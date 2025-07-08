import os
import json
from functools import partial

import torch
import torch.distributed as dist
import torchaudio  # type: ignore[import-untyped]
from lightning import LightningModule
from omegaconf import DictConfig, open_dict
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache  # type: ignore[import-untyped]

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.modules import AudioPerceptionModule
from nemo.collections.speechlm2.modules.easy_ar_tts import RVQVAE, Config
from nemo.collections.speechlm2.modules.easy_ar_tts import Model as SpeechDecoder
from nemo.collections.speechlm2.modules.easy_ar_tts import (
    Stack,
    TimestepEmbedder,
    is_first_process,
)
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    load_pretrained_nemo,
    set_model_dict_for_partial_init,
)
from nemo.utils import logging


class DuplexS2SEasyARTTSModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        llm = load_pretrained_hf(
            self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights
        ).train()
        self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        self.lm_head = llm.lm_head
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens
        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        asr = load_pretrained_nemo(ASRModel, self.cfg.pretrained_asr).eval()
        with open_dict(self.cfg):
            self.cfg.perception.preprocessor = asr.cfg.preprocessor
            self.cfg.perception.encoder = asr.cfg.encoder
            self.cfg.perception.output_dim = self.llm.config.hidden_size
        self.perception = AudioPerceptionModule(self.cfg.perception).train()
        self.perception.load_state_dict(asr.state_dict(), strict=False)

        llm_tokenizer_vocab_items = self.tokenizer.vocab
        # if vocab is a dict it already has the subword and token id, if not, get it from the tokenizer
        if isinstance(llm_tokenizer_vocab_items, dict):
            llm_tokenizer_vocab_items = llm_tokenizer_vocab_items.items()
        else:
            llm_tokenizer_vocab_items = [
                (subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword))
                for subword in llm_tokenizer_vocab_items
            ]

        self.audio_codec_ds_rate = 4_096
        self.audio_codec_sample_rate = int(
            self.source_fps * self.audio_codec_ds_rate
        )  # 44_100
        with fp32_precision():
            self.audio_codec = RVQVAE.from_pretrained(
                "/lustre/fsw/portfolios/llmservice/users/jaehyeonk/codes/2025/easy-ar-tts/logs/rvqvae-d72-adv"
            ).eval()
            for p in self.audio_codec.parameters():
                p.requires_grad = False
        self._codebook_size = self.audio_codec.config.num_mixtures
        self._num_codebooks = self.audio_codec.config.depth

        speech_decoder_config = Config(
            model=Config(
                d_in=512,
                d_ff=4608,
                d_model=1152,
                num_heads=16,
                num_layers=28,
                dropout_rate=0.1,
                eps=1e-6,
                self_attn_window=None,
                rotary_value=False,
                gated_act=False,
                post_norm=False,
                attn_logit_softcapping=None,
                gradient_checkpointing=True,
                num_mlp_layers=3,
                d_depth=16,
                d_low=64,
                num_predictions=1024,
                num_splits=1,
                label_smoothing=0.01,
                masking_mode="coarse_first",
                max_training_ratio=0.8,
                min_log_std=-4.0,
                p_uncond=0.1,
                p_lm_script=0.5,
                d_lm=self.embed_tokens.weight.size(-1),
                char_aware_subword_config=Config(
                    pretrained_tokenizer_name=self.cfg.pretrained_llm,
                    d_ff=4608,
                    d_model=1152,
                    num_heads=16,
                    num_layers=1,
                    dropout_rate=0.1,
                    eps=1e-6,
                    self_attn_window=None,
                    rotary_value=False,
                    gated_act=False,
                    post_norm=False,
                    attn_logit_softcapping=None,
                    gradient_checkpointing=True,
                ),
            ),
            workdir_path=cfg.exp_manager.explicit_log_dir,
        )

        if is_first_process():
            with open(
                os.path.join(speech_decoder_config.workdir_path, "config.json"), "w"
            ) as f:
                f.write(speech_decoder_config.as_json())
        with fp32_precision():
            self.speech_decoder = SpeechDecoder(
                speech_decoder_config.model,
                torch.stack([x.detach() for x in self.audio_codec.prvq.mus_list], 0),
                os.path.join(
                    speech_decoder_config.workdir_path, "char_aware_subword.json"
                ),
            )

            ckpt_src = torch.load(
                "/lustre/fsw/portfolios/llmservice/users/jaehyeonk/codes/2025/easy-ar-tts/logs/eng-af3-tts-packing-fixed/ema_684367.pt",
                weights_only=True,
            )
            ckpt_tgt = self.speech_decoder.state_dict()

            for k in ckpt_src.keys():
                if k not in ckpt_tgt:
                    print(f"WARM START - SKIP; `{k}` not found.")
                    continue
                if ckpt_src[k].size() != ckpt_tgt[k].size():
                    print(
                        f"WARM START - SKIP; `{k}` size mismatch (src: {ckpt_src[k].size()}, tgt: {ckpt_tgt[k].size()}.)"
                    )
                else:
                    ckpt_tgt[k] = ckpt_src[k]

            with open(
                "/lustre/fsw/portfolios/llmservice/users/jaehyeonk/codes/2025/easy-ar-tts/logs/eng-af3-tts-packing-fixed/char_aware_subword.json"
            ) as f:
                cas_dict_src = json.load(f)
            for c in cas_dict_src["char_vocab"]:
                if c in self.speech_decoder.cas_encoder.char_vocab:
                    ckpt_tgt["cas_encoder.embed_tokens.weight"][
                        self.speech_decoder.cas_encoder.char_vocab[c], :
                    ] = ckpt_src["cas_encoder.embed_tokens.weight"][
                        cas_dict_src["char_vocab"][c], :
                    ]
            self.speech_decoder.load_state_dict(ckpt_tgt)
        if getattr(cfg.model.speech_decoder, "finetuning", False):
            for n, p in self.speech_decoder.named_parameters():
                if (
                    n.endswith("adaLN_emb")
                    or n == "lm_proj.weight"
                    or n == "cas_encoder.embed_tokens.weight"
                ):
                    pass
                else:
                    if p.requires_grad:
                        p.requires_grad = False

        self._use_fsdp = False
        self._use_tp = False

        # warm start
        good_ckpt_path = "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/1.78kbps/demo_model_aug_8nodes_reproduce_demo_model_llm_init_better_prev_sd_init_old_codebase_scale_user_low_pass/checkpoints/step=100000-last.ckpt"
        good_ckpt = torch.load(good_ckpt_path)

        model_state_dict = self.state_dict()
        for k in model_state_dict.keys():
            if k in good_ckpt["state_dict"]:
                assert model_state_dict[k].size() == good_ckpt["state_dict"][k].size()
                model_state_dict[k] = good_ckpt["state_dict"][k]
        self.load_state_dict(model_state_dict)

        for n, p in self.named_parameters():
            if not n.startswith("speech_decoder."):
                p.requires_grad = False

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
        input_audio_tokens=None,
        loss_mask=None,
        target_text_tokens=None,
        target_audio_tokens=None,
        modality_adapter_emb=None,
        speaker_encoder_emb=None,
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
        """

        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out["last_hidden_state"])  # (B, T, text_vocab_size)

        if loss_mask is not None:
            # This is training Mode
            loss_mask = loss_mask[:, :, -1].reshape(
                loss_mask.size(0), loss_mask.size(1)
            )

        # if inference time, uses the target text tokens sampled from the llm backbone
        if not self.training:
            target_text_tokens = (
                torch.argmax(text_logits, dim=-1).view(B, T).contiguous()
            )

        assert target_audio_tokens is not None
        with fp32_precision(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            audio_eos_loss, audio_z_loss, audio_k_loss = self.speech_decoder(
                code=target_audio_tokens,
                audio_mask=torch.ones((B, T, 1), dtype=torch.long, device=self.device),
                lm_hidden_state=out["last_hidden_state"],
                subword_ids=target_text_tokens,
            )
            audio_loss = (
                audio_z_loss + audio_k_loss + audio_eos_loss * 0
            )  # ignore audio_eos_loss

        ans = {"text_logits": text_logits, "audio_loss": audio_loss}
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def forward_infer(
        self,
        input_embeds,
        input_audio_tokens,
        cache,
        past_key_values,
        cnt: int,
        num_iter: int,
        classifier_free_guidance_list: list[float] | None = None,
        top_p_or_k_list: list[float] | None = None,
        noise_scale_list: list[float] | None = None,
        exponent: float = 3.0,
    ):
        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        B, T = input_embeds.shape[:2]

        text_logits = self.lm_head(out["last_hidden_state"])  # (B, T, text_vocab_size)
        text_ids = text_logits.argmax(dim=-1)

        sinusoidal_pos = self.speech_decoder.pos_embedding(
            torch.zeros((1,), dtype=torch.long, device=self.device) + cnt
        ).unsqueeze(0)
        cond = torch.zeros(
            (B, 1, self.speech_decoder.config.d_model), device=self.device
        )
        if self.speech_decoder.lm_proj is not None:
            lm_emb = out["last_hidden_state"]
            cond = cond + self.speech_decoder.lm_proj(lm_emb)
        assert self.speech_decoder.config.char_aware_subword_config is not None
        assert self.speech_decoder.cas_encoder is not None
        assert self.speech_decoder.cas_proj is not None
        subword_mask = torch.ones_like(text_ids).bool()
        cas_emb = self.speech_decoder.cas_encoder(text_ids, subword_mask)
        cond = cond + self.speech_decoder.cas_proj(cas_emb)
        decoder_input_emb = self.speech_decoder.depthsum_embedding(
            input_audio_tokens, self.speech_decoder.mus, include_blank=True
        ).view(B, input_audio_tokens.size(1), -1)
        x = self.speech_decoder.forward_decoder(
            decoder_input_emb
            if classifier_free_guidance_list is None
            else decoder_input_emb.repeat(2, 1, 1),
            cond
            if classifier_free_guidance_list is None
            else torch.cat(
                [
                    cond,
                    torch.zeros_like(cond) + self.speech_decoder.decoder_null_emb,
                ],
                0,
            ),
            sinusoidal_pos,
            past_key_values=past_key_values,
        )
        code = self.speech_decoder.generate_step(
            num_iter=num_iter,
            x=x[:B],
            x_cfg=None if classifier_free_guidance_list is None else x[B:],
            classifier_free_guidance_list=classifier_free_guidance_list,
            top_p_or_k_list=top_p_or_k_list,
            noise_scale_list=noise_scale_list,
            exponent=exponent,
        )

        ans = {
            "text_ids": text_ids,
            "code": code,
            "cache": out["past_key_values"] if cache is not None else None,
            "past_key_values": past_key_values,
        }
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'loss_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["target_first_turn_audio"].size(0) == batch["target_audio"].size(0)

        source_encoded, source_encoded_lens = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
        )

        speaker_encoder_emb = None

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(
                            source_encoded.shape[0],
                            abs(diff),
                            device=source_encoded.device,
                        )
                        * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        with fp32_precision(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ), torch.no_grad():
            target_audio = torchaudio.functional.resample(
                batch["target_audio"].float(),
                self.target_sample_rate,
                self.audio_codec_sample_rate,
            )
            target_audio_lens = (
                (
                    batch["target_audio_lens"]
                    * (self.audio_codec_sample_rate / self.target_sample_rate)
                ).clamp_max(target_audio.size(-1))
                / self.audio_codec_ds_rate
            ).long() * self.audio_codec_ds_rate
            target_codes, target_codes_lens = self.audio_codec.encode(
                target_audio.unsqueeze(1)[..., : target_audio_lens.max()],
                target_audio_lens,
            )

        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self._codebook_size,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )

        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        input_embeds = self.embed_tokens(text_inputs)

        input_embeds.add_(
            source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0)
        )

        loss_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )

        if self.cfg.get("mask_sequence_loss", True):
            # set the mask based on the target_token_lens to disconsider sequence padding in loss
            for i in range(batch["target_token_lens"].size(0)):
                speech_end_idx = batch["target_token_lens"][i]
                loss_mask[i, speech_end_idx:, :] = 0

            # check new mask consistency
            mask_lengths = loss_mask[:, :, 0].sum(-1)
            assert torch.allclose(
                batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0
            )

        """
        # debug samples:
        def write_wave(one_audio_signal, file_name, sr=None):
            import numpy as np
            import soundfile as sf
            one_audio_signal = one_audio_signal.cpu().numpy()
            one_audio_signal = one_audio_signal.astype(np.float32)
            if sr is None:
                sr = self.target_sample_rate
            # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
            sf.write(file_name, one_audio_signal, sr)    

        write_wave(
            batch["target_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_target_audio_5.wav",
            sr=22050
        )
        write_wave(
            batch["target_first_turn_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_speaker_ref_5.wav",
            sr=22050
        )
        write_wave(
            batch["source_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_input_5.wav",
            sr=16000
        )
        # reconstruct wav
        audio_labels = replace_control_speech_codes(audio_labels, self._control_codes)
        with fp32_precision(), torch.no_grad():
            lengths = torch.tensor([audio_labels.shape[1]]*audio_labels.shape[0]).to(self.audio_codec.device)
            predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                tokens=audio_labels.transpose(1, 2), tokens_len=lengths
            )
        write_wave(
            predicted_audio[-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/reconstructed_codec_audio_5.wav",
            sr=22050
        )

        # check text
        print("text_labels", text_labels)
        print("target labels from dataloader", batch["target_tokens"])
        print("text_labels", tokens_to_str(text_labels[-1:], target_codes_lens-1, tokenizer=self.tokenizer, pad_id=self.text_pad_id))
        print("target labels from dataloader",  tokens_to_str(batch["target_tokens"][-1:], target_codes_lens-1, tokenizer=self.tokenizer, pad_id=self.text_pad_id))

        zeros_begening = 0
        for t in text_labels[-1:].squeeze():
            if t == 0:
                zeros_begening += 1
            else:
                break

        print("Total aduio seconds padded input:", (zeros_begening*self.audio_codec.samples_per_frame)/ self.target_sample_rate)

        exit()
        """

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_labels": text_labels,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "loss_mask": loss_mask,
            "perception_emb": source_encoded[:, :-1],
            "speaker_encoder_emb": speaker_encoder_emb,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (
            self.perception.preprocessor,
            self.perception.encoder,
            self.llm,
            self.speech_decoder,
        ):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            loss_mask=inputs["loss_mask"],
            target_text_tokens=inputs["text_labels"],
            target_audio_tokens=inputs["audio_labels"],
            modality_adapter_emb=inputs["perception_emb"],
            speaker_encoder_emb=inputs["speaker_encoder_emb"],
        )
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            # mask audio logits to ignore sequence padding
            text_logits = forward_outputs["text_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                text_logits = text_logits * inputs["loss_mask"][:, :, 0].unsqueeze(-1)
            text_loss = (
                torch.nn.functional.cross_entropy(
                    text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="sum",
                )
                / num_frames
            )
            # mask audio logits to ignore sequence padding
            audio_loss = forward_outputs["audio_loss"]
        loss = (
            self.cfg.text_loss_weight * text_loss
            + self.cfg.audio_loss_weight * audio_loss
        )

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(
                    self.trainer.optimizers[0].param_groups[0]["lr"]
                    if self._trainer is not None
                    else 0
                )
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            with torch.no_grad():
                ans = self.training_step(dataset_batch, batch_idx)
                print("val", ans["loss"], ans["text_loss"], ans["audio_loss"])

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
            )

            if (
                self.cfg.get("audio_save_path", None) is not None
                and dist.get_rank() == 0
            ):
                os.makedirs(self.cfg.audio_save_path, exist_ok=True)
                predicted_audios = results["audio"]
                for i in range(len(predicted_audios)):
                    pred_audio = predicted_audios[i].float()
                    user_audio = torchaudio.functional.resample(
                        dataset_batch["source_audio"][i].float(),
                        self.source_sample_rate,
                        self.audio_codec_sample_rate,
                    )

                    T1, T2 = pred_audio.shape[0], user_audio.shape[0]
                    max_len = max(T1, T2)
                    pred_audio_padded = torch.nn.functional.pad(
                        pred_audio, (0, max_len - T1), mode="constant", value=0
                    )
                    user_audio_padded = torch.nn.functional.pad(
                        user_audio, (0, max_len - T2), mode="constant", value=0
                    )

                    # combine audio in a multichannel audio
                    combined_wav = torch.cat(
                        [
                            user_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                            pred_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                        ],
                        dim=0,
                    )

                    # save audio
                    out_audio_path = f"{self.cfg.audio_save_path}/{name}_{dataset_batch['sample_id'][i]}.wav"
                    torchaudio.save(
                        out_audio_path,
                        combined_wav.squeeze(),
                        self.audio_codec_sample_rate,
                    )
                    print("Audio saved at:", out_audio_path)

            with fp32_precision():  # torchaudio resample is fragile to bfloat16 default dtype as well
                self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=torchaudio.functional.resample(
                        results["audio"], self.audio_codec_sample_rate, 16000
                    ),
                    pred_audio_lens=(
                        results["audio_len"] / self.audio_codec_sample_rate * 16000
                    ).to(torch.long),
                )

            self.bleu.update(
                name=name, refs=dataset_batch["target_texts"], hyps=results["text"]
            )

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
        num_iter: int = 4,
        classifier_free_guidance: float | None = None,
        top_p_or_k: float = 0.8,
        noise_scale: float = 0.8,
        exponent: float = 3.0,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        with fp32_precision(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            source_encoded, lengths = self.perception(
                input_signal=input_signal,
                input_signal_length=input_signal_lens,
            )
            B, T_local, H = source_encoded.shape

            # Determine decoding length and pad if FSDP
            if self._use_fsdp:
                T_tensor = torch.tensor([T_local], device=source_encoded.device)
                dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
                T = int(T_tensor.item())
                if T > T_local:
                    last_frame = source_encoded[:, T_local - 1 : T_local, :]  # (B,1,H)
                    pad = last_frame.repeat(1, T - T_local, 1)  # (B, T-T_local, H)
                    source_encoded = torch.cat([source_encoded, pad], dim=1)
            else:
                T = T_local

            # Apply channel weight
            input_embeds = source_encoded.clone()
            input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

            # This cache is for self.llm
            cache = DynamicCache()
            # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
            past_key_values = DynamicCache()
            classifier_free_guidance_list = (
                None
                if classifier_free_guidance is None
                else [classifier_free_guidance] * num_iter
            )
            top_p_or_k_list = [top_p_or_k] * num_iter
            noise_scale_list = [noise_scale] * num_iter

            gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
            gen_audio = torch.empty(
                B,
                T,
                self._num_codebooks,
                device=self.device,
                dtype=torch.long,
            )

            # First step, use speech_delay token
            input_embeds[:, 0] += self._get_bos_embedding()
            first_audio = torch.full(
                [B, 1, self._num_codebooks],
                fill_value=self._codebook_size,
                device=self.device,
                dtype=torch.long,
            )

            # generation

            ans = self.forward_infer(
                input_embeds=input_embeds[:, 0:1],
                input_audio_tokens=first_audio,
                cache=cache,
                past_key_values=past_key_values,
                cnt=0,
                num_iter=num_iter,
                classifier_free_guidance_list=classifier_free_guidance_list,
                top_p_or_k_list=top_p_or_k_list,
                noise_scale_list=noise_scale_list,
                exponent=exponent,
            )
            gen_text[:, 0] = ans["text_ids"][:, -1]
            gen_audio[:, 0] = ans["code"][:, -1]

            # Autoregressive loop
            for t in range(1, T):
                last_emb = self.embed_tokens(gen_text[:, t - 1])
                input_embeds[:, t] += last_emb
                current_audio = gen_audio[:, t - 1 : t, :]
                ans = self.forward_infer(
                    input_embeds=input_embeds[:, t : t + 1],
                    input_audio_tokens=current_audio,
                    cache=ans["cache"],
                    past_key_values=ans["past_key_values"],
                    cnt=t,
                    num_iter=num_iter,
                    classifier_free_guidance_list=classifier_free_guidance_list,
                    top_p_or_k_list=top_p_or_k_list,
                    noise_scale_list=noise_scale_list,
                    exponent=exponent,
                )
                gen_text[:, t] = ans["text_ids"][:, -1]
                gen_audio[:, t] = ans["code"][:, -1]

            # Trim back to local length if padded
            if self._use_fsdp and T > T_local:
                gen_text = gen_text[:, :T_local]
                gen_audio = gen_audio[:, :T_local]

            ans = {
                "text": tokens_to_str(
                    gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id
                ),
                "tokens_text": gen_text,
                "tokens_audio": gen_audio,
                "tokens_len": lengths,
            }

            if decode_audio:
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    gen_audio, lengths
                )
                ans["audio"] = predicted_audio.squeeze(1)
                ans["audio_len"] = predicted_audio_lens
        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info(
                "Error loading model state_dict !! Retrying with partial initialization!"
            )
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)


class DuplexS2SEasyARIOModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        assert self.target_sample_rate == self.source_sample_rate
        assert self.target_sample_rate == 44_100
        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        llm = load_pretrained_hf(
            self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights
        ).train()
        self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        self.lm_head = llm.lm_head
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens
        maybe_install_lora(self)

        # Load Neural Audio Codec
        self.audio_codec_ds_rate = 4_096
        with fp32_precision():
            self.audio_codec = RVQVAE.from_pretrained(
                "/lustre/fsw/portfolios/llmservice/users/jaehyeonk/codes/2025/easy-ar-tts/logs/rvqvae-d72-adv"
            ).eval()
            for p in self.audio_codec.parameters():
                p.requires_grad = False
        self._codebook_size = self.audio_codec.config.num_mixtures
        self._num_codebooks = self.audio_codec.config.depth

        # Perception Module
        with fp32_precision():
            d_model = 768
            num_heads = 12
            num_layers = 12
            self.perception_pos_embedding = partial(
                TimestepEmbedder.timestep_embedding,
                dim=d_model // num_heads,
            )
            self.perception_pre = torch.nn.Linear(
                self.audio_codec.config.latent_dim, d_model
            )
            self.perception_encoder = Stack(
                d_model,
                d_model * 4,
                num_heads,
                num_layers,
            )
            self.perception_post = torch.nn.Linear(d_model, self.llm.config.hidden_size)

        llm_tokenizer_vocab_items = self.tokenizer.vocab
        # if vocab is a dict it already has the subword and token id, if not, get it from the tokenizer
        if isinstance(llm_tokenizer_vocab_items, dict):
            llm_tokenizer_vocab_items = llm_tokenizer_vocab_items.items()
        else:
            llm_tokenizer_vocab_items = [
                (subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword))
                for subword in llm_tokenizer_vocab_items
            ]

        # Speech Decoder
        speech_decoder_config = Config(
            model=Config(
                d_in=512,
                d_ff=4608,
                d_model=1152,
                num_heads=16,
                num_layers=28,
                dropout_rate=0.1,
                eps=1e-6,
                self_attn_window=None,
                rotary_value=False,
                gated_act=False,
                post_norm=False,
                attn_logit_softcapping=None,
                gradient_checkpointing=True,
                num_mlp_layers=3,
                d_depth=16,
                d_low=64,
                num_predictions=1024,
                num_splits=1,
                label_smoothing=0.01,
                masking_mode="coarse_first",
                max_training_ratio=0.8,
                min_log_std=-4.0,
                p_uncond=0.1,
                p_lm_script=0.5,
                d_lm=self.embed_tokens.weight.size(-1),
                char_aware_subword_config=Config(
                    pretrained_tokenizer_name=self.cfg.pretrained_llm,
                    d_ff=4608,
                    d_model=1152,
                    num_heads=16,
                    num_layers=1,
                    dropout_rate=0.1,
                    eps=1e-6,
                    self_attn_window=None,
                    rotary_value=False,
                    gated_act=False,
                    post_norm=False,
                    attn_logit_softcapping=None,
                    gradient_checkpointing=True,
                ),
            ),
            workdir_path=cfg.exp_manager.explicit_log_dir,
        )

        if is_first_process():
            with open(
                os.path.join(speech_decoder_config.workdir_path, "config.json"), "w"
            ) as f:
                f.write(speech_decoder_config.as_json())
        with fp32_precision():
            self.speech_decoder = SpeechDecoder(
                speech_decoder_config.model,
                torch.stack([x.detach() for x in self.audio_codec.prvq.mus_list], 0),
                os.path.join(
                    speech_decoder_config.workdir_path, "char_aware_subword.json"
                ),
            )

            ckpt_src = torch.load(
                "/lustre/fsw/portfolios/llmservice/users/jaehyeonk/codes/2025/easy-ar-tts/logs/eng-af3-tts-packing-fixed/ema_684367.pt",
                weights_only=True,
            )
            ckpt_tgt = self.speech_decoder.state_dict()

            for k in ckpt_src.keys():
                if k not in ckpt_tgt:
                    print(f"WARM START - SKIP; `{k}` not found.")
                    continue
                if ckpt_src[k].size() != ckpt_tgt[k].size():
                    print(
                        f"WARM START - SKIP; `{k}` size mismatch (src: {ckpt_src[k].size()}, tgt: {ckpt_tgt[k].size()}.)"
                    )
                else:
                    ckpt_tgt[k] = ckpt_src[k]

            with open(
                "/lustre/fsw/portfolios/llmservice/users/jaehyeonk/codes/2025/easy-ar-tts/logs/eng-af3-tts-packing-fixed/char_aware_subword.json"
            ) as f:
                cas_dict_src = json.load(f)
            for c in cas_dict_src["char_vocab"]:
                if c in self.speech_decoder.cas_encoder.char_vocab:
                    ckpt_tgt["cas_encoder.embed_tokens.weight"][
                        self.speech_decoder.cas_encoder.char_vocab[c], :
                    ] = ckpt_src["cas_encoder.embed_tokens.weight"][
                        cas_dict_src["char_vocab"][c], :
                    ]
            self.speech_decoder.load_state_dict(ckpt_tgt)
        if getattr(cfg.model.speech_decoder, "finetuning", False):
            for n, p in self.speech_decoder.named_parameters():
                if (
                    n.endswith("adaLN_emb")
                    or n == "lm_proj.weight"
                    or n == "cas_encoder.embed_tokens.weight"
                ):
                    pass
                else:
                    if p.requires_grad:
                        p.requires_grad = False

        self._use_fsdp = False
        self._use_tp = False

        # warm start
        good_ckpt_path = "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/results/exp/1.78kbps/demo_model_aug_8nodes_reproduce_demo_model_llm_init_better_prev_sd_init_old_codebase_scale_user_low_pass/checkpoints/step=100000-last.ckpt"
        good_ckpt = torch.load(good_ckpt_path)

        model_state_dict = self.state_dict()
        for k in model_state_dict.keys():
            if k in good_ckpt["state_dict"]:
                assert model_state_dict[k].size() == good_ckpt["state_dict"][k].size()
                model_state_dict[k] = good_ckpt["state_dict"][k]
        self.load_state_dict(model_state_dict)

        for n, p in self.named_parameters():
            if not (n.startswith("speech_decoder.") or n.startswith("perception_")):
                p.requires_grad = False

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def perception(self, code: Tensor, code_length: Tensor) -> Tensor:
        with fp32_precision(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            embs = self.speech_decoder.mus
            b, t, d = code.size()
            _, v, h = embs.size()
            device = code.device

            position_ids = torch.arange(t).to(device=device).unsqueeze(0)
            sinusoidal_pos = self.perception_pos_embedding(position_ids)

            input_emb = torch.zeros((b, t, h), device=device)
            for i in range(d):
                emb = embs[i]
                input_emb = input_emb + torch.nn.functional.embedding(code[..., i], emb)

            x = self.perception_pre(input_emb)
            x = self.perception_encoder(
                x, is_causal=True, sinusoidal_pos=sinusoidal_pos
            )
            x = self.perception_post(x)
            return x, code_length

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
        input_audio_tokens=None,
        loss_mask=None,
        target_text_tokens=None,
        target_audio_tokens=None,
        modality_adapter_emb=None,
        speaker_encoder_emb=None,
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
        """

        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out["last_hidden_state"])  # (B, T, text_vocab_size)

        if loss_mask is not None:
            # This is training Mode
            loss_mask = loss_mask[:, :, -1].reshape(
                loss_mask.size(0), loss_mask.size(1)
            )

        # if inference time, uses the target text tokens sampled from the llm backbone
        if not self.training:
            target_text_tokens = (
                torch.argmax(text_logits, dim=-1).view(B, T).contiguous()
            )

        assert target_audio_tokens is not None
        with fp32_precision(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            audio_eos_loss, audio_z_loss, audio_k_loss = self.speech_decoder(
                code=target_audio_tokens,
                audio_mask=torch.ones((B, T, 1), dtype=torch.long, device=self.device),
                lm_hidden_state=out["last_hidden_state"],
                subword_ids=target_text_tokens,
            )
            audio_loss = (
                audio_z_loss + audio_k_loss + audio_eos_loss * 0
            )  # ignore audio_eos_loss

        ans = {"text_logits": text_logits, "audio_loss": audio_loss}
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def forward_infer(
        self,
        input_embeds,
        input_audio_tokens,
        cache,
        past_key_values,
        cnt: int,
        num_iter: int,
        classifier_free_guidance_list: list[float] | None = None,
        top_p_or_k_list: list[float] | None = None,
        noise_scale_list: list[float] | None = None,
        exponent: float = 3.0,
    ):
        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        B, T = input_embeds.shape[:2]

        text_logits = self.lm_head(out["last_hidden_state"])  # (B, T, text_vocab_size)
        text_ids = text_logits.argmax(dim=-1)

        sinusoidal_pos = self.speech_decoder.pos_embedding(
            torch.zeros((1,), dtype=torch.long, device=self.device) + cnt
        ).unsqueeze(0)
        cond = torch.zeros(
            (B, 1, self.speech_decoder.config.d_model), device=self.device
        )
        if self.speech_decoder.lm_proj is not None:
            lm_emb = out["last_hidden_state"]
            cond = cond + self.speech_decoder.lm_proj(lm_emb)
        assert self.speech_decoder.config.char_aware_subword_config is not None
        assert self.speech_decoder.cas_encoder is not None
        assert self.speech_decoder.cas_proj is not None
        subword_mask = torch.ones_like(text_ids).bool()
        cas_emb = self.speech_decoder.cas_encoder(text_ids, subword_mask)
        cond = cond + self.speech_decoder.cas_proj(cas_emb)
        decoder_input_emb = self.speech_decoder.depthsum_embedding(
            input_audio_tokens, self.speech_decoder.mus, include_blank=True
        ).view(B, input_audio_tokens.size(1), -1)
        x = self.speech_decoder.forward_decoder(
            decoder_input_emb
            if classifier_free_guidance_list is None
            else decoder_input_emb.repeat(2, 1, 1),
            cond
            if classifier_free_guidance_list is None
            else torch.cat(
                [
                    cond,
                    torch.zeros_like(cond) + self.speech_decoder.decoder_null_emb,
                ],
                0,
            ),
            sinusoidal_pos,
            past_key_values=past_key_values,
        )
        code = self.speech_decoder.generate_step(
            num_iter=num_iter,
            x=x[:B],
            x_cfg=None if classifier_free_guidance_list is None else x[B:],
            classifier_free_guidance_list=classifier_free_guidance_list,
            top_p_or_k_list=top_p_or_k_list,
            noise_scale_list=noise_scale_list,
            exponent=exponent,
        )

        ans = {
            "text_ids": text_ids,
            "code": code,
            "cache": out["past_key_values"] if cache is not None else None,
            "past_key_values": past_key_values,
        }
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'loss_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["target_first_turn_audio"].size(0) == batch["target_audio"].size(0)

        with fp32_precision(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ), torch.no_grad():
            source_audio_lens = (
                (batch["source_audio_lens"]).clamp_max(batch["source_audio"].size(-1))
                / self.audio_codec_ds_rate
            ).long() * self.audio_codec_ds_rate
            target_audio_lens = (
                (batch["target_audio_lens"]).clamp_max(batch["target_audio"].size(-1))
                / self.audio_codec_ds_rate
            ).long() * self.audio_codec_ds_rate
            source_codes, source_codes_lens = self.audio_codec.encode(
                batch["source_audio"].unsqueeze(1)[..., : source_audio_lens.max()],
                source_audio_lens,
            )

            target_codes, target_codes_lens = self.audio_codec.encode(
                batch["target_audio"].unsqueeze(1)[..., : target_audio_lens.max()],
                target_audio_lens,
            )

        source_encoded, source_encoded_lens = self.perception(
            source_codes,
            source_codes_lens,
        )

        speaker_encoder_emb = None

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(
                            source_encoded.shape[0],
                            abs(diff),
                            device=source_encoded.device,
                        )
                        * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self._codebook_size,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )

        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        input_embeds = self.embed_tokens(text_inputs)

        input_embeds.add_(
            source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0)
        )

        loss_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )

        if self.cfg.get("mask_sequence_loss", True):
            # set the mask based on the target_token_lens to disconsider sequence padding in loss
            for i in range(batch["target_token_lens"].size(0)):
                speech_end_idx = batch["target_token_lens"][i]
                loss_mask[i, speech_end_idx:, :] = 0

            # check new mask consistency
            mask_lengths = loss_mask[:, :, 0].sum(-1)
            assert torch.allclose(
                batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0
            )

        """
        # debug samples:
        def write_wave(one_audio_signal, file_name, sr=None):
            import numpy as np
            import soundfile as sf
            one_audio_signal = one_audio_signal.cpu().numpy()
            one_audio_signal = one_audio_signal.astype(np.float32)
            if sr is None:
                sr = self.target_sample_rate
            # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
            sf.write(file_name, one_audio_signal, sr)    

        write_wave(
            batch["target_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_target_audio_5.wav",
            sr=22050
        )
        write_wave(
            batch["target_first_turn_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_speaker_ref_5.wav",
            sr=22050
        )
        write_wave(
            batch["source_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_input_5.wav",
            sr=16000
        )
        # reconstruct wav
        audio_labels = replace_control_speech_codes(audio_labels, self._control_codes)
        with fp32_precision(), torch.no_grad():
            lengths = torch.tensor([audio_labels.shape[1]]*audio_labels.shape[0]).to(self.audio_codec.device)
            predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                tokens=audio_labels.transpose(1, 2), tokens_len=lengths
            )
        write_wave(
            predicted_audio[-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/reconstructed_codec_audio_5.wav",
            sr=22050
        )

        # check text
        print("text_labels", text_labels)
        print("target labels from dataloader", batch["target_tokens"])
        print("text_labels", tokens_to_str(text_labels[-1:], target_codes_lens-1, tokenizer=self.tokenizer, pad_id=self.text_pad_id))
        print("target labels from dataloader",  tokens_to_str(batch["target_tokens"][-1:], target_codes_lens-1, tokenizer=self.tokenizer, pad_id=self.text_pad_id))

        zeros_begening = 0
        for t in text_labels[-1:].squeeze():
            if t == 0:
                zeros_begening += 1
            else:
                break

        print("Total aduio seconds padded input:", (zeros_begening*self.audio_codec.samples_per_frame)/ self.target_sample_rate)

        exit()
        """

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_labels": text_labels,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "loss_mask": loss_mask,
            "perception_emb": source_encoded[:, :-1],
            "speaker_encoder_emb": speaker_encoder_emb,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (
            self.perception_pre,
            self.perception_encoder,
            self.perception_post,
            self.llm,
            self.speech_decoder,
        ):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            loss_mask=inputs["loss_mask"],
            target_text_tokens=inputs["text_labels"],
            target_audio_tokens=inputs["audio_labels"],
            modality_adapter_emb=inputs["perception_emb"],
            speaker_encoder_emb=inputs["speaker_encoder_emb"],
        )
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            # mask audio logits to ignore sequence padding
            text_logits = forward_outputs["text_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                text_logits = text_logits * inputs["loss_mask"][:, :, 0].unsqueeze(-1)
            text_loss = (
                torch.nn.functional.cross_entropy(
                    text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="sum",
                )
                / num_frames
            )
            # mask audio logits to ignore sequence padding
            audio_loss = forward_outputs["audio_loss"]
        loss = (
            self.cfg.text_loss_weight * text_loss
            + self.cfg.audio_loss_weight * audio_loss
        )

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(
                    self.trainer.optimizers[0].param_groups[0]["lr"]
                    if self._trainer is not None
                    else 0
                )
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            with torch.no_grad():
                ans = self.training_step(dataset_batch, batch_idx)
                print("val", ans["loss"], ans["text_loss"], ans["audio_loss"])

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
            )

            if (
                self.cfg.get("audio_save_path", None) is not None
                and dist.get_rank() == 0
            ):
                os.makedirs(self.cfg.audio_save_path, exist_ok=True)
                predicted_audios = results["audio"]
                for i in range(len(predicted_audios)):
                    pred_audio = predicted_audios[i].float()
                    user_audio = dataset_batch["source_audio"][i].float()

                    T1, T2 = pred_audio.shape[0], user_audio.shape[0]
                    max_len = max(T1, T2)
                    pred_audio_padded = torch.nn.functional.pad(
                        pred_audio, (0, max_len - T1), mode="constant", value=0
                    )
                    user_audio_padded = torch.nn.functional.pad(
                        user_audio, (0, max_len - T2), mode="constant", value=0
                    )

                    # combine audio in a multichannel audio
                    combined_wav = torch.cat(
                        [
                            user_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                            pred_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                        ],
                        dim=0,
                    )

                    # save audio
                    out_audio_path = f"{self.cfg.audio_save_path}/{name}_{dataset_batch['sample_id'][i]}.wav"
                    torchaudio.save(
                        out_audio_path,
                        combined_wav.squeeze(),
                        self.target_sample_rate,
                    )
                    print("Audio saved at:", out_audio_path)

            with fp32_precision():  # torchaudio resample is fragile to bfloat16 default dtype as well
                self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=torchaudio.functional.resample(
                        results["audio"], self.target_sample_rate, 16000
                    ),
                    pred_audio_lens=(
                        results["audio_len"] / self.target_sample_rate * 16000
                    ).to(torch.long),
                )

            self.bleu.update(
                name=name, refs=dataset_batch["target_texts"], hyps=results["text"]
            )

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
        num_iter: int = 4,
        classifier_free_guidance: float | None = None,
        top_p_or_k: float = 0.8,
        noise_scale: float = 0.8,
        exponent: float = 3.0,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        with fp32_precision(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            input_signal_lens = (
                (input_signal_lens).clamp_max(input_signal.size(-1))
                / self.audio_codec_ds_rate
            ).long() * self.audio_codec_ds_rate
            source_codes, source_codes_lens = self.audio_codec.encode(
                input_signal.unsqueeze(1)[..., : input_signal_lens.max()],
                input_signal_lens,
            )
            source_encoded, lengths = self.perception(
                source_codes,
                source_codes_lens,
            )
            B, T_local, H = source_encoded.shape

            # Determine decoding length and pad if FSDP
            if self._use_fsdp:
                T_tensor = torch.tensor([T_local], device=source_encoded.device)
                dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
                T = int(T_tensor.item())
                if T > T_local:
                    last_frame = source_encoded[:, T_local - 1 : T_local, :]  # (B,1,H)
                    pad = last_frame.repeat(1, T - T_local, 1)  # (B, T-T_local, H)
                    source_encoded = torch.cat([source_encoded, pad], dim=1)
            else:
                T = T_local

            # Apply channel weight
            input_embeds = source_encoded.clone()
            input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

            # This cache is for self.llm
            cache = DynamicCache()
            # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
            past_key_values = DynamicCache()
            classifier_free_guidance_list = (
                None
                if classifier_free_guidance is None
                else [classifier_free_guidance] * num_iter
            )
            top_p_or_k_list = [top_p_or_k] * num_iter
            noise_scale_list = [noise_scale] * num_iter

            gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
            gen_audio = torch.empty(
                B,
                T,
                self._num_codebooks,
                device=self.device,
                dtype=torch.long,
            )

            # First step, use speech_delay token
            input_embeds[:, 0] += self._get_bos_embedding()
            first_audio = torch.full(
                [B, 1, self._num_codebooks],
                fill_value=self._codebook_size,
                device=self.device,
                dtype=torch.long,
            )

            # generation

            ans = self.forward_infer(
                input_embeds=input_embeds[:, 0:1],
                input_audio_tokens=first_audio,
                cache=cache,
                past_key_values=past_key_values,
                cnt=0,
                num_iter=num_iter,
                classifier_free_guidance_list=classifier_free_guidance_list,
                top_p_or_k_list=top_p_or_k_list,
                noise_scale_list=noise_scale_list,
                exponent=exponent,
            )
            gen_text[:, 0] = ans["text_ids"][:, -1]
            gen_audio[:, 0] = ans["code"][:, -1]

            # Autoregressive loop
            for t in range(1, T):
                last_emb = self.embed_tokens(gen_text[:, t - 1])
                input_embeds[:, t] += last_emb
                current_audio = gen_audio[:, t - 1 : t, :]
                ans = self.forward_infer(
                    input_embeds=input_embeds[:, t : t + 1],
                    input_audio_tokens=current_audio,
                    cache=ans["cache"],
                    past_key_values=ans["past_key_values"],
                    cnt=t,
                    num_iter=num_iter,
                    classifier_free_guidance_list=classifier_free_guidance_list,
                    top_p_or_k_list=top_p_or_k_list,
                    noise_scale_list=noise_scale_list,
                    exponent=exponent,
                )
                gen_text[:, t] = ans["text_ids"][:, -1]
                gen_audio[:, t] = ans["code"][:, -1]

            # Trim back to local length if padded
            if self._use_fsdp and T > T_local:
                gen_text = gen_text[:, :T_local]
                gen_audio = gen_audio[:, :T_local]

            ans = {
                "text": tokens_to_str(
                    gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id
                ),
                "tokens_text": gen_text,
                "tokens_audio": gen_audio,
                "tokens_len": lengths,
            }

            if decode_audio:
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    gen_audio, lengths
                )
                ans["audio"] = predicted_audio.squeeze(1)
                ans["audio_len"] = predicted_audio_lens
        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info(
                "Error loading model state_dict !! Retrying with partial initialization!"
            )
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
