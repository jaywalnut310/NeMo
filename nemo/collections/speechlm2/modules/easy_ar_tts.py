import glob
import importlib.machinery
import json
import logging
import math
import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal, overload

import soundfile as sf  # type: ignore[import-untyped]
import torch
import torchaudio  # type: ignore[import-untyped]
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import (  # type: ignore[import-untyped]
    AutoModelForCausalLM,
    AutoTokenizer,
    Cache,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper  # type: ignore[import-untyped]

logger = logging.getLogger()

PYTHON_CONFIG_GETTER_NAME = "get_config"
CHECKPOINT_FORMAT = "net_{}.pt"
CONFIG_NAME = "config.json"

MASKING_MODE = Literal["indep", "coarse_first", "fine_first"]

SAMPLING_RATE = 44_100
WAV_TO_TOKEN_RATIO = 4096
SCRIPT_PLACEHOLDER = "[[[<<<SCRIPT_PLACEHOLDER>>>]]]"

default_chat_template = """{% for message in messages %}
{{ bos_token }}{{ '### ' + message['role'] + '\n' }}{{ message['content'] }}{{ eos_token }}
{% endfor %}{{ bos_token + '### assistant\n' if add_generation_prompt else '' }}"""


def is_first_process() -> bool:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


def latest_checkpoint_path(dir_path: str, regex: str | None = None) -> str:
    if regex is None:
        regex = CHECKPOINT_FORMAT.format("*")
    f_list = glob.glob(os.path.join(dir_path, regex))
    assert len(f_list) > 0
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.info(
        f"The lastest checkpoint {os.path.basename(x)} is found in the {dir_path} directory."
    )
    return x


class Config(MutableMapping):
    def __init__(self, **kwargs):
        self.update(kwargs)

    def update(self, other=(), /, **kwds):
        def _register(key, value):
            if isinstance(value, Mapping):
                value = Config(**value)
            self[key] = value

        if isinstance(other, Mapping):
            for key in other:
                _register(key, other[key])
        elif hasattr(other, "keys"):
            for key in other.keys():
                _register(key, other[key])
        else:
            for key, value in other:
                _register(key, value)
        for key, value in kwds.items():
            _register(key, value)

    def as_json(self):
        return json.dumps(
            self.__dict__,
            default=lambda x: x.__dict__,
            indent=2,
        )

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError(
                f"`{self.__class__.__name__}` object has no attribute `{key}`"
            )

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __hash__(self):
        # Convert the internal state of the object to a hashable type
        # (like a tuple), and compute its hash.
        return hash(tuple(sorted(self.items())))


def get_config_from_file(config_path: str) -> Config:
    match = re.search(r".+\.((json)|(py)|(py:.+))$", config_path)
    assert (
        match is not None
    ), f"Python files (*.py) or JSON files (*.json) are only accepted as configuration files, but got {config_path}."

    py_config_name: str | None = None
    if not (config_path.endswith(".py") or config_path.endswith(".json")):
        config_path_split = config_path.split(":")
        config_path = ":".join(config_path_split[:-1])
        py_config_name = config_path_split[-1]

    assert os.path.exists(config_path) and os.path.isfile(
        config_path
    ), f"There is no such file {config_path}."

    if config_path.endswith(".json"):
        with open(config_path) as f:
            data = f.read()
        config = json.loads(data)
    else:
        config_module = importlib.machinery.SourceFileLoader(
            "_config", config_path
        ).load_module()
        assert hasattr(config_module, PYTHON_CONFIG_GETTER_NAME), (
            f"Python configuration files should define `{PYTHON_CONFIG_GETTER_NAME}` method "
            "that returns configuration dictionary."
        )
        config = getattr(config_module, PYTHON_CONFIG_GETTER_NAME)(py_config_name)
        assert isinstance(config, Mapping), (
            f"`{PYTHON_CONFIG_GETTER_NAME}` method in the python configuration file "
            "should return a python dictionary-like object."
        )
    cfg = Config(**config)
    return cfg


def get_config_from_dir(workdir_path: str) -> Config:
    config_save_path = os.path.join(workdir_path, CONFIG_NAME)
    cfg = get_config_from_file(config_save_path)
    cfg.workdir_path = workdir_path
    return cfg


def batch_matmul_pytorch(x, w, y, *args, **kwargs):
    result = torch.bmm(w[y], x.unsqueeze(2)).squeeze(2)
    return result


try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    @triton.jit
    def batch_matmul_kernel(
        x_ptr,  # Pointer to x: [b, d_in]
        w_ptr,  # Pointer to w: [n, d_out, d_in]
        y_ptr,  # Pointer to y: [b]
        result_ptr,  # Pointer to result: [b, d_out]
        b,
        d_in,
        d_out,
        n,
        BLOCK_SIZE_DIN: tl.constexpr,
        BLOCK_SIZE_DOUT: tl.constexpr,
    ):
        # Program IDs for batch and output dimension
        batch_id = tl.program_id(axis=0)
        dout_block_id = tl.program_id(axis=1)

        # Return if batch_id is out of bounds
        if batch_id >= b:
            return

        # Load index idx = y[batch_id]
        idx = tl.load(y_ptr + batch_id)

        # Compute offsets
        x_offset = x_ptr + batch_id * d_in
        w_offset = w_ptr + idx * d_out * d_in

        # Output dimension offsets for this block
        dout_offsets = dout_block_id * BLOCK_SIZE_DOUT + tl.arange(0, BLOCK_SIZE_DOUT)
        dout_mask = dout_offsets < d_out

        # Initialize result block
        result_block = tl.zeros([BLOCK_SIZE_DOUT], dtype=tl.float32)

        # Loop over d_in in chunks
        for din_start in range(0, d_in, BLOCK_SIZE_DIN):
            din_offsets = din_start + tl.arange(0, BLOCK_SIZE_DIN)
            din_mask = din_offsets < d_in

            # Load x_i[din_offsets]
            x_i = tl.load(x_offset + din_offsets, mask=din_mask, other=0.0)

            # Load w_i[dout_offsets, din_offsets]
            w_i_block = tl.load(
                w_offset + dout_offsets[:, None] * d_in + din_offsets[None, :],
                mask=(dout_mask[:, None] & din_mask[None, :]),
                other=0.0,
            )

            # Compute partial dot product and accumulate
            partial = tl.sum(w_i_block * x_i[None, :], axis=1)
            result_block += partial
        # Store the result
        result_offset = result_ptr + batch_id * d_out + dout_offsets
        tl.store(result_offset, result_block, mask=dout_mask)

    # Triton function to call the kernel
    def batch_matmul_triton(
        x, w, y, *args, BLOCK_SIZE_DIN: int = 16, BLOCK_SIZE_DOUT: int = 64
    ):
        assert x.is_contiguous()
        assert w.is_contiguous()
        assert y.is_contiguous()
        assert math.log2(BLOCK_SIZE_DIN).is_integer()
        assert math.log2(BLOCK_SIZE_DOUT).is_integer()

        b, d_in = x.shape
        n, d_out, _ = w.shape
        result = torch.empty(b, d_out, device=x.device, dtype=torch.float32)

        def grid(meta):
            return (b, triton.cdiv(d_in, meta["BLOCK_SIZE_DOUT"]))

        batch_matmul_kernel[grid](
            x.float(),
            w.float(),
            y,
            result,
            b,
            d_in,
            d_out,
            n,
            BLOCK_SIZE_DIN=BLOCK_SIZE_DIN,
            BLOCK_SIZE_DOUT=BLOCK_SIZE_DOUT,
        )
        return result.to(dtype=x.dtype)

    batch_matmul = batch_matmul_triton
    logger.info("Discovered triton lang - will use it instead of PyTorch")
except ImportError:
    batch_matmul = batch_matmul_pytorch


def gumbel_like(tensor: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Generate a tensor with Gumbel noise of the same shape as the input tensor.

    Args:
    tensor (torch.Tensor): The input tensor to base the shape on.
    eps (float): A small value to ensure numerical stability.

    Returns:
    torch.Tensor: A tensor with Gumbel noise.
    """
    u = torch.rand_like(tensor)
    return -torch.log(-torch.log(u + eps) + eps)


# From: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L68  # noqa: E501
def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """  # noqa: E501
    for p in module.parameters():
        p.detach().zero_()
    return module


def may_mask(
    x: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    if mask is not None:
        x = x * mask
    return x


@overload
def may_resize_mask(x: Tensor, mask: None, dim: int = 2) -> None: ...


@overload
def may_resize_mask(x: Tensor, mask: Tensor, dim: int = 2) -> Tensor: ...


def may_resize_mask(
    x: Tensor,
    mask: Tensor | None = None,
    dim: int = 2,
) -> Tensor | None:
    if mask is None:
        return None

    x_len = x.size(dim)
    mask_len = mask.size(dim)
    if x_len > mask_len:
        assert (
            x_len % mask_len == 0
        ), f"x_len ({x_len}) should be divisible by mask_len ({mask_len})."
        rate = x_len // mask_len
        mask = torch.repeat_interleave(mask, rate, dim)
    elif x_len < mask_len:
        assert (
            mask_len % x_len == 0
        ), f"mask_len ({mask_len}) should be divisible by x_len ({x_len})."
        rate = mask_len // x_len
        mask = torch.index_select(
            mask, dim=dim, index=torch.arange(0, mask_len, rate, device=mask.device)
        )
    return mask


def sequence_mask(lengths: Tensor, max_length: Tensor | int | None = None) -> Tensor:
    if max_length is None:
        max_length = lengths.max()
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)  # type: ignore[arg-type]
    return x.unsqueeze(0) < lengths.unsqueeze(1)


"""
Spectrogram Functions
"""


def spectrogram(
    wav: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn=torch.hann_window,
) -> Tensor:
    """
    wav: [batch_size?, time_steps], where batch_size? is an optional batch dimension
    """
    pad_size_l = (n_fft - hop_length) // 2
    pad_size_r = (n_fft - hop_length) - pad_size_l
    with torch.autocast(device_type=wav.device.type, enabled=False):
        wav = F.pad(wav, (pad_size_l, pad_size_r)).float()
        spec = torch.stft(
            wav,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window_fn(win_length).to(wav),
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    return spec


def spec_to_wav(
    spec,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window_fn=torch.hann_window,
    constrain_value_range: bool = False,
) -> Tensor:
    """
    spec: [batch_size?, dim, time_steps], where batch_size? is an optional batch dimension
    """
    with torch.autocast(device_type=spec.device.type, enabled=False):
        if window_fn == torch.hann_window:
            pad = (win_length - hop_length) // 2
            T = spec.size(-1)

            window = window_fn(win_length).to(device=spec.device)

            # Inverse FFT
            ifft = torch.fft.irfft(spec, n_fft, dim=-2, norm="backward")
            window_unsqz = window.unsqueeze(-1)
            if constrain_value_range:
                # contatrain the values, which will be overlap-added, to be less than window values
                ifft = torch.where(
                    ifft >= 0,
                    torch.minimum(ifft, window_unsqz),
                    torch.maximum(ifft, -window_unsqz),
                )
            ifft = ifft * window_unsqz

            # Overlap and Add
            output_size = (T - 1) * hop_length + win_length
            wav = torch.nn.functional.fold(
                ifft,
                output_size=(1, output_size),
                kernel_size=(1, win_length),
                stride=(1, hop_length),
            )[..., 0, 0, pad:-pad]

            # Window envelope
            window_sq = window.square().expand(T, -1).transpose(0, 1)
            window_envelope = torch.nn.functional.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, win_length),
                stride=(1, hop_length),
            ).squeeze()[pad:-pad]

            # Normalize
            assert (window_envelope > 1e-11).all()
            wav = wav / window_envelope
            return wav
        else:
            raise ValueError(
                f"`window_fn` should be 'hann_window', but got '{window_fn}'."
            )


def get_masking_ratio(ratio: Tensor, exponent: float = 3.0) -> Tensor:
    # return torch.cos(math.pi / 2.0 * ratio)
    return (1 - ratio.pow(exponent)).pow(1 / exponent)


def get_ratio(masking_ratio: Tensor, exponent: float = 3.0) -> Tensor:
    # return torch.acos(masking_ratio) * (2.0 / math.pi)
    return (1 - masking_ratio.pow(exponent)).pow(1 / exponent)


@torch.compile
def get_random_mask(
    code_mask: Tensor,
    padding_mask: Tensor,
    num_masking: Tensor,
    masking_mode: str = "indep",
    unmasking: bool = False,
    confidence: Tensor | None = None,
    min_value_bf16: float = torch.finfo(torch.bfloat16).min,
    min_value_fp16: float = torch.finfo(torch.float16).min,
    min_value_fp32: float = torch.finfo(torch.float32).min,
    min_value_fp64: float = torch.finfo(torch.float64).min,
) -> Tensor:
    """
    code_mask: BoolTensor, [b, t, d]
    padding_mask: BoolTensor, [b, t, d or 1]
    num_masking: LongTensor, [b]
    """
    assert torch.all(
        torch.logical_and(code_mask, padding_mask) == code_mask
    ), "`code_mask` should be a subset of `padding_mask`."
    if unmasking:
        code_mask = (~code_mask) * padding_mask  # complementary code mask

    batch_size, length, depth = code_mask.size()
    num_masking_max = num_masking.max()

    if confidence is None:
        v = torch.rand((batch_size, length, depth), device=code_mask.device)
    else:
        v = confidence
    if v.dtype == torch.bfloat16:
        min_value = min_value_bf16
    elif v.dtype == torch.float16:
        min_value = min_value_fp16
    elif v.dtype == torch.float32:
        min_value = min_value_fp32
    elif v.dtype == torch.float64:
        min_value = min_value_fp64
    else:
        raise RuntimeError(
            f"Expected `confidence` to be floating-point, got {v.dtype}."
        )
    v = v * code_mask + min_value * (~code_mask)
    ids_max = (
        v.view(batch_size, length * depth)
        .topk(num_masking_max, -1, largest=True, sorted=True)
        .indices
    )  # type: ignore[arg-type]
    ids_mask = sequence_mask(num_masking, num_masking_max)
    ids = ids_max * ids_mask + ids_max[:, :1] * (~ids_mask)
    mask_sel = torch.zeros(
        (batch_size, length * depth), dtype=torch.bool, device=code_mask.device
    )
    mask_sel.scatter_(dim=1, index=ids, value=1)
    mask_sel = torch.where(
        num_masking.unsqueeze(-1) == 0, torch.zeros_like(mask_sel), mask_sel
    )
    if masking_mode == "indep":
        mask = (~mask_sel.view(batch_size, length, depth)) * code_mask
        if unmasking:
            mask = (~mask) * padding_mask
    else:
        mask_sel = mask_sel.view(batch_size * length, depth)

        num_sel = mask_sel.sum(-1)
        num_org = code_mask.sum(-1).view(batch_size * length)
        num_tot = num_org - num_sel
        mask = sequence_mask(num_tot, depth).view(batch_size, length, depth)
        if unmasking:
            mask = (~mask).flip([-1]) * padding_mask

        if masking_mode == "fine_first":
            mask = mask.flip([-1])
    return mask


def may_squeeze_chunk(x: Tensor, num_splits: int, dim: int = 2) -> tuple[Tensor, ...]:
    return tuple(
        y.squeeze(dim) if y.size(dim) == 1 else y for y in x.chunk(num_splits, dim)
    )


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: an (B?, N) Tensor of timestep. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (B?, N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FF(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        gated_act: bool = True,
    ):
        super().__init__()
        self.gated_act = gated_act
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        if gated_act:
            self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.GELU(approximate="tanh")

        # init
        self.wi_0.weight.data.normal_(mean=0.0, std=d_model**-0.5)
        if gated_act:
            self.wi_1.weight.data.normal_(mean=0.0, std=d_model**-0.5)
        self.wo.weight.data.normal_(mean=0.0, std=d_ff**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.wi_0(x))
        if self.gated_act:
            h_linear = self.wi_1(x)
            h = h * h_linear
        x = self.wo(self.dropout(h))
        return x


class LayerFF(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        eps: float = 1e-6,
        cond: bool = False,
        gated_act: bool = True,
        post_norm: bool = False,
    ):
        super().__init__()
        self.cond = cond
        self.pre_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.ff = FF(d_model, d_ff, dropout_rate, gated_act=gated_act)
        self.dropout = nn.Dropout(dropout_rate)
        self.adaLN_emb = nn.Parameter(torch.zeros(3, d_model))

        self.post_norm: nn.Module | None = (
            nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
            if post_norm
            else None
        )

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        b, t, _ = x.size()
        if c is not None and self.cond:
            shift, scale, gate = may_squeeze_chunk(
                self.adaLN_emb + c.view(b, t, 3, -1), 3, dim=2
            )
        else:
            shift, scale, gate = (
                self.adaLN_emb[0],
                self.adaLN_emb[1],
                self.adaLN_emb[2],
            )
        y = self.pre_norm(x)
        y = (1 + scale) * y + shift
        y = self.ff(y)
        if self.post_norm is not None:
            y = self.post_norm(y)
        x = x + gate * self.dropout(y)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        logit_softcapping: float | None = None,
        num_key_value_heads: int | None = None,
        layer_idx: int = -1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_key_value_heads is None or (
            d_model % num_key_value_heads == 0 and num_key_value_heads <= num_heads
        )

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.dropout_rate = dropout_rate
        self.logit_softcapping = logit_softcapping
        self.layer_idx = layer_idx

        self.d_kv = d_model // num_heads

        self.q = nn.Linear(d_model, self.num_heads * self.d_kv, bias=False)
        self.kv = nn.Linear(
            d_model, 2 * self.num_key_value_heads * self.d_kv, bias=False
        )
        self.o = nn.Linear(self.num_heads * self.d_kv, d_model, bias=False)

        # init
        with torch.no_grad():
            self.q.weight.data.copy_(
                torch.randn(self.num_heads * self.d_kv, d_model)
                * ((d_model * self.d_kv) ** -0.5)
            )
            self.kv.weight.data.copy_(
                torch.randn(2 * self.num_key_value_heads * self.d_kv, d_model)
                * (d_model**-0.5)
            )
        self.o.weight.data.normal_(mean=0.0, std=(self.num_heads * self.d_kv) ** -0.5)

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        sinusoidal_pos: Tensor | None = None,
        rotary_value: bool = False,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> Tensor:
        if is_causal:
            attn_mask = None

        b, t_q, h = query_states.size()
        t_kv = key_value_states.size(1)
        q = (
            self.q(query_states).view(b, t_q, self.num_heads, self.d_kv).transpose(1, 2)
        )  # [b, n, t_q, d]
        kv = (
            self.kv(key_value_states)
            .view(b, t_kv, self.num_key_value_heads, 2 * self.d_kv)
            .transpose(1, 2)
        )  # [b, n, t_kv, 2 * d]
        k, v = torch.split(kv, [self.d_kv] * 2, -1)  # [b, n, t_kv, d] x 2

        if sinusoidal_pos is not None:
            # sin [batch_size or 1, num_heads or 1, sequence_length, embed_size_per_head//2]
            # cos [batch_size or 1, num_heads or 1, sequence_length, embed_size_per_head//2]
            if sinusoidal_pos.ndim == 3:
                sinusoidal_pos = sinusoidal_pos.unsqueeze(1)
            cos, sin = sinusoidal_pos.chunk(2, dim=-1)
            # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
            sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
            # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
            cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
            # rotate_half_q [-q1,q0,-q3,q2......,-qd-1,qd-2]
            rotate_half_q = torch.stack(
                [-q[..., 1::2], q[..., ::2]], dim=-1
            ).reshape_as(q)
            q = q * cos_pos + rotate_half_q * sin_pos
            # rotate_half_k [-k1,k0,-k3,k2......,-kd-1,kd-2]
            rotate_half_k = torch.stack(
                [-k[..., 1::2], k[..., ::2]], dim=-1
            ).reshape_as(k)
            k = k * cos_pos + rotate_half_k * sin_pos

            if rotary_value:
                # rotate_half_v [-v1,v0,-v3,v2......,-vd-1,vd-2]
                rotate_half_v = torch.stack(
                    [-v[..., 1::2], v[..., ::2]], dim=-1
                ).reshape_as(v)
                v = v * cos_pos + rotate_half_v * sin_pos

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

            assert is_causal
            is_causal = False
            t_kv = k.size(2)
            attn_mask = (
                torch.ones((b, t_q, t_kv), dtype=torch.bool, device=q.device)
                .tril(diagonal=t_kv - t_q)
                .unsqueeze(1)
            )

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=-3)
            v = v.repeat_interleave(self.num_key_value_groups, dim=-3)

        if self.logit_softcapping is not None:
            attn_bias = torch.zeros(b, 1, t_q, t_kv, dtype=q.dtype, device=q.device)
            if is_causal:
                temp_mask = torch.ones(
                    t_q, t_kv, dtype=torch.bool, device=q.device
                ).tril(diagonal=0)
                attn_bias.masked_fill_(
                    temp_mask.logical_not(), torch.finfo(q.dtype).min
                )
            elif attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(
                        attn_mask.logical_not(), torch.finfo(q.dtype).min
                    )
                else:
                    attn_bias += attn_mask
            logits = (q * (1 / math.sqrt(q.size(-1)))) @ k.transpose(-2, -1)
            logits = (
                torch.tanh(logits / self.logit_softcapping) * self.logit_softcapping
            )
            attn_weight = torch.softmax(logits + attn_bias, -1).type_as(q)
            attn_weight = F.dropout(attn_weight, self.dropout_rate, self.training)
            attn_output = attn_weight @ v
        else:
            dropout_rate = self.dropout_rate if self.training else 0.0
            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.MATH,
                    *([SDPBackend.EFFICIENT_ATTENTION] if not self.training else []),
                ]
            ):
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_rate if self.training else 0.0,
                    is_causal=is_causal,
                )
        o = self.o(
            attn_output.transpose(1, 2).reshape(b, t_q, self.num_heads * self.d_kv)
        )  # [b, t_q, d]
        return o


class LayerSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        eps: float = 1e-6,
        cond: bool = False,
        post_norm: bool = False,
        logit_softcapping: float | None = None,
        num_key_value_heads: int | None = None,
        layer_idx: int = -1,
    ):
        super().__init__()
        self.cond = cond
        self.pre_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.self_attn = Attention(
            d_model,
            num_heads,
            attention_dropout_rate,
            logit_softcapping=logit_softcapping,
            num_key_value_heads=num_key_value_heads,
            layer_idx=layer_idx,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.adaLN_emb = nn.Parameter(torch.zeros(3, d_model))

        self.post_norm: nn.Module | None = (
            nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
            if post_norm
            else None
        )

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        sinusoidal_pos: Tensor | None = None,
        rotary_value: bool = False,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> Tensor:
        b, t, _ = x.size()
        if c is not None and self.cond:
            shift, scale, gate = may_squeeze_chunk(
                self.adaLN_emb + c.view(b, t, 3, -1), 3, dim=2
            )
        else:
            shift, scale, gate = (
                self.adaLN_emb[0],
                self.adaLN_emb[1],
                self.adaLN_emb[2],
            )
        y = self.pre_norm(x)
        y = (1 + scale) * y + shift
        y = self.self_attn(
            y,
            y,
            attn_mask=attn_mask,
            is_causal=is_causal,
            sinusoidal_pos=sinusoidal_pos,
            rotary_value=rotary_value,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        if self.post_norm is not None:
            y = self.post_norm(y)
        x = x + gate * self.dropout(y)
        return x


class LayerCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        eps: float = 1e-6,
        post_norm: bool = False,
        logit_softcapping: float | None = None,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.cross_attn = Attention(
            d_model,
            num_heads,
            attention_dropout_rate,
            logit_softcapping=logit_softcapping,
            num_key_value_heads=num_key_value_heads,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.adaLN_emb = nn.Parameter(torch.zeros(3, d_model))

        self.post_norm: nn.Module | None = (
            nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
            if post_norm
            else None
        )

    def forward(
        self,
        x: Tensor,
        key_value_states: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        b, t, _ = x.size()
        shift, scale, gate = (
            self.adaLN_emb[0],
            self.adaLN_emb[1],
            self.adaLN_emb[2],
        )
        y = self.pre_norm(x)
        y = (1 + scale) * y + shift
        y = self.cross_attn(y, key_value_states, attn_mask=attn_mask)
        if self.post_norm is not None:
            y = self.post_norm(y)
        x = x + gate * self.dropout(y)
        return x


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float = 0.1,
        eps: float = 1e-6,
        cross_attn: bool = False,
        cond: bool = False,
        gated_act: bool = True,
        post_norm: bool = False,
        logit_softcapping: float | None = None,
        num_key_value_heads: int | None = None,
        layer_idx: int = -1,
    ):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = cross_attn
        self.cond = cond

        self.layer_self_attn = LayerSelfAttention(
            d_model,
            num_heads,
            dropout_rate,
            attention_dropout_rate,
            eps,
            cond=cond,
            post_norm=post_norm,
            logit_softcapping=logit_softcapping,
            num_key_value_heads=num_key_value_heads,
            layer_idx=layer_idx,
        )
        if cross_attn:
            self.layer_cross_attn = LayerCrossAttention(
                d_model,
                num_heads,
                dropout_rate,
                attention_dropout_rate,
                eps,
                post_norm=post_norm,
                logit_softcapping=logit_softcapping,
                num_key_value_heads=num_key_value_heads,
            )
        self.layer_ff = LayerFF(
            d_model,
            d_ff,
            dropout_rate,
            eps,
            cond=cond,
            gated_act=gated_act,
            post_norm=post_norm,
        )

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        is_causal: bool = False,
        sinusoidal_pos: Tensor | None = None,
        rotary_value: bool = False,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> Tensor:
        if c is not None and self.cond:
            c_sa, c_ff = c.chunk(2, dim=-1)
        else:
            c_sa, c_ff = None, None
        x = self.layer_self_attn(
            x,
            c_sa,
            attn_mask=self_attn_mask,
            is_causal=is_causal,
            sinusoidal_pos=sinusoidal_pos,
            rotary_value=rotary_value,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        if self.cross_attn:
            x = self.layer_cross_attn(
                x, key_value_states=key_value_states, attn_mask=cross_attn_mask
            )
        x = self.layer_ff(x, c_ff)
        return x


class Stack(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        eps: float = 1e-6,
        cross_attn: bool = False,
        self_attn_window: int | None = None,
        d_cond: int | None = None,
        gated_act: bool = True,
        post_norm: bool = False,
        logit_softcapping: float | None = None,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        self.self_attn_window = self_attn_window
        self.d_cond = d_cond

        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    d_ff,
                    num_heads,
                    dropout_rate,
                    attention_dropout_rate,
                    eps,
                    cross_attn,
                    cond=d_cond is not None,
                    gated_act=gated_act,
                    post_norm=post_norm,
                    logit_softcapping=logit_softcapping,
                    num_key_value_heads=num_key_value_heads,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.adaLN_emb = nn.Parameter(torch.zeros(2, d_model))
        if d_cond is not None:
            self.c_block = nn.Sequential(
                nn.Linear(d_cond, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, 8 * d_model),
            )

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        key_value_mask: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        is_causal: bool = False,
        sinusoidal_pos: Tensor | None = None,
        rotary_value: bool = False,
        gradient_checkpointing: bool = False,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> Tensor:
        b, t, d = x.size()
        if c is not None and self.d_cond is not None:
            c, c_final = self.c_block(c).split([6 * d, 2 * d], dim=2)
            shift, scale = may_squeeze_chunk(
                self.adaLN_emb + c_final.view(b, t, 2, -1), 2, dim=2
            )
        else:
            shift, scale = self.adaLN_emb[0], self.adaLN_emb[1]
        self_attn_mask, cross_attn_mask = self.get_attn_masks(
            x,
            mask,
            key_value_states,
            key_value_mask,
            self.self_attn_window,
            self_attn_mask,
            cross_attn_mask,
        )
        for block in self.blocks:
            if self.training and gradient_checkpointing:
                assert past_key_values is None
                x = checkpoint(
                    block.forward,
                    x,
                    c,
                    key_value_states=key_value_states,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    is_causal=is_causal,
                    sinusoidal_pos=sinusoidal_pos,
                    rotary_value=rotary_value,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    c,
                    key_value_states=key_value_states,
                    self_attn_mask=self_attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    is_causal=is_causal,
                    sinusoidal_pos=sinusoidal_pos,
                    rotary_value=rotary_value,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
        x = self.final_norm(x)
        x = (1 + scale) * x + shift
        return may_mask(x, mask)

    def get_attn_masks(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        key_value_mask: Tensor | None = None,
        self_attn_window: int | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self_attn_mask is None:
            if self_attn_window is None:
                if mask is None:
                    self_attn_mask = None
                else:
                    self_attn_mask = (
                        torch.ones_like(mask) * mask.transpose(1, 2)
                    ).unsqueeze(1)

                if self_attn_mask is not None and torch.all(self_attn_mask):
                    self_attn_mask = None
            else:
                if mask is None:
                    self_attn_mask = (
                        torch.ones(
                            (x.size(0), x.size(1), x.size(1)),
                            dtype=torch.bool,
                            device=x.device,
                        )
                        .tril(self_attn_window)
                        .triu(-self_attn_window)
                        .unsqueeze(1)
                    )
                else:
                    self_attn_mask = (
                        (torch.ones_like(mask) * mask.transpose(1, 2))
                        .tril(self_attn_window)
                        .triu(-self_attn_window)
                        .unsqueeze(1)
                    )
                self_attn_mask = torch.where(
                    self_attn_mask.sum(-1, keepdim=True) != 0,
                    self_attn_mask,
                    torch.ones_like(self_attn_mask),
                )

        if cross_attn_mask is None:
            if key_value_states is None:
                cross_attn_mask = None
            else:
                if mask is None and key_value_mask is None:
                    cross_attn_mask = None
                else:
                    if key_value_mask is None:
                        key_value_mask = torch.ones(
                            (
                                key_value_states.size(0),
                                key_value_states.size(1),
                                1,
                            ),
                            dtype=torch.bool,
                            device=key_value_states.device,
                        )
                    cross_attn_mask = (
                        torch.ones(
                            (x.size(0), x.size(1), 1),
                            dtype=torch.bool,
                            device=x.device,
                        )
                        * key_value_mask.transpose(1, 2)
                    ).unsqueeze(1)

            if cross_attn_mask is not None and torch.all(cross_attn_mask):
                cross_attn_mask = None
        return self_attn_mask, cross_attn_mask


class EMAVariance(nn.Module):
    """
    Exponential Moving Average of Variance
    """

    def __init__(self, momentum: float = 0.999, initial_value: float = 1.0):
        super().__init__()
        self.momentum = momentum
        self.variance = nn.Parameter(
            torch.tensor(initial_value),
            requires_grad=False,
        )

    @torch.no_grad()
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        difference: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        :param difference: (Optional) a Tensor of distance between variables and the mean
        :param mask: (Optional) a mask Tensor
        :return: the updated variance
        """
        if not self.training or difference is None:
            return self.variance

        difference = difference.float()
        n = may_mask(torch.ones_like(difference), mask).sum().long()
        v_sum = may_mask(difference.pow(2), mask).sum()

        if torch.distributed.is_initialized():
            # gather values from all devices if distributed training
            torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(v_sum, op=torch.distributed.ReduceOp.SUM)

        v_avg = (v_sum / n.clamp_min(1)).type(self.variance.dtype)

        # updated variance
        self.variance.mul_(self.momentum)
        self.variance.add_((1 - self.momentum) * v_avg)
        return v_avg


class ProbabilisticVQ(nn.Module):
    def __init__(
        self,
        channels: int,
        num_mixtures: int,
        depth: int = 1,
        momentum: float = 0.999,
        logging_prefix: str = "vq",
    ):
        super().__init__()
        self.channels = channels
        self.num_mixtures = num_mixtures
        self.depth = depth
        self.momentum = momentum
        self.logging_prefix = logging_prefix

        self.mus_list = nn.ParameterList(
            [
                nn.Parameter(
                    F.normalize(torch.randn(num_mixtures, channels), p=2.0, dim=1)
                    * ((depth - i) / depth),
                    requires_grad=False,
                )
                for i in range(depth)
            ]
        )
        self._variance_list = nn.ModuleList([EMAVariance() for _ in range(depth)])

    @property
    def log_std(self) -> Tensor:
        return torch.log(self._variance_list[-1]()) * 0.5

    @overload
    def encode(self, z: Tensor, return_z_q: Literal[False]) -> list[Tensor]: ...

    @overload
    def encode(
        self, z: Tensor, return_z_q: Literal[True]
    ) -> tuple[list[Tensor], Tensor]: ...

    def encode(
        self, z: Tensor, return_z_q: bool = False
    ) -> list[Tensor] | tuple[list[Tensor], Tensor]:
        r = z
        ids_sel = []
        for i in range(self.depth):
            mus = self.mus_list[i]
            idx_sel = self._dist_sq(r, mus).argmin(-1)  # [b, ?, h], [v, h] -> [b, ?]
            r = r - F.embedding(idx_sel, mus)
            ids_sel.append(idx_sel)
        if return_z_q:
            return ids_sel, z - r
        return ids_sel

    def decode(self, ids_sel: list[Tensor], include_blank: bool = True) -> Tensor:
        z = torch.zeros((*ids_sel[0].size(), self.channels), device=ids_sel[0].device)
        for i in range(len(ids_sel)):
            mus = self.mus_list[i]
            if include_blank:
                mus = F.pad(mus, [0, 0, 0, 1])
            z = z + F.embedding(ids_sel[i], mus)
        return z  # [b, ?, h]

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, z: Tensor, mask: Tensor, return_commitment_loss: bool = False
    ) -> tuple[list[Tensor], Tensor] | tuple[list[Tensor], Tensor, Tensor]:
        z = z.float()

        ids_sel = []
        z_q = torch.zeros_like(z)

        r = z.view(-1, self.channels)  # [b x ?, h]
        if mask.dtype != torch.bool:
            mask = mask.bool()
        mask = mask.view(-1, 1)  # [b x ?, 1]

        if return_commitment_loss:
            commitment_loss = torch.zeros_like(r)
        for i in range(self.depth):
            with torch.no_grad():
                mus = self.mus_list[i].float()  # [v, h]
                dist_sq = self._dist_sq(r, mus)
                idx_sel = dist_sq.argmin(-1)  # [b x ?, h], [v, h] -> [b x ?]
                mus_sel = F.embedding(idx_sel, mus)  # [b x ?, h]

            r = r - mus_sel
            if return_commitment_loss:
                commitment_loss = commitment_loss + r.pow(2)

            with torch.no_grad():
                ids_sel.append(idx_sel.view(z_q.size()[:-1]))
                z_q = z_q + mus_sel.view_as(z_q)

                if self.training:
                    log_std = 0.5 * torch.log(self._variance_list[i](r, mask))

                    log_prob_normal = self._dist_sq_to_log_prob_normal(
                        dist_sq, log_std
                    ).view(-1, self.num_mixtures)  # [b x ?, v]

                    pi_logits = F.log_softmax(log_prob_normal, -1).masked_fill(
                        ~mask, torch.finfo(log_prob_normal.dtype).min
                    )  # [b x ?, v]
                    pi_norm = torch.softmax(pi_logits, 0).detach()

                    if torch.distributed.is_initialized():
                        max_logit = pi_logits.max(0, keepdim=True).values  # [1, v]
                        denom = torch.exp(pi_logits - max_logit).sum(
                            0, keepdim=True
                        )  # [1, v]
                        tensor_to_gather = torch.cat([max_logit, denom], 0)  # [1, v]
                        tensor_list = [
                            torch.zeros_like(tensor_to_gather)
                            for _ in range(torch.distributed.get_world_size())
                        ]
                        torch.distributed.all_gather(tensor_list, tensor_to_gather)

                        tensor_gathered = torch.stack(tensor_list, 0)  # [N, 2, v]
                        max_logit_gathered, denom_gathered = (
                            tensor_gathered[:, 0],
                            tensor_gathered[:, 1],
                        )  # [N, v] x 2
                        global_max_logit = max_logit_gathered.max(
                            0, keepdim=True
                        ).values  # [1, v]
                        denom_scaler = torch.exp(
                            max_logit_gathered - global_max_logit
                        )  # [N, v]

                        denom_total = (denom_gathered * denom_scaler).sum(
                            0, keepdim=True
                        )  # [N, v], [N, v] -> [1, v]
                        scaler = (
                            denom
                            * torch.exp(max_logit - global_max_logit)
                            / denom_total
                        )  # [1, v], [1, v] -> [1, v]
                        pi_norm = pi_norm * scaler

                    mus_new = pi_norm.transpose(0, 1) @ (r + mus_sel)  # [v, h]

                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(
                            mus_new, op=torch.distributed.ReduceOp.SUM
                        )

                    # update mus
                    self.mus_list[i].mul_(self.momentum)
                    self.mus_list[i].add_((1 - self.momentum) * mus_new)
        if return_commitment_loss:
            commitment_loss = (commitment_loss * mask).mean(-1).sum() / mask.sum()
            return ids_sel, z_q, commitment_loss
        return ids_sel, z_q

    def _dist_sq(self, z: Tensor, mus: Tensor) -> Tensor:
        """
        z: [b, ?, d?, h]
        mus: [d?, v, h]
        """
        return (
            z.pow(2).sum(-1, keepdim=True)  # [b, ?, d?, 1]
            + mus.pow(2).sum(-1)  # [d?, v]
            - 2
            * (z.unsqueeze(-2) @ mus.transpose(-1, -2)).squeeze(
                -2
            )  # [b, ?, d?, h] , [d?, h, v] -> [b, ?, d?, v]
        )

    def _dist_sq_to_log_prob_normal(self, dist_sq: Tensor, log_std: Tensor) -> Tensor:
        """
        dist_sq: [b, ?, d?, v]
        log_std: [d?, 1]
        """
        constants = (math.log(2 * math.pi) + 2 * log_std) * self.channels
        return -0.5 * (constants + torch.exp(-2 * log_std) * dist_sq)

    def _log_prob_normal(self, z: Tensor, mus: Tensor, log_std: Tensor) -> Tensor:
        """
        z: [b, ?, d?, h]
        mus: [d?, v, h]
        log_std: [d?, 1]
        """
        constants = math.log(2 * math.pi) + 2 * log_std
        return -0.5 * (
            (constants + z.pow(2) * torch.exp(-2 * log_std)).sum(
                -1, keepdim=True
            )  # [b, ?, d?, 1]
            + (mus.pow(2) * torch.exp(-2 * log_std.unsqueeze(-2))).sum(-1)  # [d?, v]
            - 2
            * (
                z.unsqueeze(-2)
                @ (mus * torch.exp(-2 * log_std.unsqueeze(-2))).transpose(-1, -2)
            ).squeeze(-2)  # [b, ?, d?, h] , [d?, h, v] -> [b, ?, d?, v]
        )


class LayerNorm(nn.LayerNorm):
    def __init__(self, num_channels: int):
        super().__init__(num_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, -1)
        x = super().forward(x)
        x = x.transpose(1, -1)
        return x


class CausalConv1dCache:
    def __init__(self) -> None:
        self.cache: dict[int, Tensor] = {}

    def __getitem__(self, layer_idx: int) -> Tensor:
        return self.cache[layer_idx]

    def update(
        self,
        states: Tensor,
        layer_idx: int,
        padding: int,
        padding_value: int = 0,
        flush: bool = False,
    ) -> Tensor:
        device = states.device
        dtype = states.dtype
        b, c, t = states.size()

        if layer_idx not in self.cache:
            padding_tensor = (
                torch.zeros((b, c, padding), dtype=dtype, device=device) + padding_value
            )
        else:
            padding_tensor = self.cache[layer_idx]
            assert padding_tensor.size(2) == padding
        padded_states = torch.cat([padding_tensor, states], 2)
        self.cache[layer_idx] = padded_states[:, :, -padding:]
        if flush:
            del self.cache[layer_idx]
        return padded_states


class CausalConvNeXtBlock(nn.Module):
    """Causal ConvNeXt 1D Block adapted from https://github.com/charactr-platform/vocos
    which is adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        identity_init: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.dwconv = nn.Sequential(
            nn.ConstantPad1d((6, 0), 0),
            nn.Conv1d(dim, dim, kernel_size=7, groups=dim),
        )  # depthwise conv
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)  # pointwise/1x1 convs
        self.act = nn.GELU()
        if identity_init:
            self.pwconv2 = zero_module(nn.Conv1d(intermediate_dim, dim, 1))
        else:
            self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
    ) -> Tensor:
        residual = x
        if cache is not None:
            if mask is not None:
                assert torch.all(mask)
            x = cache.update(x, self.layer_idx, 6, flush=flush)
            x = self.dwconv[1](x)
        else:
            x = self.dwconv(may_mask(x, mask))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = residual + x
        return may_mask(x, mask)


class GroupLin(nn.Module):
    def __init__(
        self, in_features, out_features, groups=1, bias=True, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert in_features % groups == 0
        assert out_features % groups == 0
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.in_features_per_group = in_features // groups
        self.weight = nn.Parameter(
            torch.empty(
                (groups, in_features // groups, out_features // groups),
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.in_features_per_group
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out_size = list(x.size())[:-1] + [self.out_features]
        x = torch.einsum(
            "bgi,gij->bgj",
            x.view(-1, self.groups, self.in_features_per_group),
            self.weight,
        ).reshape(*out_size)
        if self.bias is not None:
            return x + self.bias
        return x


class MoGProj(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out: int,
        num_predictions: int,
        num_groups: int = 1,
        d_low: int | None = None,
        bias: bool = True,
        num_splits: int = 1,
    ):
        super().__init__()
        if d_low is not None:
            assert d_low <= d_out
            if d_low == d_out:
                d_low = None
        if num_splits != 1:
            assert num_splits > 1
            assert d_model % num_splits == 0
            assert num_predictions % num_splits == 0
        self.d_out = d_out
        self.d_low = d_low
        self.num_predictions = num_predictions
        self.num_groups = num_groups
        self.num_splits = num_splits
        self._dist_var_buffer = None
        self._proj_params_buffer = None

        if d_low is None:
            if num_splits == 1:
                self.proj = nn.Linear(
                    d_model,
                    num_predictions
                    + num_predictions * num_groups * d_out
                    + num_groups * d_out
                    + num_groups,
                )
            else:
                self.proj = nn.Linear(
                    d_model,
                    num_predictions + num_groups * d_out + num_groups,
                )
                self.proj_else = GroupLin(
                    d_model,
                    num_predictions * num_groups * d_out,
                    groups=num_splits,
                )
        else:
            assert d_low < d_out
            if num_splits == 1:
                self.proj = nn.Linear(
                    d_model,
                    num_predictions
                    + num_predictions * num_groups * d_low
                    + num_groups * d_out
                    + num_groups,
                )
            else:
                self.proj = nn.Linear(
                    d_model,
                    num_predictions + num_groups * d_out + num_groups,
                )
                self.proj_else = GroupLin(
                    d_model,
                    num_predictions * num_groups * d_low,
                    groups=num_splits,
                )

            self.bias: nn.Parameter | None = (
                nn.Parameter(torch.zeros(num_predictions, num_groups, d_out))
                if bias
                else None
            )
            self.low_mat = nn.Parameter(
                torch.randn(num_predictions, num_groups, d_out, d_low) * (d_low**-0.5)
            )

    def store_dist_vars(self):
        self._dist_var_buffer = (
            self.low_mat.transpose(-1, -2) @ self.low_mat,
            self.bias.pow(2).sum(-1) if self.bias is not None else None,
            (
                torch.einsum("ngi,ngij->ngj", self.bias, self.low_mat)
                if self.bias is not None
                else None
            ),
        )

    def clear_dist_vars(self):
        self._dist_var_buffer = None

    def store_proj_params_splits(self):
        assert self.num_splits == 1
        n = self.num_predictions
        g = self.num_groups
        d = self.d_low or self.d_out

        proj_weight = torch.cat(
            [
                self.proj.weight.detach()[:n],
                self.proj.weight.detach()[n + n * g * d : -g],
            ],
            0,
        )
        else_weight = self.proj.weight.detach()[n : n + n * g * d]
        logs_weight = self.proj.weight.detach()[-g:]
        proj_bias = torch.cat(
            [self.proj.bias.detach()[:n], self.proj.bias.detach()[n + n * g * d : -g]],
            0,
        )
        else_bias = self.proj.bias.detach()[n : n + n * g * d]
        logs_bias = self.proj.bias.detach()[-g:]

        self._proj_params_buffer = (
            proj_weight,
            else_weight,
            logs_weight,
            proj_bias,
            else_bias,
            logs_bias,
        )

    def clear_proj_params(self):
        self._proj_params_buffer = None

    def infer(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        classifier_free_guidance: float = 0.0,
        top_p_or_k: float | int = 1.0,
        min_log_std: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, t, _ = x.size()
        n = self.num_predictions
        g = self.num_groups
        d = self.d_low or self.d_out

        assert self.num_splits == 1
        assert self.num_groups == 1
        if self._proj_params_buffer is not None:
            (
                proj_weight,
                else_weight,
                logs_weight,
                proj_bias,
                else_bias,
                logs_bias,
            ) = self._proj_params_buffer
        else:
            proj_weight = torch.cat(
                [
                    self.proj.weight.detach()[:n],
                    self.proj.weight.detach()[n + n * g * d : -g],
                ],
                0,
            )
            else_weight = self.proj.weight.detach()[n : n + n * g * d]
            logs_weight = self.proj.weight.detach()[-g:]
            proj_bias = torch.cat(
                [
                    self.proj.bias.detach()[:n],
                    self.proj.bias.detach()[n + n * g * d : -g],
                ],
                0,
            )
            else_bias = self.proj.bias.detach()[n : n + n * g * d]
            logs_bias = self.proj.bias.detach()[-g:]

        if classifier_free_guidance > 0:
            b //= 2
            x_a, x_b = x.chunk(2, dim=0)
            log_std = F.linear(
                x_a + classifier_free_guidance * (x_a - x_b), logs_weight, logs_bias
            )

            x = x_a + classifier_free_guidance * (x_a - x_b)
        else:
            log_std = F.linear(x, logs_weight, logs_bias)
        log_std = log_std.view(b, t, g, 1)
        if min_log_std is not None:
            log_std = log_std.clamp_min(min_log_std)

        stats = may_mask(F.linear(x, proj_weight, proj_bias), mask)
        logits, mu_res = torch.split(stats, [n, g * self.d_out], -1)
        logits = F.log_softmax(
            (
                TopPLogitsWarper(top_p_or_k)(
                    None,
                    logits.view(-1, n),
                ).view(*logits.size())
                if isinstance(top_p_or_k, float)
                else TopKLogitsWarper(top_p_or_k)(
                    None,
                    logits.view(-1, n),
                ).view(*logits.size())
            ),
            -1,
        )
        mu_res = mu_res.view(b, t, g, self.d_out)

        # sampling
        mixture_indices = (logits + gumbel_like(logits)).argmax(-1)
        logit = torch.gather(logits, 2, mixture_indices.unsqueeze(-1)).squeeze(-1)

        mu = (
            batch_matmul(
                x.view(b * t, -1),
                else_weight.view(n, d, -1),
                mixture_indices.view(b * t),
            ).view(b, t, g, d)
            + else_bias.view(n, g, d)[mixture_indices]
        )
        if self.d_low is not None:
            batch_matmul_func = (
                batch_matmul
                if math.log2(d).is_integer() and math.log2(self.d_out).is_integer()
                else batch_matmul_pytorch
            )
            mu = batch_matmul_func(
                mu.view(b * t, -1),
                self.low_mat.reshape(n, self.d_out, -1).detach(),
                mixture_indices.view(b * t),
                BLOCK_SIZE_DIN=d,
                BLOCK_SIZE_DOUT=self.d_out,
            ).view(b, t, g, self.d_out)
            if self.bias is not None:
                mu = mu + self.bias[mixture_indices]
            # mu = self.proj_out(mu, mixture_indices)
        mu = mu * torch.exp(log_std) + mu_res

        return logit, mu, log_std

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b, t, _ = x.size()
        n = self.num_predictions
        g = self.num_groups
        d = self.d_low or self.d_out

        stats = may_mask(self.proj(x), mask)
        if self.num_splits == 1:
            logits, mus, mu_res, log_std = torch.split(
                stats, [n, n * g * d, g * self.d_out, g], -1
            )
        else:
            logits, mu_res, log_std = torch.split(stats, [n, g * self.d_out, g], -1)
            mus = may_mask(self.proj_else(x), mask)
        mus = mus.view(b, t, n, g, d)
        mu_res = mu_res.view(b, t, g, self.d_out)
        log_std = log_std.view(b, t, g, 1)
        return logits, mus, mu_res, log_std

    def dist(self, mus: Tensor, mu: Tensor) -> Tensor:
        """
        mus: [b, t, n, g, d]
        mu: [b, t, g, d]

        return: [b, t, n, g]
        """
        if self.d_low is None:
            return (mus - mu.unsqueeze(-3)).pow(2).sum(-1)
        else:
            if self._dist_var_buffer is not None:
                low_mat_sq, b_sq, wb = self._dist_var_buffer
            else:
                low_mat_sq, b_sq, wb = (
                    self.low_mat.transpose(-1, -2) @ self.low_mat,
                    self.bias.pow(2).sum(-1) if self.bias is not None else None,
                    (
                        torch.einsum("ngi,ngij->ngj", self.bias, self.low_mat)
                        if self.bias is not None
                        else None
                    ),
                )
            x, y = mus, mu
            b, t, n, g, d_l = x.size()
            wx_sq = (
                x
                * torch.einsum(
                    "btngi,ngij->btngj",
                    x,
                    low_mat_sq.to(x),
                )
            ).sum(-1)  # [b, t, n, g]
            y_sq = y.pow(2).sum(-1).unsqueeze(-2)  # [b, t, 1, g]
            xwy = (x * torch.einsum("btgi,ngij->btngj", y, self.low_mat.to(y))).sum(
                -1
            )  # [b, t, n, g, d_l], [n, g, d_i, d_l], [b, t, g, d_i] -> [b, t, n, g]

            dist = wx_sq + y_sq - 2 * xwy
            if self.bias is not None:
                by = torch.einsum("btgi,ngi->btng", y, self.bias.to(y))  # [b, t, n, g]
                xwb = (x * wb).sum(-1)  # [b, t, n, g]
                dist = dist + (b_sq - 2 * by + 2 * xwb)
            # if self.training and torch.any(dist < 0):
            #     logger.info("negative distance encountered.")
            return torch.abs(dist)

    def proj_out(
        self, mu: Tensor, mixture_indices: Tensor, inplace: bool = False
    ) -> Tensor:
        if self.d_low is None:
            return mu
        else:
            mu = torch.einsum("btgi,btgji->btgj", mu, self.low_mat[mixture_indices])
            if self.bias is not None:
                mu = mu + self.bias[mixture_indices]
            return mu


class MLPStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        dropout_rate: float = 0.1,
        eps: float = 1e-6,
        d_cond: int | None = None,
        gated_act: bool = True,
        post_norm: bool = False,
    ):
        super().__init__()
        self.d_cond = d_cond

        self.blocks = nn.ModuleList(
            [
                LayerFF(
                    d_model,
                    d_ff,
                    dropout_rate,
                    eps,
                    cond=d_cond is not None,
                    gated_act=gated_act,
                    post_norm=post_norm,
                )
                for i in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=eps)
        self.adaLN_emb = nn.Parameter(torch.zeros(2, d_model))
        if d_cond is not None:
            self.c_block = nn.Sequential(
                nn.Linear(d_cond, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, 5 * d_model),
            )

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        gradient_checkpointing: bool = False,
    ) -> Tensor:
        b, t, d = x.size()
        if c is not None and self.d_cond is not None:
            c, c_final = self.c_block(c).split([3 * d, 2 * d], dim=2)
            shift, scale = may_squeeze_chunk(
                self.adaLN_emb + c_final.view(b, t, 2, -1), 2, dim=2
            )
        else:
            shift, scale = self.adaLN_emb[0], self.adaLN_emb[1]
        for block in self.blocks:
            if self.training and gradient_checkpointing:
                x = checkpoint(block.forward, x, c, use_reentrant=False)
            else:
                x = block(x, c)
        x = self.final_norm(x)
        x = (1 + scale) * x + shift
        return x


@dataclass
class RVQVAEConfig:
    # model specific configs
    base_dim: int = 256
    num_blocks: int = 2
    channel_mult: tuple[int, ...] = (1, 2, 4)
    downsample_rates: tuple[int, ...] = (8, 8, 8)

    # stft
    n_fft: int = 32
    hop_length: int = 8

    # p-RVQ
    latent_dim: int = 256
    num_mixtures: int = 1024
    depth: int = 64
    groups: int = 1

    def __post_init__(self):
        assert len(self.channel_mult) == len(self.downsample_rates)


class RVQVAE(nn.Module):
    def __init__(
        self,
        config: Config,
        encoder_freeze: bool = False,
    ):
        super().__init__()
        field_names = {f.name for f in fields(RVQVAEConfig)}

        self.config = RVQVAEConfig(
            **{k: v for k, v in config.items() if k in field_names}
        )

        self.encoder_freeze = encoder_freeze

        self.prvq = ProbabilisticVQ(
            channels=self.config.latent_dim,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

        base_dim = self.config.base_dim
        num_levels = len(self.config.channel_mult)
        num_blocks = self.config.num_blocks
        self.spec_cache_idx = num_levels * num_blocks

        encoder_modules: list[nn.Module] = [
            nn.Conv1d(self.config.n_fft + 2, base_dim, 1, bias=False)
        ]
        for i in range(num_levels):
            ch_mult, ds_rate = (
                self.config.channel_mult[i],
                self.config.downsample_rates[i],
            )
            dim = base_dim * ch_mult
            for j in range(num_blocks):
                encoder_modules.append(
                    CausalConvNeXtBlock(
                        dim, dim * 4, True, layer_idx=i * num_blocks + j
                    )
                )
            encoder_modules.append(
                nn.Conv1d(
                    dim,
                    (
                        base_dim * self.config.channel_mult[i + 1]
                        if i != num_levels - 1
                        else self.config.latent_dim
                    ),
                    ds_rate,
                    stride=ds_rate,
                    bias=False,
                )
            )
        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules: list[nn.Module] = []
        for i in reversed(range(num_levels)):
            ch_mult, ds_rate = (
                self.config.channel_mult[i],
                self.config.downsample_rates[i],
            )
            dim = base_dim * ch_mult
            decoder_modules.append(
                nn.ConvTranspose1d(
                    (
                        base_dim * self.config.channel_mult[i + 1]
                        if i != num_levels - 1
                        else self.config.latent_dim
                    ),
                    dim,
                    ds_rate,
                    stride=ds_rate,
                    bias=False,
                )
            )
            for j in range(num_blocks):
                decoder_modules.append(
                    CausalConvNeXtBlock(
                        dim, dim * 4, True, layer_idx=i * num_blocks + j
                    )
                )
        decoder_modules.append(
            nn.Conv1d(base_dim, self.config.n_fft + 2, 1, bias=False)
        )
        self.decoder = nn.Sequential(*decoder_modules)

        if encoder_freeze:
            for p in self.prvq.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_dir: str,
        cfg: Config | None = None,
        encoder_freeze: bool = True,
    ) -> "RVQVAE":
        pretrained_cfg = get_config_from_dir(pretrained_dir).model
        if cfg is not None:
            pretrained_cfg.update(cfg)
            logger.info(
                f"The loaded config of the pretrained model is updated to: {pretrained_cfg}"
            )
        model = cls(pretrained_cfg, encoder_freeze=encoder_freeze)
        model_state_dict = model.state_dict()
        model_state_dict.update(
            torch.load(
                latest_checkpoint_path(pretrained_dir, "ema_*.pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        model.load_state_dict(model_state_dict)
        return model

    def get_optimizer_param_groups(self, weight_decay: float = 0.0) -> list[dict]:
        def _get_conv_weight_names(module):
            """
            Returns the names of the model parameters that are not inside a forbidden layer.
            """
            result = []
            if any(
                isinstance(module, conv_module)
                for conv_module in [nn.Conv1d, nn.ConvTranspose1d]
            ):
                result.append("weight")
                return result
            else:
                for name, child in module.named_children():
                    result += [f"{name}.{n}" for n in _get_conv_weight_names(child)]
            return result

        params_w_decay, params_wo_decay = [], []

        param_names_w_decay = set(_get_conv_weight_names(self))
        for n, p in self.named_parameters():
            if p.requires_grad:
                if n in param_names_w_decay:
                    params_w_decay.append(p)
                else:
                    params_wo_decay.append(p)
        return [
            {"params": params_w_decay, "weight_decay": weight_decay},
            {"params": params_wo_decay, "weight_decay": 0.0},
        ]

    def ae_encode(
        self, x: Tensor, mask: Tensor, cache: CausalConv1dCache | None = None
    ) -> tuple[Tensor, Tensor]:
        assert x.size(1) == 1 and x.dim() == 3
        b, _, t = x.size()
        assert (
            t % (self.config.hop_length * math.prod(self.config.downsample_rates)) == 0
        )
        if cache is not None:
            raise NotImplementedError  # TODO

        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = spectrogram(
                x.float().squeeze(1),
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.n_fft,
                window_fn=torch.hann_window,
            )
            mag, ph = torch.view_as_real(spec).chunk(2, dim=-1)
            x = torch.cat([mag, ph], 1).squeeze(-1)
        for layer in self.encoder:
            if isinstance(layer, CausalConvNeXtBlock):
                x = layer(x, cache=cache)
            else:
                x = layer(x)

        mask = may_resize_mask(x, mask, dim=2)
        return may_mask(x, mask), mask

    def ae_decode(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        constrain_value_range: bool = False,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if cache is not None:
            if mask is not None:
                assert torch.all(mask)
        for layer in self.decoder:
            if isinstance(layer, CausalConvNeXtBlock):
                x = layer(x, cache=cache, flush=flush)
            else:
                x = layer(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            max_mag = 100.0
            mag, ph = x.chunk(2, dim=1)
            mag = max_mag * torch.exp(
                -F.softplus(-mag + math.log(max_mag))
            )  # safeguard to prevent excessively large magnitudes
            # wrapping happens here. These two lines produce real and imaginary value
            mag_l, mag, mag_r = mag.split([1, mag.size(1) - 2, 1], dim=1)
            real_l, real, real_r = torch.cos(ph).split([1, ph.size(1) - 2, 1], dim=1)
            imag_l, imag, imag_r = torch.sin(ph).split([1, ph.size(1) - 2, 1], dim=1)

            # recalculating phase here does not produce anything new
            # only costs time
            # better directly produce the complex value
            spec = torch.cat(
                [mag_l * real_l, mag * (real + 1j * imag), mag_r * real_r], 1
            )
            if cache is not None:
                half_spec_padding = math.ceil(
                    ((self.config.n_fft - self.config.hop_length) // 2)
                    / self.config.hop_length
                )
                spec = cache.update(
                    spec,
                    self.spec_cache_idx,
                    padding=half_spec_padding * 2,
                    flush=flush,
                )
                if flush:
                    spec = F.pad(spec, [0, half_spec_padding])
            x = spec_to_wav(
                spec,
                self.config.n_fft,
                self.config.hop_length,
                self.config.n_fft,
                constrain_value_range=constrain_value_range,
            ).unsqueeze(1)
        if cache is not None:
            half_wav_padding = half_spec_padding * self.config.hop_length
            x = x[:, :, half_wav_padding:-half_wav_padding]
            if mask is not None:
                mask = torch.ones_like(x[:, :1]).bool()
            return x, mask
        else:
            mask = may_resize_mask(x, mask, dim=2)
            return may_mask(x, mask), mask

    def encode(
        self, x: Tensor, x_len: Tensor, cache: CausalConv1dCache | None = None
    ) -> tuple[Tensor, Tensor]:
        mask = sequence_mask(x_len).unsqueeze(1)
        z, z_mask = self.ae_encode(x, mask, cache=cache)

        ids_sel = self.prvq.encode(z.transpose(1, 2), return_z_q=False)
        return torch.stack(ids_sel, -1), z_mask.sum([1, 2]).long()

    def decode(
        self,
        code: Tensor,
        code_len: Tensor | None = None,
        constrain_value_range: bool = False,
        cache: CausalConv1dCache | None = None,
        flush: bool = False,
        include_blank: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        ids_sel = [x.squeeze(-1) for x in torch.split(code, 1, -1)]
        z_mask: Tensor | None = None
        if code_len is not None:
            z_mask = sequence_mask(code_len).unsqueeze(1)

        z_q = self.prvq.decode(ids_sel, include_blank=include_blank).transpose(1, 2)
        x_hat, mask = self.ae_decode(
            z_q,
            z_mask,
            constrain_value_range=constrain_value_range,
            cache=cache,
            flush=flush,
        )
        return x_hat, mask.sum([1, 2]).long() if mask is not None else None

    def forward(
        self,
        x: Tensor,
        x_len: Tensor,
        constrain_value_range: bool = False,
    ) -> tuple[Tensor, Tensor] | Tensor:
        if not self.encoder_freeze:
            mask = sequence_mask(x_len).unsqueeze(1)
            z, z_mask = self.ae_encode(x, mask)
            ids_sel, z_q, commitment_loss = self.prvq(
                z.transpose(1, 2), z_mask.transpose(1, 2), return_commitment_loss=True
            )
            z_q = z_q.transpose(1, 2)
            x_hat, _ = self.ae_decode(z_q + (z - z.detach()), z_mask)
            return x_hat, commitment_loss
        else:
            with torch.no_grad():
                mask = sequence_mask(x_len).unsqueeze(1)
                z, z_mask = self.ae_encode(x, mask)
                z_q = self.prvq.encode(z.transpose(1, 2), return_z_q=True)[1].transpose(
                    1, 2
                )
            x_hat, _ = self.ae_decode(
                z_q, z_mask, constrain_value_range=constrain_value_range
            )
            return x_hat


@dataclass
class CharAwareSubwordConfig:
    # init configs
    pretrained_tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # model specific configs
    d_ff: int = 4608
    d_model: int = 1152
    num_heads: int = 16
    num_layers: int = 1
    dropout_rate: float = 0.1
    eps: float = 1e-6
    self_attn_window: int | None = None
    rotary_value: bool = False
    gated_act: bool = False
    post_norm: bool = False
    attn_logit_softcapping: float | None = None
    num_key_value_heads: int | None = None
    gradient_checkpointing: bool = False

    # decoder specific configs
    num_langs: int = 109


def build_vocabs(
    pretrained_tokenizer_name: str, vocab_save_path: str | None = None
) -> tuple[dict[int, tuple[int, ...]], dict[str, int], int]:
    char_vocab: dict[str, int]
    if vocab_save_path is not None and os.path.exists(vocab_save_path):
        with open(vocab_save_path) as f:
            data = json.load(f)
        subword_id_to_char_ids = {
            int(k): v for k, v in data["subword_id_to_char_ids"].items()
        }
        char_vocab = data["char_vocab"]
        subword_padding_idx = data["subword_padding_idx"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)

        org_char_vocab = {
            subword: subword_id
            for subword, subword_id in tokenizer.vocab.items()
            if len(subword) == 1
        }
        sorted_char_vocab = dict(sorted(org_char_vocab.items(), key=lambda x: x[1]))
        char_vocab = {k: i for i, (k, _) in enumerate(sorted_char_vocab.items())}
        assert sorted(char_vocab.values()) == list(range(len(char_vocab)))
        subword_id_to_char_ids = {
            subword_id: tuple(char_vocab[char] for char in subword)
            for subword, subword_id in tokenizer.vocab.items()
        }

        assert max(subword_id_to_char_ids) == len(subword_id_to_char_ids) - 1
        # add padding_idx
        subword_padding_idx = len(subword_id_to_char_ids)
        subword_id_to_char_ids[subword_padding_idx] = (len(char_vocab),)

        if vocab_save_path is not None:
            if is_first_process():
                with open(vocab_save_path, "w") as f:
                    json.dump(
                        {
                            "subword_id_to_char_ids": subword_id_to_char_ids,
                            "char_vocab": char_vocab,
                            "subword_padding_idx": subword_padding_idx,
                        },
                        f,
                    )
    return subword_id_to_char_ids, char_vocab, subword_padding_idx


class CharAwareSubwordEncoder(nn.Module):
    def __init__(self, config: Config, vocab_save_path: str | None = None):
        super().__init__()
        field_names = {f.name for f in fields(CharAwareSubwordConfig)}
        self.config = CharAwareSubwordConfig(
            **{k: v for k, v in config.items() if k in field_names}
        )

        self.subword_id_to_char_ids, self.char_vocab, self.subword_padding_idx = (
            build_vocabs(self.config.pretrained_tokenizer_name, vocab_save_path)
        )

        self.pos_embedding = partial(
            TimestepEmbedder.timestep_embedding,
            dim=self.config.d_model // self.config.num_heads,
        )

        self.embed_tokens = nn.Embedding(
            self.vocab_size + 1, self.config.d_model, padding_idx=self.vocab_size
        )

        self.encoder = Stack(
            self.config.d_model,
            self.config.d_ff,
            self.config.num_heads,
            self.config.num_layers,
            dropout_rate=self.config.dropout_rate,
            attention_dropout_rate=self.config.dropout_rate,
            eps=self.config.eps,
            cross_attn=False,
            self_attn_window=self.config.self_attn_window,
            d_cond=None,
            gated_act=self.config.gated_act,
            post_norm=self.config.post_norm,
            logit_softcapping=self.config.attn_logit_softcapping,
            num_key_value_heads=self.config.num_key_value_heads,
        )

    @property
    def vocab_size(self):
        return len(self.char_vocab)

    def prepare_inputs(
        self, subword_ids: Tensor, padding_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        device = subword_ids.device

        subword_id_list = torch.masked_select(subword_ids, padding_mask).cpu().tolist()
        char_id_list = [list(self.subword_id_to_char_ids[x]) for x in subword_id_list]

        char_lengths = torch.tensor(
            [len(x) for x in char_id_list], dtype=torch.long, device=device
        )
        batch_size = char_lengths.size(0)

        char_ids = torch.full(
            (batch_size, int(char_lengths.max().item())),
            self.vocab_size,
            dtype=torch.long,
        )
        for i in range(batch_size):
            char_ids[i, : char_lengths[i]] = torch.tensor(char_id_list[i])
        char_ids = char_ids.to(device=device)
        return char_ids, char_lengths

    def forward(
        self, subword_ids: Tensor, subword_mask: Tensor | None = None
    ) -> Tensor:
        device = subword_ids.device
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids).bool()
        if subword_mask.ndim == 3:
            subword_mask = subword_mask.squeeze(-1)

        char_ids, char_lengths = self.prepare_inputs(subword_ids, subword_mask)
        char_mask = sequence_mask(char_lengths).unsqueeze(-1)

        sinusoidal_pos = self.pos_embedding(
            torch.arange(char_ids.size(1)).to(device=device)
        ).unsqueeze(0)
        x = self.encoder(
            self.embed_tokens(char_ids),
            mask=char_mask,
            sinusoidal_pos=sinusoidal_pos,
            rotary_value=self.config.rotary_value,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )

        mean_emb = ((x / char_mask.sum(1, keepdim=True)) * char_mask).sum(1)
        subword_emb = torch.zeros(
            (subword_mask.size(0), subword_mask.size(1), mean_emb.size(-1)),
            device=device,
        )
        subword_emb[subword_mask.unsqueeze(-1).expand(-1, -1, mean_emb.size(-1))] = (
            mean_emb.view(-1)
        )
        return subword_emb


@dataclass
class ModelConfig:
    # model specific configs
    d_in: int = 512
    d_ff: int = 4608
    d_model: int = 1152
    num_heads: int = 16
    num_layers: int = 28
    dropout_rate: float = 0.1
    eps: float = 1e-6
    self_attn_window: int | None = None
    rotary_value: bool = False
    gated_act: bool = False
    post_norm: bool = False
    attn_logit_softcapping: float | None = None
    num_key_value_heads: int | None = None
    gradient_checkpointing: bool = False

    # method specific configs
    num_mlp_layers: int = 3
    d_depth: int = 16
    d_low: int | None = 64
    num_predictions: int = 1024
    num_splits: int = 1
    label_smoothing: float = 0.01
    masking_mode: MASKING_MODE = "coarse_first"
    max_training_ratio: float = 0.8
    min_log_std: float = -4.0
    p_uncond: float = 0.1
    p_lm_script: float = 0.5

    # conditional inputs
    d_lm: int | None = 4096
    char_aware_subword_config: Config | None = None


@torch.compile
def depthsum_encoding_step(
    embs: Tensor,
    r: Tensor,
    code: Tensor,
    depth_str: int = 0,
    k: int = 72,
) -> Tensor:
    for i in range(depth_str, depth_str + k):
        idx_sel = (
            embs[i].pow(2).sum(-1)  # [g?, v]
            - 2
            * (r.unsqueeze(-2) @ embs[i].transpose(-1, -2)).squeeze(
                -2
            )  # [b, ?, g?, h] , [g?, h, v] -> [b, ?, g?, v]
        ).argmin(-1)
        emb_i = F.embedding(idx_sel, embs[i])
        r = r - emb_i
        code[..., i : i + 1] = idx_sel
    return code


class Model(nn.Module):
    def __init__(
        self,
        config: Config,
        mus: Tensor,
        vocab_save_path: str | None = None,
    ):
        super().__init__()
        field_names = {f.name for f in fields(ModelConfig)}
        self.config = ModelConfig(
            **{k: v for k, v in config.items() if k in field_names}
        )

        self.depth = mus.size(0)
        self.vocab_size = mus.size(1)
        self.blank_id = mus.size(1)
        self.register_buffer("mus", mus.detach().clone())

        self.pos_embedding = partial(
            TimestepEmbedder.timestep_embedding,
            dim=self.config.d_model // self.config.num_heads,
        )
        if self.config.d_lm is None:
            self.lm_proj = None
        else:
            self.lm_proj = nn.Linear(self.config.d_lm, self.config.d_model)

        if self.config.char_aware_subword_config is None:
            self.cas_encoder = None
            self.cas_proj = None
        else:
            self.cas_encoder = CharAwareSubwordEncoder(
                self.config.char_aware_subword_config, vocab_save_path
            )
            self.cas_proj = nn.Linear(
                self.cas_encoder.config.d_model, self.config.d_model
            )

        self.decoder_null_emb = nn.Parameter(torch.randn((self.config.d_model,)))
        self.decoder_pre = nn.Linear(self.config.d_in, self.config.d_model)
        self.decoder = Stack(
            self.config.d_model,
            self.config.d_ff,
            self.config.num_heads,
            self.config.num_layers,
            dropout_rate=self.config.dropout_rate,
            attention_dropout_rate=self.config.dropout_rate,
            eps=self.config.eps,
            cross_attn=False,
            self_attn_window=self.config.self_attn_window,
            d_cond=None,
            gated_act=self.config.gated_act,
            post_norm=self.config.post_norm,
            logit_softcapping=self.config.attn_logit_softcapping,
            num_key_value_heads=self.config.num_key_value_heads,
        )

        self.eos_head = nn.Linear(self.config.d_model, 1)

        self.depth_embs = nn.Parameter(
            torch.randn((self.depth, 2, self.config.d_depth))
        )
        self.mlp_null_emb = nn.Parameter(torch.randn((self.config.d_model,)))
        self.mlp_pre = nn.Linear(self.config.d_in, self.config.d_model)
        self.mlp = MLPStack(
            self.config.d_model,
            self.config.d_ff,
            self.config.num_mlp_layers,
            dropout_rate=self.config.dropout_rate,
            eps=self.config.eps,
            d_cond=self.depth * self.config.d_depth,
            gated_act=self.config.gated_act,
            post_norm=self.config.post_norm,
        )
        self.mlp_post = MoGProj(
            d_model=self.config.d_model,
            d_out=self.config.d_in,
            num_predictions=self.config.num_predictions,
            num_groups=1,
            d_low=self.config.d_low,
            num_splits=self.config.num_splits,
        )

    def get_optimizer_param_groups(self, weight_decay: float = 0.0) -> list[dict]:
        def _get_linear_weight_names(module):
            """
            Returns the names of the model parameters that are not inside a forbidden layer.
            """
            result = []
            if isinstance(module, nn.Linear):
                result.append("weight")
                return result
            else:
                for name, child in module.named_children():
                    result += [f"{name}.{n}" for n in _get_linear_weight_names(child)]
            return result

        params_w_decay, params_wo_decay = [], []

        param_names_w_decay = {
            "decoder." + x for x in _get_linear_weight_names(self.decoder)
        }
        if self.cas_encoder is not None:
            param_names_w_decay.update(
                {
                    "cas_encoder.encoder." + x
                    for x in _get_linear_weight_names(self.cas_encoder.encoder)
                }
            )

        for n, p in self.named_parameters():
            if p.requires_grad:
                if n in param_names_w_decay:
                    params_w_decay.append(p)
                else:
                    params_wo_decay.append(p)
        return [
            {"params": params_w_decay, "weight_decay": weight_decay},
            {"params": params_wo_decay, "weight_decay": 0.0},
        ]

    def depthwise_embedding(
        self, ids: Tensor, embs: Tensor, include_blank: bool = False
    ) -> Tensor:
        """
        ids: [b, t, d]
        embs: [d, v, h]

        ret: [b, t, d, h]
        """
        depth, vocab_size, _ = embs.size()
        device = ids.device

        if include_blank:
            vocab_size = vocab_size + 1
            embs = F.pad(embs, [0, 0, 0, 1])

        ret = F.embedding(
            ids + torch.arange(depth).to(device=device)[: ids.size(-1)] * vocab_size,
            embs.view(depth * vocab_size, -1),
        )
        return ret

    def depthsum_embedding(
        self, ids: Tensor, embs: Tensor, include_blank: bool = False
    ) -> Tensor:
        """
        ids: [b, t, d]
        embs: [d, v, h]

        ret: [b, t, 1, h]
        """
        b, t, d = ids.size()
        _, v, h = embs.size()
        device = ids.device

        ret = torch.zeros((b, t, h), device=device)
        if include_blank:
            embs = F.pad(embs, [0, 0, 0, 1])
        for i in range(d):
            emb = embs[i]
            ret = ret + F.embedding(ids[..., i], emb)
        return ret.unsqueeze(-2)

    def dist_sq(self, z: Tensor, mus: Tensor) -> Tensor:
        """
        z: [b, ?, g?, h]
        mus: [g?, v, h]
        """
        return (
            z.pow(2).sum(-1, keepdim=True)  # [b, ?, g?, 1]
            + mus.pow(2).sum(-1)  # [g?, v]
            - 2
            * (z.unsqueeze(-2) @ mus.transpose(-1, -2)).squeeze(
                -2
            )  # [b, ?, g?, h] , [g?, h, v] -> [b, ?, g?, v]
        )

    def preprocess_inputs(self, code: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, t, d = code.size()
        device = code.device

        ratio = torch.rand(b * t, device=device) * self.config.max_training_ratio
        masking_ratio = get_masking_ratio(ratio)
        num_masking = torch.ceil(masking_ratio * d).long()

        code_mask = torch.ones((b * t, 1, d), dtype=torch.bool, device=device)
        code_mask = get_random_mask(
            code_mask,
            code_mask[..., :1],
            num_masking,
            masking_mode=self.config.masking_mode,
        )
        code_mask = code_mask.view(b, t, d)
        masked_code = code * code_mask + (torch.zeros_like(code) + self.blank_id) * (
            ~code_mask
        )

        z = self.depthsum_embedding(code, self.mus, include_blank=True).detach()

        return (
            masked_code,
            code_mask,
            z,
        )

    def forward_mlp(
        self,
        input_emb: Tensor,
        depth_emb: Tensor,
        hidden_state: Tensor,
    ) -> Tensor:
        x = self.mlp_pre(input_emb) + hidden_state
        x = self.mlp(
            x,
            depth_emb,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )
        return x

    def forward_decoder(
        self,
        input_emb: Tensor,
        cond: Tensor,
        sinusoidal_pos: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> Tensor:
        x = self.decoder_pre(input_emb) + cond
        x = self.decoder(
            x,
            is_causal=True if attention_mask is None else False,
            self_attn_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            rotary_value=self.config.rotary_value,
            gradient_checkpointing=self.config.gradient_checkpointing,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        return x

    def forward(
        self,
        code: Tensor,
        audio_mask: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        lm_hidden_state: Tensor | None = None,
        subword_ids: Tensor | None = None,
        subword_mask: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, t, d = code.size()
        device = code.device

        (random_code, random_code_mask, z) = self.preprocess_inputs(code)

        if position_ids is None:
            position_ids = torch.arange(t).to(device=device).unsqueeze(0)
        sinusoidal_pos = self.pos_embedding(position_ids)

        uncond_dec_flag = torch.rand((b, 1, 1), device=device) < self.config.p_uncond
        cond = torch.zeros((b, t, self.config.d_model), device=device)
        if self.lm_proj is not None and lm_hidden_state is not None:
            script_mask = torch.where(
                torch.rand((b, 1, 1), device=device) < self.config.p_lm_script,
                torch.ones_like(audio_mask),
                ~audio_mask,
            )
            cond = cond + self.lm_proj(lm_hidden_state * script_mask)
        if (
            self.config.char_aware_subword_config is not None
            and subword_ids is not None
        ):
            assert self.cas_encoder is not None
            assert self.cas_proj is not None
            if subword_mask is None:
                subword_mask = audio_mask
                if lm_hidden_state is not None:
                    subword_mask = torch.logical_and(
                        subword_mask,
                        torch.any(lm_hidden_state != 0, dim=-1, keepdim=True),
                    )
            cas_emb = self.cas_encoder(subword_ids, subword_mask)
            cond = cond + self.cas_proj(cas_emb)
        decoder_input_emb = F.pad(z.view(b, t, -1)[:, :-1], [0, 0, 1, 0])
        h_dec = self.forward_decoder(
            decoder_input_emb,
            torch.where(
                uncond_dec_flag, torch.zeros_like(cond) + self.decoder_null_emb, cond
            ),
            sinusoidal_pos,
            attention_mask=attention_mask,
        )

        eos_logit = self.eos_head(h_dec)

        uncond_mlp_flag = (
            torch.rand((b, t, 1), device=device) < 0.0
        )  # self.config.p_uncond
        mlp_input_emb = (
            self.depthsum_embedding(random_code, self.mus, include_blank=True)
            .detach()
            .view(b, t, -1)
        )
        depth_emb = self.depthwise_embedding(
            random_code_mask.long(), self.depth_embs
        ).view(b, t, -1)
        h_mlp = self.forward_mlp(
            mlp_input_emb,
            depth_emb,
            torch.where(
                uncond_mlp_flag, torch.zeros_like(h_dec) + self.mlp_null_emb, h_dec
            ),
        )
        logits_p, mus_p, mu_res_p, log_std_p = self.mlp_post(h_mlp)
        log_std_p = log_std_p.clamp_min(self.config.min_log_std)

        with torch.autocast(device.type, enabled=False):
            eos_logit = eos_logit.float()
            eos_target = torch.logical_and(
                torch.logical_not(F.pad(audio_mask[:, 1:], [0, 0, 0, 1])), audio_mask
            ).float()
            eos_loss = (
                F.binary_cross_entropy_with_logits(
                    eos_logit, eos_target, reduction="none"
                )
                * audio_mask
            ).sum() / audio_mask.sum()

            z = z.float()
            mu_base = mlp_input_emb.float().view(b, t, 1, self.config.d_in)
            logits_p = logits_p.float()
            mus_p = mus_p.float()
            mu_res_p = mu_res_p.float()
            log_std_p = log_std_p.float()

            z_transformed = (z - mu_base - mu_res_p) * torch.exp(-log_std_p)

            bias_logp_z = (-log_std_p - 0.5 * math.log(2.0 * math.pi)).transpose(
                -1, -2
            ) * self.config.d_in  # [b, t, 1, g]
            unscaled_logp_z = -0.5 * self.mlp_post.dist(
                mus_p, z_transformed
            )  # [b, t, n, g]

            q_kc = (
                torch.softmax(
                    (unscaled_logp_z).sum(-1),
                    -1,
                )
                * (1 - self.config.label_smoothing)
                + self.config.label_smoothing / self.config.num_predictions
            ).detach()  # [b, t, n]
            logq_kc = torch.log(q_kc).detach()

            target_mask = ~random_code_mask * audio_mask
            reduced_target_mask = target_mask.sum(-1) != 0

            z_loss = (
                (q_kc * -(unscaled_logp_z + bias_logp_z).sum(-1)).sum(-1)
                * reduced_target_mask
            ).sum() / target_mask.sum()

            k_loss = (
                (q_kc * (logq_kc - F.log_softmax(logits_p, -1))).sum(-1)
                * reduced_target_mask
            ).sum() / target_mask.sum()
        return eos_loss, z_loss, k_loss

    @torch.no_grad()
    def generate_step(
        self,
        num_iter: int,
        x: Tensor,
        x_cfg: Tensor | None = None,
        code: Tensor | None = None,
        code_mask: Tensor | None = None,
        classifier_free_guidance_list: list[float] | None = None,
        top_p_or_k_list: list[float] | None = None,
        noise_scale_list: list[float] | None = None,
        exponent: float = 3.0,
    ) -> Tensor:
        # preparation
        device = x.device

        b_org, t_org = x.size(0), x.size(1)
        b = b_org * t_org
        t = 1
        d = self.depth

        if code is not None:
            code = code.view(b, t, d)
        else:
            code = (
                torch.zeros(
                    (b, t, d),
                    dtype=torch.long,
                    device=device,
                )
                + self.blank_id
            )
        if code_mask is not None:
            code_mask = code_mask.view(b, t, d)
        else:
            code_mask = torch.zeros((b, t, d), dtype=torch.bool, device=device)
        padding_mask = torch.ones((b, t, 1), dtype=torch.bool, device=device)

        x = x.view(b, t, -1)
        if x_cfg is None:
            x_cfg = x[:0, :0, :0]
        else:
            x_cfg = x_cfg.view(b, t, -1)
        assert x_cfg is not None
        if classifier_free_guidance_list is not None and any(
            cfg > 0.0 for cfg in classifier_free_guidance_list
        ):
            assert x_cfg.size() == x.size()

        ratios = torch.linspace(0.0, 1.0, num_iter + 1, device=device)[:-1].unsqueeze(
            -1
        )
        masking_ratios = get_masking_ratio(ratios, exponent=exponent)
        num_maskings = torch.ceil(d * masking_ratios).long()
        ks = num_maskings - F.pad(num_maskings[1:], [0, 0, 0, 1])

        num_maskings_total = ((code == self.blank_id) * padding_mask).sum([1, 2])
        ratios = torch.linspace(0.0, 1.0, num_iter + 1, device=device)[:-1].unsqueeze(
            -1
        )
        ratios_min = get_ratio(
            (
                (
                    num_maskings_total.unsqueeze(0)
                    - torch.arange(num_iter, device=device).unsqueeze(-1)
                )
                / num_maskings_total
            ).clamp_min(0),
            exponent=exponent,
        )
        masking_ratios = get_masking_ratio(
            torch.maximum(ratios, ratios_min), exponent=exponent
        )

        num_maskings = torch.ceil(num_maskings_total * masking_ratios).long()
        ks = num_maskings - F.pad(num_maskings[1:], [0, 0, 0, 1])

        cnt = 0
        # iteration
        for step, k in enumerate(ks):
            assert code_mask is not None
            if torch.all(k == 0):
                continue

            if classifier_free_guidance_list is None:
                classifier_free_guidance = 0.0
            else:
                classifier_free_guidance = classifier_free_guidance_list[step]
            if top_p_or_k_list is None:
                top_p_or_k = 1.0
            else:
                top_p_or_k = top_p_or_k_list[step]
                if not (top_p_or_k > 0.0):
                    top_p_or_k = 1
            if noise_scale_list is None:
                noise_scale = 1.0
            else:
                noise_scale = noise_scale_list[step]

            mlp_input_emb = self.depthsum_embedding(
                code, self.mus, include_blank=True
            ).view(b, t, -1)
            depth_emb = self.depthwise_embedding(
                code_mask.long(), self.depth_embs
            ).view(b, t, -1)

            logit_p, mu_p, log_std_p = self.mlp_post.infer(
                x=(
                    self.forward_mlp(
                        mlp_input_emb.repeat(2, 1, 1),
                        depth_emb.repeat(2, 1, 1),
                        torch.cat([x, x_cfg], 0),
                    )
                    if classifier_free_guidance > 0
                    else self.forward_mlp(
                        mlp_input_emb,
                        depth_emb,
                        x,
                    )
                ),
                classifier_free_guidance=classifier_free_guidance,
                top_p_or_k=top_p_or_k,
                min_log_std=self.config.min_log_std,
            )
            z = mu_p + torch.randn_like(mu_p) * torch.exp(log_std_p) * noise_scale

            new_code_mask = get_random_mask(
                code_mask,
                padding_mask,
                k,
                self.config.masking_mode,
                unmasking=True,
            )

            code = depthsum_encoding_step(self.mus, z, code, cnt, k[0].item())
            cnt += k[0].item()
            code_mask = new_code_mask
            assert code is not None
        return code.view(b_org, t_org, d)


def load_models(
    cfg: Config, device: torch.device, load_lm: bool = True
) -> tuple[PreTrainedTokenizer, PreTrainedModel | None, nn.Embedding, RVQVAE, Model]:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        cfg.data.pretrained_tokenizer_name
    )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = default_chat_template

    if load_lm:
        try:
            lm: PreTrainedModel | None = (
                AutoModelForCausalLM.from_pretrained(
                    cfg.data.pretrained_lm_name, torch_dtype=torch.bfloat16
                )
                .to(device=device)
                .eval()
            )
        except (ValueError, OSError):
            lm = None
    else:
        lm = None
    if lm is not None:
        lm_embedding_weight = lm.get_input_embeddings().weight.detach()
    else:
        lm_embedding_weight = torch.load(
            cfg.data.pretrained_lm_embedding_path, map_location="cpu", weights_only=True
        )["weight"]
    vocab_size, d_in = lm_embedding_weight.size()
    embed_tokens = nn.Embedding(vocab_size, d_in).to(dtype=torch.bfloat16)
    embed_tokens.weight.data.copy_(lm_embedding_weight)
    embed_tokens.weight.requires_grad = False
    embed_tokens.to(device).eval()

    ae = RVQVAE.from_pretrained(cfg.data.pretrained_ae_dir).to(device).eval()
    mus = torch.stack([x.detach() for x in ae.prvq.mus_list], 0)

    net = (
        Model(cfg.model, mus, os.path.join(cfg.workdir_path, "char_aware_subword.json"))
        .to(device)
        .eval()
    )

    net_state_dict = net.state_dict()
    ckpt_path = latest_checkpoint_path(cfg.workdir_path, "ema_*.pt")
    net_state_dict.update(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    _ = net.load_state_dict(net_state_dict)

    return tokenizer, lm, embed_tokens, ae, net


def get_inputs_from_texts_and_wav_path(
    text_prompt: str,
    wav_path: str,
    resampler_dict: dict[int, torchaudio.transforms.Resample],
    tokenizer: AutoTokenizer,
    no_desc: bool = False,
) -> dict[str, Any]:
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav).mean(1)

    if sr not in resampler_dict:
        resampler_dict[sr] = torchaudio.transforms.Resample(
            sr,
            SAMPLING_RATE,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    wav = resampler_dict[sr](wav)
    wav_length = wav.size(0)
    audio_token_length = math.ceil(wav_length // WAV_TO_TOKEN_RATIO)
    audio_prompt_length = audio_token_length * WAV_TO_TOKEN_RATIO
    pad_length = audio_prompt_length - wav_length
    wav = F.pad(wav, [pad_length, 0])

    messages = []
    messages.append(
        {
            "role": "system",
            "content": (
                "You engage in conversation with the user. "
                "When delivering your response as speech, if the user provides a description such as "
                "emotions, scene details, or speaker style, "
                "you adjust your speaking style accordingly when delivering the response. "
                "However, this description should influence only the delivery of your response, "
                "not its content. "
                "Your response should remain independent of any stylistic instructions."
            )
            if not no_desc
            else "",
        }
    )
    if not no_desc:
        messages.append(
            {
                "role": "user",
                "content": "I'd like to hear something from you.",
            }
        )
    messages.append(
        {
            "role": "assistant",
            "content": SCRIPT_PLACEHOLDER,
        }
    )
    if not no_desc:
        messages.append(
            {
                "role": "user",
                "content": "I'd like to hear something from you.",
            }
        )
    messages.append({"role": "assistant", "content": SCRIPT_PLACEHOLDER})
    non_script_list = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ).split(SCRIPT_PLACEHOLDER + tokenizer.eos_token)[:-1]
    script_list = []
    script_list.append(text_prompt)

    input_ids = []
    aligned_script_mask = []
    aligned_text_mask = []
    aligned_audio_mask = []
    init_inputs_length = 0
    eos_token_id = tokenizer.vocab[tokenizer.eos_token]

    # prompt
    desc_ids = tokenizer.encode(non_script_list[0], add_special_tokens=False)
    script_ids = tokenizer.encode(script_list[0], add_special_tokens=False) + [
        eos_token_id
    ]
    script_ids = script_ids[-audio_token_length:]

    desc_token_length = len(desc_ids)
    script_token_length = len(script_ids)
    text_token_length = len(desc_ids) + len(script_ids)
    aligned_token_length = desc_token_length + audio_token_length
    init_inputs_length += aligned_token_length

    input_ids += desc_ids + script_ids

    aligned_text_mask += [1] * text_token_length + [0] * (
        aligned_token_length - text_token_length
    )
    aligned_script_mask += (
        [0] * desc_token_length
        + [1] * script_token_length
        + [0] * (aligned_token_length - text_token_length)
    )
    aligned_audio_mask += [0] * desc_token_length + [1] * (
        aligned_token_length - desc_token_length
    )

    # target
    desc_ids = tokenizer.encode(non_script_list[-1], add_special_tokens=False)

    desc_token_length = len(desc_ids)
    text_token_length = len(desc_ids)
    init_inputs_length += desc_token_length

    input_ids += desc_ids

    aligned_text_mask += [1] * text_token_length
    aligned_script_mask += [0] * desc_token_length
    aligned_audio_mask += [0] * text_token_length

    return {
        "input_ids": input_ids,
        "aligned_text_mask": aligned_text_mask,
        "aligned_script_mask": aligned_script_mask,
        "aligned_audio_mask": aligned_audio_mask,
        "wav": wav,
        "init_inputs_length": init_inputs_length,
    }


def get_batched_inputs(
    device,
    text_ref_list,
    wav_ref_list,
    resampler_dict,
    tokenizer,
    lm,
    embed_tokens,
    ae,
):
    input_ids_list = []
    aligned_text_mask_list = []
    aligned_script_mask_list = []
    aligned_audio_mask_list = []
    wav_list = []
    init_inputs_length = 0
    for text_ref, wav_path in zip(text_ref_list, wav_ref_list, strict=True):
        inputs = get_inputs_from_texts_and_wav_path(
            text_ref, wav_path, resampler_dict, tokenizer
        )
        input_ids_list.append(inputs["input_ids"])
        aligned_text_mask_list.append(inputs["aligned_text_mask"])
        aligned_script_mask_list.append(inputs["aligned_script_mask"])
        aligned_audio_mask_list.append(inputs["aligned_audio_mask"])
        wav_list.append(inputs["wav"])
        if init_inputs_length == 0:
            init_inputs_length = inputs["init_inputs_length"]
        else:
            assert init_inputs_length == inputs["init_inputs_length"]
    # batching
    max_token_length = max([len(x) for x in input_ids_list])
    max_aligned_token_length = max([len(x) for x in aligned_text_mask_list])
    # Append trailing zeros for inputs.
    input_ids = torch.tensor(
        [x + [0] * (max_token_length - len(x)) for x in input_ids_list],
        dtype=torch.long,
        device=device,
    )
    aligned_text_mask = torch.tensor(
        [x + [0] * (max_aligned_token_length - len(x)) for x in aligned_text_mask_list],
        dtype=torch.bool,
        device=device,
    )
    aligned_script_mask = torch.tensor(
        [
            x + [0] * (max_aligned_token_length - len(x))
            for x in aligned_script_mask_list
        ],
        dtype=torch.bool,
        device=device,
    )
    aligned_audio_mask = torch.tensor(
        [
            x + [0] * (max_aligned_token_length - len(x))
            for x in aligned_audio_mask_list
        ],
        dtype=torch.bool,
        device=device,
    )
    wav_length = torch.tensor(
        [wav.size(0) for wav in wav_list], dtype=torch.long, device=device
    )
    wav = (
        torch.stack(
            [F.pad(wav, [0, wav_length.max() - wav.size(0)]) for wav in wav_list], 0
        )
        .to(device)
        .unsqueeze(1)
    )

    b, t = aligned_text_mask.size()
    blank_id = ae.config.num_mixtures

    # process batches
    code, code_length = ae.encode(wav, wav_length)
    aligned_code = (
        torch.zeros((b, t, code.size(2)), dtype=code.dtype, device=device) + blank_id
    )
    for i in range(b):
        aligned_code[i].masked_scatter_(aligned_audio_mask[i].unsqueeze(-1), code[i])

    inputs_embeds = embed_tokens(input_ids)
    if lm is not None:
        lm_hidden_state = lm.model(
            inputs_embeds=inputs_embeds,
        ).last_hidden_state
    else:
        lm_hidden_state = inputs_embeds
    aligned_lm_hidden_state = torch.zeros(
        (b, t, lm_hidden_state.size(2)), dtype=lm_hidden_state.dtype, device=device
    )
    aligned_input_ids = torch.zeros((b, t), dtype=input_ids.dtype, device=device)
    for i in range(b):
        aligned_lm_hidden_state[i].masked_scatter_(
            aligned_text_mask[i].unsqueeze(-1), lm_hidden_state[i]
        )
        aligned_input_ids[i].masked_scatter_(aligned_text_mask[i], input_ids[i])
    return (
        aligned_code,
        aligned_audio_mask.unsqueeze(-1),
        aligned_lm_hidden_state,
        aligned_input_ids,
        aligned_script_mask.unsqueeze(-1),
        init_inputs_length,
    )


def main_inference(
    device,
    tokenizer,
    lm,
    embed_tokens,
    ae,
    net,
    batch_list,
    num_iter,
    classifier_free_guidance_list,
    top_p_or_k_list,
    noise_scale_list,
    end_threshold,
    out_dir,
    max_num_steps=200,
    no_desc: bool = False,
    exponent: float = 3.0,
):
    resampler_dict: dict[int, torchaudio.transforms.Resample] = {}  # type: ignore[annotation-unchecked]
    no_cfg = not (any(x > 0.0 for x in classifier_free_guidance_list))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for file_name_list, text_ref_list, wav_ref_list in batch_list:
            (
                code,
                audio_mask,
                lm_hidden_state,
                subword_ids,
                subword_mask,
                init_inputs_length,
            ) = get_batched_inputs(
                text_ref_list,
                wav_ref_list,
                resampler_dict,
                tokenizer,
                lm,
                embed_tokens,
                ae,
            )

            #########
            ### TTS
            #########
            past_key_values = DynamicCache()
            b = code.size(0)

            assert init_inputs_length > 0
            sinusoidal_pos = net.pos_embedding(
                torch.arange(init_inputs_length, device=device)
            ).unsqueeze(0)

            cond = torch.zeros(
                (b, init_inputs_length, net.config.d_model), device=device
            )
            if net.lm_proj is not None:
                lm_emb = lm_hidden_state
                cond = cond + net.lm_proj(lm_emb[:, :init_inputs_length])
            if (
                net.config.char_aware_subword_config is not None
                and subword_ids is not None
            ):
                assert net.cas_encoder is not None
                assert net.cas_proj is not None
                if torch.any(subword_mask):
                    cas_emb = net.cas_encoder(subword_ids, subword_mask)
                else:
                    cas_emb = torch.zeros(
                        (
                            b,
                            subword_ids.size(1),
                            net.config.char_aware_subword_config.d_model,
                        ),
                        device=device,
                    )
                cond = cond + net.cas_proj(cas_emb[:, :init_inputs_length])

            z = net.depthsum_embedding(
                code[:, :init_inputs_length], net.mus, include_blank=True
            )
            decoder_input_emb = F.pad(z.view(b, cond.size(1), -1)[:, :-1], [0, 0, 1, 0])
            x = net.forward_decoder(
                decoder_input_emb if no_cfg else decoder_input_emb.repeat(2, 1, 1),
                cond
                if no_cfg
                else torch.cat(
                    [cond, torch.zeros_like(cond) + net.decoder_null_emb], 0
                ),
                sinusoidal_pos,
                past_key_values=past_key_values,
            )

            audio_gen_only_steps = 0
            cnt = init_inputs_length + 0
            code = code[:, cnt - 1 : cnt]
            pos_init = torch.zeros((1,), dtype=torch.long, device=device)
            cond_init = torch.zeros((b, 1, net.config.d_model), device=device)
            cond_last = cond_init
            if net.lm_proj is not None:
                cond_last = cond_last + net.lm_proj(torch.zeros_like(lm_emb[:, -1:]))
            if net.config.char_aware_subword_config is not None:
                assert net.cas_proj is not None
                cond_last = cond_last + net.cas_proj(torch.zeros_like(cas_emb[:, -1:]))
            end_flag = torch.zeros((b,), dtype=torch.bool, device=device)

            # Combined generation loop
            codes = []
            end_flags = []
            while True:
                if cnt >= lm_hidden_state.size(1):
                    audio_gen_only_steps += 1
                if audio_gen_only_steps == 0:
                    cond = cond_init
                    if net.lm_proj is not None:
                        cond = cond + net.lm_proj(lm_emb[:, cnt : cnt + 1])
                    if net.config.char_aware_subword_config is not None:
                        assert net.cas_proj is not None
                        cond = cond + net.cas_proj(cas_emb[:, cnt : cnt + 1])
                else:
                    cond = cond_last

                sinusoidal_pos = net.pos_embedding(pos_init + cnt).unsqueeze(0)
                z = net.depthsum_embedding(code, net.mus, include_blank=True)
                decoder_input_emb = z.view(b, 1, -1)
                x = net.forward_decoder(
                    decoder_input_emb if no_cfg else decoder_input_emb.repeat(2, 1, 1),
                    cond
                    if no_cfg
                    else torch.cat(
                        [cond, torch.zeros_like(cond) + net.decoder_null_emb], 0
                    ),
                    sinusoidal_pos,
                    past_key_values=past_key_values,
                )
                code = net.generate_step(
                    num_iter=num_iter,
                    x=x[:b],
                    x_cfg=None if no_cfg else x[b:],
                    classifier_free_guidance_list=None
                    if no_cfg
                    else classifier_free_guidance_list,
                    top_p_or_k_list=top_p_or_k_list,
                    noise_scale_list=noise_scale_list,
                    exponent=exponent,
                )
                end_flag = torch.logical_or(
                    net.eos_head(x[:b]).squeeze() > end_threshold, end_flag
                )
                cnt += 1
                codes.append(code)
                end_flags.append(end_flag)
                if torch.all(end_flag) or (cnt - init_inputs_length) >= max_num_steps:
                    break
            code_sample = torch.cat(codes, 1)
            end_flags_tensor = torch.stack(end_flags, 1)
            code_sample_length = torch.where(
                torch.any(end_flags_tensor, 1),
                torch.argmax(end_flags_tensor.long(), 1) + 1,  # Length is index + 1
                max_num_steps,
            )

            for i in range(b):
                code_sample_len = int(code_sample_length[i].item())
                wav_sample = ae.decode(
                    code_sample[i : i + 1, :code_sample_len], constrain_value_range=True
                )[0][0].cpu()
                torchaudio.save(
                    os.path.join(out_dir, file_name_list[i]),
                    wav_sample,
                    sample_rate=SAMPLING_RATE,
                    format="wav",
                )


# class EasyARTTS(nn.Module):
#     def __init__(self, cfg: Config, device: torch.device | None = None):
#         super().__init__()
#         self.net_cache = None
#         self.ae_cache = None
#
#         if device is None:
#             device = torch.device("cpu")
#         self.tokenizer, self.lm, self.embed_tokens, self.ae, self.net = load_models(
#             cfg, device, load_lm=False
#         )
#
#         for n, p in self.named_parameters():
#             if not n.startswith("net."):
#                 p.requires_grad = False
#
#     @classmethod
#     def from_pretrained(
#         cls, pretrained_dir: str, device: torch.device | None = None
#     ) -> "EasyARTTS":
#         cfg = get_config_from_dir(pretrained_dir)
#         return cls(cfg, device)
#
#     def reset_input_and_kv_cache(self, use_cache: bool = True):
#         if use_cache:
#             self.net_cache = DynamicCache()
#             # self.ae_cache = CausalConv1dCache()
#         else:
#             self.net_cache = None
#             # self.ae_cache = None
#
#     def decode(
#         self,
#         tokens: Tensor,
#         tokens_len: Tensor | None = None,
#         constrain_value_range: bool = True,
#         flush: bool = False,
#     ) -> tuple[Tensor, Tensor | None]:
#         return self.ae.decode(
#             tokens,
#             tokens_len,
#             constrain_value_range=constrain_value_range,
#             cache=self.ae_cache,
#             flush=flush,
#         )
