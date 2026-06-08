# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pallas-Mosaic-GPU RMSNorm with optional activation quantization."""

import dataclasses
import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
import qwix
from tokamax._src import gpu_utils


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
  """Configuration for the RMSNorm Mosaic GPU kernel."""

  block_m: int = 1

  def __post_init__(self):
    if self.block_m <= 0:
      raise ValueError(f"{self.block_m=} must be positive.")


def _check_supported_qtype(qtype: jax.typing.DTypeLike) -> jnp.dtype:
  qtype = jnp.dtype(qtype)
  if qtype not in (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.int8)):
    raise NotImplementedError(f"Unsupported RMSNorm quantization {qtype=}.")
  return qtype


def _quantize(
    x: jax.Array,
    qtype: jnp.dtype,
    *,
    subchannel_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes one normalized row using Qwix-compatible absmax scales."""
  x_tiled = x.reshape((x.shape[0] // subchannel_size, subchannel_size))
  absmax = jnp.abs(x_tiled).max(axis=1)
  if qtype == jnp.dtype(jnp.int8):
    qmax = jnp.array(127.5, dtype=x.dtype)
  else:
    qmax = jnp.array(jnp.finfo(qtype).max, dtype=x.dtype)
  scale = absmax / qmax
  scale = jnp.where(absmax == 0.0, jnp.array(1.0, dtype=x.dtype), scale)
  inv_scale = lax.broadcast_in_dim(1.0 / scale, x_tiled.shape, [0])
  qvalue = x_tiled * inv_scale
  if qtype == jnp.dtype(jnp.int8):
    qvalue = jnp.round(jnp.clip(qvalue, -127.5, 126.75)).astype(qtype)
  else:
    qinfo = jnp.finfo(qtype)
    qvalue = jnp.clip(qvalue, qinfo.min, qinfo.max).astype(qtype)
  return qvalue.reshape(x.shape), scale


def _rms_norm_row(
    x: jax.Array,
    scale: jax.Array | None,
    *,
    epsilon: float,
    out_dtype: jnp.dtype,
) -> jax.Array:
  x = x.astype(jnp.float32)
  rms = lax.rsqrt(jnp.mean(jnp.square(x), axis=0) + epsilon)
  x *= rms
  if scale is not None:
    x *= scale.astype(jnp.float32)
  # The unfused path normalizes to the input dtype before quantizing.
  return x.astype(out_dtype).astype(jnp.float32)


def get_heuristics_config(x: jax.Array) -> Config:
  """Returns a shape-aware default config.

  Decode usually has very small leading dimensions, where one row per CTA keeps
  latency predictable. Prefill has many rows, so small channel counts can group
  a few rows per CTA without losing occupancy. Large d_model rows stay at
  block_m=1 to avoid excessive register pressure.
  """
  m = math.prod(x.shape[:-1])
  c = x.shape[-1]
  if m <= 16:
    return Config(block_m=1)
  return Config(block_m=max(1, min(8, 4096 // c)))


def rms_norm(
    x: jax.Array,
    scale: jax.Array | None = None,
    *,
    epsilon: float = 1e-6,
    quantize: bool = False,
    qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    subchannel_size: int = 512,
    quant_scale_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    config: Config | None = None,
) -> jax.Array | qwix.QArray:
  """Runs RMSNorm with optional activation quantization on Mosaic GPU.

  This implements a Mosaic GPU RMSNorm kernel. If ``quantize=True``, the kernel
  also performs the common MoE activation quantization epilogue:
  BF16 RMSNorm(x, scale) -> BF16 -> absmax tiled quantization.

  Args:
    x: Input activations with shape ``(*B, C)``.
    scale: Optional RMSNorm scale of shape ``(C,)``.
    epsilon: RMSNorm epsilon.
    quantize: If ``True``, return a Qwix QArray. Otherwise return dense RMSNorm
      output with the same dtype and shape as ``x``.
    qtype: Quantized activation dtype. ``float8_e4m3fn`` is intended for the
      SM100 FP8 ragged-dot lhs path. ``int8`` is also supported for parity with
      the existing fused epilogue quantizer.
    subchannel_size: Last-axis tile size for quantization scales.
    quant_scale_dtype: Dtype for returned quantization scales.
    config: Optional kernel config. If omitted, a shape-aware default is chosen.

  Returns:
    Dense RMSNorm output if ``quantize=False``. Otherwise, a Qwix QArray with
    ``qvalue.shape == x.shape`` and
    ``scale.shape == (*B, C // subchannel_size)``.
  """
  if not gpu_utils.has_mosaic_gpu_support():
    raise NotImplementedError("Mosaic GPU not supported on this platform.")
  if not gpu_utils.is_sm100():
    raise NotImplementedError(
        "Mosaic GPU RMSNorm is currently only enabled for SM100 GPUs."
    )

  orig_shape = x.shape
  if x.ndim < 1:
    raise ValueError("Expected RMSNorm input to have at least one dimension.")
  c = orig_shape[-1]
  out_dtype = jnp.dtype(x.dtype)
  if scale is not None and scale.shape != (c,):
    raise ValueError(f"Expected scale shape {(c,)}, got {scale.shape}.")
  if config is None:
    config = get_heuristics_config(x)

  x_2d = x.reshape((-1, c))
  m = x_2d.shape[0]
  block_m = config.block_m
  grid = ((m + block_m - 1) // block_m,)

  def normalize_row(x_row, scale_row):
    return _rms_norm_row(
        x_row,
        scale_row,
        epsilon=epsilon,
        out_dtype=out_dtype,
    )

  if not quantize:

    if scale is None:

      @functools.partial(
          plgpu.kernel,
          out_shape=jax.ShapeDtypeStruct((m, c), out_dtype),
          grid=grid,
          grid_names=("row_blocks",),
          kernel_name="rms_norm_sm100",
          compiler_params=plgpu.CompilerParams(approx_math=True),
      )
      def kernel(x_gmem, out_gmem):
        row_start = lax.axis_index("row_blocks") * block_m
        for i in range(block_m):
          row = row_start + i

          @pl.when(row < m)
          def _():
            out_gmem[row, :] = normalize_row(x_gmem[row, :], None).astype(
                out_dtype
            )

      out = kernel(x_2d)
    else:

      @functools.partial(
          plgpu.kernel,
          out_shape=jax.ShapeDtypeStruct((m, c), out_dtype),
          grid=grid,
          grid_names=("row_blocks",),
          kernel_name="rms_norm_sm100",
          compiler_params=plgpu.CompilerParams(approx_math=True),
      )
      def kernel(x_gmem, scale_gmem, out_gmem):
        row_start = lax.axis_index("row_blocks") * block_m
        scale_values = scale_gmem[:]
        for i in range(block_m):
          row = row_start + i

          @pl.when(row < m)
          def _():
            out_gmem[row, :] = normalize_row(
                x_gmem[row, :], scale_values
            ).astype(out_dtype)

      out = kernel(x_2d, scale)

    return out.reshape(orig_shape)

  qtype = _check_supported_qtype(qtype)
  quant_scale_dtype = jnp.dtype(quant_scale_dtype)
  if subchannel_size <= 0:
    raise ValueError(f"{subchannel_size=} must be positive.")
  if c % subchannel_size != 0:
    raise NotImplementedError(
        f"Expected last dimension {c} to be divisible by {subchannel_size=}."
    )
  num_scale_tiles = c // subchannel_size

  def quantize_row(x_row, scale_row):
    x_row = _rms_norm_row(
        x_row,
        scale_row,
        epsilon=epsilon,
        out_dtype=out_dtype,
    )
    qvalue, quant_scale = _quantize(
        x_row,
        qtype,
        subchannel_size=subchannel_size,
    )
    return qvalue, quant_scale.astype(quant_scale_dtype)

  if scale is None:

    @functools.partial(
        plgpu.kernel,
        out_shape=(
            jax.ShapeDtypeStruct((m, c), qtype),
            jax.ShapeDtypeStruct((m, num_scale_tiles), quant_scale_dtype),
        ),
        grid=grid,
        grid_names=("row_blocks",),
        kernel_name="rms_norm_quant_sm100",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )
    def kernel(x_gmem, qvalue_gmem, quant_scale_gmem):
      row_start = lax.axis_index("row_blocks") * block_m
      for i in range(block_m):
        row = row_start + i

        @pl.when(row < m)
        def _():
          qvalue, quant_scale = quantize_row(x_gmem[row, :], None)
          qvalue_gmem[row, :] = qvalue
          quant_scale_gmem[row, :] = quant_scale

    qvalue, quant_scale = kernel(x_2d)
  else:

    @functools.partial(
        plgpu.kernel,
        out_shape=(
            jax.ShapeDtypeStruct((m, c), qtype),
            jax.ShapeDtypeStruct((m, num_scale_tiles), quant_scale_dtype),
        ),
        grid=grid,
        grid_names=("row_blocks",),
        kernel_name="rms_norm_quant_sm100",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )
    def kernel(x_gmem, scale_gmem, qvalue_gmem, quant_scale_gmem):
      row_start = lax.axis_index("row_blocks") * block_m
      scale_values = scale_gmem[:]
      for i in range(block_m):
        row = row_start + i

        @pl.when(row < m)
        def _():
          qvalue, quant_scale = quantize_row(x_gmem[row, :], scale_values)
          qvalue_gmem[row, :] = qvalue
          quant_scale_gmem[row, :] = quant_scale

    qvalue, quant_scale = kernel(x_2d, scale)

  return qwix.QArray(
      qvalue.reshape(orig_shape),
      quant_scale.reshape((*orig_shape[:-1], num_scale_tiles)),
      qtype=qtype,
  )
