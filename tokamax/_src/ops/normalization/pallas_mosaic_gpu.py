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
"""Pallas-Mosaic-GPU RMSNorm and fused RMSNorm + packed activation quant.

This module exposes two separate, intentionally differently-named entry points
so callers are never surprised by a polymorphic return type:

* ``rms_norm`` -- plain RMSNorm, returns a dense ``jax.Array``.
* ``rms_norm_fuse_quant_packed`` -- RMSNorm fused with absmax tiled activation
  quantization, returns a single **packed** ``uint8`` buffer whose every token
  row is ``[ qvalue bytes | scale bytes ]``.

The packed buffer exists because ``jax.lax.ragged_all_to_all`` (the MoE
dispatch collective) permutes *rows* of a plain dense array and cannot carry a
``qwix.QArray``. Packing quant value+scale into one contiguous row lets the
collective move both together, and -- because the row is fp8 (~1 byte/elem)
rather than bf16 (2 bytes/elem) -- halves the all-to-all byte volume. Use
``unpack_rms_norm_quant`` on the receiving side to rebuild the ``qwix.QArray``
that the fp8 ragged-dot consumes.
"""

import dataclasses
import functools
import math

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp
from tokamax._src import gpu_utils
from tokamax._src.ops.normalization import packed_quant

# Re-exported here so callers can reach the packed wire format alongside the
# kernels (e.g. ``pallas_mosaic_gpu.unpack_rms_norm_quant``). The definitions
# live in ``packed_quant`` so they import without a GPU.
packed_width = packed_quant.packed_width
unpack_rms_norm_quant = packed_quant.unpack_rms_norm_quant


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
  """Configuration for the RMSNorm Mosaic GPU kernel."""

  block_m: int = 1

  def __post_init__(self):
    if self.block_m <= 0:
      raise ValueError(f"{self.block_m=} must be positive.")


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


# -----------------------------------------------------------------------------
# Kernels.
# -----------------------------------------------------------------------------


def _check_sm100_support() -> None:
  if not gpu_utils.has_mosaic_gpu_support():
    raise NotImplementedError("Mosaic GPU not supported on this platform.")
  if not gpu_utils.is_sm100():
    raise NotImplementedError(
        "Mosaic GPU RMSNorm is currently only enabled for SM100 GPUs."
    )


def rms_norm(
    x: jax.Array,
    scale: jax.Array | None = None,
    *,
    epsilon: float = 1e-6,
    config: Config | None = None,
) -> jax.Array:
  """Runs plain RMSNorm on Mosaic GPU.

  Args:
    x: Input activations with shape ``(*B, C)``.
    scale: Optional RMSNorm scale of shape ``(C,)``.
    epsilon: RMSNorm epsilon.
    config: Optional kernel config. If omitted, a shape-aware default is chosen.

  Returns:
    Dense RMSNorm output with the same dtype and shape as ``x``.
  """
  _check_sm100_support()

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

  if scale is None:

    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((m, c), out_dtype),
        grid=grid,
        grid_names=("row_blocks",),
        kernel_name="rms_norm_sm100",
        compiler_params=plgpu.CompilerParams(approx_math=False),
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
        compiler_params=plgpu.CompilerParams(approx_math=False),
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


def rms_norm_fuse_quant_packed(
    x: jax.Array,
    scale: jax.Array | None = None,
    *,
    epsilon: float = 1e-6,
    qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    subchannel_size: int = 512,
    quant_scale_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    config: Config | None = None,
) -> jax.Array:
  """Runs RMSNorm fused with packed absmax activation quantization.

  Performs the common MoE activation-quant epilogue --
  ``BF16 RMSNorm(x, scale) -> absmax tiled quantization`` -- and packs the
  result so it can be transported through ``ragged_all_to_all``: the returned
  array is a single ``uint8`` buffer whose every token row is
  ``[ qvalue bytes | scale bytes ]``. Use ``unpack_rms_norm_quant`` on the
  receiving side to rebuild the ``qwix.QArray`` for the fp8 ragged-dot.

  Args:
    x: Input activations with shape ``(*B, C)``.
    scale: Optional RMSNorm scale of shape ``(C,)``.
    epsilon: RMSNorm epsilon.
    qtype: Quantized activation dtype. ``float8_e4m3fn`` targets the SM100 FP8
      ragged-dot lhs path; ``int8`` is also supported.
    subchannel_size: Last-axis tile size for quantization scales.
    quant_scale_dtype: Dtype of the packed quantization scales.
    config: Optional kernel config. If omitted, a shape-aware default is chosen.

  Returns:
    A ``uint8`` array of shape ``(*B, packed_width(C, ...))`` -- see
    ``packed_width`` for the per-row byte layout.
  """
  _check_sm100_support()

  orig_shape = x.shape
  if x.ndim < 1:
    raise ValueError("Expected RMSNorm input to have at least one dimension.")
  c = orig_shape[-1]
  out_dtype = jnp.dtype(x.dtype)
  if scale is not None and scale.shape != (c,):
    raise ValueError(f"Expected scale shape {(c,)}, got {scale.shape}.")
  if config is None:
    config = get_heuristics_config(x)

  qtype = packed_quant.check_supported_qtype(qtype)
  quant_scale_dtype = jnp.dtype(quant_scale_dtype)
  if subchannel_size <= 0:
    raise ValueError(f"{subchannel_size=} must be positive.")
  if c % subchannel_size != 0:
    raise NotImplementedError(
        f"Expected last dimension {c} to be divisible by {subchannel_size=}."
    )
  num_scale_tiles = c // subchannel_size
  width = packed_quant.packed_width(c, qtype, subchannel_size, quant_scale_dtype)
  qmax = 127.5 if qtype == jnp.dtype(jnp.int8) else float(jnp.finfo(qtype).max)

  x_2d = x.reshape((-1, c))
  m = x_2d.shape[0]
  block_m = config.block_m
  grid = ((m + block_m - 1) // block_m,)

  out_shape = (
      jax.ShapeDtypeStruct((m, c), qtype),
      jax.ShapeDtypeStruct((m, num_scale_tiles), quant_scale_dtype),
  )

  def quant_write_row(x_gmem, weight_gmem, qvalue_gmem, scale_gmem, row):
    """Fused RMSNorm + per-subchannel quant for one row, tile-by-tile.

    Mosaic GPU cannot slice a register vector or reduce over an inner axis of a
    reshaped one, so each subchannel is re-loaded fresh from GMEM (slice loads
    and full-vector reductions both lower). RMS is a single scalar from the
    full-row read.
    """
    xr = x_gmem[row, :].astype(jnp.float32)
    rms = lax.rsqrt(jnp.mean(jnp.square(xr), axis=0) + epsilon)
    for t in range(num_scale_tiles):
      lo = t * subchannel_size
      xt = x_gmem[row, lo : lo + subchannel_size].astype(jnp.float32) * rms
      if weight_gmem is not None:
        xt = xt * weight_gmem[lo : lo + subchannel_size].astype(jnp.float32)
      # Round-trip through the input dtype to match the unfused reference.
      xt = xt.astype(out_dtype).astype(jnp.float32)
      absmax = jnp.max(jnp.abs(xt))
      s = absmax / qmax
      s = jnp.where(absmax == 0.0, jnp.float32(1.0), s)
      # Round the scale to the stored dtype and quantize by dividing by exactly
      # that value, so the kernel divides by the same scale dequant multiplies
      # back (otherwise fp8 codes flip at rounding boundaries vs. qwix).
      s = s.astype(quant_scale_dtype)
      q = xt / s.astype(jnp.float32)
      if qtype == jnp.dtype(jnp.int8):
        q = jnp.round(jnp.clip(q, -127.5, 126.75)).astype(qtype)
      else:
        qinfo = jnp.finfo(qtype)
        q = jnp.clip(q, float(qinfo.min), float(qinfo.max)).astype(qtype)
      qvalue_gmem[row, lo : lo + subchannel_size] = q
      scale_gmem[row, t] = s

  if scale is None:

    @functools.partial(
        plgpu.kernel,
        out_shape=out_shape,
        grid=grid,
        grid_names=("row_blocks",),
        kernel_name="rms_norm_fuse_quant_packed_sm100",
        compiler_params=plgpu.CompilerParams(approx_math=False),
    )
    def kernel(x_gmem, qvalue_gmem, scale_gmem):
      row_start = lax.axis_index("row_blocks") * block_m
      for i in range(block_m):
        row = row_start + i

        @pl.when(row < m)
        def _():
          quant_write_row(x_gmem, None, qvalue_gmem, scale_gmem, row)

    qvalue, quant_scale = kernel(x_2d)
  else:

    @functools.partial(
        plgpu.kernel,
        out_shape=out_shape,
        grid=grid,
        grid_names=("row_blocks",),
        kernel_name="rms_norm_fuse_quant_packed_sm100",
        compiler_params=plgpu.CompilerParams(approx_math=False),
    )
    def kernel(x_gmem, weight_gmem, qvalue_gmem, scale_gmem):
      row_start = lax.axis_index("row_blocks") * block_m
      for i in range(block_m):
        row = row_start + i

        @pl.when(row < m)
        def _():
          quant_write_row(x_gmem, weight_gmem, qvalue_gmem, scale_gmem, row)

    qvalue, quant_scale = kernel(x_2d, scale)

  out = packed_quant.pack_qvalue_scale(qvalue, quant_scale)
  return out.reshape((*orig_shape[:-1], width))
