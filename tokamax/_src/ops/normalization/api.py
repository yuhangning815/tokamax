# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
"""Layer Normalization API."""

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias

from absl import logging
import immutabledict
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import gpu_utils
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import packed_quant

# Pure-JAX inverse of ``rms_norm_fuse_quant_packed``; rebuilds the QArray on the
# all-to-all receiving side. Re-exported here (and from ``tokamax``) so callers
# do not need to reach into the Mosaic GPU module, which may be unavailable.
unpack_rms_norm_quant = packed_quant.unpack_rms_norm_quant


Implementation: TypeAlias = Literal['xla', 'triton']
RmsNormImplementation: TypeAlias = Literal['xla', 'mosaic_gpu']

_IMPLEMENTATIONS = dict(xla=base.Normalization())
_DEFAULT_IMPLEMENTATIONS = ('xla',)

try:
  from tokamax._src.ops.normalization import pallas_triton  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  _IMPLEMENTATIONS['triton'] = pallas_triton.PallasTritonNormalization()
  _DEFAULT_IMPLEMENTATIONS = ('triton',) + _DEFAULT_IMPLEMENTATIONS
except ImportError:
  pass

try:
  # pylint: disable=g-import-not-at-top
  from tokamax._src.ops.normalization import pallas_mosaic_gpu
  # pylint: enable=g-import-not-at-top
except ImportError:
  pallas_mosaic_gpu = None


IMPLEMENTATIONS: Final[immutabledict.immutabledict[str, Callable[..., Any]]] = (
    immutabledict.immutabledict(_IMPLEMENTATIONS)
)
del _IMPLEMENTATIONS


def layer_norm(
    x: jax.Array,
    scale: jax.Array | None,
    offset: jax.Array | None,
    *,
    axis: int = -1,
    epsilon: float = 1e-06,
    scale_offset: float = 0.0,
    subtract_mean: bool = True,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> jax.Array:
  """Normalization layer.

  Implements LayerNorm (https://arxiv.org/abs/1607.06450), and RMSNorm
  (https://arxiv.org/abs/1910.07467) if `subtract_mean=False`.

  FP16/BF16 inputs will first be upcast to FP32, and all computations will be
  done in FP32. The result will downcasted to the input data type.

  Arguments:
    x: The array to be normalized.
    scale: An optional one-dimensional array of length `x.shape[axis]`.
    offset: An optional one-dimensional array of length `x.shape[axis]`.
    axis: The axis along which to normalize. Default is `-1`.
    epsilon: Epsilon value added to the denominator to avoid division by zero.
      Default is 1e-6.
    scale_offset: An offset added to the scale factors before scaling. Default
      is 0.0.
    subtract_mean: If `True`, use standard variance calculation. If `False`,
      assume mean is zero (i.e. RMS norm). Default is `True`.
    implementation: The implementation to use. If `None` (default), an
      implementation is automatically chosen that will work on all platforms.
      'xla' will use an XLA only implementation and works on any platform, and
      'triton' will use a Triton GPU kernel. A sequence of implementations can
      be passed, in which case all implementations will be attempted and the
      first successful result will be returned.

  Raises:
    ExceptionGroup: if all implementations fail with their error messages.

  Returns:
    The normalized array with the same shape as the input `x`.
  """
  if implementation is None:
    implementation = _DEFAULT_IMPLEMENTATIONS
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError('`implementation` must not be an empty sequence.')

  # TODO: switch to using the offline autotuned result for the
  # the None automatic case.

  errors = []
  fn = base.Normalization()
  for impl in implementation:
    if isinstance(impl, str):
      if impl == 'triton' and not gpu_utils.has_triton_support():
        errors.append(
            NotImplementedError(
                'Triton not supported on this platform. Please use XLA'
                ' implementation.'
            )
        )
        continue
      if impl not in IMPLEMENTATIONS:
        raise ValueError(f'Unknown implementation: {impl}')
      fn = IMPLEMENTATIONS[impl]

    try:
      return fn(
          x=x,
          scale=scale,
          offset=offset,
          axis=axis,
          epsilon=epsilon,
          scale_offset=scale_offset,
          subtract_mean=subtract_mean,
      )
    except NotImplementedError as e:
      logging.exception('Failed to run implementation')
      errors.append(e)

  raise ExceptionGroup('all implementations failed', errors)


# The ``_rms_norm_xla_*`` paths are the portable reference implementation: they
# serve as the golden in tests AND as the runtime fallback when not on an SM100
# GPU (see the ``('mosaic_gpu', 'xla')`` dispatch order). The fused Mosaic GPU
# kernel in ``pallas_mosaic_gpu`` is the production SM100 path.
def _rms_norm_xla_dense(
    x: jax.Array,
    scale: jax.Array | None,
    *,
    epsilon: float,
) -> jax.Array:
  return base.Normalization()(
      x=x,
      scale=scale,
      offset=None,
      epsilon=epsilon,
      subtract_mean=False,
  )


def _rms_norm_xla_packed(
    x: jax.Array,
    scale: jax.Array | None,
    *,
    epsilon: float,
    qtype: jax.typing.DTypeLike,
    subchannel_size: int,
    quant_scale_dtype: jax.typing.DTypeLike,
) -> jax.Array:
  """XLA reference for ``rms_norm_fuse_quant_packed`` (golden, any platform)."""
  y = _rms_norm_xla_dense(x, scale, epsilon=epsilon)
  if subchannel_size <= 0:
    raise ValueError(f'{subchannel_size=} must be positive.')
  if x.shape[-1] % subchannel_size != 0:
    raise NotImplementedError(
        f'Expected last dimension {x.shape[-1]} to be divisible by'
        f' {subchannel_size=}.'
    )
  tiled_axes = {axis: 1 for axis in range(y.ndim - 1)}
  tiled_axes[y.ndim - 1] = subchannel_size
  q = qwix.quantize(y, qtype, tiled_axes=tiled_axes)
  num_scale_tiles = x.shape[-1] // subchannel_size
  scale = q.scale.reshape((*x.shape[:-1], num_scale_tiles))
  return packed_quant.pack_qvalue_scale(
      q.qvalue, scale.astype(quant_scale_dtype)
  )


def _resolve_implementations(
    implementation: (
        RmsNormImplementation | Sequence[RmsNormImplementation] | None
    ),
) -> Sequence[RmsNormImplementation]:
  if implementation is None:
    return ('mosaic_gpu', 'xla')
  if isinstance(implementation, str):
    return (implementation,)
  if not implementation:
    raise ValueError('`implementation` must not be an empty sequence.')
  return implementation


def _dispatch_rms_norm(implementation, mosaic_fn, xla_fn):
  """Tries each implementation in order; returns the first success.

  ``mosaic_fn`` / ``xla_fn`` are zero-arg thunks already bound to their args,
  so the dense and packed entry points share the SM100 guard logic.
  """
  errors = []
  for impl in _resolve_implementations(implementation):
    if impl == 'mosaic_gpu':
      if pallas_mosaic_gpu is None:
        errors.append(
            NotImplementedError('Mosaic GPU implementation unavailable.')
        )
        continue
      if not gpu_utils.has_mosaic_gpu_support():
        errors.append(
            NotImplementedError('Mosaic GPU not supported on this platform.')
        )
        continue
      if not gpu_utils.is_sm100():
        errors.append(
            NotImplementedError(
                'Mosaic GPU RMSNorm is currently only enabled for SM100 GPUs.'
            )
        )
        continue

    try:
      if impl == 'mosaic_gpu':
        return mosaic_fn()
      elif impl == 'xla':
        return xla_fn()
      else:
        raise ValueError(f'Unknown implementation: {impl}')
    except NotImplementedError as e:
      logging.exception('Failed to run implementation')
      errors.append(e)

  raise ExceptionGroup('all implementations failed', errors)


def _check_rms_norm_args(x, scale, d_model):
  if d_model is not None and x.shape[-1] != d_model:
    raise ValueError(f'Expected last dimension {d_model=}, got {x.shape[-1]}.')
  if scale is not None and scale.shape != (x.shape[-1],):
    raise ValueError(
        f'Expected scale shape {(x.shape[-1],)}, got {scale.shape}.'
    )


def rms_norm(
    x: jax.Array,
    scale: jax.Array | None = None,
    *,
    epsilon: float = 1e-6,
    d_model: int | None = None,
    implementation: (
        RmsNormImplementation
        | Sequence[RmsNormImplementation]
        | None
    ) = None,
) -> jax.Array:
  """Plain RMSNorm. Returns a dense array.

  RMSNorm math is performed in float32 and cast back to ``x.dtype``. For the
  fused RMSNorm + activation-quant epilogue (which returns a packed buffer for
  all-to-all transport) use ``rms_norm_fuse_quant_packed`` instead.

  Args:
    x: Input activations with shape ``(*B, C)``.
    scale: Optional RMSNorm scale with shape ``(C,)``.
    epsilon: RMSNorm epsilon.
    d_model: Optional expected last-axis size.
    implementation: ``'mosaic_gpu'`` for the Mosaic GPU kernel, ``'xla'`` for
      the JAX reference path, or ``None`` to try Mosaic GPU first and fall back
      to XLA.

  Returns:
    Dense RMSNorm output with the same dtype and shape as ``x``.
  """
  _check_rms_norm_args(x, scale, d_model)
  return _dispatch_rms_norm(
      implementation,
      mosaic_fn=lambda: pallas_mosaic_gpu.rms_norm(x, scale, epsilon=epsilon),
      xla_fn=lambda: _rms_norm_xla_dense(x, scale, epsilon=epsilon),
  )


def rms_norm_fuse_quant_packed(
    x: jax.Array,
    scale: jax.Array | None = None,
    *,
    epsilon: float = 1e-6,
    qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    subchannel_size: int = 512,
    quant_scale_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    d_model: int | None = None,
    implementation: (
        RmsNormImplementation
        | Sequence[RmsNormImplementation]
        | None
    ) = None,
) -> jax.Array:
  """RMSNorm fused with packed absmax activation quantization.

  Performs ``RMSNorm(x, scale) -> absmax tiled quantization`` and packs the
  result into a single ``uint8`` buffer whose every token row is
  ``[ qvalue bytes | scale bytes ]`` -- a layout that survives a
  ``ragged_all_to_all`` row permutation, halving the collective's byte volume
  versus transporting bf16. Use ``unpack_rms_norm_quant`` on the receiving side
  to rebuild the ``qwix.QArray`` for the fp8 ragged-dot.

  Args:
    x: Input activations with shape ``(*B, C)``.
    scale: Optional RMSNorm scale with shape ``(C,)``.
    epsilon: RMSNorm epsilon.
    qtype: Quantized activation dtype. ``float8_e4m3fn`` targets the SM100 FP8
      ragged-dot lhs path; ``int8`` is also supported.
    subchannel_size: Last-axis quantization tile size. Must match the fp8
      ragged-dot's ``tile_k``. Defaults to 512.
    quant_scale_dtype: Dtype of the packed quantization scales.
    d_model: Optional expected last-axis size.
    implementation: ``'mosaic_gpu'`` for the Mosaic GPU kernel, ``'xla'`` for
      the JAX reference path, or ``None`` to try Mosaic GPU first and fall back
      to XLA.

  Returns:
    A ``uint8`` array of shape ``(*B, packed_width(C, ...))``. See
    ``unpack_rms_norm_quant`` / ``packed_quant.packed_width`` for the layout.
  """
  _check_rms_norm_args(x, scale, d_model)
  return _dispatch_rms_norm(
      implementation,
      mosaic_fn=lambda: pallas_mosaic_gpu.rms_norm_fuse_quant_packed(
          x,
          scale,
          epsilon=epsilon,
          qtype=qtype,
          subchannel_size=subchannel_size,
          quant_scale_dtype=quant_scale_dtype,
      ),
      xla_fn=lambda: _rms_norm_xla_packed(
          x,
          scale,
          epsilon=epsilon,
          qtype=qtype,
          subchannel_size=subchannel_size,
          quant_scale_dtype=quant_scale_dtype,
      ),
  )
