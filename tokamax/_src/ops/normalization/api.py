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


def _rms_norm_xla(
    x: jax.Array,
    scale: jax.Array | None,
    *,
    epsilon: float,
    quantize: bool,
    qtype: jax.typing.DTypeLike,
    subchannel_size: int,
) -> jax.Array | qwix.QArray:
  y = base.Normalization()(
      x=x,
      scale=scale,
      offset=None,
      epsilon=epsilon,
      subtract_mean=False,
  )
  if not quantize:
    return y
  if subchannel_size <= 0:
    raise ValueError(f'{subchannel_size=} must be positive.')
  if x.shape[-1] % subchannel_size != 0:
    raise NotImplementedError(
        f'Expected last dimension {x.shape[-1]} to be divisible by'
        f' {subchannel_size=}.'
    )
  tiled_axes = {axis: 1 for axis in range(y.ndim - 1)}
  tiled_axes[y.ndim - 1] = subchannel_size
  return qwix.quantize(y, qtype, tiled_axes=tiled_axes)


def rms_norm(
    x: jax.Array,
    scale: jax.Array | None = None,
    *,
    epsilon: float = 1e-6,
    quantize: bool = False,
    qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    subchannel_size: int = 512,
    d_model: int | None = None,
    implementation: (
        RmsNormImplementation
        | Sequence[RmsNormImplementation]
        | None
    ) = None,
) -> jax.Array | qwix.QArray:
  """RMSNorm with optional tiled activation quantization.

  RMSNorm math is performed in float32 and cast back to ``x.dtype``. If
  ``quantize=True``, the result is then absmax-quantized. This lets baseline
  RMSNorm and fused RMSNorm+quantize share the same kernel path.

  Args:
    x: Input activations with shape ``(*B, C)``.
    scale: Optional RMSNorm scale with shape ``(C,)``.
    epsilon: RMSNorm epsilon.
    quantize: If ``True``, return a Qwix QArray. Otherwise return dense RMSNorm
      output with the same dtype and shape as ``x``.
    qtype: Quantized activation dtype used when ``quantize=True``. Defaults to
      ``float8_e4m3fn``.
    subchannel_size: Last-axis quantization tile size used when
      ``quantize=True``. Defaults to 512.
    d_model: Optional expected last-axis size.
    implementation: ``'mosaic_gpu'`` for the Mosaic GPU kernel, ``'xla'`` for
      the JAX reference path, or ``None`` to try Mosaic GPU first and fall back
      to XLA.

  Returns:
    Dense RMSNorm output if ``quantize=False``. Otherwise, a Qwix QArray with
    ``qvalue.shape == x.shape`` and tiled scales of shape
    ``(*B, C // subchannel_size)``.
  """
  if d_model is not None and x.shape[-1] != d_model:
    raise ValueError(f'Expected last dimension {d_model=}, got {x.shape[-1]}.')
  if scale is not None and scale.shape != (x.shape[-1],):
    raise ValueError(f'Expected scale shape {(x.shape[-1],)}, got {scale.shape}.')

  if implementation is None:
    implementation = ('mosaic_gpu', 'xla')
  elif isinstance(implementation, str):
    implementation = (implementation,)
  elif not implementation:
    raise ValueError('`implementation` must not be an empty sequence.')

  errors = []
  for impl in implementation:
    if impl == 'mosaic_gpu':
      if pallas_mosaic_gpu is None:
        errors.append(NotImplementedError('Mosaic GPU implementation unavailable.'))
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
        return pallas_mosaic_gpu.rms_norm(
            x,
            scale,
            epsilon=epsilon,
            quantize=quantize,
            qtype=qtype,
            subchannel_size=subchannel_size,
        )
      elif impl == 'xla':
        return _rms_norm_xla(
            x,
            scale,
            epsilon=epsilon,
            quantize=quantize,
            qtype=qtype,
            subchannel_size=subchannel_size,
        )
      else:
        raise ValueError(f'Unknown implementation: {impl}')
    except NotImplementedError as e:
      logging.exception('Failed to run implementation')
      errors.append(e)

  raise ExceptionGroup('all implementations failed', errors)
