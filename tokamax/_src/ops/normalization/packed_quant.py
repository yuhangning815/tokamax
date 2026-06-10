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
"""Wire format for transporting quantized activations through all-to-all.

``jax.lax.ragged_all_to_all`` (the MoE dispatch collective) permutes *rows* of
a plain dense array and cannot carry a ``qwix.QArray``. These helpers pack a
quantized activation's value+scale into one contiguous ``uint8`` row so the
collective moves both together, and unpack it back into a ``qwix.QArray`` on the
receiving side.

These are *production* helpers, not just test references -- they compile to
lightweight XLA ops (slice / bitcast / concatenate) and run on whatever device
holds the data, i.e. the GPU in deployment:

* ``unpack_rms_norm_quant`` runs on the all-to-all *receiving side* in the real
  pipeline (GPU), rebuilding the QArray that the fp8 ragged-dot consumes.
* ``pack_qvalue_scale`` is the *reference / fallback* packer; the production
  packing is fused inside the Mosaic GPU kernel
  (``pallas_mosaic_gpu.rms_norm_fuse_quant_packed``). It is also used by the XLA
  reference path.

The module is deliberately pure JAX with no Mosaic GPU / Pallas import so it
loads on *any* platform -- the receiving side may not be an SM100 GPU, and the
tests run on CPU. "Pure JAX" means no SM100 dependency, not "CPU only".

Per-row packed layout (``dtype=uint8``)::

    [  C * itemsize(qtype) bytes of qvalue  |  T * itemsize(scale) bytes of scale  ]
    T = num_scale_tiles = C // subchannel_size
"""

import jax
from jax import lax
import jax.numpy as jnp
import qwix


def check_supported_qtype(qtype: jax.typing.DTypeLike) -> jnp.dtype:
  qtype = jnp.dtype(qtype)
  if qtype not in (jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.int8)):
    raise NotImplementedError(f"Unsupported RMSNorm quantization {qtype=}.")
  return qtype


def packed_width(
    c: int,
    qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    subchannel_size: int = 512,
    quant_scale_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> int:
  """Returns the per-row byte width of the packed quant buffer.

  ``packed_width = C * itemsize(qtype) + itemsize(scale) * (C // subchannel)``.
  For the default fp8 value + bf16 scale this is ``C + 2 * (C // subchannel)``.
  """
  if subchannel_size <= 0 or c % subchannel_size != 0:
    raise ValueError(f"{c=} must be a positive multiple of {subchannel_size=}.")
  num_scale_tiles = c // subchannel_size
  value_bytes = c * jnp.dtype(qtype).itemsize
  scale_bytes = num_scale_tiles * jnp.dtype(quant_scale_dtype).itemsize
  return value_bytes + scale_bytes


def pack_qvalue_scale(qvalue: jax.Array, quant_scale: jax.Array) -> jax.Array:
  """Packs ``(*B, C)`` qvalue and ``(*B, T)`` scale into one ``uint8`` buffer.

  Each row becomes ``[ qvalue bytes | scale bytes ]`` so it stays contiguous
  under a ragged-all-to-all row permutation. Used by the XLA reference and as
  the post-kernel fallback packer for the Mosaic GPU kernel.
  """
  *batch, _ = qvalue.shape
  qbytes = lax.bitcast_convert_type(qvalue, jnp.uint8).reshape((*batch, -1))
  sbytes = lax.bitcast_convert_type(quant_scale, jnp.uint8).reshape((*batch, -1))
  return lax.concatenate([qbytes, sbytes], dimension=qvalue.ndim - 1)


def unpack_rms_norm_quant(
    packed: jax.Array,
    *,
    c: int,
    qtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    subchannel_size: int = 512,
    quant_scale_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> qwix.QArray:
  """Inverse of ``pack_qvalue_scale`` / ``rms_norm_fuse_quant_packed``.

  Reconstructs the ``qwix.QArray`` (qvalue ``(*B, C)``, scale
  ``(*B, C // subchannel_size)``) that the fp8 ragged-dot lhs path expects.
  Run this on the receiving side after the all-to-all.
  """
  qtype = check_supported_qtype(qtype)
  quant_scale_dtype = jnp.dtype(quant_scale_dtype)
  if subchannel_size <= 0 or c % subchannel_size != 0:
    raise ValueError(f"{c=} must be a positive multiple of {subchannel_size=}.")
  num_scale_tiles = c // subchannel_size
  value_bytes = c * qtype.itemsize
  expected = packed_width(c, qtype, subchannel_size, quant_scale_dtype)
  if packed.shape[-1] != expected:
    raise ValueError(
        f"Expected packed width {expected}, got {packed.shape[-1]}."
    )
  *batch, _ = packed.shape

  qbytes = packed[..., :value_bytes]
  qvalue = lax.bitcast_convert_type(qbytes, qtype).reshape((*batch, c))

  sbytes = packed[..., value_bytes:]
  sbytes = sbytes.reshape((*batch, num_scale_tiles, quant_scale_dtype.itemsize))
  scale = lax.bitcast_convert_type(sbytes, quant_scale_dtype)
  scale = scale.reshape((*batch, num_scale_tiles))

  return qwix.QArray(qvalue, scale, qtype=qtype)
