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
"""Tests for the all-to-all packed quantization wire format.

These cover ``packed_quant``, which is *production* code (``unpack_rms_norm_quant``
runs on the GPU receiving side of the all-to-all). Because the module has no
SM100 / Pallas dependency, the byte math runs on any backend, so these tests are
the portable gate that runs in ordinary CI (no GPU required). The fused Mosaic
GPU kernel that produces the packed buffer is covered separately, and only on
SM100, in ``pallas_mosaic_gpu_test``.
"""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import qwix
from tokamax._src.ops.normalization import packed_quant


def _quantize_reference(y, qtype, subchannel_size):
  """A qwix.QArray golden for normalized activations ``y`` of shape (*B, C)."""
  tiled_axes = {axis: 1 for axis in range(y.ndim - 1)}
  tiled_axes[y.ndim - 1] = subchannel_size
  return qwix.quantize(y, qtype, tiled_axes=tiled_axes)


class PackedQuantTest(parameterized.TestCase):

  @parameterized.product(
      c=(512, 1024),
      qtype=(jnp.float8_e4m3fn, jnp.int8),
      subchannel_size=(512, 256, 128),
      quant_scale_dtype=(jnp.bfloat16, jnp.float32),
  )
  def test_packed_width(self, c, qtype, subchannel_size, quant_scale_dtype):
    if c % subchannel_size != 0:
      self.skipTest("c must be divisible by subchannel_size.")
    width = packed_quant.packed_width(
        c, qtype, subchannel_size, quant_scale_dtype
    )
    value_bytes = c * jnp.dtype(qtype).itemsize
    scale_bytes = (c // subchannel_size) * jnp.dtype(quant_scale_dtype).itemsize
    self.assertEqual(width, value_bytes + scale_bytes)

  def test_packed_width_rejects_bad_subchannel(self):
    with self.assertRaises(ValueError):
      packed_quant.packed_width(1024, jnp.float8_e4m3fn, subchannel_size=300)

  def test_check_supported_qtype_rejects_unsupported(self):
    with self.assertRaises(NotImplementedError):
      packed_quant.check_supported_qtype(jnp.float16)

  @parameterized.product(
      shape=((128, 512), (2, 128, 1024), (8, 1024)),
      qtype=(jnp.float8_e4m3fn, jnp.int8),
      subchannel_size=(512, 256, 128),
      quant_scale_dtype=(jnp.bfloat16, jnp.float32),
  )
  def test_pack_unpack_roundtrip_is_exact(
      self, shape, qtype, subchannel_size, quant_scale_dtype
  ):
    if shape[-1] % subchannel_size != 0:
      self.skipTest("shape[-1] must be divisible by subchannel_size.")
    c = shape[-1]
    num_scale_tiles = c // subchannel_size

    rng_q, rng_s = jax.random.split(jax.random.PRNGKey(0))
    # Arbitrary in-range quant values and positive scales.
    qvalue = (jax.random.normal(rng_q, shape) * 8).astype(qtype)
    quant_scale = jax.random.uniform(
        rng_s, (*shape[:-1], num_scale_tiles), minval=0.1, maxval=2.0
    ).astype(quant_scale_dtype)

    packed = packed_quant.pack_qvalue_scale(qvalue, quant_scale)
    width = packed_quant.packed_width(
        c, qtype, subchannel_size, quant_scale_dtype
    )
    self.assertEqual(packed.dtype, jnp.dtype(jnp.uint8))
    self.assertEqual(packed.shape, (*shape[:-1], width))

    restored = packed_quant.unpack_rms_norm_quant(
        packed,
        c=c,
        qtype=qtype,
        subchannel_size=subchannel_size,
        quant_scale_dtype=quant_scale_dtype,
    )
    self.assertEqual(restored.qtype, jnp.dtype(qtype))
    # Bytes survive the round-trip exactly (bit-for-bit), not just approximately.
    chex.assert_trees_all_equal(
        restored.qvalue.view(jnp.uint8), qvalue.view(jnp.uint8)
    )
    chex.assert_trees_all_equal(
        restored.scale.view(jnp.uint8), quant_scale.view(jnp.uint8)
    )

  @parameterized.product(
      shape=((128, 512), (2, 128, 1024)),
      qtype=(jnp.float8_e4m3fn, jnp.int8),
      subchannel_size=(512, 256),
  )
  def test_survives_row_permutation(self, shape, qtype, subchannel_size):
    """A ragged-all-to-all permutes rows; value+scale must travel together."""
    if shape[-1] % subchannel_size != 0:
      self.skipTest("shape[-1] must be divisible by subchannel_size.")
    c = shape[-1]
    num_scale_tiles = c // subchannel_size

    rng_q, rng_s = jax.random.split(jax.random.PRNGKey(1))
    qvalue = (jax.random.normal(rng_q, shape) * 8).astype(qtype)
    quant_scale = jax.random.uniform(
        rng_s, (*shape[:-1], num_scale_tiles), minval=0.1, maxval=2.0
    ).astype(jnp.bfloat16)

    packed = packed_quant.pack_qvalue_scale(qvalue, quant_scale)
    width = packed.shape[-1]

    # Flatten to (M, width) and permute rows, as the collective would.
    flat = packed.reshape((-1, width))
    perm = jax.random.permutation(jax.random.PRNGKey(2), flat.shape[0])
    restored = packed_quant.unpack_rms_norm_quant(
        flat[perm], c=c, qtype=qtype, subchannel_size=subchannel_size
    )

    qflat = qvalue.reshape((-1, c))
    sflat = quant_scale.reshape((-1, num_scale_tiles))
    chex.assert_trees_all_equal(
        restored.qvalue.view(jnp.uint8), qflat[perm].view(jnp.uint8)
    )
    chex.assert_trees_all_equal(
        restored.scale.view(jnp.uint8), sflat[perm].view(jnp.uint8)
    )

  @parameterized.product(
      shape=((128, 512), (2, 128, 1024)),
      qtype=(jnp.float8_e4m3fn, jnp.int8),
      subchannel_size=(512, 256),
  )
  def test_unpacked_dequant_matches_qwix(self, shape, qtype, subchannel_size):
    """unpack(pack(qwix.quantize(y))) dequantizes back to the qwix golden."""
    if shape[-1] % subchannel_size != 0:
      self.skipTest("shape[-1] must be divisible by subchannel_size.")
    c = shape[-1]
    num_scale_tiles = c // subchannel_size

    y = jax.random.normal(jax.random.PRNGKey(3), shape, dtype=jnp.bfloat16)
    golden = _quantize_reference(y, qtype, subchannel_size)

    scale = golden.scale.reshape((*shape[:-1], num_scale_tiles))
    packed = packed_quant.pack_qvalue_scale(
        golden.qvalue, scale.astype(jnp.bfloat16)
    )
    restored = packed_quant.unpack_rms_norm_quant(
        packed, c=c, qtype=qtype, subchannel_size=subchannel_size
    )
    chex.assert_trees_all_close(
        qwix.dequantize(restored),
        qwix.dequantize(golden),
        atol=5e-3,
        rtol=5e-3,
    )

  def test_unpack_rejects_wrong_width(self):
    packed = jnp.zeros((4, 999), dtype=jnp.uint8)
    with self.assertRaises(ValueError):
      packed_quant.unpack_rms_norm_quant(
          packed, c=512, qtype=jnp.float8_e4m3fn, subchannel_size=512
      )


if __name__ == "__main__":
  absltest.main()
