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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import gpu_utils
from tokamax._src.ops.normalization import api
from tokamax._src.ops.normalization import pallas_mosaic_gpu


class PallasMosaicGpuRmsNormTest(parameterized.TestCase):
  """Tests the production SM100 Mosaic GPU kernels against the XLA golden.

  SM100-only (skipped elsewhere). The portable byte-level wire-format checks
  live in ``packed_quant_test`` and run without a GPU.
  """

  def setUp(self):
    if not gpu_utils.is_sm100():
      self.skipTest("Mosaic GPU RMSNorm is only enabled on SM100.")
    super().setUp()

  @parameterized.product(
      shape=((128, 512), (8, 128, 1024)),
      subchannel_size=(512, 256),
      use_scale=(True, False),
  )
  def test_fuse_quant_packed_roundtrip(self, shape, subchannel_size, use_scale):
    if shape[-1] % subchannel_size != 0:
      self.skipTest("shape[-1] must be divisible by subchannel_size.")

    rng_x, rng_scale = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng_x, shape, dtype=jnp.bfloat16)
    scale = (
        jax.random.uniform(rng_scale, (shape[-1],), dtype=jnp.bfloat16)
        if use_scale
        else None
    )
    qtype = jnp.float8_e4m3fn

    packed = pallas_mosaic_gpu.rms_norm_fuse_quant_packed(
        x,
        scale,
        qtype=qtype,
        subchannel_size=subchannel_size,
    )
    width = pallas_mosaic_gpu.packed_width(
        shape[-1], qtype, subchannel_size, jnp.bfloat16
    )
    self.assertEqual(packed.dtype, jnp.dtype(jnp.uint8))
    self.assertEqual(packed.shape, (*shape[:-1], width))

    restored = pallas_mosaic_gpu.unpack_rms_norm_quant(
        packed, c=shape[-1], qtype=qtype, subchannel_size=subchannel_size
    )
    self.assertEqual(restored.qvalue.shape, x.shape)
    self.assertEqual(
        restored.scale.shape, (*shape[:-1], shape[-1] // subchannel_size)
    )
    self.assertEqual(restored.qtype, jnp.dtype(qtype))

    expected_packed = api.rms_norm_fuse_quant_packed(
        x,
        scale,
        qtype=qtype,
        subchannel_size=subchannel_size,
        implementation="xla",
    )
    expected = pallas_mosaic_gpu.unpack_rms_norm_quant(
        expected_packed, c=shape[-1], qtype=qtype, subchannel_size=subchannel_size
    )
    chex.assert_trees_all_close(
        qwix.dequantize(restored),
        qwix.dequantize(expected),
        atol=5e-3,
        rtol=5e-3,
    )

  @parameterized.product(
      shape=((128, 512), (8, 128, 1024)),
      use_scale=(True, False),
  )
  def test_dense_matches_xla(self, shape, use_scale):
    rng_x, rng_scale = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng_x, shape, dtype=jnp.bfloat16)
    scale = (
        jax.random.uniform(rng_scale, (shape[-1],), dtype=jnp.bfloat16)
        if use_scale
        else None
    )

    actual = pallas_mosaic_gpu.rms_norm(x, scale)
    expected = api.rms_norm(x, scale, implementation="xla")

    chex.assert_trees_all_close(actual, expected, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
  absltest.main()
