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

  def setUp(self):
    if not gpu_utils.is_sm100():
      self.skipTest("Mosaic GPU RMSNorm is only enabled on SM100.")
    super().setUp()

  @parameterized.product(
      shape=((128, 512), (8, 128, 1024)),
      subchannel_size=(512, 256),
      use_scale=(True, False),
  )
  def test_quantized_matches_xla_quantize(
      self, shape, subchannel_size, use_scale
  ):
    if shape[-1] % subchannel_size != 0:
      self.skipTest("shape[-1] must be divisible by subchannel_size.")

    rng_x, rng_scale = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng_x, shape, dtype=jnp.bfloat16)
    scale = (
        jax.random.uniform(rng_scale, (shape[-1],), dtype=jnp.bfloat16)
        if use_scale
        else None
    )

    actual = pallas_mosaic_gpu.rms_norm(
        x,
        scale,
        quantize=True,
        qtype=jnp.float8_e4m3fn,
        subchannel_size=subchannel_size,
    )
    expected = api.rms_norm(
        x,
        scale,
        quantize=True,
        qtype=jnp.float8_e4m3fn,
        subchannel_size=subchannel_size,
        implementation="xla",
    )

    self.assertEqual(actual.qvalue.shape, x.shape)
    self.assertEqual(
        actual.scale.shape, (*shape[:-1], shape[-1] // subchannel_size)
    )
    self.assertEqual(actual.qtype, jnp.dtype(jnp.float8_e4m3fn))
    chex.assert_trees_all_close(
        qwix.dequantize(actual),
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

    actual = pallas_mosaic_gpu.rms_norm(x, scale, quantize=False)
    expected = api.rms_norm(
        x,
        scale,
        quantize=False,
        implementation="xla",
    )

    chex.assert_trees_all_close(actual, expected, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
  absltest.main()
