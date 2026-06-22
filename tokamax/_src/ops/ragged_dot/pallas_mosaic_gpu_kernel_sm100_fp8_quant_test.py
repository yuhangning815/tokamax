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

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import qwix
from tokamax._src import gpu_utils
from tokamax._src import quantization
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_kernel_sm100_fp8_quant_bf16_fp8 as sm100_fp8_quant_bf16_fp8
from tokamax._src.ops.ragged_dot import test_base
from typing_extensions import override

# Config for enabling use_native_int8_mma for testing quant_i8 kernel.
_CONFIG = pallas_mosaic_gpu.Config(
    block_m=16,
    block_n=128,
    block_k=128,
    num_stages=2,
    split_k=1,
    persistent=True,
    collective=False,
    grid_minor_dim=pallas_mosaic_gpu.common.MatmulDimension.M,
    grid_tile_width=1,
)


class PallasMosaicGpuKernelSm100FP8QuantTest(test_base.RaggedDotTestBase):
  """Tests for Pallas Mosaic GPU kernel with fp8xi4 quantization."""

  def __init__(self, *args):
    op = pallas_mosaic_gpu.PallasMosaicGpuRaggedDot()

    def fn(lhs, rhs, **kwargs):
      config = test_base.test_config.get()
      expect_supported = True
      lhs_ = jax.eval_shape(quantization.as_array_or_qarray, lhs)
      rhs_ = jax.eval_shape(quantization.as_array_or_qarray, rhs)

      device_kind = jax.devices()[0].device_kind.lower()
      if "b200" not in device_kind:
        expect_supported = False

      if config is None:
        config = _CONFIG

      tile_k = rhs_.scale_tile_shape[1] if isinstance(rhs_, qwix.QArray) else 0
      config = dataclasses.replace(
          config,
          block_k=min(config.block_k, max(tile_k, 32)),
      )
      if (
          not isinstance(rhs_, qwix.QArray)
          or (rhs_.qtype != jnp.int4)
          or (rhs_.scale_tile_shape[0] != 1)
          or (rhs_.scale_tile_shape[1] % config.block_k != 0)
          or (rhs_.scale_tile_shape[2] != 1)
      ):
        expect_supported = False

      if isinstance(lhs_, qwix.QArray) and lhs_.qtype != jnp.float8_e4m3fn:
        expect_supported = False

      if expect_supported:
        return op.replace(config=config)(lhs, rhs, **kwargs)

      with self.assertRaises(NotImplementedError) as e:
        _ = op.replace(config=config)(lhs, rhs, **kwargs)
      self.skipTest(f"Test not supported: {e.exception}")

    super().__init__(*args, dot_fn=fn)

  @override
  def _test_quantized(
      self,
      a_dtype,
      b_dtype,
      a_tile_shape,
      b_tile_shape,
      use_as_qarray,
      activation=None,
      # (num_groups, m, k, n)
      task=(8, 512, 256, 512),
  ):
    self.skipTest("Not supported.")

  @parameterized.product(
      subchannels=(512, 256, 128),
      use_as_qarray=(True, False),
      activation=(None, test_base.relu, jax.nn.tanh),
      task=(
          (8, 512, 512, 512),
          (8, 512, 512, 512),
          (32, 4096, 4096, 4096),
          (32, 16384, 4096, 4096),
          (16, 8192, 4096, 4096),
      ),
      block_m=(16, 32),
      block_k=(512, 256, 128),
  )
  def test_wi4_afp8_quantized(
      self, subchannels, use_as_qarray, activation, task, block_m, block_k
  ):
    if subchannels < block_k:
      self.skipTest("subchannels < block_k")
    config = dataclasses.replace(
        _CONFIG, block_m=block_m, block_k=block_k
    )
    with test_base.ConfigManager(config):
      super()._test_quantized(
          "float8_e4m3fn",
          "int4",
          (1, subchannels),
          (1, subchannels, 1),
          use_as_qarray,
          activation,
          task,
      )

  @override
  def _test_preferred_element_type(self, out_type):
    self.skipTest("Not supported.")

  @override
  def _test_vjp(self, num_groups, m, k, n, activation=None):
    self.skipTest("Not supported.")

  @override
  def _test_bench(self, spec):
    self.skipTest("Not supported.")

  @override
  def _test_simple(self, dtype):
    self.skipTest("Not supported.")

  @override
  def test_padded(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        random_groups=True,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, 128),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 128, 1),
    )
    expected = test_base.ref(a, b, group_sizes)
    actual = self._dot_fn(a, b, group_sizes=group_sizes, activation=None)
    count = sum(group_sizes)
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.01, rtol=0.005
    )

  @override
  def test_group_sizes(self):
    num_groups, m, k, n = 8, 1024, 128, 256
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        random_groups=False,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, 128),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 128, 1),
    )
    expected = test_base.ref(a, b, group_sizes=group_sizes)
    group_sizes = base.GroupSizes(group_sizes, (1,) * num_groups)
    actual = self._dot_fn(a, b, group_sizes=group_sizes, activation=None)
    chex.assert_trees_all_close(actual, expected, atol=0.01, rtol=0.005)

  @override
  def test_zero_group_sizes(self):
    num_groups, m, k, n = 8, 1024, 512, 256
    a, b, group_sizes = self._create_inputs(
        num_groups,
        m,
        k,
        n,
        jnp.bfloat16,
        random_groups=True,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, 256),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 256, 1),
    )

    # Test all possible patterns of zero group sizes.
    for i in range(2**num_groups):
      group_sizes_ = jnp.where(jnp.unpackbits(jnp.uint8(i)), group_sizes, 0)
      with self.subTest(f"group_sizes={group_sizes_.tolist()}"):
        expected = test_base.ref(a, b, group_sizes_)
        actual = self._dot_fn(a, b, group_sizes=group_sizes_, activation=None)
        count = sum(group_sizes_)
        chex.assert_trees_all_close(
            actual[:count], expected[:count], atol=0.01, rtol=0.005
        )

  def _dequant(self, qarray, subchannel):
    q = qarray.qvalue.astype(jnp.float32)
    s = jnp.repeat(qarray.scale.astype(jnp.float32), subchannel, axis=1)
    return q * s

  @parameterized.product(block_k=(128,), activation=(None, test_base.relu))
  def test_epilogue_quant(self, block_k, activation):
    # New arch: ONE accumulator of block_n N-cols per CTA -> the fused output
    # subchannel must equal block_n (= 128). Activation subchannel matched.
    num_groups, m, k, n = 8, 512, 256, 512
    sub = 128
    a, b, group_sizes = self._create_inputs(
        num_groups, m, k, n, jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    config = dataclasses.replace(
        _CONFIG,
        block_m=16,
        block_n=128,
        block_k=block_k,
        epilogue_quant_qtype=common.EpilogueQuantDType.FLOAT8_E4M3FN,
        epilogue_quant_subchannel_size=sub,
    )
    out = sm100_fp8_quant_bf16_fp8.ragged_dot_gpu_fp8_quant_bf16_fp8_blackwell_kernel(
        a, b, jnp.asarray(group_sizes), jnp.bfloat16, config, activation
    )
    self.assertIsInstance(out, qwix.QArray)
    self.assertEqual(out.qvalue.dtype, jnp.float8_e4m3fn)
    actual = self._dequant(out, sub)
    expected = test_base.ref(a, b, group_sizes, activation)
    count = int(sum(group_sizes))
    # fp8_e4m3fn-quantized output -> a few % error.
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.06, rtol=0.1
    )

  @parameterized.product(block_k=(128,), activation=(None, test_base.relu))
  def test_relaxed_activation_subchannel(self, block_k, activation):
    # Activation subchannel (128) finer than weight subchannel (512); dense out.
    # NOTE: uses a 4x ratio (weight 512 / act 128). A 2x ratio (tile_k ==
    # 2*tile_xk) collides with the production kernel's x_sum-encoded-scale
    # detection (`scales.shape[1] == 2 * (k_x // tile_k)`) and is unsupported.
    num_groups, m, k, n = 8, 512, 512, 512
    a, b, group_sizes = self._create_inputs(
        num_groups, m, k, n, jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, 128),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, 512, 1),
    )
    config = dataclasses.replace(
        _CONFIG, block_m=16, block_n=128, block_k=block_k
    )
    out = sm100_fp8_quant_bf16_fp8.ragged_dot_gpu_fp8_quant_bf16_fp8_blackwell_kernel(
        a, b, jnp.asarray(group_sizes), jnp.bfloat16, config, activation
    )
    expected = test_base.ref(a, b, group_sizes, activation)
    count = int(sum(group_sizes))
    chex.assert_trees_all_close(
        out[:count], expected[:count], atol=0.01, rtol=0.005
    )

  @parameterized.product(
      block_k=(128,),
      activation=(None, test_base.relu),
      block_m=(16, 64),
      group_sizes_pat=(
          # cumsum starts not multiples of align_tile(8) -> a tile straddles two
          # groups (start_within_block != 0); the fused scale store must write
          # ONLY this group's valid rows, not the whole aligned bucket.
          (33, 67, 50, 80, 70, 60, 90, 62),  # mixed ragged
          # size-1 groups (actual_size==1 single-row scatter) next to big groups
          # that span many tiles (first ragged, middle aligned, last partial).
          (1, 7, 199, 1, 100, 50, 151, 3),
          # zero-size groups (actual_size==0 -> no write) interleaved.
          (0, 130, 0, 200, 0, 98, 84, 0),
      ),
  )
  def test_epilogue_quant_ragged(
      self, block_k, activation, block_m, group_sizes_pat
  ):
    # Corner cases for the ragged fused scale store: a tile straddling two
    # groups must scatter only [start_within_block, +actual_size); the old store
    # wrote the whole bucket and corrupted neighbours' scale rows.
    num_groups, m, k, n = 8, 512, 256, 512
    sub = 128
    a, b, _ = self._create_inputs(
        num_groups, m, k, n, jnp.bfloat16,
        use_as_qarray=False,
        quant_a_dtype=jnp.float8_e4m3fn,
        a_tile_shape=(1, sub),
        quant_b_dtype=jnp.int4,
        b_tile_shape=(1, sub, 1),
    )
    group_sizes = jnp.array(group_sizes_pat, jnp.int32)
    assert int(group_sizes.sum()) == m, group_sizes_pat
    config = dataclasses.replace(
        _CONFIG,
        block_m=block_m,
        block_n=128,
        block_k=block_k,
        epilogue_quant_qtype=common.EpilogueQuantDType.FLOAT8_E4M3FN,
        epilogue_quant_subchannel_size=sub,
    )
    out = sm100_fp8_quant_bf16_fp8.ragged_dot_gpu_fp8_quant_bf16_fp8_blackwell_kernel(
        a, b, group_sizes, jnp.bfloat16, config, activation
    )
    self.assertIsInstance(out, qwix.QArray)
    actual = self._dequant(out, sub)
    expected = test_base.ref(a, b, group_sizes, activation)
    count = int(group_sizes.sum())
    chex.assert_trees_all_close(
        actual[:count], expected[:count], atol=0.06, rtol=0.1
    )

  def setUp(self):
    if jax.default_backend() == "tpu":
      self.skipTest("Not supported on TPUs.")
    if not gpu_utils.is_sm100():
      self.skipTest("Not supported on non-sm100 GPUs.")
    super().setUp()


if __name__ == "__main__":
  absltest.main()
