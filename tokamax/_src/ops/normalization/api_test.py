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
from collections.abc import Callable, Mapping
import functools
import typing
from typing import Final
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import export
import jax.numpy as jnp
import qwix
from tokamax._src import gpu_utils
from tokamax._src import hlo_utils
from tokamax._src.ops.normalization import api
from tokamax._src.ops.normalization import base
from tokamax._src.ops.normalization import pallas_triton_vjp as pl_norm_vjp
from tokamax._src.ops.normalization import test_base

_IMPLEMENTATIONS: Final[tuple[str | None, ...]] = typing.get_args(
    api.Implementation
) + (None,)

_IMPLEMENTATIONS_VJP: Mapping[str, Callable[..., jax.Array]] = dict(
    xla=base.NormalizationVjp(),
    triton=pl_norm_vjp.PallasTritonNormalizationVjp(),
)

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv[:1])  # Parse absl flags before absltest.main()


def _get_input_data(shape, dtype=jnp.bfloat16):
  rng0, rng1 = jax.random.split(jax.random.PRNGKey(0))
  x = jax.random.normal(rng0, shape, dtype=dtype)
  scale = jax.random.uniform(rng1, (shape[-1],), dtype=dtype)
  offset = jax.random.uniform(rng1, (shape[-1],), dtype=dtype)
  return x, scale, offset


class LayerNormTest(parameterized.TestCase):

  @parameterized.parameters(*_IMPLEMENTATIONS)
  def test_basic_api(self, implementation):
    if implementation == "triton" and not gpu_utils.has_triton_support():
      self.skipTest("Triton not supported on this platform.")

    x, scale, offset = _get_input_data(shape=(128, 32), dtype=jnp.bfloat16)

    @jax.jit
    @functools.partial(jax.value_and_grad)
    def norm_fn(x, scale, offset):
      out = api.layer_norm(x, scale, offset, implementation=implementation)
      return jnp.sum(out)

    @jax.jit
    @functools.partial(jax.value_and_grad)
    def norm_fn_ref(x, scale, offset):
      out = api.layer_norm(x, scale, offset, implementation="xla")
      return jnp.sum(out)

    out = norm_fn(x, scale, offset)
    out_golden = norm_fn_ref(x, scale, offset)

    with self.subTest("value"):
      chex.assert_trees_all_close(out, out_golden)

    with self.subTest("symbolic serialization"):
      (a,) = export.symbolic_shape("a")
      x_shape = jax.ShapeDtypeStruct((a, 32), x.dtype)
      param_shape = jax.ShapeDtypeStruct(scale.shape, x.dtype)

      if implementation == "triton":
        with self.assertRaisesRegex(Exception, "all implementations failed"):
          export.export(
              norm_fn,
              disabled_checks=hlo_utils.DISABLE_JAX_EXPORT_CHECKS,
          )(x_shape, param_shape, param_shape)
        # Change shape to non-symbolic to test standard export.
        x_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

      exported = export.export(
          norm_fn,
          disabled_checks=hlo_utils.DISABLE_JAX_EXPORT_CHECKS,
      )(x_shape, param_shape, param_shape)
      serialized: bytearray = exported.serialize()
      rehydrated_norm: export.Exported = export.deserialize(serialized)
      out_rehydrated = jax.jit(rehydrated_norm.call)(x, scale, offset)
      chex.assert_trees_all_close(out_rehydrated, out_golden)

    with self.subTest("correct_implementation_used"):
      opspecs = hlo_utils.get_opspecs(
          norm_fn.lower(x, scale, offset),
          include_xla_kernels=(implementation == "xla"),
      )
      triton_impl = type(api.IMPLEMENTATIONS.get("triton"))
      triton_vjp_impl = type(_IMPLEMENTATIONS_VJP["triton"])
      match implementation:
        case "triton":
          self.assertIsInstance(opspecs[0].op, triton_impl)
          self.assertIsInstance(opspecs[1].op, triton_vjp_impl)
        case "xla":
          self.assertIsInstance(opspecs[0].op, type(api.IMPLEMENTATIONS["xla"]))
        case None:
          if jax.default_backend() == "gpu":
            # Ensure the Triton implementation is used.
            self.assertIsInstance(opspecs[0].op, triton_impl)
            self.assertIsInstance(opspecs[1].op, triton_vjp_impl)


class RmsNormTest(parameterized.TestCase):

  @parameterized.product(
      shape=((128, 512), (2, 128, 1024)),
      subchannel_size=(512, 256),
      use_scale=(True, False),
  )
  def test_xla_quantized_api(self, shape, subchannel_size, use_scale):
    if shape[-1] % subchannel_size != 0:
      self.skipTest("shape[-1] must be divisible by subchannel_size.")

    rng_x, rng_scale = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng_x, shape, dtype=jnp.bfloat16)
    scale = (
        jax.random.uniform(rng_scale, (shape[-1],), dtype=jnp.bfloat16)
        if use_scale
        else None
    )
    actual = api.rms_norm(
        x,
        scale,
        quantize=True,
        qtype=jnp.float8_e4m3fn,
        subchannel_size=subchannel_size,
        implementation="xla",
    )
    expected_y = api.layer_norm(
        x,
        scale,
        offset=None,
        subtract_mean=False,
        implementation="xla",
    )
    tiled_axes = {axis: 1 for axis in range(expected_y.ndim - 1)}
    tiled_axes[expected_y.ndim - 1] = subchannel_size
    expected = qwix.quantize(
        expected_y,
        jnp.float8_e4m3fn,
        tiled_axes=tiled_axes,
    )

    self.assertEqual(actual.qvalue.shape, x.shape)
    self.assertEqual(
        actual.scale.shape,
        (*shape[:-1], shape[-1] // subchannel_size),
    )
    chex.assert_trees_all_close(
        qwix.dequantize(actual), qwix.dequantize(expected)
    )

  @parameterized.product(
      shape=((128, 512), (2, 128, 1024)),
      use_scale=(True, False),
  )
  def test_xla_dense_api(self, shape, use_scale):
    rng_x, rng_scale = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(rng_x, shape, dtype=jnp.bfloat16)
    scale = (
        jax.random.uniform(rng_scale, (shape[-1],), dtype=jnp.bfloat16)
        if use_scale
        else None
    )
    actual = api.rms_norm(
        x,
        scale,
        quantize=False,
        implementation="xla",
    )
    expected = api.layer_norm(
        x,
        scale,
        offset=None,
        subtract_mean=False,
        implementation="xla",
    )
    chex.assert_trees_all_close(actual, expected)


class LayerNormTritonTest(test_base.NormalizationTestBase):
  IMPL = "triton"

  def __init__(self, *args):
    if not gpu_utils.has_triton_support():
      norm_fn = functools.partial(api.layer_norm, implementation="xla")
      super().__init__(*args, norm_fn=norm_fn)
    else:
      norm_fn = functools.partial(api.layer_norm, implementation=self.IMPL)
      super().__init__(*args, norm_fn=norm_fn)


class LayerNormXlATest(test_base.NormalizationTestBase):
  IMPL = "xla"

  def __init__(self, *args):
    norm_fn = functools.partial(api.layer_norm, implementation=self.IMPL)
    super().__init__(*args, norm_fn=norm_fn)


if __name__ == "__main__":
  absltest.main()
