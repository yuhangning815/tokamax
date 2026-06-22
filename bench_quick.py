"""Fast 1-point bench: dot vs epi-dot at block_k=512, block_m=64 (prod shape).

For quick perf iteration on the fused-output epilogue. Prints overhead = epi-dot
- dot (us). Run on B200 with the cuda13 ptxas on PATH.
"""
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import qwix

from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_kernel_sm100_fp8_quant as prod
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_kernel_sm100_fp8_quant_bf16_fp8 as epi


def _bench(f, *a, warmup=15, iters=80):
  o = f(*a); jax.block_until_ready(o)
  for _ in range(warmup):
    o = f(*a)
  jax.block_until_ready(o)
  t0 = time.perf_counter()
  for _ in range(iters):
    o = f(*a)
  jax.block_until_ready(o)
  return (time.perf_counter() - t0) / iters


def main():
  g, m, k, n = 128, 4096, 3072, 4096
  sub, out_sub = 512, 128
  rng = np.random.default_rng(0)
  a_bf16 = jnp.asarray(rng.normal(0, 0.1, (m, k)).astype(np.float32), jnp.bfloat16)
  b_bf16 = jnp.asarray(rng.normal(0, 0.1, (g, k, n)).astype(np.float32), jnp.bfloat16)
  b = qwix.quantize(b_bf16, jnp.int4, tiled_axes={0: 1, 1: sub, 2: 1})
  a8 = qwix.quantize(a_bf16, jnp.float8_e4m3fn, tiled_axes={0: 1, 1: sub})
  gs = jnp.array([
      39, 44, 28, 39, 29, 33, 32, 25, 28, 30, 39, 32, 24, 26, 36, 32, 33, 32,
      32, 25, 24, 33, 40, 30, 35, 38, 24, 22, 27, 33, 30, 37, 19, 35, 38, 40,
      23, 42, 26, 29, 35, 38, 28, 34, 32, 28, 41, 28, 33, 32, 31, 22, 29, 35,
      39, 33, 26, 34, 41, 24, 28, 23, 33, 34, 30, 33, 38, 29, 22, 30, 29, 37,
      37, 42, 36, 39, 26, 35, 31, 27, 34, 30, 28, 33, 30, 37, 30, 28, 33, 22,
      32, 35, 28, 29, 32, 27, 29, 33, 34, 35, 30, 33, 30, 28, 33, 37, 29, 41,
      32, 34, 35, 34, 28, 33, 26, 29, 42, 31, 35, 32, 32, 33, 33, 38, 42, 35,
      34, 33], jnp.int32)

  def cfg(epilogue):
    # Production autotuned config: block_m=32, block_k=256.
    c = common.Config(block_m=32, block_n=128, block_k=256, num_stages=2,
                      split_k=1, split_m=1, persistent=True, post_scale=False,
                      collective=False, grid_minor_dim=common.MatmulDimension.M,
                      grid_tile_width=1)
    if epilogue:
      c = dataclasses.replace(c,
          epilogue_quant_qtype=common.EpilogueQuantDType.FLOAT8_E4M3FN,
          epilogue_quant_subchannel_size=out_sub)
    return c

  dot = jax.jit(lambda a, b, gs: prod.ragged_dot_gpu_fp8_quant_blackwell_kernel(a, b, gs, jnp.bfloat16, cfg(False), None))
  edot = jax.jit(lambda a, b, gs: epi.ragged_dot_gpu_fp8_quant_bf16_fp8_blackwell_kernel(a, b, gs, jnp.bfloat16, cfg(True), None))
  td = _bench(dot, a8, b, gs)
  te = _bench(edot, a8, b, gs)
  print(f"block_k=256 block_m=32: dot {td*1e6:7.1f}us | epi-dot {te*1e6:7.1f}us | overhead {(te-td)*1e6:+6.1f}us")


if __name__ == "__main__":
  main()
