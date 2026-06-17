"""Config sweep for the fused-output kernel on the production shape (new arch).

The new kernel honors block_m / grid_minor_dim / grid_tile_width (num_stages and
persistent are ignored -- stages computed internally). The dot is latency-bound,
so block_m (~= avg group size M/G = 32) is the prime lever. block_k=512, block_n
fixed at 128 (== out_sub).
"""
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import qwix

from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_kernel_sm100_fp8_quant_bf16_fp8 as epi


def _bench(f, *args, warmup=10, iters=50):
  o = f(*args)
  jax.block_until_ready(o)
  for _ in range(warmup):
    o = f(*args)
  jax.block_until_ready(o)
  t0 = time.perf_counter()
  for _ in range(iters):
    o = f(*args)
  jax.block_until_ready(o)
  return (time.perf_counter() - t0) / iters


def main():
  g, m, k, n = 128, 4096, 3072, 4096
  sub, out_sub = 512, 128
  rng = np.random.default_rng(0)
  a_bf16 = jnp.asarray(rng.normal(0, 0.1, (m, k)).astype(np.float32), jnp.bfloat16)
  b_bf16 = jnp.asarray(rng.normal(0, 0.1, (g, k, n)).astype(np.float32), jnp.bfloat16)
  b = qwix.quantize(b_bf16, jnp.int4, tiled_axes={0: 1, 1: sub, 2: 1})
  a_fp8 = qwix.quantize(a_bf16, jnp.float8_e4m3fn, tiled_axes={0: 1, 1: sub})
  gs = jnp.array([
      39, 44, 28, 39, 29, 33, 32, 25, 28, 30, 39, 32, 24, 26, 36, 32, 33, 32,
      32, 25, 24, 33, 40, 30, 35, 38, 24, 22, 27, 33, 30, 37, 19, 35, 38, 40,
      23, 42, 26, 29, 35, 38, 28, 34, 32, 28, 41, 28, 33, 32, 31, 22, 29, 35,
      39, 33, 26, 34, 41, 24, 28, 23, 33, 34, 30, 33, 38, 29, 22, 30, 29, 37,
      37, 42, 36, 39, 26, 35, 31, 27, 34, 30, 28, 33, 30, 37, 30, 28, 33, 22,
      32, 35, 28, 29, 32, 27, 29, 33, 34, 35, 30, 33, 30, 28, 33, 37, 29, 41,
      32, 34, 35, 34, 28, 33, 26, 29, 42, 31, 35, 32, 32, 33, 33, 38, 42, 35,
      34, 33], jnp.int32)

  print(f"shape: G={g} M={m} K={k} N={n}, out fp8 sub={out_sub}, block_k=512")
  print("  epi-dot us | block_m grid_minor grid_tile_width")
  results = []
  for block_m in (32, 64, 96, 128):
    for gm in (common.MatmulDimension.N,):
      for gtw in (1,):
        cfg = common.Config(
            block_m=block_m, block_n=128, block_k=512, num_stages=2,
            split_k=1, split_m=1, persistent=True, post_scale=False,
            collective=False, grid_minor_dim=gm, grid_tile_width=gtw,
            epilogue_quant_qtype=common.EpilogueQuantDType.FLOAT8_E4M3FN,
            epilogue_quant_subchannel_size=out_sub)
        tag = f"bm={block_m:2d} gm={gm.name} gtw={gtw}"
        try:
          f = jax.jit(lambda a, b, gs, c=cfg: epi.ragged_dot_gpu_fp8_quant_bf16_fp8_blackwell_kernel(a, b, gs, jnp.bfloat16, c, None))
          t = _bench(f, a_fp8, b, gs)
          results.append((t, tag))
          print(f"   {t*1e6:7.1f}   {tag}")
        except Exception as e:  # pylint: disable=broad-except
          print(f"   FAIL      {tag}: {str(e)[:55]}")
  results.sort()
  print("\n=== TOP 5 ===")
  for t, c in results[:5]:
    print(f"   {t*1e6:7.1f}us   {c}")
  if results:
    print(f"\nbest {results[0][0]*1e6:.1f}us vs bench default (bm=16 gm=M gtw=1) ~837us")


if __name__ == "__main__":
  main()
