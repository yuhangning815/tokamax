"""Benchmark: epilogue-fused fp8 ragged dot vs the UNFUSED pipeline (new arch).

Re-architected production (a3a1e93) stores ONE accumulator of block_n N-columns
per CTA, so the fused output subchannel == block_n (= 128). Each stage is its
own jitted launch that materializes to HBM (the real pipeline; quant is its own
kernel with an fp8 roundtrip).

  unfused  = quant_kernel(bf16->fp8)  +  dot_kernel(fp8->bf16)  +  quant(bf16->fp8)
  interim  = quant_kernel(bf16->fp8)  +  epilogue_fused_dot(fp8->fp8)

Calls the kernels directly (production vs the fused-output kernel) to compare
apples-to-apple. Run on B200 with the cuda13 ptxas on PATH.
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


def _bench(f, *args, warmup=15, iters=100):
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

  def cfg(bk, block_m=16, epilogue=False):
    c = common.Config(
        block_m=block_m, block_n=128, block_k=bk, num_stages=2, split_k=1,
        split_m=1, persistent=True, post_scale=False, collective=False,
        grid_minor_dim=common.MatmulDimension.M, grid_tile_width=1)
    if epilogue:
      c = dataclasses.replace(
          c, epilogue_quant_qtype=common.EpilogueQuantDType.FLOAT8_E4M3FN,
          epilogue_quant_subchannel_size=out_sub)
    return c

  q_in = jax.jit(lambda a: qwix.quantize(a, jnp.float8_e4m3fn, tiled_axes={0: 1, 1: sub}))
  q_out = jax.jit(lambda o: qwix.quantize(o, jnp.float8_e4m3fn, tiled_axes={0: 1, 1: out_sub}))

  us = lambda t: f"{t*1e6:7.1f}us"
  print(f"shape: G={g} M={m} K={k} N={n}, int4 sub={sub}, out fp8 sub={out_sub}")
  t_qin = _bench(q_in, a_bf16)
  o_bf16 = prod.ragged_dot_gpu_fp8_quant_blackwell_kernel(a_fp8, b, gs, jnp.bfloat16, cfg(512), None)
  t_qout = _bench(q_out, o_bf16)
  print(f"  standalone input  quant (bf16->fp8 sub={sub}): {us(t_qin)}")
  print(f"  standalone output quant (bf16->fp8 sub={out_sub}): {us(t_qout)}")
  print("  block_k |  dot    | epi-dot | unfused(qin+dot+qout) | interim(qin+epi) | epi win")
  for bk in (512, 256, 128):
    dot = jax.jit(lambda a, b, gs, c=cfg(bk): prod.ragged_dot_gpu_fp8_quant_blackwell_kernel(a, b, gs, jnp.bfloat16, c, None))
    epidot = jax.jit(lambda a, b, gs, c=cfg(bk, epilogue=True): epi.ragged_dot_gpu_fp8_quant_bf16_fp8_blackwell_kernel(a, b, gs, jnp.bfloat16, c, None))
    t_d = _bench(dot, a_fp8, b, gs)
    t_e = _bench(epidot, a_fp8, b, gs)
    t_unfused = t_qin + t_d + t_qout
    t_interim = t_qin + t_e
    win = (t_unfused - t_interim) / t_unfused * 100.0
    print(f"   {bk:4d}   | {us(t_d)} | {us(t_e)} | {us(t_unfused)}          | {us(t_interim)}      | {win:4.1f}%")


if __name__ == "__main__":
  main()
