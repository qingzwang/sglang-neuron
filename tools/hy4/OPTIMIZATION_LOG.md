# Hy4-FP8 on Trn2 (sglang-neuron): optimization log

Model: tencent/Hy4-FP8 (78 layers, MLA + DSA indexer, iHC hc_mult 4, 256 experts top-8, block-FP8).
Hardware: trn2.48xlarge, TP64 EP64, LNC2 (64 logical cores). Reference: 8x B200, official SGLang (lmsysorg/sglang:hy4).
Code: `/mnt/nvme/sglang-neuron`, branch `hy4-trn2` (commits below). Work dir: `/mnt/nvme/work/hy4`.

## Results over time (full model, same client: `b200prof/client.py`, `tests/itl.py`)

| stage (commit) | per 1K prefill chunk @32K / 128K / 512K | cold 400K | decode ITL | AgentX conc1 |
|---|---|---|---|---|
| bring-up, bucket bug fixed | 1.69 / 3.1 / 6.2 s | ~40 min | ~89 ms | 37 req/h, TTFT p50 13.1 s, ITL 84 ms |
| compile fixes (indexer einsum, layer-0 HC) | 1.21 / 1.70 / 2.43 s | 860 s | ~90 ms | |
| NKI hc / mla_dense / index_scores / moe_prefill (`f208967`) | 0.76 / 1.04 / 2.10 s | 701 s | ~90 ms | |
| mla_sparse + 512K bucket, decode MoE v2, fewer collectives (`13c392d`) | ~1.1 / 1.2 / 2.4 s (+1K TTFT) | 403 s | 77 ms | |
| column-tiled MoE + hwdge, decode compaction (`fa86dd9`) | same | 403 s | 54 ms | 71 req/h, TTFT p50 2.6 s, ITL 55 ms |
| decode HC post+pre kernel, batched expert DMAs (`46c1c53`) | same | 403 s | 49.4 ms (itl.py @32K) | |
| **B200** | 0.21 / 0.16 / 0.26 s | 39 s | 10.5 ms | 177 req/h, TTFT p50 0.89 s, ITL 10.4 ms |

## Steps, measurements and findings

1. **Prefill context buckets broke long context** (needles failed at 36K+). `keep_tensor_guards_unsafe` dropped the
   prologue's int `blocks` guard: every bucket reran the first bucket's program. Fix: tensor-only guards for the group
   program only, compile every bucket on the first prefill of a shape (`runtime.py`).
2. **First neuron-explorer profile** (`prof_nx/prologue256k_*`, `group256k_*`): prologue 857 ms with MFU 1.25%:
   the indexer's `einsum("thk,th->tk")` lowered to a batched matmul fed by layout DMAs (Scalar 550 ms + Sync 237 ms);
   layer 0's HC mixes over the broadcast embedding became a vector reduction (167 ms). Group: 44 GB spill traffic.
   Fixes: multiply+sum; replicated-stream HC path. Prologue 695 -> 78 ms, group 218 -> 131 ms (6-layer, 128K).
3. **NKI kernels** (`sglang_neuron/models/hy4/kernels/`), each unit-tested vs torch then in the served model:
   - `index_scores.py` (indexer scores in SBUF), `mla_dense.py` (flash attention, two-pass softmax <= 2048 keys),
     `hc.py` (prefill HC pre/post, streams read once), `moe_prefill.py` (routed 128-token blocks, FP8 dequant in SBUF,
     `dma_compute` scatter-add), `mla_sparse.py` (per-row-block gather of selected rows), `moe_decode2.py` (decode
     experts; `moe_decode_fp8_ct` column-tiled), `decode_compact.py` (nonzero_with_count compaction), `hc_decode.py`.
   - 128K group HBM traffic 24 -> 11.7 GB; compute-idle-waiting-on-DMA 23 -> 10.8 ms (`reports/g128k_nki3_dma.txt`,
     `reports/g128k_all1.txt`).
4. **B200 profile** (`b200prof/trace/`, torch profiler at 392K + 8K prefill): indexer FP8 logits 27%, MoE 16%,
   sparse attention 16%, allreduce 11%, iHC Triton kernels 14%, top-k 8%.
5. **Decode ablations** (`logs/ablate_full.out`, full model, ITL @32K, baseline 51.2 ms): routed experts 13.0 ms,
   HC 6.4, attention core 5.0, all-gather 3.6, all-reduce 3.5, reduce-scatter 1.3, remainder ~18.

## Pitfalls found (also in Claude memory: nki-in-xla-graph-pitfalls, dynamo-tensor-guards-int-args)

- A kernel output whose layout the consumer collective wants differently (token-major into a reduce-scatter over
  heads) -> XLA relayout with element-sized DMAs, 133M packets, 3.7 s a group. Emit the consumer's layout.
- PSUM: explicit `accumulate=False` into one region of a bank broke other accumulations sharing the bank; use
  `accumulate=None` (compiler-inferred) when several regions share a bank (`kbench/accum*.py`).
- Dynamic-offset (scalar_offset) DMAs default to the GpSimd software DGE (~0.7 us a DMA): use
  `dge_mode=hwdge, engine=sync/scalar` for weight loads (decode MoE 0.18 -> 0.09 ms an expert).
- Tensor-engine column tiling (`tile_position/tile_size`, stationary <= 32 columns): 4 matmuls at once, 3.3x the rate
  (`kbench/coltile.py`).
- Summed instruction durations overlap (pipelined): judge by the span (first start .. last end of one execution on a
  core). A decode GEMV in NKI was no faster than XLA (both weight-DMA bound, `kcap_gemv_*`).
- Short decode kernels are bound by fixed latencies (program start ~4 us, activation-table loads ~1.3 us each,
  dependent DMA chains) -> the case for a decode giga kernel.
- Sharding q_a / the router over the ranks slowed prefill 17% (all-gather layout) and did not help decode: reverted.
- Requantizing the 128x128 block scales to per-column-block adds ~2.6% output error (about the FP8 error itself):
  not done (`requant/err.py`).

## Artifacts

- `prof_nx/<name>.neff` + `<name>_rank_0.ntff` (raw), `<name>_pq/` (parquet), `reports/<name>.txt` (text summary).
  Names: `prologue256k`/`group256k` (first profile, pre-fix); `g128k_fix1` (after compile fixes); `g32k_nki`
  (the 3.7 s relayout bug); `g128k_nki3` (+`_dpq` with DMA trace: head-major attention); `g128k_all1` (all prefill
  kernels); `g512k_sp1` (sparse 512K); `dec1`/`dec3`/`dec4` (decode group before/after MoE v3 / compaction);
  `kcap_*` (single kernels with real inputs).
- Tools: `prof_nx/prof2.sh <neff> <name>` (64-rank capture + parquet + reports), `prof_nx/kcap.sh <name> <test>`
  (one kernel with its test's real inputs), `dma_overlap2.py`, `agg3.py`, `hbm_by_src.py`.
- Benchmarks: `agentx-results-trn2/` (run1_baseline = before kernels; top level = after `fa86dd9`),
  `agentx-results-b200/`, `b200prof/*.json` (client timings), `logs/` (server logs, ablations).

## Next: decode giga kernel

Step 1 (done, off by default, `HY4_NKI_MOE_BLOCK=1`): `kernels/moe_block.py` — router, top-8, renormalization, local affinities, routed experts
(column-tiled, static per-expert 0/1-trip loops) and the shared expert in one kernel. Step 2: the attention block.
Step 3: whole layers (and HC) in one kernel with in-kernel collectives (`nki.collectives`) and weight prefetch.

### Giga kernel step 1 result (2026-11-22)

- Device spans (kernel-only, real inputs, `prof_nx/kcap_mblk*`): no hit 101 us vs 114 us for the current path
  (XLA routing + `moe_decode_fp8_ct` + XLA shared MLP, `kcap_mblkx_*`); two hits 290 vs 300 us.
- First version was slower (305 / 127 us): four 0/1-trip device loops cost ~9 us each even with 0 trips, and the
  shared expert ran serially after the experts. Fixed with one loop over a `nonzero_with_count` hit list and the
  shared expert fused into the router's row-block loop.
- Full model ITL 49.1 vs 49.4 ms: within noise. Fusing one block saves only its small ops; the per-layer cost left is
  the expert weights' DMA (~85 us an expert, ~1.4 hit experts on the slowest rank a layer), each kernel's start/finish
  (~10-20 us) and the collectives outside. Large gains need cross-layer overlap (prefetching the next layer's
  weights during this layer's MoE / collectives), i.e. a multi-layer kernel.

## Speculative decoding (DFly2), started 2026-11-22

- The checkpoint bundles a 5-layer DFly2 drafter (`model.dfly_layers.*`, 3.3B): block-parallel (bonus anchor + 6 mask
  queries, `dfly_block_size` 7), each draft layer sharing the KV cache and indexer top-2048 of target layer
  61/65/69/73/77, a sequential hidden-correction chain over the target LM head proposes the drafts.
- Reference implementation copied from `lmsysorg/sglang:hy4` into `ref_dfly/` (models/hunyuan_v4_dfly.py,
  configs/hy_v4_dfly.py, speculative/dfly_worker_v2.py, dfly_utils.py, kernels/ops/{attention,speculative}/dfly.py).
- B200 with `--speculative-algorithm DFLY` (`b200prof/dfly_accept_b200.log`): accept length ~2.2-4.3 a step (mostly
  ~2.8; 2.2 overall incl. T=0.9), bs1 generation 130-240 tok/s vs ~95 without.
- Trn2 estimate: verify of 7 rows ~60 ms + draft ~5 ms per step, ~2.8 tokens a step -> ~23 ms effective ITL (vs 49).
- Plan: verify program on the decode graph (7 rows a request), draft program (5 layers + correction chain), device
  acceptance (greedy; speculative-sampling rule for T > 0), BaseSpecWorker via NeuronSpeculativeWorker (pattern of
  Kimi-K3 DSPARK), then AgentX on both trn2 and B200 with DFLY for a fair comparison.

### DFly on Trn2: implementation and results (2026-11-22)

- Code: `models/hy4/dfly.py` (drafter + DFlyStep program), `dfly_worker.py` (SGLang DFLASH spec-v2 worker),
  `runtime.py` (decode buckets in rows = Q x requests, the draft program after the verify in the same call),
  `weights.py` (Fp8Slice: the drafter's block-FP8 projections through the target's bf16 loaders), platform hooks.
  Launch: `run_dfly.sh` (`--speculative-algorithm DFLASH --speculative-draft-model-path <model>
  --speculative-num-draft-tokens 7 --disable-overlap-schedule`).
- Verify = the unchanged decode graph over 7 rows a request (write-before-attend + key_pos <= row pos makes the
  block causal); each row sampled by the device sampler, a draft accepted while it equals the sample of the row
  before (exact speculative sampling for a greedy draft). The drafter reads the producer layers' compacted
  selections straight from the group programs' `state` and the verify's final streams; one readback a step.
- Correctness: drafts off = plain decode exactly (6/6 prompts); with drafts 3/6 identical over 200 tokens, 3 split
  at near-ties. Not DFly: 7 identical requests in a plain 7-row decode split at the same points (the decode graph's
  numerics depend on a token's row; pre-existing, `tests/rowpos_check.py`).
- Step (full model, bs1, 128K): prologue 5.6 + 19 groups 71 + head 2.8 + draft 6.5 = ~92 ms; accept length 1.9-6.7
  (mean ~3.2 on the test prompts). 7-row verify ablations: routed experts 22.1 ms (1-row 13.0), attention 11.2
  (5.0), HC 9.5 (6.4). A token-batched HC kernel (`hc_decode_batched`, 75 -> 57 us at 7 rows) saved ~1 ms in the
  graph; the sparse prefill kernel for multi-row decode attention saved nothing (`HY4_NKI_SPARSE_DECODE`, off).

| AgentX conc1 | req | out tok/s | TTFT p50 / p90 s | ITL p50 ms | E2E p50 s |
|---|---|---|---|---|---|
| trn2 | 71 | 14.6 | 2.61 / 5.69 | 55.1 | 17.9 |
| trn2 + DFly | 95 | 21.7 | 3.16 / 14.09 | 38.0 | 17.1 |
| B200 | 177 | 74.1 | 0.89 / 1.85 | 10.4 | 8.6 |
| B200 + DFLY | 232 | 115.8 | 0.95 / 1.76 | 5.8 | 6.2 |

DFly: trn2 x1.49 output throughput (ITL x1.45), B200 x1.56 (ITL x1.8). Gap with both speculative: 5.3x throughput,
6.6x ITL. The B200's verify of 7 rows costs about what one row does; trn2's costs ~1.8x (expert weight DMA,
serialized collectives) -> the multi-layer giga kernel (prefetching the next layer's experts) is the lever.

### Giga kernel groundwork (2026-11-22): real-input profile and per-op kernels

- Real-input profile of the 7-row verify group (`prof_nx/vg7r`: a live server's group call dumped with
  HY4_DUMP_GROUP, matched to the NEFF by byte size, `prof_nx/match_dump.py`): span ~2.8 ms. A shared layer ~530 us:
  ~215 us MoE all-reduce -> query all-gather (HC post + pre, norm, q_a / kv_a / q_b), ~120 us attention with its 4
  collectives, ~190 us o_proj all-reduce -> MoE all-reduce; full layer ~1 ms; collectives ~113 us a layer, serial.
  q_a (25 MB) 105 us and kv_a (7 MB) 75 us of PE time in XLA (PE bound at T = 7).
- Host is not the bottleneck: a decode step's 21 calls are issued in ~20.5 ms of a ~52 ms period
  (HY4_DEBUG_LAUNCH=1); fences ~0.87 ms each are host-side.
- Measured limits: HBM->SBUF ~320 GB/s a physical core sustained with 2 KB packets (rows < 2 KB split into ~650 B
  packets, ~half rate; `kbench/dma_bw.py`); PE GEMV with 512-column moving tiles 720 GB/s fp8 / 1.5 TB/s bf16 a
  program vs 274 GB/s for 128-column tiles with rotating stationaries (`kbench/gemv_rate.py`).
- `kernels/moe_wide.py` (4 pre-scaled token copies a column tile, 512-column moving, PSUM accumulation, diagonal
  extraction; exact): 213 vs 224 us for 2 experts: the per-expert startup and the loop's lack of cross-expert
  prefetch remain. Not wired in.
- `kernels/gemv_wide.py` for q_a + kv_a (exact): standalone 94 vs XLA 88 us; in the served verify 72.2 vs 70.2 ms
  for the 19 groups (slower; `HY4_NKI_PROJ`, off).
- Conclusion: like the MoE block, HC-batched and sparse-decode kernels before it, replacing single ops with kernels
  does not shorten the step: each kernel is an ordering boundary in the XLA schedule, and the step is a serial chain
  of ~15 dependent stages a layer with ~110 us of collectives. Only fusing whole layers (collectives in-kernel, the
  next stage's weights streaming during the current stage's collectives) removes that chain.

### Decode fusion kernels (2026-11-22/23), all behind flags (default off)

7-row DFly verify, 19 groups (`ab_dfly.sh`, HY4_DEBUG_TIMING):

| config | groups ms |
|---|---|
| baseline | 70.1 |
| + HY4_NKI_MOE_SUB=1 (`kernels/moe_sublayer.py`: norm, routing, shared + routed experts, in-kernel all-reduce) | 66.5 |
| + HY4_HC_CHAIN=1 (layer-boundary HC post + pre in hc_decode) | 65.8 |
| + HY4_NKI_ATTN_IN=1 (`kernels/attn_in.py`: norm, q_a/kv_a 512-wide GEMV, latent norms, q_b, RoPE, cache write, query all-gather) | **64.8** |
| + HY4_NKI_ATTN_CORE=1 (`kernels/attn_core.py`: sparse attention + merge + output, 3 in-kernel collectives) | 65.6 |
| + HY4_NKI_LAYER=1 (`kernels/layer_decode.py`: a shared layer's 5 stages in one trace) | 65.1 |

- Best: MOE_SUB + HC_CHAIN + ATTN_IN (-7.6%), accept rates unchanged. In-kernel 64-rank collectives work
  (all-reduce, all-gather, reduce-scatter); a kernel can write the KV cache in place (`wrap_nki(...,
  alias_outputs_in_place=True)`, `kbench/alias_test.py`).
- Group span (real inputs, `prof_nx/vg7d`, `vg7l`): 3.12 -> 2.72 ms; shared layer ~527 -> ~450 us. Idle (no DMA, no
  compute) ~28%: ~70% of it inside collectives (Mesh, latency bound: AR 86 KB ~23 us, AG ~8-10 us, RS 1 MB fp32
  ~28 us; ~92 us a shared layer), ~17% semaphore waits, ~12% tiny gaps (`prof_nx/idle.py`).
- DMA packets: 54% of the bytes in 256 B-1 KB packets (hardware DGE mean 816 B); rows < 2 KB split.
- Experts in moe_sublayer stream at ~290 GB/s a physical core (peak ~320): at the DMA wall. The waste is the
  windows without DMA: collectives, routing (~25-30 us), MoE tail (~20 us), attention chain (~30-40 us). Bandwidth
  floor of the 19 groups at 7 rows ~21 ms vs 65 now.
- Putting stages in one trace (layer_decode) does not overlap them: NKI keeps program order; prefetch must be
  explicit (next: a group kernel prefetching the next stage's known weights during collectives).

### TP-16 x EP-4 decode experts (2026-11-23), HY4_MOE_TP=16 (needs HY4_NKI_MOE_SUB / HY4_NKI_LAYER)

- `kernels/moe_tp.py`: each EP group of 16 ranks holds 64 experts; each rank 1/16 of each (128 gate/up columns, 128
  down rows). A hit expert is 2.4 MB on each of 16 ranks instead of 37.7 MB on one: removes the rank expert-load
  imbalance (slowest rank ~3 experts vs ~0.9 mean at 7 rows). Hits processed 8 slots per device-loop chunk,
  hardware-DGE DMAs at scalar offsets, out-of-range slots skipped.
- moe_sublayer TP mode: group-local routing (iota one-hot, 3 instructions), hit list padded with E, chunks = ceil/8.
  Standalone vs torch: rel err 0.005-0.006 for group bases 0/64/128/192, 1-42 hits (`tests/moe_sub_tp_test.py`).
- Loaders (`weights.experts_tp_loaders`) bit-identical to `tp_layouts`.
- Truncated 22 layers, DFly 7-row verify (ab_tb0/tb1): groups 19.2 -> **17.5 ms (-9%)**. Outputs (garbage model)
  agree 5-30 tokens then drift (rounding order).
- **Full model** (HY4_MOE_TP=16 replaces the EP layout; prefill on `kernels/moe_prefill_tp.py`), ab_ftp1 vs ab_ain3:
  decode groups 64.8 -> **57.8 ms (-10.8%)**; accept lengths unchanged (2.42/4.0/2.29/6.74/2.42/6.4 vs
  2.17/4.0/2.29/6.74/2.67/6.4), outputs coherent (2 of 6 identical, the rest diverge mid-text).
  Prefill groups a 1024-row chunk 619 -> 713 ms (+15%): moe_prefill_tp is PE / vector bound (48 token transposes +
  12K-element dequant a 128-token block, 64 blocks a rank; standalone 4.8 vs 3.4 ms EP).
- moe_prefill_tp tuning (standalone, 1024 rows, one core's device span; `kbench/moe_prefill_tp_v4_pairs.py`):
  v1 (per-block gather + 48 transposes, committed) 1985 us; scaled-token gate/up (2x matmuls) 2096; dense
  transpose once + 256 B column reads 2078; (tile, expert) pairs 2428 -> 2036 (2 pairs/iteration) -> 2150
  (metadata on hwdge, one DMA per tensor, double-buffered across iterations). EP kernel (4 experts): wall 3.4 vs
  4.8 ms. Floor: ~29 us PE + ~22 us vector (dequant) a 128-slot pair; TP repeats each token's handling on 16 ranks.
  Kept v1.
- With TP experts (full model, groups ms / accept lengths): ftp1 (MOE_SUB+HC_CHAIN+ATTN_IN+TP) 57.8 /
  2.41 4.0 2.28 6.73 2.41 6.4; + ATTN_CORE 58.4; + ATTN_CORE + LAYER 56.9 but **wrong** (counting prompt
  "18, 18, 20 .. 22, 22", accept 5.56; the pre-TP layer1 run too: 4.57): layer_decode had no barrier on the KV cache
  between attn_in's in-place row writes and attn_core's indirect key gather. Fixed (core_barrier(cache)):
  **57.3** / 1.88 3.87 2.32 6.73 2.37 6.73, counting output correct.
- Group profile (prof_nx/vprof.sh vg8, real inputs, best flags): group span 3041 us = shared layers ~520 us each,
  full layer ~930 us of which the XLA indexer ~440 us (projections ~70, key gather + scores ~90, 2 bisection rounds
  of 256 bins ~100 each). Program size: HY4_GROUP=8 (42-layer model, 5 groups of 8 vs 10 of 4) 31.7 vs 31.4 ms:
  launches cost nothing measurable.
- Decode selection bisection (full model, best flags): 256 bins x 2 rounds 57.3 ms -> 64 x 3 54.8 -> **16 x 4 54.5**
  (same resolution as 256 x 2), accept lengths and outputs identical. Default now 16 x 4 (HY4_DECODE_SELECT).
- HY4_QKV_SH=1 (attn_in: q_a / kv_a sharded over the 64 ranks, 41 columns + the gate a rank packed partition-major,
  one all-gather of [T, 41] slices; 0.5 MB of weights a layer instead of 32 MB): groups 54.5 -> **53.4 ms**, accepts
  2.61 3.87 2.37 6.73 2.50 6.73, counting output correct (texts diverge mid-way: fp32 order).
- Served step without the timing syncs (ab_wall.sh, HY4_DFLY_PROFILE=1): prep 0.7, launch (async) 22.6, device wait
  38 ms a verify step (~61-64 ms device-bound); the scheduler's time between decode steps is the next request's
  prefill (~1.3 s a 1024-row chunk), not host overhead. AgentX conc1 (last run): ISL avg 195K (97% cached), OSL 820,
  TTFT 6.2 s, ITL 41.9 ms: decode ~80% of request latency, at 4x the A/B's context (8192 keys a rank).
- 512K context capacity (AgentX's CTX; 8192 key slots a rank): best flags groups 61.8 ms (vs 53.4 at 128K).
  HY4_NKI_INDEX_DECODE=1 (kernels/index_scores.py index_scores_decode: a request's 7 verify rows share one key
  gather; 2 query blocks of 128 (row, head) a 512-key tile, relu + head-weighted sum in-core): groups 60.0 ms,
  prologue unchanged (6.2; the kernel is kept out of layer 0, whose program got slower with it: ~10 ms), accepts
  identical.
- 512K, on top: index_scores_decode transposes the gathered keys on the PE (no XLA transpose) 60.0 -> 59.9;
  HY4_IDXQ_SH=1 (the decode indexer's idx_q sharded, 64 columns a rank + an all-gather, instead of the replicated
  16 MB a full layer) -> **56.9 ms**, accepts identical.

### AgentX conc1, best decode flags (2026-11-23, agentx-results-trn2/dfly_v2)
MOE_SUB HC_CHAIN ATTN_IN MOE_TP=16 ATTN_CORE LAYER QKV_SH NKI_INDEX_DECODE IDXQ_SH; vs dfly (v1, 2026-11-22):
output 21.5 -> **24.7 tok/s**, ITL avg 41.9 -> 33.5 ms (p50 38.0 -> 28.2), active decode 28.9 -> 36.3 tok/s, request
latency 42.8 -> 39.5 s, TTFT 6.2 -> **8.0 s** (worse: the TP-layout prefill). 105 requests, 0 errors. B200 DFLY: 115.8.
- DFly draft layers' attention + output through attn_core (HY4_NKI_ATTN_CORE): draft 6.6 -> **5.8 ms** at 512K,
  accepts 2.61 3.87 2.78 6.73 2.56 6.4. The first try padded the draft's 263 keys a row to 320: mla_sparse needs
  2 x keys a multiple of 512 and silently computed garbage (accept 6.73 -> 3.12; the target's own output stayed
  correct, the verify catching it); padded to 512 now, and mla_sparse asserts the width.
- Prefill chunk size (fresh 15.6K-token prompts, best flags, 512K capacity; ttft_ab.sh): CHUNK=1024 13.4 s
  (~1160 tok/s); 2048 (KV pool cut to 1.5M tokens to fit) 14.6 s; 4096 OOM (NRT_RESOURCE). Prefill is compute bound:
  bigger chunks do not amortize anything.
- Prefill group profile (prof_nx/pf1, a 1024-row chunk at ~10K context, real inputs): 37.5 ms a group = XLA 8.6,
  moe_blocks' XLA (argsort [T, 64] 2.0 + two [NB, 128] gathers 3.4) 5.4, mla_dense 5.1, hc 4.0, moe_prefill_tp 2.7.
  With 64 local experts (TP) moe_blocks' static bound T E / 128 + E = 576 blocks; a token routes to at most 8:
  bound T min(k, E) / 128 + E = 128. TTFT of a fresh 15.6K prompt 13.4 -> **12.0 s**.
- moe_blocks orders each expert's routed tokens with decode_compact (GpSimd nonzero_with_count) instead of an XLA
  argsort over [T, E]: TTFT 12.0 -> **11.5 s** (fresh 15.6K prompt; 13.4 before both changes), decode outputs
  bit-identical to the argsort version.
- decode moe_blocks fast path (T==128): CPU-equal, but device OOB gather at bucket 14 warmup -> dropped
- 1M context (CTX=1048576, buckets ...,1048576): needles 120K/475K/902K tokens all found (902K prefill 1311 s)
- HY4_MERGE_BF16: TTFT 11.23 -> 10.80 s (15.6K), outputs identical, needles 114K/285K ok
- HY4_KV_FP8: needles 57K/114K/285K/475K ok; 3/6 greedy identical; decode same speed
- dense per-expert MoE blocks at 112 rows: moe_prefill_tp 0.9 -> 9.3 ms/group (scatter-add of all 128 rows x H a block): dropped

## 2026-11-24: decode at concurrency, GLM-5.2 giga kernel port

- HY4_TOPK_SELECT (GpSimd nisa.topk local top-256, all-gather, global 2048th threshold): 7-row 19 groups 53.1 -> 51.5 ms
  (timing mode); 4/6 greedy identical. Commit db05505.
- 112-row (16 req x 7) baseline: groups 323 ms/step (timing), device group 12.3 ms = hc_decode 2.8, XLA 3.7, MoE blocks
  1.7 + TP 0.7, mla_sparse 1.5 (prof_nx/r112p). GLM-5.2 1P2D: 128 rows ~65 ms/round.
- attn_in / attn_core multi-row (<= 128 rows; PSUM bank split, 16-token stat chunks): exact at 7/28/112; XLA 3.7 -> 2.4 ms
  but the kernels add 1.1 ms (prof_nx/r112q). HY4_HC_DEC_ROWS=16 (XLA HC past 16 rows): group 12.3 -> 10.9 ms.
  hc_decode_batched in-kernel 8-row chunks (SBUF). Commits 27f8da8, 128b1d7.
- MoE block list on chip (moe_decode_tp, every local expert a block): 0.9 -> 8.2 ms/group when routing is narrow;
  dense per-expert blocks 9.3 ms (scatter-add of 128 rows x H a block). Both off.
- GLM52 runtime knobs NEURON_RT_DBG_RDH_DOUBLE_BUF / DISABLE_EXECUTION_BARRIER: ITL 66.15 -> 65.91 ms, not kept.
- AgentX PD v2 (Mooncake): c1 24.65 tok/s TPOT p50 27.3 ms; c4 38.8 tok/s TPOT p50 78.3 ms (results_agentx/pd_v2).
- GLM-5.2 giga kernel port (decode, one node: 64 ranks LNC2 CP4, 16 seqs x 8 rows): Kaena nkilib test harness runs on the
  dev node (/mnt/nvme/work/hy4gk/run_test.sh). The 64-rank CP4 attention layer test passes as is; with Hy4's learned
  sinks (extra term in the CP combine) and the sigmoid output gate (normed rows all-gathered SBUF->SBUF over the
  o_proj group) it passes (cos >= 0.993); without either it fails (cos 0.95 / 0.83). Patch
  /mnt/nvme/work/hy4gk/hy4_attention.patch. The vLLM snapshot's trn2 attention does not match the Kaena reference
  (diverged): the port takes Kaena's attention + the snapshot's one-node EP64 MoE (block-scaled FP8; per-channel
  requant measured 2.65% weight error, not used).
