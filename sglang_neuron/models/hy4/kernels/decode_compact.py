# SPDX-License-Identifier: Apache-2.0
"""The decode's compaction of a full layer's selection in one NKI kernel: each row's selected local keys, as cache rows.

Hy4Graph._compact took each row's ``cap`` highest-scoring selected keys with torch.topk over the rank's keys (8208 at
a 512K context): on one partition a row, 32 rounds of max8 / find_index8 over the keys, ~1.7 ms a full layer. A row
selects ~index_topk / ranks (32) keys on a rank, far below ``cap`` (256), and the attention does not depend on the
order of its keys, so the selection itself is the compaction: the GpSimd engine's nonzero_with_count lists a row's
selected positions (eight rows at once, one on partition 16 j of each GpSimd core), and nc_n_gather reads their cache
rows. A row with more than ``cap`` selected keys keeps its first ``cap`` by position (the XLA form kept the
highest-scoring; past cap per rank the selection is heavily concentrated on one rank's keys, which the context
parallel layout makes rare)."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
GROUP = 16  # partitions a GpSimd core reads its 0th of
ROWS = P // GROUP  # rows an instruction compacts


def decode_compact(sel, key_rows, cap: int = 256, pad_row: int = 0):
    """sel [B, K] fp32 (1 on a selected key); key_rows [B, K] int32 (each key's cache row) -> (rows [B, cap] int32,
    keep [B, cap] fp32), a pad slot's row ``pad_row``."""
    B, K = sel.shape
    rows_out = nl.ndarray((B, cap), dtype=nl.int32, buffer=nl.shared_hbm, name="compact_rows")
    keep_out = nl.ndarray((B, cap), dtype=nl.float32, buffer=nl.shared_hbm, name="compact_keep")
    for b0 in range(0, B, ROWS):
        nb = min(ROWS, B - b0)
        s = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
        kr = nl.ndarray((P, K + 1), dtype=nl.int32, buffer=nl.sbuf)
        # the pad slot's row, after the row's keys (the pad index K gathers it)
        nisa.memset(dst=kr[0:P, K:K + 1], value=pad_row)
        for j in range(nb):
            nisa.dma_copy(dst=s[j * GROUP:j * GROUP + 1, 0:K], src=sel.ap(pattern=[[K, 1], [1, K]], offset=(b0 + j) * K))
            nisa.dma_copy(dst=kr[j * GROUP:j * GROUP + 1, 0:K],
                          src=key_rows.ap(pattern=[[K, 1], [1, K]], offset=(b0 + j) * K))
        nz = nl.ndarray((P, K + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.nonzero_with_count(dst=nz[0:P, 0:K + 1], src=s[0:P, 0:K], index_offset=0, padding_val=K)
        idx = nl.ndarray((P, cap), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idx[0:P, 0:cap], src=nz[0:P, 0:cap], engine=nisa.gpsimd_engine)
        g = nl.ndarray((P, cap), dtype=nl.int32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=g[0:P, 0:cap], data=kr[0:P, 0:K + 1], indices=idx[0:P, 0:cap])
        kp = nl.ndarray((P, cap), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=kp[0:P, 0:cap], data=nz[0:P, 0:cap], op0=nl.less, operand0=K)
        for j in range(nb):
            nisa.dma_copy(dst=rows_out.ap(pattern=[[cap, 1], [1, cap]], offset=(b0 + j) * cap),
                          src=g[j * GROUP:j * GROUP + 1, 0:cap])
            nisa.dma_copy(dst=keep_out.ap(pattern=[[cap, 1], [1, cap]], offset=(b0 + j) * cap),
                          src=kp[j * GROUP:j * GROUP + 1, 0:cap])
    return rows_out, keep_out
