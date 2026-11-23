# SPDX-License-Identifier: Apache-2.0
"""Hy4's KV: a context-parallel store of the graphs' own, and SGLang's pools that only count tokens.

The store holds, on each rank of the attention TP (``tp`` ranks), every layer's MLA rows (the 512-wide normalized
latent and the 64-wide rotated RoPE key, bf16) and every full indexer layer's index keys (128 wide, bf16) of the
rank's share of the tokens. A page of SGLang's allocator is ``page = tp * block`` tokens; rank r stores the page's
tokens ``[r * block, (r + 1) * block)`` as its local rows ``[p * block, (p + 1) * block)`` of page ``p``, so a block
table is the same on every rank and each rank reads its own rows of every page of a request. One copy of the KV across
the node: at TP64 a rank holds 1/64 of every sequence, which is what lets a single node serve contexts of millions of
tokens.

SGLang still schedules by its token allocator and request rows: its KV pool is one minimal layer, kept on the host
(``Hy4AccountingPool``), and the platform's paged allocator hands out the pages, whose ids address the store. The
store is written by the graphs only; a page is never read past its request's length, so a freed page needs no
zeroing (every row ever written is a finite value), and page 0, SGLang's reserved page, is where the tokens a rank
does not own write (models/hy4/model._write_slots)."""

from __future__ import annotations

import functools
import logging

import torch

from sglang_neuron.models.hy4.config import DTYPE, Hy4Config

logger = logging.getLogger(__name__)

_ACCOUNTING_HEAD_DIM = 8


class Hy4KVStore:
    def __init__(self, config: Hy4Config, *, tp: int, page: int, tokens: int, device):
        if page % tp:
            raise ValueError(f"--page-size {page}: Hy4's KV is context-parallel over the {tp} attention-TP ranks, so a "
                             f"page must be a whole number of tokens on each ({tp} x a block)")
        self.tp, self.page, self.block = tp, page, page // tp
        # SGLang's pages, and its reserved page 0
        self.pages = tokens // page + 1
        self.rows = self.pages * self.block
        self.latent = [torch.zeros(self.rows, config.compact_dim, dtype=DTYPE, device=device)
                       for _ in range(config.num_hidden_layers)]
        self.index = [torch.zeros(self.rows, config.index_head_dim, dtype=DTYPE, device=device)
                      for _ in config.full_layers]
        self.index_layers = tuple(config.full_layers)
        logger.info("Hy4 KV store: %d pages of %d tokens (%d a rank), %d local rows, %.2f GiB a rank", self.pages,
                    page, self.block, self.rows, self.bytes() / 2**30)

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.latent + self.index)

    @staticmethod
    def bytes_per_token(config: Hy4Config, tp: int) -> float:
        row = config.num_hidden_layers * config.compact_dim + len(config.full_layers) * config.index_head_dim
        return row * 2 / tp


@functools.lru_cache(maxsize=None)
def accounting_pool_cls():
    """SGLang's token accounting over one minimal layer, its buffers on the host: nothing reads or writes them."""
    from sglang_neuron.mem_cache.memory_pool import NeuronMHATokenToKVPoolKind

    class Hy4AccountingPool(NeuronMHATokenToKVPoolKind):
        hy4_store = None  # the graphs' KV store (configure_kv_cache), whose regions a PD transfer moves

        def _create_buffers(self) -> None:
            k_shape, shape = self.get_kv_buffer_shape()
            self.k_buffer = [torch.zeros((1, *k_shape[1:]), dtype=self.store_dtype, device="cpu")]
            self.v_buffer = [torch.zeros((1, *shape[1:]), dtype=self.store_dtype, device="cpu")]

        def get_contiguous_buf_infos(self):
            """(ptrs, lens, item_lens) of the store's regions for a PD transfer: every layer's MLA rows, then every
            full layer's index keys. Page p's rows on this rank are its local rows [p block, (p + 1) block), one run
            at ptr + p item_len, so a prefill rank's pages land in the same rank's rows of an equal-TP decode (the
            context-parallel split is the same on both sides)."""
            from sglang_neuron.backend.device_io import device_address

            store = self.hy4_store
            if store is None:
                raise RuntimeError("Hy4's PD transfer needs the KV store bound to the accounting pool")
            regions = store.latent + store.index
            ptrs = [device_address(t) for t in regions]
            lens = [t.numel() * t.element_size() for t in regions]
            return ptrs, lens, [store.block * t.shape[1] * t.element_size() for t in regions]

        def get_kv_layer_ids(self):
            """The model layer of each region get_contiguous_buf_infos lists (both PD peers pair the entries by
            them): every layer's MLA rows, then each full layer's index keys."""
            store = self.hy4_store
            return list(range(len(store.latent))) + list(store.index_layers)

    return Hy4AccountingPool


def configure_kv_cache(configurator, config: Hy4Config, *, tp: int, device):
    """SGLang's request and token pools over the accounting pool and the platform's paged allocator, and the graphs'
    KV store for the launch's --max-total-tokens."""
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigResult
    from sglang.srt.platforms import current_platform
    from sglang.srt.runtime_context import get_schedule

    schedule = get_schedule()
    page = int(schedule.page_size)
    stated = schedule.max_total_tokens
    if not stated:
        per_token = Hy4KVStore.bytes_per_token(config, tp)
        raise ValueError(f"Hy4 sizes its context-parallel KV store from --max-total-tokens ({per_token / 1024:.1f} KiB "
                         "a token on each rank); launch with it")
    tokens = int(stated) // page * page
    running = int(schedule.max_running_requests or 0)
    if not running:
        raise ValueError("Hy4: launch with --max-running-requests, the request rows and the largest decode batch")
    req_pool = configurator._build_req_to_token_pool(max_num_reqs=running)
    pool = accounting_pool_cls()(
        tokens, page_size=page, dtype=torch.bfloat16, head_num=1, head_dim=_ACCOUNTING_HEAD_DIM, layer_num=1,
        device=configurator.device, enable_memory_saver=False, start_layer=0, end_layer=0)
    allocator = current_platform.get_paged_allocator_cls()(
        tokens, page_size=page, dtype=torch.bfloat16, device=configurator.device, kvcache=pool, need_sort=True)
    store = Hy4KVStore(config, tp=tp, page=page, tokens=tokens, device=device)
    pool.hy4_store = store
    result = KVCacheConfigResult(
        max_total_num_tokens=tokens, max_running_requests=running, full_max_total_num_tokens=None,
        swa_max_total_num_tokens=None, req_to_token_pool=req_pool, token_to_kv_pool=pool,
        token_to_kv_pool_allocator=allocator, memory_pool_config=None)
    return result, store
