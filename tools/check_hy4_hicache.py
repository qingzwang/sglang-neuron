#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check of Hy4's HiCache host tier (models/hy4/kv.py Hy4HostPool) over a small context-parallel store.

Builds a Hy4KVStore (2 latent layers, 1 index layer by default; --device neuron puts it on a NeuronCore), fills it with
random rows, stores device pages {3, 7} into host pages {1, 4}, loads those into device pages {5, 9}, and checks every
region bit-exact (and that page 0 and the untouched pages are unchanged).

usage: python tools/check_hy4_hicache.py [--device cpu|neuron] [--tp 64] [--page 1024]
Prints one JSON line and exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tp", type=int, default=64)
    ap.add_argument("--page", type=int, default=1024)
    ap.add_argument("--pages", type=int, default=16)
    args = ap.parse_args()
    if args.device == "neuron":
        import torch_neuronx  # noqa: F401
    from check_mla_hicache import published_topology

    from sglang_neuron.models.hy4.kv import Hy4KVStore, hy4_host_pool_cls

    cfg = types.SimpleNamespace(num_hidden_layers=2, compact_dim=576, index_head_dim=128, full_layers=(1,))
    store = Hy4KVStore(cfg, tp=args.tp, page=args.page, tokens=args.pages * args.page, device=args.device)
    torch.manual_seed(0)
    ref = []
    for t in store.latent + store.index:
        r = torch.randn(t.shape, dtype=t.dtype)
        r[:store.block] = 0  # page 0: the reserved page
        t.copy_(r.to(t.device))
        ref.append(r)
    pool = types.SimpleNamespace(hy4_store=store, size=args.pages * args.page)
    with published_topology():
        host = hy4_host_pool_cls()(pool, host_to_device_ratio=2.0, host_size=0, page_size=args.page,
                                   layout="layer_first", allocator_type=None)
        P = args.page

        def idx(pages):
            return torch.cat([torch.arange(p * P, (p + 1) * P) for p in pages])

        host.backup_from_device_all_layer(None, idx([1, 4]), idx([3, 7]), None)
        host.engine.wait(host.engine._seq) if hasattr(host.engine, "wait") else None
        host.load_to_device_per_layer(None, idx([1, 4]), idx([5, 9]), 0, None)
        host.engine.wait(host.engine._seq) if hasattr(host.engine, "wait") else None
    bad = 0
    for t, r in zip(store.latent + store.index, ref):
        got = t.to("cpu").view(store.pages, store.block, -1)
        r = r.view(store.pages, store.block, -1)
        want = r.clone()
        want[5], want[9] = r[3], r[7]
        bad += int(not torch.equal(got.view(torch.int16), want.view(torch.int16)))
    rec = dict(check="hy4_hicache", device=args.device, regions=len(ref), host_pages=host.page_num,
               size_per_token=host.size_per_token, bad_regions=bad)
    print(json.dumps(rec), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
