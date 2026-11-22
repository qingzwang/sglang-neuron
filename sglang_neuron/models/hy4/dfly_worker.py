# SPDX-License-Identifier: Apache-2.0
"""Hy4's DFly speculative worker: SGLang's spec-v2 worker (``--speculative-algorithm DFLASH``, whose scheduling
reserves a fixed verify block of ``Q`` = ``--speculative-num-draft-tokens`` positions a request, twice over) over the
Hy4 decode programs and the DFly step (dfly.py), built by the platform's speculative-worker seam
(model_executor.spec_worker) from the host's ``speculative_worker_cls``.

- An extend batch is the target's prefill as it is (the drafter needs none of it).
- A decode batch is one call of the decode runtime over ``Q`` rows a request, built here on the host: each request
  verifies its bonus token (its last output, at its committed length ``L``) and the drafts the last step drafted for
  it, or no draft (-1) when it has none (its first decode step, or a step it missed); the call ends in the DFly step,
  which accepts, takes the new bonus and drafts the next block. One read of the step's tokens, accepted counts and
  drafts: SGLang commits ``accepted + 1`` tokens of each request (the target's tokens of its rows ``0 .. accepted``),
  and the drafts wait here, keyed by request and its new committed length.

Every step's rows are sampled by the decode runtime's device sampler (greedy or with the request's temperature); a
draft is accepted while it equals the target's sample of the row before, which for the greedy drafter is the
speculative-sampling rule (accepting draft d with probability p(d), and otherwise sampling from p without d).

The scheduling runs synchronously (``--disable-overlap-schedule``): a step's rows depend on the last step's accepted
counts."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# debug: verify no draft (each step commits one token), for comparing a DFly server's outputs with a plain decode's
_NO_DRAFTS = __import__("os").environ.get("HY4_DFLY_NO_DRAFTS") == "1"


def worker_reasons(server_args, model) -> list:
    reasons = []
    graph = getattr(model, "model_graphs", {}).get("decode")
    d = getattr(graph, "dfly", None)
    if d is None:
        return ["the checkpoint has no DFly drafter (dfly_num_hidden_layers)"]
    q = int(getattr(server_args, "speculative_num_draft_tokens", 0) or 0)
    if q != d.dfly.block:
        reasons.append(f"--speculative-num-draft-tokens {q}: the drafter's block is {d.dfly.block}")
    if not bool(getattr(server_args, "disable_overlap_schedule", False)):
        reasons.append("the overlap schedule: a DFly step's rows need the last step's accepted counts; launch with "
                       "--disable-overlap-schedule")
    if bool(getattr(server_args, "enable_mixed_chunk", False)):
        reasons.append("--enable-mixed-chunk")
    return reasons


class _Sampling:
    """The attribute the decode runtime's sampling kwargs read of a forward batch."""

    def __init__(self, sampling_info):
        self.sampling_info = sampling_info


def _worker_class():
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

    from sglang_neuron.attention.backend import NeuronForwardMetadata

    class Hy4DFlyWorker(BaseSpecWorker):
        def __init__(self, server_args, gpu_id: int, nccl_port: int, target_worker, **kwargs):
            super().__init__()
            self._target_worker = target_worker
            self._draft_worker = None  # the drafter is the decode runtime's DFly step
            self.server_args = server_args
            self.model = target_worker.model_runner.model
            reasons = worker_reasons(server_args, self.model)
            if reasons:
                raise ValueError("the Hy4 DFly worker cannot serve this launch:\n  - " + "\n  - ".join(reasons))
            self.q = int(server_args.speculative_num_draft_tokens)
            self.speculative_num_steps, self.speculative_num_draft_tokens, self.topk = 1, self.q, 1
            self.drafts = {}  # rid -> (the committed length the drafts follow, [Q - 1] drafts)
            self.steps = 0
            self.accepted = 0
            self.verified = 0
            self.t_last = time.perf_counter()
            logger.info("Hy4 DFly worker: Q = %d rows a request", self.q)

        @property
        def target_worker(self):
            return self._target_worker

        @property
        def draft_worker(self):
            return None

        def init_cuda_graphs(self):
            return None

        def init_attention_backends(self):
            return None

        def clear_cache_pool(self):
            self.drafts.clear()

        def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
            self.drafts.pop(rid, None)

        def block_accept_estimate_log_suffix(self) -> Optional[str]:
            return None

        def dump_info_records(self) -> Optional[dict]:
            return None

        def clear_info_records(self) -> None:
            return None

        def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None,
                                     **kwargs):
            if pp_proxy_tensors is not None:
                raise ValueError("the Hy4 DFly worker runs no pipeline stage")
            mode = batch.forward_mode
            if mode.is_extend() or getattr(batch, "is_extend_in_batch", False):
                result = self._target_worker.forward_batch_generation(batch, **kwargs)
                result.new_seq_lens = batch.seq_lens
                n = len(batch.reqs)
                result.next_draft_input = make_draft_input_v2(bonus_tokens=torch.zeros(n, dtype=torch.int64),
                                                              new_seq_lens=torch.as_tensor(batch.seq_lens).to("cpu"))
                for req in batch.reqs:
                    self.drafts.pop(req.rid, None)
            elif mode.is_idle() or batch.batch_size() == 0:
                empty = torch.zeros(0, dtype=torch.int64)
                result = GenerationBatchResult(
                    logits_output=None, next_token_ids=empty, accept_lens=torch.zeros(0, dtype=torch.int32),
                    next_draft_input=make_draft_input_v2(bonus_tokens=empty, new_seq_lens=empty),
                    speculative_num_draft_tokens=self.q, new_seq_lens=empty, can_run_cuda_graph=False)
            elif mode.is_decode():
                result = self._decode(batch)
            else:
                raise RuntimeError(f"the Hy4 DFly worker runs extend, decode and idle batches, not {mode}")
            if on_publish is not None:
                on_publish(result.new_seq_lens)
            return result

        def _decode(self, batch):
            from sglang.srt.utils.common import DynamicGradMode

            from sglang_neuron.backend import device_io
            from sglang_neuron.model_executor.async_execution import refuse_invalid_token_ids
            from sglang_neuron.models.hy4.runtime import dfly_inputs

            rt = self.model._runtime_for("decode")
            q = self.q
            page = int(rt.block_size)
            r2t = batch.req_to_token_pool.req_to_token
            r2t = r2t.numpy() if isinstance(r2t, torch.Tensor) else np.asarray(r2t)
            n = len(batch.reqs)
            rows = n * q
            lens = np.array([int(req.kv.kv_committed_len) for req in batch.reqs], dtype=np.int64)
            pools = np.array([int(req.kv.req_pool_idx) for req in batch.reqs], dtype=np.int64)
            cand = np.full((n, q), -1, dtype=np.int64)
            for i, req in enumerate(batch.reqs):
                cand[i, 0] = req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
                held = None if _NO_DRAFTS else self.drafts.get(req.rid)
                if held is not None and held[0] == lens[i]:
                    cand[i, 1:] = held[1]
            offs = np.arange(q, dtype=np.int64)
            positions = (lens[:, None] + offs[None, :])  # [n, Q]
            slots = r2t[pools[:, None], positions].astype(np.int64)
            ahead = r2t[pools[:, None], lens[:, None] + 1 + np.arange(2 * q - 1)[None, :]].astype(np.int64)
            pages = int(-(-(int(lens.max()) + q) // page))
            table = (r2t[pools, : pages * page: page] // page).astype(np.int32)
            table = np.repeat(table, q, axis=0)
            live = np.arange(0, pages * page, page)[None, :] < (positions.reshape(-1) + 1)[:, None]
            table = np.where(live, table, -1).astype(np.int32)
            md = NeuronForwardMetadata(
                is_prefill=False, batch_size=rows, num_tokens=rows, block_table=torch.from_numpy(table),
                slot_mapping=torch.from_numpy(slots.reshape(-1)), seq_lens=torch.from_numpy(positions.reshape(-1) + 1),
                positions=torch.from_numpy(positions.reshape(-1)), cached_seq_len=0)
            bucket = rt.buckets.seqs_bucket(rows)
            sampling = rt._sampling_kwargs(_Sampling(batch.sampling_info))
            sampling = {k: v.repeat_interleave(q, dim=0) if isinstance(v, torch.Tensor) and v.dim() else v
                        for k, v in sampling.items()}
            ids = torch.from_numpy(np.clip(cand, 0, None).reshape(-1).astype(np.int32))
            kwargs = rt.build_model_inputs(md, token_ids=ids, bucket=bucket, **sampling)
            b = bucket // q
            pad_cand = torch.full((bucket,), -1, dtype=torch.int32)
            pad_cand[:rows] = torch.from_numpy(cand.reshape(-1).astype(np.int32))
            pad_base = torch.zeros(b, dtype=torch.int64)
            pad_base[:n] = torch.from_numpy(lens)
            pad_slots = torch.zeros(b, 2 * q - 1, dtype=torch.int64)
            pad_slots[:n] = torch.from_numpy(ahead)
            kwargs.update(dfly_inputs(bucket, q, pad_cand, pad_base, pad_slots))
            with DynamicGradMode(), torch.compiler.set_stance("fail_on_recompile"):
                device_io.before_launch()
                with device_io.pre_execute_scope():
                    out = rt.compiled(**kwargs)
                event = device_io.after_launch()
            tokens, accepted, drafts = device_io.readback_many(
                [out["token_ids"], out["dfly_accepted"], out["dfly_drafts"]], event)
            tokens = tokens[:rows].to(torch.int64).view(n, q)
            refuse_invalid_token_ids(tokens.reshape(-1), rt.vocab_size)
            accepted = accepted[:n].to(torch.int64)
            drafts = drafts[:n].to(torch.int64).numpy()
            accept_lens = (accepted + 1).to(torch.int32)
            new_lens = torch.from_numpy(lens) + accept_lens.to(torch.int64)
            for i, req in enumerate(batch.reqs):
                self.drafts[req.rid] = (int(new_lens[i]), drafts[i])
            self._log(n, int(accepted.sum()), cand)
            out_tokens = tokens.clone()
            out_tokens[torch.arange(q)[None, :] > accepted[:, None]] = -1
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None), next_token_ids=out_tokens.reshape(-1),
                accept_lens=accept_lens,
                next_draft_input=make_draft_input_v2(bonus_tokens=tokens.gather(1, accepted[:, None])[:, 0],
                                                     new_seq_lens=new_lens),
                speculative_num_draft_tokens=q, new_seq_lens=new_lens, can_run_cuda_graph=False)

        def _log(self, n: int, accepted: int, cand) -> None:
            self.steps += 1
            self.verified += n
            self.accepted += accepted
            if self.steps % 64 == 0:
                now = time.perf_counter()
                logger.info("Hy4 DFly: %d steps, mean accepted drafts %.2f a request-step (accept length %.2f); "
                            "last 64 steps %.1f ms each; with drafts %d of %d", self.steps,
                            self.accepted / max(self.verified, 1), 1 + self.accepted / max(self.verified, 1),
                            (now - self.t_last) * 1e3 / 64, int((cand[:, 1] >= 0).sum()), n)
                self.t_last = now

    return Hy4DFlyWorker


_WORKER = None


def worker_class() -> type:
    global _WORKER
    if _WORKER is None:
        _WORKER = _worker_class()
    return _WORKER
