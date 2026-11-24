import sys, time, torch, torch_neuronx  # noqa
sys.path.insert(0, "/mnt/nvme/sglang-neuron")
from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.hy4.kernels.mla_sparse import mla_sparse, block_slots
from sglang_neuron.models.hy4.kernels.mla_dense import onehot_rows
from sglang_neuron.utils.hardware_config import get_logical_nc_config
bf = torch.bfloat16; T, H, D, L, CAP, R = 1024, 64, 576, 512, 256, 16400; SC = 0.0625; MASKED = -1.0e30
torch.manual_seed(0)
q = torch.randn(T, H, D).to(bf); cache = torch.randn(R, D).to(bf); cache[:, :L] = (cache[:, :L] * 8).clamp(-224, 224)
pack = lambda x: torch.cat([x[:, :L].to(torch.float8_e4m3fn).view(torch.uint8), x[:, L:].contiguous().view(torch.uint8)], 1)
unpack = lambda c: torch.cat([c[:, :L].contiguous().view(torch.float8_e4m3fn).to(bf), c[:, L:].contiguous().view(bf)], 1)
cache8 = pack(cache); cache = unpack(cache8)  # the reference over the FP8 values; the kernel reads cache8
rows = torch.randint(0, R, (T, CAP)); nkeep = torch.randint(0, CAP + 1, (T,)); nkeep[:8] = 0; nkeep[8:16] = CAP
keep = torch.arange(CAP)[None, :] < nkeep[:, None]
# reference: each token over its kept rows (a token with none: m = MASKED, l = 0 is not compared)
kk = cache[rows].float()
s = torch.einsum("thd,tkd->thk", q.float(), kk) * SC + torch.where(keep, 0.0, MASKED)[:, None, :]
m_ref = s.amax(-1); p = torch.exp(s - m_ref[..., None]); l_ref = p.sum(-1)
o_ref = torch.einsum("thk,tkr->thr", p, kk[..., :L]) / l_ref[..., None]
kern = wrap_nki(mla_sparse)[get_logical_nc_config()]
def nki_fn(q, cache, rows, keep):
    slots, bias = block_slots(rows, keep, H, R)
    out, ml = kern(q.reshape(-1, 128, D), cache, slots, bias, onehot_rows(H, q.device), scale=SC)
    return out.transpose(0, 1), ml[..., 0].t(), ml[..., 1].t()
def xla_fn(q, cache, rows, keep):
    ms, ls, outs = [], [], []
    for st in range(0, T, 256):
        keys = cache[rows[st:st + 256]]
        s = torch.einsum("rhd,rkd->rhk", q[st:st + 256], keys).float() * SC
        s = torch.where(keep[st:st + 256, None, :], s, torch.full_like(s, MASKED))
        m = s.amax(-1); p = torch.exp(s - m[..., None]); ls.append(p.sum(-1)); ms.append(m)
        outs.append(torch.einsum("rhk,rkd->rhd", p.to(bf), keys[..., :L]).float())
    return torch.cat(outs), torch.cat(ms), torch.cat(ls)
args = [t.to("neuron") for t in (q, cache8, rows, keep)]
ok = nkeep > 0
for name, fn in [("nki_fp8", nki_fn)]:
    c = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
    o, m, l = (x.cpu() for x in c(*args))
    on = o / l[..., None]
    err = ((on[ok] - o_ref[ok]).abs().max() / o_ref[ok].abs().max()).item(); merr = (m[ok] - m_ref[ok]).abs().max().item()
    finite = bool(torch.isfinite(o).all() and torch.isfinite(m).all())
    r = c(*args); r[0].cpu(); t = time.time()
    for _ in range(5): r = c(*args)
    r[0].cpu()
    print(f"{name}: out relerr {err:.2e} m maxerr {merr:.2e} finite {finite} {(time.time()-t)/5*1e3:.2f} ms wall", flush=True)

