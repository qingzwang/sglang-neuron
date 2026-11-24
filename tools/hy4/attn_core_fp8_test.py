import sys, time, torch, torch_neuronx  # noqa
sys.path.insert(0, "/mnt/nvme/sglang-neuron")
from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.hy4.kernels.attn_core import attn_core_decode
from sglang_neuron.models.hy4.kernels.mla_sparse import block_slots
from sglang_neuron.models.hy4.kernels.mla_dense import onehot_rows
bf = torch.bfloat16; T = int(sys.argv[1]); H, D, L, V, Hd, R, CAP = 64, 576, 512, 256, 6144, 2048, 256
torch.manual_seed(0); scale = (192 + 64) ** -0.5
q = (torch.randn(T, H, D)).to(bf); cache = (torch.randn(R, D)).to(bf); cache[:, :L] = (cache[:, :L] * 8).clamp(-224, 224)
cache8 = torch.cat([cache[:, :L].to(torch.float8_e4m3fn).view(torch.uint8), cache[:, L:].contiguous().view(torch.uint8)], 1)
cache = torch.cat([cache8[:, :L].contiguous().view(torch.float8_e4m3fn).to(bf), cache8[:, L:].contiguous().view(bf)], 1)
rows = torch.stack([torch.randperm(R - 1)[:CAP] + 1 for _ in range(T)]).to(torch.int32)
keep = torch.rand(T, CAP) < 0.6
sinks = torch.randn(H); wuv = (torch.randn(V, L) * 0.05).to(bf); gx = torch.rand(T, V).to(bf); o = (torch.randn(V, Hd) * 0.05).to(bf)
# reference: one rank, head 0 kept
keys = cache[rows.long()]  # [T, CAP, D]
s = torch.einsum("thd,tkd->thk", q, keys).float() * scale
s = torch.where(keep[:, None, :], s, torch.full_like(s, -1e30))
m = s.amax(-1); p = torch.exp(s - m[..., None]); l = p.sum(-1)
out = torch.einsum("thk,tkr->thr", p.to(bf), keys[..., :L]).float()
M = torch.maximum(m, sinks[None]); tot = torch.exp(m - M) * l + torch.exp(sinks[None] - M)
att = out * (torch.exp(m - M) / tot)[..., None]
values = (att[:, 0].to(bf) @ wuv.t())
ref = ((values * gx) @ o)
Tp = T + (-T % 4)
qp = torch.cat([q, q.new_zeros(Tp - T, H, D)]); rp = torch.cat([rows, rows.new_zeros(Tp - T, CAP)]); kp = torch.cat([keep, keep.new_zeros(Tp - T, CAP)])
slots, bias = block_slots(rp, kp, H, R)
k = wrap_nki(attn_core_decode)[2]
def f(q, cache, slots, bias, oh, sinks, wuv, gx, o):
    return k(q, cache, slots, bias, oh, sinks, wuv, gx, o, ranks=(), scale=scale, t_real=T)
args = [t.to("neuron").contiguous() for t in (qp.reshape(-1, 128, D), cache8, slots, bias, onehot_rows(H), sinks.reshape(1, H), wuv, gx, o)]
c = torch.compile(f, backend="neuron", fullgraph=True, dynamic=False)
r = c(*args).cpu()
print("relerr", ((r.float() - ref.float()).abs().max() / ref.float().abs().max()).item())
t = time.time()
for _ in range(20): r = c(*args)
r.cpu(); print(f"ATTNCORE T={T}: {(time.time() - t) / 20 * 1e3:.3f} ms", flush=True)
import kcap_util; kcap_util.profile(f, *args)
