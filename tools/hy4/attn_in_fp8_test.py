import sys, time, torch, torch_neuronx  # noqa
sys.path.insert(0, "/mnt/nvme/sglang-neuron")
from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.hy4.kernels.attn_in import attn_in_decode, qkv_shard_layout
from sglang_neuron.utils.hardware_config import get_logical_nc_config
bf = torch.bfloat16; T = int(sys.argv[1]); H, QL, NOPE, RP, L = 6144, 2048, 192, 64, 512
torch.manual_seed(0)
x = (torch.randn(T, H) * 2).to(bf); lnw = (torch.rand(H) + 0.5).to(bf)
qa = (torch.randn(H, QL) * 0.02).to(bf); qaw = (torch.rand(QL) + 0.5).to(bf)
qb = (torch.randn(QL, NOPE + RP) * 0.02).to(bf); wuk = (torch.randn(1, NOPE, L) * 0.05).to(bf)
kva = (torch.randn(H, L + RP) * 0.02).to(bf); kvw = (torch.rand(L) + 0.5).to(bf)
pos = torch.arange(T) + 1000; inv = 1.0 / (10000 ** (torch.arange(0, RP, 2).float() / RP))
fr = pos.float()[:, None] * inv[None]; emb = torch.cat([fr, fr], -1); cos, sin = emb.cos().to(bf), emb.sin().to(bf)
def rms(v, w, eps):
    v32 = v.float(); return w * (v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True) + eps)).to(bf)
def rot(v):
    h = v.shape[-1] // 2; return v * cos + torch.cat([-v[..., h:], v[..., :h]], -1) * sin
xn = rms(x, lnw, 1e-5)
ql = rms(xn @ qa, qaw, 1e-6)
q = ql @ qb; qn, qp = q[:, :NOPE], q[:, NOPE:]
qabs = torch.einsum("td,dr->tr", qn, wuk[0]); qry = torch.cat([qabs, rot(qp)], -1)
gw = (torch.randn(H, 256) * 0.02).to(bf); gref = torch.sigmoid(xn @ gw)
lp = xn @ kva; comp = torch.cat([rms(lp[:, :L], kvw, 1e-6), rot(lp[:, L:])], -1)
k = wrap_nki(attn_in_decode, alias_outputs_in_place=True)[get_logical_nc_config()]
SH = len(sys.argv) > 2 and sys.argv[2] == "sh"
def f(x, lnw, qa, qaw, qb, wuk, kva, kvw, cos, sin, cache, wr, gw, *sh):
    r = k(x, lnw, qa, qaw, qb, wuk, kva, kvw, cos, sin, cache, wr, gate=gw, ranks=(), eps=1e-5, lat_eps=1e-6,
          qkv_sh=sh[0] if sh else None)
    return r[0], r[1], r[2], r[4]
args = [t.to("neuron").contiguous() for t in (x, lnw.reshape(1, H), qa, qaw.reshape(1, QL), qb, wuk, kva, (kvw * 8).reshape(1, L), cos, sin)]
cache = torch.zeros(64, 640, dtype=torch.uint8).to('neuron'); wr = torch.arange(1, T + 1, dtype=torch.int32).reshape(1, T).to('neuron')
args += [cache, wr, gw.to('neuron')]
if SH: args.append(qkv_shard_layout(qa, kva, gw, 0, 1).to('neuron'))
c = torch.compile(f, backend="neuron", fullgraph=True, dynamic=False)
oq, oc, ol, og = (t.cpu() for t in c(*args))
print('gate relerr', ((og.float() - gref.float()).abs().max()).item())
e = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
c8 = cache.cpu()[1:T + 1]
lat = c8[:, :512].contiguous().view(torch.float8_e4m3fn).to(bf); rope = c8[:, 512:].contiguous().view(bf)
want = torch.cat([oc[:, :L].clamp(-224, 224).to(torch.float8_e4m3fn).view(torch.uint8), oc[:, L:].contiguous().view(torch.uint8)], 1)
print("fp8 cache rows == host pack of compact", torch.equal(c8, want), "lat relerr vs 8 comp", e(lat, comp[:, :L] * 8), "rope ok", torch.equal(rope, oc[:, L:]))
print(f"relerr query {e(oq[:, 0], qry):.2e} compact {e(oc[:, :L], comp[:, :L] * 8):.2e} q_lat {e(ol, ql):.2e}")
t = time.time()
for _ in range(20): r = c(*args)
r[0].cpu(); print(f"ATTNIN{'SH' if SH else ''} T={T}: {(time.time() - t) / 20 * 1e3:.3f} ms", flush=True)
import kcap_util
kcap_util.profile(f, *args)
