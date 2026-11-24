"""hicache_check.py PORT N_TOKENS FILLERS: a request cold, then FILLERS other requests of the same size (evicting its
pages from the device to HiCache's host tier), then the request again: its cached tokens (the host tier's load) and
whether its greedy output matches the cold run."""
import json, sys, time, random, urllib.request

port, n, fillers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
words = "the quick brown fox jumps over a lazy dog while seven bright stars glow above quiet hills".split()


def prompt(seed):
    rnd = random.Random(seed)
    code = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(4)) + "-" + str(rnd.randint(1000, 9999))
    f = [rnd.choice(words) for _ in range(int(n * 0.95))]
    f.insert(len(f) // 3, f". Remember this: the secret passphrase is {code}. ")
    return " ".join(f) + "\n\nQuestion: what is the secret passphrase mentioned in the text above?\nAnswer:", code


def gen(text):
    body = json.dumps({"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 64}}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/generate", body,
                                                                {"Content-Type": "application/json"}), timeout=7200))
    return r["text"], r["meta_info"].get("cached_tokens"), time.time() - t0


p, code = prompt(0)
a, c0, t0 = gen(p)
print(f"HICACHE cold cached={c0} time={t0:.0f}s match={code in a} text={a[-60:]!r}", flush=True)
for i in range(fillers):
    _, c, t = gen(prompt(i + 1)[0])
    print(f"HICACHE filler{i} cached={c} time={t:.0f}s", flush=True)
b, c1, t1 = gen(p)
print(f"HICACHE warm cached={c1} time={t1:.0f}s same={a == b} match={code in b} text={b[-60:]!r}", flush=True)
