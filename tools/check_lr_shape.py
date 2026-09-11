import math, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def lr_at(step, peak, warmup=2500, total=228000, schedule="cosine", decay_frac=0.35):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    if schedule == "cosine":
        return peak * (0.1 + 0.45 * (1.0 + math.cos(math.pi * t)))
    decay_start = 1.0 - decay_frac
    if t < decay_start:
        return peak
    u = (t - decay_start) / decay_frac
    return peak * (0.1 + 0.9 * (1.0 - u))

for sched in ("cosine", "wsd"):
    pts = [0, 1250, 2500, 25000, 57000, 114000, 171000, 205000, 227999]
    row = [lr_at(s, 7e-4, schedule=sched) for s in pts]
    print(sched.ljust(7), " ".join(f"{v/7e-4:6.3f}" for v in row), "(x peak @ step", pts, ")")
