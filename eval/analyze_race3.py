# Race3 curve morphology: per-arm trajectories, end slopes, phase analysis.
# Questions: (a) cosine shape -> no plateau? (b) converged or still rising?
# (c) control saturation vs new-arch headroom. CPU only.
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rows = []
for line in open("runs/race3_train_log.jsonl", encoding="utf-8"):
    try: rec = json.loads(line)
    except Exception: continue
    if rec.get("panel"): rows.append(rec)

arms = ["control", "resmlp", "swiglu"]
def seg(key, lo, hi):
    xs = [r["step"] for r in rows if lo <= r["step"] < hi and f"agree_{key}" in r]
    ys = [r[f"agree_{key}"] for r in rows if lo <= r["step"] < hi and f"agree_{key}" in r]
    return xs, ys

def slope_per10k(xs, ys):
    if len(xs) < 2: return float("nan")
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    d = sum((x-mx)**2 for x in xs)
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/d * 10000 if d else float("nan")

print(f"{'arm':9s} {'@20k':>7s} {'@60k':>7s} {'@120k':>7s} {'@180k':>7s} {'final':>7s} | slope 60-120k  120-180k  last30k  (agree pts/10k)")
for a in arms:
    def at(s):
        xs, ys = seg(a, s, s+2000)
        return ys[-1] if ys else float("nan")
    final = rows[-1][f"agree_{a}"]
    s1 = slope_per10k(*seg(a, 60000, 120000))
    s2 = slope_per10k(*seg(a, 120000, 180000))
    s3 = slope_per10k(*seg(a, 198000, 1 << 30))
    print(f"{a:9s} {at(20000):7.4f} {at(60000):7.4f} {at(120000):7.4f} {at(180000):7.4f} {final:7.4f} |   {s1:+.4f}     {s2:+.4f}    {s3:+.4f}")

print("\ndev_nll trajectory:")
for a in arms:
    def atn(s):
        xs = [r["step"] for r in rows if s <= r["step"] < s+2000]
        ys = [r[f"dev_nll_{a}"] for r in rows if s <= r["step"] < s+2000]
        return ys[-1] if ys else float("nan")
    print(f"{a:9s} 20k {atn(20000):.3f}  60k {atn(60000):.3f}  120k {atn(120000):.3f}  180k {atn(180000):.3f}  228k {atn(227000):.3f}")

print("\nn_eff / dead final:", {a: (rows[-1][f"n_eff_{a}"], rows[-1][f"dead_{a}"]) for a in arms})

# milestones: when did each arm cross agree 0.70 / 0.75 / 0.78?
for th in (0.70, 0.75, 0.78):
    line = {}
    for a in arms:
        xs, ys = seg(a, 0, 1 << 30)
        cross = next((x for x, y in zip(xs, ys) if y >= th), None)
        line[a] = f"{cross/1000:.0f}k" if cross else "never"
    print(f"agree >= {th}: " + "  ".join(f"{a}={v}" for a, v in line.items()))
