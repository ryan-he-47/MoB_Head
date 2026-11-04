
# (a) did shallow1l converge? (b) did it show a second-stage climb in the WSD
# decay tail like deep5 did? Compare shapes vs control WSD and deep5/wide arms.
# CPU only. WSD decay-frac 0.35 over 245k -> decay onset ~159k.
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DECAY_ONSET = 159250

def panels(path, keys):
    out = []
    for line in open(path, encoding="utf-8"):
        try: rec = json.loads(line)
        except Exception: continue
        if rec.get("panel"):
            row = {"step": rec["step"]}
            for k in keys:
                if k in rec: row[k] = rec[k]
            out.append(row)
    return out

def seg(rows, key, lo, hi):
    xs = [r["step"] for r in rows if lo <= r["step"] < hi and key in r]
    ys = [r[key] for r in rows if lo <= r["step"] < hi and key in r]
    return xs, ys

def slope_per10k(xs, ys):
    if len(xs) < 2: return float("nan")
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    denom = sum((x-mx)**2 for x in xs)
    if denom == 0: return float("nan")
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/denom * 10000

def report(name, rows, key, scale=1.0):
    pre_x, pre_y = seg(rows, key, 60000, DECAY_ONSET)          # constant-LR phase
    tail_x, tail_y = seg(rows, key, DECAY_ONSET, 1<<30)        # decay tail
    end_x, end_y = seg(rows, key, 215000, 1<<30)               # last 30k
    pre_slope = slope_per10k(pre_x, pre_y)*scale
    tail_gain = (tail_y[-1]-tail_y[0])*scale if len(tail_y)>=2 else float("nan")
    end_slope = slope_per10k(end_x, end_y)*scale
    at_onset = pre_y[-1] if pre_y else float("nan")
    final = rows[-1][key]*scale if key in rows[-1] else float("nan")
    print(f"{name:22s} onset@{DECAY_ONSET/1000:.0f}k: {at_onset:.4f} | final: {final:.4f} | "
          f"pre-decay slope {pre_slope:+.4f}/10k | tail gain {tail_gain:+.4f} | end slope {end_slope:+.4f}/10k")

shallow = panels("runs/shallow_dual_log.jsonl", ["agree_shallow1l","dev_nll_shallow1l","agree_linear","dev_nll_linear"])
head    = panels("runs/headscale_dual_log.jsonl", ["agree_wide","agree_deep","dev_nll_wide","dev_nll_deep"])
wsd     = panels("runs/train_log_wsd.jsonl",     ["dev_top1_agree","dev_nll"])

print("=== agreement (points/10k) ===")
report("shallow1l", shallow, "agree_shallow1l")
report("linear",    shallow, "agree_linear")
report("deep5",     head,    "agree_deep")
report("wide3",     head,    "agree_wide")
report("control_wsd", wsd,   "dev_top1_agree")
print("=== dev_nll (nats/10k) ===")
report("shallow1l", shallow, "dev_nll_shallow1l")
report("deep5",     head,    "dev_nll_deep")
report("control_wsd", wsd,   "dev_nll")

# step-by-step trajectory around the tail for shallow1l (compact table)
print("\n=== shallow1l tail detail (agree) ===")
for r in shallow:
    if r["step"] >= 140000 and r["step"] % 5000 == 0:
        print(f"  {r['step']:>6d}  agree {r['agree_shallow1l']:.4f}  dev_nll {r['dev_nll_shallow1l']:.4f}")
print("\n=== control WSD tail detail ===")
for r in wsd:
    if r["step"] >= 140000 and r["step"] % 5000 == 0 and "dev_top1_agree" in r:
        print(f"  {r['step']:>6d}  agree {r['dev_top1_agree']:.4f}  dev_nll {r['dev_nll']:.4f}")
