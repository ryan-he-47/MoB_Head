# Compare dual headscale arms vs the ORIGINAL WSD run (K=128, 3072x3, 32.3M,
# job 520825) at matched steps. CPU-only.
import json
import sys

RUNS = r".\runs"


def load_panels(path, kind):
    panels = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if not d.get("panel"):
            continue
        if kind == "dual":
            panels[d["step"]] = (d["agree_wide"], d["agree_deep"],
                                 d["dev_nll_wide"], d["dev_nll_deep"])
        else:
            panels[d["step"]] = d["dev_top1_agree"]
    return panels


dual = load_panels(RUNS + r"\headscale_dual_log.jsonl", "dual")
orig = load_panels(RUNS + r"\train_log_wsd.jsonl", "orig")

print(f"{'step':>7} | {'orig(32M)':>9} | {'wide(51M)':>9} | {'deep(51M)':>9} | w-orig | d-orig")
for s in sorted(dual):
    o = orig.get(s)
    if o is None:
        continue
    w, d, nw, nd = dual[s]
    print(f"{s:>7} | {o:9.4f} | {w:9.4f} | {d:9.4f} | {w-o:+6.4f} | {d-o:+6.4f}")

last = dual[max(dual)]
s = max(dual)
o = orig.get(s)
print(f"\nlatest: step {s}, dev_nll wide {last[2]:.4f} deep {last[3]:.4f}")
