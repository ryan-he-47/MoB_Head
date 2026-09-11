# Dump all LR-scan result JSONs as a compact table (CPU only).
import json, glob, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rows = []
for p in sorted(glob.glob("runs/lr_scan/*.json")):
    d = json.load(open(p))
    rows.append(d)
print(f"{'variant':9s} {'peak':>7s} {'steps':>5s} {'final_nll':>9s} {'dev_nll':>8s} {'agree':>7s} {'spikes':>6s} {'slope_last':>10s}")
for d in rows:
    print(f"{d['variant']:9s} {d['peak']:>7.4f} {d['steps']:>5d} {d['final_nll']:>9.4f} {d['dev_nll']:>8.4f} "
          f"{d.get('dev_agree', float('nan')):>7.4f} {d['spikes']:>6d} {d['slope_last']:>+10.4f}")
