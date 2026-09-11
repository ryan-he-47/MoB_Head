import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
sys.path.insert(0, "scripts")
from mob.mob_common import SwiGLUHead, ResMLPHead, MoBHead

for name, head in [("swiglu1-fixed", SwiGLUHead(k=128, hidden=7300, m=17)),
                   ("resmlp5", ResMLPHead(k=128, width=3072, depth=5, m=17)),
                   ("mlp3", MoBHead(k=128, width=3072, depth=3, m=17))]:
    n = sum(p.numel() for p in head.parameters())
    bl, pl = head(torch.randn(4, 2048))
    assert bl.shape == (4, 128, 17) and pl.shape == (4, 128)
    print(f"{name:15s} {n/1e6:.1f}M  forward OK")
