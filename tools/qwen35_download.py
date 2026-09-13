# Qwen3.5-2B prep: weights + dataset shards downloader (2026-09-14, decisions:
# instruct variant; EN/ZH/Code/Instruct = 50/30/15/5 of ~2B tokens; scratch dir;
# download ONLY the needed shards, never whole datasets).
# Sources (docs/22 Q3.5): EN fineweb-edu sample-100BT x2 parquet; ZH CCI3-HQ
# (fallback FineWeb-2 cmn_Hani + TeleChat-PTD); Code stack-v3-train shards;
# Instruct HuggingFaceTB/smoltalk. Run on the cluster login->tiny job.
import os
import subprocess
import sys

ROOT = "/gpfs1/scratch/yihe47/qwen35"
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")

ENV = dict(os.environ)
ENV["HF_ENDPOINT"] = "https://hf-mirror.com"
ENV["HF_HUB_DISABLE_XET"] = "1"


def run(desc, cmd, env=ENV, ok_rc=(0,)):
    print(f"\n==== {desc}\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=env)
    print(f"rc={r.returncode}", flush=True)
    return r.returncode in ok_rc


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    # ---- 1. weights (ModelScope direct) ----
    if not run("Qwen3.5-2B weights (ModelScope)",
               ["modelscope", "download", "--model", "Qwen/Qwen3.5-2B",
                "--local_dir", os.path.join(MODELS, "Qwen3.5-2B")]):
        print("WARN: weights download failed", flush=True)

    # ---- 2. EN: fineweb-edu sample-100BT, 2 parquets (~1.43B tok, trim later) ----
    run("EN fineweb-edu 100BT x2 parquet",
        ["hf", "download", "HuggingFaceFW/fineweb-edu", "--repo-type", "dataset",
         "--include", "sample/100BT/000_00000.parquet", "sample/100BT/000_00001.parquet",
         "--local-dir", os.path.join(DATA, "fineweb_edu_100bt")])

    # ---- 3. ZH: CCI3-HQ (needs MS login+agreement); fallback FineWeb-2 cmn_Hani ----
    zh_ok = run("ZH CCI3-HQ first shards (ModelScope)",
                ["modelscope", "download", "--dataset", "BAAI/CCI3-HQ",
                 "--include", "data/part_00000[0-3].jsonl",
                 "--local_dir", os.path.join(DATA, "cci3hq")])
    cci_dir = os.path.join(DATA, "cci3hq")
    cci_files = os.listdir(cci_dir) if os.path.isdir(cci_dir) else []
    if (not zh_ok) or (not cci_files):
        print("CCI3-HQ unavailable -> fallback FineWeb-2 cmn_Hani + TeleChat-PTD", flush=True)
        run("ZH fallback FineWeb-2 cmn_Hani shard0",
            ["hf", "download", "HuggingFaceFW/fineweb-2", "--repo-type", "dataset",
             "--include", "data/cmn_Hani/train/000_00000.parquet",
             "--local-dir", os.path.join(DATA, "fineweb2_zh")])
        run("ZH fallback TeleChat-PTD file0",
            ["hf", "download", "Tele-AI/TeleChat-PTD", "--repo-type", "dataset",
             "--include", "data/0.jsonl.gz",
             "--local-dir", os.path.join(DATA, "telechat_ptd")])

    # ---- 4. Code: stack-v3-train shards (permissive filtered at cache build) ----
    run("Code stack-v3-train ~15 shards",
        ["hf", "download", "HuggingFaceCode/stack-v3-train", "--repo-type", "dataset",
         "--include", "data/part-0000*-7d88aa48*", "data/part-0001[0-4]-7d88aa48*",
         "--local-dir", os.path.join(DATA, "stack_v3")])

    # ---- 5. Instruct: smoltalk (ungated) ----
    run("Instruct smoltalk",
        ["hf", "download", "HuggingFaceTB/smoltalk", "--repo-type", "dataset",
         "--include", "data/train-0000[0-1]*",
         "--local-dir", os.path.join(DATA, "smoltalk")])

    print("\nDOWNLOAD_DONE", flush=True)
    for root, _, files in os.walk(ROOT):
        for f in files[:8]:
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p)/1e6:9.1f} MB  {p}", flush=True)


if __name__ == "__main__":
    main()
