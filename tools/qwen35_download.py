# Qwen3.5-2B prep: weights + dataset shards downloader (2026-09-14, decisions:
# instruct variant; EN/ZH/Code/Instruct = 50/30/15/5 of ~2B tokens; scratch dir;
# download ONLY the needed shards, never whole datasets).
# Pure-Python API version (the modelscope/hf CLIs are not on PATH in cose2).
import os
import sys

ROOT = "/gpfs1/scratch/yihe47/qwen35"
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    from modelscope.hub.snapshot_download import snapshot_download as ms_model_dl
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download as ms_data_dl
    except ImportError:
        from modelscope import dataset_snapshot_download as ms_data_dl
    from huggingface_hub import snapshot_download as hf_dl

    # ---- 1. weights (ModelScope direct) ----
    try:
        ms_model_dl("Qwen/Qwen3.5-2B", local_dir=os.path.join(MODELS, "Qwen3.5-2B"))
        print("[1] weights OK", flush=True)
    except Exception as e:
        print(f"[1] weights FAILED: {e}", flush=True)

    # ---- 2. EN: fineweb-edu sample-100BT, 2 parquets ----
    try:
        hf_dl("HuggingFaceFW/fineweb-edu", repo_type="dataset",
              allow_patterns=["sample/100BT/000_00000.parquet", "sample/100BT/000_00001.parquet"],
              local_dir=os.path.join(DATA, "fineweb_edu_100bt"))
        print("[2] EN OK", flush=True)
    except Exception as e:
        print(f"[2] EN FAILED: {e}", flush=True)

    # ---- 3. ZH: CCI3-HQ (needs MS login+agreement); fallback FineWeb-2 + TeleChat ----
    zh_done = False
    try:
        ms_data_dl("BAAI/CCI3-HQ", allow_file_pattern="data/part_00000[0-3].jsonl",
                   local_dir=os.path.join(DATA, "cci3hq"))
        zh_done = True
        print("[3] ZH CCI3-HQ OK", flush=True)
    except Exception as e:
        print(f"[3] CCI3-HQ FAILED ({e}) -> fallbacks", flush=True)
    if not zh_done:
        for repo, pat, sub in (
            ("HuggingFaceFW/fineweb-2", ["data/cmn_Hani/train/000_00000.parquet"], "fineweb2_zh"),
            ("Tele-AI/TeleChat-PTD", ["data/0.jsonl.gz"], "telechat_ptd"),
        ):
            try:
                hf_dl(repo, repo_type="dataset", allow_patterns=pat,
                      local_dir=os.path.join(DATA, sub))
                print(f"[3] ZH fallback {repo} OK", flush=True)
            except Exception as e:
                print(f"[3] ZH fallback {repo} FAILED: {e}", flush=True)

    # ---- 4. Code: stack-v3-train shards ----
    try:
        hf_dl("HuggingFaceCode/stack-v3-train", repo_type="dataset",
              allow_patterns=["data/part-0000*-7d88aa48*", "data/part-0001[0-4]-7d88aa48*"],
              local_dir=os.path.join(DATA, "stack_v3"))
        print("[4] Code OK", flush=True)
    except Exception as e:
        print(f"[4] Code FAILED: {e}", flush=True)

    # ---- 5. Instruct: smoltalk ----
    try:
        hf_dl("HuggingFaceTB/smoltalk", repo_type="dataset",
              allow_patterns=["data/train-0000[0-1]*"],
              local_dir=os.path.join(DATA, "smoltalk"))
        print("[5] Instruct OK", flush=True)
    except Exception as e:
        print(f"[5] Instruct FAILED: {e}", flush=True)

    print("\nDOWNLOAD_DONE", flush=True)
    for root, _, files in os.walk(ROOT):
        for f in files[:8]:
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p)/1e6:9.1f} MB  {p}", flush=True)


if __name__ == "__main__":
    main()
