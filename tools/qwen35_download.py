# Qwen3.5-2B prep v3 (2026-09-14): list-then-download (never guess filenames).
# Weights via ModelScope (resume until no .incomplete); datasets via hf-mirror
# with explicit file lists from list_repo_files. Scratch dir only.
import os
import sys
import time

ROOT = "/gpfs1/scratch/yihe47/qwen35"
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"


def hf_list(api, repo, prefix, suffix=".parquet", limit=None):
    files = api.list_repo_files(repo, repo_type="dataset")
    hits = sorted(f for f in files if f.startswith(prefix) and f.endswith(suffix))
    if limit:
        hits = hits[:limit]
    return hits


def dl_with_retry(desc, fn, tries=3):
    for i in range(tries):
        try:
            fn()
            print(f"{desc} OK", flush=True)
            return True
        except Exception as e:
            print(f"{desc} attempt {i+1} failed: {str(e)[:300]}", flush=True)
            time.sleep(10)
    return False


def glob_jsonl(d):
    import glob as g
    return g.glob(os.path.join(d, "**", "*.jsonl"), recursive=True) if os.path.isdir(d) else []


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])

    # ---- 1. weights (ModelScope, resume until complete) ----
    def weights():
        from modelscope.hub.snapshot_download import snapshot_download as ms_dl
        ms_dl("Qwen/Qwen3.5-2B", local_dir=os.path.join(MODELS, "Qwen3.5-2B"))
        inc = [f for f in os.listdir(os.path.join(MODELS, "Qwen3.5-2B"))
               if f.endswith(".incomplete")]
        if inc:
            raise RuntimeError(f"incomplete files remain: {inc}")
    dl_with_retry("[1] weights Qwen3.5-2B", weights, tries=4)

    # ---- 2. EN: fineweb-edu sample-100BT, first 2 parquets ----
    en_files = hf_list(api, "HuggingFaceFW/fineweb-edu", "sample/100BT/", limit=2)
    print("[2] EN files:", en_files, flush=True)

    def en():
        for f in en_files:
            hf_hub_download("HuggingFaceFW/fineweb-edu", f, repo_type="dataset",
                            local_dir=os.path.join(DATA, "fineweb_edu_100bt"))
    dl_with_retry("[2] EN fineweb-edu", en)

    # ---- 3. ZH: cci3hq already present (3.9G from v1); verify, else fallback ----
    cci = glob_jsonl(os.path.join(DATA, "cci3hq"))
    print(f"[3] cci3hq jsonl files present: {len(cci)}", flush=True)
    if not cci:
        def zh_fb():
            for f in hf_list(api, "HuggingFaceFW/fineweb-2", "data/cmn_Hani/train/", limit=1):
                hf_hub_download("HuggingFaceFW/fineweb-2", f, repo_type="dataset",
                                local_dir=os.path.join(DATA, "fineweb2_zh"))
            for f in hf_list(api, "Tele-AI/TeleChat-PTD", "data/", suffix=".jsonl.gz", limit=1):
                hf_hub_download("Tele-AI/TeleChat-PTD", f, repo_type="dataset",
                                local_dir=os.path.join(DATA, "telechat_ptd"))
        dl_with_retry("[3] ZH fallbacks", zh_fb)

    # ---- 4. Code: stack-v3-train, first 15 data parquets by list ----
    code_files = hf_list(api, "HuggingFaceCode/stack-v3-train", "data/", limit=15)
    print(f"[4] code shards matched: {len(code_files)}", flush=True)

    def code():
        for f in code_files:
            hf_hub_download("HuggingFaceCode/stack-v3-train", f, repo_type="dataset",
                            local_dir=os.path.join(DATA, "stack_v3"))
    dl_with_retry("[4] code stack-v3", code)

    # ---- 5. Instruct: smoltalk, list and pick train shards ----
    st_files = api.list_repo_files("HuggingFaceTB/smoltalk", repo_type="dataset")
    st_train = sorted(f for f in st_files if f.endswith(".parquet") and "/train" in f)
    if not st_train:
        st_train = sorted(f for f in st_files if f.startswith("data/train") and f.endswith(".parquet"))
    pick = st_train[:4]
    print("[5] smoltalk files:", pick, flush=True)

    def ins():
        for f in pick:
            hf_hub_download("HuggingFaceTB/smoltalk", f, repo_type="dataset",
                            local_dir=os.path.join(DATA, "smoltalk"))
    dl_with_retry("[5] instruct smoltalk", ins)

    print("\nDOWNLOAD_DONE", flush=True)
    for root, _, files in os.walk(ROOT):
        for f in files[:6]:
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p)/1e6:9.1f} MB  {p}", flush=True)


if __name__ == "__main__":
    main()
