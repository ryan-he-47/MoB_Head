# Qwen3.5-2B prep v4 (2026-09-14): ModelScope-only (hf-mirror's LFS bridge is
# broken from the cluster; MS file trees verified via API recon). Idempotent:
# skips sources whose target files already exist. Scratch dir only.
# Verified paths (MS API, 2026-09-14):
#   EN   HuggingFaceFW/fineweb-edu: sample/100BT/000_0000{0,1}.parquet (2.15GB each)
#   Code HuggingFaceCode/stack-v3-train: data/part-0000[0-7]-4beed122-* (batch UUID)
#   SFT  HuggingFaceTB/smoltalk: data/all/train-0000[0-1]*
#   ZH   BAAI/CCI3-HQ: data/part_00000[0-3].jsonl (already on disk from v1)
import os
import sys
import time

ROOT = "/gpfs1/scratch/yihe47/qwen35"
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")


def have(d, suffix):
    import glob
    if not os.path.isdir(d):
        return False
    return any(glob.glob(os.path.join(d, "**", f"*{suffix}"), recursive=True))


def dl(desc, fn, tries=3):
    for i in range(tries):
        try:
            fn()
            print(f"{desc} OK", flush=True)
            return True
        except Exception as e:
            print(f"{desc} attempt {i+1} failed: {str(e)[:300]}", flush=True)
            time.sleep(15)
    return False


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    from modelscope.hub.snapshot_download import snapshot_download as ms_model_dl
    try:
        from modelscope.hub.snapshot_download import dataset_snapshot_download as ms_data_dl
    except ImportError:
        from modelscope import dataset_snapshot_download as ms_data_dl

    # ---- 1. weights (resume until no .incomplete) ----
    mdir = os.path.join(MODELS, "Qwen3.5-2B")
    if os.path.isdir(mdir) and not any(f.endswith(".incomplete") for f in os.listdir(mdir)):
        print("[1] weights already complete, skip", flush=True)
    else:
        def weights():
            ms_model_dl("Qwen/Qwen3.5-2B", local_dir=mdir)
            inc = [f for f in os.listdir(mdir) if f.endswith(".incomplete")]
            if inc:
                raise RuntimeError(f"incomplete: {inc}")
        dl("[1] weights", weights, tries=5)

    # ---- 2. EN: fineweb-edu sample-100BT x2 ----
    d = os.path.join(DATA, "fineweb_edu_100bt")
    if have(d, ".parquet"):
        print("[2] EN already present, skip", flush=True)
    else:
        dl("[2] EN", lambda: ms_data_dl(
            "HuggingFaceFW/fineweb-edu",
            allow_file_pattern=["sample/100BT/000_00000.parquet",
                                "sample/100BT/000_00001.parquet"],
            local_dir=d))

    # ---- 3. ZH: cci3hq present from v1 (verify only) ----
    if have(os.path.join(DATA, "cci3hq"), ".jsonl"):
        print("[3] ZH cci3hq present, skip", flush=True)
    else:
        dl("[3] ZH CCI3-HQ", lambda: ms_data_dl(
            "BAAI/CCI3-HQ", allow_file_pattern="data/part_00000[0-3].jsonl",
            local_dir=os.path.join(DATA, "cci3hq")))

    # ---- 4. Code: stack-v3 one UUID batch, 8 shards ----
    d = os.path.join(DATA, "stack_v3")
    if have(d, ".parquet"):
        print("[4] code already present, skip", flush=True)
    else:
        dl("[4] code", lambda: ms_data_dl(
            "HuggingFaceCode/stack-v3-train",
            allow_file_pattern="data/part-0000[0-7]-4beed122-*",
            local_dir=d))

    # ---- 5. Instruct: smoltalk 'all' config, 2 shards ----
    d = os.path.join(DATA, "smoltalk")
    if have(d, ".parquet"):
        print("[5] instruct already present, skip", flush=True)
    else:
        dl("[5] instruct", lambda: ms_data_dl(
            "HuggingFaceTB/smoltalk", allow_file_pattern="data/all/train-0000[0-1]*",
            local_dir=d))

    print("\nDOWNLOAD_DONE", flush=True)
    for root, _, files in os.walk(ROOT):
        for f in files[:6]:
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p)/1e6:9.1f} MB  {p}", flush=True)


if __name__ == "__main__":
    main()
