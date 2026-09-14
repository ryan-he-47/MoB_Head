import glob

import pyarrow.parquet as pq

files = sorted(glob.glob("/gpfs1/scratch/yihe47/qwen35/data/smoltalk/**/*.parquet", recursive=True))
print("SMOLTALK files:", [f.split("/")[-1] for f in files])
if files:
    pf = pq.ParquetFile(files[0])
    print("arrow schema names:", pf.schema_arrow.names)
    b = next(pf.iter_batches(batch_size=1))
    for k, v in b.to_pydict().items():
        print(f"  {k}: {repr(v[0])[:200]}")
