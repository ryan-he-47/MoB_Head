import glob
import sys

import pyarrow.parquet as pq

p = sorted(glob.glob("/gpfs1/scratch/yihe47/qwen35/data/stack_v3/**/*.parquet", recursive=True))[0]
pf = pq.ParquetFile(p)
print("file:", p.split("/")[-1])
print("schema.names:", pf.schema.names)
print("arrow schema:")
print(pf.schema_arrow)
b = next(pf.iter_batches(batch_size=2))
print("batch keys:", list(b.to_pydict().keys()))
