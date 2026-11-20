"""Writes sid_file_map.csv: the participant index (sid) used in all result files
and the .mat file it belongs to.

sid is the position of the file in the sorted list of .mat files of the dataset
(the same order and the same control/ADHD rule as find_files() in
full-pipeline.py).  Usage:  python make_sid_file_map.py <dataset_folder> [out.csv]
"""
import glob, os, sys
import pandas as pd

data_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATASET_PATH")
out_csv = sys.argv[2] if len(sys.argv) > 2 else "sid_file_map.csv"
assert data_dir, "give the dataset folder as the first argument or set DATASET_PATH"

rows = []
for f in sorted(glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)):
    rel = os.path.relpath(f, data_dir)
    low = rel.lower()
    if "control" in low:
        label = 0
    elif "adhd" in low:
        label = 1
    else:
        continue
    rows.append({"sid": len(rows), "label": label, "file": rel})
df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print(len(df), "participants;", int((df.label == 1).sum()), "ADHD,", int((df.label == 0).sum()), "control ->", out_csv)
