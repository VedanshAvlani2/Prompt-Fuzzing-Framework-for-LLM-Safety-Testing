from pathlib import Path
import csv

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def read_seeds_csv(path):
    if not Path(path).exists():
        print("⚠️ No seeds.csv found!"); return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if row: out.append(row[0])
    return out
