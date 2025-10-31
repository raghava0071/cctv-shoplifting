#!/usr/bin/env python3
import os, re, pandas as pd, numpy as np

LABELS = "output/labels.csv"

if not os.path.exists(LABELS):
    raise SystemExit(f"ERROR: {LABELS} not found – run your labeling first.")

df = pd.read_csv(LABELS)

def extract_prob(name: str) -> float:
    m = re.search(r"_p([0-9]+\.[0-9]+)(?=\.mp4|_)", str(name))
    return float(m.group(1)) if m else np.nan

df["prob"] = df["file"].apply(extract_prob)
df["label_int"] = df["label"].str.upper().map({"TRUE":1, "FALSE":0})

clean = df.dropna(subset=["prob","label_int"]).copy()
print("Usable rows:", len(clean))
print("TRUE:", (clean["label_int"]==1).sum(), "FALSE:", (clean["label_int"]==0).sum())

rows = []
for thr in np.arange(0.530, 0.585, 0.0025):
    pred = (clean["prob"] >= thr).astype(int)
    tp = int(((pred==1)&(clean["label_int"]==1)).sum())
    fp = int(((pred==1)&(clean["label_int"]==0)).sum())
    tn = int(((pred==0)&(clean["label_int"]==0)).sum())
    fn = int(((pred==0)&(clean["label_int"]==1)).sum())

    prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
    acc  = (tp+tn) / len(clean)
    rows.append(dict(thr=round(thr,4), prec=prec, rec=rec, acc=acc,
                     tp=tp, fp=fp, tn=tn, fn=fn))

res = pd.DataFrame(rows)
res["f1"] = 2*(res["prec"]*res["rec"])/(res["prec"]+res["rec"]+1e-6)

os.makedirs("output", exist_ok=True)
res.to_csv("output/threshold_sweep.csv", index=False)

print("\nTop thresholds (by F1):")
print(res.sort_values("f1", ascending=False).head(10).to_string(index=False))
print("\nSaved to output/threshold_sweep.csv")
