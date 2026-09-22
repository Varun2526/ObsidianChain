import pandas as pd, numpy as np, json
pd.set_option('display.width', 200)
D="/mnt/user-data/uploads/obsidianchain/data/models/ps_native/datasets/"
tr=pd.read_parquet(D+"train.parquet"); va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
man=json.load(open(D+"manifest.json")); FEATS=man["feature_list"]

print("="*78); print("SHAPES / PREVALENCE (vs manifest)")
for n,d in [("train",tr),("validation",va),("test",te)]:
    print(f"{n:11s} rows={len(d):7,d} pos={int(d.y.sum()):6,d} prev={d.y.mean():.5f}  manifest_rows={man['row_counts'][n]:,d} pos={man['positive_counts'][n]:,d}")
print("columns:", list(tr.columns))

print("="*78); print("DUPLICATES / OVERLAP")
for n,d in [("train",tr),("validation",va),("test",te)]:
    print(f"{n:11s} unique addresses={d.address.nunique():,d} / {len(d):,d}  dup_rows={len(d)-d.address.nunique():,d}  dup_full={int(d.duplicated().sum()):,d}")
A,B,C=set(tr.address),set(va.address),set(te.address)
print(f"address overlap  train&val={len(A&B)}  train&test={len(A&C)}  val&test={len(B&C)}")
print(f"txid overlap     train&val={len(set(tr.txid)&set(va.txid))}  train&test={len(set(tr.txid)&set(te.txid))}  val&test={len(set(va.txid)&set(te.txid))}")

print("="*78); print("TIMESTAMP COVERAGE (surrogate t -> step)")
STEP=lambda ts:((ts-1400000000)//1209600+1)
for n,d in [("train",tr),("validation",va),("test",te)]:
    s=STEP(d.timestamp); print(f"{n:11s} step min={s.min()} max={s.max()} n_steps={s.nunique()}  distinct_timestamps={d.timestamp.nunique()}")

print("="*78); print("FEATURE DEGENERACY (all splits pooled)")
al=pd.concat([tr,va,te],ignore_index=True)
rows=[]
for f in FEATS:
    v=al[f]; nun=v.nunique(dropna=True)
    rows.append(dict(feature=f, n_unique=nun, pct_nan=100*v.isna().mean(), pct_zero=100*(v==0).mean(),
                     const=(nun<=1), mn=v.min(), mx=v.max()))
deg=pd.DataFrame(rows)
print(deg.to_string(index=False, float_format=lambda x:f"{x:.4g}"))
print("\nCONSTANT FEATURES:", list(deg[deg.const].feature) or "none")
print("NEAR-CONSTANT (<=2 unique):", list(deg[deg.n_unique<=2].feature))

print("="*78); print("EXACT DUPLICATE / COLLINEAR FEATURE PAIRS (pooled, identical values)")
import itertools
vals={f:al[f].fillna(-99999).to_numpy() for f in FEATS}
dups=[]
for a,b in itertools.combinations(FEATS,2):
    if np.array_equal(vals[a],vals[b]): dups.append((a,b))
print("identical pairs:", dups or "none")
