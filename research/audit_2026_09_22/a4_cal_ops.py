import pandas as pd, numpy as np, joblib
from sklearn.metrics import average_precision_score
U="/mnt/user-data/uploads/obsidianchain/data/models/"; D=U+"ps_native/datasets/"; V=U+"ps_native/v1/"
va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
art=joblib.load(V+"model.joblib"); mdl=art["model"]; cal=art["calibrator"]; FE=list(art["features"])
BANDS=art["severity_bands"]
def raw(d):
    X=np.nan_to_num(d[FE].to_numpy(dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    return mdl.predict_proba(X)[:,1]

print("="*78); print("A. TIE STRUCTURE OF **CALIBRATED** SCORES (what production actually ranks on)")
rng=np.random.default_rng(0)
def pak_stable(y,s,k): return float(np.asarray(y)[np.argsort(-s,kind="stable")[:k]].mean())
def pak_rand(y,s,k,n=2000):
    y=np.asarray(y); o=np.empty(n)
    for i in range(n):
        j=rng.permutation(len(s)); o[i]=y[j][np.argsort(-s[j],kind="stable")[:k]].mean()
    return o
def pak_best(y,s,k):
    y=np.asarray(y); return float(y[np.lexsort((-y,-s))[:k]].mean())
def pak_worst(y,s,k):
    y=np.asarray(y); return float(y[np.lexsort((y,-s))[:k]].mean())
rows=[]
for nm,d in [("validation",va),("test",te)]:
    c=cal.predict(raw(d)); yy=d.y.to_numpy()
    u=np.unique(c); srt=np.sort(c)[::-1]
    print(f"\n[{nm}] distinct CALIBRATED scores = {len(u):,d}  (raw had ~{len(np.unique(raw(d))):,d})")
    for k in (100,500):
        s100=srt[k-1]; above=(c>s100).sum(); blk=(c==s100).sum()
        print(f"  rank{k}: {above:,d} strictly above, tie block at cutoff = {blk:,d} rows "
              f"-> {k-above} of {k} slots drawn from that block")
    for k in (10,50,100,500,1000):
        r=pak_rand(yy,c,k)
        rows.append(dict(split=nm,k=k,stable=pak_stable(yy,c,k),rand_mean=r.mean(),
                         rand_lo=np.percentile(r,2.5),rand_hi=np.percentile(r,97.5),
                         best=pak_best(yy,c,k),worst=pak_worst(yy,c,k)))
print("\nP@K on CALIBRATED scores under different tie-breaking:")
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

print("\n"+"="*78); print("B. SEVERITY BANDS: fitted on validation -> what do they deliver on TEST?")
print("bands:",[(b['band'],round(b['threshold'],4)) for b in BANDS])
for nm,d in [("validation",va),("test",te)]:
    c=cal.predict(raw(d)); yy=d.y.to_numpy()
    print(f"\n[{nm}] n={len(yy):,d} prev={yy.mean():.4f}")
    prev_thr=1.1
    for b in BANDS:
        t=b["threshold"]; m=(c>=t)&(c<prev_thr)
        cum=(c>=t)
        print(f"  {b['band']:9s} thr>={t:.4f}  band_n={m.sum():6,d} band_prec={yy[m].mean() if m.sum() else float('nan'):.4f} | "
              f"cumulative_n={cum.sum():6,d} cum_prec={yy[cum].mean() if cum.sum() else float('nan'):.4f} "
              f"cum_recall={yy[cum].sum()/yy.sum():.4f}  (target {b['target_precision']:.2f})")
        prev_thr=t
    m=c<prev_thr; print(f"  {'LOW':9s} thr< {prev_thr:.4f}  band_n={m.sum():6,d} band_prec={yy[m].mean():.4f}")

print("\n"+"="*78); print("C. PRECISION@K CURVE ON TEST (calibrated) - how deep does the good top go?")
c=cal.predict(raw(te)); yy=te.y.to_numpy(); o=np.argsort(-c,kind="stable")
ys=yy[o]; cum=np.cumsum(ys)
for k in (10,25,50,100,200,300,500,750,1000,2000,2518,5000,10000):
    if k<=len(ys):
        print(f"  P@{k:<6,d}={cum[k-1]/k:.4f}   R@{k:<6,d}={cum[k-1]/yy.sum():.4f}   (total positives={int(yy.sum()):,d})")
