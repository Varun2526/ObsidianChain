import pandas as pd, numpy as np, joblib
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
U="/mnt/user-data/uploads/obsidianchain/data/models/"; D=U+"ps_native/datasets/"; V=U+"ps_native/v1/"
va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
art=joblib.load(V+"model.joblib"); mdl=art["model"]; cal=art["calibrator"]; FE=list(art["features"])
def raw(d):
    X=np.nan_to_num(d[FE].to_numpy(dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    return mdl.predict_proba(X)[:,1]

print("="*78); print("A. DOES 'CALIBRATED' EXPLAIN THE TEST MISMATCH?")
ref_te=dict(pr=0.20792324917190835,roc=0.7367124754721598,br=0.04510267547248221,p100=0.97,p500=0.504)
def pak(y,s,k,kind="stable"):
    k=min(k,len(y)); return float(np.asarray(y)[np.argsort(-s,kind=kind)[:k]].mean())
y=te.y.to_numpy(); r=raw(te); c=cal.predict(r)
for nm,s in [("raw",r),("calibrated",c)]:
    print(f"  test/{nm:11s} PR={average_precision_score(y,s):.10f} ROC={roc_auc_score(y,s):.10f} "
          f"Brier={brier_score_loss(y,s):.10f} P@100={pak(y,s,100):.4f} P@500={pak(y,s,500):.4f}")
print(f"  FROZEN         PR={ref_te['pr']:.10f} ROC={ref_te['roc']:.10f} Brier={ref_te['br']:.10f} "
      f"P@100={ref_te['p100']:.4f} P@500={ref_te['p500']:.4f}")

print("="*78); print("B. SCORE TIE STRUCTURE (raw RF probabilities)")
for nm,d in [("validation",va),("test",te)]:
    s=raw(d); yy=d.y.to_numpy()
    u,cnt=np.unique(s,return_counts=True)
    order=np.argsort(-u); top_val=u[order[0]]; top_n=cnt[order[0]]
    # how many rows share the score of the 100th-ranked item
    srt=np.sort(s)[::-1]; s100=srt[99]
    blk=(s==s100).sum(); above=(s>s100).sum()
    print(f"\n[{nm}] n={len(s):,d}  distinct scores={len(u):,d}")
    print(f"  highest score={top_val:.6f} shared by {top_n:,d} rows (illicit share in that block="
          f"{yy[s==top_val].mean():.4f})")
    print(f"  score at rank100={s100:.6f}: {above:,d} rows strictly above, {blk:,d} rows tied at it "
          f"-> {100-above} of the 100 slots drawn from a tie block of {blk:,d}")

print("="*78); print("C. PRECISION@K UNDER DIFFERENT TIE-BREAKING (the frozen code uses 'stable' = file order)")
rng=np.random.default_rng(0)
def pak_random(y,s,k,n=2000):
    y=np.asarray(y); out=np.empty(n)
    for i in range(n):
        j=rng.permutation(len(s))
        out[i]=y[j][np.argsort(-s[j],kind="stable")[:k]].mean()
    return out
def pak_best(y,s,k):   # adversarial best: positives first within ties
    y=np.asarray(y); o=np.lexsort((-y,-s)); return float(y[o[:k]].mean())
def pak_worst(y,s,k):
    y=np.asarray(y); o=np.lexsort((y,-s)); return float(y[o[:k]].mean())
rows=[]
for nm,d in [("validation",va),("test",te)]:
    s=raw(d); yy=d.y.to_numpy()
    for k in (10,50,100,500):
        rs=pak_random(yy,s,k)
        rows.append(dict(split=nm,k=k,stable=pak(yy,s,k),rand_mean=rs.mean(),rand_lo=np.percentile(rs,2.5),
                         rand_hi=np.percentile(rs,97.5),best=pak_best(yy,s,k),worst=pak_worst(yy,s,k)))
df=pd.DataFrame(rows)
print(df.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
