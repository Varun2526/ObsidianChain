import pandas as pd, numpy as np, warnings, joblib
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
D="/mnt/user-data/uploads/obsidianchain/data/models/ps_native/datasets/"
tr=pd.read_parquet(D+"train.parquet"); va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
FE=list(joblib.load("/mnt/user-data/uploads/obsidianchain/data/models/ps_native/v1/model.joblib")["features"])
STEP=lambda d:((d.timestamp-1400000000)//1209600+1).astype(int)
for d in (tr,va,te): d["step"]=STEP(d)
Xtr=tr[FE].to_numpy(np.float32); ytr=tr.y.to_numpy(np.int8)
Xte=te[FE].to_numpy(np.float32); yte=te.y.to_numpy(np.int8)
def mk(n,s=20260919):
    if n=="LR": return Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=1000,random_state=s,class_weight="balanced"))])
    if n=="RF": return RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=20,random_state=s,n_jobs=-1)
    return LGBMClassifier(n_estimators=250,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,
                          colsample_bytree=0.8,random_state=s,n_jobs=-1,verbose=-1)
S={n:mk(n).fit(Xtr,ytr).predict_proba(Xte)[:,1] for n in ("LR","RF","LGBM")}

print("="*78); print("PAIRED BOOTSTRAP on TEST (800 resamples): is LR's advantage real?")
rng=np.random.default_rng(11); n=len(yte); B=800
acc={k:[] for k in S}; d_lr_rf=[]; d_lr_lgbm=[]
for b in range(B):
    i=rng.integers(0,n,n); yb=yte[i]
    if yb.sum()<5: continue
    a={k:average_precision_score(yb,S[k][i]) for k in S}
    for k in S: acc[k].append(a[k])
    d_lr_rf.append(a["LR"]-a["RF"]); d_lr_lgbm.append(a["LR"]-a["LGBM"])
for k in S:
    v=np.array(acc[k]); print(f"  {k:5s} test AP  {v.mean():.4f}  95% CI [{np.percentile(v,2.5):.4f}, {np.percentile(v,97.5):.4f}]")
for nm,d in [("LR - RF",np.array(d_lr_rf)),("LR - LightGBM",np.array(d_lr_lgbm))]:
    print(f"  delta {nm:14s} = {d.mean():+.4f}  95% CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  "
          f"P(LR better)={np.mean(d>0):.3f}")

print("\n"+"="*78); print("CORRECTED SELECTION PROTOCOL: inner chronological split of TRAIN ONLY")
print("(inner-train steps 1-27, inner-val steps 28-34). Test is NOT consulted for selection.\n")
itr=tr[tr.step<=27]; iva=tr[tr.step>27]
Xi=itr[FE].to_numpy(np.float32); yi=itr.y.to_numpy(np.int8)
Xv=iva[FE].to_numpy(np.float32); yv=iva.y.to_numpy(np.int8)
print(f"  inner-train n={len(itr):,d} prev={yi.mean():.4f} | inner-val n={len(iva):,d} prev={yv.mean():.4f}")
sel=[]
for k in ("LR","RF","LGBM"):
    mm=mk(k).fit(Xi,yi); sv=mm.predict_proba(Xv)[:,1]
    sel.append(dict(model=k,inner_val_AP=average_precision_score(yv,sv),inner_val_ROC=roc_auc_score(yv,sv),
                    held_out_test_AP=average_precision_score(yte,S[k]),held_out_test_ROC=roc_auc_score(yte,S[k])))
sdf=pd.DataFrame(sel); print(sdf.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print(f"\n  inner-val would select : {sdf.loc[sdf.inner_val_AP.idxmax(),'model']}")
print(f"  repo's validation split selected : RandomForest")
print(f"  best on held-out TEST  : {sdf.loc[sdf.held_out_test_AP.idxmax(),'model']}")

print("\n"+"="*78); print("RANK CORRELATION: does validation AP predict test AP across families?")
vm={"LR":0.2665,"RF":0.5549,"LGBM":0.5328}; tm={k:average_precision_score(yte,S[k]) for k in S}
from scipy.stats import spearmanr
ks=list(vm); print("  val AP :",{k:round(vm[k],4) for k in ks}); print("  test AP:",{k:round(tm[k],4) for k in ks})
print(f"  Spearman(val,test) across 3 families = {spearmanr([vm[k] for k in ks],[tm[k] for k in ks]).statistic:+.2f}")
