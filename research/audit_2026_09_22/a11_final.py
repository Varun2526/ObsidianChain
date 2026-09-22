import pandas as pd, numpy as np, warnings, joblib
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
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
fits={n:mk(n).fit(Xtr,ytr) for n in ("LR","RF","LGBM")}
S={n:fits[n].predict_proba(Xte)[:,1] for n in fits}
def pak(y,s,k):
    k=min(k,len(y)); return float(np.asarray(y)[np.argsort(-s,kind="stable")[:k]].mean())

print("="*78); print("E1. MECHANISM: performance vs FORWARD HORIZON (test timesteps 42..49)")
print("If trees exploit window-local structure, they should decay faster with horizon.\n")
rows=[]
for st in sorted(te.step.unique()):
    m=(te.step==st).to_numpy(); y=yte[m]
    if y.sum()<5: continue
    r=dict(step=int(st),n=int(m.sum()),prev=float(y.mean()))
    for k in S: r[f"{k}_AP"]=average_precision_score(y,S[k][m])
    for k in S: r[f"{k}_nAP"]=(r[f"{k}_AP"]-r["prev"])/(1-r["prev"])
    rows.append(r)
hz=pd.DataFrame(rows); print(hz.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\n  nAP decay (first 2 test steps -> last 2 test steps):")
for k in S:
    a=hz[f"{k}_nAP"].iloc[:2].mean(); b=hz[f"{k}_nAP"].iloc[-2:].mean()
    print(f"    {k:5s} {a:.4f} -> {b:.4f}   retained={100*b/a if a>0 else float('nan'):.1f}%")

print("\n"+"="*78); print("E2. INVESTIGATIVE QUEUE: precision@K on TEST (the operational question)")
ks=[25,50,100,200,300,500,1000,2000]
q=pd.DataFrame({"K":ks})
for k in S: q[k]= [pak(yte,S[k],x) for x in ks]
q["RF_recall"]=[pak(yte,S["RF"],x)*x/yte.sum() for x in ks]
q["LR_recall"]=[pak(yte,S["LR"],x)*x/yte.sum() for x in ks]
print(q.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

print("\n"+"="*78); print("E3. DOES AN ENSEMBLE HELP? (mean of rank-normalised scores) - test before adopting")
from scipy.stats import rankdata
ens=np.mean([rankdata(S[k])/len(yte) for k in ("LR","RF","LGBM")],axis=0)
ens2=np.mean([rankdata(S[k])/len(yte) for k in ("LR","LGBM")],axis=0)
for nm,s in [("LR alone",S["LR"]),("LR+LGBM rank-mean",ens2),("LR+RF+LGBM rank-mean",ens)]:
    print(f"  {nm:22s} AP={average_precision_score(yte,s):.4f} ROC={roc_auc_score(yte,s):.4f} "
          f"P@200={pak(yte,s,200):.4f} P@500={pak(yte,s,500):.4f}")

print("\n"+"="*78); print("E4. TRAINING ON train+validation (more data, closer in time) - one variable")
trv=pd.concat([tr,va],ignore_index=True); Xv=trv[FE].to_numpy(np.float32); yv=trv.y.to_numpy(np.int8)
for k in ("LR","RF","LGBM"):
    m2=mk(k).fit(Xv,yv); s2=m2.predict_proba(Xte)[:,1]
    print(f"  {k:5s} train-only AP={average_precision_score(yte,S[k]):.4f} -> train+val AP={average_precision_score(yte,s2):.4f} "
          f"| P@200 {pak(yte,S[k],200):.4f} -> {pak(yte,s2,200):.4f}")

print("\n"+"="*78); print("E5. CALIBRATION: isotonic-on-validation, applied to TEST (Brier + score resolution)")
Xva=va[FE].to_numpy(np.float32); yva=va.y.to_numpy(np.int8)
for k in ("LR","RF","LGBM"):
    rv=fits[k].predict_proba(Xva)[:,1]; cal=IsotonicRegression(out_of_bounds="clip").fit(rv,yva)
    raw=S[k]; c=cal.predict(raw)
    print(f"  {k:5s} raw  Brier={brier_score_loss(yte,raw):.5f} AP={average_precision_score(yte,raw):.4f} distinct={len(np.unique(raw)):,d}")
    print(f"        cal  Brier={brier_score_loss(yte,c):.5f} AP={average_precision_score(yte,c):.4f} distinct={len(np.unique(c)):,d}")

print("\n"+"="*78); print("E6. FEATURE PRUNING applied to LR (7 degenerate/redundant removed)")
DEG=["is_peeling_candidate","input_amount_std","output_amount_std","equal_output_count",
     "in_degree_asof_t","out_degree_asof_t","output_entropy"]
K2=[f for f in FE if f not in DEG]
for nm,ff in [("all 30",FE),("23 pruned",K2)]:
    m3=mk("LR").fit(tr[ff].to_numpy(np.float32),ytr); s3=m3.predict_proba(te[ff].to_numpy(np.float32))[:,1]
    print(f"  LR / {nm:10s} test AP={average_precision_score(yte,s3):.4f} ROC={roc_auc_score(yte,s3):.4f} "
          f"P@200={pak(yte,s3,200):.4f} P@500={pak(yte,s3,500):.4f}")
