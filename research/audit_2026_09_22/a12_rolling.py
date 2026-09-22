import pandas as pd, numpy as np, warnings, joblib
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
D="/mnt/user-data/uploads/obsidianchain/data/models/ps_native/datasets/"
FE=list(joblib.load("/mnt/user-data/uploads/obsidianchain/data/models/ps_native/v1/model.joblib")["features"])
al=pd.concat([pd.read_parquet(D+f) for f in ("train.parquet","validation.parquet","test.parquet")],ignore_index=True)
al["step"]=((al.timestamp-1400000000)//1209600+1).astype(int)
print(f"pooled n={len(al):,d}  steps {al.step.min()}..{al.step.max()}  prevalence={al.y.mean():.4f}")
print("CAVEAT: the original build dropped addresses spanning the 34/41 boundaries, so those")
print("addresses are absent from this pool. Folds are therefore comparable to each other,")
print("but not a clean re-derivation from raw Elliptic++.\n")
def mk(n,s=20260919):
    if n=="LR": return Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=1000,random_state=s,class_weight="balanced"))])
    if n=="RF": return RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=20,random_state=s,n_jobs=-1)
    return LGBMClassifier(n_estimators=250,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,
                          colsample_bytree=0.8,random_state=s,n_jobs=-1,verbose=-1)
def pak(y,s,k):
    k=min(k,len(y)); return float(np.asarray(y)[np.argsort(-s,kind="stable")[:k]].mean())
print("="*78); print("ROLLING-ORIGIN EVALUATION: train on steps<=k, evaluate on steps k+1..k+4")
rows=[]
for k in (20,24,28,32,36,40,44):
    trn=al[al.step<=k]; tst=al[(al.step>k)&(al.step<=k+4)]
    if len(tst)<500 or tst.y.sum()<20: continue
    Xa=trn[FE].to_numpy(np.float32); ya=trn.y.to_numpy(np.int8)
    Xb=tst[FE].to_numpy(np.float32); yb=tst.y.to_numpy(np.int8)
    r=dict(fold=f"<={k} -> {k+1}-{k+4}",n_tr=len(trn),n_te=len(tst),prev=float(yb.mean()))
    for m in ("LR","RF","LGBM"):
        s=mk(m).fit(Xa,ya).predict_proba(Xb)[:,1]
        ap=average_precision_score(yb,s)
        r[f"{m}_nAP"]=(ap-r["prev"])/(1-r["prev"]); r[f"{m}_P200"]=pak(yb,s,200)
    rows.append(r); print("  fold done:",r["fold"],flush=True)
df=pd.DataFrame(rows); df.to_csv("rolling.csv",index=False)
print("\nNORMALISED AP (prevalence-controlled) PER FOLD")
print(df[["fold","n_te","prev","LR_nAP","RF_nAP","LGBM_nAP"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\nPRECISION@200 PER FOLD")
print(df[["fold","prev","LR_P200","RF_P200","LGBM_P200"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\nSUMMARY ACROSS FOLDS")
for m in ("LR","RF","LGBM"):
    a=df[f"{m}_nAP"]; p=df[f"{m}_P200"]
    print(f"  {m:5s} nAP mean={a.mean():.4f} sd={a.std():.4f} median={a.median():.4f} | P@200 mean={p.mean():.4f} sd={p.std():.4f}")
wins={m:int((df[[f'{x}_nAP' for x in ('LR','RF','LGBM')]].idxmax(axis=1)==f"{m}_nAP").sum()) for m in ("LR","RF","LGBM")}
print(f"  folds won on nAP: {wins}  (n_folds={len(df)})")
