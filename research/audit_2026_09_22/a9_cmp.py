import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
D="/mnt/user-data/uploads/obsidianchain/data/models/ps_native/datasets/"
tr=pd.read_parquet(D+"train.parquet"); va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
import joblib; FE=list(joblib.load("/mnt/user-data/uploads/obsidianchain/data/models/ps_native/v1/model.joblib")["features"])
Xtr=tr[FE].to_numpy(np.float32); ytr=tr.y.to_numpy(np.int8)
Xva=va[FE].to_numpy(np.float32); yva=va.y.to_numpy(np.int8)
Xte=te[FE].to_numpy(np.float32); yte=te.y.to_numpy(np.int8)
def pak(y,s,k): return float(np.asarray(y)[np.argsort(-s)[:k]].mean())
def mk(name,seed):
    if name=="LogisticRegression":
        return Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=1000,random_state=seed,class_weight="balanced"))])
    if name=="RandomForest":
        return RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=20,random_state=seed,n_jobs=-1)
    return LGBMClassifier(n_estimators=250,learning_rate=0.05,num_leaves=31,min_child_samples=30,
                          subsample=0.8,colsample_bytree=0.8,random_state=seed,n_jobs=-1,verbose=-1)
rows=[]
plan=[("LogisticRegression",[20260919]),("RandomForest",[20260919,1,2,3,4]),("LightGBM",[20260919,1,2,3,4])]
for name,seeds in plan:
    for sd in seeds:
        t=time.time(); mm=mk(name,sd).fit(Xtr,ytr); ft=time.time()-t
        sv=mm.predict_proba(Xva)[:,1]; st=mm.predict_proba(Xte)[:,1]
        rows.append(dict(model=name,seed=sd,fit_s=round(ft,2),
            val_AP=average_precision_score(yva,sv),val_P100=pak(yva,sv,100),
            test_AP=average_precision_score(yte,st),test_ROC=roc_auc_score(yte,st),
            test_P100=pak(yte,st,100),test_P200=pak(yte,st,200),test_P500=pak(yte,st,500)))
        print(f"  done {name} seed={sd} in {ft:.1f}s",flush=True)
df=pd.DataFrame(rows); df.to_csv("model_seeds.csv",index=False)
print("\n"+"="*78); print("PER-RUN RESULTS"); print(df.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\n"+"="*78); print("MEAN +/- SD ACROSS SEEDS")
g=df.groupby("model")[["val_AP","test_AP","test_ROC","test_P100","test_P200","test_P500","fit_s"]].agg(["mean","std"])
print(g.to_string(float_format=lambda x:f"{x:.4f}"))
print("\nRepo selected RandomForest on a SINGLE seed by val_AP.")
r=df[df.model=="RandomForest"].val_AP; l=df[df.model=="LightGBM"].val_AP
print(f"  val_AP  RF {r.mean():.4f}+/-{r.std():.4f}   LGBM {l.mean():.4f}+/-{l.std():.4f}   gap={r.mean()-l.mean():+.4f}")
rt=df[df.model=="RandomForest"].test_AP; lt=df[df.model=="LightGBM"].test_AP
print(f"  test_AP RF {rt.mean():.4f}+/-{rt.std():.4f}   LGBM {lt.mean():.4f}+/-{lt.std():.4f}   gap={rt.mean()-lt.mean():+.4f}")
r2=df[df.model=="RandomForest"].test_P200; l2=df[df.model=="LightGBM"].test_P200
print(f"  test_P@200 RF {r2.mean():.4f}+/-{r2.std():.4f}  LGBM {l2.mean():.4f}+/-{l2.std():.4f}  gap={r2.mean()-l2.mean():+.4f}")
