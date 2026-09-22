import pandas as pd, numpy as np, joblib, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
U="/mnt/user-data/uploads/obsidianchain/data/models/"; D=U+"ps_native/datasets/"; V=U+"ps_native/v1/"
tr=pd.read_parquet(D+"train.parquet"); va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
art=joblib.load(V+"model.joblib"); FE=list(art["features"]); MF=art["model"]
Xtr=tr[FE].to_numpy(np.float32); ytr=tr.y.to_numpy(np.int8)
Xva=va[FE].to_numpy(np.float32); yva=va.y.to_numpy(np.int8)
Xte=te[FE].to_numpy(np.float32); yte=te.y.to_numpy(np.int8)
def pak(y,s,k): return float(np.asarray(y)[np.argsort(-s)[:k]].mean())

print("="*78); print("EXACT REPRODUCTION with the TRAINING SCRIPT's config (min_samples_leaf=20 included)")
m=RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=20,random_state=20260919,n_jobs=-1).fit(Xtr,ytr)
sv=m.predict_proba(Xva)[:,1]; sf=MF.predict_proba(Xva)[:,1]
print(f"  refit val AP  = {average_precision_score(yva,sv):.10f}")
print(f"  frozen val AP = {average_precision_score(yva,sf):.10f}")
print(f"  identical score vectors: {np.allclose(sv,sf)}  max|diff|={np.abs(sv-sf).max():.2e}")
print("  -> frozen model IS reproducible from train.parquet + train_ps_model.py config")
print("  -> docs/ML_PIPELINE.md config (120 trees/depth14/balanced_subsample/seed42) is NOT the shipped model")

print("\n"+"="*78); print("MODEL SELECTION UNDER SEED NOISE (validation, RAW scores, as the repo selects)")
print("repo used ONE seed (20260919) and picked RF over LightGBM by +0.0474 val PR-AUC\n")
def mk(name,seed):
    if name=="LogisticRegression":
        return Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=1000,random_state=seed,class_weight="balanced"))])
    if name=="RandomForest":
        return RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=20,random_state=seed,n_jobs=-1)
    return LGBMClassifier(n_estimators=250,learning_rate=0.05,num_leaves=31,min_child_samples=30,
                          subsample=0.8,colsample_bytree=0.8,random_state=seed,n_jobs=-1,verbose=-1)
rows=[]
SEEDS=[20260919,1,2,3,4,5,6,7]
for name in ["LogisticRegression","RandomForest","LightGBM"]:
    for sd in SEEDS:
        t=time.time(); mm=mk(name,sd).fit(Xtr,ytr); ft=time.time()-t
        s_v=mm.predict_proba(Xva)[:,1]; s_t=mm.predict_proba(Xte)[:,1]
        rows.append(dict(model=name,seed=sd,fit_s=round(ft,2),
                         val_AP=average_precision_score(yva,s_v),val_ROC=roc_auc_score(yva,s_v),
                         val_P100=pak(yva,s_v,100),
                         test_AP=average_precision_score(yte,s_t),test_ROC=roc_auc_score(yte,s_t),
                         test_P100=pak(yte,s_t,100),test_P200=pak(yte,s_t,200),test_P500=pak(yte,s_t,500)))
df=pd.DataFrame(rows); df.to_csv("model_seeds.csv",index=False)
agg=df.groupby("model").agg(["mean","std"])[["val_AP","test_AP","test_ROC","test_P100","test_P200","test_P500"]]
print(df.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\nMEAN +/- SD ACROSS 8 SEEDS:")
print(agg.to_string(float_format=lambda x:f"{x:.4f}"))
