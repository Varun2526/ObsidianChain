import pandas as pd, numpy as np, joblib, hashlib, json
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, f1_score
U="/mnt/user-data/uploads/obsidianchain/data/models/"
D=U+"ps_native/datasets/"; V=U+"ps_native/v1/"
tr=pd.read_parquet(D+"train.parquet"); va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
man=json.load(open(V+"feature_schema.json")) if False else None
mj=V+"model.joblib"
sha=hashlib.sha256(open(mj,'rb').read()).hexdigest()
print("model sha256:", sha)
print("matches manifest 7d9c1b40...:", sha=="7d9c1b404872fd1b05ecb4ff99648904c89f6f5ee1502e6c2915bc6f4d7ad65d")
art=joblib.load(mj)
print("artifact keys:", list(art.keys()))
mdl=art["model"]; cal=art["calibrator"]; FE=list(art["features"])
print("model:", type(mdl).__name__, "n_estimators",getattr(mdl,'n_estimators',None), "max_depth",getattr(mdl,'max_depth',None),
      "class_weight",getattr(mdl,'class_weight',None),"random_state",getattr(mdl,'random_state',None))
print("calibrator:", type(cal).__name__)
print("n features:", len(FE))

def raw(d):
    X=d[FE].to_numpy(dtype=np.float32); X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
    return mdl.predict_proba(X)[:,1]

def pak_stable(y,s,k):
    k=min(k,len(y)); top=np.argsort(-s,kind="stable")[:k]; return float(np.asarray(y)[top].mean())

print("="*78); print("REPRODUCING metrics.json (RAW probabilities, as the file's brier implies)")
ref={"validation":{"pr":0.5701972891343364,"roc":0.8606422389649213,"br":0.04186015888399271,"p100":0.99,"p500":0.944},
     "test":{"pr":0.20792324917190835,"roc":0.7367124754721598,"br":0.04510267547248221,"p100":0.97,"p500":0.504}}
for name,d in [("validation",va),("test",te)]:
    y=d.y.to_numpy(); s=raw(d)
    pr=average_precision_score(y,s); roc=roc_auc_score(y,s); br=brier_score_loss(y,s)
    p100=pak_stable(y,s,100); p500=pak_stable(y,s,500)
    r=ref[name]
    print(f"\n[{name}]  n={len(y):,d} prev={y.mean():.5f}")
    print(f"  PR-AUC   repro={pr:.10f}  frozen={r['pr']:.10f}  match={abs(pr-r['pr'])<1e-9}")
    print(f"  ROC-AUC  repro={roc:.10f}  frozen={r['roc']:.10f}  match={abs(roc-r['roc'])<1e-9}")
    print(f"  Brier    repro={br:.10f}  frozen={r['br']:.10f}  match={abs(br-r['br'])<1e-9}")
    print(f"  P@100    repro={p100:.4f}  frozen={r['p100']:.4f}  match={abs(p100-r['p100'])<1e-9}")
    print(f"  P@500    repro={p500:.4f}  frozen={r['p500']:.4f}  match={abs(p500-r['p500'])<1e-9}")
