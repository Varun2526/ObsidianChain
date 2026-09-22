import pandas as pd, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
U="/mnt/user-data/uploads/obsidianchain/data/models/"; D=U+"ps_native/datasets/"; V=U+"ps_native/v1/"
tr=pd.read_parquet(D+"train.parquet"); va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
art=joblib.load(V+"model.joblib"); M=art["model"]; FE=list(art["features"])
def X(d): return np.nan_to_num(d[FE].to_numpy(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
def sc(m,d): return m.predict_proba(X(d))[:,1]
def ap(y,s): return average_precision_score(y,s)

print("="*78); print("H: WAS THE FROZEN MODEL TRAINED ON THE VALIDATION SPLIT?")
print("Frozen model reported val AP = 0.5702, test AP(raw) = 0.2182\n")
cfg=dict(n_estimators=100,max_depth=12,random_state=20260919,n_jobs=-1)
rows=[]
for nm,trn in [("train only",tr),("train+validation",pd.concat([tr,va],ignore_index=True))]:
    m=RandomForestClassifier(**cfg).fit(X(trn),trn.y.to_numpy())
    rows.append(dict(trained_on=nm,
                     val_AP=ap(va.y,sc(m,va)), val_ROC=roc_auc_score(va.y,sc(m,va)),
                     test_AP=ap(te.y,sc(m,te)), test_ROC=roc_auc_score(te.y,sc(m,te))))
rows.append(dict(trained_on="FROZEN ARTIFACT",val_AP=ap(va.y,sc(M,va)),val_ROC=roc_auc_score(va.y,sc(M,va)),
                 test_AP=ap(te.y,sc(M,te)),test_ROC=roc_auc_score(te.y,sc(M,te))))
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

print("\n"+"="*78); print("DECISIVE TEST: memorisation fingerprint — out-of-bag vs in-bag behaviour")
print("A tree ensemble that saw a row predicts it far more confidently than an unseen row.")
print("Compare the frozen model's score distribution on TRAIN (seen) vs VAL vs TEST:\n")
for nm,d in [("train",tr),("validation",va),("test",te)]:
    s=sc(M,d); y=d.y.to_numpy()
    print(f"  {nm:11s} AP={ap(y,s):.4f}  mean_score_pos={s[y==1].mean():.4f}  mean_score_neg={s[y==0].mean():.4f}  "
          f"frac_pos_scored>0.9={np.mean(s[y==1]>0.9):.4f}")

print("\n"+"="*78); print("CONTROL: same stats for a model we KNOW trained on train only")
m_tr=RandomForestClassifier(**cfg).fit(X(tr),tr.y.to_numpy())
for nm,d in [("train",tr),("validation",va),("test",te)]:
    s=sc(m_tr,d); y=d.y.to_numpy()
    print(f"  {nm:11s} AP={ap(y,s):.4f}  mean_score_pos={s[y==1].mean():.4f}  mean_score_neg={s[y==0].mean():.4f}  "
          f"frac_pos_scored>0.9={np.mean(s[y==1]>0.9):.4f}")

print("\n"+"="*78); print("CONTROL: model we KNOW trained on train+validation")
m_tv=RandomForestClassifier(**cfg).fit(X(pd.concat([tr,va],ignore_index=True)),pd.concat([tr,va]).y.to_numpy())
for nm,d in [("train",tr),("validation",va),("test",te)]:
    s=sc(m_tv,d); y=d.y.to_numpy()
    print(f"  {nm:11s} AP={ap(y,s):.4f}  mean_score_pos={s[y==1].mean():.4f}  mean_score_neg={s[y==0].mean():.4f}  "
          f"frac_pos_scored>0.9={np.mean(s[y==1]>0.9):.4f}")
