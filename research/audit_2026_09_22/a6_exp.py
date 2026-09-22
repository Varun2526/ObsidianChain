import pandas as pd, numpy as np, joblib, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
U="/mnt/user-data/uploads/obsidianchain/data/models/"; D=U+"ps_native/datasets/"; V=U+"ps_native/v1/"
tr=pd.read_parquet(D+"train.parquet"); va=pd.read_parquet(D+"validation.parquet"); te=pd.read_parquet(D+"test.parquet")
art=joblib.load(V+"model.joblib"); FE=list(art["features"])
STEP=lambda d:((d.timestamp-1400000000)//1209600+1).astype(int)
for d in (tr,va,te): d["step"]=STEP(d)

print("="*78); print("X0. MECHANISM CHECK: output_entropy is a deterministic function of output_count")
al=pd.concat([tr,va,te]); oc=al.output_count.to_numpy(); ent=al.output_entropy.to_numpy()
pred=np.where(oc>1,np.log2(np.maximum(oc,1)),0.0)
print(f"  max|output_entropy - log2(output_count)| = {np.abs(ent-pred).max():.3e}  -> identical: {np.allclose(ent,pred,atol=1e-9)}")
print(f"  is_mixing_candidate == (input_count>=3 & output_count>=3): "
      f"{np.array_equal(al.is_mixing_candidate.to_numpy().astype(int), ((al.input_count>=3)&(al.output_count>=3)).to_numpy().astype(int))}")

DEGEN=["is_peeling_candidate","input_amount_std","output_amount_std","equal_output_count",
       "in_degree_asof_t","out_degree_asof_t","output_entropy"]
KEEP=[f for f in FE if f not in DEGEN]
print(f"\n  degenerate/redundant identified: {len(DEGEN)} -> reduced feature set = {len(KEEP)} of {len(FE)}")

def fit_eval(feats, Xtr, seed=20260919, n_est=100, depth=12):
    m=RandomForestClassifier(n_estimators=n_est,max_depth=depth,random_state=seed,n_jobs=-1)
    m.fit(np.nan_to_num(Xtr[feats].to_numpy(np.float32),nan=0.0), Xtr.y.to_numpy())
    return m
def sc(m,feats,d): return m.predict_proba(np.nan_to_num(d[feats].to_numpy(np.float32),nan=0.0))[:,1]
def pak(y,s,k): k=min(k,len(y)); return float(np.asarray(y)[np.argsort(-s,kind="stable")[:k]].mean())
def row(nm,y,s):
    return dict(config=nm,AP=average_precision_score(y,s),ROC=roc_auc_score(y,s),
                P100=pak(y,s,100),P200=pak(y,s,200),P500=pak(y,s,500))

print("\n"+"="*78); print("X1. FEATURE PRUNING (one variable: feature set). Train=train, eval on val+test, RAW scores")
res=[]
for nm,feats in [("all 30 features",FE),("24 non-degenerate",KEEP)]:
    m=fit_eval(feats,tr)
    for split,d in [("val",va),("test",te)]:
        r=row(f"{nm} / {split}",d.y.to_numpy(),sc(m,feats,d)); res.append(r)
print(pd.DataFrame(res).to_string(index=False,float_format=lambda x:f"{x:.4f}"))

print("\n"+"="*78); print("X2. SEED STABILITY of the frozen configuration (10 seeds, test set, RAW)")
rows=[]
for s in range(10):
    m=fit_eval(FE,tr,seed=1000+s); sco=sc(m,FE,te); y=te.y.to_numpy()
    rows.append(row(f"seed{1000+s}",y,sco))
dfm=pd.DataFrame(rows)
print(dfm.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\n  mean±sd  AP={:.4f}±{:.4f}  ROC={:.4f}±{:.4f}  P@100={:.4f}±{:.4f}  P@200={:.4f}±{:.4f}  P@500={:.4f}±{:.4f}".format(
    dfm.AP.mean(),dfm.AP.std(),dfm.ROC.mean(),dfm.ROC.std(),dfm.P100.mean(),dfm.P100.std(),
    dfm.P200.mean(),dfm.P200.std(),dfm.P500.mean(),dfm.P500.std()))

print("\n"+"="*78); print("X3. BOOTSTRAP 95% CI on TEST for the frozen model (2000 resamples, RAW scores)")
m0=joblib.load(V+"model.joblib")["model"]
s_te=m0.predict_proba(np.nan_to_num(te[FE].to_numpy(np.float32),nan=0.0))[:,1]; y_te=te.y.to_numpy()
rng=np.random.default_rng(7); n=len(y_te); B=2000
aps=np.empty(B); p100=np.empty(B); p200=np.empty(B)
for b in range(B):
    i=rng.integers(0,n,n); yb=y_te[i]; sb=s_te[i]
    if yb.sum()==0: aps[b]=np.nan; p100[b]=np.nan; p200[b]=np.nan; continue
    aps[b]=average_precision_score(yb,sb); p100[b]=pak(yb,sb,100); p200[b]=pak(yb,sb,200)
for nm,arr in [("test AP",aps),("test P@100",p100),("test P@200",p200)]:
    print(f"  {nm:12s} point={np.nanmean(arr):.4f}  95% CI [{np.nanpercentile(arr,2.5):.4f}, {np.nanpercentile(arr,97.5):.4f}]")
