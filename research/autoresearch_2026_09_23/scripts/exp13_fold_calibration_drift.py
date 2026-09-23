import sys, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from obsidianchain.ml.protocol import rolling_origin_folds, normalised_average_precision
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
P='data/models/ps_native/datasets/'
d=pd.concat([pd.read_parquet(P+'train.parquet'),pd.read_parquet(P+'validation.parquet')],ignore_index=True)
d['t']=((d.timestamp-1400000000)//1209600+1).astype(int)
F=[c for c in d.columns if c not in('address','txid','timestamp','y','t')]
def ece(y,p,n=10):
    q=np.quantile(p,np.linspace(0,1,n+1)); b=np.clip(np.searchsorted(q,p,side='right')-1,0,n-1)
    return sum(abs(y[b==i].mean()-p[b==i].mean())*(b==i).mean() for i in range(n) if (b==i).any())
def nap_w(y,s,w):
    from sklearn.metrics import average_precision_score
    pr=np.average(y,weights=w); return (average_precision_score(y,s,sample_weight=w)-pr)/(1-pr)
rows=[]
for f in rolling_origin_folds():
    tr=d[d.t<=f.train_end]; ev=d[(d.t>=f.eval_start)&(d.t<=f.eval_end)]
    cal=tr[tr.t>f.train_end-2]; fit=tr[tr.t<=f.train_end-2]   # temporal calibration split
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=31,min_child_samples=50,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1,random_state=0).fit(fit[F],fit.y)
    s=m.predict_proba(ev[F])[:,1]; sc=m.predict_proba(cal[F])[:,1]; y=ev.y.values
    iso=IsotonicRegression(out_of_bounds='clip').fit(sc,cal.y); pl=LogisticRegression().fit(np.log(sc/(1-sc+1e-12)+1e-12).reshape(-1,1),cal.y)
    si=iso.predict(s); sp=pl.predict_proba(np.log(s/(1-s+1e-12)+1e-12).reshape(-1,1))[:,1]
    nap,prev=normalised_average_precision(y,s)
    # txid-weighted: each tx counts once
    w=1/ev.txid.map(ev.txid.value_counts()).values
    naw=nap_w(y,s,w)
    # top-100 concentration
    top=ev.assign(s=s).nlargest(100,'s'); 
    postx=ev[ev.y==1].txid.nunique()
    trs=m.predict_proba(fit[F])[:,1]
    rows.append(dict(fold=f'{f.eval_start}-{f.eval_end}',n=len(ev),pos=int(y.sum()),pos_txids=postx,prev=round(prev,3),cal_prev=round(cal.y.mean(),3),
      nap=round(nap,3),nap_txw=round(naw,3),auc=round(roc_auc_score(y,s),3),top100_distinct_tx=top.txid.nunique(),top100_prec=top.y.mean(),
      mean_p=round(s.mean(),3),brier_raw=round(brier_score_loss(y,s),4),brier_iso=round(brier_score_loss(y,si),4),brier_platt=round(brier_score_loss(y,sp),4),
      ece_raw=round(ece(y,s),3),ece_iso=round(ece(y,si),3),ece_platt=round(ece(y,sp),3),iso_levels=len(np.unique(si)),train_auc=round(roc_auc_score(fit.y,trs),3)))
r=pd.DataFrame(rows); pd.set_option('display.width',250); print(r.to_string())
print(r[['nap','nap_txw','auc','brier_raw','brier_iso','brier_platt','ece_raw','ece_iso','ece_platt','train_auc']].agg(['mean','std']).round(3).to_string())
print('corr nap vs prev',np.corrcoef(r.nap,r.prev)[0,1],' nap vs pos_txids',np.corrcoef(r.nap,np.log(r.pos_txids))[0,1])
# adversarial validation train(<=34) vs val(35-41)
from sklearn.model_selection import cross_val_predict
a=d.copy(); a['z']=(a.t>34).astype(int)
p=cross_val_predict(lgb.LGBMClassifier(n_estimators=200,verbose=-1),a[F],a.z,cv=5,method='predict_proba')[:,1]
print('adversarial AUC train vs val',roc_auc_score(a.z,p))
m=lgb.LGBMClassifier(n_estimators=200,verbose=-1).fit(a[F],a.z); imp=pd.Series(m.booster_.feature_importance('gain'),F).sort_values(ascending=False); print((imp/imp.sum()).head(6).round(3).to_string())
