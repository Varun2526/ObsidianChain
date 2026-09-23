import sys, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from obsidianchain.ml.protocol import rolling_origin_folds, normalised_average_precision, paired_verdict
import lightgbm as lgb
from sklearn.metrics import average_precision_score
P='data/models/ps_native/datasets/'
d=pd.concat([pd.read_parquet(P+'train.parquet'),pd.read_parquet(P+'validation.parquet')],ignore_index=True)
d['t']=((d.timestamp-1400000000)//1209600+1).astype(int)
print('fee==0 rows',(d.fee==0).mean(),'illicit rate fee0',d[d.fee==0].y.mean(),'vs',d[d.fee>0].y.mean(), 'distinct txids fee0',d[d.fee==0].txid.nunique())
F=[c for c in d.columns if c not in('address','txid','timestamp','y','t')]
d['txsize']=d.txid.map(d.txid.value_counts())
# variant features
d['fee_missing']=(d.fee==0).astype(int)
for c in ['fee','fee_ratio','total_input_amount','output_amount_mean','input_amount_mean','mean_fee_ratio_asof_t']:
    d[c+'_steprank']=d.groupby('t')[c].rank(pct=True)
d.loc[d.fee==0,['fee','fee_ratio']]=np.nan
RED=['total_output_amount','tx_velocity_per_hour','btc_sent_total_asof_t','mean_fee_ratio_asof_t','gap_since_last_tx','is_mixing_candidate']
V={'base':(F,False),'txweight':(F,True),
   'fee_nan+missflag':(F+['fee_missing'],False),
   'steprank':(F+['fee_missing']+[c for c in d if c.endswith('_steprank')],False),
   'steprank+txw':(F+['fee_missing']+[c for c in d if c.endswith('_steprank')],True),
   'pruned':([c for c in F if c not in RED],False)}
def nap_w(y,s,w):
    pr=np.average(y,weights=w); return (average_precision_score(y,s,sample_weight=w)-pr)/(1-pr)
res={}
base_d=d.copy(); base_d[['fee','fee_ratio']]=base_d[['fee','fee_ratio']].fillna(0)
for name,(feats,txw) in V.items():
    D=base_d if name in('base','txweight','pruned') else d
    a=[];b=[]
    for f in rolling_origin_folds():
        tr=D[D.t<=f.train_end]; ev=D[(D.t>=f.eval_start)&(D.t<=f.eval_end)]
        w=1/tr.txsize if txw else None
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,min_child_samples=50,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1,random_state=0).fit(tr[feats],tr.y,sample_weight=w)
        s=m.predict_proba(ev[feats])[:,1]
        a.append(normalised_average_precision(ev.y,s)[0]); b.append(nap_w(ev.y.values,s,1/ev.txsize.values))
    res[name]=(np.array(a),np.array(b)); print(f'{name:18s} nAP={np.mean(a):.3f}±{np.std(a):.3f}  nAP_txw={np.mean(b):.3f}±{np.std(b):.3f}',flush=True)
for name in V:
    if name=='base': continue
    for i,lab in enumerate(['nAP','nAP_txw']):
        v=paired_verdict(res[name][i]-res['base'][i],a=name,b='base'); print(name,lab,{k:v.get(k) for k in ('verdict','mean_difference','p_value','folds_won_by_a')})
