from __future__ import annotations

import hashlib, json, os, platform, sys, time, warnings
from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SEED = 20251024
INPUT = ROOT / "input_sources" / "Task2_Complete_Data_Quality_and_Cleaning.xlsx"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
PREDS = RESULTS / "predictions"
LOGS = RESULTS / "logs"
ARTIFACTS = ROOT / "artifacts" / "models"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
for d in [TABLES, PREDS, LOGS, ARTIFACTS, FIGURES, REPORTS, ROOT/'data']:
    d.mkdir(parents=True, exist_ok=True)

SELECTED12 = ['Tenure','Complain','CashbackAmount','PreferedOrderCat','WarehouseToHome',
 'SatisfactionScore','DaySinceLastOrder','MaritalStatus','NumberOfAddress',
 'OrderAmountHikeFromlastYear','PreferredPaymentMode','CityTier']
CAT = ['PreferredLoginDevice','PreferredPaymentMode','Gender','PreferedOrderCat','MaritalStatus']

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def load_data():
    df=pd.read_excel(INPUT,sheet_name='Cleaned Data')
    df['PreferredLoginDevice']=df['PreferredLoginDevice'].replace({'Phone':'Mobile Phone'})
    df['PreferedOrderCat']=df['PreferedOrderCat'].replace({'Mobile':'Mobile Phone'})
    predictors=[c for c in df.columns if c not in ['CustomerID','Churn']]
    key=df[predictors].astype(str).agg('\x1f'.join,axis=1)
    df['ProfileGroupID']=key.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    return df,predictors

def add_features(df, mode='none'):
    z=df.copy()
    if mode in ('paper','all'):
        z['TenurePerDevice']=z['Tenure']/z['NumberOfDeviceRegistered'].clip(lower=1)
    if mode in ('behavior','all'):
        z['ComplaintXSatisfaction']=z['Complain']*z['SatisfactionScore']
        z['TenureXDevices']=z['Tenure']*z['NumberOfDeviceRegistered']
        z['RecencyXTenture']=z['DaySinceLastOrder']*z['Tenure']
        z['OrdersXCoupons']=z['OrderCount']*z['CouponUsed']
        z['DistanceXCityTier']=z['WarehouseToHome']*z['CityTier']
        z['CouponPerOrder']=z['CouponUsed']/z['OrderCount'].clip(lower=1)
    return z

def make_preprocessor(cols):
    cats=[c for c in cols if c in CAT]
    nums=[c for c in cols if c not in cats]
    return ColumnTransformer([
      ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
      ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)
    ],verbose_feature_names_out=False)

def estimator(name, params=None, class_weight=None):
    p=params or {}
    if name=='logistic': return LogisticRegression(max_iter=3000,class_weight=class_weight,random_state=SEED,**p)
    if name=='tree': return DecisionTreeClassifier(random_state=SEED,class_weight=class_weight,**p)
    if name=='rf': return RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=SEED,class_weight=class_weight,**p)
    if name=='extra': return ExtraTreesClassifier(n_estimators=500,n_jobs=-1,random_state=SEED,class_weight=class_weight,**p)
    if name=='xgb': return XGBClassifier(n_estimators=500,n_jobs=-1,random_state=SEED,eval_metric='logloss',**p)
    if name=='lgbm': return LGBMClassifier(n_estimators=500,n_jobs=-1,random_state=SEED,verbosity=-1,**p)
    if name=='cat': return CatBoostClassifier(iterations=500,random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=-1,**p)
    raise KeyError(name)

def pipeline(name, cols, params=None, balance='none', selector='none'):
    steps=[('prep',make_preprocessor(cols))]
    if selector=='model': steps.append(('select',SelectFromModel(ExtraTreesClassifier(n_estimators=250,random_state=SEED,n_jobs=-1),threshold='median')))
    elif selector=='rfe': steps.append(('select',RFE(LogisticRegression(max_iter=2000),n_features_to_select=12,step=.2)))
    if balance=='ros': steps.append(('sample',RandomOverSampler(random_state=SEED)))
    elif balance=='smote': steps.append(('sample',SMOTE(random_state=SEED,k_neighbors=5)))
    cw='balanced' if balance=='class_weight' and name in {'logistic','tree','rf','extra'} else None
    pp=dict(params or {})
    if balance=='class_weight' and name in {'xgb','lgbm','cat'}:
        pp['scale_pos_weight']=4.94
    steps.append(('model',estimator(name,pp,cw)))
    return ImbPipeline(steps)

def metrics(y,p,t=.5):
    yh=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,yh),precision=precision_score(y,yh,zero_division=0),
      recall=recall_score(y,yh),f1=f1_score(y,yh),specificity=tn/(tn+fp),
      roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),
      balanced_accuracy=balanced_accuracy_score(y,yh),mcc=matthews_corrcoef(y,yh),
      brier=brier_score_loss(y,p),log_loss=log_loss(y,p,labels=[0,1]),
      tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=float(t))

def threshold_select(y,p,floor=.93):
    candidates=np.unique(np.r_[np.linspace(.05,.95,181),p])
    rows=[]
    for t in candidates:
        m=metrics(y,p,t); rows.append(m)
    valid=[r for r in rows if r['accuracy']>=floor]
    best=max(valid or rows,key=lambda r:(r['f1'],r['recall'],r['accuracy']))
    return best['threshold'],pd.DataFrame(rows)

def bootstrap_diff(y,pa,pb,ta,tb,n=2000):
    rng=np.random.default_rng(SEED); y=np.asarray(y); diffs=[]
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y)); yy=y[ix]
        if len(np.unique(yy))<2: continue
        diffs.append([f1_score(yy,(pb[ix]>=tb))-f1_score(yy,(pa[ix]>=ta)),
                      average_precision_score(yy,pb[ix])-average_precision_score(yy,pa[ix])])
    a=np.asarray(diffs)
    return {'f1_diff':float(a[:,0].mean()),'f1_lo':float(np.quantile(a[:,0],.025)),'f1_hi':float(np.quantile(a[:,0],.975)),
            'ap_diff':float(a[:,1].mean()),'ap_lo':float(np.quantile(a[:,1],.025)),'ap_hi':float(np.quantile(a[:,1],.975))}

def search_spaces():
    return {
      'rf':{'model__max_depth':[None,8,12,18],'model__min_samples_leaf':randint(1,8),'model__max_features':['sqrt',.6,1.0]},
      'extra':{'model__max_depth':[None,8,12,18],'model__min_samples_leaf':randint(1,6),'model__max_features':['sqrt',.6,1.0]},
      'xgb':{'model__max_depth':randint(2,7),'model__learning_rate':loguniform(.02,.2),'model__subsample':uniform(.7,.3),'model__colsample_bytree':uniform(.7,.3),'model__min_child_weight':randint(1,8)},
      'lgbm':{'model__num_leaves':randint(10,64),'model__learning_rate':loguniform(.02,.2),'model__subsample':uniform(.7,.3),'model__colsample_bytree':uniform(.7,.3),'model__min_child_samples':randint(10,50)},
      'cat':{'model__depth':randint(4,9),'model__learning_rate':loguniform(.02,.2),'model__l2_leaf_reg':loguniform(1,20)} }

def main():
    started=time.time(); df,base_cols=load_data(); y=df.Churn.to_numpy(); groups=df.ProfileGroupID.to_numpy()
    split=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(df,y,groups))
    dev_idx,test_idx=split; dev=df.iloc[dev_idx].copy(); test=df.iloc[test_idx].copy()
    manifest=pd.DataFrame({'CustomerID':df.CustomerID,'ProfileGroupID':groups,'split':np.where(np.isin(np.arange(len(df)),test_idx),'test','development'),'Churn':y})
    manifest.to_csv(ROOT/'data/split_manifest.csv',index=False)
    audit=pd.DataFrame([{'source':INPUT.name,'sha256':sha256(INPUT),'rows':len(df),'columns':20,'churners':int(y.sum()),'non_churners':int((1-y).sum()),'churn_rate':y.mean(),'profile_groups':df.ProfileGroupID.nunique(),'repeated_profile_groups':int((df.ProfileGroupID.value_counts()>1).sum()),'rows_in_repeated_profiles':int(df.ProfileGroupID.map(df.ProfileGroupID.value_counts()).gt(1).sum()),'conflicting_label_groups':int(df.groupby('ProfileGroupID').Churn.nunique().gt(1).sum()),'dev_rows':len(dev),'test_rows':len(test),'group_overlap':len(set(dev.ProfileGroupID)&set(test.ProfileGroupID))}])
    audit.to_csv(TABLES/'source_and_split_audit.csv',index=False)
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    inner=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+1)
    Xdev=dev[base_cols]; ydev=dev.Churn.to_numpy(); gdev=dev.ProfileGroupID.to_numpy()
    best_params={}; search_rows=[]
    for name,space in search_spaces().items():
        st=time.time(); rs=RandomizedSearchCV(pipeline(name,base_cols,balance='none'),space,n_iter=12,scoring='average_precision',cv=inner,n_jobs=1,random_state=SEED,refit=True,return_train_score=False)
        rs.fit(Xdev,ydev,groups=gdev); best_params[name]={k.replace('model__',''):v.item() if hasattr(v,'item') else v for k,v in rs.best_params_.items()}
        for i,(score,pars) in enumerate(zip(rs.cv_results_['mean_test_score'],rs.cv_results_['params'])):
            search_rows.append({'stage':'E1','model':name,'trial':i+1,'cv_average_precision':score,'parameters':json.dumps(pars,default=str),'runtime_seconds':time.time()-st,'status':'complete'})
    pd.DataFrame(search_rows).to_csv(TABLES/'search_trials.csv',index=False)
    configs=[]
    def add(label,name,features='all',eng='none',balance='none',selector='none',params=None): configs.append(dict(label=label,name=name,features=features,eng=eng,balance=balance,selector=selector,params=params or best_params.get(name,{})))
    add('Majority baseline','majority'); add('Logistic regression','logistic'); add('Decision tree','tree',params={'max_depth':8,'min_samples_leaf':8})
    for n in ['rf','extra','xgb','lgbm','cat']: add(n.upper()+' tuned',n)
    add('RF historical 12-feature ROS matched','rf','selected12','none','ros',params={'max_depth':None,'min_samples_leaf':1,'max_features':'sqrt'})
    add('RF all-feature ROS','rf','all','none','ros')
    for eng in ['paper','behavior','all']: add('LGBM '+eng+' features', 'lgbm','all',eng)
    for bal in ['class_weight','ros','smote']: add('LGBM '+bal,'lgbm','all','all',bal)
    for sel in ['model','rfe']: add('LGBM selector '+sel,'lgbm','all','all','none',sel)
    oof_rows=[]; pred_store={}; fitted={}
    for cfg in configs:
        st=time.time(); label=cfg['label']
        if cfg['name']=='majority': p=np.repeat(ydev.mean(),len(ydev)); fit_sec=0
        else:
            d=add_features(dev,cfg['eng']); cols=SELECTED12 if cfg['features']=='selected12' else [c for c in d.columns if c not in ['CustomerID','Churn','ProfileGroupID']]
            pipe=pipeline(cfg['name'],cols,cfg['params'],cfg['balance'],cfg['selector'])
            p=cross_val_predict(pipe,d[cols],ydev,groups=gdev,cv=cv,method='predict_proba',n_jobs=1)[:,1]
            fit_sec=time.time()-st
        t,curve=threshold_select(ydev,p,.93); m05=metrics(ydev,p,.5); mt=metrics(ydev,p,t)
        for kind,m in [('0.50',m05),('oof_selected',mt)]: oof_rows.append({**cfg,'params':json.dumps(cfg['params'],default=str),'evaluation':'development_oof','threshold_source':kind,'runtime_seconds':fit_sec,'status':'complete',**m})
        pred_store[label]=(p,t,cfg)
        curve.assign(model=label).to_csv(TABLES/('threshold_'+hashlib.md5(label.encode()).hexdigest()[:8]+'.csv'),index=False)
    oof=pd.DataFrame(oof_rows); oof.to_csv(TABLES/'oof_model_comparison.csv',index=False)
    ranked=oof[oof.threshold_source=='oof_selected'].sort_values(['f1','average_precision','recall'],ascending=False)
    candidates=ranked[ranked.model!='majority'] if 'model' in ranked else ranked
    best_label=ranked.iloc[0].label
    # diversity-aware blend among top three non-majority models
    top=[x for x in ranked.label if x!='Majority baseline'][:3]
    blend=np.mean([pred_store[x][0] for x in top],axis=0); bt,bcurve=threshold_select(ydev,blend,.93)
    bm=metrics(ydev,blend,bt); oof_rows.append({'label':'Calibrated probability blend','name':'blend','features':'mixed','eng':'mixed','balance':'mixed','selector':'none','params':json.dumps({'members':top}),'evaluation':'development_oof','threshold_source':'oof_selected','runtime_seconds':0,'status':'complete',**bm})
    pred_store['Calibrated probability blend']=(blend,bt,{'members':top})
    oof=pd.DataFrame(oof_rows); oof.to_csv(TABLES/'oof_model_comparison.csv',index=False)
    if bm['f1']>metrics(ydev,pred_store[best_label][0],pred_store[best_label][1])['f1']: best_label='Calibrated probability blend'
    # calibration study on top single model with nested OOF calibrators
    top_single=ranked.iloc[0].label; p0,t0,cfg0=pred_store[top_single]
    d=add_features(dev,cfg0['eng']); cols=SELECTED12 if cfg0['features']=='selected12' else [c for c in d.columns if c not in ['CustomerID','Churn','ProfileGroupID']]
    basepipe=pipeline(cfg0['name'],cols,cfg0['params'],cfg0['balance'],cfg0['selector'])
    cal_rows=[]
    for method in ['sigmoid','isotonic']:
        cal=CalibratedClassifierCV(basepipe,method=method,cv=3)
        pc=cross_val_predict(cal,d[cols],ydev,groups=gdev,cv=cv,method='predict_proba',n_jobs=1)[:,1]
        tc,_=threshold_select(ydev,pc,.93); mm=metrics(ydev,pc,tc); cal_rows.append({'model':top_single,'calibration':method,**mm})
        pred_store[top_single+' calibrated '+method]=(pc,tc,{'base':cfg0,'calibration':method})
    pd.DataFrame(cal_rows).to_csv(TABLES/'calibration_comparison.csv',index=False)
    # freeze selection on OOF only; evaluate test once
    all_rank=[]
    for label,(p,t,c) in pred_store.items():
        m=metrics(ydev,p,t); all_rank.append((m['f1'],m['average_precision'],m['recall'],label))
    selected=max(all_rank)[3]
    def fit_predict(label):
        if label=='Calibrated probability blend':
            probs=[]; models=[]
            for member in top:
                _,_,c=pred_store[member]; dd=add_features(dev,c['eng']); tt=add_features(test,c['eng']); cc=SELECTED12 if c['features']=='selected12' else [x for x in dd.columns if x not in ['CustomerID','Churn','ProfileGroupID']]
                mdl=pipeline(c['name'],cc,c['params'],c['balance'],c['selector']).fit(dd[cc],ydev); probs.append(mdl.predict_proba(tt[cc])[:,1]); models.append((member,mdl,cc,c['eng']))
            return np.mean(probs,axis=0),models
        if ' calibrated ' in label:
            base=label.split(' calibrated ')[0]; method=label.rsplit(' ',1)[-1]; _,_,c=pred_store[base]
            dd=add_features(dev,c['eng']); tt=add_features(test,c['eng']); cc=SELECTED12 if c['features']=='selected12' else [x for x in dd.columns if x not in ['CustomerID','Churn','ProfileGroupID']]
            mdl=CalibratedClassifierCV(pipeline(c['name'],cc,c['params'],c['balance'],c['selector']),method=method,cv=3).fit(dd[cc],ydev)
            return mdl.predict_proba(tt[cc])[:,1],(mdl,cc,c['eng'])
        _,_,c=pred_store[label]; dd=add_features(dev,c['eng']); tt=add_features(test,c['eng']); cc=SELECTED12 if c['features']=='selected12' else [x for x in dd.columns if x not in ['CustomerID','Churn','ProfileGroupID']]
        mdl=pipeline(c['name'],cc,c['params'],c['balance'],c['selector']).fit(dd[cc],ydev); return mdl.predict_proba(tt[cc])[:,1],(mdl,cc,c['eng'])
    # selected and matched baseline, same locked test
    final=[]; test_preds={}
    for label in [selected,'RF historical 12-feature ROS matched']:
        pt,mdl=fit_predict(label); th=pred_store[label][1]; mm=metrics(test.Churn.to_numpy(),pt,th); final.append({'model':label,'predictor_count':12 if '12-feature' in label else 'mixed/all','evaluation':'locked_test_once','threshold_source':'development_oof','historical_reference':False,**mm}); test_preds[label]=pt
        joblib.dump(mdl,ARTIFACTS/(hashlib.md5(label.encode()).hexdigest()[:10]+'.joblib'))
    hist={'model':'Historical reported RF','predictor_count':12,'evaluation':'historical_reference','threshold_source':'historical_validation','historical_reference':True,'accuracy':.9547069271758437,'precision':.8926553672316384,'recall':.8315789473684211,'f1':.8610354223433242,'specificity':.9797008547008547,'roc_auc':.985863697705803,'average_precision':.9531464893497235,'balanced_accuracy':(.8315789473684211+.9797008547008547)/2,'mcc':np.nan,'brier':np.nan,'log_loss':np.nan,'tn':917,'fp':19,'fn':32,'tp':158,'threshold':.5}
    final.append(hist); pd.DataFrame(final).to_csv(TABLES/'final_test_results.csv',index=False)
    basep=test_preds['RF historical 12-feature ROS matched']; selp=test_preds[selected]
    dif=bootstrap_diff(test.Churn.to_numpy(),basep,selp,pred_store['RF historical 12-feature ROS matched'][1],pred_store[selected][1])
    pd.DataFrame([{'selected':selected,'baseline':'RF historical 12-feature ROS matched',**dif}]).to_csv(TABLES/'paired_bootstrap_differences.csv',index=False)
    out=pd.DataFrame({'CustomerID':test.CustomerID,'split':'locked_test','actual':test.Churn,'selected_probability':selp,'selected_prediction':(selp>=pred_store[selected][1]).astype(int),'matched_rf_probability':basep,'matched_rf_prediction':(basep>=pred_store['RF historical 12-feature ROS matched'][1]).astype(int)})
    out.to_csv(PREDS/'locked_test_predictions.csv',index=False)
    # contact-at-k
    rows=[]
    for label,p in [(selected,selp),('Matched RF',basep)]:
        order=np.argsort(-p)
        for frac in [.05,.10,.20]:
            n=int(np.ceil(len(p)*frac)); ix=order[:n]; rows.append({'model':label,'top_fraction':frac,'contacted':n,'precision_at_k':test.Churn.to_numpy()[ix].mean(),'recall_at_k':test.Churn.to_numpy()[ix].sum()/test.Churn.sum()})
    pd.DataFrame(rows).to_csv(TABLES/'contact_at_k.csv',index=False)
    # feature importance on matched data via permutation
    dd=add_features(dev,cfg0['eng']); cc=SELECTED12 if cfg0['features']=='selected12' else [c for c in dd.columns if c not in ['CustomerID','Churn','ProfileGroupID']]
    model=pipeline(cfg0['name'],cc,cfg0['params'],cfg0['balance'],cfg0['selector']).fit(dd[cc],ydev)
    try:
        pi=permutation_importance(model,dd[cc],ydev,n_repeats=8,random_state=SEED,scoring='average_precision',n_jobs=-1)
        pd.DataFrame({'feature':cc,'importance_mean':pi.importances_mean,'importance_std':pi.importances_std}).sort_values('importance_mean',ascending=False).to_csv(TABLES/'permutation_importance.csv',index=False)
    except Exception as e: pd.DataFrame([{'error':str(e)}]).to_csv(TABLES/'permutation_importance.csv',index=False)
    meta={'selected_model':selected,'selected_threshold':pred_store[selected][1],'selected_oof_metrics':metrics(ydev,pred_store[selected][0],pred_store[selected][1]),'source_sha256':sha256(INPUT),'seed':SEED,'python':sys.version,'platform':platform.platform(),'runtime_seconds':time.time()-started,'best_params':best_params,'feature_order':base_cols}
    (ARTIFACTS/'model_metadata.json').write_text(json.dumps(meta,indent=2,default=str))
    print(json.dumps(meta,indent=2,default=str))

if __name__=='__main__': main()
