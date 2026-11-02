from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import os
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'work/xdg-cache'))
import matplotlib
matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from src.experiment_runner import load_data, add_features, pipeline, SELECTED12, SEED

OUT=ROOT/'research_paper/figures'; OUT.mkdir(parents=True,exist_ok=True)
(ROOT/'figures').mkdir(exist_ok=True)
sns.set_theme(style='whitegrid',context='paper'); plt.rcParams.update({'figure.dpi':160,'savefig.dpi':300,'font.size':9})
BLUE='#24557a'; ORANGE='#d47a2c'; GREEN='#3a7d44'; RED='#a23b3b'; GRAY='#6c757d'
def save(name):
    plt.tight_layout(); plt.savefig(OUT/name,bbox_inches='tight'); plt.savefig(ROOT/'figures'/name,bbox_inches='tight'); plt.close()

df,cols=load_data(); y=df.Churn.to_numpy(); g=df.ProfileGroupID.to_numpy(); cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
dev_idx,test_idx=next(cv.split(df,y,g)); dev=df.iloc[dev_idx].copy(); test=df.iloc[test_idx].copy(); yd=dev.Churn.to_numpy(); gd=dev.ProfileGroupID.to_numpy()
meta=json.loads((ROOT/'artifacts/models/model_metadata.json').read_text())
models={
 'Extra Trees':pipeline('extra',cols,meta['best_params']['extra']),
 'CatBoost':pipeline('cat',cols,meta['best_params']['cat']),
 'XGBoost':pipeline('xgb',cols,meta['best_params']['xgb']),
 'Matched RF':pipeline('rf',SELECTED12,{'max_depth':None,'min_samples_leaf':1,'max_features':'sqrt'},'ros')}
oof_path=ROOT/'results/predictions/development_oof_predictions.csv'
if oof_path.exists():
    saved=pd.read_csv(oof_path); probs={k:saved[k+'_probability'].to_numpy() for k in models}
else:
    probs={}
    for name,m in models.items():
        use=SELECTED12 if name=='Matched RF' else cols
        probs[name]=cross_val_predict(m,dev[use],yd,groups=gd,cv=cv,method='predict_proba',n_jobs=1)[:,1]
    pd.DataFrame({'CustomerID':dev.CustomerID,'actual':yd,**{k+'_probability':v for k,v in probs.items()}}).to_csv(oof_path,index=False)

# 1 architecture
fig,ax=plt.subplots(figsize=(7.2,3.2)); ax.axis('off')
boxes=[('Immutable cleaned\nworkbook',.02),('Synonym map +\nprofile groups',.21),('Locked group\nsplit',.40),('Nested grouped CV\nfold-local pipeline',.58),('OOF threshold +\ncalibration',.78)]
for text,x in boxes:
 ax.add_patch(plt.Rectangle((x,.58),.17,.22,facecolor='#eaf1f6',edgecolor=BLUE,lw=1.2)); ax.text(x+.085,.69,text,ha='center',va='center',fontsize=7)
 if x<.78: ax.annotate('',xy=(x+.19,.69),xytext=(x+.17,.69),arrowprops=dict(arrowstyle='->',color=GRAY))
ax.add_patch(plt.Rectangle((.28,.10),.44,.22,facecolor='#eef5ea',edgecolor=GREEN,lw=1.2))
ax.text(.5,.21,'Freeze recipe → refit development\nEvaluate locked test once',ha='center',va='center',fontsize=7.5,linespacing=1.25)
ax.annotate('',xy=(.5,.32),xytext=(.865,.58),arrowprops=dict(arrowstyle='->',color=GRAY)); ax.set_xlim(0,1); ax.set_ylim(0,1); save('fig01_methodology.pdf')

# 2 class balance
fig,ax=plt.subplots(figsize=(4.8,3.2)); counts=df.Churn.value_counts().sort_index(); ax.bar(['Stay','Churn'],counts,color=[BLUE,ORANGE])
for i,v in enumerate(counts):
    ax.text(i,v+80,f'{v:,}\n({v/len(df):.1%})',ha='center')
ax.set_ylabel('Customers'); ax.set_title('Target distribution (n=5,630)'); save('fig02_class_balance.pdf')

# 3 correlation
num=[c for c in cols if c not in ['PreferredLoginDevice','PreferredPaymentMode','Gender','PreferedOrderCat','MaritalStatus']]
corr=df[num+['Churn']].corr(method='spearman')
fig,ax=plt.subplots(figsize=(7.2,5.3)); sns.heatmap(corr,cmap='vlag',center=0,vmin=-1,vmax=1,ax=ax,cbar_kws={'label':'Spearman correlation'}); ax.set_title('Numeric-feature association matrix'); save('fig03_feature_association.pdf')

# 4 model comparison
o=pd.read_csv(ROOT/'results/tables/oof_model_comparison.csv'); o=o[o.threshold_source=='oof_selected'].sort_values('f1',ascending=False).head(10)
long=o.melt(id_vars='label',value_vars=['accuracy','precision','recall','f1'],var_name='Metric',value_name='Score')
fig,ax=plt.subplots(figsize=(7.2,4.8)); sns.barplot(data=long,y='label',x='Score',hue='Metric',ax=ax,palette=[BLUE,GREEN,ORANGE,RED]); ax.set_xlim(.72,1); ax.set_ylabel(''); ax.set_title('Development OOF model comparison'); ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,-.25)); save('fig04_model_comparison.pdf')

# 5 ROC
fig,ax=plt.subplots(figsize=(5.2,4.3))
for (name,p),c in zip(probs.items(),[BLUE,ORANGE,GREEN,RED]): fpr,tpr,_=roc_curve(yd,p); ax.plot(fpr,tpr,label=f'{name} (AUC {auc(fpr,tpr):.3f})',color=c)
ax.plot([0,1],[0,1],'--',color=GRAY); ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='Development OOF ROC curves'); ax.legend(); save('fig05_roc_curves.pdf')

# 6 PR
fig,ax=plt.subplots(figsize=(5.2,4.3))
for (name,p),c in zip(probs.items(),[BLUE,ORANGE,GREEN,RED]): pr,re,_=precision_recall_curve(yd,p); ax.plot(re,pr,label=f'{name} (AP {np.trapezoid(pr[::-1],re[::-1]):.3f})',color=c)
ax.axhline(yd.mean(),ls='--',color=GRAY,label=f'Prevalence {yd.mean():.3f}'); ax.set(xlabel='Recall',ylabel='Precision',title='Development OOF precision–recall curves'); ax.legend(); save('fig06_pr_curves.pdf')

# 7 confusion matrix
pred=pd.read_csv(ROOT/'results/predictions/locked_test_predictions.csv'); cm=confusion_matrix(pred.actual,pred.selected_prediction)
fig,ax=plt.subplots(figsize=(4.2,3.5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,ax=ax); ax.set(xlabel='Predicted class',ylabel='Actual class',title='Extra Trees locked-test confusion matrix'); ax.set_xticklabels(['Stay','Churn']); ax.set_yticklabels(['Stay','Churn'],rotation=0); save('fig07_confusion_matrix.pdf')

# 8 importance
imp=pd.read_csv(ROOT/'results/tables/permutation_importance.csv').dropna().head(12).sort_values('importance_mean')
fig,ax=plt.subplots(figsize=(6.4,4.2)); ax.barh(imp.feature,imp.importance_mean,xerr=imp.importance_std,color=BLUE,alpha=.9); ax.set(xlabel='Decrease in development AP after permutation',title='Permutation importance (top 12)'); save('fig08_feature_importance.pdf')

# 9 ablation
keep=o=pd.read_csv(ROOT/'results/tables/oof_model_comparison.csv'); keep=keep[(keep.threshold_source=='oof_selected') & keep.label.str.contains('LGBM|RF historical|EXTRA')].drop_duplicates('label').sort_values('f1')
fig,ax=plt.subplots(figsize=(6.8,4.6)); ax.barh(keep.label,keep.f1,color=np.where(keep.label.eq('EXTRA tuned'),ORANGE,BLUE)); ax.set_xlim(.78,.90); ax.set_xlabel('Development OOF churn F1'); ax.set_title('Model and ablation contributions'); save('fig09_ablation.pdf')

# 10 search trials
s=pd.read_csv(ROOT/'results/tables/search_trials.csv')
fig,ax=plt.subplots(figsize=(6.4,4));
for name,grp in s.groupby('model'): ax.plot(grp.trial,grp.cv_average_precision,marker='o',ms=3,label=name.upper())
ax.set(xlabel='Random-search trial',ylabel='Inner-CV average precision',title='Hyperparameter search traces'); ax.legend(ncol=3); save('fig10_hyperparameter_search.pdf')

# 11 threshold tradeoff for Extra Trees
p=probs['Extra Trees']; rows=[]
from sklearn.metrics import accuracy_score,recall_score,f1_score
for t in np.linspace(.05,.9,86):
 yh=p>=t; rows.append((t,accuracy_score(yd,yh),recall_score(yd,yh),f1_score(yd,yh)))
th=pd.DataFrame(rows,columns=['Threshold','Accuracy','Recall','F1'])
fig,ax=plt.subplots(figsize=(6.4,4));
for m,c in [('Accuracy',BLUE),('Recall',ORANGE),('F1',GREEN)]: ax.plot(th.Threshold,th[m],label=m,color=c)
ax.axvline(.45,ls='--',color=RED,label='Selected 0.45'); ax.set(xlabel='Decision threshold',ylabel='Score',title='Extra Trees OOF threshold trade-off'); ax.legend(); save('fig11_threshold_tradeoff.pdf')

# 12 paired bootstrap CI
b=pd.read_csv(ROOT/'results/tables/paired_bootstrap_differences.csv').iloc[0]
fig,ax=plt.subplots(figsize=(5.8,3.2)); vals=[b.f1_diff,b.ap_diff]; lo=[b.f1_lo,b.ap_lo]; hi=[b.f1_hi,b.ap_hi]
ax.errorbar(vals,[1,0],xerr=[np.array(vals)-np.array(lo),np.array(hi)-np.array(vals)],fmt='o',color=BLUE,capsize=5); ax.axvline(0,color=GRAY,ls='--'); ax.set_yticks([1,0],['F1 difference','AP difference']); ax.set_xlabel('Extra Trees minus matched RF'); ax.set_title('Paired bootstrap differences on locked test (95% CI)'); save('fig12_paired_bootstrap.pdf')

# supplemental calibration
fig,ax=plt.subplots(figsize=(5.2,4));
for (name,p),c in zip(probs.items(),[BLUE,ORANGE,GREEN,RED]): frac,mean=calibration_curve(yd,p,n_bins=10,strategy='quantile'); ax.plot(mean,frac,marker='o',label=name,color=c)
ax.plot([0,1],[0,1],'--',color=GRAY); ax.set(xlabel='Mean predicted probability',ylabel='Observed churn fraction',title='Development OOF reliability'); ax.legend(); save('fig13_calibration.pdf')
print('generated',len(list(OUT.glob('*.pdf'))),'figures')
