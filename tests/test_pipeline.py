import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from src.experiment_runner import load_data, metrics, add_features, SELECTED12, ROOT

def test_input_dimensions_and_target():
    df,cols=load_data(); assert df.shape[0]==5630; assert len(cols)==18; assert set(df.Churn)=={0,1}; assert df.Churn.sum()==948
def test_no_id_or_target_in_features():
    _,cols=load_data(); assert 'CustomerID' not in cols and 'Churn' not in cols
def test_deterministic_synonyms():
    df,_=load_data(); assert 'Phone' not in set(df.PreferredLoginDevice); assert 'Mobile' not in set(df.PreferedOrderCat)
def test_split_group_overlap_and_reproducibility():
    m=pd.read_csv(ROOT/'data/split_manifest.csv'); assert not (set(m[m.split=='test'].ProfileGroupID)&set(m[m.split=='development'].ProfileGroupID)); assert len(m)==5630
def test_confusion_arithmetic():
    y=np.array([0,0,1,1]); p=np.array([.1,.7,.4,.9]); m=metrics(y,p,.5); assert (m['tn'],m['fp'],m['fn'],m['tp'])==(1,1,1,1); assert m['accuracy']==.5
def test_engineered_features_finite():
    df,_=load_data(); z=add_features(df,'all'); assert np.isfinite(z.select_dtypes(include=np.number)).all().all()
def test_saved_probabilities_valid():
    p=pd.read_csv(ROOT/'results/predictions/locked_test_predictions.csv'); assert p.filter(like='probability').apply(lambda s:s.between(0,1).all()).all()
def test_metadata_and_model_reload():
    meta=json.loads((ROOT/'artifacts/models/model_metadata.json').read_text()); files=list((ROOT/'artifacts/models').glob('*.joblib')); assert meta['source_sha256']; assert files; assert joblib.load(files[0]) is not None
def test_final_test_access_contract():
    p=pd.read_csv(ROOT/'results/tables/final_test_results.csv'); assert set(p.evaluation)<={'locked_test_once','historical_reference'}
def test_required_tables_exist():
    for f in ['source_and_split_audit.csv','oof_model_comparison.csv','final_test_results.csv','paired_bootstrap_differences.csv']: assert (ROOT/'results/tables'/f).exists()
