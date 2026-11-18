from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import pytest
from powerguard.data import validate_upload,load_hourly,calendar,sequences,anomaly_features,inject_events
from powerguard.inference import Engine
from powerguard import alerts
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def data():return load_hourly(ROOT/'data/household_hourly.csv')

@pytest.fixture(scope='module')
def engine():return Engine(ROOT/'artifacts')

def test_data_checksum_and_provenance(data):
    p=json.loads((ROOT/'data/provenance.json').read_text())
    assert hashlib.sha256((ROOT/'data/household_hourly.csv').read_bytes()).hexdigest()==p['hourly_csv_sha256']
    assert len(data)==p['hours'] and int(data.observed.sum())==p['observed_hours']
    assert data.timestamp.is_unique and data.timestamp.is_monotonic_increasing
    assert (data.timestamp.diff().dropna()==pd.Timedelta(hours=1)).all()

def test_scaler_uses_training_only(data,engine):
    tr=data.iloc[:engine.meta['train_end_exclusive']].query('observed').load_kw.dropna()
    assert np.isclose(tr.mean(),engine.mean) and np.isclose(tr.std(ddof=0),engine.std)

def test_saved_models_reproduce_test_outputs(data,engine):
    va=engine.meta['validation_end_exclusive']; actual=engine.predict_rows(data.iloc[va-48:va+100].reset_index(drop=True)).iloc[48:]
    expected=pd.read_csv(ROOT/'reports/test_predictions.csv').iloc[48:148]
    for key in ['forecast_kw','ae_score','if_score']:
        np.testing.assert_allclose(actual[key],expected[key],atol=2e-5,equal_nan=True)

def test_future_target_cannot_affect_prediction(data,engine):
    h=data.iloc[29600:29650].copy().reset_index(drop=True)
    a=engine.predict_rows(h).forecast_kw.iloc[48]
    h.loc[48:,'load_kw']=200
    assert engine.predict_rows(h).forecast_kw.iloc[48]==a

def test_upload_valid_and_long_gaps():
    raw=(ROOT/'data/sample_upload.csv').read_bytes(); df=validate_upload(raw)
    assert len(df)>=72
    cut=df.drop(index=range(10,15))
    out=validate_upload(cut[['timestamp','load_kw']].to_csv(index=False).encode())
    assert not out.observed.iloc[10] and out.load_kw.iloc[10]==out.load_kw.iloc[9]
    assert pd.isna(out.load_kw.iloc[13])

@pytest.mark.parametrize('kind',['negative','duplicate','subhour','offset','nan'])
def test_invalid_uploads(kind):
    df=pd.read_csv(ROOT/'data/sample_upload.csv')
    if kind=='negative':df.loc[0,'load_kw']=-1
    if kind=='duplicate':df.loc[1,'timestamp']=df.loc[0,'timestamp']
    if kind=='subhour':df.loc[0,'timestamp']='2010-04-01 00:30:00'
    if kind=='offset':df['timestamp']=df.timestamp+'+05:30'
    if kind=='nan':df.loc[0,'load_kw']=np.nan
    with pytest.raises(ValueError):validate_upload(df.to_csv(index=False).encode())

def test_injections_stay_in_test(data,engine):
    start=engine.meta['validation_end_exclusive']; changed=inject_events(data,start)
    pd.testing.assert_series_equal(changed.load_kw.iloc[:start],data.load_kw.iloc[:start])
    assert changed.injected.any() and not changed.injected.iloc[:start].any()

def test_alert_register_deduplicates_and_reviews(tmp_path):
    df=pd.read_csv(ROOT/'reports/injection_challenge.csv').head(500)
    path=tmp_path/'alerts.db'
    alerts.record(path,df,'test','Autoencoder',1.402969)
    a=alerts.read(path,'test')
    assert len(a)>0
    alerts.record(path,df,'test','Autoencoder',1.402969)
    assert len(alerts.read(path,'test'))==len(a)
    alerts.mark_reviewed(path,'test')
    assert alerts.read(path,'test').reviewed.eq(1).all()

def test_streamlit_pages_run():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not at.exception
    for label in ['Model results','Upload readings','Project review','Monitor & replay']:
        at.sidebar.radio[0].set_value(label).run()
        assert not at.exception
