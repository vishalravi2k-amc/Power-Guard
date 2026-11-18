"""Export trusted trained artifacts to portable, non-executable browser JSON."""
import json, sys, shutil, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch, joblib
from powerguard.inference import Engine
from powerguard.data import load_hourly

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False))

def main():
    target = ROOT/'web'
    meta = json.loads((ROOT/'artifacts/metadata.json').read_text())
    models = {'meta':meta}
    for name in ['lstm','autoencoder']:
        state = torch.load(ROOT/f'artifacts/{name}.pt', map_location='cpu', weights_only=True)
        models[name] = {k:v.numpy().tolist() for k,v in state.items()}
    forest = joblib.load(ROOT/'artifacts/isolation_forest.joblib')
    models['forest'] = {'max_samples':int(forest.max_samples_), 'trees':[]}
    for est, features in zip(forest.estimators_, forest.estimators_features_):
        t = est.tree_
        models['forest']['trees'].append({'left':t.children_left.tolist(), 'right':t.children_right.tolist(),
            'feature':t.feature.tolist(), 'threshold':t.threshold.tolist(), 'count':t.n_node_samples.tolist(),
            'features':features.tolist()})
    write(target/'models/weights.json',models)
    for name, source in [('original','test_predictions'),('challenge','injection_challenge')]:
        df=pd.read_csv(ROOT/f'reports/{source}.csv')
        write(target/f'data/{name}.json', json.loads(df.to_json(orient='records',double_precision=12)))
    for source in ['reports/metrics.json','data/provenance.json','data/sample_upload.csv','reports/EVALUATION.md','DATA_LICENSE.md']:
        shutil.copyfile(ROOT/source,target/'data'/Path(source).name)
    # Golden cases from Python, used to verify the browser port independently.
    engine=Engine(ROOT/'artifacts')
    df=load_hourly(ROOT/'data/household_hourly.csv')
    cases=[]
    for end in [29480, 29600, 30000, 31000, 32000, 33000, len(df)]:
        h=df.iloc[end-49:end].copy().reset_index(drop=True)
        if h.load_kw.isna().any() or not h.observed.iloc[-1]:continue
        result=engine.score_latest(h)
        cases.append({'rows':json.loads(h.to_json(orient='records',date_format='iso')),
            'expected':{k:float(result[k]) for k in ['forecast_kw','ae_score','if_score']},
            'next':engine.next_hour(h)['forecast_kw']})
    write(ROOT/'tests/golden_cases.json',cases)
    print(f'Exported weights, real historical data, metrics and {len(cases)} golden cases.')

if __name__=='__main__':main()
