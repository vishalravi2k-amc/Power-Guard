"""Train once, select on validation, then evaluate the untouched chronological test."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import copy
import json
import hashlib
import platform
import time
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, confusion_matrix
from powerguard.data import load_hourly, sequences, anomaly_features, inject_events, FEATURE_NAMES, LOOKBACK
from powerguard.models import LoadLSTM, LoadAutoencoder
from powerguard.inference import Engine


def fit(model, arrays, targets, validation, epochs, batch_size, lr=0.001):
    dataset = TensorDataset(*[torch.tensor(a, dtype=torch.float32) for a in arrays], torch.tensor(targets, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, stale, log = float('inf'), None, 0, []
    va, vy = validation
    with torch.no_grad():
        va = [torch.tensor(a, dtype=torch.float32) for a in va]
        vy = torch.tensor(vy, dtype=torch.float32)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(*batch[:-1]), batch[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()*len(batch[-1])
        model.eval()
        with torch.inference_mode():
            sums = []
            for b in range(0, len(vy), 512):
                p = model(*[a[b:b+512] for a in va])
                sums.append(((p-vy[b:b+512])**2).sum().item())
            val = sum(sums)/vy.numel()
        log.append({'epoch':ep, 'train_mse':total/len(dataset), 'validation_mse':val})
        print(f'{type(model).__name__} {ep:02d}: train={log[-1]["train_mse"]:.4f} val={val:.4f}', flush=True)
        if val < best-1e-6:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 5: break
    model.load_state_dict(best_state)
    model.eval()
    return log


def detection_metrics(labels, predicted):
    p, r, f, _ = precision_recall_fscore_support(labels, predicted, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[False, True]).ravel()
    return {'precision':float(p),'recall':float(r),'f1':float(f),'tp':int(tp),'fp':int(fp),
            'fn':int(fn),'tn':int(tn),'background_alert_rate':float(fp/max(1,tn+fp))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--ae-epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.ae_epochs < 1: parser.error('Epoch counts must be positive.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    begin = time.perf_counter()
    df = load_hourly(ROOT/'data'/'household_hourly.csv')
    n = len(df); tr, va = int(n*.7), int(n*.85)
    observed_train = df.iloc[:tr].query('observed').load_kw.dropna()
    mean, std = float(observed_train.mean()), float(observed_train.std(ddof=0))
    x, c, y, idx = sequences(df, mean, std)
    train, val, test = idx < tr, (idx>=tr)&(idx<va), idx>=va
    lstm = LoadLSTM()
    log_lstm = fit(lstm, [x[train],c[train]], y[train], ([x[val],c[val]],y[val]), args.epochs, 256)

    feats = anomaly_features(df).to_numpy()
    valid = np.isfinite(feats).all(axis=1) & df.observed.to_numpy()
    ftrain = valid & (np.arange(n)<tr)
    fval = valid & (np.arange(n)>=tr)&(np.arange(n)<va)
    fm, fs = feats[ftrain].mean(axis=0), feats[ftrain].std(axis=0)
    fs = np.maximum(fs,1e-6)
    z = ((feats-fm)/fs).astype('float32')
    # Heuristic screening only. UCI provides no verified fault labels.
    med = np.median(feats[ftrain,:3],axis=0)
    mad = np.maximum(np.median(np.abs(feats[ftrain,:3]-med),axis=0)*1.4826, 0.1)
    screen = (np.abs((feats[:,:3]-med)/mad) <= 6).all(axis=1)
    ae = LoadAutoencoder()
    log_ae = fit(ae, [z[ftrain & screen]], z[ftrain & screen],
                 ([z[fval]], z[fval]), args.ae_epochs, 256)
    forest = IsolationForest(n_estimators=150, max_samples=256, random_state=args.seed, n_jobs=2)
    forest.fit(z[ftrain & screen])
    with torch.inference_mode():
        val_ae = ((ae(torch.from_numpy(z[fval])).numpy()-z[fval])**2).mean(axis=1)
    val_if = -forest.score_samples(z[fval])
    artifact = ROOT/'artifacts'; artifact.mkdir(exist_ok=True)
    meta = {'seed':args.seed, 'lookback':LOOKBACK, 'forecast_horizon_hours':1,
            'load_mean':mean, 'load_std':std, 'feature_mean':fm.tolist(),'feature_std':fs.tolist(),
            'feature_names':FEATURE_NAMES, 'ae_threshold':float(np.quantile(val_ae,.99)),
            'if_threshold':float(np.quantile(val_if,.99)), 'threshold_quantile':.99,
            'train_end_exclusive':tr, 'validation_end_exclusive':va,
            'train_last':str(df.timestamp.iloc[tr-1]),'validation_first':str(df.timestamp.iloc[tr]),
            'validation_last':str(df.timestamp.iloc[va-1]),'test_first':str(df.timestamp.iloc[va]),
            'test_last':str(df.timestamp.iloc[-1]),'total_hours':n,
            'forecast_samples':{'train':int(train.sum()),'validation':int(val.sum()),'test':int(test.sum())},
            'anomaly_training_samples':int((ftrain&screen).sum()),
            'screened_out_training_samples':int((ftrain&~screen).sum()),
            'screening':'Load, 1h change, 24h change within six train-only robust standard deviations.',
            'python':platform.python_version(),
            'versions':{p:importlib.metadata.version(p) for p in ['torch','scikit-learn','numpy','pandas','streamlit','plotly']},
            'model_version':'powerguard-v1'}
    torch.save(lstm.state_dict(),artifact/'lstm.pt')
    torch.save(ae.state_dict(),artifact/'autoencoder.pt')
    joblib.dump(forest,artifact/'isolation_forest.joblib')
    meta['model_fingerprint'] = hashlib.sha256(b''.join((artifact/name).read_bytes() for name in
        ['lstm.pt','autoencoder.pt','isolation_forest.joblib']) +
        json.dumps([meta['ae_threshold'],meta['if_threshold']]).encode()).hexdigest()[:16]
    (artifact/'metadata.json').write_text(json.dumps(meta,indent=2))
    engine = Engine(artifact)
    evaluated = engine.predict_rows(df)
    test_df = evaluated.iloc[va:].dropna(subset=['forecast_kw','persistence_kw','previous_day_kw'])
    regression = []
    for title, col in [('LSTM','forecast_kw'),('Persistence','persistence_kw'),('Previous day','previous_day_kw')]:
        regression.append({'model':title,'mae_kw':float(mean_absolute_error(test_df.load_kw,test_df[col])),
                           'rmse_kw':float(np.sqrt(mean_squared_error(test_df.load_kw,test_df[col]))), 'samples':len(test_df)})
    challenge = inject_events(df, va, args.seed)
    challenge_result = engine.predict_rows(challenge)
    ct = challenge_result.iloc[va:].dropna(subset=['ae_score','if_score'])
    anomaly = {name:detection_metrics(ct.injected, ct[f'{prefix}_alert'])
               for name,prefix in [('Autoencoder','ae'),('Isolation Forest','if')]}
    by_type = {}
    for kind in ['Spike','Sustained high load','Near-zero drop']:
        event_rows = ct[ct.event_type.eq(kind)]
        by_type[kind] = {'hours':len(event_rows),
                         'autoencoder_recall':float(event_rows.ae_alert.mean()) if len(event_rows) else None,
                         'isolation_forest_recall':float(event_rows.if_alert.mean()) if len(event_rows) else None}
    background = evaluated.iloc[va:].dropna(subset=['ae_score','if_score'])
    valid_end = int(idx[test][0])+1
    hist = df.iloc[:valid_end]
    for _ in range(5): engine.score_latest(hist)
    times = []
    for _ in range(100):
        t = time.perf_counter(); engine.score_latest(hist); times.append((time.perf_counter()-t)*1000)
    metrics = {'forecast':regression,'injected_event_detection':anomaly,'injected_event_types':by_type,
               'unmodified_test_alerts':{'hours':len(background),'autoencoder':int(background.ae_alert.sum()),
                                         'isolation_forest':int(background.if_alert.sum())},
               'injection_label_scope':'Pointwise recovery of known injected modifications. Unmodified readings are background, not verified fault-free labels. These scores are not real fault accuracy.',
               'latency':{'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
                          'iterations':100,'scope':'CPU: feature construction, LSTM and both detectors for one latest hour; excludes model loading, network, SQLite and dashboard rendering.',
                          'machine':platform.machine(),'processor':platform.processor(),'threads':2},
               'training_seconds':time.perf_counter()-begin,
               'epoch_history':{'lstm':log_lstm,'autoencoder':log_ae}}
    reports = ROOT/'reports'; reports.mkdir(exist_ok=True)
    (reports/'metrics.json').write_text(json.dumps(metrics,indent=2))
    evaluated.iloc[max(0,va-LOOKBACK):].to_csv(reports/'test_predictions.csv',index=False)
    challenge_result.iloc[max(0,va-LOOKBACK):].to_csv(reports/'injection_challenge.csv',index=False)
    challenge_result.iloc[va:].query('ae_alert').to_csv(reports/'challenge_alerts.csv',index=False)
    print(json.dumps({k:v for k,v in metrics.items() if k!='epoch_history'},indent=2), flush=True)


if __name__ == '__main__': main()
