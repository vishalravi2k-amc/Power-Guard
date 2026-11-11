from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .data import LOOKBACK, sequences, calendar, anomaly_features
from .models import LoadLSTM, LoadAutoencoder


class Engine:
    def __init__(self, artifact_dir: Path):
        torch.set_num_threads(2)
        self.meta = json.loads((artifact_dir/'metadata.json').read_text())
        self.lstm, self.ae = LoadLSTM(), LoadAutoencoder()
        self.lstm.load_state_dict(torch.load(artifact_dir/'lstm.pt', map_location='cpu', weights_only=True))
        self.ae.load_state_dict(torch.load(artifact_dir/'autoencoder.pt', map_location='cpu', weights_only=True))
        self.lstm.eval(); self.ae.eval()
        # Only load the trusted model shipped with this project, never uploaded pickle files.
        self.forest = joblib.load(artifact_dir/'isolation_forest.joblib')
        self.mean, self.std = self.meta['load_mean'], self.meta['load_std']
        self.fm = np.array(self.meta['feature_mean'])
        self.fs = np.array(self.meta['feature_std'])

    def predict_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ['forecast_kw', 'persistence_kw', 'previous_day_kw', 'ae_score', 'if_score']:
            out[col] = np.nan
        if len(df) > LOOKBACK:
            x, c, _, idx = sequences(df, self.mean, self.std)
            preds = []
            with torch.inference_mode():
                for b in range(0, len(x), 512):
                    preds.extend(self.lstm(torch.from_numpy(x[b:b+512]), torch.from_numpy(c[b:b+512])).numpy())
            out.loc[out.index[idx], 'forecast_kw'] = np.maximum(0, np.array(preds)*self.std+self.mean)
            out.loc[out.index[idx], 'persistence_kw'] = df.load_kw.shift(1).iloc[idx].to_numpy()
            out.loc[out.index[idx], 'previous_day_kw'] = df.load_kw.shift(24).iloc[idx].to_numpy()
        feats = anomaly_features(df).to_numpy()
        valid = np.isfinite(feats).all(axis=1) & df.observed.to_numpy()
        z = ((feats[valid]-self.fm)/self.fs).astype('float32')
        if len(z):
            with torch.inference_mode():
                recon = self.ae(torch.from_numpy(z)).numpy()
            out.loc[valid, 'ae_score'] = ((recon-z)**2).mean(axis=1)
            out.loc[valid, 'if_score'] = -self.forest.score_samples(z)
        out['ae_alert'] = out.ae_score.gt(self.meta['ae_threshold'])
        out['if_alert'] = out.if_score.gt(self.meta['if_threshold'])
        return out

    def next_hour(self, history: pd.DataFrame) -> dict:
        if len(history) < LOOKBACK or history.load_kw.tail(LOOKBACK).isna().any():
            raise ValueError('Need 48 consecutive usable historical hours.')
        h = history.tail(LOOKBACK)
        x = np.column_stack([(h.load_kw-self.mean)/self.std, calendar(h.timestamp)]).astype('float32')
        target = pd.Timestamp(h.timestamp.iloc[-1])+pd.Timedelta(hours=1)
        with torch.inference_mode():
            y = self.lstm(torch.from_numpy(x[None]), torch.from_numpy(calendar([target]))).item()
        return {'timestamp': str(target), 'forecast_kw': max(0.0, y*self.std+self.mean)}

    def score_latest(self, history: pd.DataFrame) -> dict:
        """One reading's features, LSTM forecast and detector scores; no UI or disk IO."""
        if len(history) < LOOKBACK + 1:
            raise ValueError('Need 49 hours including the latest reading.')
        tail = history.tail(LOOKBACK+1).reset_index(drop=True)
        return self.predict_rows(tail).iloc[-1].to_dict()
