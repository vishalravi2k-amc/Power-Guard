"""Data validation, causal preparation and controlled evaluation scenarios."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

DATASET_URL = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
LOOKBACK = 48


def uci_to_hourly(archive: Path) -> pd.DataFrame:
    """Hourly mean kW, requiring 45 valid minute readings in an hour."""
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith('household_power_consumption.txt'))
        with z.open(name) as f:
            raw = pd.read_csv(f, sep=';', usecols=['Date', 'Time', 'Global_active_power'], na_values=['?'])
    index = pd.to_datetime(raw['Date'] + ' ' + raw['Time'], format='%d/%m/%Y %H:%M:%S')
    series = pd.Series(pd.to_numeric(raw.Global_active_power, errors='coerce').to_numpy(), index=index)
    series = series.where(series.ge(0) & np.isfinite(series))
    grouped = series.resample('h')
    frame = pd.DataFrame({'load_kw': grouped.mean(), 'valid_minutes': grouped.count()})
    frame['observed'] = frame.valid_minutes.ge(45)
    frame.loc[~frame.observed, 'load_kw'] = np.nan
    # No backward fill/interpolation: future readings never fill historical inputs.
    frame['load_kw'] = frame.load_kw.ffill(limit=3)
    frame.index.name = 'timestamp'
    return frame.reset_index()


def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['observed'] = df['observed'].astype(str).str.lower().eq('true')
    return df


def validate_upload(content: bytes) -> pd.DataFrame:
    if len(content) > 10 * 1024 * 1024:
        raise ValueError('Use a CSV smaller than 10 MB.')
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError('The file must be a readable CSV.') from e
    if not {'timestamp', 'load_kw'}.issubset(df.columns):
        raise ValueError('CSV needs timestamp and load_kw columns.')
    df = df[['timestamp', 'load_kw']].copy()
    try:
        df.timestamp = pd.to_datetime(df.timestamp, errors='raise')
        df.load_kw = pd.to_numeric(df.load_kw, errors='raise')
    except Exception as e:
        raise ValueError('Use ISO timestamps and numeric load_kw values.') from e
    if df.timestamp.isna().any() or df.timestamp.duplicated().any():
        raise ValueError('Timestamps must be present and unique.')
    if not pd.api.types.is_datetime64_dtype(df.timestamp.dtype):
        raise ValueError('Use local timestamps without timezone offsets, consistently for one site.')
    if not df.timestamp.eq(df.timestamp.dt.floor('h')).all():
        raise ValueError('Provide hourly mean readings at exact hours, such as 2026-11-01 09:00:00.')
    if not np.isfinite(df.load_kw).all() or df.load_kw.lt(0).any():
        raise ValueError('Load must contain finite, non-negative kW values.')
    df = df.sort_values('timestamp').set_index('timestamp')
    span_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600 if len(df) else 0
    if len(df) < 72 or span_hours > 24 * 366 * 5 or len(df) > 50000:
        raise ValueError('Provide 72 to 50,000 hourly rows spanning at most five years.')
    df = df.asfreq('h')
    df['observed'] = df.load_kw.notna()
    df.load_kw = df.load_kw.ffill(limit=3)
    df.index.name = 'timestamp'
    if not df.load_kw.notna().rolling(LOOKBACK + 1).sum().eq(LOOKBACK + 1).any():
        raise ValueError('At least 49 consecutive usable hours are needed.')
    return df.reset_index()


def calendar(times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    return np.column_stack([np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24),
                            np.sin(2*np.pi*t.dayofweek/7), np.cos(2*np.pi*t.dayofweek/7)]).astype('float32')


def sequences(df, mean, std, lookback=LOOKBACK):
    load = df.load_kw.to_numpy(dtype='float32')
    features = np.column_stack([(load - mean) / std, calendar(df.timestamp)]).astype('float32')
    # Target i uses [i-lookback, i), never its own reading.
    idx = np.arange(lookback, len(df))
    windows = np.lib.stride_tricks.sliding_window_view(features, lookback, axis=0).transpose(0, 2, 1)[:-1]
    ok = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(load[idx]) & df.observed.to_numpy()[idx]
    return windows[ok].copy(), calendar(df.timestamp.iloc[idx[ok]]), ((load[idx[ok]]-mean)/std), idx[ok]


FEATURE_NAMES = ['load_kw', 'change_1h_kw', 'change_24h_kw', 'previous_mean_24h_kw',
                 'previous_std_24h_kw', 'hour_sin', 'hour_cos', 'weekend']


def anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df.load_kw
    h = df.timestamp.dt.hour
    return pd.DataFrame({
        'load_kw': p, 'change_1h_kw': p-p.shift(1), 'change_24h_kw': p-p.shift(24),
        'previous_mean_24h_kw': p.shift(1).rolling(24).mean(),
        'previous_std_24h_kw': p.shift(1).rolling(24).std(),
        'hour_sin': np.sin(2*np.pi*h/24), 'hour_cos': np.cos(2*np.pi*h/24),
        'weekend': df.timestamp.dt.dayofweek.ge(5).astype(float),
    }, index=df.index)


def inject_events(df: pd.DataFrame, start: int, seed=42) -> pd.DataFrame:
    """Fixed controlled challenge. Labels identify modifications, not physical faults."""
    out = df.copy()
    out['injected'] = False
    out['event_type'] = 'Unmodified'
    rng = np.random.default_rng(seed)
    for n, pos in enumerate(range(start + 72, len(out) - 36, 120)):
        pos += int(rng.integers(0, 12))
        kind, duration = [('Spike', 1), ('Sustained high load', 8), ('Near-zero drop', 4)][n % 3]
        sl = out.index[pos:pos+duration]
        if len(sl) != duration or not out.loc[sl, 'observed'].all() or not np.isfinite(out.loc[sl, 'load_kw']).all():
            continue
        if kind == 'Spike': out.loc[sl, 'load_kw'] += 5.0
        elif kind == 'Sustained high load': out.loc[sl, 'load_kw'] += 3.0
        else: out.loc[sl, 'load_kw'] *= 0.05
        out.loc[sl, 'injected'] = True
        out.loc[sl, 'event_type'] = kind
    return out


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(df[['timestamp', 'load_kw']].to_csv(index=False).encode()).hexdigest()[:16]
