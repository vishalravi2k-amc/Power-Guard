"""Local SQLite alert register. Same observation is logged once per scenario/model."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import pandas as pd


@contextmanager
def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute('''CREATE TABLE IF NOT EXISTS alerts (
        source TEXT, timestamp TEXT, detector TEXT, load_kw REAL, score REAL,
        threshold REAL, simulated INTEGER, event_type TEXT, reviewed INTEGER DEFAULT 0,
        PRIMARY KEY (source, timestamp, detector))''')
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def record(path: Path, rows: pd.DataFrame, source: str, detector: str, threshold: float):
    prefix = 'ae' if detector == 'Autoencoder' else 'if'
    flagged = rows[rows[f'{prefix}_alert']]
    with connect(path) as con:
        con.executemany('INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?,?,?,?,0)', [
            (source, str(r.timestamp), detector, float(r.load_kw), float(getattr(r, f'{prefix}_score')),
             threshold, int(bool(getattr(r, 'injected', False))), str(getattr(r, 'event_type', 'Unmodified')))
            for r in flagged.itertuples()])


def read(path: Path, source=None):
    with connect(path) as con:
        if source:
            return pd.read_sql_query('SELECT * FROM alerts WHERE source=? ORDER BY timestamp DESC', con, params=[source])
        return pd.read_sql_query('SELECT * FROM alerts ORDER BY timestamp DESC', con)


def mark_reviewed(path: Path, source: str):
    with connect(path) as con:
        con.execute('UPDATE alerts SET reviewed=1 WHERE source=?', (source,))
