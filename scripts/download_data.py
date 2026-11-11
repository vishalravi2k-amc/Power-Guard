"""Rebuild the bundled hourly dataset from the public UCI archive."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import hashlib
import urllib.request
from powerguard.data import uci_to_hourly, DATASET_URL

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=Path, help='Use an existing UCI zip instead of downloading.')
args = parser.parse_args()
archive = args.archive or root/'data'/'uci_original.zip'
archive.parent.mkdir(parents=True, exist_ok=True)
if not archive.exists():
    print('Downloading public UCI data...', flush=True)
    with urllib.request.urlopen(DATASET_URL, timeout=120) as source, archive.open('wb') as dest:
        import shutil
        shutil.copyfileobj(source, dest)
df = uci_to_hourly(archive)
path = root/'data'/'household_hourly.csv'
df.to_csv(path, index=False)
manifest = {'source': DATASET_URL, 'doi':'10.24432/C58K54', 'license':'CC BY 4.0',
            'authors':'Georges Hebrail and Alice Berard', 'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'hourly_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'hours':len(df), 'observed_hours':int(df.observed.sum()),
            'forward_filled_hours':int((~df.observed & df.load_kw.notna()).sum()),
            'remaining_missing_hours':int(df.load_kw.isna().sum()),
            'first':str(df.timestamp.min()), 'last':str(df.timestamp.max()),
            'timestamp_policy':'Original source wall-clock timestamps; no UTC conversion.',
            'transform':'Mean active power per hour, minimum 45 valid minutes; past-only fill limited to 3 hours.'}
(root/'data'/'provenance.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
