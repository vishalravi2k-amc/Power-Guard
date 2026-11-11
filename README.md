# Power Guard — working review prototype

**AI-Driven Micro-Grid Energy Load Forecasting and Anomaly Detection for Urban Apartments**  
AMC Engineering College (Autonomous), Department of CSE (AI & ML)  
Team T2: P Vishal Ravi, Sagar J, Jeevan U Acharya, Varshith M  
Guide: Dr. Nandeeswar S B

This project implements the software workflow in the supplied November 2026 synopsis. It includes a trained LSTM, a trained autoencoder, Isolation Forest, a Streamlit dashboard, historical replay, CSV analysis, SQLite alerts, baseline comparisons and reproducible tests. It runs on a computer with Python; no sensor purchase is needed for this version.

**First action: open the project on a laptop or desktop.** An iPhone can download the ZIP and view documentation, but cannot run this Python application directly.

## Start on Windows

1. Install **Python 3.12 (64-bit)** from https://www.python.org/downloads/ with the Python launcher enabled.
2. Extract the entire ZIP. Open the `power_guard` folder, where `app.py` is located.
3. Double-click **`start_windows.bat`**.
4. The first launch creates a virtual environment and downloads packages. PyTorch is a large download. Keep the command window open.
5. Open **http://localhost:8501** if the browser does not open automatically.
6. Later launches reuse the installed environment. Internet is not needed for the bundled dataset/model demo after installation.

Do this setup before review day. The saved models are already trained; do not retrain just to launch the demo.

Manual Windows commands, from this folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
```

## Start on macOS or Linux

Use Python 3.12. From the extracted project folder:

```bash
bash start_mac_linux.sh
```

Or create `.venv`, install requirements with its Python and run `python -m streamlit run app.py`. On Linux, the launcher installs the CPU build of PyTorch. macOS uses its platform build. The Python application was tested on Linux; the Windows/macOS launchers are provided but have not been executed on those operating systems.

## Demonstrate in this order

1. **Monitor & replay / Original UCI readings:** explain actual load, one-hour-ahead forecast and score threshold.
2. **Injected event challenge:** show a known spike marked with a hollow amber circle and any detector alert as a coral diamond. Advance 24 hours or enable auto replay.
3. Switch **Autoencoder / Isolation Forest** to compare behavior. Download the alert CSV.
4. **Model results:** show measured test errors, controlled detection metrics, missed event types and CPU latency.
5. **Upload readings:** download and re-upload the included sample. Read the domain-shift note.
6. **Project review:** show the architecture and explain the next development steps.

See `docs/REVIEW_GUIDE.md` for a short two-presenter script and viva answers. `reports/EVALUATION.md` contains the measured experiment report.

## What is complete, and what remains

| Implemented in this package | Later work |
|---|---|
| UCI hourly preprocessing with explicit quality flags | Representative local apartment data |
| 48-hour LSTM input and one-hour prediction | Longer horizons and a GRU comparison |
| Persistence and previous-day benchmarks | Additional sites, seasons and repeated-seed experiments |
| Dense temporal-feature autoencoder and Isolation Forest | Better sustained/low-load anomaly detection |
| Validation-calibrated thresholds | Site-specific calibration and verified event labels |
| Historical replay, CSV import and local alert register | A live meter connector and operational notifications |
| Test results and documented limitations | Field evaluation of usefulness and savings |

This is an **advisory software prototype**, not a protection controller. It does not operate relays, diagnose physical faults, dispatch batteries, or measure energy savings.

## Folder map

| Path | Purpose |
|---|---|
| `app.py` | Dashboard pages, controls, charts and replay |
| `powerguard/data.py` | UCI conversion, validation, causal features and injected events |
| `powerguard/models.py` | PyTorch LSTM and autoencoder definitions |
| `powerguard/inference.py` | Saved-model loading and predictions |
| `powerguard/alerts.py` | SQLite logging and review status |
| `scripts/download_data.py` | Rebuild hourly data from the UCI archive |
| `scripts/train.py` | Training, calibration, test evaluation and latency measurement |
| `scripts/build_report.py` | Rebuild the readable evaluation report |
| `artifacts/` | Weights, Isolation Forest and fitted preprocessing parameters |
| `data/` | Hourly UCI data, provenance and upload sample |
| `reports/` | Forecasts, challenge results, metrics and test output |
| `tests/` | Causality, validation, inference, persistence and app tests |
| `runtime/alerts.db` | Created locally when the app runs; not included in the distribution |

## Reproduce training

The processed dataset is included. Run from the project folder with the virtual environment's Python:

```bash
python scripts/train.py
python scripts/build_report.py
```

Default: seed 42, up to 25 LSTM epochs and 30 autoencoder epochs, early stopping after five non-improving validation epochs. This rebuilds model and report files. Stop and restart Streamlit afterward to clear cached models. Exact floating-point results and latency can vary with platform and package versions.

To reproduce preprocessing from the original public archive:

```bash
python scripts/download_data.py
python scripts/train.py
python scripts/build_report.py
```

To use an already downloaded official archive:

```bash
python scripts/download_data.py --archive /path/to/individual_household.zip
```

No original minute-level archive is included. The smaller hourly derivative is included with attribution, provenance and a checksum. Uploaded CSVs are analyzed by the saved UCI model; uploading does **not** automatically retrain the model or establish suitability for another site.

## Run tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests require the included model/data/report artifacts. App tests write example events to `runtime/alerts.db`. For a fresh demo, stop the app and remove that local database only if those practice alerts are no longer needed. Reset replay changes the cursor and does not delete saved alerts.

## CSV format

```csv
timestamp,load_kw
2026-11-01 00:00:00,0.72
2026-11-01 01:00:00,0.61
```

- Supply at least 72 hourly rows for one site, up to 50,000 rows / 10 MB.
- Each value is **mean active power in kW**, not a cumulative meter counter or kWh reading.
- Use unique exact-hour timestamps, consistently in the site's local wall-clock time without timezone offsets.
- Missing hourly slots are marked; forward fill is limited to three hours. Long gaps interrupt model eligibility.
- The app rejects negative, nonnumeric or nonfinite loads, duplicates, subhourly times and mixed/offset timestamps.
- No fault labels are required for inference. Uploads do not produce fault precision/recall because no verified labels are available.

## Troubleshooting

| Symptom | Action |
|---|---|
| `py` not found on Windows | Install Python 3.12 with its launcher; reopen the command window. |
| `No matching distribution` | Check 64-bit Python 3.12 and package connectivity. Do not silently change the scikit-learn version used by the saved forest. |
| Import error | Run the launcher or use `.venv` Python; avoid mixing global Python and the virtual environment. |
| Port 8501 in use | Run `python -m streamlit run app.py --server.port 8502`. |
| Next forecast unavailable | The last 48 hours contain a long gap or insufficient history. Use another range/file. |
| Old results after retraining | Stop Streamlit with Ctrl+C and restart. |
| Many alerts after upload | The new site's consumption differs from the training household. Collect local data and retrain/calibrate. |
| Package install interrupted | Launch again. The ready marker is written only after installation succeeds. |

## Optional Docker

```bash
docker build -t power-guard .
docker run --rm -p 127.0.0.1:8501:8501 -v powerguard-alerts:/app/runtime power-guard
```

The Docker recipe is supplied as an optional path; the container build was not tested in this environment. There is no login system. The default launcher binds to the local computer. Do not expose this prototype publicly without authentication and deployment review.

## Dataset attribution

Georges Hebrail and Alice Berard (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C58K54. Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/.

The included CSV is a modified hourly derivative: minute power aggregated into hourly means, low-coverage hours marked, and short gaps filled from the past. It describes one French household, not measurements from AMC or an Indian apartment. Source timestamps retain their original wall-clock representation. See `data/provenance.json` and `DATA_LICENSE.md`.
