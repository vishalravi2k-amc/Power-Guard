"""Generate an evaluation report from actual saved metrics, never hand-entered scores."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'reports'/'metrics.json').read_text())
a=json.loads((ROOT/'artifacts'/'metadata.json').read_text())
p=json.loads((ROOT/'data'/'provenance.json').read_text())
f=m['forecast']
improvement=(f[1]['mae_kw']-f[0]['mae_kw'])/f[1]['mae_kw']*100
lines=[
'# Power Guard — measured evaluation, prototype v1','',
'Prepared from the saved experiment artifacts. These are measured results from one run, not targets.','',
'## Data and evaluation split','',
f'- Source: UCI Individual Household Electric Power Consumption, DOI https://doi.org/10.24432/C58K54.',
f'- Hourly slots: {p["hours"]:,}; observed: {p["observed_hours"]:,}; short fills: {p["forward_filled_hours"]:,}; remaining missing: {p["remaining_missing_hours"]:,}.',
f'- Training: {p["first"]} to {a["train_last"]}.',
f'- Validation: {a["validation_first"]} to {a["validation_last"]}.',
f'- Test: {a["test_first"]} to {a["test_last"]}.',
f'- Usable forecast samples: {a["forecast_samples"]["train"]:,} training, {a["forecast_samples"]["validation"]:,} validation, {a["forecast_samples"]["test"]:,} test.',
'- Models use past-only windows and training-only scaling; thresholds use original validation data.',
'- Data describes one household in France. This experiment does not validate an apartment micro-grid.','',
'## Next-hour forecasting','',
'| Model | MAE (kW) | RMSE (kW) | Test hours |','|---|---:|---:|---:|']
for row in f: lines.append(f'| {row["model"]} | {row["mae_kw"]:.4f} | {row["rmse_kw"]:.4f} | {row["samples"]:,} |')
lines += ['',f'The LSTM reduces MAE by {improvement:.1f}% relative to persistence in this test period. This is a relative error reduction, not an “accuracy percentage.” All three methods use the same eligible timestamps.',
'','Forecasts are rolling one-step estimates, with earlier observed test readings available. A prediction does not use its own target or later load. No confidence interval or multiple-seed robustness study has been established.','',
'## Anomaly detection: controlled injected-event recovery','',
'Three known modifications were applied to test readings only: one-hour +5 kW spikes, eight-hour +3 kW increases and four-hour 95% drops. No simulated test event was used to fit a model or choose its threshold.','',
'| Detector | Precision | Recall | F1 | TP | FP | FN | TN |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,row in m['injected_event_detection'].items():
    lines.append(f'| {name} | {row["precision"]:.3f} | {row["recall"]:.3f} | {row["f1"]:.3f} | {row["tp"]} | {row["fp"]} | {row["fn"]} | {row["tn"]} |')
lines += ['',m['injection_label_scope'],'',
'Each hour is evaluated independently. An alert on an unmodified hour counts as a false positive for this challenge, including delayed effects following an injection. These results are not verified real-fault precision or recall.','',
'| Injected event | Modified hours | Autoencoder recall | Isolation Forest recall |','|---|---:|---:|---:|']
for name,row in m['injected_event_types'].items():
    lines.append(f'| {name} | {row["hours"]} | {row["autoencoder_recall"]:.1%} | {row["isolation_forest_recall"]:.1%} |')
u=m['unmodified_test_alerts']
lines += ['',f'On {u["hours"]:,} usable unmodified test hours, the autoencoder flags {u["autoencoder"]} and Isolation Forest flags {u["isolation_forest"]}. Without verified fault labels, these counts are neither confirmed faults nor a measured real-world false-positive rate.','',
'The initial detectors are much better at isolated high spikes than sustained increases or near-zero drops. Low-load consumption may look normal to an unsupervised detector. A forecast-residual feature, sequence model and local event labels are priorities for the next iteration.','',
'## Calibration and training','',
f'- Autoencoder threshold: {a["ae_threshold"]:.6f}; Isolation Forest threshold: {a["if_threshold"]:.6f}.',
'- Both thresholds are the 99th percentile of their original validation scores.',
f'- Anomaly training rows retained: {a["anomaly_training_samples"]:,}; screened out: {a["screened_out_training_samples"]:,}. Screening is heuristic, not verified normality.',
f'- Seed: {a["seed"]}; LSTM epochs run: {len(m["epoch_history"]["lstm"])}; autoencoder epochs run: {len(m["epoch_history"]["autoencoder"])}.',
f'- Full training and report generation in this run: {m["training_seconds"]:.1f} seconds. Timing varies by machine.','',
'## Processing latency','',
f'- Warm repeated calls: {m["latency"]["iterations"]}.',
f'- Median: {m["latency"]["median_ms"]:.2f} ms; 95th percentile: {m["latency"]["p95_ms"]:.2f} ms.',
f'- Scope: {m["latency"]["scope"]}',
f'- Architecture: {m["latency"]["machine"]}; configured PyTorch threads: {m["latency"]["threads"]}.',
'- This is a local benchmark after receipt of an hourly reading. It does not measure sensor sampling delay, end-to-end alert delivery or performance on the presentation laptop.','',
'## Reproducibility and next evidence','',
'See `artifacts/metadata.json` for versions and scaling, `data/provenance.json` for checksums, `reports/training_console.txt` for epoch output and `reports/test_results.txt` for the test run. Full forecast and challenge CSVs are included. Rebuild with `python scripts/train.py` followed by `python scripts/build_report.py`.','',
'Before a later review, collect local data, improve detection using validation experiments, reserve a new final test period and report repeated-seed variation. No energy savings, hardware protection effectiveness or site deployment has been demonstrated.','']
(ROOT/'reports'/'EVALUATION.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote reports/EVALUATION.md')
