# Power Guard — measured evaluation, prototype v1

Prepared from the saved experiment artifacts. These are measured results from one run, not targets.

## Data and evaluation split

- Source: UCI Individual Household Electric Power Consumption, DOI https://doi.org/10.24432/C58K54.
- Hourly slots: 34,589; observed: 34,151; short fills: 30; remaining missing: 408.
- Training: 2006-12-16 17:00:00 to 2009-09-20 12:00:00.
- Validation: 2009-09-20 13:00:00 to 2010-04-24 16:00:00.
- Test: 2010-04-24 17:00:00 to 2010-11-26 21:00:00.
- Usable forecast samples: 23,877 training, 5,006 validation, 4,884 test.
- Models use past-only windows and training-only scaling; thresholds use original validation data.
- Data describes one household in France. This experiment does not validate an apartment micro-grid.

## Next-hour forecasting

| Model | MAE (kW) | RMSE (kW) | Test hours |
|---|---:|---:|---:|
| LSTM | 0.3463 | 0.4946 | 4,884 |
| Persistence | 0.3903 | 0.5896 | 4,884 |
| Previous day | 0.5158 | 0.7667 | 4,884 |

The LSTM reduces MAE by 11.3% relative to persistence in this test period. This is a relative error reduction, not an “accuracy percentage.” All three methods use the same eligible timestamps.

Forecasts are rolling one-step estimates, with earlier observed test readings available. A prediction does not use its own target or later load. No confidence interval or multiple-seed robustness study has been established.

## Anomaly detection: controlled injected-event recovery

Three known modifications were applied to test readings only: one-hour +5 kW spikes, eight-hour +3 kW increases and four-hour 95% drops. No simulated test event was used to fit a model or choose its threshold.

| Detector | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Autoencoder | 0.292 | 0.235 | 0.260 | 42 | 102 | 137 | 4651 |
| Isolation Forest | 0.652 | 0.335 | 0.443 | 60 | 32 | 119 | 4721 |

Pointwise recovery of known injected modifications. Unmodified readings are background, not verified fault-free labels. These scores are not real fault accuracy.

Each hour is evaluated independently. An alert on an unmodified hour counts as a false positive for this challenge, including delayed effects following an injection. These results are not verified real-fault precision or recall.

| Injected event | Modified hours | Autoencoder recall | Isolation Forest recall |
|---|---:|---:|---:|
| Spike | 15 | 86.7% | 100.0% |
| Sustained high load | 112 | 23.2% | 40.2% |
| Near-zero drop | 52 | 5.8% | 0.0% |

On 4,932 usable unmodified test hours, the autoencoder flags 31 and Isolation Forest flags 12. Without verified fault labels, these counts are neither confirmed faults nor a measured real-world false-positive rate.

The initial detectors are much better at isolated high spikes than sustained increases or near-zero drops. Low-load consumption may look normal to an unsupervised detector. A forecast-residual feature, sequence model and local event labels are priorities for the next iteration.

## Calibration and training

- Autoencoder threshold: 1.402969; Isolation Forest threshold: 0.619643.
- Both thresholds are the 99th percentile of their original validation scores.
- Anomaly training rows retained: 23,512; screened out: 461. Screening is heuristic, not verified normality.
- Seed: 42; LSTM epochs run: 14; autoencoder epochs run: 30.
- Full training and report generation in this run: 141.6 seconds. Timing varies by machine.

## Processing latency

- Warm repeated calls: 100.
- Median: 59.87 ms; 95th percentile: 202.79 ms.
- Scope: CPU: feature construction, LSTM and both detectors for one latest hour; excludes model loading, network, SQLite and dashboard rendering.
- Architecture: x86_64; configured PyTorch threads: 2.
- This is a local benchmark after receipt of an hourly reading. It does not measure sensor sampling delay, end-to-end alert delivery or performance on the presentation laptop.

## Reproducibility and next evidence

See `artifacts/metadata.json` for versions and scaling, `data/provenance.json` for checksums, `reports/training_console.txt` for epoch output and `reports/test_results.txt` for the test run. Full forecast and challenge CSVs are included. Rebuild with `python scripts/train.py` followed by `python scripts/build_report.py`.

Before a later review, collect local data, improve detection using validation experiments, reserve a new final test period and report repeated-seed variation. No energy savings, hardware protection effectiveness or site deployment has been demonstrated.
