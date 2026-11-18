# Validation — 18 November 2026

## Automated results

- `python -m pytest -q tests/test_pipeline.py`: **13 passed**.
- `node --test tests/web.test.mjs`: **15 passed**.
- `node --check web/app.js` and `node --check web/engine.js`: passed.
- Static website build: passed.

Python tests verify the real hourly CSV checksum/provenance, chronological spacing, training-only load scaler, saved model outputs versus the original report, causal forecasting, valid/invalid uploads, short-gap handling, injection isolation, SQLite deduplication/review status, and all four Streamlit pages.

Node tests verify six Python golden cases (LSTM predictions, next-hour forecasts, autoencoder and forest scores), actual inference on the sample CSV, exclusion of target/future load, rejected invalid dates/offsets/duplicates/numbers, causal gap filling, independently recomputed test MAE and both challenge confusion matrices. Portable model tolerance is 0.00002 absolute.

## Browser verification

The desktop dashboard rendered correctly in Chromium. Tested source switching, 24-hour replay advancement, detector switching, per-alert review, filtering, navigation, CSV import and export. Uploaded the real 240-row sample through the file chooser; the worker completed all three models and produced working chart values and an exported 72-row visible window.

A 390-pixel browser frame tested the responsive overview, model performance and data explorer. Actual document width was 375 pixels after the scrollbar and matched scroll width on those pages: no page-level horizontal overflow. Table scrolling and horizontal navigation are intentional. Physical iOS/Safari testing was not available.

## Limits

The existing measured experiment remains unchanged; no newly fabricated training metrics are substituted. The sample upload is an integration test, not a new generalization result. Current anomaly recall remains weak on sustained high load and near-zero drops. Python latency results are not browser or end-to-end telemetry timing. No physical meter, apartment deployment, external notifications, shared alert database or savings measurement was tested.

Windows/macOS launchers and the optional Docker recipe remain untested on those platforms. The original `reports/test_results.txt` describes an earlier package run; the current test suite is committed under `tests/` and the counts above refer to this update.

## Location monitoring update

Added coordinate/provenance validation and per-meter next-hour summary tests. Browser verification registered the real 240-row UCI sample as an explicitly labeled demonstration placement, confirmed the 2007-02-06 09:00 forecast and 24-hour coverage, exported the location report and opened the correct uploaded dataset in Overview. No Bengaluru meter or utility connection was claimed or tested. Location registrations are session-only. The earlier mobile checks cover the original pages; the new location page was checked on desktop only.
