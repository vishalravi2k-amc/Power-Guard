# Website guide

URL: https://power-guard-energy.vishalravi20k.chatgpt.site

The deployed site is public at the owner's request. Anyone with the URL can open it without signing in.

## Overview

1. Read the source banner. Original readings are real UCI observations. The challenge contains deliberately simulated events. Uploaded files are identified by filename.
2. Current load is hourly mean active power in **kW**. Next-hour forecast uses the last 48 usable readings and the target hour's calendar features.
3. Observed energy is the sum of hourly mean kW × one hour for observed hours only. Missing/imputed hours are excluded; coverage is displayed.
4. Choose 24 hours, 3 days or 7 days. Use the date picker, slider, advance button, reset or replay. Replay advances one historical hour every 1.8 seconds.
5. The line forecast is a rolling one-hour prediction, not a multiday forecast. Historical outputs are precomputed from saved trained models; the next-hour card is computed in the browser.

## Alerts

Select Autoencoder or Isolation Forest. Each uses its frozen 99th-percentile validation threshold. The table displays alerts through the replay cursor. A score/limit ratio above one triggers an alert. “Reviewed” is a user's acknowledgement, not proof of a fault or a resolved incident.

Use filters, pagination and the CSV export. Status is scoped by source, detector and timestamp in browser local storage. Uploaded sources use a content hash. It does not sync across devices. Resetting replay hides later events but does not erase review flags. Storage failures are reported and review state then lasts for the session only.

## CSV analysis

Download the included sample CSV, choose it with the file picker, wait for analysis, and press **Open analysis**. The worker calculates actual saved LSTM forecasts, autoencoder errors and Isolation Forest scores. Progress and cancel controls are available.

- Required columns: `timestamp,load_kw`.
- 72–50,000 rows, up to 10 MB, spanning at most five years.
- ISO local timestamps at exact hours, no timezone offsets.
- Unique dates and finite non-negative hourly mean kW values.
- Extra columns are ignored. Quoted plain field values are accepted; embedded commas/newlines in field values are not supported.
- Missing hours are inserted, forward-filled for up to three hours, then left missing. At least 49 consecutive usable hours are required.
- First 48 hours warm up the forecast; first 24 hours warm up anomaly features.
- The site holds uploaded data in memory. It does not send readings to a server, retrain models or claim generalization to the new site.

The included sample is a real UCI subset, intended to test file handling, not to serve as an independent evaluation dataset.

## Model performance

Metrics come from the existing `reports/metrics.json` experiment. Tests independently recompute forecast MAE and both injected-event confusion matrices. The displayed Python latency is from the original CPU benchmark; it is not a browser or network latency estimate. Full scope and failure cases are in `reports/EVALUATION.md`.

## Implementation details

`engine.js` implements the PyTorch LSTM gate equations, dense ReLU autoencoder and sklearn Isolation Forest path-length scoring using exported JSON. Six independently produced Python cases compare one-hour forecasts, both detector scores and next-hour predictions with a 0.00002 absolute tolerance. JavaScript uses double precision with float32 anomaly inputs; small numerical differences near thresholds are possible.

The UI uses ES modules, SVG charts and a module web worker. It has no JavaScript CDN dependency. Google Fonts are optional; system fonts are the fallback. Browser testing used Chromium at desktop and a narrow 390-pixel frame. Physical iPhone/Safari execution was not available; the implementation uses standard features available in Safari 16, with the `findLastIndex` call replaced by a compatible loop.

## Update and deploy

1. Edit files in `web/`; change the Python pipeline if model/data changes are needed.
2. After changing model artifacts, run `python scripts/export_web.py`.
3. Run `python -m pytest -q` and `npm test`, then `npm run build`.
4. Review the website and commit/push the source to GitHub.
5. Publish the matching `dist/` to the existing Sites project. Its ID is in `.openai/hosting.json`; do not create a duplicate project.

GitHub CI runs tests and builds the static bundle. It does not change Site permissions or deploy public Pages. A Pages deployment is an optional separate hosting choice; follow [GitHub's Pages guide](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site) and [custom workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) only when public hosting is intended.

## Location monitoring

Upload one hourly CSV per meter in Data explorer, then open Location monitoring. Enter a site name, area and meter GPS coordinates and specify whether the readings are user-supplied local readings or a demonstration/sample mapped to a location. The built-in UCI sample must be labelled as demonstration if placed in Bengaluru. Coordinates are supplied by the user; they are not geocoded or independently verified.

Register additional distinct meter files to compare their last 24 hours. Identical dataset content cannot be registered as multiple meters. The table shows forecast timestamp, flagged/scored hours, observed coverage, and review status. Missing scores yield “Incomplete evidence”. “No flags in window” is not a safe-area or reliability determination. The next forecast is one hour after the final source timestamp, not a live current-time prediction.

Open analysis to inspect a meter's detailed charts and alerts, or View GPS pin to send those coordinates to OpenStreetMap in a new tab. Export location report saves the summary and provenance as JSON. Sites and readings remain in session memory; the export does not contain the underlying CSV data. Different time periods and different load scales are not directly comparable; local calibration remains necessary. Feeder/transformer topology, outage/voltage signals and verified fault labels are required to move from meter anomalies to grid-fault localization.
