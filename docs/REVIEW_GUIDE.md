# Team T2 review presentation

## Five-minute demonstration

- **Vishal, 1 minute:** Introduce the problem, the UCI household proxy, kW versus kWh, and actual versus predicted load.
- **Vishal, 1 minute:** Play historical replay, change the range, and show next-hour forecasts. Explain 48-hour input and one-hour horizon.
- **Sagar, 1 minute:** Choose the injected-event challenge and replay through 27 April 2010, 18:00. Open alerts, switch detectors, mark one reviewed and export the register.
- **Sagar, 1 minute:** Show model comparisons. The LSTM reduces MAE by 11.3% versus persistence. Discuss poor sustained/low-load anomaly recall honestly.
- **Both, 1 minute:** Import the sample CSV, show actual model analysis, and explain the architecture and next field-validation steps.

## Viva points

**Why chronological splitting?** Random splitting can mix future and past conditions and exaggerate time-series performance. Training precedes validation, which precedes testing.

**How is leakage avoided?** Training-only scaling; past-only filling; rolling windows exclude the target; thresholds selected on original validation data; injected test events are not training examples.

**Why use baselines?** A neural model is useful only if it improves on simple credible alternatives at the same timestamps.

**Is an alert a fault?** No. It is an unusual feature pattern requiring review. UCI has no verified fault labels.

**What is the project contribution?** A documented integration of forecasting, anomaly scoring, transparent evaluation and reviewable dashboard outputs. No novelty claim about inventing the algorithms is needed.

**Is the website pretending to run ML?** No. Historical outputs are model-produced. New next-hour predictions and uploaded-file analysis execute exported trained weights and trees in JavaScript, checked against Python reference results.

**What remains?** Representative apartment data, site calibration, verified event labels, live meter integration, multiuser persistence and field evaluation. These are outside this household-proxy software prototype.

## Location-aware extension

Present this as “Power Guard: Location-aware Energy Forecasting and Anomaly Monitoring”. Upload at least two distinct meter datasets, provide their areas/GPS coordinates and show the timestamped comparison. Explain that a location pin identifies a supplied meter, not a model-discovered fault. Use explicit demonstration provenance for any UCI data assigned to Bengaluru coordinates. Avoid claiming “unsafe neighbourhoods”, real outages or an exact failed transformer from household kW alone.
