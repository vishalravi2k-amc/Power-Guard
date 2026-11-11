# Validation notes

The supplied run passed 19 automated tests, including:

- Causal historical windows and preceding-only rolling features.
- Fitting the saved load scaler on training observations only.
- Exclusion of imputed targets.
- Injection isolation to the test period.
- CSV rejection for malformed numeric values, timestamps, duplicate rows and short inputs.
- Short-gap filling without using future readings.
- Saved-model outputs matching the supplied forecast and detector CSVs.
- Recomputed regression metrics and controlled-challenge confusion counts.
- Persistent SQLite alert deduplication and reviewed status.
- Streamlit navigation, scenario switching, replay advancement and detector switching using AppTest.

See `test_results.txt` for the test run. A host-environment pytest plugin deprecation warning is unrelated to the project.

The project was trained and tested with Python 3.12 on Linux. A full visual browser review could not be completed because the available browser could not reach the local application and a local browser process could not start under the host's process restrictions. Streamlit's application tests exercised the Python UI without exceptions. Validate the rendered charts and upload interaction on the presentation laptop before the review.

Windows/macOS launcher execution, Docker build, physical meter integration, actual apartment data and real fault labels have not been validated. Uploaded-data validation and inference are tested as Python functions; no end-to-end browser file-upload test was completed.
