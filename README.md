# Diabetes 30-Day Readmission Risk Portal

Veritas AI International High School Hackathon 2026 · AI + Health

## Live app
[paste Hugging Face Space link] (temporary Gradio link: https://2370c40e9ac2956a4b.gradio.live)

## Results (held-out test set, 20,075 encounters)
- ROC-AUC 0.667, PR-AUC 0.231 (random guessing: 0.115)
- After isotonic calibration: average predicted risk 0.114 vs observed 0.115; Brier 0.097
- Threshold 0.12 (chosen on validation data)

This project is a prototype and is not medical advice.

## Files
- `notebook.ipynb`: full training run with outputs
- `app.py`: Gradio app (loads saved model, no retraining)
- `requirements.txt`: pinned library versions
- `readmission_diabetes_cache.joblib`, `calibrator.joblib`, `app_assets.json`: saved model and assets
- `Project_Report_Diabetes_Readmission.docx`: full report
- `Credits.txt`: datasets, libraries and AI tools

## Data
UCI Diabetes 130-US Hospitals (1999–2008). The notebook downloads it automatically.
