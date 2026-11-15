# Run this in Colab in the SAME session right after readmission_diabetes.py finishes.
# It writes app_assets.json, which app.py needs (choices, defaults, presets, test results).
import json

assets = {
    "ui_cols": UI_COLS,
    "num_cols": NUM_COLS,
    "model_features": MODEL_FEATURES,
    "choices": {c: sorted(X_all[c].astype(str).unique().tolist())
                for c in UI_COLS if c not in NUM_COLS},
    "defaults": DEFAULTS,
    "presets": PRESETS,
    "preset_truth": PRESET_TRUTH,
    "threshold": THRESHOLD,
    "test_labels": [int(v) for v in y_test.values],
    "test_probs": [float(v) for v in p_test_prob],
}
with open("app_assets.json", "w") as f:
    json.dump(assets, f)
print("Wrote app_assets.json with", len(assets["choices"]), "categorical fields")
