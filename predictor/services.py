"""Prediction service: the ONE place that loads the trained pipelines and scores students.
Both the HTML views and the Ninja API call into this module."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.utils import timezone

from .mlkit import ALL_FEATURES, EARLY_FEATURES


class ModelNotTrained(RuntimeError):
    pass


def _paths():
    d = settings.ML_ARTIFACT_DIR
    return d / "model_early.joblib", d / "model_full.joblib", d / "metrics.json"


@lru_cache(maxsize=2)
def _load(stamp):
    early, full, meta = _paths()
    return {
        "early": joblib.load(early),
        "full": joblib.load(full),
        "meta": json.loads(meta.read_text()),
    }


def _artifacts():
    paths = _paths()
    if not all(p.exists() for p in paths):
        raise ModelNotTrained("No trained model found. Run: python manage.py train_model")
    # Keyed on file mtimes, so retraining is picked up without restarting the server.
    try:
        return _load(tuple(p.stat().st_mtime_ns for p in paths))
    except Exception as exc:  # e.g. models pickled under a different scikit-learn version
        raise ModelNotTrained(
            f"Could not load the saved model ({type(exc).__name__}). "
            "Retrain it on this machine: python manage.py train_model") from exc


def get_metrics():
    """Training metrics dict, or None if the model has not been trained yet."""
    try:
        return _artifacts()["meta"]
    except ModelNotTrained:
        return None


def _level(prob, threshold):
    if prob >= threshold:
        return "high"
    if prob >= threshold * 0.6:
        return "medium"
    return "low"


def predict_frame(df):
    """Score a DataFrame holding the ALL_FEATURES columns (G1/G2 may be NaN).

    Rows with both G1 and G2 use the 'full' model; all others use the 'early-warning' model.
    """
    art = _artifacts()
    df = df.copy()
    for col in ("G1", "G2"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    has_grades = df[["G1", "G2"]].notna().all(axis=1).to_numpy()

    probs = np.zeros(len(df))
    used = np.array(["early"] * len(df), dtype=object)
    if (~has_grades).any():
        probs[~has_grades] = art["early"].predict_proba(df[~has_grades][EARLY_FEATURES])[:, 1]
    if has_grades.any():
        probs[has_grades] = art["full"].predict_proba(df[has_grades][ALL_FEATURES])[:, 1]
        used[has_grades] = "full"

    results = []
    for p, m in zip(probs, used):
        thr = art["meta"]["variants"][m]["threshold"]
        results.append({
            "probability": round(float(p), 4),
            "at_risk": bool(p >= thr),
            "risk_level": _level(p, thr),
            "model_used": str(m),
            "threshold": thr,
        })
    return results


def predict_one(features):
    """features: dict with the ALL_FEATURES keys (G1/G2 optional)."""
    row = {f: features.get(f) for f in ALL_FEATURES}
    return predict_frame(pd.DataFrame([row]))[0]


def student_row(student):
    return {f: getattr(student, f) for f in ALL_FEATURES}


def apply_predictions(students):
    """Score Student instances and set the prediction fields in place (caller saves)."""
    students = list(students)
    if not students:
        return students
    df = pd.DataFrame([student_row(s) for s in students])
    now = timezone.now()
    for s, r in zip(students, predict_frame(df)):
        s.risk_probability = r["probability"]
        s.at_risk = r["at_risk"]
        s.risk_level = r["risk_level"]
        s.model_used = r["model_used"]
        s.predicted_at = now
    return students
