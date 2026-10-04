"""Offline training: compare models with cross-validation, tune the alert threshold,
evaluate on a held-out test set, and save pipelines + metrics + plots.

Run it with:  python manage.py train_model
"""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate,
    train_test_split,
)

from .mlkit import (  # noqa: E402
    ALL_FEATURES, EARLY_FEATURES, NUMERIC_FEATURES, PASS_MARK, build_pipeline, read_table,
)

SEED = 42
TARGET_RECALL = 0.80  # share of truly at-risk students the alert threshold should catch

MODEL_FACTORIES = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=SEED),
}


def load_dataset(path):
    df = read_table(path)
    missing = [c for c in ALL_FEATURES + ["G3"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["at_risk"] = (df["G3"] < PASS_MARK).astype(int)
    return df


def data_quality(df):
    """Objective (ii): document what cleaning will do before training."""
    num = df[NUMERIC_FEATURES]
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    outliers = {}
    for col in NUMERIC_FEATURES:
        if iqr[col] > 0:
            n = int(((num[col] < q1[col] - 1.5 * iqr[col]) | (num[col] > q3[col] + 1.5 * iqr[col])).sum())
            if n:
                outliers[col] = n
    return {
        "rows": int(len(df)),
        "missing_cells": int(df[ALL_FEATURES].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "outliers_clipped": outliers,
    }


def _scores(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }


def _plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    ax.imshow(cm, cmap="Blues")
    top = max(max(r) for r in cm) or 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=15,
                    color="white" if cm[i][j] > top / 2 else "black")
    ax.set_xticks([0, 1], ["Pass", "At risk"])
    ax.set_yticks([0, 1], ["Pass", "At risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_importance(ranking, title, path):
    names = [r[0] for r in ranking][::-1]
    vals = [r[1] for r in ranking][::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.barh(names, vals, color="#2f6fed")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _train_variant(df, name, features, img_dir, log):
    X, y = df[features], df["at_risk"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    cv_results = {}
    for model_name, make in MODEL_FACTORIES.items():
        res = cross_validate(build_pipeline(make(), features), X_tr, y_tr,
                             cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[model_name] = {
            m: {"mean": round(float(res[f"test_{m}"].mean()), 4),
                "std": round(float(res[f"test_{m}"].std()), 4)}
            for m in scoring
        }
        log(f"  [{name}] {model_name}: AUC={cv_results[model_name]['roc_auc']['mean']:.3f} "
            f"recall={cv_results[model_name]['recall']['mean']:.3f}")

    best = max(cv_results, key=lambda m: cv_results[m]["roc_auc"]["mean"])

    # Alert threshold: the HIGHEST cut-off that still catches >= TARGET_RECALL of at-risk
    # students on out-of-fold training predictions. Missing a struggling student costs more
    # than a false alarm, and "highest" keeps the alert list as short as that goal allows.
    # (Maximising F2 was tried first: with a weak model it degenerates into "flag everyone".)
    oof = cross_val_predict(
        build_pipeline(MODEL_FACTORIES[best](), features), X_tr, y_tr,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    grid = np.arange(0.05, 0.96, 0.01)
    reachable = [t for t in grid if recall_score(y_tr, (oof >= t).astype(int)) >= TARGET_RECALL]
    threshold = round(float(max(reachable) if reachable else grid[0]), 2)

    pipe = build_pipeline(MODEL_FACTORIES[best](), features).fit(X_tr, y_tr)
    test = _scores(y_te, pipe.predict_proba(X_te)[:, 1], threshold)

    imp = permutation_importance(pipe, X_te, y_te, scoring="roc_auc",
                                 n_repeats=20, random_state=SEED, n_jobs=-1)
    ranking = sorted(zip(features, imp.importances_mean), key=lambda t: -t[1])[:10]

    label = "early-warning (no grades)" if name == "early" else "with G1/G2 grades"
    _plot_confusion(test["confusion"], f"Hold-out test: {label}", img_dir / f"{name}_confusion.png")
    _plot_importance(ranking, f"Top features: {label}", img_dir / f"{name}_importance.png")

    # Deployment model: refit on all rows with the settings chosen above.
    final = build_pipeline(MODEL_FACTORIES[best](), features).fit(X, y)

    info = {
        "label": label,
        "features": features,
        "best_model": best,
        "threshold": threshold,
        "cv": cv_results,
        "test": test,
        "importance": [{"feature": f, "importance": round(float(v), 4)} for f, v in ranking],
        "confusion_img": f"predictor/img/{name}_confusion.png",
        "importance_img": f"predictor/img/{name}_importance.png",
    }
    log(f"  [{name}] best={best} threshold={threshold} "
        f"test recall={test['recall']:.3f} acc={test['accuracy']:.3f}")
    return final, info


def train(data_path, artifact_dir, img_dir, log=print):
    artifact_dir, img_dir = Path(artifact_dir), Path(img_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    log(f"Loaded {len(df)} rows; at-risk rate {df['at_risk'].mean():.1%} (G3 < {PASS_MARK})")

    out = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": f"G3 < {PASS_MARK}",
        "threshold_rule": f"highest cut-off catching >= {TARGET_RECALL:.0%} of at-risk students (out-of-fold)",
        "at_risk_rate": round(float(df["at_risk"].mean()), 4),
        "quality": data_quality(df),
        "variants": {},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, feats in {"early": EARLY_FEATURES, "full": ALL_FEATURES}.items():
            log(f"Training variant '{name}' ({len(feats)} features)...")
            pipe, info = _train_variant(df, name, feats, img_dir, log)
            joblib.dump(pipe, artifact_dir / f"model_{name}.joblib")
            out["variants"][name] = info

    (artifact_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    log(f"Saved models and metrics to {artifact_dir}")
    return out
