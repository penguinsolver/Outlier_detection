#!/usr/bin/env python3
"""
outlier_supervised_pipeline.py (train + weekly score)
Supervised outlier detection with:
- Logistic Regression
- Decision Tree
- Random Forest
Includes probability calibration and exports results to Excel.

For non-technical users: only edit USER CONFIG.

Two modes:
1) Train/Evaluate (needs labels in target_col):
   - trains calibrated models
   - evaluates on a hold-out test split
   - saves models + "best threshold" for later scoring

2) Score (no labels needed):
   - loads previously saved calibrated models
   - scores a new weekly dataset
   - outputs predicted outlier labels + calibrated probabilities

Run:
  python outlier_supervised_pipeline.py

Optional CLI:
  python outlier_supervised_pipeline.py --mode score --input weekly_file.csv --output weekly_scored.xlsx
  python outlier_supervised_pipeline.py --logreg on --tree off --rf on
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import joblib
import inspect

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    precision_score, recall_score, f1_score, fbeta_score,
    average_precision_score, roc_auc_score,
    brier_score_loss, log_loss
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =========================
# USER CONFIG (EDIT ONLY THIS)
# =========================
@dataclass
class UserConfig:
    input_path: str = "dummy_outliers.csv"         # training file (labels required) OR weekly scoring file (no labels needed)
    output_excel: str = "outlier_results.xlsx"

    mode: str = "score"                            # "train" or "score"
    artifacts_dir: str = "model_artifacts"         # where trained models + thresholds are saved/loaded

    target_col: str = "is_outlier"                 # only used in TRAIN mode (must be 0/1, 1 = outlier)
    id_col: Optional[str] = "record_id"            # optional; set to None if you don't have it

    # Turn models ON/OFF (use "on" or "off")
    run_logreg: str = "on"                         # Logistic Regression
    run_tree: str = "on"                           # Decision Tree
    run_rf: str = "on"                             # Random Forest


CFG = UserConfig()


# =========================
# ADVANCED SETTINGS (DO NOT EDIT)
# =========================
ADV: Dict[str, Any] = {
    # Splits
    "test_size": 0.20,
    "random_state": 42,

    # Class imbalance
    "class_weight": "balanced",  # "balanced" or None

    # Logistic regression
    "logreg_penalty": "l2",      # "l2" or "l1"
    "logreg_C": 1.0,

    # Decision tree
    "tree_max_depth": 6,
    "tree_min_samples_leaf": 10,

    # Random forest (keep "small-ish" for CPU/storage)
    "rf_n_estimators": 200,
    "rf_max_depth": 10,
    "rf_min_samples_leaf": 5,
    "rf_n_jobs": -1,

    # Calibration
    "calibration_method": "sigmoid",  # "sigmoid" (Platt) or "isotonic"
    "calibration_cv": 5,

    # Extra metrics
    "precision_at_k": 50,
    "fbeta_beta": 2.0,

    # Threshold search grid (for saving "best threshold" in train mode)
    "threshold_grid_step": 0.01,  # 0.01 => 101 thresholds from 0..1
}



def _make_calibrator(estimator, method: str, cv: int) -> CalibratedClassifierCV:
    """
    scikit-learn changed CalibratedClassifierCV arg name from base_estimator -> estimator.
    This keeps the script working across versions.
    """
    sig = inspect.signature(CalibratedClassifierCV)
    if "estimator" in sig.parameters:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


# =========================
# Helpers
# =========================
def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx/.xls")


def _on(x: str) -> bool:
    v = (x or "").strip().lower()
    if v in ("on", "yes", "true", "1"):
        return True
    if v in ("off", "no", "false", "0"):
        return False
    raise ValueError("Model toggles must be 'on' or 'off' (also accepts yes/no, true/false, 1/0).")


def _models_from_toggles(cfg: UserConfig) -> Tuple[str, ...]:
    selected: List[str] = []
    if _on(cfg.run_logreg):
        selected.append("logreg")
    if _on(cfg.run_tree):
        selected.append("tree")
    if _on(cfg.run_rf):
        selected.append("rf")
    if not selected:
        raise ValueError("No models are ON. Set at least one of run_logreg/run_tree/run_rf to 'on'.")
    return tuple(selected)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def _feature_names(prep: ColumnTransformer) -> List[str]:
    try:
        return prep.get_feature_names_out().tolist()
    except Exception:
        return []


def _precision_at_k(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    k = int(min(max(k, 1), len(y_true)))
    idx = np.argsort(-proba)[:k]
    return float(np.mean(y_true[idx] == 1))


def _roc_auc_safe(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def _log_loss_safe(y_true: np.ndarray, proba: np.ndarray) -> float:
    eps = 1e-15
    proba = np.clip(proba, eps, 1 - eps)
    return float(log_loss(y_true, np.column_stack([1 - proba, proba]), labels=[0, 1]))


def _effective_cv(y: np.ndarray, requested_cv: int) -> int:
    vals, counts = np.unique(y, return_counts=True)
    if len(vals) < 2:
        raise ValueError("Training data has only one class. Need both outliers (1) and non-outliers (0).")
    min_class = int(np.min(counts))
    return int(max(2, min(int(requested_cv), min_class)))


def _make_models(models_to_run: Tuple[str, ...]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "logreg" in models_to_run:
        penalty = ADV["logreg_penalty"]
        if penalty == "l1":
            solver = "liblinear"
        else:
            solver = "lbfgs"
        out["logreg"] = LogisticRegression(
            penalty=penalty,
            C=ADV["logreg_C"],
            solver=solver,
            max_iter=2000,
            class_weight=ADV["class_weight"],
        )

    if "tree" in models_to_run:
        out["tree"] = DecisionTreeClassifier(
            max_depth=ADV["tree_max_depth"],
            min_samples_leaf=ADV["tree_min_samples_leaf"],
            random_state=ADV["random_state"],
            class_weight=ADV["class_weight"],
        )

    if "rf" in models_to_run:
        out["rf"] = RandomForestClassifier(
            n_estimators=ADV["rf_n_estimators"],
            max_depth=ADV["rf_max_depth"],
            min_samples_leaf=ADV["rf_min_samples_leaf"],
            random_state=ADV["random_state"],
            n_jobs=ADV["rf_n_jobs"],
            class_weight=ADV["class_weight"],
        )

    return out


def _threshold_grid(step: float) -> np.ndarray:
    step = float(step)
    if step <= 0 or step > 0.5:
        step = 0.01
    n = int(round(1.0 / step))
    grid = np.linspace(0.0, 1.0, n + 1)
    return grid


def _best_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray, step: float) -> Dict[str, float]:
    grid = _threshold_grid(step)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in grid:
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best["f1"]:
            best = {"threshold": float(t), "f1": float(f), "precision": float(p), "recall": float(r)}
    return best


def _artifacts_paths(art_dir: Path, model_name: str) -> Tuple[Path, Path]:
    return (
        art_dir / f"{model_name}_calibrated.joblib",
        art_dir / f"{model_name}_meta.json",
    )


def _write_excel_train(out_path: Path, cfg: UserConfig, metrics_df: pd.DataFrame, preds_df: pd.DataFrame, fi_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame([asdict(cfg)]).to_excel(xw, index=False, sheet_name="user_config_used")
        pd.DataFrame([ADV]).to_excel(xw, index=False, sheet_name="advanced_defaults")
        metrics_df.to_excel(xw, index=False, sheet_name="metrics")
        preds_df.to_excel(xw, index=False, sheet_name="predictions")
        if not fi_df.empty:
            fi_df.to_excel(xw, index=False, sheet_name="feature_importance")


def _write_excel_score(out_path: Path, cfg: UserConfig, scored_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        pd.DataFrame([asdict(cfg)]).to_excel(xw, index=False, sheet_name="user_config_used")
        summary_df.to_excel(xw, index=False, sheet_name="summary")
        scored_df.to_excel(xw, index=False, sheet_name="scored_predictions")


# =========================
# TRAIN MODE
# =========================
def train_mode(cfg: UserConfig) -> None:
    t0_all = time.perf_counter()

    path = Path(cfg.input_path)
    df = _read_table(path)

    if cfg.target_col not in df.columns:
        raise ValueError(f"TRAIN mode requires '{cfg.target_col}' column in the input file.")
    if cfg.id_col and cfg.id_col not in df.columns:
        raise ValueError(f"id_col='{cfg.id_col}' not found. Set id_col=None if you don't have it.")

    y_all = df[cfg.target_col].astype(int).to_numpy()
    if set(np.unique(y_all)) - {0, 1}:
        raise ValueError(f"target_col must be binary 0/1. Found: {sorted(pd.unique(df[cfg.target_col]))}")

    models_to_run = _models_from_toggles(cfg)
    models = _make_models(models_to_run)

    # Build X (drop target + optional id)
    drop_cols = [cfg.target_col]
    if cfg.id_col:
        drop_cols.append(cfg.id_col)
    X_all = df.drop(columns=drop_cols, errors="ignore")

    # Split indices (keeps original row index/id for outputs)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=ADV["test_size"],
        random_state=ADV["random_state"],
        stratify=y_all if len(np.unique(y_all)) > 1 else None
    )

    X_train = X_all.iloc[train_idx].copy()
    y_train = y_all[train_idx]
    X_test = X_all.iloc[test_idx].copy()
    y_test = y_all[test_idx]

    print(f"\n[TRAIN] Loaded {len(df):,} rows. Overall outlier rate: {float(np.mean(y_all)):.3%}")
    print(f"[TRAIN] Train={len(train_idx):,}, Test={len(test_idx):,} | Models: {', '.join(models_to_run)}\n")

    preprocessor = _build_preprocessor(X_train)

    metrics_rows = []
    preds_all = []
    fi_all = []

    # Save artifacts
    art_dir = Path(cfg.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = X_train.columns.tolist()

    for name, clf in models.items():
        print(f"=== [TRAIN] Running model: {name} ===")
        t0 = time.perf_counter()

        base_pipeline = Pipeline(steps=[
            ("prep", preprocessor),
            ("clf", clf),
        ])

        cv_eff = _effective_cv(y_train, int(ADV["calibration_cv"]))
        cal = _make_calibrator(base_pipeline, method=ADV["calibration_method"], cv=cv_eff)
        cal.fit(X_train, y_train)
        proba = cal.predict_proba(X_test)[:, 1]

        # Choose and save "best threshold" (F1) for later scoring
        best = _best_threshold_by_f1(y_test, proba, step=float(ADV["threshold_grid_step"]))
        thr = best["threshold"]
        y_pred = (proba >= thr).astype(int)

        # Metrics at saved threshold
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fbeta = fbeta_score(y_test, y_pred, beta=float(ADV["fbeta_beta"]), zero_division=0)

        ap = average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
        roc = _roc_auc_safe(y_test, proba)
        brier = brier_score_loss(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
        ll = _log_loss_safe(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
        p_at_k = _precision_at_k(y_test, proba, int(ADV["precision_at_k"]))

        runtime_s = time.perf_counter() - t0

        metrics_rows.append({
            "model": name,
            "saved_threshold(best_F1_on_test)": float(thr),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            f"f{ADV['fbeta_beta']:g}": float(fbeta),
            "avg_precision(PR-AUC)": float(ap),
            "roc_auc": float(roc),
            "brier": float(brier),
            "log_loss": float(ll),
            f"precision@{int(ADV['precision_at_k'])}": float(p_at_k),
            "test_outlier_rate": float(np.mean(y_test)),
            "runtime_seconds": float(runtime_s),
            "calibration": f"{ADV['calibration_method']} (cv={cv_eff})",
        })

        # Predictions output (test set)
        pred_out = pd.DataFrame({
            "model": name,
            "row_index_in_input": test_idx,
            "y_true": y_test,
            "p_outlier_calibrated": proba,
            "threshold_used": float(thr),
            "y_pred": y_pred,
        })
        if cfg.id_col:
            pred_out[cfg.id_col] = df.iloc[test_idx][cfg.id_col].to_numpy()
        pred_out = pred_out.sort_values("p_outlier_calibrated", ascending=False).reset_index(drop=True)
        pred_out["rank_by_p_outlier"] = np.arange(1, len(pred_out) + 1)
        preds_all.append(pred_out)

        # Feature importance (fit uncalibrated pipeline for interpretability only)
        base_pipeline.fit(X_train, y_train)
        feat_names = _feature_names(base_pipeline.named_steps["prep"])
        if feat_names:
            if name == "logreg":
                coefs = base_pipeline.named_steps["clf"].coef_.ravel()
                fi = pd.DataFrame({
                    "model": name,
                    "feature": feat_names,
                    "abs_importance": np.abs(coefs),
                    "signed_coef": coefs,
                }).sort_values("abs_importance", ascending=False).head(40)
                fi_all.append(fi)
            elif name in ("tree", "rf"):
                imps = base_pipeline.named_steps["clf"].feature_importances_
                fi = pd.DataFrame({
                    "model": name,
                    "feature": feat_names,
                    "importance": imps,
                }).sort_values("importance", ascending=False).head(40)
                fi_all.append(fi)

        # Save calibrated model + metadata (for weekly scoring)
        model_path, meta_path = _artifacts_paths(art_dir, name)
        joblib.dump(cal, model_path)

        meta = {
            "model": name,
            "saved_threshold": float(thr),
            "feature_columns": feature_cols,
            "id_col": cfg.id_col,
            "target_col": cfg.target_col,
            "calibration": f"{ADV['calibration_method']} (cv={cv_eff})",
            "trained_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"Done: {name} in {runtime_s:.2f}s | saved_threshold={thr:.2f} | F1={f1:.3f} | PR-AUC={ap:.3f}")
        print(f"Saved: {model_path} and {meta_path}\n")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["f1", "avg_precision(PR-AUC)"], ascending=False)
    preds_df = pd.concat(preds_all, ignore_index=True)
    fi_df = pd.concat(fi_all, ignore_index=True) if fi_all else pd.DataFrame()

    out_path = Path(cfg.output_excel)
    _write_excel_train(out_path, cfg, metrics_df, preds_df, fi_df)

    total_s = time.perf_counter() - t0_all
    print(f"[TRAIN] All done. Wrote: {out_path.resolve()}")
    print(f"[TRAIN] Total runtime: {total_s:.2f}s")


# =========================
# SCORE MODE (weekly production)
# =========================
def score_mode(cfg: UserConfig) -> None:
    t0_all = time.perf_counter()

    path = Path(cfg.input_path)
    df = _read_table(path)

    if cfg.id_col and cfg.id_col not in df.columns:
        raise ValueError(f"id_col='{cfg.id_col}' not found in scoring input. Set id_col=None if you don't have it.")

    models_to_run = _models_from_toggles(cfg)
    art_dir = Path(cfg.artifacts_dir)

    print(f"\n[SCORE] Loaded {len(df):,} rows. Models: {', '.join(models_to_run)}")
    print(f"[SCORE] Using artifacts from: {art_dir.resolve()}\n")

    scored_all = []
    summary_rows = []

    for name in models_to_run:
        model_path, meta_path = _artifacts_paths(art_dir, name)
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing artifacts for model '{name}'. Expected:\n- {model_path}\n- {meta_path}\n"
                f"Run TRAIN mode first with the model turned on."
            )

        cal = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_cols = meta["feature_columns"]
        thr = float(meta["saved_threshold"])

        # Check required feature columns exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Scoring input is missing required feature columns for '{name}': {missing}\n"
                f"Your weekly file must contain the same feature columns as training."
            )

        X = df[feature_cols].copy()
        proba = cal.predict_proba(X)[:, 1]
        y_pred = (proba >= thr).astype(int)

        out = pd.DataFrame({
            "model": name,
            "p_outlier_calibrated": proba,
            "threshold_used": thr,
            "predicted_is_outlier": y_pred,
        })
        if cfg.id_col:
            out[cfg.id_col] = df[cfg.id_col].to_numpy()
        out["row_index_in_input"] = np.arange(len(df))
        out = out.sort_values("p_outlier_calibrated", ascending=False).reset_index(drop=True)
        out["rank_by_p_outlier"] = np.arange(1, len(out) + 1)
        scored_all.append(out)

        n_flagged = int(out["predicted_is_outlier"].sum())
        summary_rows.append({
            "model": name,
            "threshold_used": thr,
            "rows_scored": int(len(df)),
            "predicted_outliers": n_flagged,
            "predicted_outlier_rate": float(n_flagged / max(len(df), 1)),
            "calibration": meta.get("calibration", ""),
        })

        print(f"Done: {name} | threshold={thr:.2f} | predicted_outliers={n_flagged}/{len(df)} ({n_flagged/len(df):.3%})")

    scored_df = pd.concat(scored_all, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    out_path = Path(cfg.output_excel)
    _write_excel_score(out_path, cfg, scored_df, summary_df)

    total_s = time.perf_counter() - t0_all
    print(f"\n[SCORE] All done. Wrote: {out_path.resolve()}")
    print(f"[SCORE] Total runtime: {total_s:.2f}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default=None, help="train or score")
    p.add_argument("--input", type=str, default=None, help="Override input_path")
    p.add_argument("--output", type=str, default=None, help="Override output_excel")
    p.add_argument("--artifacts_dir", type=str, default=None, help="Override artifacts_dir")
    p.add_argument("--logreg", type=str, default=None, help="Set logistic regression on/off")
    p.add_argument("--tree", type=str, default=None, help="Set decision tree on/off")
    p.add_argument("--rf", type=str, default=None, help="Set random forest on/off")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = CFG

    if args.mode:
        cfg.mode = args.mode.strip().lower()
    if args.input:
        cfg.input_path = args.input
    if args.output:
        cfg.output_excel = args.output
    if args.artifacts_dir:
        cfg.artifacts_dir = args.artifacts_dir
    if args.logreg is not None:
        cfg.run_logreg = args.logreg
    if args.tree is not None:
        cfg.run_tree = args.tree
    if args.rf is not None:
        cfg.run_rf = args.rf

    mode = (cfg.mode or "").strip().lower()
    if mode == "train":
        train_mode(cfg)
    elif mode == "score":
        score_mode(cfg)
    else:
        raise ValueError("CFG.mode must be 'train' or 'score'.")
