import json
import os
import pickle
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from src.model import brier_score
from src.model_elo_v2 import prob_from_elo_diff
from src.model_logreg_v2 import LogRegV2Config, predict_logreg_v2, prepare_logreg_frame

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


@dataclass
class StackConfig:
    train_cutoff: int = 2022
    oof_start_season: int | None = None
    oof_end_season: int | None = None
    random_state: int = 42
    use_catboost: bool = True
    use_sigmoid_calibration: bool = True
    calibration_min_gain: float = 0.0003
    meta_C: float = 1.0
    clip_candidates: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 1.0), (0.01, 0.99), (0.02, 0.98), (0.03, 0.97)]
    )
    xgb_params: dict = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "alpha": 1.5,
            "lambda": 2.0,
            "seed": 42,
            "verbosity": 0,
        }
    )
    xgb_num_boost_round: int = 220
    cat_params: dict = field(
        default_factory=lambda: {
            "loss_function": "Logloss",
            "depth": 5,
            "learning_rate": 0.04,
            "l2_leaf_reg": 5.0,
            "iterations": 300,
            "random_seed": 42,
            "verbose": False,
        }
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fit_xgb(train_df: pd.DataFrame, feature_cols: list[str], cfg: StackConfig):
    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df["Target"])
    return xgb.train(cfg.xgb_params, dtrain, num_boost_round=cfg.xgb_num_boost_round)


def _predict_xgb(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if model is None or len(df) == 0:
        return np.full(len(df), np.nan)
    dtest = xgb.DMatrix(df[feature_cols])
    return model.predict(dtest)


def _fit_catboost(train_df: pd.DataFrame, feature_cols: list[str], cfg: StackConfig):
    if not cfg.use_catboost or CatBoostClassifier is None:
        return None
    model = CatBoostClassifier(**cfg.cat_params)
    model.fit(train_df[feature_cols], train_df["Target"])
    return model


def _predict_catboost(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if model is None or len(df) == 0:
        return np.full(len(df), np.nan)
    return model.predict_proba(df[feature_cols])[:, 1]


def _fit_sigmoid_calibrator(y_true: np.ndarray, probs: np.ndarray):
    valid = ~np.isnan(probs)
    if valid.sum() < 50:
        return None
    X = probs[valid].reshape(-1, 1)
    y = y_true[valid]
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=42)
    cal.fit(X, y)
    return cal


def _apply_sigmoid_calibrator(calibrator, probs: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return probs
    out = probs.copy()
    valid = ~np.isnan(out)
    if valid.any():
        out[valid] = calibrator.predict_proba(out[valid].reshape(-1, 1))[:, 1]
    return out


def select_clip_bounds(y_true: np.ndarray, preds: np.ndarray, candidates: list[tuple[float, float]]):
    valid = ~np.isnan(preds)
    y = y_true[valid]
    p = preds[valid]
    best = (0.0, 1.0)
    best_bs = brier_score(y, np.clip(p, 0.0, 1.0))

    for lo, hi in candidates:
        clipped = np.clip(p, lo, hi)
        bs = brier_score(y, clipped)
        if bs < best_bs:
            best_bs = bs
            best = (lo, hi)

    return best[0], best[1], float(best_bs)


def _resolve_feature_set(feature_sets: dict, key: str) -> list[str]:
    vals = feature_sets.get(key, [])
    return [c for c in vals] if vals is not None else []


def generate_oof_base_preds(
    matchups: pd.DataFrame,
    feature_sets: dict,
    lr_cfg: LogRegV2Config,
    cfg: StackConfig,
) -> pd.DataFrame:
    """
    Rolling OOF predictions for base models, with no future leakage.
    """
    df_lr, lr_feature_cols = prepare_logreg_frame(matchups, lr_cfg)
    seasons = sorted(df_lr["Season"].unique())
    if cfg.oof_start_season is not None:
        seasons = [s for s in seasons if s >= cfg.oof_start_season]
    if cfg.oof_end_season is not None:
        seasons = [s for s in seasons if s <= cfg.oof_end_season]

    xgb_features = [c for c in _resolve_feature_set(feature_sets, "xgb_features") if c in matchups.columns]
    cb_features = [c for c in _resolve_feature_set(feature_sets, "cb_features") if c in matchups.columns]
    if not cb_features:
        cb_features = xgb_features

    folds = []
    for season in seasons:
        train_lr = df_lr[df_lr["Season"] < season].copy()
        test_lr = df_lr[df_lr["Season"] == season].copy()
        if len(train_lr) == 0 or len(test_lr) == 0:
            continue

        train = matchups[matchups["Season"] < season].copy()
        test = matchups[matchups["Season"] == season].copy()
        if len(train) == 0 or len(test) == 0:
            continue

        # Elo baseline from Elo_diff.
        if "Elo_diff" in test.columns:
            elo_pred = prob_from_elo_diff(test["Elo_diff"].values)
        else:
            elo_pred = np.full(len(test), 0.5)

        valid_lr_train = train_lr[lr_feature_cols].notna().all(axis=1)
        train_lr_valid = train_lr.loc[valid_lr_train]
        if len(train_lr_valid) > 0:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_lr_valid[lr_feature_cols].values)
            y_train = train_lr_valid["Target"].values
            lr_model = LogisticRegression(
                C=lr_cfg.C, max_iter=lr_cfg.max_iter, solver="lbfgs", random_state=lr_cfg.random_state
            )
            lr_model.fit(X_train, y_train)
            lr_pred = predict_logreg_v2(lr_model, scaler, test_lr, lr_feature_cols)
        else:
            lr_pred = np.full(len(test_lr), np.nan)

        xgb_model = _fit_xgb(train, xgb_features, cfg) if xgb_features else None
        xgb_pred = _predict_xgb(xgb_model, test, xgb_features) if xgb_features else np.full(len(test), np.nan)

        cb_model = _fit_catboost(train, cb_features, cfg) if cb_features else None
        cb_pred = (
            _predict_catboost(cb_model, test, cb_features) if cb_features else np.full(len(test), np.nan)
        )

        fold = test[["Season", "TeamA", "TeamB", "Target"]].copy()
        fold["TrainMaxSeason"] = int(train["Season"].max())
        fold["elo_pred"] = elo_pred
        fold["lr_pred"] = lr_pred
        fold["xgb_pred"] = xgb_pred
        fold["cb_pred"] = cb_pred
        folds.append(fold)

    if not folds:
        raise ValueError("No OOF folds generated.")

    return pd.concat(folds, ignore_index=True)


def fit_meta_model(oof_df: pd.DataFrame, cfg: StackConfig) -> dict:
    pred_cols = ["elo_pred", "lr_pred", "xgb_pred", "cb_pred"]
    active_cols = [c for c in pred_cols if c in oof_df.columns and (~oof_df[c].isna()).any()]
    if not active_cols:
        raise ValueError("No active base prediction columns for meta model.")

    X = oof_df[active_cols].copy()
    fill_values = X.mean(axis=0).to_dict()
    X = X.fillna(fill_values)
    y = oof_df["Target"].values

    meta = LogisticRegression(C=cfg.meta_C, max_iter=3000, solver="lbfgs", random_state=cfg.random_state)
    meta.fit(X.values, y)
    raw = meta.predict_proba(X.values)[:, 1]
    raw_bs = brier_score(y, raw)

    calibrator = None
    calibrated = raw
    calibrated_bs = raw_bs
    if cfg.use_sigmoid_calibration:
        cal = _fit_sigmoid_calibrator(y, raw)
        if cal is not None:
            maybe = _apply_sigmoid_calibrator(cal, raw)
            maybe_bs = brier_score(y, maybe)
            if raw_bs - maybe_bs >= cfg.calibration_min_gain:
                calibrator = cal
                calibrated = maybe
                calibrated_bs = maybe_bs

    clip_low, clip_high, clipped_bs = select_clip_bounds(y, calibrated, cfg.clip_candidates)

    return {
        "meta_model": meta,
        "calibrator": calibrator,
        "base_cols": active_cols,
        "fill_values": fill_values,
        "oof_brier_raw": float(raw_bs),
        "oof_brier_calibrated": float(calibrated_bs),
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "oof_brier_clipped": float(clipped_bs),
    }


def train_stack_final(
    matchups: pd.DataFrame,
    feature_sets: dict,
    lr_cfg: LogRegV2Config,
    cfg: StackConfig,
    max_train_season: int | None = None,
) -> dict:
    train_df = matchups.copy()
    if max_train_season is not None:
        train_df = train_df[train_df["Season"] <= max_train_season].copy()
    if len(train_df) == 0:
        raise ValueError("No rows available for stack training.")

    oof_df = generate_oof_base_preds(train_df, feature_sets, lr_cfg, cfg)
    meta_info = fit_meta_model(oof_df, cfg)

    train_lr, lr_feature_cols = prepare_logreg_frame(train_df, lr_cfg)
    valid_lr_train = train_lr[lr_feature_cols].notna().all(axis=1)
    train_lr_valid = train_lr.loc[valid_lr_train]
    from sklearn.preprocessing import StandardScaler

    lr_scaler = StandardScaler()
    X_lr = lr_scaler.fit_transform(train_lr_valid[lr_feature_cols].values)
    y_lr = train_lr_valid["Target"].values
    lr_model = LogisticRegression(
        C=lr_cfg.C, max_iter=lr_cfg.max_iter, solver="lbfgs", random_state=lr_cfg.random_state
    )
    lr_model.fit(X_lr, y_lr)

    xgb_features = [c for c in _resolve_feature_set(feature_sets, "xgb_features") if c in train_df.columns]
    xgb_model = _fit_xgb(train_df, xgb_features, cfg) if xgb_features else None

    cb_features = [c for c in _resolve_feature_set(feature_sets, "cb_features") if c in train_df.columns]
    if not cb_features:
        cb_features = xgb_features
    cb_model = _fit_catboost(train_df, cb_features, cfg) if cb_features else None

    return {
        "stack_version": "stack_v1",
        "stack_config": asdict(cfg),
        "lr_config": asdict(lr_cfg),
        "feature_sets": feature_sets,
        "lr_model": lr_model,
        "lr_scaler": lr_scaler,
        "lr_feature_cols": lr_feature_cols,
        "xgb_model": xgb_model,
        "xgb_feature_cols": xgb_features,
        "cb_model": cb_model,
        "cb_feature_cols": cb_features,
        "meta_model": meta_info["meta_model"],
        "calibrator": meta_info["calibrator"],
        "meta_base_cols": meta_info["base_cols"],
        "meta_fill_values": meta_info["fill_values"],
        "clip_low": meta_info["clip_low"],
        "clip_high": meta_info["clip_high"],
        "oof_summary": {
            "raw": meta_info["oof_brier_raw"],
            "calibrated": meta_info["oof_brier_calibrated"],
            "clipped": meta_info["oof_brier_clipped"],
        },
    }


def predict_stack_from_matchups(matchups: pd.DataFrame, artifacts: dict, clip: bool = True) -> np.ndarray:
    n = len(matchups)
    if n == 0:
        return np.array([])

    if "Elo_diff" in matchups.columns:
        elo_pred = prob_from_elo_diff(matchups["Elo_diff"].values)
    else:
        elo_pred = np.full(n, 0.5)

    lr_pred = predict_logreg_v2(
        artifacts["lr_model"], artifacts["lr_scaler"], matchups, artifacts["lr_feature_cols"]
    )
    xgb_cols = artifacts.get("xgb_feature_cols", [])
    xgb_pred = (
        _predict_xgb(artifacts.get("xgb_model"), matchups, xgb_cols) if xgb_cols else np.full(n, np.nan)
    )
    cb_cols = artifacts.get("cb_feature_cols", [])
    cb_pred = (
        _predict_catboost(artifacts.get("cb_model"), matchups, cb_cols) if cb_cols else np.full(n, np.nan)
    )

    base = pd.DataFrame(
        {"elo_pred": elo_pred, "lr_pred": lr_pred, "xgb_pred": xgb_pred, "cb_pred": cb_pred}
    )
    cols = artifacts["meta_base_cols"]
    X = base[cols].copy()
    fill_values = artifacts.get("meta_fill_values", {})
    X = X.fillna(fill_values)

    probs = artifacts["meta_model"].predict_proba(X.values)[:, 1]
    probs = _apply_sigmoid_calibrator(artifacts.get("calibrator"), probs)
    if clip:
        probs = np.clip(probs, artifacts["clip_low"], artifacts["clip_high"])
    return probs


def evaluate_stack_holdout(
    matchups: pd.DataFrame, artifacts: dict, train_cutoff: int = 2022
) -> dict:
    test = matchups[matchups["Season"] >= train_cutoff].copy()
    if len(test) == 0:
        raise ValueError("No holdout rows found for evaluate_stack_holdout.")
    preds = predict_stack_from_matchups(test, artifacts, clip=True)
    bs = brier_score(test["Target"].values, preds)
    return {"holdout_brier": float(bs), "n_holdout": int(len(test))}


def save_stack_artifacts(artifacts: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    bundle_path = os.path.join(out_dir, "artifacts.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(artifacts, f)

    meta = {
        "stack_version": artifacts.get("stack_version", "stack_v1"),
        "clip_low": artifacts.get("clip_low"),
        "clip_high": artifacts.get("clip_high"),
        "meta_base_cols": artifacts.get("meta_base_cols", []),
        "oof_summary": artifacts.get("oof_summary", {}),
        "stack_config": artifacts.get("stack_config", {}),
        "lr_config": artifacts.get("lr_config", {}),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_stack_artifacts(in_dir: str) -> dict:
    bundle_path = os.path.join(in_dir, "artifacts.pkl")
    with open(bundle_path, "rb") as f:
        return pickle.load(f)
