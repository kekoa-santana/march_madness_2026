"""LOSO (Leave-One-Season-Out) evaluation for the full stacking pipeline.

For each test season S:
  - Base models use rolling OOF (train on seasons < S, predict S) — already computed
  - Meta-model is fit on OOF predictions from all seasons != S
  - Calibrator and clip bounds are selected on all seasons != S
  - Final prediction is made for season S

This gives ~16 per-season Brier scores instead of 4, making it much harder
to overfit to noise in a small holdout.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.model import brier_score
from src.model_stack_v1 import (
    StackConfig,
    _fit_sigmoid_calibrator,
    _apply_sigmoid_calibrator,
    select_clip_bounds,
    generate_oof_base_preds,
)
from src.model_logreg_v2 import LogRegV2Config


PRED_COLS = ["elo_pred", "lr_pred", "xgb_pred", "cb_pred"]


def loso_evaluate(
    matchups: pd.DataFrame,
    feature_sets: dict,
    lr_cfg: LogRegV2Config,
    cfg: StackConfig,
    eval_seasons: list[int] | None = None,
    alpha: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Run full LOSO evaluation on the stacking pipeline.

    Parameters
    ----------
    matchups : DataFrame with features and Target column (tournament matchups)
    feature_sets : dict with 'xgb_features' and 'cb_features' keys
    lr_cfg : LogRegV2Config for the base LR model
    cfg : StackConfig for the stacking pipeline
    eval_seasons : which seasons to evaluate on (default: all available in OOF)
    alpha : logit stretching factor (1.0 = no stretching)
    verbose : print per-season results

    Returns
    -------
    dict with 'mean_brier', 'per_season', 'n_total', 'std_brier'
    """
    # Step 1: Generate all OOF base predictions (rolling, no leakage)
    oof_df = generate_oof_base_preds(matchups, feature_sets, lr_cfg, cfg)

    all_seasons = sorted(oof_df["Season"].unique())
    if eval_seasons is not None:
        all_seasons = [s for s in all_seasons if s in eval_seasons]

    active_cols = [c for c in PRED_COLS if c in oof_df.columns and (~oof_df[c].isna()).any()]

    per_season = []
    all_preds = []
    all_targets = []

    for season in all_seasons:
        train_oof = oof_df[oof_df["Season"] != season].copy()
        test_oof = oof_df[oof_df["Season"] == season].copy()

        if len(train_oof) < 10 or len(test_oof) == 0:
            continue

        # Fit meta-model on all seasons except this one
        X_train = train_oof[active_cols].copy()
        fill_values = X_train.mean(axis=0).to_dict()
        X_train = X_train.fillna(fill_values)
        y_train = train_oof["Target"].values

        meta = LogisticRegression(
            C=cfg.meta_C, max_iter=3000, solver="lbfgs", random_state=cfg.random_state
        )
        meta.fit(X_train.values, y_train)

        # Fit calibrator on train OOF
        train_raw = meta.predict_proba(X_train.values)[:, 1]
        calibrator = None
        if cfg.use_sigmoid_calibration:
            cal = _fit_sigmoid_calibrator(y_train, train_raw)
            if cal is not None:
                train_cal = _apply_sigmoid_calibrator(cal, train_raw)
                raw_bs = brier_score(y_train, train_raw)
                cal_bs = brier_score(y_train, train_cal)
                if raw_bs - cal_bs >= cfg.calibration_min_gain:
                    calibrator = cal

        # Select clip bounds on train OOF
        train_preds = _apply_sigmoid_calibrator(calibrator, train_raw)
        clip_low, clip_high, _ = select_clip_bounds(y_train, train_preds, cfg.clip_candidates)

        # Predict on held-out season
        X_test = test_oof[active_cols].copy().fillna(fill_values)
        test_raw = meta.predict_proba(X_test.values)[:, 1]
        test_preds = _apply_sigmoid_calibrator(calibrator, test_raw)
        test_preds = np.clip(test_preds, clip_low, clip_high)

        # Apply alpha stretching
        if alpha != 1.0:
            test_preds = _stretch_preds(test_preds, alpha)

        y_test = test_oof["Target"].values
        bs = brier_score(y_test, test_preds)

        per_season.append({
            "season": season,
            "brier": float(bs),
            "n_games": len(test_oof),
        })
        all_preds.extend(test_preds.tolist())
        all_targets.extend(y_test.tolist())

        if verbose:
            print(f"  {season}: Brier={bs:.5f}  (n={len(test_oof)})")

    if not per_season:
        raise ValueError("No seasons evaluated.")

    brier_scores = [s["brier"] for s in per_season]
    mean_bs = float(np.mean(brier_scores))
    std_bs = float(np.std(brier_scores))
    pooled_bs = brier_score(np.array(all_targets), np.array(all_preds))

    if verbose:
        print(f"\n  LOSO mean:   {mean_bs:.5f} +/- {std_bs:.5f}")
        print(f"  LOSO pooled: {pooled_bs:.5f}  (n={len(all_preds)})")

    return {
        "mean_brier": mean_bs,
        "std_brier": std_bs,
        "pooled_brier": float(pooled_bs),
        "n_total": len(all_preds),
        "n_seasons": len(per_season),
        "per_season": per_season,
    }


def loso_compare(
    matchups: pd.DataFrame,
    configs: dict[str, tuple[dict, LogRegV2Config, StackConfig]],
    eval_seasons: list[int] | None = None,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Compare multiple configs via LOSO. Returns summary DataFrame.

    Parameters
    ----------
    matchups : tournament matchup DataFrame
    configs : dict mapping name -> (feature_sets, lr_cfg, stack_cfg)
    eval_seasons : seasons to evaluate on
    alpha : logit stretching factor

    Returns
    -------
    DataFrame with columns: name, mean_brier, std_brier, pooled_brier, n_seasons
    """
    rows = []
    for name, (feature_sets, lr_cfg, stack_cfg) in configs.items():
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(f"{'='*50}")
        result = loso_evaluate(
            matchups, feature_sets, lr_cfg, stack_cfg,
            eval_seasons=eval_seasons, alpha=alpha,
        )
        rows.append({
            "name": name,
            "mean_brier": result["mean_brier"],
            "std_brier": result["std_brier"],
            "pooled_brier": result["pooled_brier"],
            "n_seasons": result["n_seasons"],
        })

    df = pd.DataFrame(rows).sort_values("mean_brier")
    print(f"\n{'='*50}")
    print("LOSO Comparison Summary (sorted by mean Brier)")
    print(f"{'='*50}")
    for _, row in df.iterrows():
        print(f"  {row['name']:30s}  mean={row['mean_brier']:.5f} +/- {row['std_brier']:.5f}  pooled={row['pooled_brier']:.5f}")
    return df


def _stretch_preds(preds, alpha):
    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    logit = np.log(preds / (1 - preds))
    stretched = 1.0 / (1.0 + np.exp(-alpha * logit))
    return np.clip(stretched, 0.001, 0.999)
