import numpy as np
import pandas as pd

from src.model_logreg_v2 import LogRegV2Config
from src.model_stack_v1 import (
    StackConfig,
    evaluate_stack_holdout,
    generate_oof_base_preds,
    predict_stack_from_matchups,
    train_stack_final,
)


def _synthetic_matchups():
    rows = []
    rng = np.random.default_rng(42)
    seasons = [2018, 2019, 2020, 2021, 2022, 2023]
    for season in seasons:
        for i in range(60):
            elo_diff = rng.normal(0, 120)
            seed_diff = rng.integers(-8, 9)
            pom_diff = rng.normal(0, 20)
            off_eff_diff = rng.normal(0, 8)
            win_pct_diff = rng.normal(0, 0.2)
            net_eff_diff = rng.normal(0, 10)
            ppg_diff = rng.normal(0, 6)
            ppg_allowed_diff = rng.normal(0, 6)
            wpr_rating_diff = rng.normal(0, 12)
            wpr_sos_diff = rng.normal(0, 8)
            logit = (
                0.006 * elo_diff
                - 0.04 * seed_diff
                - 0.02 * pom_diff
                + 0.04 * off_eff_diff
                + 1.2 * win_pct_diff
            )
            prob = 1.0 / (1.0 + np.exp(-logit))
            target = int(rng.random() < prob)
            rows.append(
                {
                    "Season": season,
                    "TeamA": 1000 + i,
                    "TeamB": 2000 + i,
                    "Target": target,
                    "Elo_diff": elo_diff,
                    "SeedNum_diff": seed_diff,
                    "Rank_POM_diff": pom_diff,
                    "Off_Eff_diff": off_eff_diff,
                    "Win_pct_diff": win_pct_diff,
                    "Net_Eff_diff": net_eff_diff,
                    "PPG_diff": ppg_diff,
                    "PPG_allowed_diff": ppg_allowed_diff,
                    "WPR_Rating_diff": wpr_rating_diff,
                    "WPR_SOS_diff": wpr_sos_diff,
                }
            )
    return pd.DataFrame(rows)


def test_generate_oof_base_preds_has_no_leakage():
    matchups = _synthetic_matchups()
    feature_sets = {
        "xgb_features": [
            "Elo_diff",
            "SeedNum_diff",
            "Rank_POM_diff",
            "Off_Eff_diff",
            "Win_pct_diff",
        ],
        "cb_features": ["Elo_diff", "SeedNum_diff", "Off_Eff_diff", "Win_pct_diff"],
    }
    lr_cfg = LogRegV2Config(
        base_features=["Elo_diff", "SeedNum_diff", "Rank_POM_diff", "Off_Eff_diff", "Win_pct_diff"],
        interaction_pairs=[("Elo_diff", "SeedNum_diff")],
        C=0.5,
    )
    cfg = StackConfig(oof_start_season=2019, oof_end_season=2022, use_catboost=False)
    oof = generate_oof_base_preds(matchups, feature_sets, lr_cfg, cfg)
    assert len(oof) > 0
    assert (oof["TrainMaxSeason"] < oof["Season"]).all()
    assert {"elo_pred", "lr_pred", "xgb_pred"}.issubset(set(oof.columns))


def test_train_and_predict_stack_bounds():
    matchups = _synthetic_matchups()
    feature_sets = {
        "xgb_features": [
            "Elo_diff",
            "SeedNum_diff",
            "Rank_POM_diff",
            "Off_Eff_diff",
            "Win_pct_diff",
        ],
        "cb_features": ["Elo_diff", "SeedNum_diff", "Off_Eff_diff", "Win_pct_diff"],
    }
    lr_cfg = LogRegV2Config(
        base_features=["Elo_diff", "SeedNum_diff", "Rank_POM_diff", "Off_Eff_diff", "Win_pct_diff"],
        interaction_pairs=[("Elo_diff", "SeedNum_diff")],
        C=0.5,
    )
    cfg = StackConfig(oof_start_season=2019, oof_end_season=2021, use_catboost=False, train_cutoff=2022)
    artifacts = train_stack_final(matchups, feature_sets, lr_cfg, cfg, max_train_season=2022)

    holdout = matchups[matchups["Season"] >= 2022].copy()
    preds = predict_stack_from_matchups(holdout, artifacts, clip=True)
    assert preds.shape[0] == len(holdout)
    assert np.all((preds >= 0.0) & (preds <= 1.0))

    metrics = evaluate_stack_holdout(matchups, artifacts, train_cutoff=2022)
    assert "holdout_brier" in metrics
    assert metrics["n_holdout"] == len(holdout)
