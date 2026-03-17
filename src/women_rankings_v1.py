import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import build_matchup_matrix
from src.model import brier_score


@dataclass(frozen=True)
class WomenRankConfig:
    learning_rate: float = 14.0
    home_adv: float = 45.0
    recency_tau: float = 35.0
    reg_strength: float = 0.0015
    epoch_shrink: float = 0.995
    n_epochs: int = 4
    scale: float = 110.0


def _expected_prob(rating_a: float, rating_b: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp(-(rating_a - rating_b) / scale))


def compute_women_power_ratings(games_df: pd.DataFrame, config: WomenRankConfig) -> pd.DataFrame:
    """
    Build in-house women rankings using regular-season games only (DayNum <= 132).
    """
    reg = games_df[games_df["DayNum"] <= 132].sort_values(by=["Season", "DayNum"]).copy()
    rows = []

    for season, season_games in reg.groupby("Season"):
        season_games = season_games.sort_values("DayNum")
        teams = sorted(set(season_games["WTeamID"]).union(set(season_games["LTeamID"])))
        team_to_ix = {team: i for i, team in enumerate(teams)}
        ratings = np.zeros(len(teams), dtype=float)
        max_day = float(season_games["DayNum"].max())

        for _ in range(config.n_epochs):
            for game in season_games.itertuples(index=False):
                w_ix = team_to_ix[game.WTeamID]
                l_ix = team_to_ix[game.LTeamID]
                day_weight = math.exp(-(max_day - game.DayNum) / max(config.recency_tau, 1.0))
                w_adv = config.home_adv if game.WLoc == "H" else 0.0
                l_adv = config.home_adv if game.WLoc == "A" else 0.0

                expected_w = _expected_prob(ratings[w_ix] + w_adv, ratings[l_ix] + l_adv, config.scale)
                mov = math.log(abs(game.WScore - game.LScore) + 1.0)
                step = config.learning_rate * day_weight * mov * (1.0 - expected_w)

                ratings[w_ix] = ratings[w_ix] * (1.0 - config.reg_strength) + step
                ratings[l_ix] = ratings[l_ix] * (1.0 - config.reg_strength) - step

            ratings *= config.epoch_shrink

        # Strength of schedule proxy: average opponent rating faced.
        sos_sum = np.zeros(len(teams), dtype=float)
        sos_cnt = np.zeros(len(teams), dtype=float)
        for game in season_games.itertuples(index=False):
            w_ix = team_to_ix[game.WTeamID]
            l_ix = team_to_ix[game.LTeamID]
            sos_sum[w_ix] += ratings[l_ix]
            sos_cnt[w_ix] += 1.0
            sos_sum[l_ix] += ratings[w_ix]
            sos_cnt[l_ix] += 1.0

        sos = np.divide(sos_sum, np.maximum(sos_cnt, 1.0))
        ordinal = pd.Series(-ratings).rank(method="dense").astype(int).values

        for team_id, ix in team_to_ix.items():
            rows.append((season, team_id, ratings[ix], int(ordinal[ix]), sos[ix]))

    return pd.DataFrame(
        rows, columns=["Season", "TeamID", "WPR_Rating", "WPR_Ordinal", "WPR_SOS"]
    )


def merge_women_rank_features(team_features: pd.DataFrame, wpr_df: pd.DataFrame) -> pd.DataFrame:
    return team_features.merge(wpr_df, on=["Season", "TeamID"], how="left")


def _fit_eval_logreg(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> float:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    y_train = train_df["Target"].values
    y_test = test_df["Target"].values
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return brier_score(y_test, preds)


def tune_women_rank_config(
    regular_season_games: pd.DataFrame,
    women_tourney_games: pd.DataFrame,
    women_team_features_base: pd.DataFrame,
    config_grid: list[WomenRankConfig],
    train_cutoff: int = 2022,
) -> tuple[WomenRankConfig, pd.DataFrame]:
    """
    Tune WomenRankConfig using holdout Brier from simple LR on matchup features.
    """
    rows = []
    best_cfg = None
    best_bs = float("inf")

    for cfg in config_grid:
        wpr_df = compute_women_power_ratings(regular_season_games, cfg)
        team_features = merge_women_rank_features(women_team_features_base, wpr_df)
        matchups = build_matchup_matrix(women_tourney_games, team_features)
        train = matchups[matchups["Season"] < train_cutoff]
        test = matchups[matchups["Season"] >= train_cutoff]
        if len(train) == 0 or len(test) == 0:
            continue

        candidate = [
            "Elo_diff",
            "SeedNum_diff",
            "Net_Eff_diff",
            "WPR_Rating_diff",
            "WPR_Ordinal_diff",
            "WPR_SOS_diff",
        ]
        feature_cols = [c for c in candidate if c in train.columns]
        valid_train = train[feature_cols].notna().all(axis=1)
        valid_test = test[feature_cols].notna().all(axis=1)
        if valid_train.sum() < 50 or valid_test.sum() < 20:
            continue

        bs = _fit_eval_logreg(train.loc[valid_train], test.loc[valid_test], feature_cols)
        row = asdict(cfg)
        row["holdout_brier"] = float(bs)
        rows.append(row)

        if bs < best_bs:
            best_bs = bs
            best_cfg = cfg

    if best_cfg is None:
        raise ValueError("No WomenRankConfig evaluated. Check inputs/config_grid.")

    results = pd.DataFrame(rows).sort_values("holdout_brier", ascending=True).reset_index(drop=True)
    return best_cfg, results
