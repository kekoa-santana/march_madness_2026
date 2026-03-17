import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.feature_engineering import build_matchup_matrix, build_team_features, parse_seeds
from src.model import brier_score


@dataclass(frozen=True)
class EloConfig:
    k: float = 28.0
    home_adv: float = 90.0
    carryover: float = 0.75
    mov_alpha: float = 2.2
    mov_beta: float = 0.001
    season_decay: float = 0.15
    scale: float = 400.0


def _expected_win_prob(rating_a: float, rating_b: float, scale: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / scale))


def prob_from_elo_diff(elo_diff, scale: float = 400.0):
    elo_diff = np.asarray(elo_diff, dtype=float)
    return 1.0 / (1.0 + 10 ** (-elo_diff / scale))


def compute_elo_ratings_v2(games_df: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """
    Compute regular-season end Elo ratings with tunable parameters.
    """
    reg = games_df[games_df["DayNum"] <= 132].sort_values(by=["Season", "DayNum"]).copy()
    elos = {}
    results = []
    current_season = None

    for row in reg.itertuples(index=False):
        season = row.Season
        w_id = row.WTeamID
        l_id = row.LTeamID

        if season != current_season:
            if current_season is not None:
                for team_id, elo in elos.items():
                    results.append((current_season, team_id, elo))
            for team in elos:
                elos[team] = 1500.0 + config.carryover * (elos[team] - 1500.0)
            current_season = season

        w_elo = elos.get(w_id, 1500.0)
        l_elo = elos.get(l_id, 1500.0)

        w_adj = w_elo
        l_adj = l_elo
        if row.WLoc == "H":
            w_adj += config.home_adv
        elif row.WLoc == "A":
            l_adj += config.home_adv

        expected_w = _expected_win_prob(w_adj, l_adj, config.scale)
        score_diff = row.WScore - row.LScore
        elo_gap = abs(w_elo - l_elo)
        mov_mult = math.log(abs(score_diff) + 1.0) * (
            config.mov_alpha / (config.mov_alpha + config.mov_beta * elo_gap)
        )
        # Higher weight for late-season games.
        day_weight = 1.0 + config.season_decay * (row.DayNum / 132.0)
        update = config.k * day_weight * mov_mult * (1.0 - expected_w)

        elos[w_id] = w_elo + update
        elos[l_id] = l_elo - update

    if current_season is not None:
        for team_id, elo in elos.items():
            results.append((current_season, team_id, elo))

    return pd.DataFrame(results, columns=["Season", "TeamID", "Elo"])


def build_elo_lookup(elo_df: pd.DataFrame) -> dict:
    return elo_df.set_index(["Season", "TeamID"])["Elo"].to_dict()


def elo_probs_for_matchups(
    matchups: pd.DataFrame, elo_lookup: dict, default_elo: float = 1500.0, scale: float = 400.0
) -> np.ndarray:
    elo_a = np.array(
        [elo_lookup.get((int(s), int(t)), default_elo) for s, t in zip(matchups["Season"], matchups["TeamA"])],
        dtype=float,
    )
    elo_b = np.array(
        [elo_lookup.get((int(s), int(t)), default_elo) for s, t in zip(matchups["Season"], matchups["TeamB"])],
        dtype=float,
    )
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / scale))


def tune_elo_config(
    regular_season_games: pd.DataFrame,
    tourney_games: pd.DataFrame,
    seeds_df: pd.DataFrame,
    config_grid: list[EloConfig],
    train_cutoff: int = 2022,
) -> tuple[EloConfig, pd.DataFrame]:
    """
    Tune Elo parameters directly against tournament holdout Brier.
    """
    parsed_seeds = parse_seeds(seeds_df) if "Seed" in seeds_df.columns else seeds_df.copy()
    rows = []
    best_cfg = None
    best_bs = float("inf")

    for cfg in config_grid:
        elo_df = compute_elo_ratings_v2(regular_season_games, cfg)
        team_features = build_team_features(elo_df, seeds_df=parsed_seeds)
        matchups = build_matchup_matrix(tourney_games, team_features)
        test = matchups[matchups["Season"] >= train_cutoff]
        if len(test) == 0:
            continue

        preds = prob_from_elo_diff(test["Elo_diff"].values, scale=cfg.scale)
        bs = brier_score(test["Target"].values, preds)
        row = asdict(cfg)
        row["holdout_brier"] = float(bs)
        rows.append(row)

        if bs < best_bs:
            best_bs = bs
            best_cfg = cfg

    if best_cfg is None:
        raise ValueError("No Elo configs evaluated. Check train_cutoff and data inputs.")

    results = pd.DataFrame(rows).sort_values("holdout_brier", ascending=True).reset_index(drop=True)
    return best_cfg, results
