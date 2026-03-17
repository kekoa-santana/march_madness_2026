import numpy as np
import pandas as pd

from src.model_elo_v2 import EloConfig, compute_elo_ratings_v2, elo_probs_for_matchups


def _toy_games():
    return pd.DataFrame(
        [
            {"Season": 2021, "DayNum": 10, "WTeamID": 1, "WScore": 70, "LTeamID": 2, "LScore": 60, "WLoc": "H"},
            {"Season": 2021, "DayNum": 50, "WTeamID": 2, "WScore": 66, "LTeamID": 1, "LScore": 65, "WLoc": "N"},
            {"Season": 2021, "DayNum": 140, "WTeamID": 1, "WScore": 80, "LTeamID": 2, "LScore": 70, "WLoc": "N"},
            {"Season": 2022, "DayNum": 15, "WTeamID": 1, "WScore": 72, "LTeamID": 2, "LScore": 60, "WLoc": "A"},
        ]
    )


def test_compute_elo_ratings_v2_filters_postseason_days():
    games = _toy_games()
    cfg = EloConfig()
    elo = compute_elo_ratings_v2(games, cfg)
    assert set(elo.columns) == {"Season", "TeamID", "Elo"}
    assert len(elo) > 0
    # DayNum 140 game should not be included in regular-season Elo updates.
    assert (elo["Season"] == 2021).any()
    assert (elo["Season"] == 2022).any()


def test_elo_probs_for_matchups_returns_valid_probs():
    games = _toy_games()
    cfg = EloConfig()
    elo = compute_elo_ratings_v2(games, cfg)
    lookup = elo.set_index(["Season", "TeamID"])["Elo"].to_dict()
    matchups = pd.DataFrame(
        [{"Season": 2021, "TeamA": 1, "TeamB": 2}, {"Season": 2022, "TeamA": 1, "TeamB": 2}]
    )
    probs = elo_probs_for_matchups(matchups, lookup)
    assert probs.shape == (2,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
