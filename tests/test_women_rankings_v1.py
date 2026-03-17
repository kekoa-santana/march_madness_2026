import numpy as np
import pandas as pd

from src.women_rankings_v1 import WomenRankConfig, compute_women_power_ratings


def _toy_women_games():
    return pd.DataFrame(
        [
            {"Season": 2021, "DayNum": 10, "WTeamID": 3001, "WScore": 70, "LTeamID": 3002, "LScore": 60, "WLoc": "N"},
            {"Season": 2021, "DayNum": 30, "WTeamID": 3001, "WScore": 68, "LTeamID": 3003, "LScore": 58, "WLoc": "H"},
            {"Season": 2021, "DayNum": 45, "WTeamID": 3003, "WScore": 66, "LTeamID": 3002, "LScore": 62, "WLoc": "N"},
            {"Season": 2021, "DayNum": 140, "WTeamID": 3002, "WScore": 90, "LTeamID": 3001, "LScore": 80, "WLoc": "N"},
            {"Season": 2022, "DayNum": 20, "WTeamID": 3002, "WScore": 71, "LTeamID": 3003, "LScore": 65, "WLoc": "A"},
        ]
    )


def test_compute_women_power_ratings_schema_and_filtering():
    cfg = WomenRankConfig(n_epochs=2)
    ratings = compute_women_power_ratings(_toy_women_games(), cfg)
    assert set(ratings.columns) == {"Season", "TeamID", "WPR_Rating", "WPR_Ordinal", "WPR_SOS"}
    assert (ratings["Season"] == 2021).any()
    assert (ratings["Season"] == 2022).any()
    assert np.issubdtype(ratings["WPR_Ordinal"].dtype, np.integer)


def test_higher_win_team_gets_higher_rating_in_toy_data():
    cfg = WomenRankConfig(n_epochs=3)
    ratings = compute_women_power_ratings(_toy_women_games(), cfg)
    s2021 = ratings[ratings["Season"] == 2021].set_index("TeamID")
    # Team 3001 has stronger record in filtered regular season than 3002.
    assert s2021.loc[3001, "WPR_Rating"] > s2021.loc[3002, "WPR_Rating"]
