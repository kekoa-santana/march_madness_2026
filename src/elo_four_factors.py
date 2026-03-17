"""Four Factors-adjusted Elo rating system.

Standard Elo treats all wins/losses equally given the margin. This system
adjusts Elo updates based on how surprising the result is given each team's
offensive and defensive efficiency profile.

A high-offense team beating a bad-defense team by 15 = expected = small update.
A bad-offense team beating a good-defense team by 15 = impressive = big update.

The resulting rating captures "how well does this team perform relative to
what we'd expect from the stylistic matchup" — coaching, clutch play, intangibles.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FFEloConfig:
    k: float = 16.0              # Base K factor
    home_adv: float = 40.0       # Home court advantage in Elo points
    carryover: float = 0.90      # Season-to-season rating retention
    ff_weight: float = 0.5       # How much to weight FF-predicted margin (0=ignore FF, 1=fully trust FF)
    recency_halflife: float = 40.0  # Halflife in DayNums for rolling Four Factors
    min_games: int = 5           # Min games before using FF adjustment
    initial_elo: float = 1500.0


def compute_four_factors(fgm, fga, fgm3, fga3, ftm, fta, orb, drb, to, opp_drb, opp_orb):
    """Compute Dean Oliver's Four Factors from box score stats.

    Returns: (efg_pct, to_rate, or_pct, ft_rate)
    """
    # Effective FG% = (FGM + 0.5 * FGM3) / FGA
    efg = (fgm + 0.5 * fgm3) / max(fga, 1)

    # Turnover rate = TO / (FGA + 0.44 * FTA + TO)
    possessions_approx = fga + 0.44 * fta + to
    to_rate = to / max(possessions_approx, 1)

    # Offensive rebound % = ORB / (ORB + Opp_DRB)
    or_pct = orb / max(orb + opp_drb, 1)

    # Free throw rate = FTM / FGA
    ft_rate = ftm / max(fga, 1)

    return efg, to_rate, or_pct, ft_rate


def estimate_possessions(fga, fta, orb, to, opp_drb):
    """Estimate possessions from box score."""
    return fga - orb + to + 0.44 * fta


def predict_margin_from_ff(off_ff, def_ff, opp_off_ff, opp_def_ff, avg_poss=68.0):
    """Predict expected scoring margin based on Four Factors matchup.

    off_ff = (efg, to_rate, or_pct, ft_rate) for team's offense
    def_ff = (efg, to_rate, or_pct, ft_rate) that team allows on defense
    opp_off_ff, opp_def_ff = same for opponent

    Returns estimated point margin (positive = team A favored).
    """
    # Team A's offense faces Team B's defense
    # Blend team A's offensive tendencies with team B's defensive tendencies
    a_efg = (off_ff[0] + opp_def_ff[0]) / 2
    a_to = (off_ff[1] + opp_def_ff[1]) / 2
    a_or = (off_ff[2] + opp_def_ff[2]) / 2
    a_ft = (off_ff[3] + opp_def_ff[3]) / 2

    # Team B's offense faces Team A's defense
    b_efg = (opp_off_ff[0] + def_ff[0]) / 2
    b_to = (opp_off_ff[1] + def_ff[1]) / 2
    b_or = (opp_off_ff[2] + def_ff[2]) / 2
    b_ft = (opp_off_ff[3] + def_ff[3]) / 2

    # Points per possession ~ 2 * eFG% * (1 - TO%) * (1 + OR%_bonus) + FT_bonus
    # Simplified: higher eFG, lower TO, higher OR, higher FT = more points
    def pts_per_poss(efg, to, orr, ft):
        return 2 * efg * (1 - to) * (1 + 0.3 * orr) + 0.4 * ft

    a_ppp = pts_per_poss(a_efg, a_to, a_or, a_ft)
    b_ppp = pts_per_poss(b_efg, b_to, b_or, b_ft)

    return (a_ppp - b_ppp) * avg_poss


def compute_ff_elo(games_df, config=None):
    """Compute Four Factors-adjusted Elo ratings.

    Args:
        games_df: Detailed season results with box score columns.
                  Must have: Season, DayNum, WTeamID, LTeamID, WScore, LScore,
                  WLoc, WFGM, WFGA, WFGM3, WFGA3, WFTM, WFTA, WOR, WDR, WTO,
                  LFGM, LFGA, LFGM3, LFGA3, LFTM, LFTA, LOR, LDR, LTO
        config: FFEloConfig

    Returns:
        DataFrame with Season, TeamID, FF_Elo, FF_eFG_off, FF_eFG_def,
        FF_TO_off, FF_TO_def, FF_OR_off, FF_OR_def, FF_FT_off, FF_FT_def
    """
    if config is None:
        config = FFEloConfig()

    # Sort chronologically
    df = games_df.sort_values(['Season', 'DayNum']).reset_index(drop=True)

    elo = {}           # TeamID -> current Elo
    # Rolling Four Factors: exponentially weighted
    # Each team tracks: sum_weight, and weighted sums of each factor
    ff_off = {}        # TeamID -> [sum_w, sum_efg, sum_to, sum_or, sum_ft]
    ff_def = {}        # TeamID -> [sum_w, sum_efg, sum_to, sum_or, sum_ft]
    game_count = {}    # TeamID -> games played this season
    prev_season = None

    end_of_season = {}  # (Season, TeamID) -> final ratings

    for _, game in df.iterrows():
        season = game['Season']
        day = game['DayNum']

        # Season transition
        if season != prev_season:
            # Save end-of-season ratings
            if prev_season is not None:
                for tid, rating in elo.items():
                    end_of_season[(prev_season, tid)] = {
                        'FF_Elo': rating,
                    }
                    if tid in ff_off and game_count.get(tid, 0) >= config.min_games:
                        sw = ff_off[tid][0]
                        if sw > 0:
                            end_of_season[(prev_season, tid)].update({
                                'FF_eFG_off': ff_off[tid][1] / sw,
                                'FF_TO_off': ff_off[tid][2] / sw,
                                'FF_OR_off': ff_off[tid][3] / sw,
                                'FF_FT_off': ff_off[tid][4] / sw,
                            })
                    if tid in ff_def and game_count.get(tid, 0) >= config.min_games:
                        sw = ff_def[tid][0]
                        if sw > 0:
                            end_of_season[(prev_season, tid)].update({
                                'FF_eFG_def': ff_def[tid][1] / sw,
                                'FF_TO_def': ff_def[tid][2] / sw,
                                'FF_OR_def': ff_def[tid][3] / sw,
                                'FF_FT_def': ff_def[tid][4] / sw,
                            })

            # Regress Elos toward mean
            for tid in elo:
                elo[tid] = config.initial_elo + config.carryover * (elo[tid] - config.initial_elo)

            # Reset rolling FF stats
            ff_off.clear()
            ff_def.clear()
            game_count.clear()
            prev_season = season

        wid = int(game['WTeamID'])
        lid = int(game['LTeamID'])

        # Initialize new teams
        for tid in [wid, lid]:
            if tid not in elo:
                elo[tid] = config.initial_elo
            if tid not in game_count:
                game_count[tid] = 0

        # --- Compute this game's Four Factors ---
        w_off_ff = compute_four_factors(
            game['WFGM'], game['WFGA'], game['WFGM3'], game['WFGA3'],
            game['WFTM'], game['WFTA'], game['WOR'], game['WDR'],
            game['WTO'], game['LDR'], game['LOR']
        )
        l_off_ff = compute_four_factors(
            game['LFGM'], game['LFGA'], game['LFGM3'], game['LFGA3'],
            game['LFTM'], game['LFTA'], game['LOR'], game['LDR'],
            game['LTO'], game['WDR'], game['WOR']
        )
        # Defensive FF = what you allowed (opponent's offense)
        w_def_ff = l_off_ff  # winner's defense allowed loser's offense
        l_def_ff = w_off_ff  # loser's defense allowed winner's offense

        # --- Elo update ---
        # Home advantage
        w_adv = config.home_adv if game['WLoc'] == 'H' else (
            -config.home_adv if game['WLoc'] == 'A' else 0)

        # Standard Elo expected score
        elo_diff = elo[wid] + w_adv - elo[lid]
        expected_w = 1.0 / (1.0 + 10 ** (-elo_diff / 400))

        # MOV factor (same as Elo v2)
        actual_margin = game['WScore'] - game['LScore']
        mov_mult = math.log(abs(actual_margin) + 1)

        # Four Factors adjustment: how surprising was the margin given the matchup?
        ff_surprise_mult = 1.0
        if (game_count.get(wid, 0) >= config.min_games and
            game_count.get(lid, 0) >= config.min_games and
            wid in ff_off and lid in ff_off):

            w_sw = ff_off[wid][0]
            l_sw = ff_off[lid][0]
            if w_sw > 0 and l_sw > 0:
                # Get rolling averages
                w_off_avg = tuple(ff_off[wid][i+1] / w_sw for i in range(4))
                w_def_avg = tuple(ff_def[wid][i+1] / ff_def[wid][0] for i in range(4))
                l_off_avg = tuple(ff_off[lid][i+1] / l_sw for i in range(4))
                l_def_avg = tuple(ff_def[lid][i+1] / ff_def[lid][0] for i in range(4))

                # Predicted margin from FF matchup
                ff_predicted_margin = predict_margin_from_ff(
                    w_off_avg, w_def_avg, l_off_avg, l_def_avg
                )

                # How much did actual exceed FF prediction?
                # If actual margin >> FF predicted margin, team overperformed → bigger update
                ff_residual = actual_margin - ff_predicted_margin
                # Scale: |residual| / typical_stdev (~10 points)
                ff_surprise_mult = 1.0 + config.ff_weight * (abs(ff_residual) / 10.0 - 1.0)
                ff_surprise_mult = max(0.3, min(ff_surprise_mult, 3.0))  # clamp

        # Combined update
        update = config.k * mov_mult * ff_surprise_mult * (1.0 - expected_w)
        elo[wid] += update
        elo[lid] -= update

        # --- Update rolling Four Factors ---
        decay = 0.5 ** (1.0 / max(config.recency_halflife, 1.0))

        for tid, off_ff, def_ff_game in [
            (wid, w_off_ff, w_def_ff),
            (lid, l_off_ff, l_def_ff),
        ]:
            if tid not in ff_off:
                ff_off[tid] = [0.0, 0.0, 0.0, 0.0, 0.0]
                ff_def[tid] = [0.0, 0.0, 0.0, 0.0, 0.0]

            # Decay existing weights
            ff_off[tid][0] *= decay
            ff_def[tid][0] *= decay
            for i in range(4):
                ff_off[tid][i+1] *= decay
                ff_def[tid][i+1] *= decay

            # Add new observation
            ff_off[tid][0] += 1.0
            ff_def[tid][0] += 1.0
            for i, val in enumerate(off_ff):
                ff_off[tid][i+1] += val
            for i, val in enumerate(def_ff_game):
                ff_def[tid][i+1] += val

            game_count[tid] = game_count.get(tid, 0) + 1

    # Save final season
    if prev_season is not None:
        for tid, rating in elo.items():
            end_of_season[(prev_season, tid)] = {'FF_Elo': rating}
            if tid in ff_off and game_count.get(tid, 0) >= config.min_games:
                sw = ff_off[tid][0]
                if sw > 0:
                    end_of_season[(prev_season, tid)].update({
                        'FF_eFG_off': ff_off[tid][1] / sw,
                        'FF_TO_off': ff_off[tid][2] / sw,
                        'FF_OR_off': ff_off[tid][3] / sw,
                        'FF_FT_off': ff_off[tid][4] / sw,
                    })
                sw = ff_def[tid][0]
                if sw > 0:
                    end_of_season[(prev_season, tid)].update({
                        'FF_eFG_def': ff_def[tid][1] / sw,
                        'FF_TO_def': ff_def[tid][2] / sw,
                        'FF_OR_def': ff_def[tid][3] / sw,
                        'FF_FT_def': ff_def[tid][4] / sw,
                    })

    # Build output DataFrame
    rows = []
    for (season, tid), vals in end_of_season.items():
        row = {'Season': season, 'TeamID': tid}
        row.update(vals)
        rows.append(row)

    result = pd.DataFrame(rows)
    return result
