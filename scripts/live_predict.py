"""Live conference tournament predictor for 2026.

Trains the full stacking pipeline (O config) on all available data,
then predicts any matchup by team name.

Usage:
    python scripts/live_predict.py                    # interactive mode
    python scripts/live_predict.py "Duke" "UNC"       # single prediction
"""
import sys
import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

WOMEN_ALPHA = 1.2

def stretch_preds(preds, alpha):
    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    logit = np.log(preds / (1 - preds))
    stretched = 1.0 / (1.0 + np.exp(-alpha * logit))
    return np.clip(stretched, 0.001, 0.999)

sys.path.insert(0, '.')

from src.data_loader import load_men_data, load_women_data
from src.feature_engineering import (
    parse_seeds, compute_season_stats, compute_massey_features,
    compute_conference_strength, compute_efficiency, build_team_features,
    build_matchup_matrix,
)
from src.model_elo_v2 import EloConfig, compute_elo_ratings_v2, build_elo_lookup
from src.model_logreg_v2 import LogRegV2Config
from src.model_stack_v1 import StackConfig, train_stack_final, predict_stack_from_matchups
from src.submission_stack_v1 import _build_matchup_features, _extract_interaction_pairs
from src.women_rankings_v1 import WomenRankConfig, compute_women_power_ratings, merge_women_rank_features

# ============================================================
# O config (LOSO-validated best)
# ============================================================
MEN_ELO = EloConfig(k=14, home_adv=40, carryover=0.94, season_decay=0.15)
WOMEN_ELO = EloConfig(k=20, home_adv=40, carryover=0.92, season_decay=0.15)

men_lr_cfg = LogRegV2Config(
    base_features=['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
    interaction_pairs=[('Elo_diff', 'SeedNum_diff'), ('Elo_diff', 'Rank_POM_diff'), ('Off_Eff_diff', 'Win_pct_diff')],
    C=0.2,
)
women_lr_cfg = LogRegV2Config(
    base_features=['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff', 'WPR_Rating_diff', 'WPR_SOS_diff'],
    interaction_pairs=[],
    C=0.5,
)

men_features_cfg = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
    'cb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
}
women_features_cfg = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff',
                     'WPR_Rating_diff', 'WPR_SOS_diff', 'Off_Eff_diff', 'Def_Eff_diff', 'Win_pct_diff'],
    'cb_features': ['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff',
                     'WPR_Rating_diff', 'WPR_SOS_diff', 'Off_Eff_diff', 'Def_Eff_diff', 'Win_pct_diff'],
}


def build_name_lookup(data_dir):
    """Build separate men's and women's name → TeamID lookups."""
    men_lookup = {}
    women_lookup = {}

    for fname, target in [('MTeamSpellings.csv', men_lookup), ('WTeamSpellings.csv', women_lookup)]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='latin-1')
            for _, row in df.iterrows():
                name = str(row.iloc[0]).strip().lower()
                tid = int(row.iloc[1])
                target[name] = tid

    for fname, target in [('MTeams.csv', men_lookup), ('WTeams.csv', women_lookup)]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                name = str(row['TeamName']).strip().lower()
                tid = int(row['TeamID'])
                target[name] = tid

    return men_lookup, women_lookup


def build_id_to_name(data_dir):
    """Build TeamID → canonical name lookup."""
    lookup = {}
    for fname in ['MTeams.csv', 'WTeams.csv']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                lookup[int(row['TeamID'])] = str(row['TeamName'])
    return lookup


def resolve_team(name_input, name_lookup, id_to_name):
    """Resolve user input to a TeamID. Accepts ID or name (fuzzy)."""
    # Try as integer ID first
    try:
        tid = int(name_input)
        if tid in id_to_name:
            return tid
    except ValueError:
        pass

    # Exact match
    key = name_input.strip().lower()
    if key in name_lookup:
        return name_lookup[key]

    # Substring match
    matches = [(k, v) for k, v in name_lookup.items() if key in k]
    if len(matches) >= 1:
        unique_ids = list(set(v for _, v in matches))
        if len(unique_ids) == 1:
            return unique_ids[0]
        # Show options
        print(f"  Ambiguous '{name_input}'. Matches:")
        shown = set()
        for _, tid in sorted(matches, key=lambda x: x[1]):
            if tid not in shown:
                shown.add(tid)
                print(f"    {tid}: {id_to_name.get(tid, '?')}")
        return None

    print(f"  No match for '{name_input}'")
    return None


def resolve_team_pair(name1, name2, men_lookup, women_lookup, id_to_name):
    """Resolve two team names, auto-detecting gender.

    Tries men first. If both found as men, returns those.
    If both found as women, returns those.
    Falls back to trying each lookup separately.
    """
    # Try as integer IDs first
    for n in [name1, name2]:
        try:
            tid = int(n)
            if tid in id_to_name:
                # Gender determined by ID
                if tid >= 3000:
                    lookup = women_lookup
                else:
                    lookup = men_lookup
                id1 = resolve_team(name1, lookup, id_to_name)
                id2 = resolve_team(name2, lookup, id_to_name)
                return id1, id2
        except ValueError:
            pass

    # Try men first (more common use case)
    m1 = resolve_team(name1, men_lookup, id_to_name)
    m2 = resolve_team(name2, men_lookup, id_to_name)
    if m1 is not None and m2 is not None:
        return m1, m2

    # Try women
    w1 = resolve_team(name1, women_lookup, id_to_name)
    w2 = resolve_team(name2, women_lookup, id_to_name)
    if w1 is not None and w2 is not None:
        return w1, w2

    # Mixed - return whatever we found
    id1 = m1 if m1 is not None else w1
    id2 = m2 if m2 is not None else w2
    return id1, id2


def predict_matchup(team_a_id, team_b_id, season, artifacts, team_features, gender):
    """Predict P(lower-ID team wins) for a single matchup."""
    # Ensure TeamA < TeamB (submission format)
    a, b = min(team_a_id, team_b_id), max(team_a_id, team_b_id)

    matchup_row = pd.DataFrame({'Season': [season], 'TeamA': [a], 'TeamB': [b]})

    req = set(artifacts.get('lr_feature_cols', []))
    req.update(artifacts.get('xgb_feature_cols', []))
    req.update(artifacts.get('cb_feature_cols', []))
    req.add('Elo_diff')
    interaction_pairs = _extract_interaction_pairs(artifacts.get('lr_config', {}))

    feature_df = _build_matchup_features(
        matchup_row, team_features, sorted(req), interaction_pairs=interaction_pairs
    )

    prob = predict_stack_from_matchups(feature_df, artifacts, clip=True)
    p_a_wins = prob[0]  # P(TeamA wins) where TeamA = lower ID

    return a, b, p_a_wins


class LivePredictor:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.men_lookup, self.women_lookup = build_name_lookup(data_dir)
        self.id_to_name = build_id_to_name(data_dir)

        print("Loading data...")
        self.m_data = load_men_data(data_dir)
        self.w_data = load_women_data(data_dir)

        print("Building features...")
        self._build_features()

        print("Training models (all data through 2025 seasons)...")
        self._train_models()

        print("Ready!\n")

    def _build_features(self):
        # Men
        self.m_elo_df = compute_elo_ratings_v2(self.m_data['MComSsn'], MEN_ELO)
        self.m_features = build_team_features(
            self.m_elo_df, parse_seeds(self.m_data['MTrnySeeds']),
            compute_season_stats(self.m_data['MDetSsn']),
            compute_massey_features(self.m_data['MOrdinals']),
            compute_conference_strength(self.m_data['MConf'], self.m_elo_df),
            efficiency_df=compute_efficiency(self.m_data['MDetSsn']),
        )
        self.m_matchups = build_matchup_matrix(self.m_data['MDetTrny'], self.m_features)

        # Women
        self.w_elo_df = compute_elo_ratings_v2(self.w_data['WComSsn'], WOMEN_ELO)
        w_base = build_team_features(
            self.w_elo_df, parse_seeds(self.w_data['WTrnySeeds']),
            stats_df=compute_season_stats(self.w_data['WDetSsn']),
            conf_df=compute_conference_strength(self.w_data['WConf'], self.w_elo_df),
            efficiency_df=compute_efficiency(self.w_data['WDetSsn']),
        )
        wpr = compute_women_power_ratings(self.w_data['WComSsn'], WomenRankConfig())
        self.w_features = merge_women_rank_features(w_base, wpr)
        self.w_matchups = build_matchup_matrix(self.w_data['WDetTrny'], self.w_features)

    def _train_models(self):
        # Train on ALL available data (2010-2025 OOF, all seasons for base models)
        final_cfg = StackConfig(train_cutoff=2022, oof_start_season=2010, oof_end_season=2025)
        self.m_artifacts = train_stack_final(
            self.m_matchups, men_features_cfg, men_lr_cfg, final_cfg, max_train_season=2025
        )
        self.w_artifacts = train_stack_final(
            self.w_matchups, women_features_cfg, women_lr_cfg, final_cfg, max_train_season=2025
        )

    def predict(self, team1, team2, season=2026, gender=None):
        """Predict a matchup. Returns (team1_name, team2_name, p_team1_wins).

        gender: 'men', 'women', or None (auto-detect, men preferred).
        """
        if gender == 'women':
            id1 = resolve_team(team1, self.women_lookup, self.id_to_name)
            id2 = resolve_team(team2, self.women_lookup, self.id_to_name)
        elif gender == 'men':
            id1 = resolve_team(team1, self.men_lookup, self.id_to_name)
            id2 = resolve_team(team2, self.men_lookup, self.id_to_name)
        else:
            id1, id2 = resolve_team_pair(team1, team2, self.men_lookup, self.women_lookup, self.id_to_name)

        if id1 is None or id2 is None:
            return None

        # Determine gender
        is_women = id1 >= 3000
        if is_women != (id2 >= 3000):
            print("  Error: Can't predict cross-gender matchups")
            return None

        artifacts = self.w_artifacts if is_women else self.m_artifacts
        features = self.w_features if is_women else self.m_features

        a, b, p_a_wins = predict_matchup(id1, id2, season, artifacts, features, 'women' if is_women else 'men')

        # Apply women's alpha stretching (J config)
        if is_women:
            p_a_wins = float(stretch_preds(np.array([p_a_wins]), WOMEN_ALPHA)[0])

        # Convert to P(team1 wins)
        if id1 == a:
            p_team1 = p_a_wins
        else:
            p_team1 = 1.0 - p_a_wins

        name1 = self.id_to_name.get(id1, str(id1))
        name2 = self.id_to_name.get(id2, str(id2))

        return name1, name2, p_team1

    def predict_batch(self, matchups, season=2026, gender=None):
        """Predict a batch of matchups.

        Args:
            matchups: dict of {team1: team2} or list of (team1, team2) tuples
            season: season year (default 2026)
            gender: 'men', 'women', or None (auto-detect)

        Returns:
            pandas DataFrame with columns: Team1, Team2, Team1_Win%, Team2_Win%
        """
        if isinstance(matchups, dict):
            pairs = list(matchups.items())
        else:
            pairs = list(matchups)

        rows = []
        for team1, team2 in pairs:
            result = self.predict(team1, team2, season=season, gender=gender)
            if result is None:
                rows.append({'Team1': team1, 'Team2': team2,
                             'Team1_Win%': None, 'Team2_Win%': None})
            else:
                name1, name2, p1 = result
                rows.append({'Team1': name1, 'Team2': name2,
                             'Team1_Win%': round(p1 * 100, 1),
                             'Team2_Win%': round((1 - p1) * 100, 1)})

        return pd.DataFrame(rows)

    def predict_day(self, day_num, season=2026):
        """Fetch all games for a given DayNum from our data and predict them.

        Args:
            day_num: DayNum (0-154) from the season schedule
            season: season year (default 2026)

        Returns:
            DataFrame with Team1, Team2, Team1_Win%, Team2_Win%, Winner, T1_Score, T2_Score
        """
        rows = []
        for prefix, data, gender in [
            ('M', self.m_data, 'men'),
            ('W', self.w_data, 'women'),
        ]:
            # Check compact results (covers all seasons)
            key = f'{prefix}ComSsn'
            df = data[key]
            day_games = df[(df['Season'] == season) & (df['DayNum'] == day_num)]

            for _, game in day_games.iterrows():
                wid = int(game['WTeamID'])
                lid = int(game['LTeamID'])
                wscore = int(game['WScore'])
                lscore = int(game['LScore'])

                # Predict (team1 = lower ID for consistency)
                t1, t2 = min(wid, lid), max(wid, lid)
                name1 = self.id_to_name.get(t1, str(t1))
                name2 = self.id_to_name.get(t2, str(t2))

                result = self.predict(str(t1), str(t2), season=season, gender=gender)
                if result:
                    _, _, p1 = result
                else:
                    p1 = 0.5

                winner_name = self.id_to_name.get(wid, str(wid))
                t1_score = wscore if wid == t1 else lscore
                t2_score = lscore if wid == t1 else wscore

                # Did we predict the winner correctly?
                predicted_t1 = p1 > 0.5
                t1_won = wid == t1
                correct = predicted_t1 == t1_won

                rows.append({
                    'Gender': gender[0].upper(),
                    'Team1': name1,
                    'Team2': name2,
                    'Team1_Win%': round(p1 * 100, 1),
                    'Team2_Win%': round((1 - p1) * 100, 1),
                    'T1_Score': t1_score,
                    'T2_Score': t2_score,
                    'Winner': winner_name,
                    'Correct': correct,
                    'Brier': round((t1_won - p1) ** 2, 4),
                })

        result_df = pd.DataFrame(rows)
        if len(result_df) > 0:
            n_correct = result_df['Correct'].sum()
            n_total = len(result_df)
            mean_brier = result_df['Brier'].mean()
            print(f"DayNum {day_num} ({len(result_df)} games): "
                  f"{n_correct}/{n_total} correct ({n_correct/n_total:.0%}), "
                  f"Brier={mean_brier:.4f}")
        return result_df

    def predict_day_range(self, start_day, end_day, season=2026):
        """Predict all games across a range of DayNums.

        Returns combined DataFrame + summary stats.
        """
        all_dfs = []
        for d in range(start_day, end_day + 1):
            df = self.predict_day(d, season=season)
            if len(df) > 0:
                df['DayNum'] = d
                all_dfs.append(df)

        if not all_dfs:
            print("No games found in range.")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        n_correct = combined['Correct'].sum()
        n_total = len(combined)
        mean_brier = combined['Brier'].mean()

        men = combined[combined['Gender'] == 'M']
        women = combined[combined['Gender'] == 'W']

        print(f"\n{'='*50}")
        print(f"SUMMARY: DayNum {start_day}-{end_day}")
        print(f"{'='*50}")
        print(f"  Total: {n_correct}/{n_total} correct ({n_correct/n_total:.0%}), Brier={mean_brier:.4f}")
        if len(men) > 0:
            print(f"  Men:   {men['Correct'].sum()}/{len(men)} ({men['Correct'].mean():.0%}), Brier={men['Brier'].mean():.4f}")
        if len(women) > 0:
            print(f"  Women: {women['Correct'].sum()}/{len(women)} ({women['Correct'].mean():.0%}), Brier={women['Brier'].mean():.4f}")

        return combined

    def show_team_features(self, team, season=2026, gender=None):
        """Show a team's features for debugging."""
        if gender == 'women':
            tid = resolve_team(team, self.women_lookup, self.id_to_name)
        elif gender == 'men':
            tid = resolve_team(team, self.men_lookup, self.id_to_name)
        else:
            tid = resolve_team(team, self.men_lookup, self.id_to_name)
            if tid is None:
                tid = resolve_team(team, self.women_lookup, self.id_to_name)
        if tid is None:
            return

        is_women = tid >= 3000
        features = self.w_features if is_women else self.m_features
        row = features[(features['Season'] == season) & (features['TeamID'] == tid)]

        name = self.id_to_name.get(tid, str(tid))
        if len(row) == 0:
            print(f"  No {season} features for {name} (ID={tid})")
            # Show available seasons
            avail = features[features['TeamID'] == tid]['Season'].unique()
            if len(avail) > 0:
                print(f"  Available seasons: {sorted(avail)[-5:]}")
            return

        print(f"\n  {name} (ID={tid}, {season}):")
        for col in row.columns:
            if col not in ('Season', 'TeamID', 'ConfAbbrev'):
                val = row[col].values[0]
                if pd.notna(val):
                    print(f"    {col}: {val:.3f}" if isinstance(val, float) else f"    {col}: {val}")

    def interactive(self):
        """Interactive prediction loop."""
        print("=" * 60)
        print("LIVE MARCH MADNESS PREDICTOR (O config)")
        print("=" * 60)
        print("Commands:")
        print("  <team1> vs <team2>      - predict matchup (men auto)")
        print("  w: <team1> vs <team2>   - predict women's matchup")
        print("  info <team>             - show team features")
        print("  elo <team>              - show Elo rating")
        print("  quit                    - exit")
        print()

        predictions_log = []

        while True:
            try:
                line = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line:
                continue
            if line.lower() in ('quit', 'exit', 'q'):
                break

            if line.lower().startswith('info '):
                self.show_team_features(line[5:].strip())
                continue

            if line.lower().startswith('elo '):
                team = line[4:].strip()
                tid = resolve_team(team, self.men_lookup, self.id_to_name)
                if tid is None:
                    tid = resolve_team(team, self.women_lookup, self.id_to_name)
                if tid:
                    is_w = tid >= 3000
                    elo_df = self.w_elo_df if is_w else self.m_elo_df
                    row = elo_df[(elo_df['TeamID'] == tid) & (elo_df['Season'] == 2026)]
                    name = self.id_to_name.get(tid, str(tid))
                    if len(row) > 0:
                        print(f"  {name}: Elo = {row['Elo'].values[0]:.1f}")
                    else:
                        # Show latest available
                        latest = elo_df[elo_df['TeamID'] == tid].sort_values('Season').tail(1)
                        if len(latest) > 0:
                            print(f"  {name}: Elo = {latest['Elo'].values[0]:.1f} (season {int(latest['Season'].values[0])})")
                        else:
                            print(f"  No Elo for {name}")
                continue

            # Check for gender prefix
            gender = None
            matchline = line
            if line.lower().startswith('w:') or line.lower().startswith('w '):
                gender = 'women'
                matchline = line[2:].strip()
            elif line.lower().startswith('m:') or line.lower().startswith('m '):
                # Only if followed by something that looks like a matchup
                rest = line[2:].strip()
                if ' vs ' in rest.lower() or ' v ' in rest.lower() or ',' in rest:
                    gender = 'men'
                    matchline = rest

            # Parse "team1 vs team2"
            parts = None
            for sep in [' vs ', ' v ', ' vs. ', ',', ' - ']:
                if sep in matchline.lower():
                    idx = matchline.lower().index(sep)
                    parts = (matchline[:idx].strip(), matchline[idx+len(sep):].strip())
                    break

            if parts is None:
                # Try splitting by whitespace if exactly 2 tokens
                tokens = matchline.split()
                if len(tokens) == 2:
                    parts = (tokens[0], tokens[1])

            if parts is None:
                print("  Use: <team1> vs <team2>")
                continue

            result = self.predict(parts[0], parts[1], gender=gender)
            if result is None:
                continue

            name1, name2, p1 = result
            p2 = 1.0 - p1

            # Display
            bar_len = 40
            filled = int(round(p1 * bar_len))
            bar = '#' * filled + '-' * (bar_len - filled)

            print(f"\n  {name1} vs {name2}")
            print(f"  [{bar}]")
            print(f"  {name1}: {p1:.1%}    {name2}: {p2:.1%}")
            print()

            predictions_log.append({
                'timestamp': datetime.now().isoformat(),
                'team1': name1, 'team2': name2,
                'p_team1': round(p1, 4), 'p_team2': round(p2, 4),
            })

        # Save predictions log
        if predictions_log:
            log_path = './predictions_log.csv'
            pd.DataFrame(predictions_log).to_csv(log_path, index=False)
            print(f"\nSaved {len(predictions_log)} predictions to {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Live March Madness predictor')
    parser.add_argument('team1', nargs='?', help='First team name or ID')
    parser.add_argument('team2', nargs='?', help='Second team name or ID')
    parser.add_argument('--season', type=int, default=2026, help='Season year (default: 2026)')
    parser.add_argument('--score', type=str, help='Score predictions CSV against results')
    args = parser.parse_args()

    predictor = LivePredictor()

    if args.team1 and args.team2:
        result = predictor.predict(args.team1, args.team2, args.season)
        if result:
            name1, name2, p1 = result
            print(f"\n{name1} vs {name2}")
            print(f"  {name1}: {p1:.1%}")
            print(f"  {name2}: {1-p1:.1%}")
    else:
        predictor.interactive()


if __name__ == '__main__':
    main()
