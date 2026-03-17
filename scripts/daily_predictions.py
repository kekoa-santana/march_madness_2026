"""Fetch ALL D1 basketball games for a date, predict, and score.

Usage:
    # Morning: fetch schedule + predict all games
    python scripts/daily_predictions.py --date 2026-03-07

    # Evening: re-fetch with scores and evaluate
    python scripts/daily_predictions.py --date 2026-03-07 --score

    # Default: today
    python scripts/daily_predictions.py
    python scripts/daily_predictions.py --score
"""
import sys
import os
import json
import argparse
from datetime import datetime, date
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

# ESPN API base — returns JSON, no scraping needed
ESPN_API = "https://site.api.espn.com/apis/site/v2/sports/basketball/{sport}/scoreboard"

# ESPN conference group IDs (comprehensive list for D1)
# We iterate through all of these and deduplicate by game ID
CONFERENCE_GROUPS = list(range(1, 63))

# Status values from ESPN
STATUS_FINAL = 'STATUS_FINAL'
STATUS_SCHEDULED = 'STATUS_SCHEDULED'
STATUS_IN_PROGRESS = 'STATUS_IN_PROGRESS'


def fetch_espn_games(date_str, sport='mens-college-basketball'):
    """Fetch all D1 games for a date from ESPN API.

    Args:
        date_str: 'YYYYMMDD' format
        sport: 'mens-college-basketball' or 'womens-college-basketball'

    Returns:
        dict of game_id -> game_info
    """
    games = {}
    gender = 'M' if 'mens' in sport and 'womens' not in sport else 'W'

    for group in CONFERENCE_GROUPS:
        url = f"{ESPN_API.format(sport=sport)}?dates={date_str}&groups={group}&limit=200"
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except (URLError, json.JSONDecodeError, TimeoutError):
            continue

        for event in data.get('events', []):
            gid = event.get('id')
            if gid in games:
                continue  # already seen from another conference

            comp = event.get('competitions', [{}])[0]
            competitors = comp.get('competitors', [])
            if len(competitors) != 2:
                continue

            status_obj = comp.get('status', {}).get('type', {})
            status = status_obj.get('name', '')

            # Parse teams
            home, away = None, None
            for c in competitors:
                team = c.get('team', {})
                info = {
                    'espn_id': team.get('id'),
                    'location': team.get('location', ''),
                    'name': team.get('name', ''),
                    'abbreviation': team.get('abbreviation', ''),
                    'display_name': team.get('displayName', ''),
                    'score': c.get('score', ''),
                    'winner': c.get('winner', False),
                }
                if c.get('homeAway') == 'home':
                    home = info
                else:
                    away = info

            if not home or not away:
                continue

            games[gid] = {
                'game_id': gid,
                'gender': gender,
                'home': home,
                'away': away,
                'status': status,
                'status_detail': status_obj.get('shortDetail', ''),
                'conference': comp.get('conference', {}).get('name', ''),
            }

    return games


def build_espn_name_map(data_dir):
    """Build ESPN location name -> our TeamID mapping.

    Uses our team spellings files to create a lookup, then adds
    common ESPN-specific mappings.
    """
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


def resolve_espn_team(team_info, lookup):
    """Try to resolve an ESPN team to our TeamID.

    Tries multiple name forms: location, abbreviation, display_name.
    """
    candidates = [
        team_info['location'].lower(),
        team_info['abbreviation'].lower(),
        team_info['display_name'].lower(),
        team_info['name'].lower(),
    ]

    # Direct match
    for c in candidates:
        if c in lookup:
            return lookup[c]

    # Normalize unicode (ESPN uses accented chars like San José)
    import unicodedata
    loc = team_info['location'].lower()
    loc_ascii = unicodedata.normalize('NFKD', loc).encode('ascii', 'ignore').decode('ascii')

    # Try common transformations
    transforms = [
        loc, loc_ascii,
        loc.replace('state', 'st'), loc_ascii.replace('state', 'st'),
        loc.replace('st.', 'state'),
        loc.replace("'", ''),
        loc.replace('-', ' '), loc_ascii.replace('-', ' '),
        loc.replace('cal state ', 'cs '),
        loc.replace('california state ', 'cs '),
        loc.replace('university', ''),
        loc.replace(' of ', ' '),
        loc.replace('north carolina', 'nc'),
        loc.replace('south carolina', 'sc'),
    ]
    # ESPN-specific hardcoded mappings
    ESPN_FIXES = {
        'miami': 'miami fl',
        'app state': 'appalachian st',
        'appalachian state': 'appalachian st',
        'st. thomas-minnesota': 'st thomas mn',
        'st. thomas - minnesota': 'st thomas mn',
        'san jose state': 'san jose st',
        'san jos\xe9 state': 'san jose st',
        'uc san diego': 'uc san diego',
        'lsu': 'louisiana st',
        'vcu': 'virginia commonwealth',
        'smu': 'southern methodist',
        'uconn': 'connecticut',
        'umass': 'massachusetts',
        'ole miss': 'mississippi',
        'pitt': 'pittsburgh',
        'ucf': 'central florida',
    }
    if loc_ascii in ESPN_FIXES:
        transforms.append(ESPN_FIXES[loc_ascii])
    if loc in ESPN_FIXES:
        transforms.append(ESPN_FIXES[loc])

    for t in transforms:
        t = t.strip()
        if t in lookup:
            return lookup[t]

    # Substring match — find unique match
    matches = set()
    for key, tid in lookup.items():
        if loc_ascii in key or key in loc_ascii:
            matches.add(tid)
    if len(matches) == 1:
        return matches.pop()

    return None


def fetch_all_games(target_date, data_dir='./data'):
    """Fetch all men's + women's D1 games for a date.

    Returns DataFrame with columns:
        game_id, gender, home_name, away_name, home_espn, away_espn,
        home_id, away_id, status, home_score, away_score, conference
    """
    date_str = target_date.strftime('%Y%m%d')
    men_lookup, women_lookup = build_espn_name_map(data_dir)

    # Build ID -> name lookup
    id_to_name = {}
    for fname in ['MTeams.csv', 'WTeams.csv']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                id_to_name[int(row['TeamID'])] = str(row['TeamName'])

    rows = []

    for sport, lookup, gender in [
        ('mens-college-basketball', men_lookup, 'M'),
        ('womens-college-basketball', women_lookup, 'W'),
    ]:
        print(f"  Fetching {gender} games for {target_date}...")
        games = fetch_espn_games(date_str, sport)
        print(f"    Found {len(games)} {gender} games")

        for gid, g in games.items():
            home_id = resolve_espn_team(g['home'], lookup)
            away_id = resolve_espn_team(g['away'], lookup)

            home_score = g['home']['score']
            away_score = g['away']['score']

            rows.append({
                'game_id': gid,
                'gender': gender,
                'home_espn': g['home']['location'],
                'away_espn': g['away']['location'],
                'home_name': id_to_name.get(home_id, g['home']['location']) if home_id else g['home']['location'],
                'away_name': id_to_name.get(away_id, g['away']['location']) if away_id else g['away']['location'],
                'home_id': home_id,
                'away_id': away_id,
                'status': g['status'],
                'status_detail': g['status_detail'],
                'home_score': int(home_score) if home_score and home_score.isdigit() else None,
                'away_score': int(away_score) if away_score and away_score.isdigit() else None,
                'conference': g['conference'],
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("  No games found!")
        return df

    # Report unresolved teams
    unresolved = df[(df['home_id'].isna()) | (df['away_id'].isna())]
    if len(unresolved) > 0:
        print(f"\n  WARNING: {len(unresolved)} games with unresolved teams:")
        for _, r in unresolved.iterrows():
            missing = []
            if pd.isna(r['home_id']):
                missing.append(f"home={r['home_espn']}")
            if pd.isna(r['away_id']):
                missing.append(f"away={r['away_espn']}")
            print(f"    {r['gender']} {r['away_espn']} @ {r['home_espn']}: {', '.join(missing)}")

    return df


def predict_games(games_df, predictor):
    """Run predictions on all resolvable games.

    Args:
        games_df: DataFrame from fetch_all_games()
        predictor: LivePredictor instance

    Returns:
        games_df with added columns: pred_home, pred_away, pick, confidence
    """
    preds_home = []
    preds_away = []

    for _, row in games_df.iterrows():
        if pd.isna(row['home_id']) or pd.isna(row['away_id']):
            preds_home.append(None)
            preds_away.append(None)
            continue

        home_id = int(row['home_id'])
        away_id = int(row['away_id'])
        gender = 'women' if row['gender'] == 'W' else 'men'

        result = predictor.predict(str(home_id), str(away_id), gender=gender)
        if result is None:
            preds_home.append(None)
            preds_away.append(None)
            continue

        name1, name2, p1 = result
        # p1 = P(team1 wins), where team1 is the first arg we passed (home_id)
        preds_home.append(round(p1 * 100, 1))
        preds_away.append(round((1 - p1) * 100, 1))

    games_df = games_df.copy()
    games_df['pred_home'] = preds_home
    games_df['pred_away'] = preds_away

    # Determine pick and confidence
    def get_pick(row):
        if pd.isna(row['pred_home']) or pd.isna(row['pred_away']):
            return None
        return row['home_name'] if row['pred_home'] > row['pred_away'] else row['away_name']

    def get_confidence(row):
        if pd.isna(row['pred_home']):
            return None
        return max(row['pred_home'], row['pred_away'])

    games_df['pick'] = games_df.apply(get_pick, axis=1)
    games_df['confidence'] = games_df.apply(get_confidence, axis=1)

    return games_df


def score_games(games_df):
    """Score predictions against actual results.

    Returns games_df with added columns: winner, correct, brier
    """
    df = games_df.copy()

    def get_winner(row):
        if pd.isna(row['home_score']) or pd.isna(row['away_score']):
            return None
        if row['home_score'] > row['away_score']:
            return row['home_name']
        return row['away_name']

    df['winner'] = df.apply(get_winner, axis=1)

    def get_correct(row):
        if row['winner'] is None or row['pick'] is None:
            return None
        return row['pick'] == row['winner']

    df['correct'] = df.apply(get_correct, axis=1)

    def get_brier(row):
        if pd.isna(row['pred_home']) or pd.isna(row['home_score']) or pd.isna(row['away_score']):
            return None
        home_won = 1.0 if row['home_score'] > row['away_score'] else 0.0
        return round((home_won - row['pred_home'] / 100) ** 2, 4)

    df['brier'] = df.apply(get_brier, axis=1)

    return df


def print_predictions(df, show_scores=False):
    """Pretty-print predictions table."""
    # Split by gender
    for gender, label in [('M', 'MEN'), ('W', 'WOMEN')]:
        gdf = df[df['gender'] == gender].copy()
        if len(gdf) == 0:
            continue

        predicted = gdf[gdf['pred_home'].notna()]
        unresolved = gdf[gdf['pred_home'].isna()]

        print(f"\n{'='*80}")
        print(f"  {label}: {len(gdf)} games ({len(predicted)} predicted, {len(unresolved)} unresolved)")
        print(f"{'='*80}")

        if len(predicted) > 0:
            # Sort by confidence descending
            predicted = predicted.sort_values('confidence', ascending=False)

            for _, r in predicted.iterrows():
                conf = r['confidence']

                if show_scores and r.get('winner') is not None:
                    icon = 'OK' if r.get('correct') else 'XX'
                    brier = r.get('brier', '')
                    brier_str = f" Brier={brier:.3f}" if pd.notna(brier) else ''
                    score_str = f" {int(r['away_score'])}-{int(r['home_score'])}"
                    print(f"  {icon} {r['away_name']:>18s} @{r['home_name']:<18s}"
                          f"  Pick: {r['pick']:<18s} ({conf:.0f}%)"
                          f"  Result:{score_str}  Winner: {r['winner']}{brier_str}")
                else:
                    status = r['status_detail'] if r['status'] != STATUS_SCHEDULED else ''
                    print(f"     {r['away_name']:>18s} @{r['home_name']:<18s}"
                          f"  Pick: {r['pick']:<18s} ({conf:.0f}%)"
                          f"  {status}")

        if len(unresolved) > 0:
            print(f"\n  Unresolved ({len(unresolved)}):")
            for _, r in unresolved.iterrows():
                print(f"    {r['away_espn']} @ {r['home_espn']}")


def print_summary(df):
    """Print scoring summary."""
    scored = df[df['correct'].notna()].copy()
    if len(scored) == 0:
        print("\nNo scored games yet.")
        return

    for gender, label in [('M', 'MEN'), ('W', 'WOMEN'), (None, 'COMBINED')]:
        if gender:
            gdf = scored[scored['gender'] == gender]
        else:
            gdf = scored

        if len(gdf) == 0:
            continue

        correct = gdf['correct'].sum()
        total = len(gdf)
        brier = gdf['brier'].mean()

        print(f"\n  {label}: {correct}/{total} ({100*correct/total:.1f}%) correct, Mean Brier: {brier:.4f}")

        # Breakdown by confidence
        high = gdf[gdf['confidence'] >= 65]
        mid = gdf[(gdf['confidence'] >= 55) & (gdf['confidence'] < 65)]
        low = gdf[gdf['confidence'] < 55]
        for bucket, lbl in [(high, '65%+'), (mid, '55-65%'), (low, '<55%')]:
            if len(bucket) > 0:
                bc = bucket['correct'].sum()
                bt = len(bucket)
                print(f"    {lbl}: {bc}/{bt} ({100*bc/bt:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description='Daily game predictions')
    parser.add_argument('--date', type=str, default=None,
                        help='Date to predict (YYYY-MM-DD, default: today)')
    parser.add_argument('--score', action='store_true',
                        help='Re-fetch results and score predictions')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                        help='Path to save predictions')
    args = parser.parse_args()

    # Parse date
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = date.today()

    date_tag = target_date.strftime('%Y%m%d')
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f'predictions_{date_tag}.csv')

    if args.score:
        # Score mode: re-fetch results and evaluate
        if not os.path.exists(pred_path):
            print(f"No predictions found at {pred_path}. Run without --score first.")
            return

        print(f"Loading predictions from {pred_path}...")
        saved = pd.read_csv(pred_path)

        print("Re-fetching game results...")
        fresh = fetch_all_games(target_date, args.data_dir)

        # Merge fresh scores into saved predictions
        score_map = {}
        for _, r in fresh.iterrows():
            score_map[r['game_id']] = (r['home_score'], r['away_score'], r['status'])

        home_scores, away_scores, statuses = [], [], []
        for _, r in saved.iterrows():
            gid = str(r['game_id'])
            if gid in score_map:
                hs, as_, st = score_map[gid]
                home_scores.append(hs)
                away_scores.append(as_)
                statuses.append(st)
            else:
                home_scores.append(r.get('home_score'))
                away_scores.append(r.get('away_score'))
                statuses.append(r.get('status'))

        saved['home_score'] = home_scores
        saved['away_score'] = away_scores
        saved['status'] = statuses

        scored = score_games(saved)
        print_predictions(scored, show_scores=True)

        final_games = scored[scored['status'] == STATUS_FINAL]
        print(f"\n{'='*80}")
        print(f"  FINAL RESULTS — {target_date}")
        print(f"{'='*80}")
        print_summary(final_games)

        # Save updated
        scored.to_csv(pred_path, index=False)
        print(f"\nUpdated {pred_path}")

    else:
        # Predict mode: fetch schedule + predict
        print(f"Fetching games for {target_date}...")
        games = fetch_all_games(target_date, args.data_dir)

        if len(games) == 0:
            return

        total = len(games)
        resolvable = games[(games['home_id'].notna()) & (games['away_id'].notna())]
        print(f"\n  Total: {total} games, {len(resolvable)} resolvable")

        print("\nInitializing predictor...")
        from scripts.live_predict import LivePredictor
        predictor = LivePredictor(data_dir=args.data_dir)

        print("Running predictions...")
        predicted = predict_games(games, predictor)

        # If any games already have scores (in-progress or final), score them
        has_scores = predicted[(predicted['home_score'].notna()) & (predicted['away_score'].notna())]
        if len(has_scores) > 0:
            predicted = score_games(predicted)
            print_predictions(predicted, show_scores=True)
            final_games = predicted[predicted['status'] == STATUS_FINAL]
            if len(final_games) > 0:
                print(f"\n{'='*80}")
                print(f"  SCORED GAMES — {target_date}")
                print(f"{'='*80}")
                print_summary(final_games)
        else:
            print_predictions(predicted, show_scores=False)

        # Save
        predicted.to_csv(pred_path, index=False)
        print(f"\nSaved {len(predicted)} predictions to {pred_path}")
        print(f"Run with --score later to evaluate: python scripts/daily_predictions.py --date {target_date} --score")


if __name__ == '__main__':
    main()
