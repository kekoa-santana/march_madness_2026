"""Walk the tournament bracket using probabilities from a submission CSV.

Reads the submission file directly — no model retraining needed.
Picks the team with >50% probability in each game.

Usage:
    python scripts/bracket_from_submission.py
    python scripts/bracket_from_submission.py --submission submissions/stage2_O_baseline.csv
    python scripts/bracket_from_submission.py --gender men
"""
import argparse
import pandas as pd

DATA_DIR = './data'


def load_submission(path):
    """Load submission CSV into a lookup: (season, teamA, teamB) -> P(teamA wins)."""
    sub = pd.read_csv(path)
    lookup = {}
    for _, row in sub.iterrows():
        parts = row['ID'].split('_')
        season, ta, tb = int(parts[0]), int(parts[1]), int(parts[2])
        lookup[(season, ta, tb)] = row['Pred']
    return lookup


def get_prob(lookup, season, id1, id2):
    """Get P(id1 wins) from the submission lookup."""
    a, b = min(id1, id2), max(id1, id2)
    p_a = lookup.get((season, a, b), 0.5)
    return p_a if id1 == a else 1.0 - p_a


def slot_to_round(slot_name):
    slot_name = slot_name.strip()
    if slot_name.startswith('R') and len(slot_name) == 4:
        return int(slot_name[1])
    return 0


ROUND_NAMES = {
    0: 'Play-in',
    1: 'Round 1 (64->32)',
    2: 'Round 2 (32->16)',
    3: 'Sweet 16',
    4: 'Elite Eight',
    5: 'Final Four',
    6: 'Championship',
}


def run_bracket(lookup, season, gender):
    prefix = 'W' if gender == 'women' else 'M'

    # Load teams for name lookup
    teams_df = pd.read_csv(f'{DATA_DIR}/{prefix}Teams.csv')
    id_to_name = dict(zip(teams_df['TeamID'], teams_df['TeamName']))

    # Load seeds
    seeds_df = pd.read_csv(f'{DATA_DIR}/{prefix}NCAATourneySeeds.csv')
    season_seeds = seeds_df[seeds_df['Season'] == season]
    seed_to_team = {}
    team_to_seed = {}
    for _, row in season_seeds.iterrows():
        seed_to_team[row['Seed'].strip()] = int(row['TeamID'])
        team_to_seed[int(row['TeamID'])] = row['Seed'].strip()

    # Load slots
    slots_df = pd.read_csv(f'{DATA_DIR}/{prefix}NCAATourneySlots.csv')
    season_slots = slots_df[slots_df['Season'] == season]
    slots = []
    for _, row in season_slots.iterrows():
        slots.append((row['Slot'].strip(), row['StrongSeed'].strip(), row['WeakSeed'].strip()))
    slots.sort(key=lambda s: slot_to_round(s[0]))

    # Load regions
    seasons_df = pd.read_csv(f'{DATA_DIR}/{prefix}Seasons.csv')
    srow = seasons_df[seasons_df['Season'] == season]
    region_names = {}
    if len(srow) > 0:
        r = srow.iloc[0]
        for letter in ['W', 'X', 'Y', 'Z']:
            col = f'Region{letter}'
            if col in r and pd.notna(r[col]):
                region_names[letter] = r[col]

    def resolve(ref, results):
        if ref in results:
            return results[ref]
        if ref in seed_to_team:
            return seed_to_team[ref]
        base = ref[:3] if len(ref) > 3 else ref
        if base in seed_to_team:
            return seed_to_team[base]
        raise ValueError(f"Cannot resolve '{ref}'")

    def seed_str(tid):
        s = team_to_seed.get(tid, '?')
        return s[1:3].lstrip('0') if len(s) >= 3 else '?'

    def name(tid):
        return id_to_name.get(tid, str(tid))

    # Walk the bracket
    results = {}
    game_log = []
    for slot_name, strong_ref, weak_ref in slots:
        round_num = slot_to_round(slot_name)
        team_a = resolve(strong_ref, results)
        team_b = resolve(weak_ref, results)
        p_a = get_prob(lookup, season, team_a, team_b)
        winner = team_a if p_a >= 0.5 else team_b
        results[slot_name] = winner
        game_log.append({
            'slot': slot_name, 'round': round_num,
            'team_a': team_a, 'team_b': team_b,
            'p_a': p_a, 'winner': winner,
        })

    # Print bracket
    print(f"\n{'#'*60}")
    print(f"  {gender.upper()} BRACKET — {season}")
    print(f"{'#'*60}")

    by_round = {}
    for g in game_log:
        by_round.setdefault(g['round'], []).append(g)

    for rnd in sorted(by_round.keys()):
        games = by_round[rnd]
        rnd_name = ROUND_NAMES.get(rnd, f'Round {rnd}')
        print(f"\n{'='*60}")
        print(f"  {rnd_name}")
        print(f"{'='*60}")

        if rnd <= 4:
            region_games = {}
            for g in games:
                rl = team_to_seed.get(g['team_a'], '?')[0]
                rn = region_names.get(rl, rl)
                region_games.setdefault(rn, []).append(g)
            for rn_key in sorted(region_games.keys()):
                print(f"\n  --- {rn_key} ---")
                for g in region_games[rn_key]:
                    _print_game(g, name, seed_str)
        else:
            for g in games:
                _print_game(g, name, seed_str)

    champ = results.get('R6CH')
    if champ:
        print(f"\n  CHAMPION: [{seed_str(champ)}] {name(champ)}")


def _print_game(g, name, seed_str):
    na, nb = name(g['team_a']), name(g['team_b'])
    sa, sb = seed_str(g['team_a']), seed_str(g['team_b'])
    pa, pb = g['p_a'], 1.0 - g['p_a']
    wn = name(g['winner'])
    marker = ""
    # Mark if the lower-seeded team won
    try:
        if int(sa) < int(sb) and g['winner'] == g['team_b']:
            marker = " *UPSET*"
        elif int(sb) < int(sa) and g['winner'] == g['team_a']:
            marker = " *UPSET*"
    except ValueError:
        pass
    print(f"  [{sa:>2}] {na:<18s} {pa:5.1%}  vs  {pb:5.1%}  [{sb:>2}] {nb:<18s} --> {wn}{marker}")


def main():
    parser = argparse.ArgumentParser(description='Bracket from submission CSV')
    parser.add_argument('--submission', type=str,
                        default='./submissions/stage2_final_round_alpha.csv',
                        help='Path to submission CSV')
    parser.add_argument('--gender', choices=['men', 'women', 'both'], default='both')
    parser.add_argument('--season', type=int, default=2026)
    args = parser.parse_args()

    print(f"Loading submission: {args.submission}")
    lookup = load_submission(args.submission)
    print(f"  {len(lookup)} matchups loaded")

    genders = ['men', 'women'] if args.gender == 'both' else [args.gender]
    for g in genders:
        run_bracket(lookup, args.season, g)


if __name__ == '__main__':
    main()
