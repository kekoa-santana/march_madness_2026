"""Tournament bracket simulator using our trained stacking model.

Walks the bracket round-by-round, applying round-specific alpha calibration.

Usage:
    python scripts/bracket.py --gender men --mode chalk --season 2025
    python scripts/bracket.py --gender women --mode upset --threshold 0.40
    python scripts/bracket.py --gender men --mode montecarlo --sims 10000
    python scripts/bracket.py --gender men --mode underdogs
    python scripts/bracket.py --gender both --season 2025
"""
import sys
import os
import json
import argparse
import random

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from scripts.live_predict import LivePredictor, predict_matchup, stretch_preds
from src.round_alpha import ROUND_NAMES

DATA_DIR = './data'
ROUND_ALPHAS_PATH = './round_alphas.json'


def load_round_alphas():
    with open(ROUND_ALPHAS_PATH) as f:
        raw = json.load(f)
    # Convert string keys to int
    return {
        gender: {int(k): v for k, v in alphas.items()}
        for gender, alphas in raw.items()
    }


def slot_to_round(slot_name):
    """Determine round number from slot name.

    Play-in slots: 'W16', 'X11', etc. (no 'R' prefix, 3-4 chars) -> round 0
    Regular slots: 'R1W1' -> round 1, 'R2W1' -> round 2, ..., 'R6CH' -> round 6
    """
    slot_name = slot_name.strip()
    if slot_name.startswith('R') and len(slot_name) == 4:
        return int(slot_name[1])
    # Play-in
    return 0


class BracketSimulator:
    def __init__(self, predictor, season, gender, round_alphas=None):
        """
        Args:
            predictor: LivePredictor instance (already trained)
            season: year (e.g. 2025)
            gender: 'men' or 'women'
            round_alphas: dict {round_num: alpha} or None
        """
        self.predictor = predictor
        self.season = season
        self.gender = gender
        self.is_women = (gender == 'women')
        self.prefix = 'W' if self.is_women else 'M'

        # Load round alphas
        if round_alphas is None:
            all_alphas = load_round_alphas()
            self.round_alphas = all_alphas.get(gender, {})
        else:
            self.round_alphas = round_alphas

        # Default alpha for women (global) when round not in JSON
        self.women_default_alpha = 1.2

        # Load bracket structure
        self._load_seeds()
        self._load_slots()
        self._load_regions()

        # Prediction cache: (team_a, team_b) -> raw_prob (before alpha)
        self._cache = {}

    def _load_seeds(self):
        path = os.path.join(DATA_DIR, f'{self.prefix}NCAATourneySeeds.csv')
        seeds_df = pd.read_csv(path)
        season_seeds = seeds_df[seeds_df['Season'] == self.season]

        if len(season_seeds) == 0:
            raise ValueError(
                f"No seeds found for {self.gender} {self.season}. "
                f"Available seasons: {sorted(seeds_df['Season'].unique())[-5:]}"
            )

        self.seed_to_team = {}  # 'W01' -> TeamID
        self.team_to_seed = {}  # TeamID -> 'W01'

        for _, row in season_seeds.iterrows():
            seed = row['Seed']
            tid = int(row['TeamID'])
            self.seed_to_team[seed] = tid
            self.team_to_seed[tid] = seed

    def _load_slots(self):
        path = os.path.join(DATA_DIR, f'{self.prefix}NCAATourneySlots.csv')
        slots_df = pd.read_csv(path)
        season_slots = slots_df[slots_df['Season'] == self.season]

        # Build slot list: [(slot_name, strong_ref, weak_ref), ...]
        self.slots = []
        for _, row in season_slots.iterrows():
            self.slots.append((
                row['Slot'].strip(),
                row['StrongSeed'].strip(),
                row['WeakSeed'].strip(),
            ))

        # Sort by round order (play-ins first, then R1, R2, ...)
        self.slots.sort(key=lambda s: slot_to_round(s[0]))

    def _load_regions(self):
        path = os.path.join(DATA_DIR, f'{self.prefix}Seasons.csv')
        seasons_df = pd.read_csv(path)
        row = seasons_df[seasons_df['Season'] == self.season]
        self.region_names = {}
        if len(row) > 0:
            r = row.iloc[0]
            for letter in ['W', 'X', 'Y', 'Z']:
                col = f'Region{letter}'
                if col in r and pd.notna(r[col]):
                    self.region_names[letter] = r[col]

    def _resolve_team(self, ref, results):
        """Resolve a slot reference to a TeamID.

        ref can be:
          - A seed string like 'W01', 'X16a' -> lookup in seed_to_team
          - A slot name like 'R1W1' -> lookup winner from results dict
        """
        # If it's a prior slot result
        if ref in results:
            return results[ref]
        # Seed lookup
        if ref in self.seed_to_team:
            return self.seed_to_team[ref]
        # Try without play-in suffix (shouldn't normally happen since
        # play-ins resolve before R1 uses their base seed)
        base = ref[:3] if len(ref) > 3 else ref
        if base in self.seed_to_team:
            return self.seed_to_team[base]
        raise ValueError(f"Cannot resolve reference '{ref}' (season={self.season})")

    def _get_round_alpha(self, round_num):
        """Get alpha for a given round, with fallbacks."""
        if round_num in self.round_alphas:
            return self.round_alphas[round_num]
        # Women's rounds not in JSON -> use global women alpha
        if self.is_women:
            return self.women_default_alpha
        # Men default: no stretching
        return 1.0

    def _predict_raw(self, team_a_id, team_b_id):
        """Get raw (un-stretched) P(lower-ID wins)."""
        a, b = min(team_a_id, team_b_id), max(team_a_id, team_b_id)
        key = (a, b)
        if key in self._cache:
            return self._cache[key]

        artifacts = self.predictor.w_artifacts if self.is_women else self.predictor.m_artifacts
        features = self.predictor.w_features if self.is_women else self.predictor.m_features

        _, _, p_a_wins = predict_matchup(a, b, self.season, artifacts, features, self.gender)
        self._cache[key] = p_a_wins
        return p_a_wins

    def _predict_stretched(self, team_a_id, team_b_id, round_num):
        """Get alpha-stretched P(team_a wins)."""
        a, b = min(team_a_id, team_b_id), max(team_a_id, team_b_id)
        raw_p_a = self._predict_raw(team_a_id, team_b_id)

        alpha = self._get_round_alpha(round_num)
        if alpha != 1.0:
            p_a = float(stretch_preds(np.array([raw_p_a]), alpha)[0])
        else:
            p_a = raw_p_a

        # Convert to P(team_a wins) if team_a != lower ID
        if team_a_id == a:
            return p_a
        else:
            return 1.0 - p_a

    def _team_name(self, tid):
        return self.predictor.id_to_name.get(tid, str(tid))

    def _team_seed_str(self, tid):
        """Return seed number like '1' or '16' for display."""
        seed = self.team_to_seed.get(tid, '')
        if len(seed) >= 3:
            return seed[1:3].lstrip('0') or '0'
        return '?'

    def simulate_chalk(self):
        """Pick the model favorite every game. Returns (results, game_log)."""
        results = {}  # slot_name -> winning TeamID
        game_log = []  # list of dicts for display

        for slot_name, strong_ref, weak_ref in self.slots:
            round_num = slot_to_round(slot_name)
            team_a = self._resolve_team(strong_ref, results)
            team_b = self._resolve_team(weak_ref, results)

            p_a = self._predict_stretched(team_a, team_b, round_num)

            if p_a >= 0.5:
                winner = team_a
            else:
                winner = team_b

            results[slot_name] = winner

            game_log.append({
                'slot': slot_name,
                'round': round_num,
                'team_a': team_a,
                'team_b': team_b,
                'p_a': p_a,
                'winner': winner,
                'is_upset': (winner == team_b and p_a >= 0.5) or (winner == team_a and p_a < 0.5),
            })

        return results, game_log

    def simulate_upset(self, threshold=0.35):
        """Pick the underdog when model gives them > threshold probability."""
        results = {}
        game_log = []

        for slot_name, strong_ref, weak_ref in self.slots:
            round_num = slot_to_round(slot_name)
            team_a = self._resolve_team(strong_ref, results)
            team_b = self._resolve_team(weak_ref, results)

            p_a = self._predict_stretched(team_a, team_b, round_num)

            # Determine favorite/underdog by seed number
            seed_a = int(self._team_seed_str(team_a) or 99)
            seed_b = int(self._team_seed_str(team_b) or 99)

            if seed_a < seed_b:
                # team_a is favored by seed
                underdog_prob = 1.0 - p_a
                if underdog_prob > threshold:
                    winner = team_b
                else:
                    winner = team_a
            elif seed_b < seed_a:
                # team_b is favored by seed
                underdog_prob = p_a
                if underdog_prob > threshold:
                    winner = team_a
                else:
                    winner = team_b
            else:
                # Same seed number (cross-region) - pick model favorite
                winner = team_a if p_a >= 0.5 else team_b

            results[slot_name] = winner

            game_log.append({
                'slot': slot_name,
                'round': round_num,
                'team_a': team_a,
                'team_b': team_b,
                'p_a': p_a,
                'winner': winner,
                'is_upset': winner != (team_a if p_a >= 0.5 else team_b),
            })

        return results, game_log

    def simulate_montecarlo(self, n_sims=10000, seed=42):
        """Run N random simulations, report advancement probabilities."""
        rng = random.Random(seed)
        # Track how many times each team reaches each round
        # advancement[team_id][round_num] = count
        advancement = {}

        for sim in range(n_sims):
            results = {}
            for slot_name, strong_ref, weak_ref in self.slots:
                round_num = slot_to_round(slot_name)
                team_a = self._resolve_team(strong_ref, results)
                team_b = self._resolve_team(weak_ref, results)

                p_a = self._predict_stretched(team_a, team_b, round_num)

                if rng.random() < p_a:
                    winner = team_a
                else:
                    winner = team_b

                results[slot_name] = winner

                # Record advancement for winner
                if winner not in advancement:
                    advancement[winner] = {}
                next_round = round_num + 1
                advancement[winner][next_round] = advancement[winner].get(next_round, 0) + 1

        # Convert to probabilities
        # Column names: "R32" = made it to round of 32 (won R1), etc.
        advance_names = {
            1: 'R32',        # won round 1 (64->32)
            2: 'S16',        # won round 2 (32->16)
            3: 'E8',         # won sweet 16
            4: 'F4',         # won elite eight
            5: 'F2',         # won final four (in championship)
            6: 'Champ',      # won championship
        }
        rows = []
        for tid, rounds in advancement.items():
            row = {
                'TeamID': tid,
                'Team': self._team_name(tid),
                'Seed': self.team_to_seed.get(tid, '?'),
            }
            for rnd in range(1, 7):
                col = advance_names.get(rnd, f'R{rnd}')
                count = rounds.get(rnd, 0)
                row[col] = round(count / n_sims * 100, 1)
            rows.append(row)

        df = pd.DataFrame(rows)
        if 'Champ' in df.columns:
            df = df.sort_values('Champ', ascending=False)

        return df

    def _resolve_playins(self):
        """Simulate play-in games (chalk) and return results dict."""
        results = {}
        for slot_name, strong_ref, weak_ref in self.slots:
            if slot_to_round(slot_name) != 0:
                continue
            team_a = self._resolve_team(strong_ref, results)
            team_b = self._resolve_team(weak_ref, results)
            p_a = self._predict_stretched(team_a, team_b, 0)
            results[slot_name] = team_a if p_a >= 0.5 else team_b
        return results

    def find_underdogs(self):
        """Find R1 upset opportunities sorted by underdog probability."""
        playin_results = self._resolve_playins()
        rows = []
        for slot_name, strong_ref, weak_ref in self.slots:
            round_num = slot_to_round(slot_name)
            if round_num != 1:
                continue

            team_a = self._resolve_team(strong_ref, playin_results)
            team_b = self._resolve_team(weak_ref, playin_results)

            p_a = self._predict_stretched(team_a, team_b, round_num)

            seed_a = int(self._team_seed_str(team_a) or 99)
            seed_b = int(self._team_seed_str(team_b) or 99)

            if seed_a < seed_b:
                fav, dog = team_a, team_b
                p_dog = 1.0 - p_a
            else:
                fav, dog = team_b, team_a
                p_dog = p_a

            region = self.team_to_seed.get(fav, '?')[0]
            region_name = self.region_names.get(region, region)

            rows.append({
                'Region': region_name,
                'Matchup': f"[{self._team_seed_str(fav)}] {self._team_name(fav)} vs [{self._team_seed_str(dog)}] {self._team_name(dog)}",
                'Underdog': self._team_name(dog),
                'Seed': f"#{self._team_seed_str(dog)}",
                'Upset%': round(p_dog * 100, 1),
            })

        df = pd.DataFrame(rows).sort_values('Upset%', ascending=False)
        return df

    def print_bracket(self, game_log):
        """Pretty-print bracket results."""
        # Group by round
        by_round = {}
        for g in game_log:
            rnd = g['round']
            if rnd not in by_round:
                by_round[rnd] = []
            by_round[rnd].append(g)

        for rnd in sorted(by_round.keys()):
            games = by_round[rnd]
            rnd_name = ROUND_NAMES.get(rnd, f'Round {rnd}')

            if rnd <= 4:
                # Group by region
                print(f"\n{'='*60}")
                print(f"  {rnd_name}")
                print(f"{'='*60}")

                # Sort games by region
                region_games = {}
                for g in games:
                    region_letter = self.team_to_seed.get(g['team_a'], '?')[0]
                    region_name = self.region_names.get(region_letter, region_letter)
                    if region_name not in region_games:
                        region_games[region_name] = []
                    region_games[region_name].append(g)

                for region_name in sorted(region_games.keys()):
                    print(f"\n  --- {region_name} ---")
                    for g in region_games[region_name]:
                        self._print_game(g)
            else:
                print(f"\n{'='*60}")
                print(f"  {rnd_name}")
                print(f"{'='*60}")
                for g in games:
                    self._print_game(g)

    def _print_game(self, g):
        name_a = self._team_name(g['team_a'])
        name_b = self._team_name(g['team_b'])
        seed_a = self._team_seed_str(g['team_a'])
        seed_b = self._team_seed_str(g['team_b'])
        p_a = g['p_a']
        p_b = 1.0 - p_a
        winner_name = self._team_name(g['winner'])
        upset_tag = " *UPSET*" if g.get('is_upset') else ""

        print(f"  [{seed_a:>2}] {name_a:<18s} {p_a:5.1%}  vs  {p_b:5.1%}  [{seed_b:>2}] {name_b:<18s} --> {winner_name}{upset_tag}")


def run_bracket(predictor, season, gender, mode, threshold=0.35, sims=10000, mc_seed=42):
    """Run a single bracket simulation."""
    print(f"\n{'#'*60}")
    print(f"  {gender.upper()} BRACKET — {season} — mode: {mode}")
    print(f"{'#'*60}")

    sim = BracketSimulator(predictor, season, gender)

    if mode == 'chalk':
        results, game_log = sim.simulate_chalk()
        sim.print_bracket(game_log)
        champ = results.get('R6CH')
        if champ:
            print(f"\n  CHAMPION: [{sim._team_seed_str(champ)}] {sim._team_name(champ)}")

    elif mode == 'upset':
        results, game_log = sim.simulate_upset(threshold=threshold)
        sim.print_bracket(game_log)
        champ = results.get('R6CH')
        if champ:
            print(f"\n  CHAMPION: [{sim._team_seed_str(champ)}] {sim._team_name(champ)}")
        n_upsets = sum(1 for g in game_log if g.get('is_upset'))
        print(f"  Total upsets picked: {n_upsets}")

    elif mode == 'montecarlo':
        print(f"  Running {sims:,} simulations...")
        df = sim.simulate_montecarlo(n_sims=sims, seed=mc_seed)
        print(f"\n  Top 20 teams by championship probability:")
        display_cols = ['Seed', 'Team']
        for col in ['R32', 'S16', 'E8', 'F4', 'F2', 'Champ']:
            if col in df.columns:
                display_cols.append(col)
        top = df[display_cols].head(20)
        print(top.to_string(index=False))

    elif mode == 'underdogs':
        df = sim.find_underdogs()
        print(f"\n  Round 1 Upset Opportunities (sorted by upset probability):\n")
        print(df.to_string(index=False))

    return sim


def main():
    parser = argparse.ArgumentParser(description='Tournament bracket simulator')
    parser.add_argument('--gender', choices=['men', 'women', 'both'], default='men')
    parser.add_argument('--mode', choices=['chalk', 'upset', 'montecarlo', 'underdogs'], default='chalk')
    parser.add_argument('--season', type=int, default=2025)
    parser.add_argument('--threshold', type=float, default=0.35, help='Upset threshold (for upset mode)')
    parser.add_argument('--sims', type=int, default=10000, help='Number of Monte Carlo simulations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for Monte Carlo')
    args = parser.parse_args()

    predictor = LivePredictor()

    genders = ['men', 'women'] if args.gender == 'both' else [args.gender]

    for g in genders:
        run_bracket(predictor, args.season, g, args.mode,
                    threshold=args.threshold, sims=args.sims, mc_seed=args.seed)


if __name__ == '__main__':
    main()
