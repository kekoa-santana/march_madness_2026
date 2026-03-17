"""Preflight check and orchestrator for March Madness submission pipeline.

Validates data, checks readiness, and optionally runs the full pipeline.

Usage:
    python scripts/preflight.py                  # check only
    python scripts/preflight.py --run-all        # check + generate everything
    python scripts/preflight.py --run-submission  # generate submission only
    python scripts/preflight.py --run-bracket    # generate bracket only
"""
import sys
import os
import argparse
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

DATA_DIR = './data'
PYTHON = '/c/Users/kekoa/anaconda3/python.exe'

# ============================================================
# Required files for each pipeline stage
# ============================================================

# Core data files loaded by data_loader.py
CORE_MEN_FILES = [
    'MRegularSeasonCompactResults.csv',
    'MRegularSeasonDetailedResults.csv',
    'MNCAATourneyDetailedResults.csv',
    'MNCAATourneySeeds.csv',
    'MMasseyOrdinals.csv',
    'MTeamConferences.csv',
    'MTeams.csv',
    'MTeamCoaches.csv',
]

CORE_WOMEN_FILES = [
    'WRegularSeasonCompactResults.csv',
    'WRegularSeasonDetailedResults.csv',
    'WNCAATourneyDetailedResults.csv',
    'WNCAATourneySeeds.csv',
    'WTeamConferences.csv',
    'WTeams.csv',
]

# Additional files needed for submission generation
SUBMISSION_FILES = [
    'SampleSubmissionStage2.csv',
]

# Additional files for round-specific alphas and bracket
BRACKET_FILES = [
    'MNCAATourneySeedRoundSlots.csv',
    'MNCAATourneySlots.csv',
    'WNCAATourneySlots.csv',
    'MSeasons.csv',
    'WSeasons.csv',
]

# Name lookup files (used by live_predict / bracket)
NAME_FILES = [
    'MTeamSpellings.csv',
    'WTeamSpellings.csv',
]


def check_file(filepath, label=None):
    """Check if file exists and is non-empty. Returns (ok, message)."""
    if not os.path.exists(filepath):
        return False, f"MISSING: {filepath}"
    size = os.path.getsize(filepath)
    if size == 0:
        return False, f"EMPTY: {filepath} (0 bytes)"
    return True, f"OK: {filepath} ({size:,} bytes)"


def check_2026_data(filepath, season_col='Season'):
    """Check if a CSV contains 2026 season data. Returns (has_2026, max_season, n_2026_rows)."""
    try:
        df = pd.read_csv(filepath)
        if season_col not in df.columns:
            return None, None, None
        max_season = int(df[season_col].max())
        n_2026 = int((df[season_col] == 2026).sum())
        return n_2026 > 0, max_season, n_2026
    except Exception as e:
        return None, None, None


def check_seeds_2026(gender='men'):
    """Check if 2026 tournament seeds exist."""
    prefix = 'M' if gender == 'men' else 'W'
    path = os.path.join(DATA_DIR, f'{prefix}NCAATourneySeeds.csv')
    if not os.path.exists(path):
        return False, 0
    df = pd.read_csv(path)
    seeds_2026 = df[df['Season'] == 2026]
    return len(seeds_2026) > 0, len(seeds_2026)


def run_preflight():
    """Run all preflight checks. Returns (all_ok, post_selection_ready)."""
    print("=" * 60)
    print("  MARCH MADNESS PIPELINE — PREFLIGHT CHECK")
    print("=" * 60)

    all_ok = True
    warnings = []

    # ----------------------------------------------------------
    # 1. Core data files
    # ----------------------------------------------------------
    print("\n[1] Core data files")
    for f in CORE_MEN_FILES + CORE_WOMEN_FILES:
        path = os.path.join(DATA_DIR, f)
        ok, msg = check_file(path)
        status = "  OK " if ok else "  FAIL"
        print(f"  {status}  {f}")
        if not ok:
            all_ok = False

    # ----------------------------------------------------------
    # 2. Submission template
    # ----------------------------------------------------------
    print("\n[2] Submission template")
    for f in SUBMISSION_FILES:
        path = os.path.join(DATA_DIR, f)
        ok, msg = check_file(path)
        status = "  OK " if ok else "  FAIL"
        print(f"  {status}  {f}")
        if not ok:
            all_ok = False
        else:
            # Check it has 2026 rows
            df = pd.read_csv(path, nrows=5)
            first_id = df['ID'].iloc[0]
            season = int(first_id.split('_')[0])
            print(f"         Season in template: {season}")
            if season != 2026:
                warnings.append(f"SampleSubmissionStage2.csv has season {season}, expected 2026")

    # ----------------------------------------------------------
    # 3. Bracket / round-alpha files
    # ----------------------------------------------------------
    print("\n[3] Bracket structure files")
    for f in BRACKET_FILES:
        path = os.path.join(DATA_DIR, f)
        ok, msg = check_file(path)
        status = "  OK " if ok else "  WARN"
        print(f"  {status}  {f}")
        # These are warnings, not fatal — only needed post-Selection Sunday

    # ----------------------------------------------------------
    # 4. 2026 regular season data freshness
    # ----------------------------------------------------------
    print("\n[4] 2026 regular season data")
    for label, f in [("Men compact", "MRegularSeasonCompactResults.csv"),
                     ("Men detailed", "MRegularSeasonDetailedResults.csv"),
                     ("Women compact", "WRegularSeasonCompactResults.csv"),
                     ("Women detailed", "WRegularSeasonDetailedResults.csv")]:
        path = os.path.join(DATA_DIR, f)
        has_2026, max_season, n_rows = check_2026_data(path)
        if has_2026:
            print(f"  OK    {label}: {n_rows:,} games in 2026")
        elif max_season:
            print(f"  WARN  {label}: max season is {max_season} (no 2026 data)")
            warnings.append(f"{f} has no 2026 data — download fresh data from Kaggle")
        else:
            print(f"  FAIL  {label}: could not read")

    # Check Massey ordinals for 2026
    path = os.path.join(DATA_DIR, 'MMasseyOrdinals.csv')
    has_2026, max_season, n_rows = check_2026_data(path)
    if has_2026:
        print(f"  OK    Massey ordinals: {n_rows:,} rankings in 2026")
    else:
        print(f"  WARN  Massey ordinals: no 2026 data (max season={max_season})")
        warnings.append("MMasseyOrdinals.csv missing 2026 — Pomeroy features will be stale")

    # ----------------------------------------------------------
    # 5. 2026 tournament seeds (post-Selection Sunday)
    # ----------------------------------------------------------
    print("\n[5] 2026 tournament seeds (Selection Sunday)")
    post_selection = True
    for gender in ['men', 'women']:
        has_seeds, n_seeds = check_seeds_2026(gender)
        if has_seeds:
            print(f"  OK    {gender.title()}: {n_seeds} seeds for 2026")
        else:
            print(f"  ---   {gender.title()}: no 2026 seeds yet (expected after March 15)")
            post_selection = False

    # ----------------------------------------------------------
    # 6. Round alphas cache
    # ----------------------------------------------------------
    print("\n[6] Round alphas cache")
    alphas_path = './round_alphas.json'
    ok, msg = check_file(alphas_path)
    if ok:
        print(f"  OK    {alphas_path} exists (can use --skip-optimize)")
    else:
        print(f"  WARN  {alphas_path} not found (will need to optimize from scratch)")

    # ----------------------------------------------------------
    # 7. Output directory
    # ----------------------------------------------------------
    print("\n[7] Output directory")
    if os.path.isdir('./submissions'):
        print(f"  OK    ./submissions/ exists")
    else:
        print(f"  WARN  ./submissions/ missing — will create on first run")

    # ----------------------------------------------------------
    # 8. Feature NaN validation for current season
    # ----------------------------------------------------------
    print("\n[8] Feature NaN validation (current season)")
    try:
        from src.data_loader import load_men_data, load_women_data
        from src.feature_engineering import (
            parse_seeds, compute_season_stats, compute_massey_features,
            compute_conference_strength, compute_efficiency, build_team_features,
        )
        from src.model_elo_v2 import compute_elo_ratings_v2
        from src.women_rankings_v1 import WomenRankConfig, compute_women_power_ratings, merge_women_rank_features
        from src.config import (
            MEN_ELO, WOMEN_ELO,
            O_MEN_FEATURES, O_WOMEN_FEATURES,
            O_MEN_LR, O_WOMEN_LR,
        )

        m_data = load_men_data(DATA_DIR)
        w_data = load_women_data(DATA_DIR)

        max_season = int(m_data['MComSsn']['Season'].max())

        # Build men's features
        m_elo = compute_elo_ratings_v2(m_data['MComSsn'], MEN_ELO)
        m_seeds = parse_seeds(m_data['MTrnySeeds'])
        m_features = build_team_features(
            m_elo, m_seeds,
            compute_season_stats(m_data['MDetSsn']),
            compute_massey_features(m_data['MOrdinals']),
            compute_conference_strength(m_data['MConf'], m_elo),
            efficiency_df=compute_efficiency(m_data['MDetSsn']),
        )

        # Build women's features
        w_elo = compute_elo_ratings_v2(w_data['WComSsn'], WOMEN_ELO)
        w_base = build_team_features(
            w_elo, parse_seeds(w_data['WTrnySeeds']),
            stats_df=compute_season_stats(w_data['WDetSsn']),
            conf_df=compute_conference_strength(w_data['WConf'], w_elo),
            efficiency_df=compute_efficiency(w_data['WDetSsn']),
        )
        wpr = compute_women_power_ratings(w_data['WComSsn'], WomenRankConfig())
        w_features = merge_women_rank_features(w_base, wpr)

        # Check NaNs for teams with seeds in the latest season
        def check_feature_nans(features_df, feature_cfg, seeds_df, gender_label, season):
            """Check for NaNs in features that the pipeline actually uses."""
            season_feats = features_df[features_df['Season'] == season].copy()
            seeded_teams = seeds_df[seeds_df['Season'] == season]['TeamID'].unique()
            seeded_feats = season_feats[season_feats['TeamID'].isin(seeded_teams)]

            if len(seeded_feats) == 0:
                print(f"  ---   {gender_label}: no seeded teams in {season} (pre-Selection Sunday)")
                # Fall back to checking ALL teams in that season as a sanity check
                if len(season_feats) > 0:
                    seeded_feats = season_feats
                    print(f"         Checking all {len(seeded_feats)} teams instead...")
                else:
                    return []

            # Collect all feature columns used by the pipeline
            used_features = set()
            for key in ['xgb_features', 'cb_features']:
                for f in feature_cfg.get(key, []):
                    # Strip _diff suffix to get the base feature name
                    base = f.replace('_diff', '')
                    used_features.add(base)

            nan_issues = []
            for col in sorted(used_features):
                if col not in seeded_feats.columns:
                    nan_issues.append((col, len(seeded_feats), 'MISSING COLUMN'))
                    continue
                n_nan = int(seeded_feats[col].isna().sum())
                if n_nan > 0:
                    nan_teams = seeded_feats[seeded_feats[col].isna()]['TeamID'].tolist()
                    nan_issues.append((col, n_nan, nan_teams))

            if nan_issues:
                print(f"  WARN  {gender_label} ({season}): NaNs found in {len(nan_issues)} feature(s):")
                for col, count, detail in nan_issues:
                    if detail == 'MISSING COLUMN':
                        print(f"          {col}: COLUMN MISSING from features")
                        warnings.append(f"{gender_label} feature '{col}' is missing entirely for {season}")
                    else:
                        print(f"          {col}: {count} seeded team(s) have NaN — TeamIDs: {detail}")
                        warnings.append(f"{gender_label} {count} seeded team(s) have NaN in '{col}' for {season}")
            else:
                print(f"  OK    {gender_label} ({season}): all {len(seeded_feats)} seeded teams have complete features")

            return nan_issues

        check_feature_nans(m_features, O_MEN_FEATURES, m_seeds, "Men", max_season)
        check_feature_nans(w_features, O_WOMEN_FEATURES, parse_seeds(w_data['WTrnySeeds']), "Women", max_season)

    except Exception as e:
        print(f"  WARN  Feature validation skipped: {e}")
        warnings.append(f"Feature NaN validation failed: {e}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")

    if warnings:
        print("  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
        print()

    if not all_ok:
        print("  STATUS: NOT READY — missing required data files")
        print("  ACTION: Download fresh data from Kaggle into ./data/")
        print(f"{'=' * 60}")
        return False, False

    if post_selection:
        print("  STATUS: FULLY READY (post-Selection Sunday)")
        print()
        print("  Available commands:")
        print(f"    {PYTHON} -u scripts/gen_submission_O.py")
        print(f"      -> submissions/stage2_O_loso_tuned.csv")
        print()
        print(f"    {PYTHON} -u scripts/gen_submission_final.py --skip-optimize")
        print(f"      -> submissions/stage2_final_round_alpha.csv")
        print()
        print(f"    {PYTHON} -u scripts/bracket.py --gender both --season 2026 --mode chalk")
        print(f"    {PYTHON} -u scripts/bracket.py --gender men --mode montecarlo --sims 10000")
        print(f"    {PYTHON} -u scripts/bracket.py --gender men --mode underdogs")
    else:
        print("  STATUS: READY (pre-Selection Sunday)")
        print()
        print("  Available now:")
        print(f"    {PYTHON} -u scripts/gen_submission_O.py")
        print(f"      -> submissions/stage2_O_loso_tuned.csv  (predicts all matchups, no seeds needed)")
        print()
        print("  After Selection Sunday (March 15):")
        print(f"    {PYTHON} -u scripts/gen_submission_final.py --skip-optimize")
        print(f"    {PYTHON} -u scripts/bracket.py --gender both --season 2026")

    print(f"\n  Deadline: March 19, 2026 4PM UTC")
    print(f"{'=' * 60}")

    return True, post_selection


def run_pipeline(run_submission=True, run_bracket=True):
    """Run the full pipeline."""
    ok, post_selection = run_preflight()
    if not ok:
        print("\nAborting — fix data issues first.")
        return False

    os.makedirs('./submissions', exist_ok=True)

    if run_submission:
        print(f"\n{'#' * 60}")
        print("  GENERATING SUBMISSIONS")
        print(f"{'#' * 60}\n")

        # Always generate the O submission (works pre and post Selection Sunday)
        print(">>> Running gen_submission_O.py ...")
        ret = subprocess.run(
            [PYTHON, '-u', 'scripts/gen_submission_O.py'],
            cwd='.', timeout=600
        )
        if ret.returncode != 0:
            print("ERROR: gen_submission_O.py failed")
            return False

        # If post-Selection Sunday, also generate round-alpha submission
        if post_selection:
            print("\n>>> Running gen_submission_final.py --skip-optimize ...")
            ret = subprocess.run(
                [PYTHON, '-u', 'scripts/gen_submission_final.py', '--skip-optimize'],
                cwd='.', timeout=600
            )
            if ret.returncode != 0:
                print("ERROR: gen_submission_final.py failed")
                return False

    if run_bracket and post_selection:
        print(f"\n{'#' * 60}")
        print("  GENERATING BRACKETS")
        print(f"{'#' * 60}\n")

        for gender in ['men', 'women']:
            print(f"\n>>> {gender.title()} chalk bracket ...")
            subprocess.run(
                [PYTHON, '-u', 'scripts/bracket.py',
                 '--gender', gender, '--mode', 'chalk', '--season', '2026'],
                cwd='.', timeout=600
            )

        print(f"\n>>> Men's Monte Carlo (10k sims) ...")
        subprocess.run(
            [PYTHON, '-u', 'scripts/bracket.py',
             '--gender', 'men', '--mode', 'montecarlo', '--sims', '10000', '--season', '2026'],
            cwd='.', timeout=600
        )

        print(f"\n>>> Women's Monte Carlo (10k sims) ...")
        subprocess.run(
            [PYTHON, '-u', 'scripts/bracket.py',
             '--gender', 'women', '--mode', 'montecarlo', '--sims', '10000', '--season', '2026'],
            cwd='.', timeout=600
        )

        print(f"\n>>> Men's underdog opportunities ...")
        subprocess.run(
            [PYTHON, '-u', 'scripts/bracket.py',
             '--gender', 'men', '--mode', 'underdogs', '--season', '2026'],
            cwd='.', timeout=600
        )

        print(f"\n>>> Women's underdog opportunities ...")
        subprocess.run(
            [PYTHON, '-u', 'scripts/bracket.py',
             '--gender', 'women', '--mode', 'underdogs', '--season', '2026'],
            cwd='.', timeout=600
        )

    elif run_bracket and not post_selection:
        print("\nSkipping brackets — 2026 seeds not available yet (pre-Selection Sunday)")

    print(f"\n{'#' * 60}")
    print("  DONE")
    print(f"{'#' * 60}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Preflight check and pipeline orchestrator')
    parser.add_argument('--run-all', action='store_true',
                        help='Run full pipeline (submissions + brackets)')
    parser.add_argument('--run-submission', action='store_true',
                        help='Generate submissions only')
    parser.add_argument('--run-bracket', action='store_true',
                        help='Generate brackets only')
    args = parser.parse_args()

    if args.run_all:
        run_pipeline(run_submission=True, run_bracket=True)
    elif args.run_submission:
        run_pipeline(run_submission=True, run_bracket=False)
    elif args.run_bracket:
        run_pipeline(run_submission=False, run_bracket=True)
    else:
        run_preflight()


if __name__ == '__main__':
    main()
