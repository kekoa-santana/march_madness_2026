"""Bracket pick accuracy & Brier decomposition on 2022-2025 holdout.

Answers two questions:
  1. How often does the model pick the right winner? (by year/gender/round/seed matchup)
  2. If every pick were correct, what would the Brier be? (calibration vs pick quality)

Uses the same LOSO pipeline as eval_loso.py with O config + round alphas.

Usage:
    /c/Users/kekoa/anaconda3/python.exe -u scripts/bracket_accuracy.py
"""
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, '.')

from src.data_loader import load_men_data, load_women_data
from src.feature_engineering import (
    parse_seeds, compute_season_stats, compute_massey_features,
    compute_conference_strength, compute_efficiency, build_team_features,
    build_matchup_matrix,
)
from src.model import brier_score
from src.model_elo_v2 import compute_elo_ratings_v2, build_elo_lookup
from src.model_stack_v1 import (
    StackConfig,
    generate_oof_base_preds,
    _fit_sigmoid_calibrator,
    _apply_sigmoid_calibrator,
    select_clip_bounds,
)
from src.model_logreg_v2 import LogRegV2Config
from src.women_rankings_v1 import WomenRankConfig, compute_women_power_ratings, merge_women_rank_features
from src.round_alpha import (
    get_round_from_daynum, stretch_preds, ROUND_NAMES, DAYNUM_TO_ROUND,
    label_historical_rounds,
)
from src.config import (
    MEN_ELO, WOMEN_ELO,
    O_MEN_LR as men_lr_cfg, O_WOMEN_LR as women_lr_cfg,
    O_MEN_FEATURES as men_features_cfg, O_WOMEN_FEATURES as women_features_cfg,
    WOMEN_GLOBAL_ALPHA,
)

PRED_COLS = ["elo_pred", "lr_pred", "xgb_pred", "cb_pred"]
EVAL_SEASONS = [2022, 2023, 2024, 2025]
data_dir = './data'


def loso_with_predictions(matchups, feature_sets, lr_cfg, cfg, eval_seasons, alpha=1.0,
                          round_alphas=None):
    """LOSO evaluation returning per-game predictions (not just aggregate Brier).

    Replicates eval_loso.py logic but captures per-game results.
    """
    oof_df = generate_oof_base_preds(matchups, feature_sets, lr_cfg, cfg)

    all_seasons = sorted(oof_df["Season"].unique())
    all_seasons = [s for s in all_seasons if s in eval_seasons]

    active_cols = [c for c in PRED_COLS if c in oof_df.columns and (~oof_df[c].isna()).any()]

    results = []

    for season in all_seasons:
        train_oof = oof_df[oof_df["Season"] != season].copy()
        test_oof = oof_df[oof_df["Season"] == season].copy()

        if len(train_oof) < 10 or len(test_oof) == 0:
            continue

        # Fit meta-model on all seasons except this one
        X_train = train_oof[active_cols].copy()
        fill_values = X_train.mean(axis=0).to_dict()
        X_train = X_train.fillna(fill_values)
        y_train = train_oof["Target"].values

        meta = LogisticRegression(
            C=cfg.meta_C, max_iter=3000, solver="lbfgs", random_state=cfg.random_state
        )
        meta.fit(X_train.values, y_train)

        # Fit calibrator on train OOF
        train_raw = meta.predict_proba(X_train.values)[:, 1]
        calibrator = None
        if cfg.use_sigmoid_calibration:
            cal = _fit_sigmoid_calibrator(y_train, train_raw)
            if cal is not None:
                train_cal = _apply_sigmoid_calibrator(cal, train_raw)
                raw_bs = brier_score(y_train, train_raw)
                cal_bs = brier_score(y_train, train_cal)
                if raw_bs - cal_bs >= cfg.calibration_min_gain:
                    calibrator = cal

        # Select clip bounds on train OOF
        train_preds = _apply_sigmoid_calibrator(calibrator, train_raw)
        clip_low, clip_high, _ = select_clip_bounds(y_train, train_preds, cfg.clip_candidates)

        # Predict on held-out season
        X_test = test_oof[active_cols].copy().fillna(fill_values)
        test_raw = meta.predict_proba(X_test.values)[:, 1]
        test_preds = _apply_sigmoid_calibrator(calibrator, test_raw)
        test_preds = np.clip(test_preds, clip_low, clip_high)

        # Apply global alpha stretching
        if alpha != 1.0:
            test_preds = stretch_preds(test_preds, alpha)

        for i, (_, row) in enumerate(test_oof.iterrows()):
            results.append({
                'Season': int(row['Season']),
                'TeamA': int(row['TeamA']),
                'TeamB': int(row['TeamB']),
                'Target': int(row['Target']),
                'pred': float(test_preds[i]),
            })

    return pd.DataFrame(results)


def enrich_with_metadata(pred_df, tourney_results, seeds_df, gender):
    """Add round, seed info to prediction DataFrame."""
    # Build lookup: (Season, TeamA, TeamB) -> DayNum from tourney results
    tr = tourney_results.copy()
    tr['TeamA'] = tr[['WTeamID', 'LTeamID']].min(axis=1)
    tr['TeamB'] = tr[['WTeamID', 'LTeamID']].max(axis=1)
    tr_key = tr[['Season', 'TeamA', 'TeamB', 'DayNum']].drop_duplicates()

    df = pred_df.merge(tr_key, on=['Season', 'TeamA', 'TeamB'], how='left')

    # Map DayNum to round
    df['Round'] = df.apply(
        lambda r: get_round_from_daynum(int(r['Season']), int(r['DayNum']))
        if pd.notna(r['DayNum']) else None,
        axis=1
    )

    # Women fallback: use game-count heuristic for rounds not in standard map
    if gender == 'women':
        missing_round = df['Round'].isna() & df['DayNum'].notna()
        if missing_round.any():
            # Count games per (Season, DayNum) to infer round
            for season in df.loc[missing_round, 'Season'].unique():
                season_mask = (df['Season'] == season) & df['DayNum'].notna()
                day_counts = df.loc[season_mask].groupby('DayNum').size()
                # Sort days and assign rounds based on game counts
                sorted_days = sorted(day_counts.index)
                # Group consecutive days and sum their game counts
                round_assignment = {}
                day_groups = []
                current_group = [sorted_days[0]]
                for d in sorted_days[1:]:
                    if d - current_group[-1] <= 1:
                        current_group.append(d)
                    else:
                        day_groups.append(current_group)
                        current_group = [d]
                day_groups.append(current_group)

                # Assign rounds: largest groups first = earliest rounds
                for rnd, group in enumerate(day_groups):
                    for d in group:
                        round_assignment[d] = rnd

                for idx in df.index[missing_round & (df['Season'] == season)]:
                    dn = df.loc[idx, 'DayNum']
                    if dn in round_assignment:
                        df.loc[idx, 'Round'] = round_assignment[dn]

    # Add seed info
    seed_map = {}
    for _, row in seeds_df.iterrows():
        seed_map[(int(row['Season']), int(row['TeamID']))] = row['Seed']

    seed_a_list, seed_b_list, seednum_a_list, seednum_b_list = [], [], [], []
    for _, row in df.iterrows():
        sa = seed_map.get((int(row['Season']), int(row['TeamA'])), None)
        sb = seed_map.get((int(row['Season']), int(row['TeamB'])), None)
        seed_a_list.append(sa)
        seed_b_list.append(sb)
        seednum_a_list.append(int(sa[1:3]) if sa else None)
        seednum_b_list.append(int(sb[1:3]) if sb else None)

    df['SeedA'] = seed_a_list
    df['SeedB'] = seed_b_list
    df['SeedNumA'] = seednum_a_list
    df['SeedNumB'] = seednum_b_list

    return df


def apply_round_alphas_to_preds(df, round_alphas):
    """Apply round-specific alpha stretching to predictions in-place."""
    if round_alphas is None:
        return df
    df = df.copy()
    for rnd, alpha in round_alphas.items():
        if alpha == 1.0:
            continue
        mask = df['Round'] == rnd
        if mask.sum() > 0:
            df.loc[mask, 'pred'] = stretch_preds(df.loc[mask, 'pred'].values, alpha)
    return df


def print_accuracy_table(df, title, groupby_col, label_map=None):
    """Print accuracy table grouped by a column."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Group':<25s} {'Correct':>7s} {'Total':>6s} {'Acc%':>7s} {'Brier':>8s}")
    print(f"  {'-'*55}")

    for val in sorted(df[groupby_col].dropna().unique()):
        sub = df[df[groupby_col] == val]
        correct = ((sub['pred'] > 0.5) & (sub['Target'] == 1)) | \
                  ((sub['pred'] < 0.5) & (sub['Target'] == 0))
        n_correct = correct.sum()
        n_total = len(sub)
        acc = n_correct / n_total * 100 if n_total > 0 else 0
        bs = brier_score(sub['Target'].values, sub['pred'].values)
        label = label_map.get(val, str(val)) if label_map else str(val)
        print(f"  {label:<25s} {n_correct:>7d} {n_total:>6d} {acc:>6.1f}% {bs:>8.5f}")

    # Total row
    correct = ((df['pred'] > 0.5) & (df['Target'] == 1)) | \
              ((df['pred'] < 0.5) & (df['Target'] == 0))
    n_correct = correct.sum()
    n_total = len(df)
    acc = n_correct / n_total * 100 if n_total > 0 else 0
    bs = brier_score(df['Target'].values, df['pred'].values)
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<25s} {n_correct:>7d} {n_total:>6d} {acc:>6.1f}% {bs:>8.5f}")


def print_seed_matchup_accuracy(df):
    """Print accuracy by seed matchup (1v16, 2v15, etc.)."""
    print(f"\n{'='*60}")
    print(f"  Pick Accuracy by Seed Matchup")
    print(f"{'='*60}")
    print(f"  {'Matchup':<15s} {'Correct':>7s} {'Total':>6s} {'Acc%':>7s} {'Upsets':>7s} {'Upset%':>7s}")
    print(f"  {'-'*52}")

    valid = df.dropna(subset=['SeedNumA', 'SeedNumB'])
    valid = valid.copy()
    valid['HighSeed'] = valid[['SeedNumA', 'SeedNumB']].min(axis=1).astype(int)
    valid['LowSeed'] = valid[['SeedNumA', 'SeedNumB']].max(axis=1).astype(int)
    valid['matchup'] = valid['HighSeed'].astype(str) + 'v' + valid['LowSeed'].astype(str)

    # Is lower seed # (better team) TeamA or TeamB?
    # Target=1 means TeamA won. If SeedNumA < SeedNumB, favorite is TeamA.
    valid['fav_is_A'] = valid['SeedNumA'] < valid['SeedNumB']
    valid['fav_won'] = ((valid['fav_is_A']) & (valid['Target'] == 1)) | \
                       ((~valid['fav_is_A']) & (valid['Target'] == 0))
    valid['upset'] = ~valid['fav_won'] & (valid['SeedNumA'] != valid['SeedNumB'])

    for matchup in sorted(valid['matchup'].unique(), key=lambda x: (int(x.split('v')[0]), int(x.split('v')[1]))):
        sub = valid[valid['matchup'] == matchup]
        correct = ((sub['pred'] > 0.5) & (sub['Target'] == 1)) | \
                  ((sub['pred'] < 0.5) & (sub['Target'] == 0))
        n_correct = correct.sum()
        n_total = len(sub)
        acc = n_correct / n_total * 100 if n_total > 0 else 0
        n_upsets = sub['upset'].sum()
        upset_pct = n_upsets / n_total * 100 if n_total > 0 else 0
        print(f"  {matchup:<15s} {n_correct:>7d} {n_total:>6d} {acc:>6.1f}% {n_upsets:>7d} {upset_pct:>6.1f}%")


def print_upset_detection(df):
    """Of actual upsets, how many did the model predict?"""
    print(f"\n{'='*60}")
    print(f"  Upset Detection (higher seed # won)")
    print(f"{'='*60}")

    valid = df.dropna(subset=['SeedNumA', 'SeedNumB']).copy()
    valid['fav_is_A'] = valid['SeedNumA'] < valid['SeedNumB']
    valid['fav_won'] = ((valid['fav_is_A']) & (valid['Target'] == 1)) | \
                       ((~valid['fav_is_A']) & (valid['Target'] == 0))
    valid['upset'] = ~valid['fav_won'] & (valid['SeedNumA'] != valid['SeedNumB'])

    upsets = valid[valid['upset']]
    if len(upsets) == 0:
        print("  No upsets found.")
        return

    # Did model predict the upset? (pred on correct side)
    model_called = ((upsets['pred'] > 0.5) & (upsets['Target'] == 1)) | \
                   ((upsets['pred'] < 0.5) & (upsets['Target'] == 0))
    n_called = model_called.sum()
    n_upsets = len(upsets)
    print(f"  Total upsets:          {n_upsets}")
    print(f"  Model called upset:    {n_called} ({n_called/n_upsets*100:.1f}%)")
    print(f"  Model missed upset:    {n_upsets - n_called} ({(n_upsets-n_called)/n_upsets*100:.1f}%)")

    # By round
    print(f"\n  {'Round':<25s} {'Upsets':>7s} {'Called':>7s} {'Rate':>7s}")
    print(f"  {'-'*48}")
    for rnd in sorted(upsets['Round'].dropna().unique()):
        sub = upsets[upsets['Round'] == rnd]
        called = ((sub['pred'] > 0.5) & (sub['Target'] == 1)) | \
                 ((sub['pred'] < 0.5) & (sub['Target'] == 0))
        n = len(sub)
        nc = called.sum()
        name = ROUND_NAMES.get(int(rnd), f'Round {int(rnd)}')
        print(f"  {name:<25s} {n:>7d} {nc:>7d} {nc/n*100:>6.1f}%")


def print_brier_decomposition(df, label=""):
    """Decompose Brier into confidence component vs wrong-pick penalty."""
    print(f"\n{'='*60}")
    print(f"  Brier Decomposition{' — ' + label if label else ''}")
    print(f"{'='*60}")

    targets = df['Target'].values
    preds = df['pred'].values

    actual_brier = np.mean((targets - preds) ** 2)
    confidence = np.maximum(preds, 1 - preds)
    # If every pick were correct, Brier = mean((1 - confidence)^2)
    perfect_picks_brier = np.mean((1 - confidence) ** 2)
    wrong_pick_penalty = actual_brier - perfect_picks_brier

    print(f"  Actual Brier:           {actual_brier:.5f}")
    print(f"  Perfect-picks Brier:    {perfect_picks_brier:.5f}  (if all picks correct)")
    print(f"  Wrong-pick penalty:     {wrong_pick_penalty:.5f}")
    print(f"  % from under-confidence: {perfect_picks_brier/actual_brier*100:.1f}%")
    print(f"  % from wrong picks:      {wrong_pick_penalty/actual_brier*100:.1f}%")

    # By round
    valid = df.dropna(subset=['Round']).copy()
    if len(valid) > 0:
        print(f"\n  {'Round':<25s} {'Brier':>8s} {'Perfect':>8s} {'Penalty':>8s} {'%Conf':>6s} {'%Wrong':>6s} {'N':>5s}")
        print(f"  {'-'*70}")
        for rnd in sorted(valid['Round'].unique()):
            sub = valid[valid['Round'] == rnd]
            t = sub['Target'].values
            p = sub['pred'].values
            bs = np.mean((t - p) ** 2)
            conf = np.maximum(p, 1 - p)
            pp = np.mean((1 - conf) ** 2)
            wp = bs - pp
            name = ROUND_NAMES.get(int(rnd), f'Round {int(rnd)}')
            print(f"  {name:<25s} {bs:>8.5f} {pp:>8.5f} {wp:>8.5f} {pp/bs*100:>5.1f}% {wp/bs*100:>5.1f}% {len(sub):>5d}")


def main():
    # ============================================================
    # Step 1: Load data & build features (same as gen_submission_final.py)
    # ============================================================
    print("Loading data...")
    m_data = load_men_data(data_dir)
    w_data = load_women_data(data_dir)

    print("Building features...")
    m_elo = compute_elo_ratings_v2(m_data['MComSsn'], MEN_ELO)
    m_features = build_team_features(
        m_elo, parse_seeds(m_data['MTrnySeeds']),
        compute_season_stats(m_data['MDetSsn']),
        compute_massey_features(m_data['MOrdinals']),
        compute_conference_strength(m_data['MConf'], m_elo),
        efficiency_df=compute_efficiency(m_data['MDetSsn']),
    )
    m_matchups = build_matchup_matrix(m_data['MDetTrny'], m_features)

    w_elo = compute_elo_ratings_v2(w_data['WComSsn'], WOMEN_ELO)
    w_base = build_team_features(
        w_elo, parse_seeds(w_data['WTrnySeeds']),
        stats_df=compute_season_stats(w_data['WDetSsn']),
        conf_df=compute_conference_strength(w_data['WConf'], w_elo),
        efficiency_df=compute_efficiency(w_data['WDetSsn']),
    )
    wpr = compute_women_power_ratings(w_data['WComSsn'], WomenRankConfig())
    w_features = merge_women_rank_features(w_base, wpr)
    w_matchups = build_matchup_matrix(w_data['WDetTrny'], w_features)

    # ============================================================
    # Step 2: Run LOSO with per-game predictions
    # ============================================================
    stack_cfg = StackConfig(train_cutoff=2022, oof_start_season=2010, oof_end_season=2025)

    print("\nRunning LOSO for men...")
    m_preds = loso_with_predictions(
        m_matchups, men_features_cfg, men_lr_cfg, stack_cfg, EVAL_SEASONS, alpha=1.0,
    )
    print(f"  Men: {len(m_preds)} games")

    print("Running LOSO for women...")
    w_preds = loso_with_predictions(
        w_matchups, women_features_cfg, women_lr_cfg, stack_cfg, EVAL_SEASONS,
        alpha=WOMEN_GLOBAL_ALPHA,
    )
    print(f"  Women: {len(w_preds)} games")

    # ============================================================
    # Step 3: Enrich with metadata (round, seeds)
    # ============================================================
    print("\nEnriching with round/seed metadata...")
    m_seeds = pd.read_csv(f'{data_dir}/MNCAATourneySeeds.csv')
    w_seeds = pd.read_csv(f'{data_dir}/WNCAATourneySeeds.csv')
    m_tourney = pd.read_csv(f'{data_dir}/MNCAATourneyCompactResults.csv')
    w_tourney = pd.read_csv(f'{data_dir}/WNCAATourneyCompactResults.csv')

    m_enriched = enrich_with_metadata(m_preds, m_tourney, m_seeds, gender='men')
    w_enriched = enrich_with_metadata(w_preds, w_tourney, w_seeds, gender='women')

    # Apply round-specific alphas
    try:
        with open('./round_alphas.json') as f:
            cached = json.load(f)
        men_round_alphas = {int(k): v for k, v in cached['men'].items()}
        women_round_alphas = {int(k): v for k, v in cached['women'].items()}
        print("Applying round-specific alphas from round_alphas.json...")
        m_enriched = apply_round_alphas_to_preds(m_enriched, men_round_alphas)
        w_enriched = apply_round_alphas_to_preds(w_enriched, women_round_alphas)
    except FileNotFoundError:
        print("No round_alphas.json found, skipping round-specific stretching.")

    # Tag gender and combine
    m_enriched['Gender'] = 'Men'
    w_enriched['Gender'] = 'Women'
    all_df = pd.concat([m_enriched, w_enriched], ignore_index=True)

    # ============================================================
    # Step 4: Print analyses
    # ============================================================

    # --- Sanity check: pooled Brier ---
    pooled = brier_score(all_df['Target'].values, all_df['pred'].values)
    m_pooled = brier_score(m_enriched['Target'].values, m_enriched['pred'].values)
    w_pooled = brier_score(w_enriched['Target'].values, w_enriched['pred'].values)
    print(f"\n  Pooled Brier (combined): {pooled:.5f}")
    print(f"  Pooled Brier (men):      {m_pooled:.5f}  (n={len(m_enriched)})")
    print(f"  Pooled Brier (women):    {w_pooled:.5f}  (n={len(w_enriched)})")

    # --- Analysis 1: Pick Accuracy ---
    print_accuracy_table(all_df, "Pick Accuracy by Season", 'Season')
    print_accuracy_table(all_df, "Pick Accuracy by Gender", 'Gender')

    has_round = all_df.dropna(subset=['Round'])
    has_round = has_round.copy()
    has_round['Round'] = has_round['Round'].astype(int)
    print_accuracy_table(has_round, "Pick Accuracy by Round", 'Round', label_map=ROUND_NAMES)

    print_seed_matchup_accuracy(all_df)
    print_upset_detection(all_df)

    # --- Analysis 2: Brier Decomposition ---
    print_brier_decomposition(all_df, "Combined")
    print_brier_decomposition(m_enriched, "Men")
    print_brier_decomposition(w_enriched, "Women")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
