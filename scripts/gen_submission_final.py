"""Generate final tournament submission with round-specific alpha stretching.

Run AFTER Selection Sunday (March 15, 2026) once seeds are known.

Steps:
  1. Optimize round-specific alphas on historical data (LOSO)
  2. Train stacking pipeline on all data
  3. Generate raw predictions for all matchups
  4. Apply round-specific alphas based on bracket structure
  5. Save submission

Usage:
    python scripts/gen_submission_final.py
    python scripts/gen_submission_final.py --skip-optimize   # use cached alphas
"""
import sys
import json
import argparse

import numpy as np
import pandas as pd

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
    StackConfig, train_stack_final, predict_stack_from_matchups,
)
from src.submission_stack_v1 import generate_submission_stacked
from src.women_rankings_v1 import WomenRankConfig, compute_women_power_ratings, merge_women_rank_features
from src.round_alpha import (
    optimize_round_alphas, apply_round_alphas, stretch_preds,
    print_round_alphas, ROUND_NAMES,
)
from src.config import (
    MEN_ELO, WOMEN_ELO,
    O_MEN_LR as men_lr_cfg, O_WOMEN_LR as women_lr_cfg,
    O_MEN_FEATURES as men_features_cfg, O_WOMEN_FEATURES as women_features_cfg,
    WOMEN_GLOBAL_ALPHA,
)

data_dir = './data'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-optimize', action='store_true',
                        help='Skip alpha optimization, use cached values')
    parser.add_argument('--alphas-file', type=str, default='./round_alphas.json',
                        help='Path to cached round alphas JSON')
    args = parser.parse_args()

    # ============================================================
    # Load data and build features
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
    # Step 1: Optimize round-specific alphas (or load cached)
    # ============================================================
    if args.skip_optimize and args.alphas_file:
        print(f"\nLoading cached alphas from {args.alphas_file}...")
        with open(args.alphas_file) as f:
            cached = json.load(f)
        men_round_alphas = {int(k): v for k, v in cached['men'].items()}
        women_round_alphas = {int(k): v for k, v in cached['women'].items()}
    else:
        print("\n--- Optimizing round-specific alphas (LOSO) ---")

        stack_cfg = StackConfig(train_cutoff=2022, oof_start_season=2010, oof_end_season=2025)

        print("\nMen's round alphas:")
        men_round_alphas, men_results = optimize_round_alphas(
            m_matchups, None, men_features_cfg, men_lr_cfg, stack_cfg,
            m_data['MDetTrny'], gender='men',
        )
        print_round_alphas(men_round_alphas, men_results)

        print("\nWomen's round alphas:")
        women_round_alphas, women_results = optimize_round_alphas(
            w_matchups, None, women_features_cfg, women_lr_cfg, stack_cfg,
            w_data['WDetTrny'], gender='women',
        )
        print_round_alphas(women_round_alphas, women_results)

        # Save for re-use
        alphas_cache = {
            'men': {str(k): v for k, v in men_round_alphas.items()},
            'women': {str(k): v for k, v in women_round_alphas.items()},
        }
        with open(args.alphas_file, 'w') as f:
            json.dump(alphas_cache, f, indent=2)
        print(f"\nSaved alphas to {args.alphas_file}")

    print(f"\nMen's round alphas:   {men_round_alphas}")
    print(f"Women's round alphas: {women_round_alphas}")

    # ============================================================
    # Step 2: Train final models on all data
    # ============================================================
    print("\n--- Training final models (all data) ---")
    final_cfg = StackConfig(train_cutoff=2022, oof_start_season=2010, oof_end_season=2025)
    m_art = train_stack_final(m_matchups, men_features_cfg, men_lr_cfg, final_cfg, max_train_season=2025)
    w_art = train_stack_final(w_matchups, women_features_cfg, women_lr_cfg, final_cfg, max_train_season=2025)

    elo_lookup = build_elo_lookup(pd.concat([m_elo, w_elo], ignore_index=True))

    # ============================================================
    # Step 3: Generate raw submission
    # ============================================================
    print("\n--- Generating raw predictions ---")
    sub = generate_submission_stacked(
        f'{data_dir}/SampleSubmissionStage2.csv',
        m_art, w_art, m_features, w_features, elo_lookup,
    )

    parts = sub['ID'].str.split('_', expand=True).astype(int)
    is_women = parts[1] >= 3000
    n_men = (~is_women).sum()
    n_women = is_women.sum()
    print(f"  Raw predictions: {n_men} men, {n_women} women")

    # ============================================================
    # Step 4: Apply round-specific alphas
    # ============================================================
    print("\n--- Applying round-specific alpha stretching ---")

    # Load bracket data
    m_seeds = pd.read_csv(f'{data_dir}/MNCAATourneySeeds.csv')
    w_seeds = pd.read_csv(f'{data_dir}/WNCAATourneySeeds.csv')
    m_seed_slots = pd.read_csv(f'{data_dir}/MNCAATourneySeedRoundSlots.csv')

    # Split submission into men/women
    sub_men = sub[~is_women].copy()
    sub_women = sub[is_women].copy()

    # Men: apply round-specific alphas
    sub_men = apply_round_alphas(
        sub_men, m_seeds, m_seed_slots,
        round_alphas=men_round_alphas,
        default_alpha=1.0,  # No global stretch for men
    )

    # Women: apply round-specific alphas, with global alpha as fallback
    # For women, we use the seed round slots heuristic where possible,
    # but women's bracket structure varies more. Apply global alpha to
    # non-tournament matchups.
    # Note: women don't have a SeedRoundSlots file, so we use the men's
    # bracket structure as an approximation (same 64-team structure)
    sub_women = apply_round_alphas(
        sub_women, w_seeds, m_seed_slots,
        round_alphas=women_round_alphas,
        default_alpha=WOMEN_GLOBAL_ALPHA,
    )

    # Recombine
    sub_final = pd.concat([sub_men, sub_women], ignore_index=True)
    sub_final = sub_final.sort_values('ID').reset_index(drop=True)

    # Stats
    print(f"\n  Men pred range:   [{sub_men['Pred'].min():.4f}, {sub_men['Pred'].max():.4f}]")
    print(f"  Women pred range: [{sub_women['Pred'].min():.4f}, {sub_women['Pred'].max():.4f}]")
    print(f"  Men pred std:     {sub_men['Pred'].std():.4f}")
    print(f"  Women pred std:   {sub_women['Pred'].std():.4f}")

    # ============================================================
    # Step 5: Save
    # ============================================================
    out_path = './submissions/stage2_final_round_alpha.csv'
    sub_final.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Also save a version without round alphas for comparison
    # (just global women's alpha, same as O config)
    sub_baseline = sub.copy()
    sub_baseline.loc[is_women, 'Pred'] = stretch_preds(
        sub_baseline.loc[is_women, 'Pred'].values, WOMEN_GLOBAL_ALPHA
    )
    baseline_path = './submissions/stage2_O_baseline.csv'
    sub_baseline.to_csv(baseline_path, index=False)
    print(f"Saved baseline (O config) to {baseline_path}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("FINAL SUBMISSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Base config: O (LOSO-validated)")
    print(f"  Men's round alphas:")
    for rnd in sorted(men_round_alphas.keys()):
        print(f"    {ROUND_NAMES.get(rnd, f'R{rnd}')}: alpha={men_round_alphas[rnd]}")
    print(f"  Women's round alphas:")
    for rnd in sorted(women_round_alphas.keys()):
        print(f"    {ROUND_NAMES.get(rnd, f'R{rnd}')}: alpha={women_round_alphas[rnd]}")
    print(f"  Women's non-tournament alpha: {WOMEN_GLOBAL_ALPHA}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
