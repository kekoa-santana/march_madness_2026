"""Generate submission O: LOSO-validated improvements over J.

Changes from J:
- Men: lean XGB (5 core features, same as CB) — LOSO -0.00053
- Women: C=0.5 (was 0.2) — LOSO -0.00013
- Women: full CB (match XGB 10 features) — LOSO -0.00026
- Combined women's changes — LOSO -0.00039
- Women's alpha=1.2 stretching kept for Kaggle submission
"""
import sys
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
from src.config import (
    MEN_ELO, WOMEN_ELO,
    O_MEN_LR as men_lr_cfg, O_WOMEN_LR as women_lr_cfg,
    O_MEN_FEATURES as men_features_cfg, O_WOMEN_FEATURES as women_features_cfg,
    WOMEN_GLOBAL_ALPHA as WOMEN_ALPHA,
)

data_dir = './data'
print("Loading data...")
m_data = load_men_data(data_dir)
w_data = load_women_data(data_dir)

def stretch_preds(preds, alpha):
    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    logit = np.log(preds / (1 - preds))
    stretched = 1.0 / (1.0 + np.exp(-alpha * logit))
    return np.clip(stretched, 0.001, 0.999)


# Build features
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

holdout_cutoff = 2022
holdout_max = holdout_cutoff - 1

# ============================================================
# Stage 1: Holdout evaluation
# ============================================================
print("\n--- Stage 1: Holdout evaluation (train <= 2021) ---")
holdout_cfg = StackConfig(train_cutoff=holdout_cutoff, oof_start_season=2010, oof_end_season=holdout_max)

m_art_ho = train_stack_final(m_matchups, men_features_cfg, men_lr_cfg, holdout_cfg, max_train_season=holdout_max)
w_art_ho = train_stack_final(w_matchups, women_features_cfg, women_lr_cfg, holdout_cfg, max_train_season=holdout_max)

# Men holdout
m_test = m_matchups[m_matchups['Season'] >= holdout_cutoff].copy()
m_preds = predict_stack_from_matchups(m_test, m_art_ho, clip=True)
m_ho = brier_score(m_test['Target'].values, m_preds)

# Women holdout
w_test = w_matchups[w_matchups['Season'] >= holdout_cutoff].copy()
w_preds_raw = predict_stack_from_matchups(w_test, w_art_ho, clip=True)
w_ho_raw = brier_score(w_test['Target'].values, w_preds_raw)

w_preds_stretched = stretch_preds(w_preds_raw, WOMEN_ALPHA)
w_ho_stretched = brier_score(w_test['Target'].values, w_preds_stretched)

combined_raw = (m_ho * len(m_test) + w_ho_raw * len(w_test)) / (len(m_test) + len(w_test))
combined_stretched = (m_ho * len(m_test) + w_ho_stretched * len(w_test)) / (len(m_test) + len(w_test))

print(f"\n  Men holdout:   {m_ho:.5f}")
print(f"  Women holdout: {w_ho_raw:.5f} -> {w_ho_stretched:.5f} (alpha={WOMEN_ALPHA})")
print(f"  Combined:      {combined_raw:.5f} -> {combined_stretched:.5f}")

print(f"\n  J baseline:    Men=0.19186  Women=0.13666->0.13593  Combined=0.16339")
print(f"  O this run:    Men={m_ho:.5f}  Women={w_ho_raw:.5f}->{w_ho_stretched:.5f}  Combined={combined_stretched:.5f}")
print(f"  Delta:         Men={m_ho-0.19186:+.5f}  Women={w_ho_stretched-0.13593:+.5f}  Combined={combined_stretched-0.16339:+.5f}")

# ============================================================
# Stage 2: Generate submission (retrain on ALL data)
# ============================================================
print("\n--- Stage 2: Generate submission (train on all data) ---")
final_cfg = StackConfig(train_cutoff=holdout_cutoff, oof_start_season=2010, oof_end_season=2025)
m_art = train_stack_final(m_matchups, men_features_cfg, men_lr_cfg, final_cfg, max_train_season=2025)
w_art = train_stack_final(w_matchups, women_features_cfg, women_lr_cfg, final_cfg, max_train_season=2025)

elo_lookup = build_elo_lookup(pd.concat([m_elo, w_elo], ignore_index=True))

sub = generate_submission_stacked(
    f'{data_dir}/SampleSubmissionStage2.csv',
    m_art, w_art, m_features, w_features, elo_lookup,
)

# Apply alpha stretching to women's predictions only
parts = sub['ID'].str.split('_', expand=True).astype(int)
is_women = parts[1] >= 3000
n_women = is_women.sum()
n_men = (~is_women).sum()

print(f"\nApplying alpha={WOMEN_ALPHA} to {n_women} women's predictions ({n_men} men unchanged)")

w_before = sub.loc[is_women, 'Pred'].copy()
sub.loc[is_women, 'Pred'] = stretch_preds(sub.loc[is_women, 'Pred'].values, WOMEN_ALPHA)
w_after = sub.loc[is_women, 'Pred']

print(f"  Women pred range: [{w_before.min():.4f}, {w_before.max():.4f}] -> [{w_after.min():.4f}, {w_after.max():.4f}]")
print(f"  Women pred std:   {w_before.std():.4f} -> {w_after.std():.4f}")

out_path = './submissions/stage2_O_loso_tuned.csv'
sub.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"SUMMARY: stage1_O_loso_tuned")
print(f"{'='*60}")
print(f"  Changes from J:")
print(f"    Men:   lean XGB (5 features, was 14)")
print(f"    Women: C=0.5 (was 0.2), full CB (10 features, was 5)")
print(f"    Women: alpha=1.2 stretching (kept from J)")
print(f"  LOSO improvements:")
print(f"    Men:   -0.00053")
print(f"    Women: -0.00039")
print(f"  Holdout (2022-2025):")
print(f"    Men={m_ho:.5f}  Women={w_ho_stretched:.5f}  Combined={combined_stretched:.5f}")
print(f"{'='*60}")
