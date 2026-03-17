"""Evaluate the current best config (J) using full LOSO evaluation.

This establishes the LOSO baseline scores for both men and women,
which we can then use to reliably compare future experiments.
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')

from src.data_loader import load_men_data, load_women_data
from src.feature_engineering import (
    parse_seeds, compute_season_stats, compute_massey_features,
    compute_conference_strength, compute_efficiency, build_team_features,
    build_matchup_matrix,
)
from src.model import brier_score
from src.model_elo_v2 import compute_elo_ratings_v2
from src.model_stack_v1 import StackConfig
from src.women_rankings_v1 import WomenRankConfig, compute_women_power_ratings, merge_women_rank_features
from src.eval_loso import loso_evaluate
from src.config import (
    MEN_ELO, WOMEN_ELO,
    J_MEN_LR as men_lr_cfg, J_WOMEN_LR as women_lr_cfg,
    J_MEN_FEATURES as men_features_cfg, J_WOMEN_FEATURES as women_features_cfg,
    WOMEN_GLOBAL_ALPHA as WOMEN_ALPHA,
)

data_dir = './data'

# ============================================================
# Load and build features
# ============================================================
print("Loading data...")
t0 = time.time()
m_data = load_men_data(data_dir)
w_data = load_women_data(data_dir)

print("Building men's features...")
m_elo = compute_elo_ratings_v2(m_data['MComSsn'], MEN_ELO)
m_features = build_team_features(
    m_elo, parse_seeds(m_data['MTrnySeeds']),
    compute_season_stats(m_data['MDetSsn']),
    compute_massey_features(m_data['MOrdinals']),
    compute_conference_strength(m_data['MConf'], m_elo),
    efficiency_df=compute_efficiency(m_data['MDetSsn']),
)
m_matchups = build_matchup_matrix(m_data['MDetTrny'], m_features)

print("Building women's features...")
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

print(f"Data loaded in {time.time()-t0:.1f}s")
print(f"Men matchups: {len(m_matchups)} rows, seasons {m_matchups['Season'].min()}-{m_matchups['Season'].max()}")
print(f"Women matchups: {len(w_matchups)} rows, seasons {w_matchups['Season'].min()}-{w_matchups['Season'].max()}")

# ============================================================
# Men's LOSO (no alpha stretching)
# ============================================================
print(f"\n{'='*60}")
print("MEN'S LOSO EVALUATION (J config, no alpha)")
print(f"{'='*60}")

men_cfg = StackConfig(oof_start_season=2003, oof_end_season=2025)
t1 = time.time()
m_result = loso_evaluate(
    m_matchups, men_features_cfg, men_lr_cfg, men_cfg,
    alpha=1.0,
)
print(f"  Time: {time.time()-t1:.1f}s")

# ============================================================
# Women's LOSO (with and without alpha)
# ============================================================
print(f"\n{'='*60}")
print("WOMEN'S LOSO EVALUATION (J config, no alpha)")
print(f"{'='*60}")

women_cfg = StackConfig(oof_start_season=2010, oof_end_season=2025)
t2 = time.time()
w_result_raw = loso_evaluate(
    w_matchups, women_features_cfg, women_lr_cfg, women_cfg,
    alpha=1.0,
)
print(f"  Time: {time.time()-t2:.1f}s")

print(f"\n{'='*60}")
print(f"WOMEN'S LOSO EVALUATION (J config, alpha={WOMEN_ALPHA})")
print(f"{'='*60}")

t3 = time.time()
w_result_alpha = loso_evaluate(
    w_matchups, women_features_cfg, women_lr_cfg, women_cfg,
    alpha=WOMEN_ALPHA,
)
print(f"  Time: {time.time()-t3:.1f}s")

# ============================================================
# Compare to 4-year holdout
# ============================================================
print(f"\n{'='*60}")
print("COMPARISON: LOSO vs 4-year holdout")
print(f"{'='*60}")

# Extract 2022-2025 subset from LOSO
holdout_seasons = [2022, 2023, 2024, 2025]

m_ho_seasons = [s for s in m_result["per_season"] if s["season"] in holdout_seasons]
w_ho_seasons_raw = [s for s in w_result_raw["per_season"] if s["season"] in holdout_seasons]
w_ho_seasons_alpha = [s for s in w_result_alpha["per_season"] if s["season"] in holdout_seasons]

if m_ho_seasons:
    m_ho_briers = [s["brier"] for s in m_ho_seasons]
    m_ho_n = [s["n_games"] for s in m_ho_seasons]
    m_ho_pooled = sum(b*n for b,n in zip(m_ho_briers, m_ho_n)) / sum(m_ho_n)
    print(f"\n  Men 2022-2025 (LOSO meta):  pooled={m_ho_pooled:.5f}  mean={np.mean(m_ho_briers):.5f}")
    print(f"  Men 2022-2025 (J holdout):  pooled=0.19186  (from J script)")

if w_ho_seasons_raw:
    w_ho_briers_raw = [s["brier"] for s in w_ho_seasons_raw]
    w_ho_n = [s["n_games"] for s in w_ho_seasons_raw]
    w_ho_pooled_raw = sum(b*n for b,n in zip(w_ho_briers_raw, w_ho_n)) / sum(w_ho_n)
    print(f"\n  Women 2022-2025 raw (LOSO): pooled={w_ho_pooled_raw:.5f}  mean={np.mean(w_ho_briers_raw):.5f}")

if w_ho_seasons_alpha:
    w_ho_briers_a = [s["brier"] for s in w_ho_seasons_alpha]
    w_ho_n_a = [s["n_games"] for s in w_ho_seasons_alpha]
    w_ho_pooled_a = sum(b*n for b,n in zip(w_ho_briers_a, w_ho_n_a)) / sum(w_ho_n_a)
    print(f"  Women 2022-2025 alpha (LOSO): pooled={w_ho_pooled_a:.5f}  mean={np.mean(w_ho_briers_a):.5f}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("LOSO BASELINE SUMMARY (J config)")
print(f"{'='*60}")
print(f"  Men   LOSO mean:    {m_result['mean_brier']:.5f} +/- {m_result['std_brier']:.5f}  ({m_result['n_seasons']} seasons)")
print(f"  Women LOSO mean:    {w_result_raw['mean_brier']:.5f} +/- {w_result_raw['std_brier']:.5f}  ({w_result_raw['n_seasons']} seasons)")
print(f"  Women LOSO (a=1.2): {w_result_alpha['mean_brier']:.5f} +/- {w_result_alpha['std_brier']:.5f}  ({w_result_alpha['n_seasons']} seasons)")
print(f"  Alpha improvement:  {w_result_alpha['mean_brier'] - w_result_raw['mean_brier']:+.5f}")
print(f"\n  Total time: {time.time()-t0:.1f}s")
