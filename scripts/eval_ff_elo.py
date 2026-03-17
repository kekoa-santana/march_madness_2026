"""Compare Four Factors Elo vs standard v2 Elo in the full stacking pipeline.

Test 1: Replace v2 Elo with FF Elo as the base rating
Test 2: Keep v2 Elo but add FF Elo as an additional feature

Uses proper holdout: train <=2021, eval 2022-2025.
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
from src.elo_four_factors import FFEloConfig, compute_ff_elo
from src.model_stack_v1 import StackConfig, evaluate_stack_holdout, train_stack_final
from src.women_rankings_v1 import WomenRankConfig, compute_women_power_ratings, merge_women_rank_features
from src.config import (
    MEN_ELO, WOMEN_ELO,
    J_MEN_LR as men_lr_cfg, J_WOMEN_LR as women_lr_cfg,
    J_MEN_FEATURES as men_features_cfg, J_WOMEN_FEATURES as women_features_cfg,
)

data_dir = './data'

stack_cfg = StackConfig(train_cutoff=2022, oof_start_season=2003, oof_end_season=2021)
women_stack_cfg = StackConfig(train_cutoff=2022, oof_start_season=2010, oof_end_season=2021)


def evaluate_config(matchups, feature_sets, lr_cfg, cfg, label):
    """Train stack on <=2021, evaluate on 2022-2025."""
    artifacts = train_stack_final(matchups, feature_sets, lr_cfg, cfg, max_train_season=2021)
    result = evaluate_stack_holdout(matchups, artifacts, train_cutoff=2022)
    print(f"  {label}: holdout Brier = {result['holdout_brier']:.5f}  (n={result['n_holdout']})")
    return result['holdout_brier']


# ============================================================
# Load data
# ============================================================
print("Loading data...")
t0 = time.time()
m_data = load_men_data(data_dir)
w_data = load_women_data(data_dir)
print(f"  Loaded in {time.time()-t0:.1f}s")

# ============================================================
# Baseline: v2 Elo (current best)
# ============================================================
print("\n" + "="*60)
print("BASELINE: v2 Elo")
print("="*60)

# Men baseline
m_elo_v2 = compute_elo_ratings_v2(m_data['MComSsn'], MEN_ELO)
m_features_v2 = build_team_features(
    m_elo_v2, parse_seeds(m_data['MTrnySeeds']),
    compute_season_stats(m_data['MDetSsn']),
    compute_massey_features(m_data['MOrdinals']),
    compute_conference_strength(m_data['MConf'], m_elo_v2),
    efficiency_df=compute_efficiency(m_data['MDetSsn']),
)
m_matchups_v2 = build_matchup_matrix(m_data['MDetTrny'], m_features_v2)
m_baseline = evaluate_config(m_matchups_v2, men_features_cfg, men_lr_cfg, stack_cfg, "Men v2 Elo")

# Women baseline
w_elo_v2 = compute_elo_ratings_v2(w_data['WComSsn'], WOMEN_ELO)
w_base_v2 = build_team_features(
    w_elo_v2, parse_seeds(w_data['WTrnySeeds']),
    stats_df=compute_season_stats(w_data['WDetSsn']),
    conf_df=compute_conference_strength(w_data['WConf'], w_elo_v2),
    efficiency_df=compute_efficiency(w_data['WDetSsn']),
)
wpr = compute_women_power_ratings(w_data['WComSsn'], WomenRankConfig())
w_features_v2 = merge_women_rank_features(w_base_v2, wpr)
w_matchups_v2 = build_matchup_matrix(w_data['WDetTrny'], w_features_v2)
w_baseline = evaluate_config(w_matchups_v2, women_features_cfg, women_lr_cfg, women_stack_cfg, "Women v2 Elo")

# ============================================================
# Test 1: Replace v2 Elo with FF Elo
# ============================================================
print("\n" + "="*60)
print("TEST 1: Replace v2 Elo with Four Factors Elo")
print("="*60)

# FF Elo needs detailed results (box scores)
# Men: detailed results available from 2003
m_ff_elo = compute_ff_elo(m_data['MDetSsn'], FFEloConfig(k=16, home_adv=40, carryover=0.90))
# Rename FF_Elo -> Elo so it plugs into the existing pipeline
m_ff_elo = m_ff_elo.rename(columns={'FF_Elo': 'Elo'})
# Drop FF detail columns - we only want the Elo replacement
ff_detail_cols = [c for c in m_ff_elo.columns if c.startswith('FF_') and c != 'FF_Elo']
m_ff_elo = m_ff_elo.drop(columns=ff_detail_cols, errors='ignore')

m_features_ff = build_team_features(
    m_ff_elo, parse_seeds(m_data['MTrnySeeds']),
    compute_season_stats(m_data['MDetSsn']),
    compute_massey_features(m_data['MOrdinals']),
    compute_conference_strength(m_data['MConf'], m_ff_elo),
    efficiency_df=compute_efficiency(m_data['MDetSsn']),
)
m_matchups_ff = build_matchup_matrix(m_data['MDetTrny'], m_features_ff)
m_ff_score = evaluate_config(m_matchups_ff, men_features_cfg, men_lr_cfg, stack_cfg, "Men FF Elo (replace)")

# Women: detailed results from 2010
w_ff_elo = compute_ff_elo(w_data['WDetSsn'], FFEloConfig(k=16, home_adv=40, carryover=0.90))
w_ff_elo = w_ff_elo.rename(columns={'FF_Elo': 'Elo'})
ff_detail_cols = [c for c in w_ff_elo.columns if c.startswith('FF_')]
w_ff_elo = w_ff_elo.drop(columns=ff_detail_cols, errors='ignore')

w_base_ff = build_team_features(
    w_ff_elo, parse_seeds(w_data['WTrnySeeds']),
    stats_df=compute_season_stats(w_data['WDetSsn']),
    conf_df=compute_conference_strength(w_data['WConf'], w_ff_elo),
    efficiency_df=compute_efficiency(w_data['WDetSsn']),
)
w_features_ff = merge_women_rank_features(w_base_ff, wpr)
w_matchups_ff = build_matchup_matrix(w_data['WDetTrny'], w_features_ff)
w_ff_score = evaluate_config(w_matchups_ff, women_features_cfg, women_lr_cfg, women_stack_cfg, "Women FF Elo (replace)")

# ============================================================
# Test 2: Add FF Elo as extra feature alongside v2 Elo
# ============================================================
print("\n" + "="*60)
print("TEST 2: v2 Elo + FF Elo as additional feature")
print("="*60)

# Men: merge FF_Elo alongside existing Elo
m_ff_elo_extra = compute_ff_elo(m_data['MDetSsn'], FFEloConfig(k=16, home_adv=40, carryover=0.90))
m_ff_elo_extra = m_ff_elo_extra[['Season', 'TeamID', 'FF_Elo']].copy()

m_features_both = m_features_v2.merge(m_ff_elo_extra, on=['Season', 'TeamID'], how='left')
m_matchups_both = build_matchup_matrix(m_data['MDetTrny'], m_features_both)

# Add FF_Elo_diff to feature sets
men_features_both = {
    'xgb_features': men_features_cfg['xgb_features'] + ['FF_Elo_diff'],
    'cb_features': men_features_cfg['cb_features'] + ['FF_Elo_diff'],
}
m_both_score = evaluate_config(m_matchups_both, men_features_both, men_lr_cfg, stack_cfg, "Men v2+FF Elo")

# Women
w_ff_elo_extra = compute_ff_elo(w_data['WDetSsn'], FFEloConfig(k=16, home_adv=40, carryover=0.90))
w_ff_elo_extra = w_ff_elo_extra[['Season', 'TeamID', 'FF_Elo']].copy()

w_features_both = w_features_v2.merge(w_ff_elo_extra, on=['Season', 'TeamID'], how='left')
w_matchups_both = build_matchup_matrix(w_data['WDetTrny'], w_features_both)

women_features_both = {
    'xgb_features': women_features_cfg['xgb_features'] + ['FF_Elo_diff'],
    'cb_features': women_features_cfg['cb_features'] + ['FF_Elo_diff'],
}
w_both_score = evaluate_config(w_matchups_both, women_features_both, women_lr_cfg, women_stack_cfg, "Women v2+FF Elo")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY: Holdout Brier Scores (train<=2021, eval 2022-2025)")
print("="*60)
print(f"                          {'Men':>10s}  {'Women':>10s}")
print(f"  Baseline (v2 Elo):      {m_baseline:10.5f}  {w_baseline:10.5f}")
print(f"  FF Elo (replace):       {m_ff_score:10.5f}  {w_ff_score:10.5f}")
print(f"  v2 + FF Elo (extra):    {m_both_score:10.5f}  {w_both_score:10.5f}")
print()
print(f"  Men FF replace delta:   {m_ff_score - m_baseline:+.5f}  {'BETTER' if m_ff_score < m_baseline else 'WORSE'}")
print(f"  Men FF extra delta:     {m_both_score - m_baseline:+.5f}  {'BETTER' if m_both_score < m_baseline else 'WORSE'}")
print(f"  Women FF replace delta: {w_ff_score - w_baseline:+.5f}  {'BETTER' if w_ff_score < w_baseline else 'WORSE'}")
print(f"  Women FF extra delta:   {w_both_score - w_baseline:+.5f}  {'BETTER' if w_both_score < w_baseline else 'WORSE'}")
print(f"\n  Total time: {time.time()-t0:.1f}s")
