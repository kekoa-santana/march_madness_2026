"""Systematic LOSO sweep of hyperparameters, features, and women's config.

Tests changes ONE AT A TIME against J config baseline to isolate effects.
Runs both men and women, reports delta vs baseline.
"""
import sys
import time
import copy
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from src.data_loader import load_men_data, load_women_data
from src.feature_engineering import (
    parse_seeds, compute_season_stats, compute_massey_features,
    compute_conference_strength, compute_efficiency, build_team_features,
    build_matchup_matrix,
)
from src.model_elo_v2 import compute_elo_ratings_v2
from src.model_logreg_v2 import LogRegV2Config
from src.model_stack_v1 import StackConfig
from src.women_rankings_v1 import WomenRankConfig, compute_women_power_ratings, merge_women_rank_features
from src.eval_loso import loso_evaluate
from src.config import (
    MEN_ELO, WOMEN_ELO,
    J_MEN_LR as BASE_MEN_LR, J_WOMEN_LR as BASE_WOMEN_LR,
    J_MEN_FEATURES as BASE_MEN_FEATS, J_WOMEN_FEATURES as BASE_WOMEN_FEATS,
)

data_dir = './data'

BASE_STACK = StackConfig()

def make_stack(**overrides):
    d = {
        'meta_C': 1.0,
        'xgb_params': {
            "objective": "binary:logistic", "eval_metric": "logloss",
            "max_depth": 3, "eta": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 5,
            "alpha": 1.5, "lambda": 2.0, "seed": 42, "verbosity": 0,
        },
        'xgb_num_boost_round': 220,
        'cat_params': {
            "loss_function": "Logloss", "depth": 5, "learning_rate": 0.04,
            "l2_leaf_reg": 5.0, "iterations": 300, "random_seed": 42, "verbose": False,
        },
    }
    for k, v in overrides.items():
        if k == 'xgb_max_depth':
            d['xgb_params'] = {**d['xgb_params'], 'max_depth': v}
        elif k == 'xgb_eta':
            d['xgb_params'] = {**d['xgb_params'], 'eta': v}
        elif k == 'xgb_min_child_weight':
            d['xgb_params'] = {**d['xgb_params'], 'min_child_weight': v}
        elif k == 'cat_depth':
            d['cat_params'] = {**d['cat_params'], 'depth': v}
        elif k == 'cat_lr':
            d['cat_params'] = {**d['cat_params'], 'learning_rate': v}
        elif k == 'cat_iterations':
            d['cat_params'] = {**d['cat_params'], 'iterations': v}
        elif k == 'cat_l2':
            d['cat_params'] = {**d['cat_params'], 'l2_leaf_reg': v}
        elif k in d:
            d[k] = v
    return StackConfig(
        meta_C=d['meta_C'],
        xgb_params=d['xgb_params'],
        xgb_num_boost_round=d['xgb_num_boost_round'],
        cat_params=d['cat_params'],
        oof_start_season=2003,
        oof_end_season=2025,
    )

def make_women_stack(**overrides):
    """Same as make_stack but with oof_start_season=2010."""
    cfg = make_stack(**overrides)
    cfg.oof_start_season = 2010
    return cfg


# ============================================================
# Load data
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
print(f"Data loaded in {time.time()-t0:.1f}s\n")


def run_experiment(name, gender, lr_cfg, feat_cfg, stack_cfg):
    """Run one LOSO experiment, return summary dict."""
    matchups = m_matchups if gender == 'M' else w_matchups
    t = time.time()
    result = loso_evaluate(matchups, feat_cfg, lr_cfg, stack_cfg, verbose=False)
    elapsed = time.time() - t
    return {
        'name': name,
        'gender': gender,
        'mean': result['mean_brier'],
        'std': result['std_brier'],
        'pooled': result['pooled_brier'],
        'n_seasons': result['n_seasons'],
        'time': elapsed,
        'per_season': result['per_season'],
    }


# ============================================================
# Define experiments
# ============================================================
experiments = []

# --- BASELINES ---
experiments.append(('J_baseline', 'M', BASE_MEN_LR, BASE_MEN_FEATS, make_stack()))
experiments.append(('J_baseline', 'W', BASE_WOMEN_LR, BASE_WOMEN_FEATS, make_women_stack()))

# ============================================================
# 1. HYPERPARAMETER SWEEPS
# ============================================================

# --- LR C values ---
for c_val in [0.05, 0.1, 0.5, 1.0]:
    lr = LogRegV2Config(base_features=BASE_MEN_LR.base_features, interaction_pairs=BASE_MEN_LR.interaction_pairs, C=c_val)
    experiments.append((f'men_C={c_val}', 'M', lr, BASE_MEN_FEATS, make_stack()))

for c_val in [0.05, 0.1, 0.5, 1.0]:
    lr = LogRegV2Config(base_features=BASE_WOMEN_LR.base_features, interaction_pairs=[], C=c_val)
    experiments.append((f'women_C={c_val}', 'W', lr, BASE_WOMEN_FEATS, make_women_stack()))

# --- XGB max_depth ---
for depth in [2, 4, 5]:
    experiments.append((f'men_xgb_depth={depth}', 'M', BASE_MEN_LR, BASE_MEN_FEATS, make_stack(xgb_max_depth=depth)))
    experiments.append((f'women_xgb_depth={depth}', 'W', BASE_WOMEN_LR, BASE_WOMEN_FEATS, make_women_stack(xgb_max_depth=depth)))

# --- XGB num_boost_round ---
for rounds in [100, 150, 300, 400]:
    experiments.append((f'men_xgb_rounds={rounds}', 'M', BASE_MEN_LR, BASE_MEN_FEATS, make_stack(xgb_num_boost_round=rounds)))
    experiments.append((f'women_xgb_rounds={rounds}', 'W', BASE_WOMEN_LR, BASE_WOMEN_FEATS, make_women_stack(xgb_num_boost_round=rounds)))

# --- XGB eta ---
for eta in [0.02, 0.03, 0.08, 0.1]:
    experiments.append((f'men_xgb_eta={eta}', 'M', BASE_MEN_LR, BASE_MEN_FEATS, make_stack(xgb_eta=eta)))
    experiments.append((f'women_xgb_eta={eta}', 'W', BASE_WOMEN_LR, BASE_WOMEN_FEATS, make_women_stack(xgb_eta=eta)))

# --- XGB min_child_weight ---
for mcw in [3, 10, 20]:
    experiments.append((f'men_xgb_mcw={mcw}', 'M', BASE_MEN_LR, BASE_MEN_FEATS, make_stack(xgb_min_child_weight=mcw)))
    experiments.append((f'women_xgb_mcw={mcw}', 'W', BASE_WOMEN_LR, BASE_WOMEN_FEATS, make_women_stack(xgb_min_child_weight=mcw)))

# --- CatBoost depth ---
for depth in [3, 4, 6]:
    experiments.append((f'men_cat_depth={depth}', 'M', BASE_MEN_LR, BASE_MEN_FEATS, make_stack(cat_depth=depth)))
    experiments.append((f'women_cat_depth={depth}', 'W', BASE_WOMEN_LR, BASE_WOMEN_FEATS, make_women_stack(cat_depth=depth)))

# --- CatBoost learning_rate ---
for lr_val in [0.02, 0.06, 0.08]:
    experiments.append((f'men_cat_lr={lr_val}', 'M', BASE_MEN_LR, BASE_MEN_FEATS, make_stack(cat_lr=lr_val)))
    experiments.append((f'women_cat_lr={lr_val}', 'W', BASE_WOMEN_LR, BASE_WOMEN_FEATS, make_women_stack(cat_lr=lr_val)))

# --- Meta C ---
for mc in [0.1, 0.5, 2.0]:
    experiments.append((f'men_meta_C={mc}', 'M', BASE_MEN_LR, BASE_MEN_FEATS, make_stack(meta_C=mc)))
    experiments.append((f'women_meta_C={mc}', 'W', BASE_WOMEN_LR, BASE_WOMEN_FEATS, make_women_stack(meta_C=mc)))

# ============================================================
# 2. FEATURE EXPERIMENTS
# ============================================================

# --- Men: remove Four Factors from XGB ---
men_feats_no_ff = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff', 'Net_Eff_diff'],
    'cb_features': BASE_MEN_FEATS['cb_features'],
}
experiments.append(('men_no_four_factors', 'M', BASE_MEN_LR, men_feats_no_ff, make_stack()))

# --- Men: remove interactions from LR ---
men_lr_no_int = LogRegV2Config(base_features=BASE_MEN_LR.base_features, interaction_pairs=[], C=0.2)
experiments.append(('men_no_interactions', 'M', men_lr_no_int, BASE_MEN_FEATS, make_stack()))

# --- Men: add Net_Eff to LR ---
men_lr_plus_neteff = LogRegV2Config(
    base_features=BASE_MEN_LR.base_features + ['Net_Eff_diff'],
    interaction_pairs=BASE_MEN_LR.interaction_pairs, C=0.2,
)
experiments.append(('men_lr+Net_Eff', 'M', men_lr_plus_neteff, BASE_MEN_FEATS, make_stack()))

# --- Men: lean XGB (same as CB) ---
men_feats_lean_xgb = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
    'cb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
}
experiments.append(('men_lean_xgb', 'M', BASE_MEN_LR, men_feats_lean_xgb, make_stack()))

# --- Men: expand CB to match XGB ---
men_feats_full_cb = {
    'xgb_features': BASE_MEN_FEATS['xgb_features'],
    'cb_features': BASE_MEN_FEATS['xgb_features'],
}
experiments.append(('men_full_cb', 'M', BASE_MEN_LR, men_feats_full_cb, make_stack()))

# --- Women: add interactions back ---
women_lr_with_int = LogRegV2Config(
    base_features=BASE_WOMEN_LR.base_features,
    interaction_pairs=[('Elo_diff', 'SeedNum_diff')],
    C=0.2,
)
experiments.append(('women_add_Elo*Seed', 'W', women_lr_with_int, BASE_WOMEN_FEATS, make_women_stack()))

# --- Women: fewer LR features (lean) ---
women_lr_lean = LogRegV2Config(
    base_features=['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff'],
    interaction_pairs=[], C=0.2,
)
experiments.append(('women_lr_lean_3feat', 'W', women_lr_lean, BASE_WOMEN_FEATS, make_women_stack()))

# --- Women: add eFG to XGB ---
women_feats_efg = {
    'xgb_features': BASE_WOMEN_FEATS['xgb_features'] + ['eFG_off_diff', 'eFG_def_diff'],
    'cb_features': BASE_WOMEN_FEATS['cb_features'],
}
experiments.append(('women_xgb+eFG', 'W', BASE_WOMEN_LR, women_feats_efg, make_women_stack()))

# --- Women: add Four Factors to XGB ---
women_feats_ff = {
    'xgb_features': BASE_WOMEN_FEATS['xgb_features'] + [
        'eFG_off_diff', 'eFG_def_diff', 'TO_rate_off_diff', 'TO_rate_def_diff',
        'OR_pct_diff', 'DR_pct_diff', 'FT_rate_off_diff', 'FT_rate_def_diff'],
    'cb_features': BASE_WOMEN_FEATS['cb_features'],
}
experiments.append(('women_xgb+FourFactors', 'W', BASE_WOMEN_LR, women_feats_ff, make_women_stack()))

# --- Women: expand CB features ---
women_feats_full_cb = {
    'xgb_features': BASE_WOMEN_FEATS['xgb_features'],
    'cb_features': BASE_WOMEN_FEATS['xgb_features'],
}
experiments.append(('women_full_cb', 'W', BASE_WOMEN_LR, women_feats_full_cb, make_women_stack()))

# --- Women: remove WPR from everything ---
women_lr_no_wpr = LogRegV2Config(
    base_features=['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff'],
    interaction_pairs=[], C=0.2,
)
women_feats_no_wpr = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff',
                     'Off_Eff_diff', 'Def_Eff_diff', 'Win_pct_diff'],
    'cb_features': ['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff'],
}
experiments.append(('women_no_WPR', 'W', women_lr_no_wpr, women_feats_no_wpr, make_women_stack()))

# ============================================================
# 3. WOMEN-SPECIFIC CONFIG CHANGES
# ============================================================

# --- Women: different Elo configs (need to rebuild features) ---
# We'll handle these separately after the main sweep

# ============================================================
# Run all experiments
# ============================================================
print(f"Running {len(experiments)} experiments...\n")

results = []
baselines = {}

for i, (name, gender, lr_cfg, feat_cfg, stack_cfg) in enumerate(experiments):
    label = f"[{i+1}/{len(experiments)}] {gender} {name}"
    print(f"{label}...", end=' ', flush=True)
    r = run_experiment(name, gender, lr_cfg, feat_cfg, stack_cfg)
    results.append(r)
    print(f"mean={r['mean']:.5f} ({r['time']:.1f}s)")

    if name == 'J_baseline':
        baselines[gender] = r['mean']

# ============================================================
# Print sorted results by gender
# ============================================================
for gender, label in [('M', "MEN'S"), ('W', "WOMEN'S")]:
    gender_results = [r for r in results if r['gender'] == gender]
    gender_results.sort(key=lambda x: x['mean'])
    baseline = baselines[gender]

    print(f"\n{'='*70}")
    print(f"{label} RESULTS (sorted by LOSO mean, baseline={baseline:.5f})")
    print(f"{'='*70}")
    print(f"  {'Name':40s} {'Mean':>8s} {'Delta':>8s} {'Std':>8s} {'Pooled':>8s}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in gender_results:
        delta = r['mean'] - baseline
        marker = ' ***' if delta < -0.001 else ' **' if delta < -0.0005 else ' *' if delta < 0 else ''
        print(f"  {r['name']:40s} {r['mean']:.5f} {delta:+.5f} {r['std']:.5f} {r['pooled']:.5f}{marker}")

print(f"\n  (* = better than baseline, ** = >0.0005 better, *** = >0.001 better)")
print(f"\nTotal time: {time.time()-t0:.1f}s")
