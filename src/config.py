"""Centralized model configurations.

All scripts should import configs from here instead of defining them inline.
This prevents config drift between evaluation, sweeps, and submission scripts.

Two configs are maintained:
  - J: The evaluation baseline (Kaggle best: 0.14502)
  - O: LOSO-validated improvements over J (lean XGB, women C=0.5, full CB)
"""
from src.model_elo_v2 import EloConfig
from src.model_logreg_v2 import LogRegV2Config
from src.model_stack_v1 import StackConfig

# ============================================================
# Elo configs (shared by J and O)
# ============================================================
MEN_ELO = EloConfig(k=14, home_adv=40, carryover=0.94, season_decay=0.15)
WOMEN_ELO = EloConfig(k=20, home_adv=40, carryover=0.92, season_decay=0.15)

# ============================================================
# J config — evaluation baseline (Kaggle best: 0.14502)
# ============================================================
J_MEN_LR = LogRegV2Config(
    base_features=['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
    interaction_pairs=[('Elo_diff', 'SeedNum_diff'), ('Elo_diff', 'Rank_POM_diff'), ('Off_Eff_diff', 'Win_pct_diff')],
    C=0.2,
)
J_WOMEN_LR = LogRegV2Config(
    base_features=['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff', 'WPR_Rating_diff', 'WPR_SOS_diff'],
    interaction_pairs=[],
    C=0.2,
)
J_MEN_FEATURES = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff', 'Net_Eff_diff',
                     'eFG_off_diff', 'eFG_def_diff', 'TO_rate_off_diff', 'TO_rate_def_diff',
                     'OR_pct_diff', 'DR_pct_diff', 'FT_rate_off_diff', 'FT_rate_def_diff'],
    'cb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
}
J_WOMEN_FEATURES = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff',
                     'WPR_Rating_diff', 'WPR_SOS_diff', 'Off_Eff_diff', 'Def_Eff_diff', 'Win_pct_diff'],
    'cb_features': ['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'WPR_Rating_diff'],
}

# ============================================================
# O config — LOSO-validated improvements over J
#   Men:   lean XGB (5 features, same as CB)
#   Women: C=0.5, full CB (match XGB 10 features)
# ============================================================
O_MEN_LR = LogRegV2Config(
    base_features=['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
    interaction_pairs=[('Elo_diff', 'SeedNum_diff'), ('Elo_diff', 'Rank_POM_diff'), ('Off_Eff_diff', 'Win_pct_diff')],
    C=0.2,
)
O_WOMEN_LR = LogRegV2Config(
    base_features=['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff', 'WPR_Rating_diff', 'WPR_SOS_diff'],
    interaction_pairs=[],
    C=0.5,
)
O_MEN_FEATURES = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
    'cb_features': ['Elo_diff', 'SeedNum_diff', 'Rank_POM_diff', 'Off_Eff_diff', 'Win_pct_diff'],
}
O_WOMEN_FEATURES = {
    'xgb_features': ['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff',
                     'WPR_Rating_diff', 'WPR_SOS_diff', 'Off_Eff_diff', 'Def_Eff_diff', 'Win_pct_diff'],
    'cb_features': ['Elo_diff', 'SeedNum_diff', 'Net_Eff_diff', 'PPG_diff', 'PPG_allowed_diff',
                     'WPR_Rating_diff', 'WPR_SOS_diff', 'Off_Eff_diff', 'Def_Eff_diff', 'Win_pct_diff'],
}

# ============================================================
# Shared constants
# ============================================================
WOMEN_GLOBAL_ALPHA = 1.2
