import numpy as np
import pandas as pd
import xgboost as xgb
from src.feature_engineering import elo_to_prob


def generate_submission(sample_path, men_model, men_features, men_feat_cols,
                        women_model, women_features, women_feat_cols,
                        elo_lookup, men_blend_w=0.55, women_blend_w=0.60,
                        clip_low=0.02, clip_high=0.98):
    '''
    Generate blended XGBoost + Elo predictions for all matchups in a sample submission.

    For each matchup ID (SSSS_XXXX_YYYY):
    - Looks up features for both teams in season SSSS
    - Computes XGBoost prediction from feature diffs
    - Computes Elo probability
    - Blends: pred = w * xgb + (1-w) * elo

    Args:
        sample_path: path to SampleSubmissionStage1.csv or Stage2.csv
        men_model: trained xgb.Booster for men
        men_features: DataFrame from build_team_features() for men
        men_feat_cols: list of feature column names used by men_model
        women_model: trained xgb.Booster for women
        women_features: DataFrame from build_team_features() for women
        women_feat_cols: list of feature column names used by women_model
        elo_lookup: dict of (Season, TeamID) -> Elo rating
        men_blend_w: XGBoost weight for men's predictions
        women_blend_w: XGBoost weight for women's predictions
        clip_low: minimum prediction value
        clip_high: maximum prediction value

    Returns:
        DataFrame with columns [ID, Pred]
    '''
    sub = pd.read_csv(sample_path)

    # Parse IDs
    parts = sub['ID'].str.split('_', expand=True).astype(int)
    sub['Season'] = parts[0]
    sub['TeamA'] = parts[1]  # lower ID
    sub['TeamB'] = parts[2]  # higher ID

    # Split men's and women's
    sub['is_women'] = sub['TeamA'] >= 3000

    # Compute Elo probabilities for all rows
    default_elo = 1500
    sub['EloA'] = sub.apply(
        lambda r: elo_lookup.get((r['Season'], r['TeamA']), default_elo), axis=1
    )
    sub['EloB'] = sub.apply(
        lambda r: elo_lookup.get((r['Season'], r['TeamB']), default_elo), axis=1
    )
    sub['elo_pred'] = elo_to_prob(sub['EloA'].values, sub['EloB'].values)

    # Generate XGBoost predictions for men
    men_mask = ~sub['is_women']
    sub.loc[men_mask, 'xgb_pred'] = _predict_xgb(
        sub[men_mask], men_model, men_features, men_feat_cols
    )
    sub.loc[men_mask, 'blend_w'] = men_blend_w

    # Generate XGBoost predictions for women
    women_mask = sub['is_women']
    sub.loc[women_mask, 'xgb_pred'] = _predict_xgb(
        sub[women_mask], women_model, women_features, women_feat_cols
    )
    sub.loc[women_mask, 'blend_w'] = women_blend_w

    # Blend: where XGBoost prediction exists, blend; otherwise pure Elo
    sub['Pred'] = (
        sub['blend_w'] * sub['xgb_pred'] +
        (1 - sub['blend_w']) * sub['elo_pred']
    )

    # Clip
    sub['Pred'] = sub['Pred'].clip(clip_low, clip_high)

    # Stats
    has_xgb = sub['xgb_pred'].notna()
    n_men = men_mask.sum()
    n_women = women_mask.sum()
    n_xgb = has_xgb.sum()
    n_elo_fallback = (~has_xgb).sum()
    print(f'Total matchups: {len(sub)} (men: {n_men}, women: {n_women})')
    print(f'XGBoost predictions: {n_xgb}, Elo fallback: {n_elo_fallback}')
    print(f'Pred range: [{sub["Pred"].min():.4f}, {sub["Pred"].max():.4f}]')
    print(f'Pred mean: {sub["Pred"].mean():.4f}')

    return sub[['ID', 'Pred']]


def _predict_xgb(matchup_rows, model, team_features, feat_cols):
    '''
    Generate XGBoost predictions for a set of matchup rows.

    Returns array of predictions, with NaN where features are missing.
    '''
    # Identify feature columns (without _diff suffix) from the feat_cols
    # feat_cols are like ['Elo_diff', 'SeedNum_diff', ...] — strip _diff to get base names
    base_cols = [c.replace('_diff', '') for c in feat_cols]

    # Build feature matrix via merge
    df = matchup_rows[['Season', 'TeamA', 'TeamB']].copy()

    # Merge TeamA features
    a_rename = {col: f'{col}_A' for col in team_features.columns if col not in ('Season', 'TeamID', 'ConfAbbrev')}
    a_feats = team_features.rename(columns=a_rename)
    df = df.merge(a_feats, left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'], how='left')
    df = df.drop(columns=['TeamID'], errors='ignore')

    # Merge TeamB features
    b_rename = {col: f'{col}_B' for col in team_features.columns if col not in ('Season', 'TeamID', 'ConfAbbrev')}
    b_feats = team_features.rename(columns=b_rename)
    df = df.merge(b_feats, left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'], how='left')
    df = df.drop(columns=['TeamID'], errors='ignore')

    # Compute diffs
    for base in base_cols:
        diff_name = f'{base}_diff'
        col_a = f'{base}_A'
        col_b = f'{base}_B'
        if col_a in df.columns and col_b in df.columns:
            df[diff_name] = df[col_a] - df[col_b]

    # Predict only where we have features for both teams
    available_feats = [c for c in feat_cols if c in df.columns]
    feature_matrix = df[available_feats]

    # Rows where all features are present get XGBoost prediction
    # Rows with any NaN get NaN (will fall back to Elo)
    dmatrix = xgb.DMatrix(feature_matrix)
    preds = model.predict(dmatrix)

    return preds
