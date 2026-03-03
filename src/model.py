import numpy as np
import pandas as pd
import xgboost as xgb


def brier_score(y_true, y_pred):
    '''Mean squared error between true labels and predicted probabilities.'''
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def get_feature_cols(matchups):
    '''Return list of diff feature columns from a matchup matrix.'''
    return [c for c in matchups.columns if c.endswith('_diff')]


def leave_one_season_out_cv(matchups, xgb_params=None, num_boost_round=300) -> pd.DataFrame:
    '''
    Leave-one-season-out cross-validation for XGBoost on tournament matchups.

    For each season, trains on all other seasons and predicts the held-out season.
    Returns out-of-fold predictions with Brier score per season.

    Args:
        matchups: DataFrame from build_matchup_matrix() with diff features and Target
        xgb_params: dict of XGBoost parameters (optional, uses tuned defaults)
        num_boost_round: number of boosting rounds

    Returns:
        DataFrame with [Season, TeamA, TeamB, Target, Pred]
    '''
    if xgb_params is None:
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1.0,
            'lambda': 1.0,
            'seed': 42,
            'verbosity': 0,
        }

    feature_cols = get_feature_cols(matchups)
    seasons = sorted(matchups['Season'].unique())
    oof_preds = []

    for season in seasons:
        train = matchups[matchups['Season'] < season]
        test = matchups[matchups['Season'] == season]

        if len(test) == 0:
            continue
        if len(train) == 0:
            print(f'Season {season} skipped because no prior year training data')
            continue

        dtrain = xgb.DMatrix(train[feature_cols], label=train['Target'])
        dtest = xgb.DMatrix(test[feature_cols])

        model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)
        preds = model.predict(dtest)

        bs = brier_score(test['Target'].values, preds)

        fold = test[['Season', 'TeamA', 'TeamB', 'Target']].copy()
        fold['Pred'] = preds
        oof_preds.append(fold)

        print(f'Season {season}: Brier={bs:.5f}  (n={len(test)})')

    results = pd.concat(oof_preds, ignore_index=True)
    overall = brier_score(results['Target'].values, results['Pred'].values)
    print(f'\nOverall OOF Brier (all features): {overall:.5f}')

    return results


def train_final_model(matchups, feature_cols=None, xgb_params=None, num_boost_round=300):
    '''
    Train XGBoost on all available tournament matchups.

    Args:
        matchups: DataFrame from build_matchup_matrix()
        xgb_params: dict of XGBoost parameters (optional)
        num_boost_round: number of boosting rounds

    Returns:
        Tuple of (fitted xgb.Booster, list of feature column names)
    '''
    if xgb_params is None:
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1.0,
            'lambda': 1.0,
            'seed': 42,
            'verbosity': 0,
        }

    if feature_cols is None:
        feature_cols = get_feature_cols(matchups)
    dtrain = xgb.DMatrix(matchups[feature_cols], label=matchups['Target'])
    model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)

    return model, feature_cols


def print_feature_importance(model, top_n=15):
    '''Print top features by XGBoost gain importance.'''
    scores = model.get_score(importance_type='gain')
    importance = pd.Series(scores).sort_values(ascending=False).head(top_n)
    print('Top features by importance:')
    for feat, score in importance.items():
        print(f'  {feat:30s} {score:.4f}')


# Recommended lean feature set — drops redundant Massey/ranking columns
LEAN_FEATURES = [
    'Elo_diff', 'SeedNum_diff', 'Massey_median_diff', 'Rank_POM_diff',
    'eFG_off_diff', 'TO_rate_off_diff', 'OR_pct_diff', 'FT_rate_off_diff',
    'eFG_def_diff', 'TO_rate_def_diff', 'DR_pct_diff', 'FT_rate_def_diff',
    'Tempo_diff', 'PPG_diff', 'PPG_allowed_diff',
    'Win_pct_diff', 'Conf_Elo_mean_diff',
]

LEAN_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'alpha': 2.0,
    'lambda': 2.0,
    'min_child_weight': 5,
    'seed': 42,
    'verbosity': 0,
}


def leave_one_season_out_cv_lean(matchups, feature_cols=None, xgb_params=None,
                                  num_boost_round=200) -> pd.DataFrame:
    '''
    Lean CV with reduced features and stronger regularization.

    Args:
        matchups: DataFrame from build_matchup_matrix()
        feature_cols: list of feature columns to use (defaults to LEAN_FEATURES)
        xgb_params: dict of XGBoost params (defaults to LEAN_PARAMS)
        num_boost_round: number of boosting rounds

    Returns:
        DataFrame with [Season, TeamA, TeamB, Target, Pred]
    '''
    if feature_cols is None:
        feature_cols = [c for c in LEAN_FEATURES if c in matchups.columns]
    if xgb_params is None:
        xgb_params = LEAN_PARAMS

    seasons = sorted(matchups['Season'].unique())
    oof_preds = []

    for season in seasons:
        train = matchups[matchups['Season'] < season]
        test = matchups[matchups['Season'] == season]

        if len(test) == 0:
            continue
        if len(train) == 0:
            print(f'Season {season}: skipped (no prior training data)')
            continue

        dtrain = xgb.DMatrix(train[feature_cols], label=train['Target'])
        dtest = xgb.DMatrix(test[feature_cols])

        model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)
        preds = model.predict(dtest)

        bs = brier_score(test['Target'].values, preds)

        fold = test[['Season', 'TeamA', 'TeamB', 'Target']].copy()
        fold['Pred'] = preds
        oof_preds.append(fold)

        print(f'Season {season}: Brier={bs:.5f}  (n={len(test)})')

    if not oof_preds:
        raise ValueError("No OOF predictions generated; check season split/data.")

    results = pd.concat(oof_preds, ignore_index=True)
    overall = brier_score(results['Target'].values, results['Pred'].values)
    print(f'\nOverall OOF Brier (lean): {overall:.5f}')

    return results


def find_best_blend(oof_results, elo_probs):
    '''
    Find optimal blend weight between XGBoost OOF predictions and Elo probabilities.

    Tests weights from 0.0 (pure Elo) to 1.0 (pure XGBoost) in 5% increments.

    Args:
        oof_results: DataFrame from leave_one_season_out_cv with [Target, Pred]
        elo_probs: array-like of Elo-based probabilities aligned to oof_results rows

    Returns:
        Tuple of (best_weight, best_brier) where blend = w * xgb + (1-w) * elo
    '''
    y = oof_results['Target'].values
    xgb_preds = oof_results['Pred'].values
    elo_preds = np.array(elo_probs)

    best_w, best_bs = 0.0, brier_score(y, elo_preds)
    print(f'w=0.00 (pure Elo):    Brier={best_bs:.5f}')

    for w in np.arange(0.05, 1.01, 0.05):
        blend = w * xgb_preds + (1 - w) * elo_preds
        bs = brier_score(y, blend)
        marker = ' <-- pure XGB' if abs(w - 1.0) < 0.01 else ''
        print(f'w={w:.2f}:               Brier={bs:.5f}{marker}')
        if bs < best_bs:
            best_bs = bs
            best_w = round(w, 2)

    print(f'\nBest blend: w={best_w:.2f}, Brier={best_bs:.5f}')
    return best_w, best_bs
