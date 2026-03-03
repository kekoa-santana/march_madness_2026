import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.model import brier_score, get_feature_cols


# High-signal features only — the minimal set for LR
MINIMAL_FEATURES = [
    'Elo_diff', 'SeedNum_diff'
]

# Minimal + Massey for men
MINIMAL_FEATURES_MEN = [
    'Elo_diff', 'SeedNum_diff', 'Rank_POM_diff'
]


def train_logreg(matchups, feature_cols=None, C=1.0):
    '''
    Train a logistic regression model on tournament matchups.

    Args:
        matchups: DataFrame from build_matchup_matrix() with diff features and Target
        feature_cols: list of feature columns to use (defaults to all _diff columns)
        C: inverse regularization strength (smaller = more regularization)

    Returns:
        Tuple of (fitted LogisticRegression, fitted StandardScaler, list of feature cols)
    '''
    if feature_cols is None:
        feature_cols = get_feature_cols(matchups)

    X = matchups[feature_cols].values
    y = matchups['Target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, feature_cols


def predict_logreg(model, scaler, X):
    '''Predict P(TeamA wins) using fitted LR model.'''
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


def logreg_cv(matchups, feature_cols=None, C=1.0) -> pd.DataFrame:
    '''
    Leave-one-season-out CV for logistic regression.

    Trains on all seasons before the held-out season (no future leakage).

    Args:
        matchups: DataFrame with diff features and Target
        feature_cols: list of feature columns
        C: regularization parameter

    Returns:
        DataFrame with [Season, TeamA, TeamB, Target, Pred]
    '''
    if feature_cols is None:
        feature_cols = get_feature_cols(matchups)

    seasons = sorted(matchups['Season'].unique())
    oof_preds = []

    for season in seasons:
        train = matchups[matchups['Season'] < season]
        test = matchups[matchups['Season'] == season]

        if len(test) == 0 or len(train) == 0:
            continue

        model, scaler, _ = train_logreg(train, feature_cols, C=C)
        preds = predict_logreg(model, scaler, test[feature_cols].values)

        bs = brier_score(test['Target'].values, preds)

        fold = test[['Season', 'TeamA', 'TeamB', 'Target']].copy()
        fold['Pred'] = preds
        oof_preds.append(fold)

        print(f'Season {season}: Brier={bs:.5f}  (n={len(test)})')

    results = pd.concat(oof_preds, ignore_index=True)
    overall = brier_score(results['Target'].values, results['Pred'].values)
    print(f'\nOverall OOF Brier (LogReg): {overall:.5f}')

    return results


def logreg_holdout_eval(matchups, feature_cols=None, C=1.0, train_cutoff=2022):
    '''
    Train on seasons < train_cutoff, evaluate on seasons >= train_cutoff.

    Args:
        matchups: DataFrame with diff features and Target
        feature_cols: list of feature columns
        C: regularization parameter
        train_cutoff: first season in the holdout set

    Returns:
        Tuple of (predictions array, trained model, scaler, feature_cols)
    '''
    if feature_cols is None:
        feature_cols = get_feature_cols(matchups)

    train = matchups[matchups['Season'] < train_cutoff]
    test = matchups[matchups['Season'] >= train_cutoff]

    model, scaler, _ = train_logreg(train, feature_cols, C=C)
    preds = predict_logreg(model, scaler, test[feature_cols].values)

    bs = brier_score(test['Target'].values, preds)
    print(f'Holdout Brier (LogReg, C={C}): {bs:.5f}  (n={len(test)})')

    return preds, model, scaler, feature_cols


def print_logreg_coefficients(model, feature_cols):
    '''Print logistic regression coefficients sorted by absolute value.'''
    coefs = pd.Series(model.coef_[0], index=feature_cols)
    coefs_sorted = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    print('Logistic Regression Coefficients:')
    for feat, coef in coefs_sorted.items():
        print(f'  {feat:30s} {coef:+.4f}')
