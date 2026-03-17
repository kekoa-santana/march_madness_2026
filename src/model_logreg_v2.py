from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.model import brier_score


MEN_V2_BASE_FEATURES = [
    "Elo_diff",
    "SeedNum_diff",
    "Rank_POM_diff",
    "Off_Eff_diff",
    "Win_pct_diff",
]

MEN_V2_INTERACTIONS = [
    ("Elo_diff", "SeedNum_diff"),
    ("Elo_diff", "Rank_POM_diff"),
    ("Off_Eff_diff", "Win_pct_diff"),
]

WOMEN_V2_BASE_FEATURES = [
    "Elo_diff",
    "SeedNum_diff",
    "Net_Eff_diff",
    "PPG_diff",
    "PPG_allowed_diff",
    "WPR_Rating_diff",
    "WPR_SOS_diff",
]

WOMEN_V2_INTERACTIONS = [
    ("Elo_diff", "SeedNum_diff"),
    ("Elo_diff", "Net_Eff_diff"),
    ("PPG_diff", "PPG_allowed_diff"),
]


@dataclass
class LogRegV2Config:
    base_features: list[str]
    interaction_pairs: list[tuple[str, str]] = field(default_factory=list)
    C: float = 1.0
    max_iter: int = 3000
    random_state: int = 42


def interaction_name(col_a: str, col_b: str) -> str:
    return f'{col_a.replace("_diff", "")}x{col_b.replace("_diff", "")}_diff'


def add_interaction_columns(
    matchups: pd.DataFrame, interaction_pairs: list[tuple[str, str]]
) -> tuple[pd.DataFrame, list[str]]:
    df = matchups.copy()
    cols = []
    for col_a, col_b in interaction_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        name = interaction_name(col_a, col_b)
        df[name] = df[col_a] * df[col_b]
        cols.append(name)
    return df, cols


def prepare_logreg_frame(matchups: pd.DataFrame, cfg: LogRegV2Config) -> tuple[pd.DataFrame, list[str]]:
    df, interaction_cols = add_interaction_columns(matchups, cfg.interaction_pairs)
    feature_cols = [c for c in cfg.base_features if c in df.columns] + interaction_cols
    return df, feature_cols


def train_logreg_v2(matchups: pd.DataFrame, cfg: LogRegV2Config):
    df, feature_cols = prepare_logreg_frame(matchups, cfg)
    valid = df[feature_cols].notna().all(axis=1)
    train_df = df.loc[valid].copy()
    if len(train_df) == 0:
        raise ValueError("No valid rows to train logistic regression v2.")

    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[feature_cols].values)
    y = train_df["Target"].values
    model = LogisticRegression(
        C=cfg.C, max_iter=cfg.max_iter, solver="lbfgs", random_state=cfg.random_state
    )
    model.fit(X, y)
    return model, scaler, feature_cols


def predict_logreg_v2(model, scaler, matchups: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    preds = np.full(len(matchups), np.nan, dtype=float)
    missing = [c for c in feature_cols if c not in matchups.columns]
    if missing:
        return preds

    valid = matchups[feature_cols].notna().all(axis=1)
    if not valid.any():
        return preds

    X = scaler.transform(matchups.loc[valid, feature_cols].values)
    preds[valid.values] = model.predict_proba(X)[:, 1]
    return preds


def rolling_oof_logreg_v2(
    matchups: pd.DataFrame,
    cfg: LogRegV2Config,
    start_season: int | None = None,
    end_season: int | None = None,
) -> pd.DataFrame:
    df, feature_cols = prepare_logreg_frame(matchups, cfg)
    seasons = sorted(df["Season"].unique())
    if start_season is not None:
        seasons = [s for s in seasons if s >= start_season]
    if end_season is not None:
        seasons = [s for s in seasons if s <= end_season]

    folds = []
    for season in seasons:
        train = df[df["Season"] < season].copy()
        test = df[df["Season"] == season].copy()
        if len(train) == 0 or len(test) == 0:
            continue

        valid_train = train[feature_cols].notna().all(axis=1)
        train = train.loc[valid_train]
        if len(train) == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_cols].values)
        y_train = train["Target"].values
        model = LogisticRegression(
            C=cfg.C, max_iter=cfg.max_iter, solver="lbfgs", random_state=cfg.random_state
        )
        model.fit(X_train, y_train)
        preds = predict_logreg_v2(model, scaler, test, feature_cols)

        fold = test[["Season", "TeamA", "TeamB", "Target"]].copy()
        fold["Pred"] = preds
        fold["TrainMaxSeason"] = int(train["Season"].max())
        folds.append(fold)

    if not folds:
        raise ValueError("No OOF folds generated for logistic regression v2.")

    return pd.concat(folds, ignore_index=True)


def holdout_logreg_v2(
    matchups: pd.DataFrame, cfg: LogRegV2Config, train_cutoff: int = 2022
) -> tuple[np.ndarray, float]:
    df, feature_cols = prepare_logreg_frame(matchups, cfg)
    train = df[df["Season"] < train_cutoff].copy()
    test = df[df["Season"] >= train_cutoff].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Invalid split for holdout_logreg_v2.")

    valid_train = train[feature_cols].notna().all(axis=1)
    train = train.loc[valid_train]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols].values)
    y_train = train["Target"].values

    model = LogisticRegression(
        C=cfg.C, max_iter=cfg.max_iter, solver="lbfgs", random_state=cfg.random_state
    )
    model.fit(X_train, y_train)
    preds = predict_logreg_v2(model, scaler, test, feature_cols)
    valid_test = ~np.isnan(preds)
    bs = brier_score(test.loc[valid_test, "Target"].values, preds[valid_test])
    return preds, float(bs)
