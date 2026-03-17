import numpy as np
import pandas as pd

from src.feature_engineering import elo_to_prob
from src.model_stack_v1 import predict_stack_from_matchups


def _extract_interaction_pairs(lr_config: dict) -> list[tuple[str, str]]:
    pairs = lr_config.get("interaction_pairs", [])
    out = []
    for p in pairs:
        if isinstance(p, (tuple, list)) and len(p) == 2:
            out.append((p[0], p[1]))
    return out


def _build_matchup_features(
    matchup_rows: pd.DataFrame,
    team_features: pd.DataFrame,
    required_diff_cols: list[str],
    interaction_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    df = matchup_rows[["Season", "TeamA", "TeamB"]].copy()
    ignore_cols = {"Season", "TeamID", "ConfAbbrev"}
    team_num_cols = [c for c in team_features.columns if c not in ignore_cols]

    a_rename = {col: f"{col}_A" for col in team_num_cols}
    b_rename = {col: f"{col}_B" for col in team_num_cols}
    a_feats = team_features.rename(columns=a_rename)
    b_feats = team_features.rename(columns=b_rename)

    df = df.merge(a_feats, left_on=["Season", "TeamA"], right_on=["Season", "TeamID"], how="left")
    df = df.drop(columns=["TeamID"], errors="ignore")
    df = df.merge(b_feats, left_on=["Season", "TeamB"], right_on=["Season", "TeamID"], how="left")
    df = df.drop(columns=["TeamID"], errors="ignore")

    base_cols = set()
    for col in required_diff_cols:
        if col.endswith("_diff") and "x" not in col.replace("_diff", ""):
            base_cols.add(col.replace("_diff", ""))
    for a, b in interaction_pairs:
        base_cols.add(a.replace("_diff", ""))
        base_cols.add(b.replace("_diff", ""))

    for base in base_cols:
        a_col = f"{base}_A"
        b_col = f"{base}_B"
        diff_col = f"{base}_diff"
        if a_col in df.columns and b_col in df.columns:
            df[diff_col] = df[a_col] - df[b_col]

    for col_a, col_b in interaction_pairs:
        if col_a in df.columns and col_b in df.columns:
            name = f'{col_a.replace("_diff", "")}x{col_b.replace("_diff", "")}_diff'
            df[name] = df[col_a] * df[col_b]

    return df


def _group_stack_predict(
    subset: pd.DataFrame,
    artifacts: dict,
    team_features: pd.DataFrame,
    elo_lookup: dict,
    clip_override: tuple[float, float] | None = None,
) -> np.ndarray:
    pred = np.full(len(subset), np.nan, dtype=float)
    if len(subset) == 0:
        return pred

    req = set(artifacts.get("lr_feature_cols", []))
    req.update(artifacts.get("xgb_feature_cols", []))
    req.update(artifacts.get("cb_feature_cols", []))
    req.add("Elo_diff")
    interaction_pairs = _extract_interaction_pairs(artifacts.get("lr_config", {}))

    feature_df = _build_matchup_features(
        subset, team_features, sorted(req), interaction_pairs=interaction_pairs
    )
    pred = predict_stack_from_matchups(feature_df, artifacts, clip=False)

    if clip_override is not None:
        lo, hi = clip_override
    else:
        lo, hi = artifacts.get("clip_low", 0.0), artifacts.get("clip_high", 1.0)
    pred = np.clip(pred, lo, hi)

    # Elo fallback when stacked prediction is missing.
    missing = np.isnan(pred)
    if missing.any():
        elo_a = np.array(
            [
                elo_lookup.get((int(s), int(t)), 1500.0)
                for s, t in zip(subset.loc[missing, "Season"], subset.loc[missing, "TeamA"])
            ],
            dtype=float,
        )
        elo_b = np.array(
            [
                elo_lookup.get((int(s), int(t)), 1500.0)
                for s, t in zip(subset.loc[missing, "Season"], subset.loc[missing, "TeamB"])
            ],
            dtype=float,
        )
        pred[missing] = elo_to_prob(elo_a, elo_b)

    return pred


def generate_submission_stacked(
    sample_path: str,
    men_artifacts: dict,
    women_artifacts: dict,
    men_features: pd.DataFrame,
    women_features: pd.DataFrame,
    elo_lookup: dict,
    clip_low: float | None = None,
    clip_high: float | None = None,
) -> pd.DataFrame:
    """
    Generate stacked submission predictions for men and women.
    """
    sub = pd.read_csv(sample_path)
    parts = sub["ID"].str.split("_", expand=True).astype(int)
    sub["Season"] = parts[0]
    sub["TeamA"] = parts[1]
    sub["TeamB"] = parts[2]
    sub["is_women"] = sub["TeamA"] >= 3000

    clip_override = None
    if clip_low is not None and clip_high is not None:
        clip_override = (clip_low, clip_high)

    men_mask = ~sub["is_women"]
    women_mask = sub["is_women"]

    men_pred = _group_stack_predict(
        sub.loc[men_mask], men_artifacts, men_features, elo_lookup, clip_override
    )
    women_pred = _group_stack_predict(
        sub.loc[women_mask], women_artifacts, women_features, elo_lookup, clip_override
    )

    sub.loc[men_mask, "Pred"] = men_pred
    sub.loc[women_mask, "Pred"] = women_pred

    print(f"Total matchups: {len(sub)} (men: {int(men_mask.sum())}, women: {int(women_mask.sum())})")
    print(
        f'Pred range: [{sub["Pred"].min():.4f}, {sub["Pred"].max():.4f}]  '
        f'mean={sub["Pred"].mean():.4f}'
    )

    return sub[["ID", "Pred"]]
