"""Round-specific alpha stretching for tournament predictions.

Key insight: for any two tournament teams, the bracket structure
deterministically determines which round they would meet in.
Later rounds benefit from more aggressive confidence stretching.

Usage after Selection Sunday:
    from src.round_alpha import apply_round_alphas, optimize_round_alphas
"""
import numpy as np
import pandas as pd


# Standard men's tournament DayNum → round mapping
# (2021 was non-standard but we handle that separately)
DAYNUM_TO_ROUND = {
    134: 0, 135: 0,   # Play-in
    136: 1, 137: 1,   # Round 1 (64→32)
    138: 2, 139: 2,   # Round 2 (32→16)
    143: 3, 144: 3,   # Sweet 16
    145: 4, 146: 4,   # Elite Eight
    152: 5,            # Final Four
    154: 6,            # Championship
}

# 2021 had a compressed schedule in Indianapolis
DAYNUM_TO_ROUND_2021 = {
    134: 0, 135: 0,
    136: 1, 137: 1, 138: 1, 139: 1,  # R1 spread across 4 days
    140: 2, 141: 2,   # R2
    144: 3, 145: 3,   # Sweet 16
    146: 4, 147: 4,   # Elite Eight
    152: 5,            # Final Four
    154: 6,            # Championship
}


def stretch_preds(preds, alpha):
    """Logit-space alpha stretching."""
    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    logit = np.log(preds / (1 - preds))
    stretched = 1.0 / (1.0 + np.exp(-alpha * logit))
    return np.clip(stretched, 0.001, 0.999)


def get_round_from_daynum(season, daynum):
    """Map a tournament game's DayNum to its round (0-6)."""
    if season == 2021:
        return DAYNUM_TO_ROUND_2021.get(daynum)
    return DAYNUM_TO_ROUND.get(daynum)


def build_seed_slot_map(seed_round_slots_df):
    """Build {seed_str: {round: game_slot}} from MNCAATourneySeedRoundSlots.

    Args:
        seed_round_slots_df: DataFrame with Seed, GameRound, GameSlot columns

    Returns:
        dict: e.g. {'W01': {1: 'R1W1', 2: 'R2W1', ...}, ...}
    """
    slot_map = {}
    for _, row in seed_round_slots_df.iterrows():
        seed = row['Seed']
        game_round = int(row['GameRound'])
        game_slot = row['GameSlot']
        if seed not in slot_map:
            slot_map[seed] = {}
        slot_map[seed][game_round] = game_slot
    return slot_map


def get_meeting_round(seed1, seed2, slot_map):
    """Determine which round two seeds would meet.

    Args:
        seed1, seed2: seed strings like 'W01', 'X08'
        slot_map: from build_seed_slot_map()

    Returns:
        int round (1-6) or None if they can't meet
    """
    # Normalize play-in seeds (W16a → W16)
    s1 = seed1[:3] if len(seed1) > 3 else seed1
    s2 = seed2[:3] if len(seed2) > 3 else seed2

    if s1 not in slot_map or s2 not in slot_map:
        return None

    slots1 = slot_map[s1]
    slots2 = slot_map[s2]

    # Find first round where they share a game slot
    for rnd in sorted(set(slots1.keys()) & set(slots2.keys())):
        if rnd == 0:
            # Play-in: only teams with same first 3 chars play each other
            if s1 == s2:
                return 0
            continue
        if slots1[rnd] == slots2[rnd]:
            return rnd

    return None


def assign_rounds_to_matchups(submission_df, seeds_df, seed_round_slots_df):
    """Assign tournament round to each matchup in a submission.

    Args:
        submission_df: DataFrame with 'ID' column (format: SSSS_XXXX_YYYY)
        seeds_df: DataFrame with Season, Seed, TeamID columns
        seed_round_slots_df: DataFrame with Seed, GameRound, GameSlot

    Returns:
        Series of round numbers (1-6) aligned with submission_df index,
        NaN for non-tournament or indeterminate matchups
    """
    slot_map = build_seed_slot_map(seed_round_slots_df)

    # Build team → seed lookup for the submission season
    parts = submission_df['ID'].str.split('_', expand=True).astype(int)
    season = parts[0].iloc[0]

    team_seed = {}
    season_seeds = seeds_df[seeds_df['Season'] == season]
    for _, row in season_seeds.iterrows():
        team_seed[int(row['TeamID'])] = row['Seed']

    rounds = []
    for _, row in submission_df.iterrows():
        id_parts = row['ID'].split('_')
        team_a = int(id_parts[1])
        team_b = int(id_parts[2])

        seed_a = team_seed.get(team_a)
        seed_b = team_seed.get(team_b)

        if seed_a is None or seed_b is None:
            rounds.append(np.nan)
            continue

        rnd = get_meeting_round(seed_a, seed_b, slot_map)
        rounds.append(rnd if rnd is not None else np.nan)

    return pd.Series(rounds, index=submission_df.index, name='round')


def label_historical_rounds(tourney_results_df):
    """Add 'Round' column to historical tournament results.

    Args:
        tourney_results_df: compact results with Season, DayNum, etc.

    Returns:
        DataFrame with added 'Round' column (0-6)
    """
    df = tourney_results_df.copy()
    df['Round'] = df.apply(
        lambda r: get_round_from_daynum(r['Season'], r['DayNum']),
        axis=1
    )
    return df


def optimize_round_alphas(matchups_df, artifacts, feature_sets, lr_cfg, stack_cfg,
                          tourney_results_df, gender='men',
                          alpha_grid=None, eval_seasons=None):
    """Find optimal alpha per round via LOSO on historical tournaments.

    For each round, finds the alpha that minimizes pooled Brier score
    across all LOSO folds for games in that round.

    Args:
        matchups_df: full matchup matrix (from build_matchup_matrix on tourney data)
        artifacts: trained stack artifacts (or None to retrain per fold)
        feature_sets: dict with xgb_features, cb_features
        lr_cfg: LogRegV2Config
        stack_cfg: StackConfig
        tourney_results_df: compact tournament results (for round labeling)
        gender: 'men' or 'women'
        alpha_grid: list of alphas to try (default: 0.5 to 3.0)
        eval_seasons: seasons to evaluate (default: all available)

    Returns:
        dict: {round_num: optimal_alpha}
        DataFrame: detailed per-round, per-alpha Brier scores
    """
    from src.model_stack_v1 import generate_oof_base_preds, predict_stack_from_matchups
    from src.model import brier_score
    from sklearn.linear_model import LogisticRegression

    if alpha_grid is None:
        alpha_grid = np.arange(0.5, 3.05, 0.1).tolist()

    # Label rounds in tournament results
    labeled = label_historical_rounds(tourney_results_df)

    # Add round info to matchups
    matchups = matchups_df.copy()
    # Merge DayNum from tournament results to get round
    tourney_key = labeled[['Season', 'WTeamID', 'LTeamID', 'DayNum', 'Round']].copy()
    # matchups has TeamA (lower ID) and TeamB (higher ID)
    tourney_key['TeamA'] = tourney_key[['WTeamID', 'LTeamID']].min(axis=1)
    tourney_key['TeamB'] = tourney_key[['WTeamID', 'LTeamID']].max(axis=1)
    tourney_key = tourney_key[['Season', 'TeamA', 'TeamB', 'Round']].drop_duplicates()

    matchups = matchups.merge(tourney_key, on=['Season', 'TeamA', 'TeamB'], how='left')

    if eval_seasons is None:
        eval_seasons = sorted(matchups['Season'].unique())

    # Generate OOF predictions (these are raw, un-stretched)
    oof_df = generate_oof_base_preds(matchups, feature_sets, lr_cfg, stack_cfg)

    # LOSO: for each season, fit meta on rest, predict held-out
    all_preds = []

    for test_season in eval_seasons:
        train_mask = oof_df['Season'] != test_season
        test_mask = oof_df['Season'] == test_season

        if test_mask.sum() == 0:
            continue

        train_oof = oof_df[train_mask]
        test_oof = oof_df[test_mask]

        # Fit meta model
        meta_features = [c for c in train_oof.columns if c.endswith('_pred')]
        X_train = train_oof[meta_features].values
        y_train = train_oof['Target'].values

        meta = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        meta.fit(X_train, y_train)

        X_test = test_oof[meta_features].values
        raw_preds = meta.predict_proba(X_test)[:, 1]
        raw_preds = np.clip(raw_preds, 0.001, 0.999)

        for i, (idx, row) in enumerate(test_oof.iterrows()):
            all_preds.append({
                'Season': row['Season'],
                'Target': row['Target'],
                'Round': matchups.loc[idx, 'Round'] if idx in matchups.index else np.nan,
                'raw_pred': raw_preds[i],
            })

    pred_df = pd.DataFrame(all_preds)
    pred_df = pred_df.dropna(subset=['Round'])
    pred_df['Round'] = pred_df['Round'].astype(int)

    # Optimize alpha per round
    results = []
    best_alphas = {}

    for rnd in sorted(pred_df['Round'].unique()):
        rnd_df = pred_df[pred_df['Round'] == rnd]
        if len(rnd_df) < 5:
            best_alphas[rnd] = 1.0
            continue

        best_brier = float('inf')
        best_alpha = 1.0

        for alpha in alpha_grid:
            stretched = stretch_preds(rnd_df['raw_pred'].values, alpha)
            b = brier_score(rnd_df['Target'].values, stretched)
            results.append({'Round': rnd, 'alpha': alpha, 'brier': b, 'n_games': len(rnd_df)})
            if b < best_brier:
                best_brier = b
                best_alpha = alpha

        best_alphas[rnd] = round(best_alpha, 1)

    results_df = pd.DataFrame(results)

    return best_alphas, results_df


def apply_round_alphas(submission_df, seeds_df, seed_round_slots_df,
                       round_alphas, default_alpha=1.0):
    """Apply round-specific alpha stretching to a submission.

    Args:
        submission_df: DataFrame with ID, Pred columns
        seeds_df: MNCAATourneySeeds or WNCAATourneySeeds
        seed_round_slots_df: MNCAATourneySeedRoundSlots
        round_alphas: dict {round_num: alpha} from optimize_round_alphas
        default_alpha: alpha for non-tournament matchups (default 1.0 = no change)

    Returns:
        Modified submission DataFrame with stretched predictions
    """
    df = submission_df.copy()
    rounds = assign_rounds_to_matchups(df, seeds_df, seed_round_slots_df)
    df['_round'] = rounds

    for rnd, alpha in round_alphas.items():
        if alpha == 1.0:
            continue
        mask = df['_round'] == rnd
        if mask.sum() > 0:
            df.loc[mask, 'Pred'] = stretch_preds(df.loc[mask, 'Pred'].values, alpha)

    # Apply default alpha to non-tournament matchups
    if default_alpha != 1.0:
        mask = df['_round'].isna()
        if mask.sum() > 0:
            df.loc[mask, 'Pred'] = stretch_preds(df.loc[mask, 'Pred'].values, default_alpha)

    df = df.drop(columns=['_round'])
    return df


ROUND_NAMES = {
    0: 'Play-in',
    1: 'Round 1 (64->32)',
    2: 'Round 2 (32->16)',
    3: 'Sweet 16',
    4: 'Elite Eight',
    5: 'Final Four',
    6: 'Championship',
}


def print_round_alphas(alphas, results_df=None):
    """Pretty-print round-specific alpha results."""
    print(f"\n{'Round':<25s} {'Alpha':>6s} {'Brier':>8s} {'N games':>8s}")
    print("-" * 50)
    for rnd in sorted(alphas.keys()):
        name = ROUND_NAMES.get(rnd, f'Round {rnd}')
        alpha = alphas[rnd]
        if results_df is not None:
            best = results_df[(results_df['Round'] == rnd) &
                             (results_df['alpha'] == alpha)]
            if len(best) > 0:
                brier = best['brier'].values[0]
                n = best['n_games'].values[0]
                print(f"{name:<25s} {alpha:>6.1f} {brier:>8.5f} {n:>8d}")
                continue
        print(f"{name:<25s} {alpha:>6.1f}")
