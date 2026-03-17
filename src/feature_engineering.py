import pandas as pd
import math

def compute_elo_ratings(games_df, k=32, home_adv=100, carryover=0.75) -> pd.DataFrame:
    '''
    Compute Elo Ratings from game level results

    Args:
        games_df: DataFrame from M/WRegularSeasonCompactResults.csv
        k: base K-factor controlling update magnitude
        home_adv: Elo points added to home team's rating for expected score calculation
        carryover: fraction of deviation from 1500 retained between seasons (0.75 = regress 25%)

    Returns:
        DataFrame with columns [season, team_id, elo] - end of regular season at the team level
    '''

    elos = {}
    results = []

    # Only regular season games
    reg = games_df[games_df['DayNum'] <= 132].sort_values(by=['Season', 'DayNum'])

    current_season = None

    for row in reg.itertuples(index=False):
        season = row.Season
        w_id = row.WTeamID
        l_id = row.LTeamID

        # Season transition
        if season != current_season:
            if current_season is not None:
                # Snapshot end of season ratings before regressing to 1500
                for team_id, elo in elos.items():
                    results.append((current_season, team_id, elo))
            # Regress
            for team in elos:
                elos[team] = 1500 + carryover * (elos[team] - 1500)
            current_season = season
        
        w_elo = elos.get(w_id, 1500)
        l_elo = elos.get(l_id, 1500)

        # Home court adjustment (only for expected score, not stored)
        w_elo_adj = w_elo
        l_elo_adj = l_elo
        if row.WLoc == 'H':
            w_elo_adj += home_adv
        if row.WLoc == 'A':
            l_elo_adj += home_adv
        
        # Expected win probability for the winner
        expected_w = 1.0 / (1.0 + 10 ** ((l_elo_adj - w_elo_adj)/ 400))

        # Margin of victory multiplier
        score_diff = row.WScore - row.LScore
        elo_diff = abs(w_elo - l_elo)
        mov_mult = math.log(abs(score_diff) + 1) * (2.2 / (2.2 + 0.001 * elo_diff))

        # Update
        update = k * mov_mult * (1-expected_w)
        elos[w_id] = w_elo + update
        elos[l_id] = l_elo - update

    # Snapshot the final season
    if current_season is not None:
        for team_id, elo in elos.items():
            results.append((current_season, team_id, elo))
        
    return pd.DataFrame(results, columns = ['Season', 'TeamID', 'Elo'])

def elo_to_prob(elo_a, elo_b):
    # P(team A beats team B) based on elo difference
    return 1.0/ (1.0 + 10 ** ((elo_b - elo_a) / 400))

def parse_seeds(seeds_df):
    '''
    Parse tournament seed strings into numeric values

    Args:
        seeds_df: DataFrame with columns [Season, Seed, TeamID]

    Returns:
        DataFrame with columns [Season, TeamID, SeedNum]
        SeedNum is an int 1-16
    '''
    df = seeds_df.copy()
    df['SeedNum'] = df['Seed'].str[1:3].astype(int)
    return df[['Season','TeamID', 'SeedNum']]

def compute_season_stats(detail_df) -> pd.DataFrame:
    '''
    Compute per-team per-season aggregated stats from detailed box scores.

    Computes offensive/defensive Four Fators, tempo, scoring and win percentage.
    Four Factors are computed at game-levelfirst then averaged across the season
    Args:
        detail_df: DataFrame from M/WRegularSeasonDetailedResults.csv

    Returns:
        DataFrame with columns [Season, TeamID, eFG_off, TO_rate_off, OR_pct,
        FT_rate_off, eFG_def, TO_rate_def, DR_pct, FT_rate_def, Tempo, PPG,
        PPG_allowed, Win_pct, Last10_Win_pct]
    '''

    # Filter for regular season
    reg = detail_df[detail_df['DayNum'] <=132].copy()

    # Build winner rows (team = winner, opponent = loser)
    w_rows = pd.DataFrame({
        'Season': reg['Season'],
        'DayNum': reg['DayNum'],
        'TeamID': reg['WTeamID'],
        'Win': 1,
        'Points': reg['WScore'],
        'OppPoints': reg['LScore'],
        'FGM': reg['WFGM'],
        'FGA': reg['WFGA'],
        'FGM3': reg['WFGM3'],
        'FTM': reg['WFTM'],
        'FTA': reg['WFTA'],
        'OR': reg['WOR'],
        'DR': reg['WDR'],
        'TO': reg['WTO'],
        'Opp_FGM': reg['LFGM'],
        'Opp_FGA': reg['LFGA'],
        'Opp_FGM3': reg['LFGM3'],
        'Opp_FTM': reg['LFTM'],
        'Opp_FTA': reg['LFTA'],
        'Opp_OR': reg['LOR'],
        'Opp_DR': reg['LDR'],
        'Opp_TO': reg['LTO']
    })

    # Build loser rows (team = loser, opponent = winner)
    l_rows = pd.DataFrame({
        'Season': reg['Season'],
        'DayNum': reg['DayNum'],
        'TeamID': reg['LTeamID'],
        'Win': 0,
        'Points': reg['LScore'],
        'OppPoints': reg['WScore'],
        'FGM': reg['LFGM'],
        'FGA': reg['LFGA'],
        'FGM3': reg['LFGM3'],
        'FTM': reg['LFTM'],
        'FTA': reg['LFTA'],
        'OR': reg['LOR'],
        'DR': reg['LDR'],
        'TO': reg['LTO'],
        'Opp_FGM': reg['WFGM'],
        'Opp_FGA': reg['WFGA'],
        'Opp_FGM3': reg['WFGM3'],
        'Opp_FTM': reg['WFTM'],
        'Opp_FTA': reg['WFTA'],
        'Opp_OR': reg['WOR'],
        'Opp_DR': reg['WDR'],
        'Opp_TO': reg['WTO']
    })

    games = pd.concat([w_rows, l_rows], ignore_index=True)

    # Compute per game Four Factors and tempo
    games['eFG_off'] = (games['FGM'] + 0.5 * games['FGM3']) / games['FGA']
    games['TO_rate_off'] = games['TO'] / (games['FGA'] + 0.44 * games['FTA'] + games['TO'])
    games['OR_pct'] = games['OR'] / (games['OR'] + games['Opp_DR'])
    games['FT_rate_off'] = games['FTM'] / games['FGA']

    games['eFG_def'] = (games['Opp_FGM'] + 0.5 * games['Opp_FGM3']) / games['Opp_FGA']
    games['TO_rate_def'] = games['Opp_TO'] / (games['Opp_FGA'] + 0.44*games['Opp_FTA'] + games['Opp_TO'])
    games['DR_pct'] = games['DR'] / (games['DR'] + games['Opp_OR'])
    games['FT_rate_def'] = games['Opp_FTM'] / games['Opp_FGA']

    games['Tempo'] = games['FGA'] - games['OR'] + games['TO'] + 0.44 * games['FTA']

    # Season-level agg
    agg = games.groupby(['Season', 'TeamID']).agg(
        eFG_off=('eFG_off', 'mean'),
        TO_rate_off=('TO_rate_off', 'mean'),
        OR_pct=('OR_pct', 'mean'),
        FT_rate_off = ('FT_rate_off', 'mean'),
        eFG_def=('eFG_def', 'mean'),
        TO_rate_def=('TO_rate_def', 'mean'),
        DR_pct=('DR_pct', 'mean'),
        FT_rate_def=('FT_rate_def', 'mean'),
        Tempo=('Tempo', 'mean'),
        PPG=('Points', 'mean'),
        PPG_allowed=('OppPoints', 'mean'),
        Win_pct=('Win', 'mean'),
        TotalGames=('Win', 'count'),
    ).reset_index()

    games_sorted = games.sort_values(['Season', 'TeamID', 'DayNum'])
    last10 = games_sorted.groupby(['Season', 'TeamID']).tail(10)
    last10_wp =last10.groupby(['Season', 'TeamID'])['Win'].mean().reset_index()
    last10_wp.columns=['Season', 'TeamID', 'Last10_Win_pct']

    # Merge and select final columns
    result = agg.merge(last10_wp, on=['Season', 'TeamID'], how='left')

    return result[['Season', 'TeamID', 'eFG_off', 'TO_rate_off', 'OR_pct', 'FT_rate_off',
                    'eFG_def', 'TO_rate_def', 'DR_pct', 'FT_rate_def',
                    'Tempo', 'PPG', 'PPG_allowed', 'Win_pct', 'Last10_Win_pct']]

def compute_efficiency(detail_df) -> pd.DataFrame:
    '''
    Compute tempo-adjusted efficiency metrics from detailed box scores.

    Offensive efficiency = points scored per 100 possessions
    Defensive efficiency = points allowed per 100 possessions
    Net efficiency = offensive - defensive (the key predictive metric)

    Possessions estimated as: FGA - OR + TO + 0.44 * FTA
    (same formula used in compute_season_stats for Tempo)

    Args:
        detail_df: DataFrame from M/WRegularSeasonDetailedResults.csv

    Returns:
        DataFrame with columns [Season, TeamID, Off_Eff, Def_Eff, Net_Eff]
    '''
    reg = detail_df[detail_df['DayNum'] <= 132].copy()

    # Winner rows
    w_rows = pd.DataFrame({
        'Season': reg['Season'],
        'TeamID': reg['WTeamID'],
        'Points': reg['WScore'],
        'OppPoints': reg['LScore'],
        'FGA': reg['WFGA'],
        'OR': reg['WOR'],
        'TO': reg['WTO'],
        'FTA': reg['WFTA'],
        'Opp_FGA': reg['LFGA'],
        'Opp_OR': reg['LOR'],
        'Opp_TO': reg['LTO'],
        'Opp_FTA': reg['LFTA'],
    })

    # Loser rows
    l_rows = pd.DataFrame({
        'Season': reg['Season'],
        'TeamID': reg['LTeamID'],
        'Points': reg['LScore'],
        'OppPoints': reg['WScore'],
        'FGA': reg['LFGA'],
        'OR': reg['LOR'],
        'TO': reg['LTO'],
        'FTA': reg['LFTA'],
        'Opp_FGA': reg['WFGA'],
        'Opp_OR': reg['WOR'],
        'Opp_TO': reg['WTO'],
        'Opp_FTA': reg['WFTA'],
    })

    games = pd.concat([w_rows, l_rows], ignore_index=True)

    # Possessions (team's offensive possessions)
    games['Poss'] = games['FGA'] - games['OR'] + games['TO'] + 0.44 * games['FTA']
    # Opponent possessions (team's defensive possessions)
    games['Opp_Poss'] = games['Opp_FGA'] - games['Opp_OR'] + games['Opp_TO'] + 0.44 * games['Opp_FTA']

    # Per-game efficiency (per 100 possessions)
    games['Off_Eff'] = 100 * games['Points'] / games['Poss']
    games['Def_Eff'] = 100 * games['OppPoints'] / games['Opp_Poss']

    # Season averages
    agg = games.groupby(['Season', 'TeamID']).agg(
        Off_Eff=('Off_Eff', 'mean'),
        Def_Eff=('Def_Eff', 'mean'),
    ).reset_index()

    agg['Net_Eff'] = agg['Off_Eff'] - agg['Def_Eff']

    return agg[['Season', 'TeamID', 'Off_Eff', 'Def_Eff', 'Net_Eff']]


def compute_massey_features(ordinals_df, systems=None) -> pd.DataFrame:
    '''
    Compute aggregated ranking features from Massey Ordinals (Men only)

    Uses end-of-regular season rankings (RankingDayNum = 133) when available,
    otherwise falls back to the latest available RankingDayNum per season.

    Args:
        ordinals_df: DataFrame from MMasseyOrdinals.csv
        systems: list of specific system names to extract as individual features

    Returns:
        DataFrame with columns
    '''
    if systems is None:
        systems = {'POM', 'SAG', 'RPI'}

    # Use DayNum 133 where available; for seasons without it, use max available
    max_day_per_season = ordinals_df.groupby('Season')['RankingDayNum'].max().reset_index()
    max_day_per_season.columns = ['Season', 'MaxDay']
    max_day_per_season['UseDay'] = max_day_per_season['MaxDay'].clip(upper=133)
    merged = ordinals_df.merge(max_day_per_season[['Season', 'UseDay']], on='Season', how='left')
    eos = merged[merged['RankingDayNum'] == merged['UseDay']].drop(columns=['UseDay']).copy()

    # Aggregate stats across systems
    agg = eos.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(
        Massey_median = 'median',
        Massey_mean = 'mean', 
        Massey_min='min',
        Massey_std='std'
    ).reset_index()

    # Extract individual system rankings
    for sys_name in systems:
        sys_df = eos[eos['SystemName'] == sys_name][['Season', 'TeamID', 'OrdinalRank']].copy()
        sys_df = sys_df.rename(columns={'OrdinalRank': f'Rank_{sys_name}'})
        agg = agg.merge(sys_df, on=['Season', 'TeamID'], how='left')

    return agg

def compute_conference_strength(conf_df, elo_df) -> pd.DataFrame:
    '''
    Compute conference strength as the average Elo of all teams in each conference

    Args:
        conf_df: DataFrame from M/WTeamConference.csv with [Season, TeamID, ConfAbbrev]

    Returns:
        DataFrame with columns [Season, TeamID, ConfAbbrev, Conf_Elo_mean]
    '''

    # Merge team conferences with their Elo ratings
    merged = conf_df.merge(elo_df, on=['Season', 'TeamID'], how='left')

    # Average Elo per conference per season
    conf_strength = merged.groupby(['Season', 'ConfAbbrev'])['Elo'].mean().reset_index()
    conf_strength = conf_strength.rename(columns={'Elo': 'Conf_Elo_mean'})

    # Map back to each team
    result = conf_df.merge(conf_strength, on=['Season', 'ConfAbbrev'], how='left')

    return result[['Season', 'TeamID', 'ConfAbbrev', 'Conf_Elo_mean']]


def compute_advanced_stats(detail_df) -> pd.DataFrame:
    '''
    Compute additional per-team per-season stats not in compute_season_stats.

    New features:
    - FG3_pct: 3-point field goal percentage (FGM3 / FGA3)
    - FG3_rate: 3-point attempt rate (FGA3 / FGA)
    - Ast_TO_ratio: assist-to-turnover ratio
    - Stl_rate: steals per possession
    - Blk_rate: blocks per opponent FGA
    - Reb_margin: total rebounds - opponent total rebounds per game
    - Scoring_margin: PPG - PPG_allowed (point differential)

    Args:
        detail_df: DataFrame from M/WRegularSeasonDetailedResults.csv

    Returns:
        DataFrame with [Season, TeamID, FG3_pct, FG3_rate, Ast_TO_ratio,
                        Stl_rate, Blk_rate, Reb_margin, Scoring_margin]
    '''
    reg = detail_df[detail_df['DayNum'] <= 132].copy()

    w_rows = pd.DataFrame({
        'Season': reg['Season'],
        'TeamID': reg['WTeamID'],
        'FGM3': reg['WFGM3'], 'FGA3': reg['WFGA3'],
        'FGA': reg['WFGA'],
        'Ast': reg['WAst'], 'TO': reg['WTO'],
        'Stl': reg['WStl'], 'Blk': reg['WBlk'],
        'OR': reg['WOR'], 'DR': reg['WDR'],
        'Opp_OR': reg['LOR'], 'Opp_DR': reg['LDR'],
        'Opp_FGA': reg['LFGA'], 'Opp_FTA': reg['LFTA'], 'Opp_TO': reg['LTO'],
        'FTA': reg['WFTA'],
        'Points': reg['WScore'], 'OppPoints': reg['LScore'],
    })
    l_rows = pd.DataFrame({
        'Season': reg['Season'],
        'TeamID': reg['LTeamID'],
        'FGM3': reg['LFGM3'], 'FGA3': reg['LFGA3'],
        'FGA': reg['LFGA'],
        'Ast': reg['LAst'], 'TO': reg['LTO'],
        'Stl': reg['LStl'], 'Blk': reg['LBlk'],
        'OR': reg['LOR'], 'DR': reg['LDR'],
        'Opp_OR': reg['WOR'], 'Opp_DR': reg['WDR'],
        'Opp_FGA': reg['WFGA'], 'Opp_FTA': reg['WFTA'], 'Opp_TO': reg['WTO'],
        'FTA': reg['LFTA'],
        'Points': reg['LScore'], 'OppPoints': reg['WScore'],
    })

    games = pd.concat([w_rows, l_rows], ignore_index=True)

    # Per-game metrics
    games['FG3_pct'] = games['FGM3'] / games['FGA3'].replace(0, float('nan'))
    games['FG3_rate'] = games['FGA3'] / games['FGA']
    games['Ast_TO_ratio'] = games['Ast'] / games['TO'].replace(0, float('nan'))
    poss = games['FGA'] - games['OR'] + games['TO'] + 0.44 * games['FTA']
    games['Stl_rate'] = games['Stl'] / poss.replace(0, float('nan'))
    games['Blk_rate'] = games['Blk'] / games['Opp_FGA'].replace(0, float('nan'))
    games['Reb_margin'] = (games['OR'] + games['DR']) - (games['Opp_OR'] + games['Opp_DR'])
    games['Scoring_margin'] = games['Points'] - games['OppPoints']

    agg = games.groupby(['Season', 'TeamID']).agg(
        FG3_pct=('FG3_pct', 'mean'),
        FG3_rate=('FG3_rate', 'mean'),
        Ast_TO_ratio=('Ast_TO_ratio', 'mean'),
        Stl_rate=('Stl_rate', 'mean'),
        Blk_rate=('Blk_rate', 'mean'),
        Reb_margin=('Reb_margin', 'mean'),
        Scoring_margin=('Scoring_margin', 'mean'),
    ).reset_index()

    return agg


def build_team_features(elo_df, seeds_df=None, stats_df=None, massey_df=None,
                        conf_df=None, efficiency_df=None, advanced_df=None) -> pd.DataFrame:
    '''
    Merge all per-team per-season features into a single DataFrame.

    Args:
        elo_df: DataFrame from compute_elo_ratings() with [Season, TeamID, Elo]
        seeds_df: DataFrame from parse_seeds() with [Season, TeamID, SeedNum] (optional)
        stats_df: DataFrame from compute_season_stats() (optional)
        massey_df: DataFrame from compute_massey_features() (optional, men only)
        conf_df: DataFrame from compute_conference_strength() (optional)
        efficiency_df: DataFrame from compute_efficiency() (optional)

    Returns:
        DataFrame with [Season, TeamID, ...all numeric features]
    '''
    features = elo_df.copy()

    if seeds_df is not None:
        features = features.merge(seeds_df, on=['Season', 'TeamID'], how='left')

    if stats_df is not None:
        features = features.merge(stats_df, on=['Season', 'TeamID'], how='left')

    if massey_df is not None:
        features = features.merge(massey_df, on=['Season', 'TeamID'], how='left')

    if conf_df is not None:
        conf_numeric = conf_df[['Season', 'TeamID', 'Conf_Elo_mean']]
        features = features.merge(conf_numeric, on=['Season', 'TeamID'], how='left')

    if efficiency_df is not None:
        features = features.merge(efficiency_df, on=['Season', 'TeamID'], how='left')

    if advanced_df is not None:
        features = features.merge(advanced_df, on=['Season', 'TeamID'], how='left')

    return features


def build_matchup_matrix(tourney_df, team_features) -> pd.DataFrame:
    '''
    Build training feature matrix from historical tournament games.

    For each game, TeamA = lower ID, TeamB = higher ID (matching submission format).
    Features are differences (TeamA - TeamB) for all numeric features.
    Target: 1 if TeamA (lower ID) won, 0 otherwise.

    Args:
        tourney_df: DataFrame from M/WNCAATourneyCompactResults.csv
        team_features: DataFrame from build_team_features()

    Returns:
        DataFrame with [Season, TeamA, TeamB, <feature>_diff columns..., Target]
    '''
    games = tourney_df.copy()

    # TeamA = lower ID, TeamB = higher ID
    games['TeamA'] = games[['WTeamID', 'LTeamID']].min(axis=1)
    games['TeamB'] = games[['WTeamID', 'LTeamID']].max(axis=1)
    games['Target'] = (games['WTeamID'] == games['TeamA']).astype(int)

    # Identify numeric feature columns
    non_feature_cols = {'Season', 'TeamID', 'ConfAbbrev'}
    feature_cols = [c for c in team_features.columns if c not in non_feature_cols]

    # Merge features for TeamA
    a_rename = {c: f'{c}_A' for c in feature_cols}
    a_feats = team_features.rename(columns=a_rename)
    games = games.merge(a_feats, left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'], how='left')
    games = games.drop(columns=['TeamID'], errors='ignore')

    # Merge features for TeamB
    b_rename = {c: f'{c}_B' for c in feature_cols}
    b_feats = team_features.rename(columns=b_rename)
    games = games.merge(b_feats, left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'], how='left')
    games = games.drop(columns=['TeamID'], errors='ignore')

    # Compute diffs (TeamA - TeamB)
    diff_cols = []
    for col in feature_cols:
        diff_name = f'{col}_diff'
        games[diff_name] = games[f'{col}_A'] - games[f'{col}_B']
        diff_cols.append(diff_name)

    return games[['Season', 'TeamA', 'TeamB'] + diff_cols + ['Target']]