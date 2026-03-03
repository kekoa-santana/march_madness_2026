import pandas as pd
import numpy as np
from pathlib import Path

def load_men_data(data_dir) -> dict[str, pd.DataFrame]:
    return {
        'MComSsn': pd.read_csv(f'{data_dir}/MRegularSeasonCompactResults.csv'), 
        'MDetSsn': pd.read_csv(f'{data_dir}/MRegularSeasonDetailedResults.csv'), 
        'MDetTrny': pd.read_csv(f'{data_dir}/MNCAATourneyDetailedResults.csv'), 
        'MTrnySeeds': pd.read_csv(f'{data_dir}/MNCAATourneySeeds.csv'),
        'MOrdinals': pd.read_csv(f'{data_dir}/MMasseyOrdinals.csv'),
        'MConf': pd.read_csv(f'{data_dir}/MTeamConferences.csv'),
        'MTeam': pd.read_csv(f'{data_dir}/MTeams.csv'),
        'MCoaches': pd.read_csv(f'{data_dir}/MTeamCoaches.csv'),
    }

def load_women_data(data_dir) -> dict[str, pd.DataFrame]:
    return {
        'WComSsn': pd.read_csv(f'{data_dir}/WRegularSeasonCompactResults.csv'), 
        'WDetSsn': pd.read_csv(f'{data_dir}/WRegularSeasonDetailedResults.csv'),
        'WDetTrny': pd.read_csv(f'{data_dir}/WNCAATourneyDetailedResults.csv'),
        'WTrnySeeds': pd.read_csv(f'{data_dir}/WNCAATourneySeeds.csv'),
        'WConf': pd.read_csv(f'{data_dir}/WTeamConferences.csv'),
        'WTeam': pd.read_csv(f'{data_dir}/WTeams.csv')
    }