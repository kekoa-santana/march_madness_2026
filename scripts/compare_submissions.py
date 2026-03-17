"""Compare submission CSVs against actual tournament results (2022-2025)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from src.data_loader import load_men_data, load_women_data
from src.model import brier_score

data_dir = './data'
m_data = load_men_data(data_dir)
w_data = load_women_data(data_dir)

# Build actual results lookup from tournament detailed results
def build_actuals(det_trny, is_women=False):
    """Build dict of (Season, TeamA, TeamB) -> outcome for TeamA."""
    results = {}
    for _, row in det_trny.iterrows():
        s = int(row['Season'])
        w = int(row['WTeamID'])
        l = int(row['LTeamID'])
        a, b = min(w, l), max(w, l)
        results[(s, a, b)] = 1.0 if w == a else 0.0
    return results

m_actuals = build_actuals(m_data['MDetTrny'])
w_actuals = build_actuals(w_data['WDetTrny'])
all_actuals = {**m_actuals, **w_actuals}

# Submissions to compare
submissions = {
    'stage1_logreg (0.15837)': './submissions/stage1_logreg.csv',
    'stage1_tuned_elo (new)': './submissions/stage1_tuned_elo.csv',
}

# Also check for other submissions
import os
for f in sorted(os.listdir('./submissions')):
    if f.endswith('.csv') and f.startswith('stage1'):
        path = f'./submissions/{f}'
        if path not in submissions.values():
            submissions[f] = path

print(f"{'Submission':<35s} {'Men':>8s} {'Women':>8s} {'Combined':>10s} {'N_men':>6s} {'N_women':>7s}")
print("-" * 80)

for label, path in sorted(submissions.items()):
    try:
        sub = pd.read_csv(path)
    except Exception as e:
        print(f"{label:<35s} ERROR: {e}")
        continue

    parts = sub['ID'].str.split('_', expand=True).astype(int)
    sub['Season'] = parts[0]
    sub['TeamA'] = parts[1]
    sub['TeamB'] = parts[2]

    # Only score rows where we have actuals (2022-2025 tourney games)
    sub['Actual'] = sub.apply(
        lambda r: all_actuals.get((r['Season'], r['TeamA'], r['TeamB']), np.nan), axis=1
    )
    scored = sub.dropna(subset=['Actual'])

    is_women = scored['TeamA'] >= 3000
    men = scored[~is_women]
    women = scored[is_women]

    if len(men) > 0:
        m_bs = brier_score(men['Actual'].values, men['Pred'].values)
    else:
        m_bs = np.nan
    if len(women) > 0:
        w_bs = brier_score(women['Actual'].values, women['Pred'].values)
    else:
        w_bs = np.nan

    combined = brier_score(scored['Actual'].values, scored['Pred'].values)

    print(f"{label:<35s} {m_bs:>8.5f} {w_bs:>8.5f} {combined:>10.5f} {len(men):>6d} {len(women):>7d}")
