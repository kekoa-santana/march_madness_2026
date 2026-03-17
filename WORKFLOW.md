# March Madness Pipeline Workflow

## Quick Start

```bash
# Check data readiness
/c/Users/kekoa/anaconda3/python.exe -u scripts/preflight.py

# Run everything (submissions + brackets)
/c/Users/kekoa/anaconda3/python.exe -u scripts/preflight.py --run-all
```

## Timeline

| Date | Event | Action |
|------|-------|--------|
| Now - March 15 | Regular season | Can generate O submission (all-matchup predictions, no seeds needed) |
| March 15 | Selection Sunday | Download updated data with 2026 seeds |
| March 15 - 19 | Seeds known | Generate round-alpha submission + brackets |
| March 19 4PM UTC | **Deadline** | Final submission must be uploaded to Kaggle |
| March 19 - April 6 | Tournament | Watch results play out on leaderboard |

## Step-by-Step

### 1. Download Fresh Data

Download the latest data drop from Kaggle into `./data/`. Kaggle will release at least one update before the deadline.

### 2. Run Preflight Check

```bash
/c/Users/kekoa/anaconda3/python.exe -u scripts/preflight.py
```

This validates:
- All 14 core CSV files exist and are non-empty
- `SampleSubmissionStage2.csv` is present with 2026 matchups
- 2026 regular season games are in the data
- 2026 Massey ordinals are present
- Whether 2026 seeds are available (pre vs post Selection Sunday)
- Bracket structure files and round alphas cache

### 3. Generate Submissions

**Pre-Selection Sunday** (available now):
```bash
/c/Users/kekoa/anaconda3/python.exe -u scripts/gen_submission_O.py
# -> submissions/stage2_O_loso_tuned.csv
```
This predicts every possible 2026 matchup. No seeds needed. Uses the O config (current best Kaggle score: 0.14502).

**Post-Selection Sunday** (after March 15):
```bash
/c/Users/kekoa/anaconda3/python.exe -u scripts/gen_submission_final.py --skip-optimize
# -> submissions/stage2_final_round_alpha.csv  (main submission)
# -> submissions/stage2_O_baseline.csv         (comparison)
```
Applies round-specific alpha stretching based on bracket structure. Requires 2026 seeds in the data.

### 4. Generate Brackets (Post-Selection Sunday)

```bash
# Chalk bracket (model favorite every game)
/c/Users/kekoa/anaconda3/python.exe -u scripts/bracket.py --gender both --season 2026 --mode chalk

# Monte Carlo simulations (advancement probabilities)
/c/Users/kekoa/anaconda3/python.exe -u scripts/bracket.py --gender men --mode montecarlo --sims 10000
/c/Users/kekoa/anaconda3/python.exe -u scripts/bracket.py --gender women --mode montecarlo --sims 10000

# Underdog betting opportunities (R1 upset probabilities)
/c/Users/kekoa/anaconda3/python.exe -u scripts/bracket.py --gender men --mode underdogs
/c/Users/kekoa/anaconda3/python.exe -u scripts/bracket.py --gender women --mode underdogs

# Upset-biased bracket (pick underdog when >35% chance)
/c/Users/kekoa/anaconda3/python.exe -u scripts/bracket.py --gender men --mode upset --threshold 0.35
```

### 5. Upload to Kaggle

1. Go to the competition submission page
2. Upload `submissions/stage2_final_round_alpha.csv` (primary)
3. Upload `submissions/stage2_O_loso_tuned.csv` (backup)
4. **Manually select** the two submissions you want scored — do not rely on auto-selection

## Or Just Run Everything

```bash
# Submissions + brackets in one command
/c/Users/kekoa/anaconda3/python.exe -u scripts/preflight.py --run-all

# Submissions only
/c/Users/kekoa/anaconda3/python.exe -u scripts/preflight.py --run-submission

# Brackets only
/c/Users/kekoa/anaconda3/python.exe -u scripts/preflight.py --run-bracket
```

## Pipeline Architecture

```
Fresh Kaggle data (./data/)
    │
    ├── data_loader.py          Load CSVs
    ├── model_elo_v2.py         Elo ratings (MOV + day-weighting)
    ├── feature_engineering.py  Season stats, seeds, Massey, efficiency
    ├── women_rankings_v1.py    Women's power ratings (WPR)
    ├── model_logreg_v2.py      Logistic regression base learner
    ├── model_stack_v1.py       XGB + CatBoost + LR meta stacking
    │
    ├── gen_submission_O.py     All-matchup submission (pre-Selection Sunday)
    ├── gen_submission_final.py Round-alpha submission (post-Selection Sunday)
    ├── round_alpha.py          Round-specific confidence calibration
    │
    └── bracket.py              Bracket simulator (chalk/upset/montecarlo/underdogs)
```

## What Automatically Updates with Fresh Data

- Elo ratings incorporate all 2026 regular season games
- Season stats (PPG, efficiency, win%) reflect full 2026 season
- Massey/Pomeroy rankings update with latest ordinals
- Conference strength recalculated from 2026 Elo ratings
- Women's power ratings (WPR) update with 2026 games

## What Does NOT Change

- Model hyperparameters (Elo K, LR C, feature sets) — already LOSO-tuned
- Training data for stacking: historical tournament results 2003-2025 (2026 tournament hasn't happened)
- Round alphas: cached in `round_alphas.json`, optimized on historical data
- `max_train_season=2025` is correct — we train on past tournaments, predict 2026 using 2026 features

## Kaggle Scores (Submission History)

| Submission | Kaggle Score |
|-----------|-------------|
| stage1_elo_baseline | 0.18056 |
| stage1_blended | 0.16567 |
| stage1_logreg | 0.16201 |
| stage1_stack_v1 (CatBoost) | 0.14826 |
| stage1_E_v2tuned | 0.14822 |
| stage1_F_feat_expansion | 0.14634 |
| stage1_G_c_tuned | 0.14566 |
| **stage1_J_women_alpha** | **0.14502** (current best) |
