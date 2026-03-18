# Data

This folder holds the raw dataset. The CSV is excluded from version control.

## Dataset: Give Me Some Credit

Source: [Kaggle — Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)

### Download via Kaggle CLI

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip -d data/
```

### Or download manually

1. Go to https://www.kaggle.com/competitions/GiveMeSomeCredit/data
2. Download `cs-training.csv`
3. Place it in this `data/` folder

### Features

| Column | Description |
|--------|-------------|
| `SeriousDlqin2yrs` | **Target** — 1 if 90+ days past due in 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Credit card utilization ratio |
| `age` | Age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30–59 days late |
| `DebtRatio` | Monthly debt / monthly income |
| `MonthlyIncome` | Monthly income (has ~19% missing) |
| `NumberOfOpenCreditLinesAndLoans` | Open credit lines |
| `NumberOfTimes90DaysLate` | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Mortgage/real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60–89 days late |
| `NumberOfDependents` | Dependents (has ~2.5% missing) |

**Class balance:** ~6.7% positive (default) — imbalanced dataset.
