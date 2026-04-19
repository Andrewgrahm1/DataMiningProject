# Progress report — March 20, 2026

## Project state

We have an initial **feature-engineering pipeline** (e.g. time-to-close / bars-since-open, batched percent-change features, binary trade targets from take-profit / stop-loss rules) and shared helpers under `lib/common`. On top of that, there is a repeatable workflow to **build training tables, hold out a chronological test split, train classifiers, and compare** them with the same evaluation metrics (recall on profitable opportunities, precision-style win rate on predicted entries, profit framing vs. a theoretical max).

The **`experiments/`** tree is organized **one subfolder per group member** so each person can run and extend their own scripts without stepping on others. There are currently **four** such folders: `andrew`, `orion`, `trace`, and `zhongjie` (each holds a copy of the latest state for feature engineering and model training, run experiment.py to pull the data, train, and test the models.).

## Current model results (preliminary)

The numbers below come from a single run of the shared experiment script (cached minute bars, same features and split). **Symbol:** MARA. **Date range:** 2022-01-01 through 2025-12-31. **Take profit:** 4%. **Stop loss:** 2%. **Trade cost:** 0.4%. **Break-even precision** (minimum win rate on taken trades to cover costs): **40%**.

**Data split:** 312,130 train rows / 78,032 test rows (last 20% chronological).

| Model | Profitable trades taken (recall) | Win rate (precision) | Loss rate | Profit vs. max possible* |
|-------|----------------------------------|----------------------|-----------|---------------------------|
| Decision Tree | 1,034 / 5,249 (19.70%) | 1,034 / 8,903 (11.61%) | 88.39% | −15,163.20% / 3,722.40% |
| Naive Bayes | 1,609 / 5,249 (30.65%) | 1,609 / 8,841 (18.20%) | 81.80% | −11,564.40% / 5,792.40% |
| K-Nearest Neighbors | 210 / 5,249 (4.00%) | 210 / 1,421 (14.78%) | 85.22% | −2,150.40% / 756.00% |
| Random Forest | 39 / 5,249 (0.74%) | 39 / 323 (12.07%) | 87.93% | −541.20% / 140.40% |
| AdaBoost | 4,658 / 5,249 (88.74%) | 4,658 / 42,934 (10.85%) | 89.15% | −75,093.60% / 16,768.80% |
| XGBoost | 3,827 / 5,249 (72.91%) | 3,827 / 29,654 (12.91%) | 87.09% | −48,207.60% / 13,777.20% |

\*Reporting convention matches the experiment’s printed “Profit: …% of max possible …%” lines.

**Takeaway:** Under this first-pass setup, **no model clears the 40% break-even precision bar**; several models trade too often (high recall, low precision) or too rarely (e.g. Random Forest). This establishes a **baseline** for iteration—tuning hyperparameters, thresholding predicted probabilities, revisiting features, or class imbalance handling next.
