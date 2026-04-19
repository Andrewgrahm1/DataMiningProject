# Linear SVM Progress Summary

## Goal
Use the existing Orion data pipeline and add a `LinearSVC` baseline to test how a linear margin-based model performs under the same data, feature, label, and split setup as the current group experiments.

## What was reused
The following parts were reused from the existing Orion workflow:

- data fetching and CSV caching
- market-hours cleaning and gap forward-filling
- feature construction
- target generation
- chronological train / validation / test split
- z-score normalization

This kept the setup directly comparable to the existing baselines.

## What was added
A new `LinearSVC` experiment was added on top of the existing pipeline.

Main changes:
- created a personal experiment entry point
- reused the same `load_*_training_frames(...)` structure
- trained a `LinearSVC` model
- evaluated the model with a custom trade-only evaluation function

## Why a custom evaluation was added
The original evaluation function treated all confusion-matrix outcomes as profit/loss contributions, including true negatives.  
That makes the printed “Profit” more like a classification-based approximation than a realized trading result.

To better reflect trading behavior, a new evaluation function was added that only counts trades actually taken:

- TP: counted as a winning trade
- FP: counted as a losing trade
- TN / FN: not counted in realized profit

## Data split used
The experiment used the same chronological split as Orion:

- **Train:** 253,606 rows, 26,297 positives (10.37%), 2022-01-03 → 2024-08-06
- **Validation:** 58,524 rows, 3,481 positives (5.95%), 2024-08-06 → 2025-03-14
- **Test:** 78,032 rows, 4,458 positives (5.71%), 2025-03-14 → 2025-12-30

## Tuning process
A simple validation-based tuning process was used.

Parameters explored:
- `C`
- `class_weight`
- decision threshold

The model was trained on the training set, candidate settings were compared on the validation set, and the final selected setup was evaluated once on the held-out test set.

## Best setup found so far
- `C = 0.01`
- `class_weight = None`
- `threshold = 0.5`

## Test result summary
Final `LinearSVC` test result:

- **Recall:** 122 / 4,458 = 2.74%
- **Precision:** 122 / 228 = 53.51%
- **Accuracy:** 73,590 / 78,032 = 94.31%
- **Trades Taken:** 228
- **Realized Profit:** -13.60%
- **Max Possible Profit:** 8,024.40%
- **Profit Capture vs Max Possible:** -0.17%

## Interpretation
The tuned `LinearSVC` became much more conservative than the earlier version.

Compared with the earlier run:
- far fewer trades were taken
- precision improved substantially
- realized profit moved much closer to break-even

However, under the current setup:
- take profit = 2%
- stop loss = 2%
- cost = 0.2%

the break-even win rate is roughly 55%, while the model achieved 53.51% precision, so the final realized profit is still slightly negative.

## Current conclusion
The `LinearSVC` baseline is now fully integrated into the existing pipeline and produces a meaningful result under the same experiment setup as the other models.

It is not profitable yet, but:
- it is much more stable than the initial version
- it is close to break-even
- it provides a useful linear baseline for comparison with other models

## Next possible steps
- search smaller `C` values
- search thresholds more finely around the current best setting
- require a minimum number of validation trades during model selection
- compare against Logistic Regression
- test whether a lightweight nonlinear approximation is more scalable than full RBF SVM

# Logistic Regression Progress Summary

## Goal
Use the existing Orion data pipeline and add a `LogisticRegression` baseline to compare a probabilistic linear model under the same data, feature, label, and split setup as the other group experiments.

## What was reused
The following parts were reused from the existing Orion workflow:

- data fetching and CSV caching
- market-hours cleaning and minute-bar gap forward-filling
- feature construction
- target generation
- chronological train / validation / test split
- z-score normalization

This kept the Logistic Regression experiment directly comparable to the other baselines.

## What was added
A new `LogisticRegression` experiment was added on top of the reused pipeline.

Main additions:
- created a personal Logistic Regression experiment entry point
- reused the same `load_*_training_frames(...)` structure
- trained `LogisticRegression` with validation-based selection
- used the custom trade-only evaluation function for realized profit

## Why a custom evaluation was used
The original evaluation function in the project treated all confusion-matrix outcomes as profit/loss contributions, including true negatives.  
That makes the printed “Profit” more like a classification-based approximation than a realized trading result.

To better reflect actual trading behavior, the custom trade-only evaluation only counts trades actually taken:

- TP: counted as a winning trade
- FP: counted as a losing trade
- TN / FN: not counted in realized profit

## Data split used
The Logistic Regression experiment used the same chronological split:

- **Train:** 253,606 rows, 26,297 positives (10.37%), 2022-01-03 → 2024-08-06
- **Validation:** 58,524 rows, 3,481 positives (5.95%), 2024-08-06 → 2025-03-14
- **Test:** 78,032 rows, 4,458 positives (5.71%), 2025-03-14 → 2025-12-30

## Tuning process
A simple validation-based tuning process was used.

Parameters explored:
- `C`
- `class_weight`
- probability threshold

The model was trained on the training set, candidate settings were compared on the validation set, and the final selected setup was evaluated once on the held-out test set.

## Best setup found so far
- `C = 0.1`
- `class_weight = None`
- `threshold = 0.7`

## Test result summary
Final `LogisticRegression` test result:

- **Recall:** 460 / 4,458 = 10.32%
- **Precision:** 460 / 1,308 = 35.17%
- **Accuracy:** 73,186 / 78,032 = 93.79%
- **Trades Taken:** 1,308
- **Realized Profit:** -1,037.60%
- **Max Possible Profit:** 8,024.40%
- **Profit Capture vs Max Possible:** -12.93%

## Interpretation
The tuned Logistic Regression model was less conservative than the tuned LinearSVC.

Compared with the tuned LinearSVC:
- it took many more trades
- recall was higher
- precision was much lower
- realized profit was substantially worse

Under the current setup:
- take profit = 2%
- stop loss = 2%
- cost = 0.2%

the break-even win rate is roughly 55%, while Logistic Regression achieved only 35.17% precision, so the final realized profit remained clearly negative.

The chosen threshold of 0.7 shows that the validation process already pushed the model toward a more conservative decision rule, but even with that stricter threshold, trade quality was still not high enough.

## Current conclusion
The Logistic Regression baseline is fully integrated into the same Orion-based experiment pipeline and provides a comparable linear baseline.

However, in its current form:
- it trades too often relative to its win rate
- it remains far below break-even precision
- it performs worse than the tuned LinearSVC under the current trade-only profit definition

## Next possible steps
- search higher thresholds more finely above 0.7
- test smaller or larger `C` values around 0.1
- require a minimum or bounded number of validation trades during selection
- compare directly against the tuned LinearSVC as the stronger linear baseline
- consider feature reduction or stricter filtering to improve trade quality