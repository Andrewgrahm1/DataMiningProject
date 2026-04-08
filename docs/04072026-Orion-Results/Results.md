# Orion Results

I started with `experiments/orion/experiment.py`, where the goal was to predict whether the target would hit a 2% take-profit before a 2% stop-loss or a 90-bar time limit. Under that setup, the reported break-even win rate was 55%. None of the tested models cleared that threshold on precision. The Random Forest had the best overall accuracy at 61.58%, but its win rate on predicted trades was only 36.29%. XGBoost reached 59.43% accuracy with 39.58% precision, and the other models were weaker. In other words, the models were somewhat good at labeling rows overall, but they were not good enough at selecting profitable long trades.

| Model | Accuracy | Precision | Recall | Profit |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 60.28% | 42.66% | 60.27% | 3917.20% |
| Naive Bayes | 49.27% | 33.62% | 55.62% | -4250.80% |
| K-Nearest Neighbors | 55.88% | 31.85% | 29.93% | 653.20% |
| Random Forest | 61.58% | 36.29% | 22.23% | 4881.20% |
| XGBoost | 59.43% | 39.58% | 44.28% | 3289.20% |

One useful clue came from the follow-up profit simulation in the same script. When the Random Forest predictions were translated into a simple allocation rule that bought TQQQ on positive predictions and bought SQQQ otherwise, the test-set simulation showed a net simple return of 1113.25% across 18,554 rows. That result suggested the model might be more useful for identifying when not to be long than for identifying clean long entries. Because of that, I changed the framing of the problem. Instead of asking one model to predict both timing and direction, I split the task into two separate questions.

In `experiments/orion2/chgeIdnt.py`, I asked whether TQQQ would move by at least 2% in either direction within the next 90 bars. This problem turned out to be much more learnable. On the test set, the best single-model results were 83.37% accuracy for the Random Forest and 83.06% accuracy for the neural network. The unanimous ensemble was even stronger on the rows where all models agreed, reaching 87.71% accuracy and 73.86% precision on 10,330 unanimous test rows. These results show that large moves can be identified with fairly high reliability.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 81.82% | 63.16% | 62.87% | 0.6302 | 0.7544 |
| Naive Bayes | 78.59% | 64.68% | 28.83% | 0.3988 | 0.6184 |
| K-Nearest Neighbors | 81.09% | 62.78% | 57.02% | 0.5976 | 0.7299 |
| Random Forest | 83.37% | 66.19% | 66.39% | 0.6629 | 0.7765 |
| XGBoost | 82.28% | 63.76% | 65.03% | 0.6439 | 0.7647 |
| Neural Network | 83.06% | 64.66% | 68.85% | 0.6669 | 0.7827 |
| Unanimous Ensemble | 87.71% | 73.86% | 67.14% | 0.7034 | 0.8028 |

In `experiments/orion2/drctIdnt.py`, I then tried to predict the direction of the move only on rows where a 2% move was going to happen. That turned out to be much harder. After balancing the training and validation data so the models could not simply learn the majority class, every model landed near chance on the test set. The best accuracy was 52.25% from the neural network, with XGBoost at 51.55% and the other models clustered around 50% or worse. ROC AUC values were also close to 0.50. An earlier direction model had looked much better, but that turned out to be mostly a class-imbalance artifact rather than a real predictive edge. The main conclusion is that detecting an upcoming move is much easier than predicting whether that move will be up or down.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 48.18% | 45.47% | 42.50% | 0.4393 | 0.4794 |
| Naive Bayes | 49.31% | 47.12% | 50.05% | 0.4854 | 0.4934 |
| K-Nearest Neighbors | 50.91% | 48.39% | 41.53% | 0.4470 | 0.5051 |
| Random Forest | 47.60% | 42.42% | 27.14% | 0.3311 | 0.4673 |
| XGBoost | 51.55% | 49.18% | 42.60% | 0.4565 | 0.5117 |
| Neural Network | 52.25% | 50.03% | 43.21% | 0.4637 | 0.5187 |
| Unanimous Ensemble | 52.98% | 50.95% | 38.86% | 0.4409 | 0.5236 |

Finally, `experiments/orion2/allShort.py` tested whether the strong change-identification signal could still support a trading rule that benefited from high accuracy even if precision was not especially high. In this setup, the model predicts whether a large move is coming, and every predicted move opens an SQQQ position while predicted non-moves stay flat. The classification numbers looked promising at first because the test set was highly imbalanced: only 7.17% of rows were positive, so a model could post very high accuracy by being conservative. Naive Bayes reached 91.80% accuracy, and the unanimous ensemble reached 91.74% accuracy on the rows where all models agreed. Even so, every version of the strategy lost money after trading costs. The least bad result was Naive Bayes at -297.34%, while the unanimous ensemble finished at -473.90%. So far, this is the closest version of the idea to working, but it still does not make money.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision Tree | 84.66% | 26.10% | 62.14% | 0.3676 | 0.7427 | -1050.19% |
| Naive Bayes | 91.80% | 42.64% | 41.28% | 0.4195 | 0.6849 | -297.34% |
| K-Nearest Neighbors | 88.15% | 32.17% | 58.84% | 0.4160 | 0.7462 | -692.78% |
| Random Forest | 86.97% | 30.86% | 65.79% | 0.4201 | 0.7720 | -900.59% |
| XGBoost | 87.22% | 31.48% | 66.36% | 0.4270 | 0.7760 | -904.89% |
| Neural Network | 83.89% | 26.09% | 67.96% | 0.3771 | 0.7654 | -1107.02% |
| Unanimous Ensemble | 91.74% | 42.87% | 67.94% | 0.5257 | 0.8070 | -473.90% |

Overall, the experiments suggest a clear pattern. Predicting whether TQQQ is about to make a large move appears feasible. Predicting the direction of that move does not. That gap matters because a strategy with a 2% take-profit, a 2% stop-loss, and a 0.4% round-trip trading cost needs a 60% win rate just to break even. Based on the current results, the models do not yet provide enough directional information to clear that bar consistently.

The main takeaway is not that the idea is impossible, but that the current signal is incomplete. The change-identification model may still be useful as one component in a broader system, especially if it is combined with another filter for direction, market regime, or volatility. However, based on the current backtests, move detection alone is not enough to support a profitable always-short-the-breakout strategy. Even if later models improve the final return, they would still need to be evaluated for drawdown and path dependence, since a strategy can be profitable on paper while still losing too much capital before the edge shows up.
