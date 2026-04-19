# Zhongjie Results

This attempt started from the same Orion pipeline and reused the existing data fetching, feature construction, target definition, chronological split, and z-score normalization. On top of that setup, I added a new linear / SVM baseline line, including Logistic Regression, LinearSVC, and RBF SVM. I also update the evaluation function which only predicted positive trades contribute to realized profit.

## Evaluation function

I updated the evaluation function to evaluate_and_print_trade_zhongjie(). The main change was to make the profit calculation match a trade-only interpretation. The original evaluate_and_print() was closer to a classification-style profit approximation, because it also gave weight to correct negative predictions. For my experiments, that was not the behavior I wanted. I only wanted rows predicted as positive to affect realized profit, since those are the trades that would actually be taken. True positives are treated as winning trades and false positives as losing trades, while true negatives and false negatives are not counted in realized profit. 

## Result

I focused on model types that were not part of the original Orion experiments, namely Logistic Regression and SVM-based models. The goal was to see whether a linear or margin-based classifier could produce a cleaner trading signal under the same Orion pipeline, including the same data source, feature construction, target definition, and chronological train / validation / test split.

Three models are tested:

- `LogisticRegression` - `experiments/zhongjie/LR.py`
- `LinearSVC` - `experiments/zhongjie/linear_svm_initial.py`
- `RBF SVM` - `experiments/zhongjie/rbf_svm.py`

For each model, I tested multiple values of `C` and decision thresholds. For `RBF SVM`, I also varied `gamma`. This made it possible to compare not only the final test result, but also how sensitive each model was to parameter choice and trade-selection threshold.

| Model | Representative Params | Test Trades | Test Precision | Test Recall | Test Profit | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | C=0.1, threshold=0.7 | 1308 | 35.17% | 10.32% | -1037.60% | Too aggressive; win rate stayed far below break-even |
| RBF SVM | C=0.1, gamma=scale, class_weight=None, threshold=-0.5 | 9 | 22.22% | 0.04% | -11.80% | Traded too little to be useful and did not produce a practical signal, also long processing time. |
| LinearSVC | C=0.5, threshold=0.55 | 176 | 55.68% | 2.20% | 4.80% | Most promising linear baseline; slightly above break-even on test |

Logistic Regression did not work well in this setup. Even after tuning C and the decision threshold, it still tended to take too many trades for the win rate it achieved. The best representative result stayed well below the estimated 55% break-even win rate, so realized profit remained clearly negative. The model seemed too broad rather than selective. It labeled too many rows as tradable positives, but the quality of those trades was not high enough to offset trading costs and stop-losses. RBF SVM was also hard to justify. It was much slower to run, taking more than 6 hours to finish the full search, and the final results were still not competitive. It took only a small number of trades and had very low recall. Even though the absolute loss was not large, it did not produce a signal that would be useful in practice. In this project, that made it both expensive to tune and difficult to use. So I decide to move forward with the last model.

LinearSVC gave the strongest results of the three. Compared with Logistic Regression, it was more selective and produced higher-precision trades. Compared with RBF SVM, it was much faster to tune. Its best results appeared in a conservative parameter region, where the model traded less often but with a noticeably higher win rate. In the final confirmation runs, LinearSVC reached test precision slightly above the estimated break-even win rate. None of the other models got that close under the same evaluation setup. For that reason, the rest of this section looks more closely at LinearSVC. I use its search results and confirmation runs to examine how `C`, decision threshold, trade frequency, precision, and realized profit changed together, and why this model stood out from the other two. Since the validation precision shows unstablity, I also want to find out whether the good result was just a single lucky point, or we can find a more regular pattern.

| C | Threshold | Validation Trades | Validation Precision | Validation Profit | Test Trades | Test Precision | Test Profit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.45 | 93 | 0.00% | -204.6% | 281 | 54.80% | -2.2% |
| 0.5 | 0.50 | 85 | 0.00% | -187.0% | 225 | 58.22% | 29.0% |
| 0.5 | 0.55 | 82 | 0.00% | -180.4% | 176 | 55.68% | 4.8% |
| 1.0 | 0.55 | 84 | 0.00% | -184.8% | 177 | 55.93% | 6.6% |
| 1.0 | 0.50 | 90 | 0.00% | -198.0% | 227 | 58.15% | 28.6% |
| 2.0 | 0.50 | 95 | 0.00% | -209.0% | 226 | 57.96% | 26.8% |

I add another two round of LinearSVC searching in `experiments/zhongjie/linear_svm_narrow.py` and `experiments/zhongjie/linear_svm_confirm.py`.The main pattern in the LinearSVC experiments was straightforward. Lower thresholds increased trade count, but usually at the cost of trade quality. Higher thresholds made the model more selective and generally improved precision. The most interesting region appeared around C≈1 and threshold≈0.5, where several nearby settings produced similar test behavior. Test precision rose to about 58%, and realized profit became slightly positive. The strongest LinearSVC results were concentrated around threshold = 0.50, with C values from 0.5 to 2.0 giving very similar test behavior. It suggested the result might not be just a single lucky point. So the model’s best region did not look completely isolated, even though the validation segment remained weak across that same range.

At the same time, the validation results were still unstable. I tried several updates on the validation side, including requiring a minimum number of validation trades, recording both validation and test results for every parameter combination instead of only checking the final selected model, and running a small confirmation search around the strongest observed LinearSVC region to see whether the result was isolated. Even with those changes, the overall pattern did not improve much. The validation segment still did not reliably identify the same conservative settings that later gave the best test behavior. So although LinearSVC produced the strongest observed result in this set of experiments, the validation setup was still not reliable enough to identify that region ahead of time.


## Conclusion
Under the current setup, Logistic Regression was too aggressive, and RBF SVM was too slow without giving useful results. LinearSVC was the only one that got close to a workable trade-only outcome, especially when around C≈1 and threshold≈0.5. In that region, test precision sometimes rose slightly above the estimated 55% break-even win rate, although the validation signal remained unstable. That suggests the current feature set may contain some usable directional information, but the model selection process is still sensitive and would likely need a more reliable validation design or more changes in pipeline.