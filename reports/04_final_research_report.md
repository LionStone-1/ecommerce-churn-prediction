# Final research report

## Recommendation

Use the tuned Extra Trees pipeline with all 18 original predictors and decision threshold 0.45 as the balanced configuration. On the locked group-aware test it achieved accuracy 0.9725, precision 0.9489, recall 0.8836, F1 0.9151, AP 0.9747, ROC-AUC 0.9938, balanced accuracy 0.9370, MCC 0.8995, Brier score 0.0237, and log loss 0.0879. Its confusion matrix was TN 928, FP 9, FN 22, TP 167.

Against the newly executed matched 12-feature oversampled RF on identical test customers, Extra Trees improved F1 by 0.0612 (paired bootstrap 95% CI 0.0326 to 0.0931) and AP by 0.0230 (0.0112 to 0.0376). Relative to the historical RF reference, differences are +0.0178 accuracy, +0.0562 precision, +0.0520 recall, +0.0540 F1, +0.0215 AP, and +0.0080 ROC-AUC, but those are descriptive because the test memberships differ.

## High-recall alternative

The OOF probability blend produced recall 0.9078, precision 0.8517, F1 0.8788, and accuracy 0.9578 on development. It may suit outreach where missed churn is costlier than false contacts, but it was not selected for the locked test because the predeclared balanced ranking favored Extra Trees.

## Limitations

The data are cross-sectional; raw cleaning lineage and temporal order are unavailable; the paper uses one fewer row; 189 churners appear in the locked test; no independent external cohort exists; and business benefits are predictive targeting estimates rather than causal retention effects.
