# Paper and baseline comparison

Sun (2025) reports logistic regression accuracy/F1 of 0.7673/0.5355, decision tree 0.9432/0.8400, and XGBoost with SMOTE and Tenure Per Device 0.9343/0.8053. XGBoost has the paper's highest ROC-AUC (0.9667), but the decision tree has higher accuracy and F1.

The supplied historical workbook reports a 12-feature Random Forest with training-only random oversampling: accuracy 0.9547, precision 0.8927, recall 0.8316, F1 0.8610, AP 0.9531, and ROC-AUC 0.9859. That test was visible before this optimization and is not a pristine model-selection benchmark.

Under the new matched protocol, the same 12-feature oversampled RF recipe achieved locked-test accuracy 0.9512, precision 0.8602, recall 0.8466, F1 0.8533, AP 0.9515, and ROC-AUC 0.9890. These matched values, not the historical or paper values, support controlled comparison with the selected model.
