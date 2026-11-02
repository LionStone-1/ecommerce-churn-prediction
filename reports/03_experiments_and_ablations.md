# Experiments and ablations

Completed experiments cover majority, logistic, tree, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, the matched 12-feature RF, an all-feature oversampled RF, paper and behavioral features, class weights, random oversampling, SMOTE, two fold-local selectors, sigmoid and isotonic calibration, threshold selection, and a three-model probability blend.

Extra Trees led development OOF F1 at 0.8831. The blend had higher recall (0.9078) and AP (0.9444) but lower F1 (0.8788) and precision (0.8517), so it is the high-recall alternative rather than the balanced winner. The paper-specific feature did not improve LightGBM. Calibration degraded both AP and F1 for Extra Trees. The main gain therefore came from learner choice, with threshold selection affecting the operating point but not discrimination.
