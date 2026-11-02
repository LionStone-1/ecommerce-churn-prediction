# Data and leakage audit

The immutable modeling source contains 5,630 customers, 20 columns, 948 churners, 4,682 non-churners, zero missing cells, zero duplicate customer IDs, and 18 predictors. Its SHA-256 is `371acba4a77fc4e630792e5f9d3c8f350ba7a5d0e9f23230f89922ad34e0f0f3`.

`Phone` was deterministically mapped to `Mobile Phone` in `PreferredLoginDevice`, and `Mobile` to `Mobile Phone` in `PreferedOrderCat`. Standardized predictor hashes yielded 5,071 profile groups. The 559 repeated groups contain 1,118 rows and no conflicting labels. No group crosses development and locked test.

The supplied cleaned workbook does not expose raw cleaning code. The original raw `E Commerce Dataset.xlsx` was not present, so pre-split imputation or capping cannot be ruled out. The published paper reports 5,629 records, one fewer than the local workbook. All 5,630 legitimate local rows were retained.

The historical report lacks split IDs and runnable code. Its final-test scores are treated as historical evidence only. New model selection used a predeclared group-aware split and grouped OOF predictions. CustomerID, Churn, group IDs, and split flags never entered the predictors.
