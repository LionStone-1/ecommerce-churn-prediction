# Evidence audit

- Modeling source: immutable copy of the supplied cleaned workbook, SHA-256 `371acba4a77fc4e630792e5f9d3c8f350ba7a5d0e9f23230f89922ad34e0f0f3`.
- Dataset: 5,630 customers, 948 churners, 18 predictors, zero missing cells, zero duplicate IDs.
- Grouping: 5,071 standardized predictor profiles; 559 repeated groups covering 1,118 rows; zero conflicting-label groups.
- Historical report: retained as reference because it provides no split manifest or executable code and exposes its final-test results.
- New protocol: seed 20251024, fixed group-aware 20% holdout, grouped nested development evaluation, OOF-only thresholding, one final holdout evaluation.
- Published paper: six pages, 5,629 records, protocol not demonstrably identical. Its cross-study scores are descriptive only.
- Statistical claim: paired percentile bootstrap on identical locked-test predictions. F1 difference 0.0612 (95% CI 0.0326 to 0.0931); AP difference 0.0230 (0.0112 to 0.0376).
- Blocked audit: the original raw workbook was not supplied, so global cleaning lineage cannot be reconstructed.
