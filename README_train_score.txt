Quick start (2 steps)

STEP 1 — Train + save models (needs labels)
- Input: dummy_outliers.csv (has column is_outlier 0/1)
- Run:
    python outlier_supervised_pipeline_train_score.py

This creates:
- outlier_results.xlsx (evaluation)
- model_artifacts/  (saved calibrated models + best thresholds)

STEP 2 — Weekly scoring (no labels needed)
- Input: dummy_outliers_unlabeled.csv (no is_outlier column)
- Set in USER CONFIG:
    mode = "score"
    input_path = "dummy_outliers_unlabeled.csv"
    output_excel = "weekly_scored.xlsx"

Or run:
    python outlier_supervised_pipeline_train_score.py --mode score --input dummy_outliers_unlabeled.csv --output weekly_scored.xlsx

Outputs in score mode:
- Excel with:
  - summary (how many predicted outliers per model)
  - scored_predictions (rows ranked by calibrated outlier probability)

Note:
- This version intentionally does NOT use dates/time-split to keep config clean.