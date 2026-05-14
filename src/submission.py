import pandas as pd
from datetime import datetime
from src.config import SUBMISSIONS_DIR


def generate_submission(df, score_col="pred_score", tag="baseline"):
    """Generate submission CSV from test dataframe with predictions.

    df must have columns: srch_id, prop_id, and score_col.
    Output: srch_id,prop_id sorted by score descending within each srch_id.
    """
    ranked = (
        df[["srch_id", "prop_id", score_col]]
        .sort_values(["srch_id", score_col], ascending=[True, False])
    )

    submission = ranked[["srch_id", "prop_id"]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = SUBMISSIONS_DIR / f"submission_{tag}_{timestamp}.csv"
    submission.to_csv(filename, index=False)
    print(f"Submission saved: {filename}")
    print(f"  Rows: {len(submission)}, Unique searches: {submission['srch_id'].nunique()}")
    return filename
