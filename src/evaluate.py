import numpy as np
import pandas as pd


def dcg_at_k(relevances, k=5):
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k=5):
    best = dcg_at_k(sorted(relevances, reverse=True), k)
    if best == 0:
        return 0.0
    return dcg_at_k(relevances, k) / best


def evaluate_ndcg(df, score_col="pred_score", k=5):
    """Compute mean NDCG@k across all search queries.

    df must have columns: srch_id, relevance, and score_col.
    """
    results = []
    for srch_id, group in df.groupby("srch_id"):
        sorted_group = group.sort_values(score_col, ascending=False)
        relevances = sorted_group["relevance"].values
        results.append(ndcg_at_k(relevances, k))
    return np.mean(results)
