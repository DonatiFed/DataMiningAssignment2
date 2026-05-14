import pandas as pd
import numpy as np


def temporal_features(df):
    dt = pd.to_datetime(df["date_time"])
    df["month"] = dt.dt.month.astype(np.int8)
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["is_weekend_search"] = (df["dayofweek"] >= 5).astype(np.int8)
    return df


def price_features(df):
    hist_price = np.exp(df["prop_log_historical_price"])
    hist_price = hist_price.replace(1.0, np.nan)  # log=0 means no data

    df["price_diff_from_hist"] = df["price_usd"] - hist_price
    df["price_ratio_to_hist"] = df["price_usd"] / hist_price

    # per-night price
    los = df["srch_length_of_stay"].replace(0, 1)
    df["price_per_night"] = df["price_usd"] / los

    # total trip cost estimate
    df["total_cost"] = df["price_usd"] * df["srch_length_of_stay"]

    return df


def visitor_match_features(df):
    df["star_diff"] = df["visitor_hist_starrating"] - df["prop_starrating"]
    df["price_diff_from_hist_usd"] = df["visitor_hist_adr_usd"] - df["price_usd"]
    df["price_ratio_to_visitor_hist"] = df["price_usd"] / df["visitor_hist_adr_usd"]
    df["has_visitor_history"] = df["visitor_hist_starrating"].notna().astype(np.int8)
    return df


def competitor_features(df):
    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    comp_inv_cols = [f"comp{i}_inv" for i in range(1, 9)]
    comp_pct_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]

    existing_rate = [c for c in comp_rate_cols if c in df.columns]
    existing_inv = [c for c in comp_inv_cols if c in df.columns]
    existing_pct = [c for c in comp_pct_cols if c in df.columns]

    if existing_rate:
        rates = df[existing_rate]
        df["comp_rate_sum"] = rates.sum(axis=1, skipna=True)
        df["comp_rate_count"] = rates.notna().sum(axis=1).astype(np.int8)
        df["comp_cheaper_count"] = (rates == -1).sum(axis=1).astype(np.int8)
        df["comp_more_expensive_count"] = (rates == 1).sum(axis=1).astype(np.int8)

    if existing_inv:
        inv = df[existing_inv]
        df["comp_inv_count"] = inv.notna().sum(axis=1).astype(np.int8)
        df["comp_no_inv_count"] = (inv == 1).sum(axis=1).astype(np.int8)

    if existing_pct:
        pct = df[existing_pct]
        df["comp_rate_pct_mean"] = pct.mean(axis=1, skipna=True)
        df["comp_rate_pct_min"] = pct.min(axis=1, skipna=True)
        df["comp_rate_pct_max"] = pct.max(axis=1, skipna=True)

    return df


def listwise_features(df):
    """Within-query comparative features — how each hotel compares to others in the same search."""
    g = df.groupby("srch_id")

    # Rank within query (lower = better for price, higher = better for scores)
    df["price_rank"] = g["price_usd"].rank(method="min", ascending=True)
    df["starrating_rank"] = g["prop_starrating"].rank(method="min", ascending=False)
    df["review_rank"] = g["prop_review_score"].rank(method="min", ascending=False)
    df["location1_rank"] = g["prop_location_score1"].rank(method="min", ascending=False)
    df["location2_rank"] = g["prop_location_score2"].rank(method="min", ascending=False)

    # Normalized position within the list (0-1 scale)
    group_size = g["price_usd"].transform("count")
    df["price_rank_norm"] = df["price_rank"] / group_size
    df["starrating_rank_norm"] = df["starrating_rank"] / group_size

    # Difference from query mean/min/max
    df["price_vs_mean"] = df["price_usd"] - g["price_usd"].transform("mean")
    df["price_vs_median"] = df["price_usd"] - g["price_usd"].transform("median")
    df["price_vs_min"] = df["price_usd"] - g["price_usd"].transform("min")

    df["star_vs_mean"] = df["prop_starrating"] - g["prop_starrating"].transform("mean")
    df["review_vs_mean"] = df["prop_review_score"] - g["prop_review_score"].transform("mean")
    df["loc1_vs_mean"] = df["prop_location_score1"] - g["prop_location_score1"].transform("mean")
    df["loc2_vs_mean"] = df["prop_location_score2"] - g["prop_location_score2"].transform("mean")

    # Count of hotels in the query
    df["query_hotel_count"] = group_size.astype(np.int16)

    return df


def hotel_aggregates(train_df, target_df):
    """Compute per-hotel historical CTR and booking rate from training data.
    Returns target_df with new columns appended.
    Must be called with train-only data as source to avoid leakage.
    """
    # per prop_id
    prop_stats = train_df.groupby("prop_id").agg(
        prop_click_rate=("click_bool", "mean"),
        prop_book_rate=("booking_bool", "mean"),
        prop_count=("srch_id", "count"),
    ).reset_index()

    target_df = target_df.merge(prop_stats, on="prop_id", how="left")

    # per srch_destination_id
    dest_stats = train_df.groupby("srch_destination_id").agg(
        dest_click_rate=("click_bool", "mean"),
        dest_book_rate=("booking_bool", "mean"),
    ).reset_index()

    target_df = target_df.merge(dest_stats, on="srch_destination_id", how="left")

    # per prop_country_id
    country_stats = train_df.groupby("prop_country_id").agg(
        country_click_rate=("click_bool", "mean"),
        country_book_rate=("booking_bool", "mean"),
    ).reset_index()

    target_df = target_df.merge(country_stats, on="prop_country_id", how="left")

    return target_df


def build_features(df, agg_source=None):
    """Full feature engineering pipeline.

    df: the dataframe to transform (train or test)
    agg_source: train dataframe to compute aggregates from.
                If None, skip aggregate features.
    """
    df = df.copy()

    df = temporal_features(df)
    df = price_features(df)
    df = visitor_match_features(df)
    df = competitor_features(df)
    df = listwise_features(df)

    if agg_source is not None:
        df = hotel_aggregates(agg_source, df)

    return df
