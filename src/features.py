import pandas as pd
import numpy as np


def temporal_features(df):
    dt = pd.to_datetime(df["date_time"])
    df["month"] = dt.dt.month.astype(np.int8)
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["is_weekend_search"] = (df["dayofweek"] >= 5).astype(np.int8)
    return df


def missing_flags(df):
    df["has_location_score2"] = df["prop_location_score2"].notna().astype(np.int8)
    df["has_visitor_history"] = df["visitor_hist_starrating"].notna().astype(np.int8)
    df["has_query_affinity"] = df["srch_query_affinity_score"].notna().astype(np.int8)
    df["has_distance"] = df["orig_destination_distance"].notna().astype(np.int8)
    df["has_historical_price"] = (df["prop_log_historical_price"] > 0).astype(np.int8)
    return df


def price_features(df):
    hist_price = np.exp(df["prop_log_historical_price"])
    hist_price = hist_price.replace(1.0, np.nan)

    df["price_diff_from_hist"] = df["price_usd"] - hist_price
    df["price_ratio_to_hist"] = df["price_usd"] / hist_price

    los = df["srch_length_of_stay"].replace(0, 1)
    df["price_per_night"] = df["price_usd"] / los
    df["total_cost"] = df["price_usd"] * df["srch_length_of_stay"]

    adults = df["srch_adults_count"].replace(0, 1)
    df["price_per_person"] = df["price_usd"] / adults

    rooms = df["srch_room_count"].replace(0, 1)
    df["price_per_room"] = df["price_usd"] / rooms

    df["log_price"] = np.log1p(df["price_usd"])
    df["log_distance"] = np.log1p(df["orig_destination_distance"])

    return df


def visitor_match_features(df):
    df["star_diff"] = df["visitor_hist_starrating"] - df["prop_starrating"]
    df["abs_star_diff"] = df["star_diff"].abs()
    df["price_diff_from_visitor_hist"] = df["visitor_hist_adr_usd"] - df["price_usd"]
    df["price_ratio_to_visitor_hist"] = df["price_usd"] / df["visitor_hist_adr_usd"]
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
        df["comp_rate_advantage"] = df["comp_more_expensive_count"] - df["comp_cheaper_count"]

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


def quality_features(df):
    df["value_score"] = df["prop_starrating"] / np.log1p(df["price_usd"])
    df["star_review_product"] = df["prop_starrating"] * df["prop_review_score"]
    df["location_total"] = df["prop_location_score1"] + df["prop_location_score2"].fillna(0)
    df["is_domestic"] = (
        df["visitor_location_country_id"] == df["prop_country_id"]
    ).astype(np.int8)
    df["starrating_is_zero"] = (df["prop_starrating"] == 0).astype(np.int8)
    df["review_is_zero"] = (df["prop_review_score"] == 0).astype(np.int8)
    return df


def listwise_features(df):
    g = df.groupby("srch_id")

    df["price_rank"] = g["price_usd"].rank(method="min", ascending=True)
    df["starrating_rank"] = g["prop_starrating"].rank(method="min", ascending=False)
    df["review_rank"] = g["prop_review_score"].rank(method="min", ascending=False)
    df["location1_rank"] = g["prop_location_score1"].rank(method="min", ascending=False)
    df["location2_rank"] = g["prop_location_score2"].rank(method="min", ascending=False)

    group_size = g["price_usd"].transform("count")
    df["price_rank_norm"] = df["price_rank"] / group_size
    df["starrating_rank_norm"] = df["starrating_rank"] / group_size
    df["review_rank_norm"] = df["review_rank"] / group_size
    df["location1_rank_norm"] = df["location1_rank"] / group_size
    df["location2_rank_norm"] = df["location2_rank"] / group_size

    df["price_vs_mean"] = df["price_usd"] - g["price_usd"].transform("mean")
    df["price_vs_median"] = df["price_usd"] - g["price_usd"].transform("median")
    df["price_vs_min"] = df["price_usd"] - g["price_usd"].transform("min")
    df["price_vs_max"] = g["price_usd"].transform("max") - df["price_usd"]

    query_std = g["price_usd"].transform("std").replace(0, 1)
    df["price_z_score"] = (df["price_usd"] - g["price_usd"].transform("mean")) / query_std

    df["star_vs_mean"] = df["prop_starrating"] - g["prop_starrating"].transform("mean")
    df["review_vs_mean"] = df["prop_review_score"] - g["prop_review_score"].transform("mean")
    df["loc1_vs_mean"] = df["prop_location_score1"] - g["prop_location_score1"].transform("mean")
    df["loc2_vs_mean"] = df["prop_location_score2"] - g["prop_location_score2"].transform("mean")

    df["query_hotel_count"] = group_size.astype(np.int16)
    df["query_price_std"] = g["price_usd"].transform("std")
    df["query_star_mean"] = g["prop_starrating"].transform("mean")
    df["query_price_mean"] = g["price_usd"].transform("mean")

    # Within-query log-price z-score (price is skewed, log helps)
    log_price = np.log1p(df["price_usd"])
    log_mean = df.groupby("srch_id")["log_price"].transform("mean")
    log_std = df.groupby("srch_id")["log_price"].transform("std").replace(0, 1)
    df["log_price_z_score"] = (log_price - log_mean) / log_std

    return df


def kfold_target_encode(df, group_col, target_col, prefix, n_folds=5, seed=42, prior_weight=30):
    """K-fold target encoding to avoid leakage within training data.
    Each row's encoded value is computed from folds that don't contain that row.
    """
    global_mean = df[target_col].mean()
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)

    rng = np.random.RandomState(seed)
    fold_ids = rng.randint(0, n_folds, size=len(df))

    for fold in range(n_folds):
        mask = fold_ids == fold
        oof = df[~mask]
        stats = oof.groupby(group_col)[target_col].agg(["sum", "count"])
        stats["encoded"] = (stats["sum"] + prior_weight * global_mean) / (stats["count"] + prior_weight)
        mapping = stats["encoded"]
        result.loc[mask] = df.loc[mask, group_col].map(mapping).values

    result.fillna(global_mean, inplace=True)
    return result.astype(np.float32)


def target_encode_from_source(source_df, target_df, group_col, target_col, prefix, prior_weight=30):
    """Target encoding using a separate source (for val/test — no leakage)."""
    global_mean = source_df[target_col].mean()
    stats = source_df.groupby(group_col)[target_col].agg(["sum", "count"])
    stats["encoded"] = (stats["sum"] + prior_weight * global_mean) / (stats["count"] + prior_weight)
    mapping = stats["encoded"]
    result = target_df[group_col].map(mapping).fillna(global_mean)
    return result.astype(np.float32)


def hotel_aggregates(train_df, target_df, is_train=True):
    """Target-encoded aggregate features.
    For training: k-fold encoding to prevent leakage.
    For val/test: encode from full train source.
    """
    encode_specs = [
        ("prop_id", "click_bool", "prop_click"),
        ("prop_id", "booking_bool", "prop_book"),
        ("srch_destination_id", "click_bool", "dest_click"),
        ("srch_destination_id", "booking_bool", "dest_book"),
        ("prop_country_id", "booking_bool", "country_book"),
    ]

    if is_train:
        for group_col, target_col, prefix in encode_specs:
            target_df[f"{prefix}_rate"] = kfold_target_encode(
                target_df, group_col, target_col, prefix
            )
    else:
        for group_col, target_col, prefix in encode_specs:
            target_df[f"{prefix}_rate"] = target_encode_from_source(
                train_df, target_df, group_col, target_col, prefix
            )

    # Property count (not target-dependent, no leakage)
    prop_count = train_df.groupby("prop_id").size().reset_index(name="prop_count")
    target_df = target_df.merge(prop_count, on="prop_id", how="left")
    target_df["prop_count"] = target_df["prop_count"].fillna(0).astype(np.int32)

    # Property avg price and price deviation
    prop_price = train_df.groupby("prop_id")["price_usd"].agg(["mean", "std"]).reset_index()
    prop_price.columns = ["prop_id", "prop_mean_price", "prop_std_price"]
    target_df = target_df.merge(prop_price, on="prop_id", how="left")
    target_df["price_vs_prop_mean"] = target_df["price_usd"] - target_df["prop_mean_price"]
    target_df["prop_price_zscore"] = (
        (target_df["price_usd"] - target_df["prop_mean_price"]) /
        target_df["prop_std_price"].replace(0, np.nan)
    )

    # Property average star/review/location (not target-dependent)
    prop_quality = train_df.groupby("prop_id").agg(
        prop_avg_position=("position", "mean") if "position" in train_df.columns else ("prop_starrating", "first"),
    ).reset_index()

    # Destination average price (context for "is this destination expensive?")
    dest_price = train_df.groupby("srch_destination_id")["price_usd"].mean().reset_index()
    dest_price.columns = ["srch_destination_id", "dest_mean_price"]
    target_df = target_df.merge(dest_price, on="srch_destination_id", how="left")
    target_df["price_vs_dest_mean"] = target_df["price_usd"] - target_df["dest_mean_price"]

    return target_df


def compute_position_propensity(train_df):
    """Compute click propensity per position from random_bool=1 data."""
    rand = train_df[train_df["random_bool"] == 1]
    propensity = rand.groupby("position")["click_bool"].mean()
    return propensity


def compute_sample_weights(train_df, propensity):
    """Inverse propensity weights: upweight clicks from low positions."""
    max_prop = propensity.max()
    weights = np.ones(len(train_df), dtype=np.float32)

    if "position" in train_df.columns:
        def _ipw(p):
            prop_val = propensity.get(p, 0)
            if prop_val <= 0:
                return 1.0
            return max_prop / prop_val
        pos_weight = train_df["position"].map(_ipw)
        # Only apply IPW to non-random data (random data is already unbiased)
        is_nonrandom = train_df["random_bool"] == 0
        weights = np.where(is_nonrandom, pos_weight, 1.0).astype(np.float32)
        # Cap extreme weights
        weights = np.clip(weights, 0.1, 10.0)

    return weights


def drop_raw_comp_columns(df):
    drop = []
    for i in range(1, 9):
        for suffix in ["_rate", "_inv", "_rate_percent_diff"]:
            col = f"comp{i}{suffix}"
            if col in df.columns:
                drop.append(col)
    df.drop(columns=drop, inplace=True)
    return df


def downcast_floats(df):
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def build_features(df, agg_source=None, is_train=True):
    df = df.copy()
    df = temporal_features(df)
    df = missing_flags(df)
    df = price_features(df)
    df = visitor_match_features(df)
    df = competitor_features(df)
    df = drop_raw_comp_columns(df)
    df = quality_features(df)
    df = downcast_floats(df)
    df = listwise_features(df)
    if agg_source is not None:
        df = hotel_aggregates(agg_source, df, is_train=is_train)
    return df
