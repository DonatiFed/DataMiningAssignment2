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

    log_price = np.log1p(df["price_usd"])
    log_mean = df.groupby("srch_id")["log_price"].transform("mean")
    log_std = df.groupby("srch_id")["log_price"].transform("std").replace(0, 1)
    df["log_price_z_score"] = (log_price - log_mean) / log_std

    # --- V4: Enhanced within-query features ---
    query_min = g["price_usd"].transform("min").replace(0, 1)
    df["price_to_min_ratio"] = df["price_usd"] / query_min

    df["is_cheapest"] = (df["price_rank"] == 1).astype(np.int8)
    df["is_most_expensive"] = (df["price_usd"] == g["price_usd"].transform("max")).astype(np.int8)
    df["is_best_star"] = (df["prop_starrating"] == g["prop_starrating"].transform("max")).astype(np.int8)
    df["is_best_review"] = (df["prop_review_score"] == g["prop_review_score"].transform("max")).astype(np.int8)
    df["is_best_location1"] = (df["prop_location_score1"] == g["prop_location_score1"].transform("max")).astype(np.int8)

    df["price_percentile"] = g["price_usd"].rank(pct=True)
    df["star_percentile"] = g["prop_starrating"].rank(pct=True, ascending=True)
    df["review_percentile"] = g["prop_review_score"].rank(pct=True, ascending=True)

    df["quality_rank_avg"] = (df["starrating_rank_norm"] + df["review_rank_norm"] + df["location1_rank_norm"]) / 3
    df["value_gap"] = df["quality_rank_avg"] - df["price_rank_norm"]

    if "location_total" in df.columns:
        df["location_total_rank"] = g["location_total"].rank(method="min", ascending=False)
        df["location_total_rank_norm"] = df["location_total_rank"] / group_size

    # V5: value_score and price_per_star within-query ranks
    if "value_score" in df.columns:
        df["value_score_rank"] = g["value_score"].rank(method="min", ascending=False)
        df["value_score_rank_norm"] = df["value_score_rank"] / group_size

    if "price_per_star" in df.columns:
        df["price_per_star_rank"] = g["price_per_star"].rank(method="min", ascending=True)
        df["price_per_star_rank_norm"] = df["price_per_star_rank"] / group_size

    # V5: Count of quality "wins" this hotel has within query
    df["n_best_flags"] = (
        df.get("is_cheapest", 0) + df.get("is_best_star", 0) +
        df.get("is_best_review", 0) + df.get("is_best_location1", 0)
    ).astype(np.int8)

    # V5: Relative competitor advantage within query
    if "comp_rate_advantage" in df.columns:
        df["comp_advantage_rank"] = g["comp_rate_advantage"].rank(method="min", ascending=False)
        df["comp_advantage_rank_norm"] = df["comp_advantage_rank"] / group_size

    return df


def interaction_features(df):
    star = df["prop_starrating"].replace(0, 1)
    df["price_per_star"] = df["price_usd"] / star

    los = df["srch_length_of_stay"].replace(0, 1)
    df["price_per_night_per_star"] = (df["price_usd"] / los) / star

    df["star_x_brand"] = df["prop_starrating"] * df["prop_brand_bool"]

    if "srch_query_affinity_score" in df.columns:
        df["query_affinity_exp"] = np.exp(df["srch_query_affinity_score"].clip(upper=0))

    df["distance_x_international"] = df["log_distance"] * (1 - df.get("is_domestic", 0))

    if "price_rank_norm" in df.columns:
        df["promo_x_cheap"] = df["promotion_flag"] * (1 - df["price_rank_norm"])

    # V5: Booking window interactions
    bw = df["srch_booking_window"]
    df["is_last_minute"] = (bw <= 1).astype(np.int8)
    df["is_short_window"] = ((bw > 1) & (bw <= 7)).astype(np.int8)
    df["is_long_window"] = (bw > 30).astype(np.int8)

    # V5: Family indicators
    df["is_family"] = (df["srch_children_count"] > 0).astype(np.int8)
    df["total_guests"] = df["srch_adults_count"] + df["srch_children_count"]
    guests = df["total_guests"].replace(0, 1)
    df["price_per_guest"] = df["price_usd"] / guests

    # V5: Price deals / discount signals
    if "price_ratio_to_hist" in df.columns:
        df["is_discounted"] = (df["price_ratio_to_hist"] < 0.9).astype(np.int8)
        df["is_overpriced"] = (df["price_ratio_to_hist"] > 1.2).astype(np.int8)

    return df


def kfold_target_encode(df, group_col, target_col, prefix, n_folds=5, seed=42, prior_weight=30):
    """K-fold target encoding split by srch_id to prevent leakage.
    Rows from the same search never contribute to their own TE.
    """
    assert "srch_id" in df.columns, "kfold TE requires srch_id for leak-safe splitting"
    global_mean = df[target_col].mean()
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)

    srch_ids = df["srch_id"].unique()
    rng = np.random.RandomState(seed)
    srch_fold = pd.Series(rng.randint(0, n_folds, size=len(srch_ids)), index=srch_ids)
    fold_ids = df["srch_id"].map(srch_fold).values

    # Assert: all rows of the same srch_id are in the same fold
    _check = pd.DataFrame({"srch_id": df["srch_id"].values, "fold": fold_ids})
    assert _check.groupby("srch_id")["fold"].nunique().max() == 1, "srch_id split leakage!"
    del _check

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


def _get_srch_folds(df, n_folds=5, seed=42):
    """Assign each srch_id to a fold; return per-row fold array."""
    srch_ids = df["srch_id"].unique()
    rng = np.random.RandomState(seed)
    srch_fold = pd.Series(rng.randint(0, n_folds, size=len(srch_ids)), index=srch_ids)
    return df["srch_id"].map(srch_fold).values


def kfold_group_stat(df, group_col, value_col, agg_func, fallback, n_folds=5, seed=42):
    """OOF group statistic. Each fold's values come from other folds only."""
    result = pd.Series(np.nan, index=df.index, dtype=np.float64)
    fold_ids = _get_srch_folds(df, n_folds, seed)
    for fold in range(n_folds):
        mask = fold_ids == fold
        oof = df[~mask]
        mapping = oof.groupby(group_col)[value_col].agg(agg_func)
        result.loc[mask] = df.loc[mask, group_col].map(mapping).values
    result.fillna(fallback, inplace=True)
    return result.astype(np.float32)


def group_stat_from_source(source_df, target_df, group_col, value_col, agg_func, fallback):
    """Group statistic from separate source (for val/test)."""
    mapping = source_df.groupby(group_col)[value_col].agg(agg_func)
    result = target_df[group_col].map(mapping).fillna(fallback)
    return result.astype(np.float32)


def _make_cross_key(df, col_a, col_b):
    return df[col_a].astype(str) + "_" + df[col_b].astype(str)


def hotel_aggregates(train_df, target_df, is_train=True):
    """Fully OOF aggregate features.
    Training: all target- and position-derived stats are k-fold OOF by srch_id.
    Val/Test: all stats computed from train_df source only.
    """
    # ========= TARGET ENCODINGS (smoothed, OOF for train) =========
    single_specs = [
        ("prop_id", "click_bool", "prop_click", 30),
        ("prop_id", "booking_bool", "prop_book", 30),
        ("prop_id", "relevance", "prop_rel", 30),
        ("srch_destination_id", "click_bool", "dest_click", 30),
        ("srch_destination_id", "booking_bool", "dest_book", 30),
        ("prop_country_id", "booking_bool", "country_book", 50),
        ("site_id", "booking_bool", "site_book", 50),
    ]

    for group_col, target_col, prefix, pw in single_specs:
        if is_train:
            target_df[f"{prefix}_rate"] = kfold_target_encode(
                target_df, group_col, target_col, prefix, prior_weight=pw)
        else:
            target_df[f"{prefix}_rate"] = target_encode_from_source(
                train_df, target_df, group_col, target_col, prefix, prior_weight=pw)

    # Cross-entity TEs
    cross_specs = [
        ("prop_id", "srch_destination_id", "booking_bool", "prop_dest_book", 10),
        ("site_id", "srch_destination_id", "booking_bool", "site_dest_book", 15),
        ("prop_id", "site_id", "booking_bool", "prop_site_book", 15),
        ("visitor_location_country_id", "prop_country_id", "booking_bool", "cpair_book", 20),
        ("site_id", "prop_country_id", "booking_bool", "site_country_book", 20),
    ]

    for col_a, col_b, target_col, prefix, pw in cross_specs:
        cross_key = f"_xkey_{prefix}"
        train_tmp = train_df.assign(**{cross_key: _make_cross_key(train_df, col_a, col_b)})
        target_df[cross_key] = _make_cross_key(target_df, col_a, col_b)
        if is_train:
            target_df[f"{prefix}_rate"] = kfold_target_encode(
                target_df, cross_key, target_col, prefix, prior_weight=pw)
        else:
            target_df[f"{prefix}_rate"] = target_encode_from_source(
                train_tmp, target_df, cross_key, target_col, prefix, prior_weight=pw)
        target_df.drop(columns=[cross_key], inplace=True)
        del train_tmp

    # ========= BOOKING-GIVEN-CLICK (OOF for train) =========
    # booking_sum / click_sum per prop_id, smoothed. Applied to ALL rows.
    alpha_bgc = 10
    if is_train:
        global_bgc = target_df.loc[target_df["click_bool"] == 1, "booking_bool"].mean()
        fold_ids = _get_srch_folds(target_df)
        bgc_result = pd.Series(np.nan, index=target_df.index, dtype=np.float64)
        click_count_result = pd.Series(np.nan, index=target_df.index, dtype=np.float64)
        for fold in range(5):
            mask = fold_ids == fold
            oof = target_df[~mask]
            book_sum = oof.groupby("prop_id")["booking_bool"].sum()
            click_sum = oof.groupby("prop_id")["click_bool"].sum()
            bgc = (book_sum + alpha_bgc * global_bgc) / (click_sum + alpha_bgc)
            bgc_result.loc[mask] = target_df.loc[mask, "prop_id"].map(bgc).values
            click_count_result.loc[mask] = target_df.loc[mask, "prop_id"].map(click_sum).values
        target_df["prop_book_given_click"] = bgc_result.fillna(global_bgc).astype(np.float32)
        target_df["prop_click_count"] = click_count_result.fillna(0).astype(np.int32)
    else:
        global_bgc = train_df.loc[train_df["click_bool"] == 1, "booking_bool"].mean()
        book_sum = train_df.groupby("prop_id")["booking_bool"].sum()
        click_sum = train_df.groupby("prop_id")["click_bool"].sum()
        bgc = (book_sum + alpha_bgc * global_bgc) / (click_sum + alpha_bgc)
        target_df["prop_book_given_click"] = target_df["prop_id"].map(bgc).fillna(global_bgc).astype(np.float32)
        target_df["prop_click_count"] = target_df["prop_id"].map(click_sum).fillna(0).astype(np.int32)

    # ========= ENTITY COUNTS (OOF for train) =========
    if is_train:
        target_df["prop_count"] = kfold_group_stat(
            target_df, "prop_id", "price_usd", "count", 0).astype(np.int32)
        target_df["dest_count"] = kfold_group_stat(
            target_df, "srch_destination_id", "price_usd", "count", 0).astype(np.int32)
    else:
        prop_count = train_df.groupby("prop_id").size()
        target_df["prop_count"] = target_df["prop_id"].map(prop_count).fillna(0).astype(np.int32)
        dest_count = train_df.groupby("srch_destination_id").size()
        target_df["dest_count"] = target_df["srch_destination_id"].map(dest_count).fillna(0).astype(np.int32)
    target_df["log_prop_count"] = np.log1p(target_df["prop_count"]).astype(np.float32)

    # ========= PROPERTY PRICE STATS (OOF for train) =========
    if is_train:
        target_df["prop_mean_price"] = kfold_group_stat(
            target_df, "prop_id", "price_usd", "mean", target_df["price_usd"].mean())
        target_df["prop_std_price"] = kfold_group_stat(
            target_df, "prop_id", "price_usd", "std", 0)
    else:
        pp = train_df.groupby("prop_id")["price_usd"].agg(["mean", "std"])
        target_df["prop_mean_price"] = target_df["prop_id"].map(pp["mean"]).fillna(train_df["price_usd"].mean()).astype(np.float32)
        target_df["prop_std_price"] = target_df["prop_id"].map(pp["std"]).fillna(0).astype(np.float32)

    target_df["price_vs_prop_mean"] = target_df["price_usd"] - target_df["prop_mean_price"]
    target_df["prop_price_zscore"] = (
        (target_df["price_usd"] - target_df["prop_mean_price"]) /
        target_df["prop_std_price"].replace(0, np.nan)
    )

    # ========= PROPERTY AVG POSITION (OOF for train) =========
    if "position" in train_df.columns:
        if is_train:
            target_df["_is_nonrand"] = (target_df["random_bool"] == 0).astype(np.float32)
            target_df["_pos_x_nonrand"] = target_df["position"] * target_df["_is_nonrand"]
            fold_ids = _get_srch_folds(target_df)
            pos_result = pd.Series(np.nan, index=target_df.index, dtype=np.float64)
            for fold in range(5):
                mask = fold_ids == fold
                oof = target_df[~mask]
                oof_nonrand = oof[oof["random_bool"] == 0]
                if len(oof_nonrand) > 0:
                    prop_pos = oof_nonrand.groupby("prop_id")["position"].mean()
                    pos_result.loc[mask] = target_df.loc[mask, "prop_id"].map(prop_pos).values
            target_df["prop_avg_position"] = pos_result.fillna(20).astype(np.float32)
            target_df.drop(columns=["_is_nonrand", "_pos_x_nonrand"], inplace=True)
        else:
            nonrand = train_df[train_df["random_bool"] == 0]
            if len(nonrand) > 0:
                prop_pos = nonrand.groupby("prop_id")["position"].mean()
                target_df["prop_avg_position"] = target_df["prop_id"].map(prop_pos).fillna(20).astype(np.float32)

    # ========= DESTINATION-RELATIVE FEATURES (OOF for train) =========
    if is_train:
        target_df["dest_mean_price"] = kfold_group_stat(
            target_df, "srch_destination_id", "price_usd", "mean", target_df["price_usd"].mean())
        target_df["dest_mean_star"] = kfold_group_stat(
            target_df, "srch_destination_id", "prop_starrating", "mean", target_df["prop_starrating"].mean())
        target_df["dest_mean_review"] = kfold_group_stat(
            target_df, "srch_destination_id", "prop_review_score", "mean", target_df["prop_review_score"].mean())
    else:
        for col, name in [("price_usd", "dest_mean_price"),
                          ("prop_starrating", "dest_mean_star"),
                          ("prop_review_score", "dest_mean_review")]:
            mapping = train_df.groupby("srch_destination_id")[col].mean()
            target_df[name] = target_df["srch_destination_id"].map(mapping).fillna(train_df[col].mean()).astype(np.float32)

    target_df["price_vs_dest_mean"] = target_df["price_usd"] - target_df["dest_mean_price"]
    target_df["star_vs_dest_mean"] = target_df["prop_starrating"] - target_df["dest_mean_star"]
    target_df["review_vs_dest_mean"] = target_df["prop_review_score"] - target_df["dest_mean_review"]

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


FORBIDDEN_FEATURES = {"position", "click_bool", "booking_bool", "gross_bookings_usd",
                       "random_bool", "relevance", "date_time"}


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
    df = interaction_features(df)
    if agg_source is not None:
        df = hotel_aggregates(agg_source, df, is_train=is_train)
    # Defragment and downcast
    df = df.copy()
    df = downcast_floats(df)
    return df
