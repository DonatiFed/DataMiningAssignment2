import pandas as pd
import numpy as np
from src.config import TRAIN_FILE, TEST_FILE, NON_FEATURE_COLS, ID_COLS


def load_train(sample_frac=None, random_state=42):
    df = pd.read_csv(TRAIN_FILE, na_values="NULL")
    if sample_frac is not None and sample_frac < 1.0:
        srch_ids = df["srch_id"].unique()
        rng = np.random.RandomState(random_state)
        sampled = rng.choice(srch_ids, size=int(len(srch_ids) * sample_frac), replace=False)
        df = df[df["srch_id"].isin(sampled)].reset_index(drop=True)
    return df


def load_test():
    return pd.read_csv(TEST_FILE, na_values="NULL")


def make_target(df):
    df["relevance"] = 5 * df["booking_bool"] + df["click_bool"] * (1 - df["booking_bool"])
    return df


def get_feature_columns(df):
    drop = set(NON_FEATURE_COLS) & set(df.columns)
    return [c for c in df.columns if c not in drop and c != "relevance"]


def split_val(df, val_frac=0.1, random_state=42):
    srch_ids = df["srch_id"].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(srch_ids)
    n_val = int(len(srch_ids) * val_frac)
    val_ids = set(srch_ids[:n_val])
    mask = df["srch_id"].isin(val_ids)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
