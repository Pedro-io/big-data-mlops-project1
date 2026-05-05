"""
Feature engineering pipeline.

Reads the preprocessed transactions and writes feature sets to data/feature_store/.

Feature sets generated:
  - transaction_features.csv  : row-level enrichment (total_value, day_of_week, etc.)
  - user_features.csv         : one row per user, aggregated behaviour
  - user_product_features.csv : one row per (user, product) pair
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"
INPUT_FILE = DATA_DIR / "preprocessed" / "transactions_preprocessed.csv"
FEATURE_STORE = DATA_DIR / "feature_store"
FEATURE_STORE.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 1. Transaction-level features
# --------------------------------------------------------------------------- #

def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per original transaction, enriched with basic derived columns."""
    t = df.copy()

    t["total_value"] = t["quantity"] * t["price"]
    t["day_of_week"] = t["timestamp"].dt.dayofweek          # 0=Mon … 6=Sun
    t["month"] = t["timestamp"].dt.month
    t["is_weekend"] = t["day_of_week"].isin([5, 6]).astype(int)

    # days since previous purchase by the same user (lag 1)
    t["days_since_prev_purchase"] = (
        t.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(86_400)
    )

    return t


# --------------------------------------------------------------------------- #
# 2. User-level features
# --------------------------------------------------------------------------- #

def build_user_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate one row per user."""
    df = df.copy()
    df["total_value"] = df["quantity"] * df["price"]

    agg = df.groupby("user_id").agg(
        total_transactions=("timestamp", "count"),
        total_spend=("total_value", "sum"),
        avg_order_value=("total_value", "mean"),
        unique_products=("product_id", "nunique"),
        first_purchase=("timestamp", "min"),
        last_purchase=("timestamp", "max"),
        avg_quantity=("quantity", "mean"),
    ).reset_index()

    agg["days_since_last_purchase"] = (
        reference_date - agg["last_purchase"]
    ).dt.total_seconds().div(86_400)

    agg["customer_lifetime_days"] = (
        agg["last_purchase"] - agg["first_purchase"]
    ).dt.total_seconds().div(86_400)

    # avg days between purchases (frequency)
    agg["avg_days_between_purchases"] = (
        agg["customer_lifetime_days"] / (agg["total_transactions"] - 1)
    ).where(agg["total_transactions"] > 1)

    # rolling windows — purchases and spend in last 30 / 90 days
    cutoff_30 = reference_date - pd.Timedelta(days=30)
    cutoff_90 = reference_date - pd.Timedelta(days=90)

    last_30 = df[df["timestamp"] >= cutoff_30].groupby("user_id").agg(
        purchases_last_30d=("timestamp", "count"),
        spend_last_30d=("total_value", "sum"),
    )
    last_90 = df[df["timestamp"] >= cutoff_90].groupby("user_id").agg(
        purchases_last_90d=("timestamp", "count"),
        spend_last_90d=("total_value", "sum"),
    )

    agg = agg.merge(last_30, on="user_id", how="left")
    agg = agg.merge(last_90, on="user_id", how="left")
    agg[["purchases_last_30d", "purchases_last_90d"]] = (
        agg[["purchases_last_30d", "purchases_last_90d"]].fillna(0)
    )
    agg[["spend_last_30d", "spend_last_90d"]] = (
        agg[["spend_last_30d", "spend_last_90d"]].fillna(0)
    )

    return agg.drop(columns=["first_purchase", "last_purchase"])


# --------------------------------------------------------------------------- #
# 3. User-Product features
# --------------------------------------------------------------------------- #

def build_user_product_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate one row per (user, product) pair."""
    df = df.copy()
    df["total_value"] = df["quantity"] * df["price"]

    agg = df.groupby(["user_id", "product_id"]).agg(
        purchase_count=("timestamp", "count"),
        total_spend=("total_value", "sum"),
        avg_quantity=("quantity", "mean"),
        first_purchase=("timestamp", "min"),
        last_purchase=("timestamp", "max"),
    ).reset_index()

    agg["days_since_last_purchase"] = (
        reference_date - agg["last_purchase"]
    ).dt.total_seconds().div(86_400)

    # last quantity bought (most recent transaction for this pair)
    last_qty = (
        df.sort_values("timestamp")
        .groupby(["user_id", "product_id"])["quantity"]
        .last()
        .rename("last_quantity")
        .reset_index()
    )
    agg = agg.merge(last_qty, on=["user_id", "product_id"], how="left")

    return agg.drop(columns=["first_purchase", "last_purchase"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("Loading data...")
    df = load_data()

    reference_date = df["timestamp"].max()
    print(f"Reference date: {reference_date.date()}")

    print("Building transaction features...")
    t_feat = build_transaction_features(df)
    t_feat.to_csv(FEATURE_STORE / "transaction_features.csv", index=False)

    print("Building user features...")
    u_feat = build_user_features(df, reference_date)
    u_feat.to_csv(FEATURE_STORE / "user_features.csv", index=False)

    print("Building user-product features...")
    up_feat = build_user_product_features(df, reference_date)
    up_feat.to_csv(FEATURE_STORE / "user_product_features.csv", index=False)

    print("\nDone. Files written to:", FEATURE_STORE)
    print(f"  transaction_features.csv  -> {len(t_feat):,} rows")
    print(f"  user_features.csv         -> {len(u_feat):,} rows")
    print(f"  user_product_features.csv -> {len(up_feat):,} rows")


if __name__ == "__main__":
    main()
