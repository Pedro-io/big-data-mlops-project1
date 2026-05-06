"""
Feature engineering pipeline.

FeatureBuilder follows the fit / transform contract:
  builder.fit(df_train)       — learns all statistics from training data only
  builder.transform(df)       — applies frozen statistics to any split
  builder.fit_transform(df)   — convenience shorthand

Train/test split is temporal (not random) to respect time-series causality.
Output: a single flat DataFrame per split with all features already joined.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"
INPUT_FILE = DATA_DIR / "preprocessed" / "transactions_preprocessed.csv"
FEATURE_STORE = DATA_DIR / "feature_store"

TRAIN_FRACTION = 0.8


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by time, not randomly."""
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    cutoff = t_min + (t_max - t_min) * train_fraction
    df_train = df[df["timestamp"] < cutoff].reset_index(drop=True)
    df_test = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df_train, df_test, cutoff


class FeatureBuilder:
    """
    Builds a single flat feature table per split (transaction granularity).

    User and user-product aggregates are always computed from train data only
    and joined onto each transaction row — test rows never carry future info.

    State stored during fit():
      reference_date_          — max timestamp in training data
      _last_train_ts_          — per-user last training timestamp (bridges lag at split boundary)
      _user_features_          — frozen user aggregates from train only
      _user_product_features_  — frozen (user, product) aggregates from train only
      _user_defaults_          — global column means for users unseen at fit time
    """

    def fit(self, df_train: pd.DataFrame) -> "FeatureBuilder":
        df = df_train.copy()
        df["total_value"] = df["quantity"] * df["price"]

        self.reference_date_: pd.Timestamp = df["timestamp"].max()
        self._last_train_ts_: pd.Series = df.groupby("user_id")["timestamp"].max()

        self._fit_user_features(df)
        self._fit_user_product_features(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self._transform_transactions(df)

        result = result.merge(self._user_features_, on="user_id", how="left")
        user_numeric_cols = [c for c in self._user_defaults_]
        for col in user_numeric_cols:
            if col in result.columns:
                result[col] = result[col].fillna(self._user_defaults_[col])

        result = result.merge(
            self._user_product_features_, on=["user_id", "product_id"], how="left"
        )
        return result

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df_train).transform(df_train)

    def _fit_user_features(self, df: pd.DataFrame) -> None:
        agg = df.groupby("user_id").agg(
            total_transactions=("timestamp", "count"),
            user_total_spend=("total_value", "sum"),
            user_avg_order_value=("total_value", "mean"),
            unique_products=("product_id", "nunique"),
            first_purchase=("timestamp", "min"),
            last_purchase=("timestamp", "max"),
            user_avg_quantity=("quantity", "mean"),
        ).reset_index()

        ref = self.reference_date_
        agg["days_since_last_purchase"] = (
            ref - agg["last_purchase"]
        ).dt.total_seconds().div(86_400)

        agg["customer_lifetime_days"] = (
            agg["last_purchase"] - agg["first_purchase"]
        ).dt.total_seconds().div(86_400)

        agg["avg_days_between_purchases"] = (
            agg["customer_lifetime_days"] / (agg["total_transactions"] - 1)
        ).where(agg["total_transactions"] > 1)

        cutoff_30 = ref - pd.Timedelta(days=30)
        cutoff_90 = ref - pd.Timedelta(days=90)

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

        self._user_features_ = agg.drop(columns=["first_purchase", "last_purchase"])

        numeric_cols = (
            self._user_features_
            .select_dtypes("number")
            .columns
            .difference(["user_id"])
        )
        self._user_defaults_: dict = self._user_features_[numeric_cols].mean().to_dict()

    def _fit_user_product_features(self, df: pd.DataFrame) -> None:
        agg = df.groupby(["user_id", "product_id"]).agg(
            up_purchase_count=("timestamp", "count"),
            up_total_spend=("total_value", "sum"),
            up_avg_quantity=("quantity", "mean"),
            first_purchase=("timestamp", "min"),
            last_purchase=("timestamp", "max"),
        ).reset_index()

        agg["up_days_since_last_purchase"] = (
            self.reference_date_ - agg["last_purchase"]
        ).dt.total_seconds().div(86_400)

        last_qty = (
            df.sort_values("timestamp")
            .groupby(["user_id", "product_id"])["quantity"]
            .last()
            .rename("up_last_quantity")
            .reset_index()
        )
        agg = agg.merge(last_qty, on=["user_id", "product_id"], how="left")
        self._user_product_features_ = agg.drop(columns=["first_purchase", "last_purchase"])

    # ------------------------------------------------------------------ #
    # Transform helpers
    # ------------------------------------------------------------------ #

    def _transform_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        t = df.copy().sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        t["total_value"] = t["quantity"] * t["price"]
        t["day_of_week"] = t["timestamp"].dt.dayofweek
        t["month"] = t["timestamp"].dt.month
        t["is_weekend"] = t["day_of_week"].isin([5, 6]).astype(int)

        t["days_since_prev_purchase"] = (
            t.groupby("user_id")["timestamp"]
            .diff()
            .dt.total_seconds()
            .div(86_400)
        )

        # Bridge the gap at the train/test boundary: for the first transaction
        # per user in this split, the NaN lag is filled with the distance to
        # that user's last training timestamp — only when the row is a genuine
        # test row (timestamp strictly after _last_train_ts).
        last_ts = self._last_train_ts_.rename("_last_train_ts")
        t = t.join(last_ts, on="user_id")
        fill_mask = (
            t["days_since_prev_purchase"].isna()
            & t["_last_train_ts"].notna()
            & (t["timestamp"] > t["_last_train_ts"])
        )
        t.loc[fill_mask, "days_since_prev_purchase"] = (
            (t.loc[fill_mask, "timestamp"] - t.loc[fill_mask, "_last_train_ts"])
            .dt.total_seconds()
            .div(86_400)
        )
        return t.drop(columns=["_last_train_ts"])


def main():
    print("Loading data...")
    df = load_data()

    df_train, df_test, cutoff = temporal_split(df)
    print(
        f"Temporal split at {cutoff.date()}: "
        f"{len(df_train):,} train rows / {len(df_test):,} test rows"
    )

    builder = FeatureBuilder()

    print("Fitting on train split (test data not seen)...")
    train_features = builder.fit_transform(df_train)
    print(f"  Reference date: {builder.reference_date_.date()}")

    print("Transforming test split with frozen train statistics...")
    test_features = builder.transform(df_test)

    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    train_path = FEATURE_STORE / "train.csv"
    test_path = FEATURE_STORE / "test.csv"

    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)

    print(f"  {train_path.relative_to(DATA_DIR.parent)} -> {len(train_features):,} rows, {len(train_features.columns)} cols")
    print(f"  {test_path.relative_to(DATA_DIR.parent)} -> {len(test_features):,} rows, {len(test_features.columns)} cols")
    print("\nDone.")


if __name__ == "__main__":
    main()
