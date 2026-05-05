from faker import Faker
import numpy as np
import pandas as pd

fake = Faker()
np.random.seed(42)


def generate_transactions(n_users=100, n_products=30, max_tx=100):
    users = [f"user_{i}" for i in range(n_users)]
    products = [f"product_{i}" for i in range(n_products)]

    weights = np.linspace(1, 2, len(products))
    product_probs = weights / weights.sum()

    product_prices = {p: np.round(np.random.uniform(10, 100), 2) for p in products}

    data = []
    for user in users:
        n = np.random.randint(800, 1200)  # 👈 key change

        for _ in range(n):
            product = np.random.choice(products, p=product_probs)
            days_ago = int(np.random.beta(a=2, b=5) * 600)
            timestamp = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
            #timestamp = fake.date_time_between(
            #    start_date='-600d',
            #    end_date='now'
            #)

            quantity = np.random.randint(1, 4)
            price = product_prices[product]

            data.append([user, product, timestamp, quantity, price])

    df = pd.DataFrame(
        data, columns=["user_id", "product_id", "timestamp", "quantity", "price"]
    )

    return df


if __name__ == "__main__":
    df = generate_transactions()

    # -----------------------------
    # Save dataset
    # -----------------------------
    df.to_csv("transactions.csv", index=False)

    # -----------------------------
    # Quick sanity checks (great for teaching)
    # -----------------------------
    print("Total rows:", len(df))
    print("Unique users:", df["user_id"].nunique())
    print("Unique products:", df["product_id"].nunique())

    # Each product should have only ONE price
    print("\nPrice consistency check:")
    print(df.groupby("product_id")["price"].nunique().value_counts())
