import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# 1. Load data
df = pd.read_csv("transactions.csv")


# 2. Create baskets
transactions = (
    df.groupby("Transaction_ID")["Product"]
      .apply(list)
      .tolist()
)


# 3. One-hot encode
te = TransactionEncoder()

basket_array = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    basket_array,
    columns=te.columns_
)


# 4. Find frequent itemsets
frequent_itemsets = apriori(
    basket,
    min_support=0.03,
    use_colnames=True
)


# 5. Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3
)


# 6. Keep useful columns
rules = rules[
    [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]
]


# 7. Sort by lift
rules = rules.sort_values(
    "lift",
    ascending=False
)


print(rules)