import pandas as pd

#! TODO:
# check if flat branch is not in fact exactly the same mecanism than up branch
# add a column unrealised pnl from remaining hedge position  
# cumulative pnl for each category, avg cumul pnl per category, number of path per category


def build_btc_trinomial_tree_rule_based(
    start_price=120_000,
    step=5_000,
    months=12,
    expected_btc_interest=7.5,
    yearly_funding_rate=0.07
):
    """
    Build a BTC hedge decision tree with:
      - Hedge adjustments (sell interest / reduce hedge)
      - Cumulative realized P&L
      - Monthly funding income from short position
      - Path-level tracking of min/max BTC prices

    Rules:
      * Sell interest only if BTC >= initial (month 0) price
      * Reduce hedge if BTC < initial price
      * Monthly funding fee on current hedge notional (before adjustment)
    """

    initial_hedge = months * expected_btc_interest
    reference_price = start_price

    nodes = [{
        "month": 0,
        "path": "",
        "btc_price": start_price,
        "path_max_price": start_price,
        "path_min_price": start_price,
        "btc_price_below_initial": False,
        "sell_interest_at_spot": False,
        "profit_from_selling_interest": 0.0,
        "reduce_hedge": False,
        "profit_from_reducing_hedge": 0.0,
        "hedge_short_position": initial_hedge,
        "remaining_interest": initial_hedge,
        "monthly_funding_fee": 0.0,
        "cumulative_pnl": 0.0,
        "action": "init"
    }]

    for m in range(1, months + 1):
        new_nodes = []

        for n in [x for x in nodes if x["month"] == m - 1]:
            prev_price = n["btc_price"]
            prev_hedge = n["hedge_short_position"]
            remaining = max(n["remaining_interest"] - expected_btc_interest, 0)
            cum_pnl = n["cumulative_pnl"]
            prev_max = n["path_max_price"]
            prev_min = n["path_min_price"]

            # Monthly funding (before adjustment)
            monthly_funding_fee = prev_hedge * prev_price * (yearly_funding_rate / 12)

            # --- Common logic builder ---
            def make_node(price, move_label):
                sell_interest = price >= reference_price
                reduce_hedge = not sell_interest
                profit_sell = expected_btc_interest * price if sell_interest else 0.0
                profit_hedge = (
                    expected_btc_interest * (reference_price - price)
                    if reduce_hedge else 0.0
                )
                new_hedge = (
                    max(prev_hedge - expected_btc_interest, 0)
                    if reduce_hedge else prev_hedge
                )
                return {
                    "month": m,
                    "path": n["path"] + move_label,
                    "btc_price": price,
                    "path_max_price": max(prev_max, price),
                    "path_min_price": min(prev_min, price),
                    "btc_price_below_initial": price < reference_price,
                    "sell_interest_at_spot": sell_interest,
                    "profit_from_selling_interest": profit_sell,
                    "reduce_hedge": reduce_hedge,
                    "profit_from_reducing_hedge": profit_hedge,
                    "hedge_short_position": new_hedge,
                    "remaining_interest": remaining,
                    "monthly_funding_fee": monthly_funding_fee,
                    "cumulative_pnl": cum_pnl + profit_sell + profit_hedge + monthly_funding_fee,
                    "action": (
                        f"price {move_label} → "
                        f"{'sell interest' if sell_interest else 'reduce hedge'}"
                    )
                }

            # --- Branches ---
            new_nodes.append(make_node(prev_price + step, "U"))
            new_nodes.append(make_node(prev_price, "F"))
            new_nodes.append(make_node(prev_price - step, "D"))

        nodes.extend(new_nodes)

    return pd.DataFrame(nodes)


def classify_path(row, start_price, step, months, alpha=0.25, n_consecutive=3):
    """
    Classify each final path into categories based on price bounds and sequence patterns.
    """
    path = row["path"]
    final_price = row["btc_price"]
    path_min = row["path_min_price"]
    path_max = row["path_max_price"]

    max_move = step * months
    lower = start_price - alpha * max_move
    upper = start_price + alpha * max_move

    # Helper: compress repeated moves (e.g. "UUUDDD" → "UD")
    compressed = []
    for ch in path:
        if not compressed or compressed[-1] != ch:
            compressed.append(ch)

    # Range-Bound
    if (lower <= final_price <= upper) and (path_min >= lower) and (path_max <= upper):
        return "range_bound"

    # Extreme monotonic
    if path == "U" * len(path):
        return "extreme_uptrend"
    if path == "D" * len(path):
        return "extreme_downtrend"

    # Sharp reversal (V-shape): one switch, no 'F'
    if "F" not in path and len(compressed) == 2 and set(compressed).issubset({"U", "D"}):
        return "sharp_reversal"

    # Uptrend (Bull)
    if (final_price > upper) and ("U" * n_consecutive in path):
        return "bull"

    # Downtrend (Bear)
    if (final_price < lower) and ("D" * n_consecutive in path):
        return "bear"

    return "other"

def build_tree_and_classify(start_price=120_000,
                            step=5_000,
                            months=12,
                            alpha = 0.5):

    df_tree = build_btc_trinomial_tree_rule_based(
        start_price=start_price,
        step=step,
        months=months,
        expected_btc_interest=7.5
    )

    # Select final nodes (one per complete path)
    final_nodes = df_tree.groupby("path").tail(1).copy()

    # Apply classification
    final_nodes["category"] = final_nodes.apply(
        classify_path,
        axis=1,
        args=(start_price, step, months, alpha),
    )

    # Merge classification back into full tree
    df_tree = df_tree.merge(
        final_nodes[["path", "category"]],
        on="path",
        how="left"
    )

    df_last_month = df_tree[df_tree['month'] == df_tree['month'].max()]

    df_category_stat = (
        df_last_month.groupby("category")
        .agg(
            count=("path", "count"),
            pnl_min=("cumulative_pnl", "min"),
            pnl_max=("cumulative_pnl", "max"),
            pnl_mean=("cumulative_pnl", "mean"),
            # pnl_median=("cumulative_pnl", "median"),
            avg_funding_income=("monthly_funding_fee", "mean"),
            total_funding_income=("monthly_funding_fee", "sum")
        )
        .reset_index()
    )


    return df_tree, df_category_stat

# --- Example usage ---
if __name__ == "__main__":

    start_price=120_000
    step=5_000
    months=12

    alpha = 0.5

    df_tree = build_btc_trinomial_tree_rule_based(
        start_price=start_price,
        step=step,
        months=months,
        expected_btc_interest=7.5
    )

    print(df_tree.head(20))

    # Select final nodes (one per complete path)
    final_nodes = df_tree.groupby("path").tail(1).copy()

    # Apply classification
    final_nodes["category"] = final_nodes.apply(
        classify_path,
        axis=1,
        args=(start_price, step, months, alpha),
    )

    # Merge classification back into full tree
    df_tree = df_tree.merge(
        final_nodes[["path", "category"]],
        on="path",
        how="left"
    )

    df_last_month = df_tree[df_tree['month'] == df_tree['month'].max()]

    # Export to Excel for analysis
    output_path = "/Users/annabellesoula/Documents/GitHub/hedge_simulation/btc_trinomial_rule_tree_v2.xlsx"
    df_tree.to_excel(output_path, index=False)
    print(f"Saved to {output_path}")



