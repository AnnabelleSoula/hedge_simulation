import pandas as pd
import numpy as np
from scipy.stats import norm





#! TODO:
# check if flat branch is not in fact exactly the same mecanism than up branch
# add a column unrealised pnl from remaining hedge position  
# cumulative pnl for each category, avg cumul pnl per category, number of path per category


# --- Black-Scholes helper ---
def bs_call_price(S, K, T, sigma, r=0.0):
    """Return Black-Scholes European call price."""
    if any(v <= 0 for v in [S, K, T, sigma]):
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

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


def build_btc_trinomial_tree_sell_interest_if_flat(
    start_price=120_000,
    step=5_000,
    months=12,
    expected_btc_interest=7.5,
    yearly_funding_rate=0.07
):
    """
    BTC hedge decision tree (mechanical version):
      * If BTC price == initial → sell interest, keep hedge (higher future funding)
      * If BTC price != initial → reduce hedge by interest amount
      * Always earn monthly funding fees on hedge notional

    This provides a non–rule-based baseline for comparison.
    """

    initial_hedge = months * expected_btc_interest
    reference_price = start_price

    nodes = [{
        "month": 0,
        "path": "",
        "btc_price": start_price,
        "path_max_price": start_price,
        "path_min_price": start_price,
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

            # Funding before hedge change
            monthly_funding_fee = prev_hedge * prev_price * (yearly_funding_rate / 12)

            # --- Node builder ---
            def make_node(price, label):
                # Price movement condition
                price_moved = price != reference_price

                if price_moved:
                    # Price changed → reduce hedge
                    reduce_hedge = True
                    sell_interest = False
                    profit_from_reducing_hedge = expected_btc_interest * (reference_price - price)
                    profit_from_selling_interest = 0.0
                    new_hedge = max(prev_hedge - expected_btc_interest, 0)
                else:
                    # Price unchanged → sell interest, keep hedge
                    reduce_hedge = False
                    sell_interest = True
                    profit_from_reducing_hedge = 0.0
                    profit_from_selling_interest = expected_btc_interest * price
                    new_hedge = prev_hedge  # unchanged

                new_cum_pnl = (
                    cum_pnl
                    + profit_from_selling_interest
                    + profit_from_reducing_hedge
                    + monthly_funding_fee
                )

                return {
                    "month": m,
                    "path": n["path"] + label,
                    "btc_price": price,
                    "path_max_price": max(prev_max, price),
                    "path_min_price": min(prev_min, price),
                    "sell_interest_at_spot": sell_interest,
                    "profit_from_selling_interest": profit_from_selling_interest,
                    "reduce_hedge": reduce_hedge,
                    "profit_from_reducing_hedge": profit_from_reducing_hedge,
                    "hedge_short_position": new_hedge,
                    "remaining_interest": remaining,
                    "monthly_funding_fee": monthly_funding_fee,
                    "cumulative_pnl": new_cum_pnl,
                    "action": (
                        f"price {label} → "
                        f"{'reduce hedge' if reduce_hedge else 'sell interest, keep hedge'}"
                    )
                }

            # --- Build branches ---
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

    # Choose one model 
    # df_tree = build_btc_trinomial_tree_rule_based(
    #     start_price=start_price,
    #     step=step,
    #     months=months,
    #     expected_btc_interest=7.5
    # )

    df_tree = build_btc_trinomial_tree_sell_interest_if_flat(
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

    # retaining upside by reinvesting funding fees into options 
    # df_tree = add_bull_spread_overlay(df_tree, spread_cap=0.20)
    df_tree = add_quarterly_call_strikes(df_tree)
    df_tree = add_quarterly_call_prices_lifecycle(df_tree, iv=0.48, r=0)
    print(df_tree.head())

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


def add_quarterly_call_strikes(df_tree: pd.DataFrame,
                               start_price: float = 120_000,
                               step: float = 5_000,
                               months: int = 12) -> pd.DataFrame:
    """
    Add regime-dependent quarterly call strike levels.
    """

    df = df_tree.copy()

    # --- Build lookup of BTC price by path for fast ancestor retrieval
    path_to_price = df.set_index("path")["btc_price"].to_dict()

    # --- For each node, find BTC price 3 months ago (same lineage)
    def lag3_price(path: str):
        if len(path) <= 3:
            return np.nan
        lag_path = path[:-3]
        return path_to_price.get(lag_path, np.nan)

    df["btc_price_lag3"] = df["path"].apply(lag3_price)

    # --- Define thresholds
    cond_above = df["btc_price"] >= start_price
    cond_mid = (df["btc_price"] < start_price) & (df["btc_price"] >= start_price - 3 * step)
    cond_below = df["btc_price"] < start_price - 3 * step

    # --- Apply regime labels
    df["call_regime"] = np.select(
        [cond_above, cond_mid, cond_below],
        ["above_initial", "below_initial", "deep_bear"],
        default="other"
    )

    # --- Assign strike prices per regime
    df["call_strike"] = np.select(
        [cond_above, cond_mid, cond_below],
        [df["btc_price"], start_price, df["btc_price_lag3"]],
        default=np.nan
    )

    # --- Handle early months (no lag data)
    df.loc[df["month"] < 3, "call_strike"] = np.where(
        df.loc[df["month"] < 3, "btc_price"] >= start_price,
        df.loc[df["month"] < 3, "btc_price"],
        start_price
    )
    df.loc[df["month"] < 3, "call_regime"] = np.where(
        df.loc[df["month"] < 3, "btc_price"] >= start_price,
        "above_initial",
        "below_initial"
    )

    return df

def add_quarterly_call_prices_lifecycle(df, iv=0.48, r=0.0):
    """
    Add rolling call option lifecycle columns:
      - call_price_old: price we paid last month to buy the call
      - call_price_now: current mark of that same call (1M later)
      - call_price_new: price of new 3M call we buy today

    Rolling relationship:
        call_price_old (m+1) = call_price_now (m)
    """
    df = df.copy()

    # --- Compute current new call price (3M tenor) ---
    df["call_price_new"] = df.apply(
        lambda row: bs_call_price(
            S=row["btc_price"],
            K=row["call_strike"],
            T=0.25,
            sigma=iv,
            r=r,
        )
        if pd.notnull(row["call_strike"]) else np.nan,
        axis=1,
    )

    # --- Compute current call mark-to-market for previous strike ---
    # (we’ll fill call_price_old later via merge)
    df["call_price_now"] = df.apply(
        lambda row: bs_call_price(
            S=row["btc_price"],
            K=row["call_strike"],
            T=max(0.25 - 1/12, 1e-6),
            sigma=iv,
            r=r,
        )
        if pd.notnull(row["call_strike"]) else np.nan,
        axis=1,
    )

    # --- Parent merge: link each node with its previous month node ---
    parent_df = df.rename(
        columns={
            "month": "month_prev",
            "path": "path_prev",
            "call_price_now": "call_price_old"  # rolling link
        }
    ).copy()
    parent_df["path_prev_child"] = parent_df["path_prev"] + ""  # just for clarity

    df["path_prev"] = df["path"].str[:-1]  # find parent
    df = df.merge(
        parent_df[["path_prev", "month_prev", "call_price_old"]],
        left_on=["path_prev", df["month"] - 1],
        right_on=["path_prev", parent_df["month_prev"]],
        how="left",
    )

    # --- Root cleanup ---
    df.loc[df["month"] == 0, ["call_price_old"]] = np.nan

    df['call_pnl'] = df['call_price_now'] - df['call_price_old']

    # --- Number of calls we can buy this month ---
    df["number_calls"] = df.apply(
        lambda row: (row["monthly_funding_fee"] / row["call_price_new"])
        if pd.notnull(row["call_price_new"]) and row["call_price_new"] > 0
        else 0.0,
        axis=1,
    )

    # --- Total PnL from calls (funded by monthly funding) ---
    df["call_pnl_total"] = df["number_calls"] * df["call_pnl"]

    # --- Fill NaN for clean numeric processing ---
    df[["call_pnl", "number_calls", "call_pnl_total"]] = df[
        ["call_pnl", "number_calls", "call_pnl_total"]
    ].fillna(0.0)

    return df
    

# def add_bull_spread_overlay(df_tree: pd.DataFrame, step: float = 5_000, start_price = 120_000, spread_cap: float = 0.20):
#     """
#     Add a bull spread overlay simulation to a BTC hedge decision tree.
#     Now correctly references previous month's BTC price and funding fee.

#     Logic:
#       - At month t, we exercise the bull spread bought at t-1.
#       - Payout depends on BTC move from (t-1) → (t):
#           * If price increased: profit = +spread_cap * funding_prev
#           * If price decreased or flat: loss = -spread_cap * funding_prev
#       - Only buy bull spreads when BTC >= start_price.
#     """

#     df = df_tree.copy()

#     # Sort to ensure proper order
#     df = df.sort_values(["path", "month"])

#     # Previous month's BTC price and funding fee
#     df["btc_price_prev"] = df.groupby(df["path"].str[:-1])["btc_price"].shift(1)
#     df["funding_prev"] = df.groupby(df["path"].str[:-1])["monthly_funding_fee"].shift(1)

#     # Compute BTC price change (month-over-month)
#     df["price_move"] = df["btc_price"] - df["btc_price_prev"]

#     # Default to no investment (e.g., month 0)
#     df["funding_prev"] = df["funding_prev"].fillna(0.0)
#     df["btc_price_prev"] = df["btc_price_prev"].fillna(df["btc_price"])
#     df["price_move"] = df["price_move"].fillna(0.0)

#     # Apply rule: only buy bull spread when BTC >= start_price
#     df["buy_bull_spread"] = df["btc_price_prev"] >= start_price

#     df["buy_bull_spread"] = df["btc_price"] >= start_price
#     df["bull_spread_profit"] = np.where(
#         df["buy_bull_spread"],
#         np.where(df["price_move"] > 0, +spread_cap * df["funding_prev"], -spread_cap * df["funding_prev"]),
#         0.0  # no spread bought → no profit/loss
#     )

#     # Replace NaN for first row in each path
#     df["bull_spread_profit"] = df["bull_spread_profit"].fillna(0.0)
#     df["funding_prev"] = df["funding_prev"].fillna(0.0)
#     df["bull_spread_profitable"] = df["bull_spread_profit"] > 0

#     # Optional: cumulative sum if you want total overlay impact
#     df["bull_spread_cum_pnl"] = df.groupby("path")["bull_spread_profit"].cumsum()

#     return df


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



    # df_with_overlay = add_bull_spread_overlay(df_tree, spread_cap=0.20)
    # print(df_with_overlay[[
    #     "month", "path", "btc_price",
    #     "monthly_funding_fee",
    #     "funding_prev",
    #     "bull_spread_range",
    #     "bull_spread_profit",
    #     "bull_spread_cum_pnl"
    # ]].head(20))


    # Export to Excel for analysis
    output_path = "/Users/annabellesoula/Documents/GitHub/hedge_simulation/btc_trinomial_rule_tree_v2.xlsx"
    df_tree.to_excel(output_path, index=False)
    print(f"Saved to {output_path}")



