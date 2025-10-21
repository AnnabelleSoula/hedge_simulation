import pandas as pd

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

    Rules:
      * Sell interest only if BTC > initial (month 0) price
      * Reduce hedge if BTC < initial price
      * Monthly funding fee on current hedge notional (before adjustment)
    """

    initial_hedge = months * expected_btc_interest  # initial short hedge position (BTC)
    reference_price = start_price  # the anchor price used for comparison

    nodes = [{
        "month": 0,
        "path": "",
        "btc_price": start_price,
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

            # Monthly funding before hedge change
            monthly_funding_fee = prev_hedge * prev_price * (yearly_funding_rate / 12)

            # --- UP branch ---
            price_up = prev_price + step
            if price_up >= reference_price:
                sell_interest = True
                profit_sell = expected_btc_interest * price_up
            else:
                sell_interest = False
                profit_sell = 0.0
            profit_hedge = 0.0
            new_nodes.append({
                "month": m,
                "path": n["path"] + "U",
                "btc_price": price_up,
                "btc_price_below_initial": price_up < reference_price,
                "sell_interest_at_spot": sell_interest,
                "profit_from_selling_interest": profit_sell,
                "reduce_hedge": not sell_interest,
                "profit_from_reducing_hedge": 0.0,
                "hedge_short_position": prev_hedge,
                "remaining_interest": remaining,
                "monthly_funding_fee": monthly_funding_fee,
                "cumulative_pnl": cum_pnl + profit_sell + monthly_funding_fee,
                "action": "price up → sell interest if above initial, keep hedge"
            })

            # --- FLAT branch ---
            price_flat = prev_price
            if price_flat >= reference_price:
                sell_interest = True
                profit_sell = expected_btc_interest * price_flat
            else:
                sell_interest = False
                profit_sell = 0.0
            profit_hedge = 0.0
            new_nodes.append({
                "month": m,
                "path": n["path"] + "F",
                "btc_price": price_flat,
                "btc_price_below_initial": price_flat < reference_price,
                "sell_interest_at_spot": sell_interest,
                "profit_from_selling_interest": profit_sell,
                "reduce_hedge": not sell_interest,
                "profit_from_reducing_hedge": profit_hedge,
                "hedge_short_position": prev_hedge,
                "remaining_interest": remaining,
                "monthly_funding_fee": monthly_funding_fee,
                "cumulative_pnl": cum_pnl + profit_sell + monthly_funding_fee,
                "action": "price flat → sell interest if above initial, keep hedge"
            })

            # --- DOWN branch ---
            price_down = prev_price - step
            sell_interest = False
            profit_sell = 0.0
            profit_hedge = expected_btc_interest * reference_price
            new_nodes.append({
                "month": m,
                "path": n["path"] + "D",
                "btc_price": price_down,
                "btc_price_below_initial": price_down < reference_price,
                "sell_interest_at_spot": sell_interest,
                "profit_from_selling_interest": profit_sell,
                "reduce_hedge": True,
                "profit_from_reducing_hedge": profit_hedge,
                "hedge_short_position": max(prev_hedge - expected_btc_interest, 0),
                "remaining_interest": remaining,
                "monthly_funding_fee": monthly_funding_fee,
                "cumulative_pnl": cum_pnl + profit_hedge + monthly_funding_fee,
                "action": "price down → reduce hedge (no sale)"
            })

        nodes.extend(new_nodes)

    return pd.DataFrame(nodes)



# --- Example usage ---
if __name__ == "__main__":
    df_tree = build_btc_trinomial_tree_rule_based(
        start_price=120_000,
        step=5_000,
        months=6,
        expected_btc_interest=7.5
    )

    print(df_tree.head(20))

    # Export to Excel for analysis
    output_path = "/Users/annabellesoula/Documents/GitHub/hedge_simulation/btc_trinomial_rule_tree_v2.xlsx"
    df_tree.to_excel(output_path, index=False)
    print(f"Saved to {output_path}")



