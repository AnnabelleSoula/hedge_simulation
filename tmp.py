import pandas as pd
import numpy as np
import math

pd.options.display.float_format = "{:,.2f}".format

# Input 
btc_spot=120000
expected_change=-0.5
simulation_period_month=12
initial_btc_position=1929
expected_monthly_btc_interest=7.5
yearly_funding=0.072327
pct_funding_allocated_to_options=0.7
strike_premium=40000
iv=0.44
r=0

# --- Option pricing ---
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot, strike, ttm_years, iv, r=0):
    """Black–Scholes call option price."""
    if ttm_years <= 0:
        return max(spot - strike, 0.0)
    if spot <= 0 or strike <= 0 or iv <= 0:
        return max(spot - strike, 0.0)

    sigma_sqrt_t = iv * math.sqrt(ttm_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * ttm_years) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return spot * _norm_cdf(d1) - strike * math.exp(-r * ttm_years) * _norm_cdf(d2)


# --- Price path generator ---
def make_linear_path(spot, months, expected_change):
    """
    Create a linear BTC path over N months, ending at spot*(1+expected_change).
    """
    final_price = spot * (1 + expected_change)
    step = (final_price - spot) / (months - 1)
    return [spot + step * i for i in range(months)]


# --- Core simulation ---
def simulate_hedge_scenario(
    btc_spot=btc_spot,
    expected_change=expected_change,
    simulation_period_month=simulation_period_month,
    initial_btc_position=initial_btc_position,
    expected_monthly_btc_interest=expected_monthly_btc_interest,
    yearly_funding=yearly_funding,
    pct_funding_allocated_to_options=pct_funding_allocated_to_options,
    strike_premium=strike_premium,
    iv=iv,
    r=r,
):
    """
    Simulate the systematic hedging strategy under a linear BTC price path.
    Returns: hedge_df (full time series) and summary (dict of P&L metrics)
    """

    hedge_df = pd.DataFrame()
    hedge_df["btc_price"] = make_linear_path(
        btc_spot, simulation_period_month, expected_change
    )

    # Perp position declines as you sell received BTC each month
    hedge_df["perp_position"] = expected_monthly_btc_interest * (
        simulation_period_month - hedge_df.index - 1
    )

    # Monthly income from selling received BTC, starts at the end of the first month (so we shift)
    hedge_df["profit_from_selling_monthly_interest"] = (
        expected_monthly_btc_interest * hedge_df["btc_price"].shift(1)
    )
    hedge_df["profit_from_selling_monthly_interest"].iloc[0] = 0

    # Profit from reducing short hedge (BTC ↓ → gain on short)
    hedge_df["profit_from_reducing_hedge"] = (
        (hedge_df["btc_price"].iloc[0] - hedge_df["btc_price"])
        * expected_monthly_btc_interest
    )

    # Funding yield, starts at the end of the first month (so we shift)
    hedge_df["funding_payments"] = (
        hedge_df["perp_position"].shift(1) * hedge_df["btc_price"].shift(1) * yearly_funding / 12
    )
    hedge_df["funding_payments"].iloc[0] = 0

    # Option pricing
    hedge_df["strike_price_yearly_calls"] = hedge_df["btc_price"] + strike_premium
    hedge_df["yearly_call_price"] = hedge_df["btc_price"].apply(
        lambda s: bs_call_price(s, s + strike_premium, 1.0, iv, r)
    )

    hedge_df["num_yearly_calls"] = (
        pct_funding_allocated_to_options
        * hedge_df["funding_payments"]
        / hedge_df["yearly_call_price"]
    ).fillna(0)

    # Rolling call P&L
    price_prev_old, price_prev_new, roll_call_pnl = [None], [None], [0]

    for i in range(1, simulation_period_month):
        s_prev = hedge_df["btc_price"].iloc[i - 1]
        s_now = hedge_df["btc_price"].iloc[i]
        strike_prev = s_prev + strike_premium
        ttm_now = max(11 / 12, 0)

        price_old = hedge_df["yearly_call_price"].iloc[i - 1]
        price_new = bs_call_price(s_now, strike_prev, ttm_now, iv, r)
        pnl_per_call = price_new - price_old
        n_calls = hedge_df["num_yearly_calls"].iloc[i - 1]
        pnl = pnl_per_call * n_calls

        price_prev_old.append(price_old)
        price_prev_new.append(price_new)
        roll_call_pnl.append(pnl)

    while len(price_prev_old) < simulation_period_month:
        price_prev_old.append(price_prev_old[-1])
        price_prev_new.append(price_prev_new[-1])
        roll_call_pnl.append(0)

    hedge_df["price_prev_old"] = price_prev_old
    hedge_df["price_prev_new"] = price_prev_new
    hedge_df["roll_call_pnl"] = roll_call_pnl

    # --- Aggregate P&L components ---
    profit_without_hedge = hedge_df["profit_from_selling_monthly_interest"].sum()
    profit_from_hedge = hedge_df["profit_from_reducing_hedge"].sum()
    profit_from_funding_total = hedge_df["funding_payments"].sum()
    profit_from_funding_after_reinvesting_in_options = (
        (1 - pct_funding_allocated_to_options) * hedge_df["funding_payments"]
    ).sum()
    profit_from_options = hedge_df["roll_call_pnl"].sum()

    total_profit_with_hedge = (
        profit_without_hedge + profit_from_hedge + profit_from_funding_after_reinvesting_in_options + profit_from_options
    )

    summary = dict(
        profit_without_hedge=profit_without_hedge,
        profit_from_funding_total=profit_from_funding_total, 
        profit_from_funding_after_reinvesting_in_options=profit_from_funding_after_reinvesting_in_options,
        profit_from_options=profit_from_options,
        profit_from_reducing_hedge=profit_from_hedge,
        total_profit_with_hedge=total_profit_with_hedge,
        delta_profit_hedge = total_profit_with_hedge - profit_without_hedge
    )

    return hedge_df, summary


# --- Scenario comparison ---
def summarize_scenarios(sim_results: dict):
    """
    Combine multiple scenario summaries into a single DataFrame.
    sim_results = {
        'minus_50pct_1y': summary_dict,
        'plus_50pct_1y': summary_dict,
        ...
    }
    """
    records = []
    for name, summary in sim_results.items():
        row = {"simulation_name": name}
        row.update(summary)
        records.append(row)
    return pd.DataFrame(records)


def simulate_spread_and_puts(
    btc_spot=btc_spot,
    expected_change=expected_change,
    simulation_period_month=simulation_period_month,
    initial_btc_position=initial_btc_position,
    expected_monthly_btc_interest=expected_monthly_btc_interest,
    yearly_funding=yearly_funding,
    pct_funding_allocated_to_options=pct_funding_allocated_to_options,
    strike_premium=strike_premium,
    iv=iv,
    r=r,
    option_spread=5000,
    put_maturity_years=0.25,
    put_moneyness=1,
):
    """
    Simulates rolling monthly bull spreads and puts:
      - Each month, sell the old spread and put (mark-to-market P&L)
      - Reinvest in new 1Y bull spread and short-term put
      - Tracks P&L and detailed price evolution for debugging
    """

    df = pd.DataFrame()
    df["btc_price"] = make_linear_path(btc_spot, simulation_period_month, expected_change)

    # --- Funding base ---
    df["perp_position"] = expected_monthly_btc_interest * (
        simulation_period_month - df.index - 1
    )
    df["funding_payments"] = (
        df["perp_position"].shift(1)
        * df["btc_price"].shift(1)
        * yearly_funding
        / 12
    )
    df["funding_payments"].iloc[0] = 0

    # --- Option pricing helpers ---
    def bull_spread_price(s, ttm_years):
        k1 = s + strike_premium
        k2 = k1 + option_spread
        c1 = bs_call_price(s, k1, ttm_years, iv, r)
        c2 = bs_call_price(s, k2, ttm_years, iv, r)
        return c1 - c2

    def put_price(s, ttm_years):
        k_put = s * put_moneyness
        return bs_call_price(s, k_put, ttm_years, iv, r) - (s - k_put)  # via put–call parity

    # --- Initial pricing (t=0) ---
    df["bull_spread_price"] = df["btc_price"].apply(lambda s: bull_spread_price(s, 1.0))
    df["naked_call_price"] = df["btc_price"].apply(lambda s: bs_call_price(s, s + strike_premium, 1.0, iv, r))
    df["spread_saving_per_call"] = df["naked_call_price"] - df["bull_spread_price"]
    df["num_spreads"] = (
        pct_funding_allocated_to_options * df["funding_payments"] / df["bull_spread_price"]
    ).fillna(0)

    df["total_saving_usd"] = df["num_spreads"] * df["spread_saving_per_call"]
    df["put_price"] = df["btc_price"].apply(lambda s: put_price(s, put_maturity_years))
    df["num_puts"] = (df["total_saving_usd"] / df["put_price"]).fillna(0)

    # --- Rolling mark-to-market P&L tracking ---
    bull_pnl, put_pnl = [0], [0]
    bull_old_price, bull_new_price = [None], [None]
    put_old_price, put_new_price = [None], [None]

    for i in range(1, simulation_period_month):
        s_prev, s_now = df["btc_price"].iloc[i - 1], df["btc_price"].iloc[i]

        # Bull spread old/new values
        old_price_bull = bull_spread_price(s_prev, 1.0)
        new_price_bull = bull_spread_price(s_now, max(11 / 12, 0))
        pnl_bull = (new_price_bull - old_price_bull) * df["num_spreads"].iloc[i - 1]

        # Put old/new values
        old_price_put = put_price(s_prev, put_maturity_years)
        new_price_put = put_price(s_now, max(put_maturity_years - 1 / 12, 0))
        pnl_put = (new_price_put - old_price_put) * df["num_puts"].iloc[i - 1]

        bull_pnl.append(pnl_bull)
        put_pnl.append(pnl_put)
        bull_old_price.append(old_price_bull)
        bull_new_price.append(new_price_bull)
        put_old_price.append(old_price_put)
        put_new_price.append(new_price_put)

    # Extend lists to full length (padding)
    while len(bull_pnl) < simulation_period_month:
        bull_pnl.append(0)
        put_pnl.append(0)
        bull_old_price.append(bull_old_price[-1])
        bull_new_price.append(bull_new_price[-1])
        put_old_price.append(put_old_price[-1])
        put_new_price.append(put_new_price[-1])

    # --- Store details in DataFrame ---
    df["bull_price_old"] = bull_old_price
    df["bull_price_new"] = bull_new_price
    df["put_price_old"] = put_old_price
    df["put_price_new"] = put_new_price
    df["bull_pnl"] = bull_pnl
    df["put_pnl"] = put_pnl
    df["total_option_pnl"] = df["bull_pnl"] + df["put_pnl"]

    # --- Summary ---
    summary = dict(
        total_bull_pnl=df["bull_pnl"].sum(),
        total_put_pnl=df["put_pnl"].sum(),
        total_option_pnl=df["total_option_pnl"].sum(),
        avg_bull_spread_price=df["bull_spread_price"].mean(),
        avg_put_price=df["put_price"].mean(),
        total_spread_saving_usd=df["total_saving_usd"].sum(),
        avg_saving_per_call=df["spread_saving_per_call"].mean(),
    )

    return df, summary


def simulate_spread_and_puts(
    btc_spot=btc_spot,
    expected_change=expected_change,
    simulation_period_month=simulation_period_month,
    initial_btc_position=initial_btc_position,
    expected_monthly_btc_interest=expected_monthly_btc_interest,
    yearly_funding=yearly_funding,
    pct_funding_allocated_to_options=pct_funding_allocated_to_options,
    strike_premium=strike_premium,
    iv=iv,
    r=r,
    option_spread=5000,
    put_moneyness=0.95,
):
    """
    Simulates rolling 1Y bull spreads (mark-to-market monthly)
    and 1M puts held to expiry.
    """

    df = pd.DataFrame()
    df["btc_price"] = make_linear_path(btc_spot, simulation_period_month, expected_change)

    # --- Funding base ---
    df["perp_position"] = expected_monthly_btc_interest * (
        simulation_period_month - df.index - 1
    )
    df["funding_payments"] = (
        df["perp_position"].shift(1)
        * df["btc_price"].shift(1)
        * yearly_funding
        / 12
    )
    df["funding_payments"].iloc[0] = 0

    # --- Option pricing helpers ---
    def bull_spread_price(s, ttm_years):
        k1 = s + strike_premium
        k2 = k1 + option_spread
        c1 = bs_call_price(s, k1, ttm_years, iv, r)
        c2 = bs_call_price(s, k2, ttm_years, iv, r)
        return c1 - c2

    def put_price(s, k, ttm_years=1/12):
        """1M put premium"""
        # use put–call parity for convenience
        return bs_call_price(s, k, ttm_years, iv, r) - (s - k)

    # --- Initial pricing ---
    df["bull_spread_price"] = df["btc_price"].apply(lambda s: bull_spread_price(s, 1.0))
    df["naked_call_price"] = df["btc_price"].apply(lambda s: bs_call_price(s, s + strike_premium, 1.0, iv, r))
    df["spread_saving_per_call"] = df["naked_call_price"] - df["bull_spread_price"]

    df["num_spreads"] = (
        pct_funding_allocated_to_options * df["funding_payments"] / df["bull_spread_price"]
    ).fillna(0)
    df["total_saving_usd"] = df["num_spreads"] * df["spread_saving_per_call"]

    # --- Rolling monthly bull spreads and expiring 1M puts ---
    bull_pnl = [0]
    put_pnl = [0]
    bull_old_price = [None]
    bull_new_price = [None]
    put_strikes = [None]

    for i in range(1, simulation_period_month):
        s_prev, s_now = df["btc_price"].iloc[i - 1], df["btc_price"].iloc[i]

        # ---- Bull Spread P&L (mark-to-market) ----
        old_bull_price = bull_spread_price(s_prev, 1.0)
        new_bull_price = bull_spread_price(s_now, max(11 / 12, 0))
        pnl_bull = (new_bull_price - old_bull_price) * df["num_spreads"].iloc[i - 1]

        # ---- Put leg: buy 1M put at t=i-1, strike = 95% * S_prev ----
        k_put = s_prev * put_moneyness
        premium_put = put_price(s_prev, k_put)
        payout_put = max(k_put - s_now, 0)   # intrinsic value at expiry
        pnl_put = (payout_put - premium_put) * (df["total_saving_usd"].iloc[i - 1] / premium_put if premium_put > 0 else 0)

        # store
        bull_pnl.append(pnl_bull)
        put_pnl.append(pnl_put)
        bull_old_price.append(old_bull_price)
        bull_new_price.append(new_bull_price)
        put_strikes.append(k_put)

    # pad lists
    while len(bull_pnl) < simulation_period_month:
        bull_pnl.append(0)
        put_pnl.append(0)
        bull_old_price.append(bull_old_price[-1])
        bull_new_price.append(bull_new_price[-1])
        put_strikes.append(put_strikes[-1])

    df["bull_price_old"] = bull_old_price
    df["bull_price_new"] = bull_new_price
    df["bull_pnl"] = bull_pnl
    df["put_strike"] = put_strikes
    df["put_pnl"] = put_pnl
    df["total_option_pnl"] = df["bull_pnl"] + df["put_pnl"]

    summary = dict(
        total_bull_pnl=df["bull_pnl"].sum(),
        total_put_pnl=df["put_pnl"].sum(),
        total_option_pnl=df["total_option_pnl"].sum(),
        avg_bull_spread_price=df["bull_spread_price"].mean(),
        avg_saving_per_call=df["spread_saving_per_call"].mean(),
    )

    return df, summary


if __name__ == '__main__':

    # Input 
    btc_spot=120000
    expected_change=-0.5
    simulation_period_month=12
    initial_btc_position=1929
    expected_monthly_btc_interest=7.5
    yearly_funding=0.072327
    pct_funding_allocated_to_options=0.7
    strike_premium=0
    iv=0.44
    r=0

    # Define scenarios
    scenarios = {
        "minus_50pct_1y": (-0.5, 12),
        "minus_50pct_2y": (-0.5, 24),
        "flat_1y": (0.0, 12),
        "plus_50pct_1y": (0.5, 12),
        "plus_100pct_1y": (1, 12),
        "plus_200pct_1y": (2, 12),
    }

    sim_results = {}
    dfs = {}

    for name, (change, months) in scenarios.items():
        df, summary = simulate_hedge_scenario(expected_change=change, simulation_period_month=months)
        dfs[name] = df
        sim_results[name] = summary

        print(name)
        print(df)
        print('--------------------------')

    # Build summary table
    summary_df = summarize_scenarios(sim_results)

    # --- Save everything to Excel ---
    output_path = "/Users/annabellesoula/Documents/GitHub/check-violations/btc_hedge_simulation_results_premium.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Tab 1: summary
        summary_df.to_excel(writer, sheet_name="summary", index=False)

        # Tabs 2+: one per scenario
        for name, df in dfs.items():
            # Excel sheet names can’t exceed 31 chars
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    print(summary_df)
    print('done')

    sim_results_spread = {}
    dfs_spread = {}

    for name, (change, months) in scenarios.items():
        df_spread, summary_spread = simulate_spread_and_puts(
            expected_change=change,
            simulation_period_month=months
        )
        dfs_spread[name] = df_spread
        sim_results_spread[name] = summary_spread

        print(f"Scenario: {name}")
        print(df_spread.head())
        print('--------------------------')

    # Build summary table for all spread scenarios
    summary_spread_df = summarize_scenarios(sim_results_spread)

    # --- Save everything to Excel ---
    output_path_spread = "/Users/annabellesoula/Documents/GitHub/check-violations/btc_hedge_bull_spread_simulation_results.xlsx"

    with pd.ExcelWriter(output_path_spread, engine="openpyxl") as writer:
        # Tab 1: summary
        summary_spread_df.to_excel(writer, sheet_name="summary", index=False)

        # Tabs 2+: one per scenario
        for name, df in dfs_spread.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    print("\nBull spread + puts simulation results summary:")
    print(summary_spread_df)
    print('done')
