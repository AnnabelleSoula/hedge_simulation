# Path Analysis

For a path dependant trinomial tree over a n periods, we have 3ⁿ path-level results. In the context of our analysis, we want to simulation a hedge over a 12 months period, so we have 3¹² = 531,441 paths!

We want to categorize paths into a few meaningful behavioral types that capture different scenario 'types' for the BTC price, then analyse hedge performance at a group level.

We suggest the following categories, the numerical values in the definition correspond to an initial BTC price of 120k.

| Category                            | Definition                                                                          | Economic intuition                                                            |
| ----------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **1. Range-bound (Neutral)**        | Final BTC price ∈ [100k, 140k] and *never* above 140k nor below 120k                | Market stays sideways, funding and hedge neutralize each other → flat PnL     |
| **2. Downtrend (Bear)**             | Path ends **below 100k** and has ≥ 3 consecutive D moves, no U move (only D or F )                           | Gradual BTC decline, hedge profit offsets loss on BTC income                  |
| **3. Uptrend (Bull)**               | Path ends **above 140k** and has ≥ 3 consecutive U moves , no D move (only U or F)                           | Gradual BTC appreciation, calls benefit, hedge loses but funding covers part  |
| **4. Sharp reversal (V-shape)**     | Path has consecutive moves in one direction followed by consecutive move in the opposite  (only one direction change, no flat)   | Classic “crash then rebound” scenario → stress test of timing and optionality |


# Hedge Strategy

### 1. Objective:

To stabilize USD cash flows from BTC-denominated income while preserving partial upside and minimizing capital outlay.

### 2. Core Hedge Structure

The base strategy consists of:

Shorting BTC perpetual futures equal to the expected yearly BTC inflows.

Each month, BTC interest is received, and we reduce the hedge of the received amount. Basically, we sell all the yearly expected interest at the current price. When current price is below initial price, the hedge is profitable. When BTC trades above the initial hedge price, the strategy continues to reduce the short hedge as interest payments are received.
However, the realized profit from this reduction is smaller than what would have been earned if the BTC interest had simply been sold outright at the higher spot price (i.e., without the hedge).
In other words, while the hedge protects against downside, it also limits upside participation once the market rises above the initial level.

We receive monthly funding fees from our short hedge position. We want to reinvest these funding fees to capture an eventual upside.
Each month, funding fees generated from the short BTC hedge are reinvested into quarterly BTC call options.
The objective is to retain convex upside exposure while the base hedge controls downside risk.

#### Strike selection rule 
| Market Condition                                            | Call Tenor | Strike Definition                   | Economic Rationale                                                                                                                   |
| ----------------------------------------------------------- | ---------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **BTC price below initial price (underwater)**              | 3 months   | **Strike = initial BTC price**      | Cheap convexity toward the original hedge anchor; benefit from potential rebound.                                                    |
| **BTC price above or equal to initial price (above water)** | 3 months   | **ATM (strike = current spot)**     | Participate in continued upside efficiently, without overpaying for deep ITM convexity.                                              |
| **BTC price far below initial price (deep bear regime)**    | 3 months   | **Strike = BTC price 3 months ago** | Maintain realistic convex exposure when the anchor strike (120 k) is too far OTM.  Avoid wasting premium on low-probability options. |

#### Position Management Approaches on options
Two implementations are possible, depending on whether the goal is smooth convex participation or discrete tail protection:

| Approach                              | Description                                                                                                                                  | Typical Behavior                                                                                 | When to Use                                                                                                                  |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Rolling 3-Month Calls Monthly** | Each month, purchase a new 3-month call (strike per the rule). After one month, mark to market and roll into a fresh 3-month maturity.       | Continuous convex exposure; smoother P&L profile; gradual participation in trending up-markets.  | Suitable when funding flows are monthly and the objective is steady convex participation (preferred for the base framework). |
| **Hold 3-Month Calls to Expiry**  | Each month, purchase a 3-month call and hold it for the full tenor until expiry. Continue buying new ones each month (overlapping vintages). | Lumpy returns: most options expire worthless, but large positive months occur on strong rallies. | Appropriate when the goal is tail-event convexity or explicit long-vol exposure.                                             |








------------------------------

#### Hedge Strategy - OLD 

1. Objective

To stabilize USD cash flows from BTC-denominated income while preserving partial upside and minimizing capital outlay.

2. Core Hedge Structure

The base strategy consists of:

Shorting BTC perpetual futures equal to the expected yearly BTC inflows.

Each month, BTC interest is received and one of two actions occurs:

If spot ≥ initial price → sell received BTC at spot to generate USD liquidity.

If spot < initial price → reduce the short hedge by the received amount (realizing hedge profits).

Throughout the year, the short perpetual position earns funding fees, typically +5–15 % annualized.

This design:

Fixes the USD value of future BTC income on the downside (crash protection).

Retains full participation in BTC upside via unhedged receipts when price rises.

Generates steady funding yield as positive carry.

3. Optional Enhancement — Using Funding Income for Additional Yield

The core hedge generates a steady funding yield from the short BTC perpetual position.
Rather than leaving this yield idle, it can be reinvested into structured overlays designed to capture upside or smooth performance.

This transforms the strategy from a pure hedge into a yield-optimized, self-financing structure.

3.1. Option 1 — Rolling Monthly Collars

Use monthly funding income to finance BTC collar strategies on newly received BTC interest:

Leg	Direction	Purpose
Buy 1-month put (≈ −10 % strike)	Long	Downside protection
Sell 1-month call (≈ +10 % strike)	Short	Finances the put
Buy equivalent BTC spot	Long	Maintain short-term exposure

Typically close to zero-cost (short call ≈ long put premium).

Captures small monthly upside while maintaining crash protection.

If BTC rises moderately, the collar gains value → extra yield.

If BTC falls, losses are limited and funding income offsets cost.

This overlay allows ongoing participation in short-term rallies without additional capital risk.

In addition: in uptrend market, instead of selling 100pct of BTC interests received, we can keep a portion of it to use in the collar. 

3.2. Option 2 — Deep-Out-of-the-Money (DOTM) Calls at Crash Lows

When BTC trades near the bottom of a large drawdown (e.g. ≈ 50 % below the initial price),
a portion of accumulated funding yield can be used to buy DOTM calls:

Leg	Direction	Example
Buy long-dated call	Long	Strike = initial price (≈ 120 k)
Underlying	—	Spot ≈ 80 k (deeply depressed)

Purpose:

Capture convex upside from post-crash recoveries.

The cost of these options is small, allowing asymmetric exposure.

Fully financed by prior funding income, so no incremental capital risk.

Effect:

In deep bear markets, the hedge + funding protects downside,
and these low-cost calls re-introduce upside convexity if BTC rebounds sharply.

3.3. General Principle

The funding yield acts as a “yield budget” — a recurring source of optionality.
Each enhancement uses that budget differently:

Collars → steady incremental yield in range-bound or slightly bullish markets.

DOTM Calls → tail convexity in deep-bear recovery scenarios.

Together, they turn a static hedge into a dynamic, self-financing yield engine that adapts to market regimes.

4. Expected Behavior by Market Regime

This section summarizes the expected cash flows and performance of each component —
Short BTC Hedge, Funding Yield, BTC Interest Sales, and the Optional Enhancements (Collars / DOTM Calls) — under various market conditions.

| Market Regime                         | BTC Price Behavior                           | Core Cash Flow Dynamics                                                                                               | Hedge Performance                         | Funding Yield                       | BTC Interest Sale                                  | Collar Enhancement                                                                | Deep OTM Calls                                                                            |
| ------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Strong Bull (+50 % or more)**       | BTC rallies persistently above initial price | BTC interest sold at high spot → large USD inflows. Hedge loses mark-to-market but partially offset by funding yield. | Negative (short BTC loses)                | Positive (funding income continues) | Large positive cash flow from selling BTC interest | Collars capped → limited benefit; still allow ~+10 % per month                    | DOTM Calls may expire worthless, as BTC far above re-entry level                          |
| **Moderate Bull (+10–30 %)**          | BTC drifts upward within moderate range      | BTC interest sold above initial price; funding yield adds extra income.                                               | Small hedge loss                          | Positive                            | Positive cash flow from BTC sales                  | Collars perform best — capture upside up to call strike → **extra yield**         | Not triggered                                                                             |
| **Range-Bound (−10 % to +10 %)**      | BTC oscillates near initial price            | Regular BTC interest sales; small hedge P&L; steady funding                                                           | Flat to mildly positive                   | Positive                            | Neutral to small positive                          | Collars generate small recurring profits as options decay → **carry enhancement** | Not triggered                                                                             |
| **Moderate Bear (−10–30 %)**          | BTC trends downward                          | Hedge gains offset reduced BTC sale proceeds; reduce hedge size                                                       | Positive (hedge gains realized)           | Positive                            | Reduced BTC sale cash flow                         | Collars protect downside → limit small monthly losses                             | Not yet activated                                                                         |
| **Deep Bear / Crash (−50 % or more)** | BTC sharply lower                            | BTC interest not sold; hedge reduced → large realized gains (USD inflow)                                              | Strongly positive                         | Positive                            | None (BTC unsold)                                  | Collars at-the-money or deep ITM → payouts limited (funding offsets losses)       | **Activated:** funding used to buy long-dated DOTM calls at crash lows → upside convexity |
| **Recovery After Crash**              | BTC rebounds from deep lows                  | Smaller hedge exposure, BTC price rising                                                                              | Slightly negative (remaining short loses) | Positive                            | Resumed BTC sales above reduced hedge              | Collars start generating profits again                                            | **DOTM Calls gain strongly**, providing leveraged upside                                  |


4.2 Cash Flow Breakdown 

| Component              | Cash Flow Direction | Description                                                                    |
| ---------------------- | ------------------- | ------------------------------------------------------------------------------ |
| **Short BTC Hedge**    | ±USD                | Realized P&L when hedge reduced (positive in crashes, negative in rallies)     |
| **Funding Yield**      | +USD                | Recurring positive income (self-financing engine for enhancements)             |
| **BTC Interest Sales** | +USD                | Monthly operational income when BTC ≥ initial price                            |
| **Collar Strategy**    | ±USD                | Small monthly option P&L; positive in mild uptrends, neutral or flat otherwise |
| **Deep OTM Calls**     | +USD (in recovery)  | Optional tail hedge; large payoff if BTC rebounds sharply post-crash           |


4.3. Key Insights

Crash Protection: The short BTC hedge locks the floor, ensuring liquidity even in deep selloffs.

Carry Income: Funding yield provides steady positive carry across all regimes.

Upside Capture: Collars add incremental returns in flat-to-mild bull markets.

Convexity Return: Deep OTM calls, financed by accumulated funding, restore upside convexity after severe crashes.

Self-Financing Loop: The entire enhancement layer is powered by funding income — no incremental capital cost.


6. Open Questions & Future Considerations

While the core framework provides a consistent and self-financing BTC hedge structure, several key questions remain open for further analysis and calibration.

6.1. FX Exposure (EUR/USD Risk)

Our USD hedge stabilizes BTC/USD value, but operational expenses may be EUR-denominated.

The EUR/USD rate introduces residual volatility in EUR-equivalent cash flows.

Open question:

Should we implement a secondary FX hedge (e.g., rolling EUR/USD forwards or options) to lock the conversion rate between USD receipts and EUR outflows?

This becomes material when EUR/USD volatility is high or BTC inflows represent a large share of total EUR operating costs.

6.2. Hedge Management in a Continuous Uptrend

In a sustained BTC rally, our logic says:

We sell monthly BTC interest at spot,

Keep the short BTC perpetual hedge unchanged (since price ≥ initial).

Over time, this means we could carry a short position indefinitely while no longer needing it for downside protection.

Open questions:

How should we systematically reduce or close the short hedge when BTC trends persistently higher?

Should there be a trigger threshold (e.g., reduce hedge once unrealized loss exceeds collected funding by X%)?

Or should we switch the short hedge into a synthetic collar beyond a certain level to regain upside exposure?

6.3. Funding Rate Volatility

The strategy assumes a stable positive funding rate, but in bear markets, funding can turn negative.

Open questions:

Should we cap hedge size dynamically based on expected funding conditions?

Could we implement a funding-rate sensitivity model to decide when to keep or unwind the short hedge?

6.4. Why not bull spread?
If we use 100 % of our funding to finance bull spreads, we might lose it all in a quiet or bearish month.
But if we use 100 % of our funding to finance collars, the most we can lose is about 10 % of it — and we retain exposure to mild upside.