from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import pandas as pd
import dash

from tree import build_tree_and_classify
from figures import make_tree_fig_combined

# --- Data ---
df_tree = build_tree_and_classify(
    start_price=120_000,
    step=5_000,
    months=12,
    alpha=0.25
)

INITIAL_PATH = ""  # root node selected by default

# --- App setup ---
app = Dash(__name__)
app.title = "BTC Hedge Tree Explorer"

app.layout = html.Div([
    dcc.Graph(
        id="tree-graph",
        figure=make_tree_fig_combined(df_tree, INITIAL_PATH),
        clear_on_unhover=True
    ),
    dcc.Store(id="current-path", data=INITIAL_PATH),
    html.Hr(),

    html.Button("Reset Path", id="reset-btn", n_clicks=0, style={"marginBottom": "10px"}),

    html.Div([
        html.Div([
            html.H4("Path Selection"),
            html.Div(id="selected-path", style={"fontSize": "18px", "marginBottom": "10px"})
        ], style={"flex": "70%"}),

        html.Div([
            html.H4("Path Category"),
            html.Div(id="path-category", style={"fontSize": "18px", "color": "#006400", "fontWeight": "bold"})
        ], style={"flex": "30%", "textAlign": "right"})
    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}),

    html.Div(id="path-table-container"),
    html.Div(id="path-plot-container", style={"marginTop": "20px"})
], style={"marginBottom": "100px"})


# --- Callback ---
@app.callback(
    Output("tree-graph", "figure"),
    Output("selected-path", "children"),
    Output("path-category", "children"),
    Output("path-table-container", "children"),
    Output("path-plot-container", "children"),
    Output("current-path", "data"),
    Input("tree-graph", "clickData"),
    Input("reset-btn", "n_clicks"),
    State("current-path", "data")
)
def update_tree_and_path(clickData, reset_clicks, current_path):
    ctx = dash.callback_context

    # --- Reset ---
    if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("reset-btn"):
        return (
            make_tree_fig_combined(df_tree, INITIAL_PATH),
            "Path reset. Click U, F, or D to start.",
            "",
            "",
            "",
            INITIAL_PATH
        )

    # --- Safeguard for empty click ---
    if not clickData or "customdata" not in clickData["points"][0]:
        return (
            make_tree_fig_combined(df_tree, current_path),
            "Click a valid (colored) node to explore.",
            "",
            "",
            "",
            current_path
        )

    clicked_path = clickData["points"][0]["customdata"]

    # --- Navigation logic ---
    if current_path == INITIAL_PATH:
        if len(clicked_path) == 1:
            new_path = clicked_path
        else:
            return (
                make_tree_fig_combined(df_tree, INITIAL_PATH),
                "Invalid: start from first month (U/F/D).",
                "",
                "",
                "",
                current_path
            )
    else:
        if clicked_path.startswith(current_path) and len(clicked_path) == len(current_path) + 1:
            new_path = clicked_path
        else:
            return (
                make_tree_fig_combined(df_tree, current_path),
                f"Invalid: can only move from {current_path} to its children.",
                "",
                "",
                "",
                current_path
            )

    # --- Build path history ---
    path_prefixes = [new_path[:i] for i in range(1, len(new_path) + 1)]
    df_path = df_tree[df_tree["path"].isin(path_prefixes)].sort_values("month")

    # Get classification of latest node
    last_node = df_path.iloc[-1]
    category_label = last_node.get("category", "N/A")

    # --- Build table ---
    display_df = df_path[[
        "month",
        "btc_price",
        "sell_interest_at_spot",
        "profit_from_selling_interest",
        "reduce_hedge",
        "profit_from_reducing_hedge",
        "monthly_funding_fee",
        "hedge_short_position",
        "cumulative_pnl"
    ]].copy()

    table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in display_df.columns],
        data=display_df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center"},
        style_header={"fontWeight": "bold"},
        page_size=12,
    )

    # --- Funding + PnL plot ---
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=df_path["month"],
        y=df_path["cumulative_pnl"],
        mode="lines+markers",
        name="Cumulative PnL",
        line=dict(color="green", width=3)
    ))
    fig_pnl.add_trace(go.Bar(
        x=df_path["month"],
        y=df_path["monthly_funding_fee"],
        name="Monthly Funding Fee",
        marker_color="orange",
        opacity=0.6
    ))
    fig_pnl.update_layout(
        title=f"Funding & PnL for Path {new_path}",
        xaxis_title="Month",
        yaxis_title="USD Value",
        height=400,
        legend=dict(orientation="h", y=-0.2)
    )

    header = (
        f"Current Path: {new_path or '(root)'} | "
        f"Month: {len(new_path)} | "
        f"Final PnL: ${df_path['cumulative_pnl'].iloc[-1]:,.0f} | "
        f"Total Funding: ${df_path['monthly_funding_fee'].sum():,.0f}"
    )


    # --- Compute summary stats ---
    final_row = df_path.iloc[-1]
    final_price = final_row["btc_price"]
    max_price = df_path["btc_price"].max()
    min_price = df_path["btc_price"].min()
    final_pnl = final_row["cumulative_pnl"]
    total_funding = df_path["monthly_funding_fee"].sum()
    category = final_row.get("category", "N/A")
    
        # --- Multi-line summary ---
    header = html.Div([
        html.B("📊 Current Path: "), f"{new_path or '(root)'}", html.Br(),
        html.B("Month: "), f"{len(new_path)}", html.Br(),
        html.B("Category: "), f"{category}", html.Br(),
        html.B("Final PnL: "), f"${final_pnl:,.0f}", html.Br(),
        html.B("Total Funding: "), f"${total_funding:,.0f}", html.Br(),
        html.B("Final BTC: "), f"${final_price:,.0f}", html.Br(),
        html.B("Max BTC: "), f"${max_price:,.0f}", html.Br(),
        html.B("Min BTC: "), f"${min_price:,.0f}",
    ])

    return (
        make_tree_fig_combined(df_tree, new_path),
        header,
        f"Category: {category_label}",
        table,
        dcc.Graph(figure=fig_pnl),
        new_path
    )


# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)