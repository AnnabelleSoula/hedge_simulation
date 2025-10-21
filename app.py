# from dash import Dash, dcc, html, Input, Output, State, dash_table
# import plotly.graph_objects as go
# import pandas as pd
# import dash

# # --- Import custom modules ---
# from tree import build_btc_trinomial_tree_rule_based
# from figures import make_tree_fig_filtered  # progressive tree renderer

# # --- Generate the full decision tree data once ---
# df_tree = build_btc_trinomial_tree_rule_based(
#     start_price=120_000,
#     step=5_000,
#     months=6,
#     expected_btc_interest=7.5,
#     yearly_funding_rate=0.07
# )

# # --- Dash App setup ---
# app = Dash(__name__)
# app.title = "BTC Hedge Tree Explorer"

# app.layout = html.Div([
#     dcc.Graph(id="tree-graph", figure=make_tree_fig_filtered(df_tree, "")),
#     dcc.Store(id="current-path", data=""),  # stores the current path between clicks
#     html.Hr(),
#     html.Button("🔄 Reset Path", id="reset-btn", n_clicks=0, style={"marginBottom": "10px"}),
#     html.H4("Path Selection"),
#     html.Div(id="selected-path", style={"fontSize": "18px", "marginBottom": "10px"}),
#     html.Div(id="path-table-container"),
#     html.Div(id="path-plot-container", style={"marginTop": "20px"})
# ])


# # --- Callback: sequential path navigation + dynamic subtree display ---
# @app.callback(
#     Output("tree-graph", "figure"),
#     Output("selected-path", "children"),
#     Output("path-table-container", "children"),
#     Output("path-plot-container", "children"),
#     Output("current-path", "data"),
#     Input("tree-graph", "clickData"),
#     Input("reset-btn", "n_clicks"),
#     State("current-path", "data")
# )
# def update_tree_and_path(clickData, reset_clicks, current_path):
#     ctx = dash.callback_context

#     # --- Reset event ---
#     if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("reset-btn"):
#         return make_tree_fig_filtered(df_tree, ""), "🔄 Path reset. Click a node to start again.", "", "", ""

#     if clickData is None:
#         return make_tree_fig_filtered(df_tree, current_path), "Click a node to explore sequentially.", "", "", current_path

#     clicked_path = clickData["points"][0]["customdata"]

#     # --- Validate sequential navigation ---
#     if current_path == "":
#         if len(clicked_path) == 1:
#             new_path = clicked_path
#         else:
#             return make_tree_fig_filtered(df_tree, ""), "⚠️ Invalid selection: start from month 1 (U/F/D).", "", "", current_path
#     else:
#         if clicked_path.startswith(current_path) and len(clicked_path) == len(current_path) + 1:
#             new_path = clicked_path
#         else:
#             return make_tree_fig_filtered(df_tree, current_path), f"⚠️ Invalid step: can only select children of {current_path}.", "", "", current_path

#     # --- Extract the state evolution for the selected path ---
#     path_prefixes = [new_path[:i] for i in range(1, len(new_path) + 1)]
#     df_path = df_tree[df_tree["path"].isin(path_prefixes)].sort_values("month")

#     display_df = df_path[[
#         "month",
#         "btc_price",
#         "sell_interest_at_spot",
#         "profit_from_selling_interest",
#         "reduce_hedge",
#         "profit_from_reducing_hedge",
#         "monthly_funding_fee",
#         "hedge_short_position",
#         "cumulative_pnl"
#     ]].copy()

#     # --- Table ---
#     table = dash_table.DataTable(
#         columns=[{"name": col, "id": col} for col in display_df.columns],
#         data=display_df.to_dict("records"),
#         style_table={"overflowX": "auto"},
#         style_cell={"textAlign": "center"},
#         style_header={"fontWeight": "bold"},
#         page_size=12,
#     )

#     # --- Plot: Cumulative PnL + Funding ---
#     fig_pnl = go.Figure()
#     fig_pnl.add_trace(go.Scatter(
#         x=df_path["month"],
#         y=df_path["cumulative_pnl"],
#         mode="lines+markers",
#         name="Cumulative PnL",
#         line=dict(color="green", width=3)
#     ))
#     fig_pnl.add_trace(go.Bar(
#         x=df_path["month"],
#         y=df_path["monthly_funding_fee"],
#         name="Monthly Funding Fee",
#         marker_color="orange",
#         opacity=0.6
#     ))
#     fig_pnl.update_layout(
#         title=f"Funding & PnL for Path {new_path}",
#         xaxis_title="Month",
#         yaxis_title="USD Value",
#         height=400,
#         legend=dict(orientation="h", y=-0.2)
#     )

#     # --- Header text ---
#     header = (
#         f"✅ Current Path: {new_path} | "
#         f"Month: {len(new_path)} | "
#         f"Final PnL: ${df_path['cumulative_pnl'].iloc[-1]:,.0f} | "
#         f"Total Funding: ${df_path['monthly_funding_fee'].sum():,.0f}"
#     )

#     return make_tree_fig_filtered(df_tree, new_path), header, table, dcc.Graph(figure=fig_pnl), new_path


# # --- Run App ---
# if __name__ == "__main__":
#     app.run(debug=True)


from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import pandas as pd
import dash

from tree import build_btc_trinomial_tree_rule_based
from figures import make_tree_fig_combined

# --- Data ---
df_tree = build_btc_trinomial_tree_rule_based(
    start_price=120_000,
    step=5_000,
    months=6,
    expected_btc_interest=7.5,
    yearly_funding_rate=0.07
)

INITIAL_PATH = ""  # root node selected by default

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
    html.Button("🔄 Reset Path", id="reset-btn", n_clicks=0, style={"marginBottom": "10px"}),
    html.H4("Path Selection"),
    html.Div(id="selected-path", style={"fontSize": "18px", "marginBottom": "10px"}),
    html.Div(id="path-table-container"),
    html.Div(id="path-plot-container", style={"marginTop": "20px"})
])


# --- Callback ---
@app.callback(
    Output("tree-graph", "figure"),
    Output("selected-path", "children"),
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
        return make_tree_fig_combined(df_tree, INITIAL_PATH), (
            "🔄 Path reset. Click U, F, or D to start."
        ), "", "", INITIAL_PATH

    # --- Safe guard for empty click ---
    if not clickData or "customdata" not in clickData["points"][0]:
        return make_tree_fig_combined(df_tree, current_path), (
            "Click a valid (colored) node to explore."
        ), "", "", current_path

    clicked_path = clickData["points"][0]["customdata"]

    # --- Navigation logic ---
    if current_path == INITIAL_PATH:
        if len(clicked_path) == 1:
            new_path = clicked_path
        else:
            return make_tree_fig_combined(df_tree, INITIAL_PATH), (
                "⚠️ Invalid: start from first month (U/F/D)."
            ), "", "", current_path
    else:
        if clicked_path.startswith(current_path) and len(clicked_path) == len(current_path) + 1:
            new_path = clicked_path
        else:
            return make_tree_fig_combined(df_tree, current_path), (
                f"⚠️ Invalid: can only move from {current_path} to its children."
            ), "", "", current_path

    # --- Build path history ---
    path_prefixes = [new_path[:i] for i in range(1, len(new_path) + 1)]
    df_path = df_tree[df_tree["path"].isin(path_prefixes)].sort_values("month")

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

    # --- Table ---
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
        f"✅ Current Path: {new_path or '(root)'} | "
        f"Month: {len(new_path)} | "
        f"Final PnL: ${df_path['cumulative_pnl'].iloc[-1]:,.0f} | "
        f"Total Funding: ${df_path['monthly_funding_fee'].sum():,.0f}"
    )

    return make_tree_fig_combined(df_tree, new_path), header, table, dcc.Graph(figure=fig_pnl), new_path


# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
