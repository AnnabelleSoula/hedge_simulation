import plotly.graph_objects as go
import pandas as pd


def make_static_lattice(start_price=120_000, step=5_000, months=12):
    """
    Build the static trinomial price lattice:
    Each month: prices = start_price + k*step for k in [-m..m].
    """
    nodes, edges = [], []

    for m in range(months + 1):
        for k in range(-m, m + 1):
            price = start_price + k * step
            nodes.append({"month": m, "btc_price": price})
            if m > 0:
                for dk in [-1, 0, 1]:
                    prev_k = k - dk
                    if abs(prev_k) <= (m - 1):
                        edges.append({
                            "month0": m - 1,
                            "price0": start_price + prev_k * step,
                            "month1": m,
                            "price1": price,
                            "key": f"{m-1}_{prev_k}->{m}_{k}"
                        })

    return pd.DataFrame(nodes), pd.DataFrame(edges)


def make_tree_fig_combined(df_tree, current_path="", start_price=120_000, step=5_000, months=12):
    """
    Combine static lattice (background) with path-dependent overlay.
    Highlight:
      - blue: current node
      - green: next reachable nodes
      - lightblue: path history
      - lightgray: inactive
    Highlight the edges of the active path (black).
    """
    df_nodes, df_edges = make_static_lattice(start_price, step, months)

    # --- Build list of nodes on the active path ---
    if current_path == "":
        active_nodes = [{"month": 0, "price": start_price}]
        next_nodes = [(1, start_price + step), (1, start_price), (1, start_price - step)]
    else:
        prefixes = [current_path[:i] for i in range(len(current_path) + 1)]
        df_path = df_tree[df_tree["path"].isin(prefixes)]
        active_nodes = list(zip(df_path["month"], df_path["btc_price"]))
        last = active_nodes[-1]
        next_nodes = [
            (last[0] + 1, last[1] + step),
            (last[0] + 1, last[1]),
            (last[0] + 1, last[1] - step),
        ]

    # --- Separate edges ---
    path_edges = []
    static_edges = []

    for _, e in df_edges.iterrows():
        edge_tuple0 = (e["month0"], e["price0"])
        edge_tuple1 = (e["month1"], e["price1"])

        # edge is on the active path if both endpoints appear consecutively in active_nodes
        is_path_edge = any(
            edge_tuple0 == active_nodes[i] and edge_tuple1 == active_nodes[i + 1]
            for i in range(len(active_nodes) - 1)
        )
        if is_path_edge:
            path_edges.append(e)
        else:
            static_edges.append(e)

    # --- Draw static edges (light gray) ---
    edge_traces = [
        go.Scatter(
            x=[e["month0"], e["month1"]],
            y=[e["price0"], e["price1"]],
            mode="lines",
            line=dict(color="lightgray", width=1),
            hoverinfo="none",
            showlegend=False
        )
        for _, e in pd.DataFrame(static_edges).iterrows()
    ]

    # --- Draw active path edges (black, thicker) ---
    if path_edges:
        edge_traces.append(
            go.Scatter(
                x=sum([[e["month0"], e["month1"], None] for _, e in pd.DataFrame(path_edges).iterrows()], []),
                y=sum([[e["price0"], e["price1"], None] for _, e in pd.DataFrame(path_edges).iterrows()], []),
                mode="lines",
                line=dict(color="black", width=2.5),
                hoverinfo="none",
                showlegend=False
            )
        )

    # --- Base lattice nodes (gray) ---
    base_nodes = go.Scatter(
        x=df_nodes["month"],
        y=df_nodes["btc_price"],
        mode="markers",
        marker=dict(size=8, color="lightgray"),
        hoverinfo="none",
        name="Static Lattice"
    )

    # --- Dynamic overlay coloring ---
    overlay_data = []

    # root + first 3 if no path
    if current_path == "":
        overlay_data = [
            {"month": 0, "btc_price": start_price, "path": "", "color": "blue"},
            {"month": 1, "btc_price": start_price + step, "path": "U", "color": "green"},
            {"month": 1, "btc_price": start_price, "path": "F", "color": "green"},
            {"month": 1, "btc_price": start_price - step, "path": "D", "color": "green"},
        ]
        df_vis = pd.DataFrame(overlay_data)
    else:
        df_vis = df_tree[df_tree["path"].isin(prefixes + [current_path + s for s in ["U", "F", "D"]])].copy()

        colors = []
        for _, r in df_vis.iterrows():
            if r["path"] == current_path:
                colors.append("blue")
            elif len(r["path"]) == len(current_path) + 1 and r["path"].startswith(current_path):
                colors.append("green")
            elif r["path"] in prefixes:
                colors.append("lightblue")
            else:
                colors.append("lightgray")
        df_vis["color"] = colors

    # Assign text positions dynamically
    text_positions = []
    for color in df_vis["color"]:
        if color == "green":
            text_positions.append("middle right")  # selectable next nodes
        elif color in ["blue", "lightblue"]:
            text_positions.append("top center")    # selected path
        else:
            text_positions.append("top center")    # default

    overlay_trace = go.Scatter(
        x=df_vis["month"],
        y=df_vis["btc_price"],
        mode="markers+text",
        text=df_vis["path"],
        textposition=text_positions,  # dynamic list
        marker=dict(size=12, color=df_vis["color"], line=dict(width=1, color="black")),
        customdata=df_vis["path"],
        hovertemplate="<b>Path:</b> %{customdata}<br>Month: %{x}<br>BTC: %{y:$,.0f}<extra></extra>",
        name="Overlay"
    )

    # --- Combine all ---
    fig = go.Figure(data=edge_traces + [base_nodes, overlay_trace])
    fig.update_layout(
        title=f"BTC Hedge Trinomial Lattice — Path: {current_path or '(root)'}",
        xaxis_title="Month",
        yaxis_title="BTC Price ($)",
        plot_bgcolor="white",
        height=700,
        hovermode="closest",
    )
    return fig
