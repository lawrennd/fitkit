"""Diagnostic plotting functions for fitness-complexity analysis.

This module provides visualization tools for exploring fitness-complexity results,
including flow visualizations, dual potential plots, and ranked barcode displays.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path


def _to_flow_df(M: pd.DataFrame, W: sp.spmatrix | np.ndarray) -> pd.DataFrame:
    """Convert flow matrix W to DataFrame matching M's index/columns.

    Args:
        M: DataFrame with index/columns matching desired output.
        W: Sparse matrix or array representing flow/coupling.

    Returns:
        DataFrame with same index/columns as M, containing W values on support.
    """
    if sp.issparse(W):
        # sparse-safe: keep as sparse frame
        return pd.DataFrame.sparse.from_spmatrix(W, index=M.index, columns=M.columns)

    W_df = pd.DataFrame(W, index=M.index, columns=M.columns)
    # keep strictly on support (in case numerical noise fills zeros)
    return W_df.where(M.astype(bool), other=0.0)


def _top_subset(W_df: pd.DataFrame, top_c: int = 20, top_p: int = 30, by: str = "mass",
                 F_s: pd.Series | None = None, Q_s: pd.Series | None = None) -> pd.DataFrame:
    """Return a filtered W_df restricted to top rows/cols.

    Args:
        W_df: Flow DataFrame to filter.
        top_c: Number of top rows to keep.
        top_p: Number of top columns to keep.
        by: Ranking method - "mass" (row/col sums) or "fitness_complexity" (requires F_s/Q_s).
        F_s: Optional Series for fitness-based ranking (required if by="fitness_complexity").
        Q_s: Optional Series for complexity-based ranking (required if by="fitness_complexity").

    Returns:
        Filtered DataFrame with top rows/columns.
    """
    if by == "fitness_complexity":
        if F_s is None or Q_s is None:
            raise ValueError("F_s and Q_s required when by='fitness_complexity'")
        c_idx = list(F_s.sort_values(ascending=False).index[:top_c])
        p_idx = list(Q_s.sort_values(ascending=False).index[:top_p])
    else:
        c_idx = list(W_df.sum(axis=1).sort_values(ascending=False).index[:top_c])
        p_idx = list(W_df.sum(axis=0).sort_values(ascending=False).index[:top_p])
    return W_df.loc[c_idx, p_idx]


def plot_circular_bipartite_flow(
    W_df: pd.DataFrame,
    max_edges: int = 350,
    min_edge_mass: float | None = None,
    color_by: str = "country",
    title: str = "Circular bipartite flow (line-weighted, filtered)",
):
    """Chord-style circular bipartite flow using Bezier curves.

    Draws curves (not full ribbons) with linewidth proportional to flow mass.
    Filter to top edges to avoid hairballs.

    Args:
        W_df: DataFrame with flow values (rows=countries/users, cols=products/words).
        max_edges: Maximum number of edges to display.
        min_edge_mass: Minimum edge mass threshold (None = no threshold).
        color_by: Color edges by "country" or "product".
        title: Plot title.

    Notes:
        - This draws *curves* (not full ribbons) with linewidth ∝ w_cp.
        - Filter to top edges to avoid hairballs.
    """
    countries = list(W_df.index)
    products = list(W_df.columns)

    edges = (
        W_df.stack()
        .rename("w")
        .reset_index()
        .rename(columns={"level_0": "country", "level_1": "product"})
    )

    edges = edges[edges["w"] > 0].sort_values("w", ascending=False)
    if min_edge_mass is not None:
        edges = edges[edges["w"] >= float(min_edge_mass)]
    edges = edges.head(max_edges)

    if len(edges) == 0:
        print("No edges to plot after filtering.")
        return

    # angles: countries on left semicircle, products on right semicircle
    n_c, n_p = len(countries), len(products)
    theta_c = np.linspace(np.pi / 2, 3 * np.pi / 2, n_c, endpoint=False)
    theta_p = np.linspace(-np.pi / 2, np.pi / 2, n_p, endpoint=False)

    def pol2cart(theta, r=1.0):
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    pos_c = {c: pol2cart(theta_c[i]) for i, c in enumerate(countries)}
    pos_p = {p: pol2cart(theta_p[j]) for j, p in enumerate(products)}

    # colors
    if color_by == "product":
        cmap = plt.get_cmap("tab20")
        colors = {p: cmap(i % 20) for i, p in enumerate(products)}

        def edge_color(row):
            return colors[row["product"]]
    else:
        cmap = plt.get_cmap("tab20")
        colors = {c: cmap(i % 20) for i, c in enumerate(countries)}

        def edge_color(row):
            return colors[row["country"]]

    w = edges["w"].to_numpy()
    wmax = float(w.max())
    # linewidth scaling (tuned to look OK for typical normalized W)
    lw = 0.2 + 6.0 * (w / (wmax + 1e-30)) ** 0.75

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.axis("off")

    # node labels (lightweight)
    for c in countries:
        x, y = pos_c[c]
        ax.plot([x], [y], marker="o", ms=3, color="black")
    for p in products:
        x, y = pos_p[p]
        ax.plot([x], [y], marker="o", ms=3, color="black")

    # edges as cubic Beziers through center
    for i, row in enumerate(edges.itertuples(index=False)):
        c = row.country
        p = row.product
        x0, y0 = pos_c[c]
        x1, y1 = pos_p[p]
        # control points closer to center
        c0 = np.array([0.35 * x0, 0.35 * y0])
        c1 = np.array([0.35 * x1, 0.35 * y1])

        verts = [(x0, y0), (c0[0], c0[1]), (c1[0], c1[1]), (x1, y1)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        patch = PathPatch(
            path, facecolor="none", edgecolor=edge_color(row._asdict()),
            lw=lw[i], alpha=0.55
        )
        ax.add_patch(patch)

    ax.set_title(title + f"\n(top {len(edges)} edges)")
    plt.show()


def plot_alluvial_bipartite(
    W_df: pd.DataFrame,
    max_edges: int = 250,
    min_edge_mass: float | None = None,
    title: str = "Alluvial (Sankey-style) bipartite flow (filtered)",
):
    """Alluvial/Sankey-style plot in pure Matplotlib.

    Draws stacked nodes on left (countries) and right (products),
    with polygon bands for the largest flows.

    Args:
        W_df: DataFrame with flow values (rows=countries/users, cols=products/words).
        max_edges: Maximum number of edges to display.
        min_edge_mass: Minimum edge mass threshold (None = no threshold).
        title: Plot title.
    """
    edges = (
        W_df.stack()
        .rename("w")
        .reset_index()
        .rename(columns={"level_0": "country", "level_1": "product"})
    )
    edges = edges[edges["w"] > 0].sort_values("w", ascending=False)
    if min_edge_mass is not None:
        edges = edges[edges["w"] >= float(min_edge_mass)]
    edges = edges.head(max_edges)

    if len(edges) == 0:
        print("No edges to plot after filtering.")
        return

    countries = list(pd.Index(edges["country"]).unique())
    products = list(pd.Index(edges["product"]).unique())

    # total mass per node (restricted to displayed edges)
    out_mass = edges.groupby("country")["w"].sum().reindex(countries)
    in_mass = edges.groupby("product")["w"].sum().reindex(products)

    # normalize heights to 1
    out_mass = out_mass / out_mass.sum()
    in_mass = in_mass / in_mass.sum()

    # vertical packing with padding
    pad = 0.01

    def pack(masses: pd.Series):
        spans = {}
        y = 0.0
        for k, v in masses.items():
            y0 = y
            y1 = y + float(v)
            spans[k] = [y0, y1]
            y = y1 + pad
        # rescale to [0,1]
        total = y - pad
        for k in spans:
            spans[k][0] /= total
            spans[k][1] /= total
        return spans

    span_c = pack(out_mass)
    span_p = pack(in_mass)

    # allocate sub-spans per edge within each node
    c_cursor = {c: span_c[c][0] for c in countries}
    p_cursor = {p: span_p[p][0] for p in products}

    cmap = plt.get_cmap("tab20")
    c_color = {c: cmap(i % 20) for i, c in enumerate(countries)}

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")

    xL, xR = 0.1, 0.9
    node_w = 0.03

    # draw nodes
    for c in countries:
        y0, y1 = span_c[c]
        ax.add_patch(Polygon(
            [[xL - node_w, y0], [xL, y0], [xL, y1], [xL - node_w, y1]],
            closed=True, color="black", alpha=0.15
        ))
        ax.text(xL - node_w - 0.01, (y0 + y1) / 2, str(c), ha="right", va="center", fontsize=8)

    for p in products:
        y0, y1 = span_p[p]
        ax.add_patch(Polygon(
            [[xR, y0], [xR + node_w, y0], [xR + node_w, y1], [xR, y1]],
            closed=True, color="black", alpha=0.15
        ))
        ax.text(xR + node_w + 0.01, (y0 + y1) / 2, str(p), ha="left", va="center", fontsize=8)

    # bands
    for row in edges.itertuples(index=False):
        c = row.country
        p = row.product
        w = float(row.w)

        # band thickness within each stacked node span (relative to node mass)
        dc = w / float(edges[edges["country"] == c]["w"].sum()) * (span_c[c][1] - span_c[c][0])
        dp = w / float(edges[edges["product"] == p]["w"].sum()) * (span_p[p][1] - span_p[p][0])

        y0c, y1c = c_cursor[c], c_cursor[c] + dc
        y0p, y1p = p_cursor[p], p_cursor[p] + dp
        c_cursor[c] = y1c
        p_cursor[p] = y1p

        # simple 4-point polygon band (looks OK with alpha)
        poly = Polygon(
            [[xL, y0c], [xR, y0p], [xR, y1p], [xL, y1c]],
            closed=True,
            facecolor=c_color[c],
            edgecolor="none",
            alpha=0.45,
        )
        ax.add_patch(poly)

    ax.set_title(title + f"\n(top {len(edges)} edges)")
    plt.show()


def plot_dual_potential_bipartite(
    M: pd.DataFrame,
    W_df: pd.DataFrame,
    u: np.ndarray,
    v: np.ndarray,
    max_edges: int = 400,
    title: str = "Dual potentials (log u, log v) with flow edges",
):
    """Layered bipartite plot: node color = dual potentials, edge thickness = w_cp.

    Args:
        M: DataFrame with support matrix (rows=countries/users, cols=products/words).
        W_df: DataFrame with flow values matching M's index/columns.
        u: Row dual potentials (n_rows,).
        v: Column dual potentials (n_cols,).
        max_edges: Maximum number of edges to display.
        title: Plot title.
    """
    countries = list(M.index)
    products = list(M.columns)

    phi = pd.Series(np.log(u + 1e-30), index=countries)
    psi = pd.Series(np.log(v + 1e-30), index=products)

    # order by potential for a clean "landscape"
    c_order = list(phi.sort_values().index)
    p_order = list(psi.sort_values().index)

    # pick top edges globally
    edges = (
        W_df.loc[c_order, p_order]
        .stack()
        .rename("w")
        .reset_index()
        .rename(columns={"level_0": "country", "level_1": "product"})
    )
    edges = edges[edges["w"] > 0].sort_values("w", ascending=False).head(max_edges)

    if len(edges) == 0:
        print("No edges to plot.")
        return

    # positions
    y_c = {c: i for i, c in enumerate(c_order)}
    y_p = {p: i for i, p in enumerate(p_order)}

    x_c, x_p = 0.0, 1.0

    # color mapping
    vals = np.concatenate([phi.to_numpy(), psi.to_numpy()])
    vmin, vmax = np.percentile(vals, [5, 95])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(10, 8))

    # edges
    w = edges["w"].to_numpy()
    wmax = float(w.max())
    lw = 0.2 + 4.5 * (w / (wmax + 1e-30)) ** 0.7

    for i, row in enumerate(edges.itertuples(index=False)):
        c = row.country
        p = row.product
        ax.plot([x_c, x_p], [y_c[c], y_p[p]], color="black", alpha=0.12, lw=lw[i])

    # nodes
    ax.scatter(
        [x_c] * len(c_order), [y_c[c] for c in c_order],
        c=[cmap(norm(phi[c])) for c in c_order], s=18, edgecolor="none"
    )
    ax.scatter(
        [x_p] * len(p_order), [y_p[p] for p in p_order],
        c=[cmap(norm(psi[p])) for p in p_order], s=18, edgecolor="none"
    )

    ax.set_yticks([])
    ax.set_xticks([x_c, x_p])
    ax.set_xticklabels(["countries", "products"])
    ax.set_title(title + f"\n(node color = log dual, top {len(edges)} edges)")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("dual potential (log scale)")

    plt.tight_layout()
    plt.show()


def plot_ranked_barcodes(
    results_countries: pd.DataFrame,
    results_products: pd.DataFrame,
    top_n: int = 40,
    title: str = "Ranked barcodes (Fitness/Complexity) with degree overlays",
):
    """Two clean rank plots: countries by Fitness, products by Complexity.

    Args:
        results_countries: DataFrame with "Fitness" and "diversification_kc" columns.
        results_products: DataFrame with "Complexity" and "ubiquity_kp" columns.
        top_n: Number of top items to display.
        title: Plot title.
    """
    rc = results_countries.sort_values("Fitness", ascending=False).head(top_n)
    rp = results_products.sort_values("Complexity", ascending=False).head(top_n)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # countries
    ax0 = ax[0]
    ax0.bar(range(len(rc)), rc["Fitness"].to_numpy(), color="black", alpha=0.6)
    ax0.set_title(f"Countries (top {len(rc)})")
    ax0.set_xlabel("rank")
    ax0.set_ylabel("Fitness")

    ax0b = ax0.twinx()
    ax0b.plot(range(len(rc)), rc["diversification_kc"].to_numpy(), color="tab:blue", lw=1.5)
    ax0b.set_ylabel("diversification (kc)")

    # products
    ax1 = ax[1]
    ax1.bar(range(len(rp)), rp["Complexity"].to_numpy(), color="black", alpha=0.6)
    ax1.set_title(f"Products (top {len(rp)})")
    ax1.set_xlabel("rank")
    ax1.set_ylabel("Complexity")

    ax1b = ax1.twinx()
    ax1b.plot(range(len(rp)), rp["ubiquity_kp"].to_numpy(), color="tab:orange", lw=1.5)
    ax1b.set_ylabel("ubiquity (kp)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
