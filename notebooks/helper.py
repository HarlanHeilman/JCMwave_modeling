import numpy as np
import matplotlib.pyplot as plt

def build_control_points(heights, widths):
    """
    Construct a closed polygon boundary from stacked trapezoid widths/heights.
    
    heights: array-like of layer heights (length N)
    widths:  array-like of layer widths (length N+1)
    """
    heights = np.asarray(heights, dtype=float)
    widths  = np.asarray(widths, dtype=float)

    # --- Right side (bottom → top) ---
    y_cursor = 0.0
    right_side = []
    for i, w in enumerate(widths):
        y = heights[i] + y_cursor if i < len(heights) else y_cursor
        right_side.append((w/2, y))
        if i < len(heights):
            y_cursor += heights[i]

    # --- Left side (top → bottom) ---
    y_cursor = heights.sum()
    left_side = []
    for i, w in reversed(list(enumerate(widths))):
        if i < len(heights):
            y_cursor -= heights[i]
        y = heights[i] + y_cursor if i < len(heights) else y_cursor
        left_side.append((-w/2, y))

    # Combine into full closed boundary
    return right_side + left_side

def plot_orders_multi(dfs, normalize=False):
    """
    dfs: list of DataFrames, each with columns ['theta','order','energy','intensity']
    normalize: if True, normalize intensity per DF to max=1
    """

    # Ensure list
    if not isinstance(dfs, (list, tuple)):
        dfs = [dfs]

    # Optional normalization (per DF)
    if normalize:
        for df in dfs:
            df["intensity"] = df["intensity"] / df["intensity"].max()

    # Collect all unique orders and energies across all DataFrames
    all_orders = sorted(set().union(*[df["order"].unique() for df in dfs]))
    all_energies = sorted(set().union(*[df["energy"].unique() for df in dfs]))

    # Assign colors automatically per energy
    cmap = plt.cm.get_cmap("tab10", len(all_energies))
    energy_colors = {E: cmap(i) for i, E in enumerate(all_energies)}

    # Create subplots
    n = len(all_orders)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3*n), sharex=True)
    if n == 1:
        axes = [axes]

    # Plotting loop
    for ax, order in zip(axes, all_orders):

        for energy in all_energies:

            # Combine all dfs for this energy + order
            for df in dfs:
                subset = df[(df["order"] == order) & (df["energy"] == energy)]
                if subset.empty:
                    continue

                ax.plot(
                    subset["theta"],
                    subset["intensity"],
                    marker=".",
                    linestyle="-",
                    color=energy_colors[energy],
                    label=f"E={energy}"
                )

        ax.set_title(f"Order {order}")
        ax.set_ylabel("Intensity")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        ax.set_yscale("log")

        # Unique legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys())

    axes[-1].set_xlabel("Theta (deg)")
    plt.tight_layout()
    return fig

def plot_fit_results(df):
    """
    df: merged DataFrame with columns:
        ['theta','energy','order','exp','calc']
    normalize: normalize both curves per (energy, order)
    """

    # Unique orders and energies
    orders = sorted(df["order"].unique())
    energies = sorted(df["energy"].unique())

    # Assign colors per energy
    cmap = plt.cm.get_cmap("tab10", len(energies))
    energy_colors = {E: cmap(i) for i, E in enumerate(energies)}

    # Create subplots
    n = len(orders)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3*n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, order in zip(axes, orders):

        for energy in energies:
            subset = df[(df["order"] == order) & (df["energy"] == energy)]
            if subset.empty:
                continue

            theta = subset["theta"].values
            y_meas = subset["exp"].values
            y_pred = subset["calc"].values

            color = energy_colors[energy]

            # Plot measured
            ax.plot(
                theta,
                y_meas,
                marker=".",
                linestyle="-",
                color=color,
                label=f"E={energy} measured"
            )

            # Plot predicted
            ax.plot(
                theta,
                y_pred,
                linestyle="--",
                color='black',
                alpha=0.7,
                label=f"E={energy} predicted"
            )

        ax.set_title(f"Order {order}")
        ax.set_ylabel("Intensity")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_yscale("log")

        # Unique legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys())

    axes[-1].set_xlabel("Theta (deg)")
    plt.tight_layout()
    return fig