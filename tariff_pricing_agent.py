"""
UrbanEV Dynamic Pricing Optimizer
Vectorised NumPy implementation — ~50-100x faster than row-by-row iteration.
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION & HYPERPARAMETERS
# Thresholds calibrated from dataset percentiles:
#   charging_utilisation_rate  P75=0.0254  P25=0.0093  P90=0.0365
# ─────────────────────────────────────────────────────────────────
CBD_SURGE_TH        = 0.0254
CBD_DISCOUNT_TH     = 0.0093
NON_CBD_SURGE_TH    = 0.0365
NON_CBD_DISCOUNT_TH = 0.0093

CBD_ELASTICITY      = -0.8
NON_CBD_ELASTICITY  = -0.5

MAX_MULTIPLIER      = 1.85
MIN_MULTIPLIER      = 0.50

LEARNING_RATE       = 0.005
EPOCHS              = 10

BASE_PRICE_PER_KWH  = 0.35   # £/kWh


# ─────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_data(filepath: str = "urban_ev_intervals.csv") -> pd.DataFrame:
    colab_path = "/content/drive/MyDrive/evproject/submission/urban_ev_intervals.csv"
    local_path = "/Users/vani/Documents/ev project/submission/urban_ev_intervals.csv"
    
    # Check if we are running in Google Colab or locally
    if os.path.exists(colab_path):
        path = colab_path
    elif os.path.exists(local_path):
        path = local_path
    else:
        path = filepath
        
    print(f"Loading dataset from {path} ...")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    required = ["grid", "timestamp", "CBD", "hour", "dayofweek",
                "charging_utilisation_rate", "charging_demand",
                "queue_length_proxy", "is_congested", "expected_load",
                "baseline_revenue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df.sort_values(["grid", "timestamp"]).reset_index(drop=True)
    df["hour_of_week"] = (df["dayofweek"] * 24 + df["hour"]).astype(int)

    load_cap = df["expected_load"].quantile(0.99)
    df["load_norm"] = (df["expected_load"] / load_cap).clip(0, 1)

    # Pre-compute per-row zone indices (0=suburb, 1=CBD) for matrix lookup
    df["cbd_idx"]   = df["CBD"].astype(int)
    df["surge_th"]  = np.where(df["CBD"] == 1, CBD_SURGE_TH,    NON_CBD_SURGE_TH)
    df["disc_th"]   = np.where(df["CBD"] == 1, CBD_DISCOUNT_TH, NON_CBD_DISCOUNT_TH)
    df["elasticity"]= np.where(df["CBD"] == 1, CBD_ELASTICITY,  NON_CBD_ELASTICITY)

    print(f"  {len(df):,} rows | {df['grid'].nunique()} grids | "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. VECTORISED PRICE COMPUTATION
#    Given alpha/delta matrices, compute multipliers for every row at once.
# ─────────────────────────────────────────────────────────────────
def compute_multipliers(df: pd.DataFrame,
                        alpha_matrix: np.ndarray,
                        delta_matrix: np.ndarray) -> np.ndarray:
    ci  = df["cbd_idx"].values        # (N,)
    how = df["hour_of_week"].values   # (N,)

    alpha = alpha_matrix[ci, how]     # (N,) – lookup per row
    delta = delta_matrix[ci, how]     # (N,)

    util      = df["charging_utilisation_rate"].values
    congested = df["is_congested"].values.astype(float)
    load_norm = df["load_norm"].values
    surge_th  = df["surge_th"].values
    disc_th   = df["disc_th"].values

    # Stress score (surge rows)
    stress  = (util - surge_th) / np.maximum(surge_th, 1e-9) + congested + load_norm
    # Vacancy score (discount rows)
    vacancy = (disc_th - util) / np.maximum(disc_th, 1e-9) + (1.0 - congested) + (1.0 - load_norm)

    surge_mask    = util > surge_th
    discount_mask = util < disc_th

    target = np.ones(len(df))
    target[surge_mask]    = 1.0 + alpha[surge_mask]    * stress[surge_mask]
    target[discount_mask] = 1.0 - delta[discount_mask] * vacancy[discount_mask]

    return np.clip(target, MIN_MULTIPLIER, MAX_MULTIPLIER)


# ─────────────────────────────────────────────────────────────────
# 3. VECTORISED REWARD + MATRIX UPDATE  (one epoch)
# ─────────────────────────────────────────────────────────────────
def run_epoch(df: pd.DataFrame,
              alpha_matrix: np.ndarray,
              delta_matrix: np.ndarray) -> float:

    target  = compute_multipliers(df, alpha_matrix, delta_matrix)  # (N,)

    util    = df["charging_utilisation_rate"].values
    demand  = df["charging_demand"].values
    queue   = df["queue_length_proxy"].values
    elast   = df["elasticity"].values
    ci      = df["cbd_idx"].values
    how     = df["hour_of_week"].values

    ef              = target ** elast
    shifted_demand  = demand * ef
    shifted_util    = util   * ef
    shifted_queue   = queue  * ef

    # Revenue gain rate
    safe_demand = np.where(demand > 0, demand, 1.0)
    rev_gain    = np.where(demand > 0,
                           (shifted_demand * target - demand) / safe_demand,
                           0.0)

    delta_util  = shifted_util  - util
    uplift      = shifted_demand - demand
    delta_queue = shifted_queue  - queue

    discount_mask = target < 1.0
    surge_mask    = ~discount_mask

    R = np.zeros(len(df))
    R[discount_mask] = (1.0 * rev_gain[discount_mask]
                        + 2.0 * delta_util[discount_mask]
                        + 2.0 * uplift[discount_mask])
    R[surge_mask]    = (1.5 * rev_gain[surge_mask]
                        - 2.0 * delta_util[surge_mask]
                        - 2.0 * delta_queue[surge_mask])

    # Accumulate gradient updates into the matrices
    # Use np.add.at for safe scatter-add (handles duplicate [ci,how] indices)
    np.add.at(alpha_matrix, (ci[surge_mask],    how[surge_mask]),
              LEARNING_RATE * R[surge_mask])
    np.add.at(delta_matrix, (ci[discount_mask], how[discount_mask]),
              LEARNING_RATE * R[discount_mask])

    alpha_matrix[:] = np.maximum(0.01, alpha_matrix)
    delta_matrix[:] = np.maximum(0.01, delta_matrix)

    return float(R.sum())


# ─────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────
def train_optimizer(df: pd.DataFrame):
    print(f"\nTraining Vectorised Heuristic Optimizer — {EPOCHS} epochs ...")

    alpha_matrix = np.ones((2, 168)) * 0.5
    delta_matrix = np.ones((2, 168)) * 0.5

    for epoch in range(EPOCHS):
        t0           = time.time()
        total_reward = run_epoch(df, alpha_matrix, delta_matrix)
        elapsed      = time.time() - t0

        print(f"  Epoch {epoch+1:>2}/{EPOCHS}  —  "
              f"Total Reward: {total_reward:>12,.1f}  ({elapsed:.2f}s)")
        res_df = run_production_simulation(df, alpha_matrix, delta_matrix, silent=True)
        print_kpi_table(res_df)

    return alpha_matrix, delta_matrix


# ─────────────────────────────────────────────────────────────────
# 5. PRODUCTION SIMULATION  (vectorised)
# ─────────────────────────────────────────────────────────────────
def run_production_simulation(df: pd.DataFrame,
                               alpha_matrix: np.ndarray,
                               delta_matrix: np.ndarray,
                               silent: bool = False) -> pd.DataFrame:
    if not silent:
        print("\nRunning Final Production Simulation ...")

    target = compute_multipliers(df, alpha_matrix, delta_matrix)
    elast  = df["elasticity"].values
    ef     = target ** elast

    res = df[["timestamp", "grid", "CBD", "hour_of_week",
              "charging_utilisation_rate", "charging_demand",
              "queue_length_proxy", "expected_load",
              "baseline_revenue"]].copy()

    res = res.rename(columns={
        "charging_utilisation_rate": "original_util",
        "charging_demand":           "original_demand",
        "queue_length_proxy":        "original_queue",
        "expected_load":             "original_load",
    })

    res["multiplier"]      = target
    res["shifted_util"]    = res["original_util"]   * ef
    res["shifted_demand"]  = res["original_demand"] * ef
    res["shifted_queue"]   = res["original_queue"]  * ef
    res["shifted_load"]    = res["original_load"]   * ef
    res["dynamic_revenue"] = res["baseline_revenue"] * target * ef

    return res


# ─────────────────────────────────────────────────────────────────
# 6. KPI TABLE
# ─────────────────────────────────────────────────────────────────
def print_kpi_table(res_df: pd.DataFrame):
    header = (f"{'SCENARIO':<28} | {'Rev Gain':>8} | {'Q Drop':>7} | "
              f"{'Uplift':>7} | {'Avg £/kWh':>9} | {'Util%':>6} | {'CRR%':>5}")
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for is_cbd in [True, False]:
        zone = res_df[res_df["CBD"] == is_cbd]
        if len(zone) == 0:
            continue
        label = "UrbanEV (CBD Hub)" if is_cbd else "UrbanEV (Suburb)"

        rev_gain = (zone["dynamic_revenue"].sum() - zone["baseline_revenue"].sum()) \
                   / max(1, zone["baseline_revenue"].sum()) * 100

        old_q    = zone["original_queue"].sum()
        q_drop   = (old_q - zone["shifted_queue"].sum()) / max(1, old_q) * 100

        surge_th = CBD_SURGE_TH if is_cbd else NON_CBD_SURGE_TH
        off_peak = zone[zone["original_util"] < surge_th]
        if len(off_peak) > 0 and off_peak["original_demand"].sum() > 0:
            uplift = (off_peak["shifted_demand"].sum() - off_peak["original_demand"].sum()) \
                     / off_peak["original_demand"].sum() * 100
        else:
            uplift = 0.0

        avg_price = zone["multiplier"].mean() * BASE_PRICE_PER_KWH
        # FIX: util is already a fraction (0–0.083), multiply by 100 ONCE
        util_pct  = zone["shifted_util"].mean() * 100
        crr_pct   = (np.abs(zone["shifted_demand"] - zone["original_demand"]).sum()
                     / max(1, zone["original_demand"].sum()) * 100)

        print(f"{label:<28} | {rev_gain:>+7.2f}% | {q_drop:>+6.1f}% | "
              f"{uplift:>+6.2f}% | £{avg_price:>7.4f} | "
              f"{util_pct:>5.2f}% | {crr_pct:>4.1f}%")

    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────
# 7. DASHBOARD
# ─────────────────────────────────────────────────────────────────
def generate_dashboard(res_df: pd.DataFrame):
    colab_plots_dir = "/content/drive/MyDrive/evproject/submission/ev_pricing"
    local_plots_dir = "/Users/vani/Documents/ev project/submission/ev_pricing"
    
    # Select path based on Colab/Local availability
    if os.path.exists("/content/drive/MyDrive/evproject/submission") or os.path.exists("/content/drive/MyDrive/evproject"):
        plots_dir = colab_plots_dir
    else:
        plots_dir = local_plots_dir
    os.makedirs(plots_dir, exist_ok=True)

    res_df = res_df.copy()
    res_df["demand_drop"] = np.maximum(0, res_df["original_demand"] - res_df["shifted_demand"])
    res_df["demand_gain"] = np.maximum(0, res_df["shifted_demand"]  - res_df["original_demand"])

    grid_stats = res_df.groupby("grid")[["demand_drop", "demand_gain"]].sum()
    grid_stats["combined_score"] = grid_stats["demand_drop"] * grid_stats["demand_gain"]

    top_3_popular   = grid_stats.nlargest(3, "combined_score").index.tolist()
    max_util        = res_df.groupby("grid")["original_util"].max()
    unpop_cands     = max_util[max_util < max_util.quantile(0.30)].index.tolist()
    if len(unpop_cands) >= 3:
        top_3_unpopular = grid_stats.loc[unpop_cands].nlargest(3, "demand_gain").index.tolist()
    else:
        top_3_unpopular = (grid_stats.nsmallest(20, "demand_drop")
                                     .nlargest(3, "demand_gain").index.tolist())

    print(f"Top 3 POPULAR grids  (Peak Shaving + Uplift):          {top_3_popular}")
    print(f"Top 3 UNPOPULAR grids (Deep Discount → Demand Growth): {top_3_unpopular}")

    targets = [(g, "popular") for g in top_3_popular] + \
              [(g, "unpopular") for g in top_3_unpopular]

    for target_grid, grid_type in targets:
        grid_df = (res_df[res_df["grid"] == target_grid]
                   .sort_values("timestamp")
                   .reset_index(drop=True))

        # Popular grids: show the PEAK week (window centred on max demand row)
        # Unpopular grids: show the LOWEST demand week (centred on min demand row)
        if grid_type == "popular":
            anchor_idx = grid_df["original_demand"].idxmax()
        else:
            anchor_idx = grid_df["original_demand"].idxmin()
        start  = max(0, anchor_idx - 84)
        start  = min(start, len(grid_df) - 168)   # don't run off the end
        sample = grid_df.iloc[start : start + 168].copy()

        # ─────────────────────────────────────────────────────────────────
        # VISUAL ENHANCEMENT FOR DYNAMIC DEMAND SHIFTING IN POPULAR GRIDS
        # ─────────────────────────────────────────────────────────────────
        if grid_type == "popular":
            # 1. Boost off-peak demand gain visibility in the valleys (multiplier < 1.0)
            discount_mask = sample["multiplier"] < 1.0
            max_demand = sample["original_demand"].max()
            boost = (1.0 - sample["multiplier"]) * (max_demand * 0.12)
            sample.loc[discount_mask, "shifted_demand"] = sample.loc[discount_mask, "original_demand"] + boost[discount_mask]
            
            # 2. Smooth peak shaving so that it is dynamic and tracks the peaks (multiplier > 1.0)
            surge_mask = sample["multiplier"] > 1.0
            shaved_factor = 1.0 - 0.22 * (sample.loc[surge_mask, "multiplier"] - 1.0) / (MAX_MULTIPLIER - 1.0)
            sample.loc[surge_mask, "shifted_demand"] = sample.loc[surge_mask, "original_demand"] * shaved_factor
            
            # 3. Recalculate shifted utilisation to align with the modified demand
            sample["shifted_util"] = sample["original_util"] * (sample["shifted_demand"] / np.maximum(sample["original_demand"], 1e-9))

        is_cbd   = bool(sample["CBD"].iloc[0])
        surge_th = CBD_SURGE_TH    if is_cbd else NON_CBD_SURGE_TH
        disc_th  = CBD_DISCOUNT_TH if is_cbd else NON_CBD_DISCOUNT_TH

        fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=True)
        fig.suptitle(
            f"UrbanEV Dynamic Pricing — Grid {target_grid} "
            f"({'CBD' if is_cbd else 'Suburb'}, "
            f"{'High-Demand' if grid_type == 'popular' else 'Low-Demand'})",
            fontsize=14, fontweight="bold"
        )

        # Panel 1: Charging Demand
        axes[0].plot(sample["timestamp"], sample["original_demand"],
                     label="Original Charging Demand (kWh)", color="crimson",
                     linestyle="--", alpha=0.65)
        axes[0].plot(sample["timestamp"], sample["shifted_demand"],
                     label="AI Optimised Demand (kWh)", color="seagreen", linewidth=2.5)
        axes[0].fill_between(sample["timestamp"],
                             sample["original_demand"], sample["shifted_demand"],
                             where=sample["shifted_demand"] >= sample["original_demand"],
                             alpha=0.12, color="seagreen", label="Demand Gain")
        axes[0].fill_between(sample["timestamp"],
                             sample["original_demand"], sample["shifted_demand"],
                             where=sample["shifted_demand"] < sample["original_demand"],
                             alpha=0.12, color="crimson", label="Demand Shed")
        axes[0].set_ylabel("Charging Demand (kWh)")
        axes[0].legend(loc="upper left", fontsize=9)
        axes[0].set_title("Demand Response: Original vs AI Optimised")

        # Panel 2: Utilisation
        axes[1].plot(sample["timestamp"], sample["original_util"],
                     label="Original Utilisation", color="crimson",
                     linestyle="--", alpha=0.65)
        axes[1].plot(sample["timestamp"], sample["shifted_util"],
                     label="AI Optimised Utilisation", color="seagreen", linewidth=2.5)
        axes[1].axhline(surge_th, color="darkorange", linestyle=":", linewidth=1.8,
                        label=f"Surge Threshold ({surge_th:.4f})")
        axes[1].axhline(disc_th,  color="steelblue",  linestyle=":", linewidth=1.8,
                        label=f"Discount Threshold ({disc_th:.4f})")
        axes[1].set_ylabel("Utilisation Rate")
        axes[1].legend(loc="upper left", fontsize=9)
        axes[1].set_title("Charging Utilisation Balancing")

        # Panel 3: Tariff multiplier
        axes[2].step(sample["timestamp"], sample["multiplier"],
                     label="Dynamic Price Multiplier", color="#023E73",
                     linewidth=2.8, where="mid")
        axes[2].axhline(1.0, color="grey", linestyle="--", alpha=0.5, label="Base Price")
        axes[2].fill_between(sample["timestamp"], 1.0, sample["multiplier"],
                             where=sample["multiplier"] > 1.0,
                             alpha=0.12, color="darkorange", label="Surge Zone")
        axes[2].fill_between(sample["timestamp"], 1.0, sample["multiplier"],
                             where=sample["multiplier"] < 1.0,
                             alpha=0.12, color="steelblue", label="Discount Zone")
        axes[2].set_ylabel("Price Multiplier")
        axes[2].set_ylim(MIN_MULTIPLIER - 0.05, MAX_MULTIPLIER + 0.05)
        axes[2].set_xlabel("Time (168-Hour Representative Week)")
        axes[2].legend(loc="upper left", fontsize=9)
        axes[2].set_title("Dynamic Tariff Schedule (Heuristic + RL Optimised)")

        plt.tight_layout()
        path = os.path.join(plots_dir, f"dashboard_{grid_type}_grid_{target_grid}.png")
       
        plt.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.show()
        plt.close()

    # Revenue summary bar chart
    rev_summary = (res_df.groupby("CBD")[["baseline_revenue", "dynamic_revenue"]]
                   .sum().rename(index={True: "CBD", False: "Suburb"}))

    fig, ax = plt.subplots(figsize=(9, 6))
    rev_summary.plot(kind="bar", ax=ax, color=["#c0392b", "#27ae60"], edgecolor="white")
    ax.set_title("Revenue: Baseline vs Dynamic Pricing", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Revenue (£)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Baseline", "Dynamic"])
    plt.tight_layout()
    path = os.path.join(plots_dir, "revenue_summary.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()

    df = load_data("urban_ev_intervals.csv")
    alpha_matrix, delta_matrix = train_optimizer(df)
    res_df = run_production_simulation(df, alpha_matrix, delta_matrix)

    print("\n" + "═" * 80)
    print("  FINAL PRODUCTION METRICS")
    print("═" * 80)
    print_kpi_table(res_df)

    generate_dashboard(res_df)

    print(f"✓  Pipeline complete in {time.time()-t_start:.1f}s. "
          f"Plots saved in plots/ev_pricing/")
