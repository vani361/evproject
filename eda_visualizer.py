import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE_DIR = "/content/drive/MyDrive/evproject"
SUBMISSION_DIR = os.path.join(WORKSPACE_DIR, "submission")
PLOTS_DIR = os.path.join(SUBMISSION_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Premium styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
    'figure.figsize': (12, 6)
})

def compute_acf(series, max_lag=72):
    acf_vals = []
    for lag in range(max_lag + 1):
        acf_vals.append(series.autocorr(lag=lag) if lag > 0 else 1.0)
    return acf_vals

def plot_diurnal_split(df_acn, df_urban_cbd, df_urban_noncbd, target_col, target_label, filename):
    print(f"Generating Diurnal Split for {target_label}...")
    acn_wd = df_acn[df_acn['is_weekend'] == 0]
    acn_we = df_acn[df_acn['is_weekend'] == 1]
    cbd_wd = df_urban_cbd[df_urban_cbd['is_weekend'] == 0]
    cbd_we = df_urban_cbd[df_urban_cbd['is_weekend'] == 1]
    noncbd_wd = df_urban_noncbd[df_urban_noncbd['is_weekend'] == 0]
    noncbd_we = df_urban_noncbd[df_urban_noncbd['is_weekend'] == 1]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    color_wd = '#023E73'  # Blue for Weekday
    color_we = '#D9430D'  # Orange for Weekend
    
    # ROW 1: WEEKDAY
    sns.lineplot(data=acn_wd, x='hour', y=target_col, ax=axes[0, 0], color=color_wd, linewidth=2.5)
    axes[0, 0].set_title("1. ACN Caltech - WEEKDAY")
    axes[0, 0].set_ylabel(target_label)
    
    sns.lineplot(data=cbd_wd, x='hour', y=target_col, ax=axes[0, 1], color=color_wd, linewidth=2.5)
    axes[0, 1].set_title("2. Shenzhen CBD - WEEKDAY")
    axes[0, 1].set_ylabel(target_label)
    
    sns.lineplot(data=noncbd_wd, x='hour', y=target_col, ax=axes[0, 2], color=color_wd, linewidth=2.5)
    axes[0, 2].set_title("3. Shenzhen Non-CBD - WEEKDAY")
    axes[0, 2].set_ylabel(target_label)
    
    # ROW 2: WEEKEND
    sns.lineplot(data=acn_we, x='hour', y=target_col, ax=axes[1, 0], color=color_we, linewidth=2.5)
    axes[1, 0].set_title("4. ACN Caltech - WEEKEND")
    axes[1, 0].set_ylabel(target_label)
    
    sns.lineplot(data=cbd_we, x='hour', y=target_col, ax=axes[1, 1], color=color_we, linewidth=2.5)
    axes[1, 1].set_title("5. Shenzhen CBD - WEEKEND")
    axes[1, 1].set_ylabel(target_label)
    
    sns.lineplot(data=noncbd_we, x='hour', y=target_col, ax=axes[1, 2], color=color_we, linewidth=2.5)
    axes[1, 2].set_title("6. Shenzhen Non-CBD - WEEKEND")
    axes[1, 2].set_ylabel(target_label)
    
    for i in range(2):
        for j in range(3):
            axes[i, j].set_xlabel("Hour of Day")
            axes[i, j].set_xticks(range(0, 24, 2))
            
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()  # Added to display in notebook
    plt.close()

def plot_weekly_box(df_acn, df_urban_cbd, df_urban_noncbd, target_col, target_label, filename):
    print(f"Generating Weekly Profile for {target_label}...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.boxplot(data=df_acn, x='dayofweek', y=target_col, hue='dayofweek', ax=axes[0], palette="Blues", legend=False)
    axes[0].set_title("1. ACN Caltech (Workplace)")
    axes[0].set_ylabel(target_label)

    sns.boxplot(data=df_urban_cbd, x='dayofweek', y=target_col, hue='dayofweek', ax=axes[1], palette="Oranges", legend=False)
    axes[1].set_title("2. Shenzhen CBD (Urban Commuter)")
    axes[1].set_ylabel(target_label)

    sns.boxplot(data=df_urban_noncbd, x='dayofweek', y=target_col, hue='dayofweek', ax=axes[2], palette="Greens", legend=False)
    axes[2].set_title("3. Shenzhen Non-CBD (Residential/Fleet)")
    axes[2].set_ylabel(target_label)

    for ax in axes:
        ax.set_xlabel("Day of Week (0=Monday, 6=Sunday)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()  # Added to display in notebook
    plt.close()

def plot_period_box(df_acn, df_urban_cbd, df_urban_noncbd, target_col, target_label, filename):
    print(f"Generating Periods Volatility for {target_label}...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    order = ["Off-Peak", "Shoulder", "Peak"]

    sns.boxplot(data=df_acn, x='period', y=target_col, hue='period', order=order, ax=axes[0], palette="coolwarm", legend=False)
    axes[0].set_title("1. ACN Caltech (Workplace)")
    axes[0].set_ylabel(target_label)

    sns.boxplot(data=df_urban_cbd, x='period', y=target_col, hue='period', order=order, ax=axes[1], palette="coolwarm", legend=False)
    axes[1].set_title("2. Shenzhen CBD (Urban Commuter)")
    axes[1].set_ylabel(target_label)

    sns.boxplot(data=df_urban_noncbd, x='period', y=target_col, hue='period', order=order, ax=axes[2], palette="coolwarm", legend=False)
    axes[2].set_title("3. Shenzhen Non-CBD (Residential/Fleet)")
    axes[2].set_ylabel(target_label)

    for ax in axes:
        ax.set_xlabel("Operational Period")
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()  # Added to display in notebook
    plt.close()

def run_eda():
    print("--- STARTING TIME-SERIES EDA ---")
    
    acn_path = os.path.join(SUBMISSION_DIR, "acn_intervals.csv")
    urban_path = os.path.join(SUBMISSION_DIR, "urban_ev_intervals.csv")
    
    if not all(os.path.exists(p) for p in [acn_path, urban_path]):
        print("Error: Preprocessed files not found. Run data_aligner.py first.")
        return
        
    print("Loading datasets...")
    df_acn = pd.read_csv(acn_path)
    df_urban = pd.read_csv(urban_path)
    
    df_acn['timestamp'] = pd.to_datetime(df_acn['timestamp'])
    df_urban['timestamp'] = pd.to_datetime(df_urban['timestamp'])
    
    # -------------------------------------------------------------------------
    # BACKWARDS COMPATIBILITY FOR ACN
    # -------------------------------------------------------------------------
    if 'expected_load' not in df_acn.columns:
        df_acn['expected_load'] = df_acn['volume'] / df_acn['charging_hours'].replace(0, np.nan)
        df_acn['expected_load'] = df_acn['expected_load'].fillna(0.0)
    
    if 'queue_length_proxy' not in df_acn.columns:
        df_acn['queue_length_proxy'] = df_acn['idle_count']

    # -------------------------------------------------------------------------
    # DATA PREP
    # -------------------------------------------------------------------------
    top_grid = df_urban.groupby('grid')['occupancy'].mean().idxmax()
    df_urban_grid = df_urban[df_urban['grid'] == top_grid].sort_values(by='timestamp').copy()
    
    df_urban_sample = df_urban.sample(n=min(50000, len(df_urban)), random_state=42)
    
    df_urban_cbd = df_urban_sample[df_urban_sample['CBD'] == 1].copy()
    df_urban_noncbd = df_urban_sample[df_urban_sample['CBD'] == 0].copy()
    
    def classify_period(hour):
        if 8 <= hour < 12 or 18 <= hour < 22:
            return "Shoulder"
        elif 12 <= hour < 18:
            return "Peak"
        else:
            return "Off-Peak"
            
    df_acn['period'] = df_acn['hour'].apply(classify_period)
    df_urban_cbd['period'] = df_urban_cbd['hour'].apply(classify_period)
    df_urban_noncbd['period'] = df_urban_noncbd['hour'].apply(classify_period)

    # -------------------------------------------------------------------------
    # PLOT 1: Time Series Trend
    # -------------------------------------------------------------------------
    print("Generating Plot 1: Trend and Rolling Statistics...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(df_acn['timestamp'][:720], df_acn['volume'][:720], label='Actual Demand (kWh)', color='#0A7E8C', alpha=0.3)
    axes[0].plot(df_acn['timestamp'][:720], df_acn['volume'][:720].rolling(24).mean(), label='24h Rolling Mean', color='#023E73', linewidth=2)
    axes[0].plot(df_acn['timestamp'][:720], df_acn['volume'][:720].rolling(168).mean(), label='168h (Weekly) Rolling Mean', color='#D9430D', linewidth=2)
    axes[0].set_title("ACN Caltech - Charging Demand & Rolling Statistics")
    axes[0].set_ylabel("Charging Demand (kWh)")
    axes[0].legend(loc='upper right')
    
    axes[1].plot(df_urban_grid['timestamp'][:720], df_urban_grid['volume'][:720], label='Actual Demand (kWh)', color='#439A86', alpha=0.3)
    axes[1].plot(df_urban_grid['timestamp'][:720], df_urban_grid['volume'][:720].rolling(24).mean(), label='24h Rolling Mean', color='#012E40', linewidth=2)
    axes[1].plot(df_urban_grid['timestamp'][:720], df_urban_grid['volume'][:720].rolling(168).mean(), label='168h (Weekly) Rolling Mean', color='#D96B27', linewidth=2)
    axes[1].set_title(f"UrbanEV Grid {top_grid} (Shenzhen) - Charging Demand & Rolling Statistics")
    axes[1].set_ylabel("Charging Demand (kWh)")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "time_series_rolling_stats.png"), dpi=150)
    plt.show()  # Added to display in notebook
    plt.close()
    
    # -------------------------------------------------------------------------
    # PLOT 2: Autocorrelation (ACF) 
    # -------------------------------------------------------------------------
    print("Generating Plot 2: Autocorrelation (ACF)...")
    acf_acn = compute_acf(df_acn['volume'], max_lag=72)
    acf_urban = compute_acf(df_urban_grid['volume'], max_lag=72)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    axes[0].bar(range(len(acf_acn)), acf_acn, color='#023E73', width=0.6)
    axes[0].axhline(y=0, color='gray', linestyle='-')
    ci = 1.96 / np.sqrt(len(df_acn))
    axes[0].axhline(y=ci, color='red', linestyle='--', alpha=0.5)
    axes[0].axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title("ACN Caltech - Autocorrelation")
    axes[0].set_xlabel("Lag (Hours)")
    axes[0].set_ylabel("ACF Value")
    axes[0].set_ylim(-0.5, 1.1)
    axes[0].set_xticks(np.arange(0, 73, 12))
    for lag_val in [24, 48, 72]:
        axes[0].axvline(x=lag_val, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
    
    axes[1].bar(range(len(acf_urban)), acf_urban, color='#439A86', width=0.6)
    axes[1].axhline(y=0, color='gray', linestyle='-')
    ci_u = 1.96 / np.sqrt(len(df_urban_grid))
    axes[1].axhline(y=ci_u, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-ci_u, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title("UrbanEV Shenzhen - Autocorrelation")
    axes[1].set_xlabel("Lag (Hours)")
    axes[1].set_ylabel("ACF Value")
    axes[1].set_ylim(-0.5, 1.1)
    axes[1].set_xticks(np.arange(0, 73, 12))
    for lag_val in [24, 48, 72]:
        axes[1].axvline(x=lag_val, color='black', linestyle=':', alpha=0.5, linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "autocorrelation_acf.png"), dpi=150)
    plt.show()  # Added to display in notebook
    plt.close()

    # -------------------------------------------------------------------------
    # PLOT 3: Feature Dependencies
    # -------------------------------------------------------------------------
    print("Generating Plot 3: Feature Dependencies...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.scatterplot(data=df_urban_sample, x='expected_load_lag_24', y='expected_load', alpha=0.3, ax=axes[0, 0], color='#023E73')
    axes[0, 0].set_title("Dependency: Power Load vs 24h Lag")
    axes[0, 0].set_xlabel("Expected Load 24h Ago (kW)")
    axes[0, 0].set_ylabel("Current Expected Load (kW)")

    sns.regplot(data=df_urban_sample, x='grid_mean_dist', y='charging_utilisation_rate', 
                scatter_kws={'alpha': 0.1, 's': 15, 'color': '#D9430D'}, 
                line_kws={'color': '#012E40', 'linewidth': 3}, 
                ax=axes[0, 1])
    axes[0, 1].set_title("Dependency: Utilization vs Spatial Topology (Regression)")
    axes[0, 1].set_xlabel("Grid Mean Distance to Neighbors (km)")
    axes[0, 1].set_ylabel("Charging Utilisation Rate")
    
    sns.boxplot(data=df_urban_sample, x='is_weekend', y='volume', hue='is_weekend', ax=axes[1, 0], palette="Set2", legend=False)
    axes[1, 0].set_title("Dependency: Demand vs Time Engineered (Weekend)")
    axes[1, 0].set_xlabel("Is Weekend (0=No, 1=Yes)")
    axes[1, 0].set_ylabel("Charging Demand (kWh)")
    
    bins = [0, 50, 100, 150, 200, 500]
    labels = ['Small\n(<50)', 'Medium\n(50-100)', 'Large\n(100-150)', 'Very Large\n(150-200)', 'Massive Hub\n(>200)']
    df_urban_sample['capacity_bin'] = pd.cut(df_urban_sample['scale'], bins=bins, labels=labels)
    
    sns.barplot(data=df_urban_sample, x='capacity_bin', y='is_congested', hue='capacity_bin', ax=axes[1, 1], palette="viridis", errorbar=None, legend=False)
    axes[1, 1].set_title("Dependency: Congestion vs Capacity (Binned)")
    axes[1, 1].set_xlabel("Station Capacity Class")
    axes[1, 1].set_ylabel("Probability of Congestion")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_dependencies.png"), dpi=150)
    plt.show()  # Added to display in notebook
    plt.close()

    # -------------------------------------------------------------------------
    # MULTI-TARGET PLOTTING (Volume, Expected Load, Queue Length)
    # -------------------------------------------------------------------------
    targets = [
        ('volume', 'Charging Demand (kWh)'),
        ('expected_load', 'Expected Power Load (kW)'),
        ('queue_length_proxy', 'Queue Length (Streak)')
    ]
    
    for col, label in targets:
        plot_diurnal_split(df_acn, df_urban_cbd, df_urban_noncbd, col, label, f"diurnal_split_{col}.png")
        plot_weekly_box(df_acn, df_urban_cbd, df_urban_noncbd, col, label, f"threeway_weekly_{col}.png")
        plot_period_box(df_acn, df_urban_cbd, df_urban_noncbd, col, label, f"threeway_periods_{col}.png")
        
    print("--- EDA VISUALISATIONS GENERATED SUCCESSFULLY ---")

if __name__ == "__main__":
    run_eda()
