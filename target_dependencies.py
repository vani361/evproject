import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

WORKSPACE_DIR = r"/content/drive/MyDrive/evproject"
SUBMISSION_DIR = os.path.join(WORKSPACE_DIR, "submission")
PLOTS_DIR = os.path.join(SUBMISSION_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Loading UrbanEV data...")
df_urban = pd.read_csv(os.path.join(SUBMISSION_DIR, "urban_ev_intervals.csv"))

# Ensure backward compatibility for the new names:
if 'charging_demand' not in df_urban.columns:
    df_urban['charging_demand'] = df_urban['volume']

df_urban = df_urban.sort_values(by=['grid', 'timestamp'])
df_urban['util_lag_1'] = df_urban.groupby('grid')['charging_utilisation_rate'].shift(1).fillna(0)

# Subsample for faster scatter plotting
df_sample = df_urban.sample(n=min(10000, len(df_urban)), random_state=42)

print("Generating 5x3 Master Target Dependency Grid...")
fig, axes = plt.subplots(5, 3, figsize=(20, 30))
colors = ['#023E73', '#D9430D']  # Blue for Non-CBD, Orange for CBD

# ==========================================
# ROW 1: TARGET 1 - CHARGING DEMAND (kWh)
# ==========================================
sns.lineplot(data=df_urban, x='hour', y='charging_demand', hue='CBD', palette=colors, ax=axes[0, 0])
axes[0, 0].set_title("Charging Demand vs Time of Day")
axes[0, 0].set_ylabel("Charging Demand (kWh)")

sns.regplot(data=df_sample, x='scale', y='charging_demand', scatter_kws={'alpha': 0.1, 's': 10, 'color': '#0A7E8C'}, line_kws={'color': '#023E73'}, ax=axes[0, 1])
axes[0, 1].set_title("Charging Demand vs Station Capacity")
axes[0, 1].set_ylabel("Charging Demand (kWh)")

sns.regplot(data=df_sample, x='vol_lag_24', y='charging_demand', scatter_kws={'alpha': 0.1, 's': 10, 'color': '#D9430D'}, line_kws={'color': '#023E73'}, ax=axes[0, 2])
axes[0, 2].set_title("Demand vs 24h Demand Lag")
axes[0, 2].set_ylabel("Charging Demand (kWh)")

# ==========================================
# ROW 2: TARGET 2 - EXPECTED LOAD (Power kW)
# ==========================================
sns.lineplot(data=df_urban, x='hour', y='expected_load', hue='CBD', palette=colors, ax=axes[1, 0])
axes[1, 0].set_title("Power Load vs Time of Day")
axes[1, 0].set_ylabel("Expected Load (kW)")

sns.regplot(data=df_sample, x='scale', y='expected_load', scatter_kws={'alpha': 0.1, 's': 10, 'color': '#0A7E8C'}, line_kws={'color': '#023E73'}, ax=axes[1, 1])
axes[1, 1].set_title("Power Load vs Station Capacity")
axes[1, 1].set_ylabel("Expected Load (kW)")

sns.regplot(data=df_sample, x='expected_load_lag_24', y='expected_load', scatter_kws={'alpha': 0.1, 's': 10, 'color': '#D9430D'}, line_kws={'color': '#023E73'}, ax=axes[1, 2])
axes[1, 2].set_title("Load vs 24h Power Lag")
axes[1, 2].set_ylabel("Expected Load (kW)")

# ==========================================
# ROW 3: TARGET 3 - QUEUE LENGTH
# ==========================================
sns.lineplot(data=df_urban, x='hour', y='queue_length_proxy', hue='CBD', palette=colors, ax=axes[2, 0])
axes[2, 0].set_title("Queue Length vs Time of Day")
axes[2, 0].set_ylabel("Queue Length")

sns.regplot(data=df_sample, x='scale', y='queue_length_proxy', scatter_kws={'alpha': 0.1, 's': 10, 'color': '#0A7E8C'}, line_kws={'color': '#023E73'}, ax=axes[2, 1])
axes[2, 1].set_title("Queue Length vs Station Capacity")
axes[2, 1].set_ylabel("Queue Length")

sns.regplot(data=df_sample, x='queue_lag_24', y='queue_length_proxy', scatter_kws={'alpha': 0.1, 's': 10, 'color': '#D9430D'}, line_kws={'color': '#023E73'}, ax=axes[2, 2])
axes[2, 2].set_title("Queue vs 24h Queue Lag")
axes[2, 2].set_ylabel("Queue Length")

# ==========================================
# ROW 4: TARGET 4 - CHARGING UTILISATION RATE
# ==========================================
sns.lineplot(data=df_urban, x='hour', y='charging_utilisation_rate', hue='CBD', palette=colors, ax=axes[3, 0])
axes[3, 0].set_title("Utilisation vs Time of Day")
axes[3, 0].set_ylabel("Utilisation Rate")

sns.regplot(data=df_sample, x='util_lag_1', y='charging_utilisation_rate', scatter_kws={'alpha': 0.1, 's': 10, 'color': '#439A86'}, line_kws={'color': '#023E73'}, ax=axes[3, 1])
axes[3, 1].set_title("Utilisation vs 1h Lag")
axes[3, 1].set_ylabel("Utilisation Rate")

sns.scatterplot(data=df_sample, x='util_lag_24', y='charging_utilisation_rate', hue='CBD', palette=colors, alpha=0.4, s=20, ax=axes[3, 2])
axes[3, 2].set_title("Utilisation vs 24h Lag")
axes[3, 2].set_ylabel("Utilisation Rate")

# ==========================================
# ROW 5: TARGET 5 - CONGESTION PROBABILITY
# ==========================================
sns.barplot(data=df_urban, x='hour', y='is_congested', hue='CBD', palette=colors, errorbar=None, ax=axes[4, 0])
axes[4, 0].set_title("Congestion vs Time of Day")
axes[4, 0].set_ylabel("Congestion Probability")

df_urban['volume_bin'] = pd.qcut(df_urban['charging_demand'], q=5, duplicates='drop')
sns.barplot(data=df_urban, x='volume_bin', y='is_congested', hue='CBD', palette=colors, errorbar=None, ax=axes[4, 1])
axes[4, 1].set_title("Congestion vs Demand Quantiles")
axes[4, 1].set_ylabel("Congestion Probability")
axes[4, 1].tick_params(axis='x', rotation=15)

df_urban['lag_bin'] = pd.qcut(df_urban['vol_lag_24'], q=5, duplicates='drop')
sns.barplot(data=df_urban, x='lag_bin', y='is_congested', hue='CBD', palette=colors, errorbar=None, ax=axes[4, 2])
axes[4, 2].set_title("Congestion vs Lag Quantiles")
axes[4, 2].set_ylabel("Congestion Probability")
axes[4, 2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'target_dependencies.png'), dpi=300, bbox_inches='tight')
plt.show()

print("Successfully generated massive 5x3 Master Target grid without errors!")