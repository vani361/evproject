import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, f1_score

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
    print("[+] XGBoost is available and will be used for training.")
except Exception as e:
    XGB_AVAILABLE = False
    print("[-] XGBoost could not be loaded (missing libomp on macOS). Falling back to RandomForest.")

# Flexible workspace path handling for both Colab and local environments
WORKSPACE_DIR = "/content/drive/MyDrive/evproject"
if not os.path.exists(WORKSPACE_DIR):
    WORKSPACE_DIR = "/Users/vani/Documents/ev project"
    
SUBMISSION_DIR = os.path.join(WORKSPACE_DIR, "submission")
PLOTS_DIR = os.path.join(SUBMISSION_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

def prepare_urbanev_model_data(df, target_col):
    """Isolates target and drops leaky features."""
    df_model = df.copy()

    if 'timestamp' in df_model.columns:
        df_model['timestamp'] = pd.to_datetime(df_model['timestamp'])
        df_model = df_model.sort_values(by=['timestamp', 'grid'])
        
    df_model['is_cbd_peak'] = ((df_model['hour'] >= 14) & (df_model['hour'] <= 18) & (df_model['CBD'] == 1)).astype(int)
    df_model['is_fleet_night'] = ((df_model['hour'] >= 0) & (df_model['hour'] <= 6) & (df_model['CBD'] == 0)).astype(int)

    leaky_features = [
        'occupancy', 'volume', 'duration', 'price',
        'occupancy_density', 'avg_power_kw', 'estimated_charging_time', 
        'total_time_plugged_in', 'charging_utilisation_rate', 
        'utilization_rate_hr', 'total_utilization_rate', 
        'queue_length_proxy', 'is_congested', 'expected_load', 
        'baseline_revenue', 'global_occ_mean', 'global_vol_mean', 
        'charging_demand', 'is_congested_5m', 'baseline_energy_cost', 'baseline_profit'
    ]
    
    if target_col in leaky_features:
        leaky_features.remove(target_col)
        
    cols_to_drop = [col for col in leaky_features if col in df_model.columns]
    df_model.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    if 'timestamp' in df_model.columns:
        df_model.drop(columns=['timestamp'], inplace=True, errors='ignore')
    
    df_model.drop(columns=['grid', 'fast_charger_ratio', 'fast_count', 'slow_count'], inplace=True, errors='ignore')
    df_model.fillna(0, inplace=True) 
    
    return df_model

def train_model(X_train, y_train, model_type):
    if XGB_AVAILABLE:
        if model_type == 'Regressor':
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        elif model_type == 'Classifier':
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss')
    else:
        if model_type == 'Regressor':
            model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        elif model_type == 'Classifier':
            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    return model



def test_model(model, X_test, y_test, model_type, target_col):
    preds = model.predict(X_test)
    
    # Bounded prediction logic: XGBRegressor outputs sums of regression trees
    # which can mathematically go negative. We clip to 0.0 for physical boundaries.
    if model_type == 'Regressor':
        preds = np.maximum(0.0, preds)
        
    print("=========================================")
    print(f"RESULTS: {target_col.upper()}")
    
    # Capture scores to save to CSV
    scores = {'Target': target_col, 'Model_Type': model_type}
    
    if model_type == 'Regressor':
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f} (Accuracy Score)")
        
        scores.update({'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Accuracy': np.nan, 'ROC_AUC': np.nan, 'F1_Score': np.nan})
        return preds, scores
        
    elif model_type == 'Classifier':
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC:  {auc:.4f} (Probability Score)")
        print(f"F1 Score: {f1:.4f} (Precision/Recall Balance)")
        
        scores.update({'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Accuracy': acc, 'ROC_AUC': auc, 'F1_Score': f1})
        return probs, scores # Return probabilities for the pricing engine
        
    print("=========================================")
        
def plot_importance(model, feature_names, target_col):
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(f"Top 15 Predictive Features for {target_col.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"importance_{target_col}.png"), dpi=150)
    plt.show() 
    plt.close() 


def simulate_live_inference(trained_models, df, feature_columns, target_hour=15):
    # Dynamically find the absolute busiest grid in the dataset so we know data exists!
    busiest_grid = df.groupby('grid')['charging_demand'].mean().idxmax()
    
    print("\n" + "="*15)
    print(f" INITIATING LIVE INFERENCE SIMULATION")
    print(f"Target: Busiest Station (Grid #{busiest_grid}) @ {target_hour}:00 (3:00 PM)")
    print("="*60)
    
    # Grab a recent snapshot of this specific grid and hour
    filtered_df = df[(df['grid'] == busiest_grid) & (df['hour'] == target_hour)]
    
    # Fallback just in case this specific hour is missing
    if len(filtered_df) == 0:
        base_state = df.iloc[-1].copy()
    else:
        base_state = filtered_df.iloc[-1].copy()
    
    # We will run two inference scenarios to show the Agent reacting to live changes
    scenarios = [
        {"name": "Normal Day", "vol_lag_multiplier": 1.0},
        {"name": "Surge Event (24h ago was 3x busier)", "vol_lag_multiplier": 3.0}
    ]
    
    for scenario in scenarios:
        print(f"\n--- SCENARIO: {scenario['name']} ---")
        
        # Load the base state into a 1-row DataFrame
        live_df = pd.DataFrame([base_state])
        
        # Apply our hypothetical live scenario hack
        if 'vol_lag_24' in live_df.columns:
            live_df['vol_lag_24'] *= scenario['vol_lag_multiplier']
            
        for target_col, model in trained_models.items():
            # Systematically clean the live data exactly like we did in training
            clean_df = prepare_urbanev_model_data(live_df, target_col=target_col)
            if target_col in clean_df.columns:
                clean_df.drop(columns=[target_col], inplace=True)
                
            # Ensure the columns perfectly match the input the model was trained on
            X_new = clean_df[feature_columns[target_col]]
            
            # RUN FAST INFERENCE!
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_new)[0][1]
                print(f"  > {target_col.upper():<25}: {prob * 100:>5.1f}% Risk")
            else:
                pred = model.predict(X_new)[0]
                # Clip prediction to avoid negative values in simulation
                pred = max(0.0, float(pred))
                print(f"  > {target_col.upper():<25}: {pred:>7.2f}")

def run_demand_agent():
    print("Loading UrbanEV Intervals data")
    df = pd.read_csv(os.path.join(SUBMISSION_DIR, "urban_ev_intervals.csv"))
    
    if 'charging_demand' not in df.columns:
        df['charging_demand'] = df['volume']
        
    targets = [
        ('charging_demand', 'Regressor'),
        ('charging_utilisation_rate', 'Regressor'),
        ('expected_load', 'Regressor'),
        ('occupancy_density', 'Regressor'),
        ('queue_length_proxy', 'Regressor'),
        ('is_congested', 'Classifier')
    ]
    
    trained_models = {}
    feature_columns = {}
    all_scores = []
    
    # Setup base dataframe perfectly sorted to match our model's test split
    df_sorted = df.copy()
    if 'timestamp' in df_sorted.columns:
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
        df_sorted = df_sorted.sort_values(by=['timestamp', 'grid']).reset_index(drop=True)
        
    train_size = int(len(df_sorted) * 0.8)
    
    # We will attach our predictions directly to this test subset
    predictions_df = df_sorted.iloc[train_size:].copy()
    
    for target_col, model_type in targets:
        print(f"\n[+] Processing: {target_col.upper()}...")
        df_model = prepare_urbanev_model_data(df, target_col=target_col)
        
        train_size_model = int(len(df_model) * 0.8)
        train = df_model.iloc[:train_size_model]
        test = df_model.iloc[train_size_model:]
        
        X_train, y_train = train.drop(columns=[target_col], errors='ignore'), train[target_col]
        X_test, y_test = test.drop(columns=[target_col], errors='ignore'), test[target_col]
        
        feature_columns[target_col] = X_train.columns
        
        model = train_model(X_train, y_train, model_type)
        trained_models[target_col] = model
        
        # Capture the raw predictions and scores from our updated function
        preds, scores = test_model(model, X_test, y_test, model_type, target_col)
        
        all_scores.append(scores)
        
        # Attach the exact prediction array to our exported CSV dataframe
        predictions_df[f'predicted_{target_col}'] = preds
        
        plot_importance(model, X_train.columns, target_col)
        
    simulate_live_inference(trained_models, df, feature_columns, target_hour=15)
    
    # =========================================================
    # EXPORT RESULTS FOR THE PRICING AGENT
    # =========================================================
    print("\n--- SAVING ML OUTPUTS ---")
    
    # Save the evaluation scores matrix
    scores_csv = os.path.join(SUBMISSION_DIR, "demand_agent_scores.csv")
    pd.DataFrame(all_scores).to_csv(scores_csv, index=False)
    print(f" Saved evaluation metrics for all 6 targets to: {scores_csv}")
    
    # Save only the 6 predicted columns as requested
    preds_csv = os.path.join(SUBMISSION_DIR, "predicted_urban_ev_intervals.csv")
    predicted_cols = [f'predicted_{target_col}' for target_col, _ in targets]
    predictions_df[predicted_cols].to_csv(preds_csv, index=False)
    print(f" Saved full timeline predictions to: {preds_csv}")

if __name__ == "__main__":
    run_demand_agent()
