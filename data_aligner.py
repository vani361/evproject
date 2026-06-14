import os
import pandas as pd
import numpy as np

WORKSPACE_DIR = r"/content/drive/MyDrive/evproject"
SUBMISSION_DIR = os.path.join(WORKSPACE_DIR, "submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)



def add_cyclical_features(df, col):
    """
    Extracts time features from a timezone-aware UTC datetime column by converting
    it to America/Los_Angeles local time first
    """
    dt_series = pd.to_datetime(df[col])
    if dt_series.dt.tz is not None:
        dt_local = dt_series.dt.tz_convert('America/Los_Angeles')
    else:
        dt_local = dt_series
        
    df['hour'] = dt_local.dt.hour
    df['dayofweek'] = dt_local.dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    return df
def get_tod_cost(hour):
    
    if 12 <= hour < 18:
        return 10.0
    elif (8 <= hour < 12) or (18 <= hour < 22):
        return 7.0
    else:
        return 4.0

def preprocess_acn():
    print("--- PREPROCESSING ACN-DATA ---")
    xlsx_path = os.path.join(WORKSPACE_DIR, "acndata_sessions.json.xlsx")
    if not os.path.exists(xlsx_path): return None
    
    df_clean = pd.read_excel(xlsx_path)
    
    # Drop specified columns 
    cols_to_drop = [
        '_meta', 'end', 'min_kWh', 'start', '_items', 'userInputs', 'site',
        'WhPerMile', 'kWhRequested', 'milesRequested', 'minutesAvailable', 
        'modifiedAt', 'paymentRequired', 'requestedDeparture', 'userID.1','_id','clusterID',
        'siteID','timezone'
    ]
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
    

    #drop missing vals of session id 
    df_clean = df_clean.dropna(subset=['sessionID'])

    #cleaning for spaceid and stationid
    # For '2-39-78-362':
    df_clean['station_cluster_code'] = df_clean['stationID'].apply(lambda x: int(x.split('-')[2]))

    df_clean['station_node_code'    ] = df_clean['stationID'].apply(lambda x: int(x.split('-')[3]))
    df_clean['space_num'] = df_clean['spaceID'].apply(lambda x: int(x.split('-')[-1]))

      

    # Impute doneChargingTime using disconnectTime (handles the 8 missing values)
    df_clean['doneChargingTime'] = df_clean['doneChargingTime'].fillna(df_clean['disconnectTime'])
    
    # Impute userID for guest/anonymous sessions
    df_clean['userID'] = df_clean['userID'].fillna(-1)
    
    # Parse timestamps to UTC and convert to localized Pacific Time (US/Pacific)
    time_cols = ['connectionTime', 'disconnectTime', 'doneChargingTime']
    for col in time_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], utc=True, errors='coerce')
        df_clean[col + '_local'] = df_clean[col].dt.tz_convert('America/Los_Angeles')
        
    # Calculate Dwell Time, Charging Time, and Idle Time in hours
    # Connection Dwell Hours (Duration connected to plug)
    df_clean['dwell_hours'] = (df_clean['disconnectTime_local'] - df_clean['connectionTime_local']).dt.total_seconds() / 3600.0
    
    # Active Charging Hours (Time from plug-in to charge done)
    df_clean['charging_hours'] = (df_clean['doneChargingTime_local'] - df_clean['connectionTime_local']).dt.total_seconds() / 3600.0
    
    # Cap active charging time by dwell connection time and ensure no negative values
    df_clean['charging_hours'] = np.minimum(df_clean['charging_hours'], df_clean['dwell_hours'])
    df_clean['charging_hours'] = np.maximum(df_clean['charging_hours'], 0.0)
    
    # Idle Connection Hours (Plugged in but not charging)
    df_clean['idle_hours'] = df_clean['dwell_hours'] - df_clean['charging_hours']
    df_clean['idle_hours'] = np.maximum(df_clean['idle_hours'], 0.0)
    
   
    df_clean = df_clean[(df_clean['dwell_hours'] > 0) & (df_clean['kWhDelivered'] > 0)].copy()
    
    #  Local Temporal Features
    
    df_clean=add_cyclical_features(df_clean,'connectionTime_local')
    df_clean['date'] =  pd.to_datetime(df_clean['connectionTime_local'].dt.date)
    
    # baseline pricing metrics
   
    df_clean['baseline_revenue'] = df_clean['kWhDelivered'] * 15.0

    print("saved to drive as acn_cleaned_sessions.csv")
    
    acn_sessions_csv = os.path.join(SUBMISSION_DIR, "acn_cleaned_sessions.csv")
    df_clean.to_csv(acn_sessions_csv, index=False)
    return df_clean



def aggregate_acn_to_intervals(df_sessions):
    
    print("\n--- AGGREGATING ACN-DATA TO HOURLY INTERVALS (DIRECT HOURLY OVERLAP METHOD) ---")
    
    #  Clean and align timestamps, converting to local California time  
    df_sessions = df_sessions.copy()
    df_sessions['connectionTime'] = pd.to_datetime(df_sessions['connectionTime'], utc=True).dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
    df_sessions['disconnectTime'] = pd.to_datetime(df_sessions['disconnectTime'], utc=True).dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
    df_sessions['doneChargingTime'] = pd.to_datetime(df_sessions['doneChargingTime'], utc=True).dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
             
    # Guarantee doneChargingTime is bounded between connection and disconnect times
    df_sessions['doneChargingTime'] = np.minimum(
        np.maximum(df_sessions['doneChargingTime'], df_sessions['connectionTime']),
        df_sessions['disconnectTime']
    )
    
    # Calculate average charge rate (kW) for each session, capped at typical Level 2 charger speed (11 kW)
    df_sessions['charge_rate_kw'] = np.where(df_sessions['charging_hours'] > 0.05,
                                             df_sessions['kWhDelivered'] / df_sessions['charging_hours'], 0.0)
    df_sessions['charge_rate_kw'] = df_sessions['charge_rate_kw'].clip(upper=11.0)
    
    # hourly intervals  
    min_time = df_sessions['connectionTime'].min().floor('h')
    max_time = df_sessions['disconnectTime'].max().ceil('h')
    intervals = pd.date_range(start=min_time, end=max_time, freq='h')
    n_hours = len(intervals) - 1
    print(f"Time span: {min_time} to {max_time} (Hourly intervals: {n_hours})")
    
    avg_session_dwell = np.zeros(n_hours)
    avg_session_charging = np.zeros(n_hours)
    avg_session_idle = np.zeros(n_hours)
    avg_session_kwh = np.zeros(n_hours) 
    avg_session_revenue = np.zeros(n_hours)

    avg_cluster_code = np.zeros(n_hours)
    avg_node_code = np.zeros(n_hours)
    avg_space_num = np.zeros(n_hours)
    

    unique_users = np.zeros(n_hours, dtype=int)
    unique_stations = np.zeros(n_hours, dtype=int) 
    unique_clusters = np.zeros(n_hours, dtype=int)

    occ_array = np.zeros(n_hours)
    chg_array = np.zeros(n_hours)
    vol_array = np.zeros(n_hours)
    idle_array = np.zeros(n_hours)
    
    conn_times = df_sessions['connectionTime'].values
    disc_times = df_sessions['disconnectTime'].values
    done_times = df_sessions['doneChargingTime'].values
    rates = df_sessions['charge_rate_kw'].values
    
    dwells = df_sessions['dwell_hours'].values
    chargings = df_sessions['charging_hours'].values
    idles = df_sessions['idle_hours'].values
    kwhs = df_sessions['kWhDelivered'].values
    revenues = df_sessions['baseline_revenue'].values
    user_ids = df_sessions['userID'].fillna(-1).values
    station_ids = df_sessions['stationID'].values
    cluster_codes = df_sessions['station_cluster_code'].values
    cluster_node_codes = df_sessions['station_node_code'].values
    space_nums = df_sessions['space_num'].values
    

    #fractional overlaps with active hours
    for i in range(len(df_sessions)):
        conn, disc, done, rate = conn_times[i], disc_times[i], done_times[i], rates[i]
        
        # Only search hourly buckets that fall within the session's duration
        h_start_idx = max(0, np.searchsorted(intervals, conn) - 1)
        h_end_idx = min(n_hours, np.searchsorted(intervals, disc))
        
        for idx in range(h_start_idx, h_end_idx):
            hour_start = intervals[idx]
            hour_end = intervals[idx + 1]
            
            # Occupancy fraction (overlap of [conn, disc] with [hour_start, hour_end])
            o_overlap = (min(disc, hour_end) - max(conn, hour_start)) / pd.Timedelta(hours=1)
            o_overlap = max(0.0, o_overlap)
            occ_array[idx] += o_overlap
            
            # Charging fraction (overlap of [conn, done] with [hour_start, hour_end])
            c_overlap = (min(done, hour_end) - max(conn, hour_start)) / pd.Timedelta(hours=1)
            c_overlap = max(0.0, c_overlap)
            chg_array[idx] += c_overlap
            
            # Idle fraction (occupancy - charging duration)
            idle_array[idx] += max(0.0, o_overlap - c_overlap)
            
            # Energy delivered: rate * active charging duration in this hour
            vol_array[idx] += rate * c_overlap

    print("Aggregating session-level features for each hour...")
    for idx in range(n_hours):
            hour_start = intervals[idx]
            hour_end = intervals[idx + 1]
            
            # Active sessions in this hour: started before hour_end and ended after hour_start
            active_indices = np.where((conn_times < hour_end) & (disc_times > hour_start))[0]
            
            if len(active_indices) > 0:
                avg_session_dwell[idx] = np.mean(dwells[active_indices])
                avg_session_charging[idx] = np.mean(chargings[active_indices])
                avg_session_idle[idx] = np.mean(idles[active_indices])
                avg_session_kwh[idx] = np.mean(kwhs[active_indices])
                
                avg_cluster_code[idx] = np.mean(cluster_codes[active_indices])
                avg_node_code[idx] = np.mean(cluster_node_codes[active_indices])
                avg_space_num[idx] = np.mean(space_nums[active_indices])
                
                # Unique counts
                unique_users[idx] = len(np.unique(user_ids[active_indices]))
                unique_stations[idx] = len(np.unique(station_ids[active_indices]))
                unique_clusters[idx] = len(np.unique(cluster_codes[active_indices]))
            
    df_hourly = pd.DataFrame({
        'timestamp': intervals[:-1],
        'timestamp_idx': np.arange(n_hours) + 1,
        'occupancy': occ_array,
        'charging_count': chg_array,
        'volume': vol_array,
        'idle_count': idle_array,
        'avg_station_cluster_code': avg_cluster_code,
        'avg_station_node_code': avg_node_code,
        'avg_space_num': avg_space_num,
        'avg_session_dwell': avg_session_dwell,
        'avg_session_charging': avg_session_charging,
        'avg_session_idle': avg_session_idle,
        'avg_session_kwh': avg_session_kwh,

        'dwell_hours': occ_array,
        'charging_hours': chg_array,
        'idle_hours': idle_array,
        'kWhDelivered': vol_array,
        
        'unique_users_count': unique_users,
        'unique_stations_count': unique_stations,
        'unique_clusters_count': unique_clusters
    })  
    
    scale = 54
    
    df_hourly['occupancy'] = df_hourly['occupancy'].clip(upper=scale)
    df_hourly['charging_count'] = df_hourly['charging_count'].clip(upper=scale)
    df_hourly['idle_count'] = df_hourly['idle_count'].clip(upper=scale)
    
    df_hourly['occupancy_density'] = df_hourly['occupancy'] / scale
    
    # metrics
    df_hourly['utilization_rate_hr'] = df_hourly['charging_count'] / scale
    total_site_utilization = df_hourly['charging_count'].sum() / (scale * len(df_hourly))
    df_hourly['total_utilization_rate'] = total_site_utilization
    
    df_hourly['queue_length_proxy'] = df_hourly['idle_count']
    df_hourly = add_cyclical_features(df_hourly, 'timestamp')
    df_hourly['month'] = df_hourly['timestamp'].dt.month 

    df_hourly['is_weekend'] = (df_hourly['dayofweek'] >= 5).astype(int)

    # Add lags for time-series forecasting
    df_hourly['occ_lag_1'] = df_hourly['occupancy_density'].shift(1).fillna(0.0)
    df_hourly['occ_lag_2'] = df_hourly['occupancy_density'].shift(2).fillna(0.0)
    df_hourly['occ_lag_24'] = df_hourly['occupancy_density'].shift(24).fillna(0.0)
    
    df_hourly['vol_lag_1'] = df_hourly['volume'].shift(1).fillna(0.0)
    df_hourly['vol_lag_2'] = df_hourly['volume'].shift(2).fillna(0.0)
    df_hourly['vol_lag_24'] = df_hourly['volume'].shift(24).fillna(0.0)

    #df_hourly['energy_cost_per_kwh'] = df_hourly['hour'].apply(get_tod_cost)
    df_hourly['baseline_revenue'] = df_hourly['volume'] * 15.0
    #df_hourly['baseline_energy_cost'] = df_hourly['volume'] * df_hourly['energy_cost_per_kwh']
    #df_hourly['baseline_profit'] = df_hourly['baseline_revenue'] - df_hourly['baseline_energy_cost']
    
    acn_intervals_csv = os.path.join(SUBMISSION_DIR, "acn_intervals.csv")
    df_hourly.to_csv(acn_intervals_csv, index=False)
    print(f"Saved HOURLY ACN interval data to {acn_intervals_csv} (Rows: {len(df_hourly)})")
    return df_hourly

    
def preprocess_urbanev():
    print("\n--- PREPROCESSING URBAN-EV HOURLY ---")
    info_df = pd.read_csv(os.path.join(WORKSPACE_DIR, "information.csv"))
    time_df = pd.read_csv(os.path.join(WORKSPACE_DIR, "time.csv"))
    occ_mat = pd.read_csv(os.path.join(WORKSPACE_DIR, "occupancy.csv"))
    vol_mat = pd.read_csv(os.path.join(WORKSPACE_DIR, "volume.csv"))
    dur_mat = pd.read_csv(os.path.join(WORKSPACE_DIR, "duration.csv"))
    price_mat = pd.read_csv(os.path.join(WORKSPACE_DIR, "price.csv"))
    

    # Process spatial adjacency and distance files
    adj_df = pd.read_csv(os.path.join(WORKSPACE_DIR, "adj.csv"))
    dist_df = pd.read_csv(os.path.join(WORKSPACE_DIR, "distance.csv"))
    
    adj_df.set_index('node_id', inplace=True)
    adj_count = adj_df.sum(axis=1) - 1.0  # Subtract self-connection
    
    grid_cols = list(dist_df.columns[1:])
    dist_matrix = dist_df[grid_cols].values
    grid_ids = [int(g) for g in grid_cols]
    
    mean_dist = []
    min_dist = []
    nearby_2km = []
    nearby_5km = []
    
    for i in range(len(grid_ids)):
        row_vals = dist_matrix[i, :]
        other_vals = row_vals[row_vals > 0.0]
        mean_dist.append(np.mean(other_vals) if len(other_vals) > 0 else 0.0)
        min_dist.append(np.min(other_vals) if len(other_vals) > 0 else 0.0)
        nearby_2km.append(np.sum(other_vals <= 2.0))
        nearby_5km.append(np.sum(other_vals <= 5.0))
        
    spatial_df = pd.DataFrame({
        'grid': grid_ids,
        'grid_adj_count': adj_count.loc[grid_ids].values,
        'grid_mean_dist': mean_dist,
        'grid_min_dist': min_dist,
        'grid_nearby_count_2km': nearby_2km,
        'grid_nearby_count_5km': nearby_5km
    })
    
    info_df = info_df.merge(spatial_df, on='grid', how='left')
    
    time_df['timestamp'] = pd.to_datetime(time_df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    time_df = time_df.reset_index().rename(columns={'index': 'timestamp_idx'})
    time_df['timestamp_idx'] = time_df['timestamp_idx'] + 1
    
    def melt_matrix(mat, val_name):
        melted = pd.melt(mat, id_vars=['timestamp'], var_name='grid', value_name=val_name)
        melted['grid'] = melted['grid'].astype(int)
        return melted
        
    occ_long = melt_matrix(occ_mat, 'occupancy')
    vol_long = melt_matrix(vol_mat, 'volume')
    dur_long = melt_matrix(dur_mat, 'duration')
    price_long = melt_matrix(price_mat, 'price')
     
    merged = occ_long.merge(vol_long, on=['timestamp', 'grid'])
    merged = merged.merge(dur_long, on=['timestamp', 'grid'])
    merged = merged.merge(price_long, on=['timestamp', 'grid'])
    
    merged = merged.rename(columns={'timestamp': 'timestamp_idx'})
    merged = merged.merge(time_df[['timestamp_idx', 'timestamp']], on='timestamp_idx')
    

    # Add scale to 5-min data for utilization
    merged = merged.merge(info_df[['grid', 'count']].rename(columns={'count': 'scale'}), on='grid')
    
    merged['utilization'] = merged['occupancy'] / merged['scale']
    merged['utilization'] = merged['utilization'].replace([np.inf, -np.inf], np.nan).fillna(0)
    merged['is_congested_5m'] = (merged['occupancy'] > 80).astype(int)
    
    merged = merged.sort_values(['grid', 'timestamp'])
    # Queue Length Proxy: Bounded +1 / -1 dynamic counter
    is_full_arr = (merged['utilization'] >= 1.0).values
    grids_arr = merged['grid'].values
    
    n_rows = len(grids_arr)
    queue_arr = np.zeros(n_rows, dtype=int)
    
    current_grid = -1
    current_q = 0
    
    # Fast numpy loop to calculate the stateful bounded queue
    for i in range(n_rows):
        # Reset queue if we switch to a new grid
        if grids_arr[i] != current_grid:
            current_grid = grids_arr[i]
            current_q = 0
        
        if is_full_arr[i]:
            current_q += 1
        else:
            if current_q > 0:
                current_q -= 1
                
        queue_arr[i] = current_q
        
    merged['queue_length_proxy'] = queue_arr

    # Resample to hourly by grouping
    merged.set_index('timestamp', inplace=True)
    hourly_merged = merged.groupby('grid').resample('1h').agg({
        'occupancy': 'mean',
        'volume': 'sum',
        'duration': 'mean',
        'price': 'mean',
        'queue_length_proxy': 'last',
        'is_congested_5m': 'mean'
    }).reset_index()
    
    hourly_merged = add_cyclical_features(hourly_merged, 'timestamp')
    hourly_merged['month'] = hourly_merged['timestamp'].dt.month
    

    info_cols = ['grid', 'count', 'fast_count', 'slow_count', 'area', 'lon', 'la', 'CBD', 
                'dynamic_pricing', 'grid_adj_count', 'grid_mean_dist', 
                'grid_min_dist', 'grid_nearby_count_2km', 'grid_nearby_count_5km']

    hourly_merged = hourly_merged.merge(info_df[info_cols], on='grid')
    hourly_merged = hourly_merged.rename(columns={'count': 'scale'})
    hourly_merged['poi'] = np.where(hourly_merged['CBD'] == 1, 1.0, 0.3)
    
    hourly_merged['occupancy_density'] = (hourly_merged['occupancy'] / hourly_merged['scale']).clip(upper=1.0)
    
    # utilisation metrics
    hourly_merged['charging_utilisation_rate'] = np.where(
        hourly_merged['scale'] > 0, 
        (hourly_merged['duration'] / hourly_merged['scale']).clip(upper=1.0), 
        0.0
    )
    # Estimate average power capacity per grid 
    hourly_merged['expected_load'] = hourly_merged['volume'] / hourly_merged['duration'].replace(0, np.nan)
    hourly_merged['expected_load'] = hourly_merged['expected_load'].fillna(0.0)
    
    
    # physical occupancy utilization (Total Plugged-In Time / Total Available Capacity)
    # For congestion and pricing, the physical occupancy utilization is simply the proportion of active chargers used in this hour.
    hourly_merged['utilization_rate_hr'] = np.where(hourly_merged['scale'] > 0, (hourly_merged['occupancy'] / hourly_merged['scale']).clip(upper=1.0), 0.0)
    
    total_urban_utilization = hourly_merged['occupancy'].sum() / (hourly_merged['scale'].sum() * (len(hourly_merged)/len(hourly_merged['grid'].unique())))
    #city wide avg
    hourly_merged['total_utilization_rate'] = total_urban_utilization
    
    #hourly_merged['queue_length_proxy'] = np.maximum(0.0, hourly_merged['occupancy'] - 0.8 * hourly_merged['scale']) * hourly_merged['duration']
    hourly_merged['is_weekend'] = (hourly_merged['dayofweek'] >= 5).astype(int)
    

    print("Engineering UrbanEV lags, rolling stats, and hardware ratios...")
    hourly_merged['fast_ratio'] = np.where(hourly_merged['scale'] > 0, hourly_merged['fast_count'] / hourly_merged['scale'], 0.0)
    hourly_merged['fast_ratio'] = hourly_merged['fast_ratio'].clip(0.0, 1.0)
    
    # Lags
    hourly_merged['occ_lag_1'] = hourly_merged.groupby('grid')['occupancy_density'].shift(1).fillna(0.0)
    hourly_merged['occ_lag_2'] = hourly_merged.groupby('grid')['occupancy_density'].shift(2).fillna(0.0)
    hourly_merged['occ_lag_3'] = hourly_merged.groupby('grid')['occupancy_density'].shift(3).fillna(0.0)
    hourly_merged['occ_lag_12'] = hourly_merged.groupby('grid')['occupancy_density'].shift(12).fillna(0.0)
    hourly_merged['occ_lag_24'] = hourly_merged.groupby('grid')['occupancy_density'].shift(24).fillna(0.0)
    
    hourly_merged['vol_lag_1'] = hourly_merged.groupby('grid')['volume'].shift(1).fillna(0.0)
    hourly_merged['vol_lag_2'] = hourly_merged.groupby('grid')['volume'].shift(2).fillna(0.0)
    hourly_merged['vol_lag_3'] = hourly_merged.groupby('grid')['volume'].shift(3).fillna(0.0)
    hourly_merged['vol_lag_12'] = hourly_merged.groupby('grid')['volume'].shift(12).fillna(0.0)
    hourly_merged['vol_lag_24'] = hourly_merged.groupby('grid')['volume'].shift(24).fillna(0.0)
    
    # Rolling Statistics (3-hour window)
    hourly_merged['occ_roll_mean_3h'] = hourly_merged.groupby('grid')['occupancy_density'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    hourly_merged['occ_roll_std_3h'] = hourly_merged.groupby('grid')['occupancy_density'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0.0)
    
    hourly_merged['vol_roll_mean_3h'] = hourly_merged.groupby('grid')['volume'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    hourly_merged['vol_roll_std_3h'] = hourly_merged.groupby('grid')['volume'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0.0)
    
    
    global_occ = hourly_merged.groupby('timestamp')['occupancy_density'].mean().rename('global_occ_mean')
    global_vol = hourly_merged.groupby('timestamp')['volume'].mean().rename('global_vol_mean')
    hourly_merged = hourly_merged.merge(global_occ, on='timestamp', how='left')
    hourly_merged = hourly_merged.merge(global_vol, on='timestamp', how='left')
    
    hourly_merged['global_occ_lag_1'] = hourly_merged.groupby('grid')['global_occ_mean'].shift(1).fillna(0.0)
    hourly_merged['global_vol_lag_1'] = hourly_merged.groupby('grid')['global_vol_mean'].shift(1).fillna(0.0)
    

    hourly_merged = hourly_merged.sort_values(by=['grid', 'timestamp'])
    
    # Create the 1-hour and 24-hour lags for charging utilisation rate
    hourly_merged['util_lag_1'] = hourly_merged.groupby('grid')['charging_utilisation_rate'].shift(1)
    hourly_merged['util_lag_24'] = hourly_merged.groupby('grid')['charging_utilisation_rate'].shift(24)
    
    
    # Fill the few resulting NaNs at the beginning of each grid's timeline with 0
    hourly_merged['util_lag_1'] = hourly_merged['util_lag_1'].fillna(0)
    hourly_merged['util_lag_24'] = hourly_merged['util_lag_24'].fillna(0)
    hourly_merged['is_congested'] = (hourly_merged['is_congested_5m'] > 0).astype(int)
    
    hourly_merged['charging_demand'] = hourly_merged['volume']
   
    hourly_merged['expected_load_lag_1'] = hourly_merged.groupby('grid')['expected_load'].shift(1).fillna(0.0)
    hourly_merged['expected_load_lag_12'] = hourly_merged.groupby('grid')['expected_load'].shift(12).fillna(0.0)
    hourly_merged['expected_load_lag_24'] = hourly_merged.groupby('grid')['expected_load'].shift(24).fillna(0.0)
    
    hourly_merged['queue_lag_1'] = hourly_merged.groupby('grid')['queue_length_proxy'].shift(1).fillna(0.0)
    hourly_merged['queue_lag_12'] = hourly_merged.groupby('grid')['queue_length_proxy'].shift(12).fillna(0.0)
    hourly_merged['queue_lag_24'] = hourly_merged.groupby('grid')['queue_length_proxy'].shift(24).fillna(0.0)



#    hourly_merged['energy_cost_per_kwh'] = hourly_merged['hour'].apply(get_tod_cost)
    hourly_merged['baseline_revenue'] = hourly_merged['volume'] * 15.0
 #   hourly_merged['baseline_energy_cost'] = hourly_merged['volume'] * hourly_merged['energy_cost_per_kwh']
  #  hourly_merged['baseline_profit'] = hourly_merged['baseline_revenue'] - hourly_merged['baseline_energy_cost']
    
    urban_ev_csv = os.path.join(SUBMISSION_DIR, "urban_ev_intervals.csv")
    hourly_merged.to_csv(urban_ev_csv, index=False)
    print(f"Saved HOURLY UrbanEV interval data to {urban_ev_csv} (Rows: {len(hourly_merged)})")
    return hourly_merged



if __name__ == "__main__":
    acn_sessions = preprocess_acn()
    aggregate_acn_to_intervals(acn_sessions)
    preprocess_urbanev()
    print("\n--- PREPROCESSING COMPLETED SUCCESSFULLY ---")
