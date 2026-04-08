# src/battery_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────
# PART A: Load Real Kaggle EV Dataset
# ─────────────────────────────────────────

# Kaggle dataset: "Electric Vehicle Charging Sessions" or similar
# Download from: https://www.kaggle.com/datasets/
# Expected columns vary by dataset - we handle multiple formats

KAGGLE_DATA_PATH = "data/raw/kaggle_ev_data.csv"


def load_kaggle_ev_data(filepath=None):
    """
    Load and preprocess real Kaggle EV telemetry data.
    
    Supports multiple Kaggle dataset formats:
    1. EV Charging Sessions (michaelbryantds/electric-vehicle-charging-dataset)
    2. EV Trip Energy (custom trip-based datasets)
    3. VED Dataset (Vehicle Energy Dataset)
    
    Returns DataFrame with standardized columns:
    - slope, distance_km, speed_kmh, road_type, energy_wh
    """
    filepath = filepath or KAGGLE_DATA_PATH
    
    if not os.path.exists(filepath):
        print(f"WARNING: Kaggle dataset not found at {filepath}")
        print("   Download from Kaggle and place in data/raw/")
        print("   Falling back to synthetic data...")
        return None
    
    print(f"Loading Kaggle EV dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Raw dataset: {len(df)} rows, columns: {list(df.columns)}")
    
    # Detect dataset format and preprocess accordingly
    df_processed = _preprocess_kaggle_data(df)
    
    if df_processed is None or len(df_processed) < 1000:
        print("WARNING: Insufficient data after preprocessing, using synthetic")
        return None
    
    return df_processed


def _preprocess_kaggle_data(df):
    """
    Preprocess Kaggle data into standardized format.
    Handles multiple common EV dataset schemas.
    """
    columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = columns
    
    result = pd.DataFrame()
    
    # ─── Format 0: ZIYA EV Energy Consumption Dataset (exact match) ───
    if 'speed_kmh' in columns and 'slope_%' in columns and 'energy_consumption_kwh' in columns:
        print("Detected: ZIYA EV Energy Consumption Dataset format")
        
        result['speed_kmh'] = pd.to_numeric(df['speed_kmh'], errors='coerce')
        result['slope'] = pd.to_numeric(df['slope_%'], errors='coerce')  # Already in %/degrees
        result['distance_km'] = pd.to_numeric(df['distance_travelled_km'], errors='coerce')
        
        # Convert kWh to Wh
        energy_kwh = pd.to_numeric(df['energy_consumption_kwh'], errors='coerce')
        result['energy_wh'] = energy_kwh * 1000
        
        # Calculate Wh per km (the RATE) - this is what RF will predict
        result['wh_per_km'] = result['energy_wh'] / result['distance_km'].clip(lower=0.1)
        
        # Road type - already numeric in this dataset (0, 1, 2)
        if 'road_type' in columns:
            road_vals = df['road_type']
            if road_vals.dtype == 'object':  # String values
                road_map = {
                    'highway': 2, 'Highway': 2,
                    'city': 1, 'City': 1, 'urban': 1, 'Urban': 1,
                    'rural': 0, 'Rural': 0, 'residential': 0, 'Residential': 0
                }
                result['road_type'] = road_vals.map(road_map).fillna(0).astype(int)
            else:  # Already numeric
                result['road_type'] = pd.to_numeric(road_vals, errors='coerce').fillna(0).astype(int)
        else:
            result['road_type'] = np.where(result['speed_kmh'] > 60, 2,
                                  np.where(result['speed_kmh'] > 35, 1, 0))
        
        print(f"Mapped {len(result)} records from ZIYA dataset")
    
    # ─── Format 1: Trip-based data with distance/energy ───
    elif 'distance' in columns or 'trip_distance' in columns or 'distance_km' in columns:
        dist_col = next((c for c in columns if 'distance' in c), None)
        energy_col = next((c for c in columns if 'energy' in c or 'kwh' in c or 'wh' in c), None)
        speed_col = next((c for c in columns if 'speed' in c or 'velocity' in c), None)
        
        if dist_col and energy_col:
            result['distance_km'] = pd.to_numeric(df[dist_col], errors='coerce')
            
            # Convert energy to Wh if in kWh
            energy_vals = pd.to_numeric(df[energy_col], errors='coerce')
            if energy_vals.median() < 10:  # Likely in kWh
                result['energy_wh'] = energy_vals * 1000
            else:
                result['energy_wh'] = energy_vals
            
            # Speed - use if available, else estimate
            if speed_col:
                result['speed_kmh'] = pd.to_numeric(df[speed_col], errors='coerce')
            else:
                # Estimate speed from distance/time if available
                time_col = next((c for c in columns if 'time' in c or 'duration' in c), None)
                if time_col:
                    duration_hrs = pd.to_numeric(df[time_col], errors='coerce') / 60
                    result['speed_kmh'] = result['distance_km'] / duration_hrs.clip(lower=0.01)
                else:
                    result['speed_kmh'] = np.random.uniform(20, 50, len(df))
            
            # Slope - use if available, else generate realistic distribution
            slope_col = next((c for c in columns if 'slope' in c or 'grade' in c or 'elevation' in c), None)
            if slope_col:
                result['slope'] = pd.to_numeric(df[slope_col], errors='coerce')
            else:
                # Indian cities have gentle slopes, simulate -4 to +4 degrees
                result['slope'] = np.random.normal(0, 1.5, len(df)).clip(-6, 6)
            
            # Road type - use if available, else estimate from speed
            road_col = next((c for c in columns if 'road' in c or 'highway' in c), None)
            if road_col:
                result['road_type'] = df[road_col].map({
                    'highway': 2, 'primary': 2, 'motorway': 2,
                    'secondary': 1, 'arterial': 1,
                    'residential': 0, 'local': 0, 'urban': 0
                }).fillna(0).astype(int)
            else:
                # Estimate from speed: high speed = highway
                result['road_type'] = np.where(result['speed_kmh'] > 60, 2,
                                      np.where(result['speed_kmh'] > 35, 1, 0))
    
    # ─── Format 2: Charging session data (derive from energy) ───
    elif 'energy_kwh' in columns or 'kwh_charged' in columns:
        energy_col = next((c for c in columns if 'kwh' in c or 'energy' in c), None)
        
        if energy_col:
            # Generate synthetic trip features from charging data
            n = len(df)
            energy_kwh = pd.to_numeric(df[energy_col], errors='coerce')
            
            # Assume average 15 Wh/km, back-calculate distance
            avg_consumption = 0.015  # kWh per km
            result['distance_km'] = (energy_kwh / avg_consumption * 
                                     np.random.uniform(0.8, 1.2, n)).clip(0.1, 50)
            result['energy_wh'] = energy_kwh * 1000
            result['speed_kmh'] = np.random.uniform(15, 55, n)
            result['slope'] = np.random.normal(0, 1.5, n).clip(-6, 6)
            result['road_type'] = np.random.choice([0, 1, 2], n, p=[0.55, 0.30, 0.15])
    
    # ─── Format 3: VED-style detailed telemetry ───
    elif 'vehicle_speed' in columns or 'obd_speed' in columns:
        speed_col = next((c for c in columns if 'speed' in c), None)
        result['speed_kmh'] = pd.to_numeric(df[speed_col], errors='coerce')
        
        # Look for energy/power columns
        power_col = next((c for c in columns if 'power' in c or 'energy' in c), None)
        if power_col:
            result['energy_wh'] = pd.to_numeric(df[power_col], errors='coerce').abs()
        
        # Generate other features
        n = len(df)
        result['distance_km'] = np.random.uniform(0.1, 2.0, n)
        result['slope'] = np.random.normal(0, 1.5, n).clip(-6, 6)
        result['road_type'] = np.random.choice([0, 1, 2], n, p=[0.55, 0.30, 0.15])
    
    if result.empty:
        print("Could not parse dataset format")
        return None
    
    # Clean up
    result = result.dropna()
    result = result[(result['energy_wh'] > 0) & (result['energy_wh'] < 50000)]  # Up to 50 kWh
    result = result[(result['distance_km'] > 0.01) & (result['distance_km'] < 500)]  # Up to 500 km
    result = result[(result['speed_kmh'] > 5) & (result['speed_kmh'] < 200)]  # Up to 200 km/h
    
    # Round for cleaner output
    result['slope'] = result['slope'].round(3)
    result['distance_km'] = result['distance_km'].round(4)
    result['speed_kmh'] = result['speed_kmh'].round(2)
    result['road_type'] = result['road_type'].astype(int)
    result['energy_wh'] = result['energy_wh'].round(4)
    
    # Compute wh_per_km if not already computed (for ML model)
    if 'wh_per_km' not in result.columns:
        result['wh_per_km'] = (result['energy_wh'] / result['distance_km'].clip(lower=0.1)).round(2)
    
    print(f"Preprocessed Kaggle data: {len(result)} valid records")
    return result


def prepare_training_data(use_kaggle=True, synthetic_fallback=True, save=True):
    """
    Prepare training data from Kaggle dataset with synthetic fallback.
    
    Args:
        use_kaggle: Try to load Kaggle dataset first
        synthetic_fallback: Generate synthetic data if Kaggle unavailable
        save: Save processed data to EV_DATA_PATH
    
    Returns:
        DataFrame with training data
    """
    df = None
    data_source = "none"
    
    if use_kaggle:
        df = load_kaggle_ev_data()
        if df is not None:
            data_source = "kaggle"
    
    if df is None and synthetic_fallback:
        print("Generating synthetic EV data as fallback...")
        df = generate_synthetic_ev_data(n=80000, save=False)
        data_source = "synthetic"
    
    if df is None:
        raise ValueError("No training data available!")
    
    if save:
        df.to_csv(config.EV_DATA_PATH, index=False)
        print(f"Saved {len(df)} records ({data_source}) to {config.EV_DATA_PATH}")
    
    return df, data_source


# ─────────────────────────────────────────
# PART A-alt: Generate synthetic EV telemetry data (fallback)
# ─────────────────────────────────────────

def generate_synthetic_ev_data(n=80000, save=True):
    """
    Create realistic EV energy consumption data.
    Features: slope (deg), distance_km, speed_kmh, road_type
    Target:   energy_wh (watt-hours consumed)
    """
    print(f"Generating {n} synthetic EV telemetry records...")
    np.random.seed(config.RANDOM_SEED)

    # Feature distributions matching Jaipur roads
    slope       = np.random.uniform(-6, 6, n)           # degrees
    distance_km = np.random.uniform(0.05, 2.5, n)       # km per road segment
    speed_kmh   = np.random.uniform(10, 55, n)          # km/h
    road_type   = np.random.choice([0, 1, 2], n,
                                    p=[0.55, 0.30, 0.15])

    # Physics formula for energy (Wh)
    # Base: 15 Wh/km for level road at 30 km/h
    # Slope penalty: +2.5 Wh/km per degree uphill
    # Speed penalty: quadratic (aerodynamic drag)
    # Road type discount: smoother roads = less rolling resistance
    road_efficiency = np.where(road_type == 2, 0.90,
                     np.where(road_type == 1, 0.95, 1.00))

    energy_per_km = (
        15.0
        + slope * 2.8                          # slope energy
        + (speed_kmh - 30.0)**2 * 0.008        # speed (drag)
    ) * road_efficiency

    energy_wh = energy_per_km * distance_km
    energy_wh = energy_wh + np.random.normal(0, 0.8, n)   # sensor noise
    energy_wh = np.clip(energy_wh, 0.1, 80.0)              # physical limits

    df = pd.DataFrame({
        'slope':       np.round(slope, 3),
        'distance_km': np.round(distance_km, 4),
        'speed_kmh':   np.round(speed_kmh, 2),
        'road_type':   road_type.astype(int),
        'energy_wh':   np.round(energy_wh, 4),
        'wh_per_km':   np.round(energy_wh / distance_km, 2)  # Rate for ML model
    })

    if save:
        df.to_csv(config.EV_DATA_PATH, index=False)
        print(f"Saved {len(df)} records to {config.EV_DATA_PATH}")

    return df


# ─────────────────────────────────────────
# PART B: Train the Random Forest model
# ─────────────────────────────────────────

def train_battery_model():
    print("Loading EV telemetry data...")
    df = pd.read_csv(config.EV_DATA_PATH)
    print(f"Dataset: {len(df)} rows, columns: {list(df.columns)}")

    # Compute wh_per_km if not present
    if 'wh_per_km' not in df.columns and 'energy_wh' in df.columns and 'distance_km' in df.columns:
        df['wh_per_km'] = df['energy_wh'] / df['distance_km']
        print("Computed wh_per_km from energy_wh / distance_km")

    # Validate columns
    for col in config.ML_FEATURES + [config.ML_TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean - filter reasonable Wh/km rates (50-1000 Wh/km for EV)
    # Typical EV: 150-250 Wh/km, allow wider range for extreme conditions
    df = df[(df[config.ML_TARGET] > 50) & (df[config.ML_TARGET] < 1000)]
    df = df.dropna(subset=config.ML_FEATURES + [config.ML_TARGET])
    print(f"After cleaning: {len(df)} rows")

    X = df[config.ML_FEATURES].values
    y = df[config.ML_TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators = config.RF_N_ESTIMATORS,
        max_depth    = config.RF_MAX_DEPTH,
        min_samples_leaf = config.RF_MIN_SAMPLES,
        n_jobs       = -1,
        random_state = config.RANDOM_SEED
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"MAE : {mae:.4f} Wh")
    print(f"R²  : {r2:.4f}")

    # Feature importance
    importances = dict(zip(config.ML_FEATURES, model.feature_importances_))
    print(f"\nFeature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:15s}: {imp:.4f}")

    # Save
    joblib.dump(model, config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")
    return model


# ─────────────────────────────────────────
# PART C: Load model and predict
# ─────────────────────────────────────────

def load_battery_model():
    return joblib.load(config.MODEL_PATH)


def predict_edge_cost(model, slope, distance_km, speed_kmh, road_type):
    """
    Predict energy consumption (Wh) for one edge.
    Model predicts Wh/km rate, we multiply by distance.
    """
    # Features: slope, speed, road_type (no distance - model predicts rate)
    features = np.array([[slope, speed_kmh, road_type]])
    rate_wh_per_km = float(model.predict(features)[0])
    return rate_wh_per_km * distance_km


if __name__ == "__main__":
    # Try Kaggle data first, fall back to synthetic
    df, source = prepare_training_data(use_kaggle=True, synthetic_fallback=True)
    print(f"\nData source: {source.upper()}")
    train_battery_model()