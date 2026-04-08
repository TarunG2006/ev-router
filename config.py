# config.py — all constants live here

# Paths
RAW_GRAPH_PATH        = "data/raw/jaipur_graph.graphml"
PROCESSED_GRAPH_PATH  = "data/processed/jaipur_graph_with_attrs.graphml"
ELEVATION_PATH        = "data/raw/jaipur_elevation.tif"
EV_DATA_PATH          = "data/raw/ev_telemetry.csv"
KAGGLE_EV_DATA_PATH   = "data/raw/kaggle_ev_data.csv"  # Real Kaggle dataset
MODEL_PATH            = "models/battery_rf_model.pkl"
BENCHMARK_RESULTS     = "data/processed/benchmark_results.csv"

# Jaipur geographic bounds (west, south, east, north)
JAIPUR_PLACE_NAME = "Jaipur, Rajasthan, India"
JAIPUR_BBOX       = (75.70, 26.75, 75.95, 27.10)
JAIPUR_CENTER_LAT = 26.9124
JAIPUR_CENTER_LON = 75.7873

# EV car parameters (for Kaggle car dataset)
BATTERY_CAPACITY_WH          = 40000.0  # 40 kWh typical EV car
BATTERY_LOW_WH               = 2000.0   # 5% reserve threshold
BATTERY_DEVIATION_THRESHOLD  = 0.15    # 15% deviation triggers replan

# Road type encoding
ROAD_TYPE_MAP = {
    'motorway': 2, 'trunk': 2, 'primary': 2,
    'secondary': 1, 'tertiary': 1,
    'residential': 0, 'unclassified': 0,
    'service': 0, 'living_street': 0
}

# Simulation
N_DELIVERIES  = 500
RANDOM_SEED   = 42
MAX_LPA_ITERS = 500_000

# ML model - predicts Wh per km RATE (not total energy)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = 12
RF_MIN_SAMPLES  = 5
ML_FEATURES     = ['slope', 'speed_kmh', 'road_type']  # No distance - we predict RATE
ML_TARGET       = 'wh_per_km'  # Predict rate, multiply by distance later
