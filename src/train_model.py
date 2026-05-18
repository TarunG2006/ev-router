"""
src/train_model.py

Retrain the Random Forest battery model on blended data:
  - eVED real OBD telemetry (speed / road_type / consumption patterns)
  - Synthetic slope-augmentation (Michigan is flat; RF needs slope examples)

Why blend?
  eVED gives realistic Wh/km vs speed relationships from real Nissan Leaf OBD.
  But Michigan gradient range is only -0.01 to +0.005 (-0.6% to +0.3% grade).
  Without synthetic slope variation the RF can't learn uphill/downhill effects.
  Solution: blend real data (dominant) with targeted synthetic slope rows.

Units alignment:
  eVED   Gradient  = decimal rise/run (0.004 = 0.4% grade)
  config ML_FEATURES uses 'slope' in degrees
  Conversion: slope_deg = gradient * (180/pi) ≈ gradient * 57.296

Citation: Zhang et al., "Extended Vehicle Energy Dataset (eVED)", IEEE VTC 2025.

Usage:
    cd C:\\Users\\Tarun\\Desktop\\ev_router
    venv\\Scripts\\activate
    python -m src.train_model
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Config ────────────────────────────────────────────────────────────────────
EVED_PATH   = config.EV_DATA_PATH          # data/raw/ev_telemetry.csv
MODEL_PATH  = config.MODEL_PATH            # models/battery_rf_model.pkl

# Synthetic augmentation: slope-focused rows to supplement flat Michigan data
N_SYNTHETIC = 6000     # small — real data stays dominant
SYNTH_SEED  = 42

# eVED mean flat Wh/km (calibration anchor for synthetic slope rows)
# Will be computed from real data; this is just a fallback
EVED_FLAT_WH_KM = 107.5   # updated dynamically below


# ─────────────────────────────────────────────────────────────────────────────
def load_eved_data(path: str) -> pd.DataFrame:
    """
    Load eVED preprocessed CSV and align to config.ML_FEATURES schema.

    eVED columns : wh_per_km, gradient, speed_kmh, road_type
    Target schema: slope (deg), speed_kmh, road_type  →  wh_per_km
    """
    print(f"Loading eVED data from {path} ...")
    df = pd.read_csv(path)
    print(f"  Raw rows : {len(df)}")
    print(f"  Columns  : {list(df.columns)}")

    # Convert gradient (decimal) → slope (degrees)
    # arctan(gradient) in degrees; for small angles ≈ gradient × 57.296
    df['slope'] = np.degrees(np.arctan(df['gradient']))

    # Drop intermediate column
    df = df.drop(columns=['gradient'])

    # road_type -1 in eVED means unmapped — treat as residential (0)
    df['road_type'] = df['road_type'].replace(-1, 0).astype(int)

    # Keep only needed columns
    df = df[['slope', 'speed_kmh', 'road_type', 'wh_per_km']].copy()

    # Filter
    df = df[
        (df['wh_per_km'] > 5) & (df['wh_per_km'] < 500) &
        (df['speed_kmh'] > 2)
    ].dropna()

    print(f"  After cleaning : {len(df)} rows")
    print(f"  slope range    : {df['slope'].min():.3f}° to {df['slope'].max():.3f}°")
    print(f"  wh_per_km mean : {df['wh_per_km'].mean():.1f}")
    return df


def generate_slope_augmentation(eved_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Generate synthetic rows covering slope variation.

    Anchored to eVED's real flat consumption mean so the two datasets
    are on the same energy scale.  Only adds slope diversity — speed
    and road_type distributions mirror eVED.

    Physics (Tata Nexon class, scaled from Leaf):
        wh_per_km = flat_base + slope_effect + speed_drag
        slope_effect = slope_deg × k_slope   (uphill costs, downhill saves)
        speed_drag   = (speed - ref_speed)² × k_drag
    """
    np.random.seed(SYNTH_SEED)

    # Anchor flat base to real data (flat segments ≤ 0.1°)
    flat_mask = df_flat = eved_df[eved_df['slope'].abs() <= 0.1]
    flat_base = flat_mask['wh_per_km'].mean() if len(flat_mask) > 10 else EVED_FLAT_WH_KM
    ref_speed = eved_df['speed_kmh'].mean()

    print(f"\n  Synthetic augmentation anchor:")
    print(f"    flat_base  = {flat_base:.1f} Wh/km  (from eVED flat segments)")
    print(f"    ref_speed  = {ref_speed:.1f} km/h")

    # Physics constants calibrated to Nexon-class (40 kWh / ~160 km range)
    k_slope = 15.0    # Wh/km per degree of slope
    k_drag  = 0.025   # Wh/km per (km/h)² deviation from ref_speed

    # Sample slope from wider range than Michigan (Jaipur has gentle hills)
    slope     = np.random.uniform(-5.0, 5.0, n)
    speed_kmh = np.random.uniform(10, 60, n)
    road_type = np.random.choice([0, 1, 2], n, p=[0.55, 0.30, 0.15])

    wh_per_km = (
        flat_base
        + slope * k_slope
        + (speed_kmh - ref_speed) ** 2 * k_drag
        + np.random.normal(0, 5.0, n)       # realistic noise
    )

    df_synth = pd.DataFrame({
        'slope'     : np.round(slope, 3),
        'speed_kmh' : np.round(speed_kmh, 2),
        'road_type' : road_type.astype(int),
        'wh_per_km' : np.round(wh_per_km, 2),
    })

    # Apply same filter as real data
    df_synth = df_synth[
        (df_synth['wh_per_km'] > 5) & (df_synth['wh_per_km'] < 500)
    ]
    print(f"    synthetic rows generated : {len(df_synth)}")
    return df_synth


# ─────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame) -> RandomForestRegressor:
    features = config.ML_FEATURES   # ['slope', 'speed_kmh', 'road_type']
    target   = config.ML_TARGET     # 'wh_per_km'

    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}  (have: {list(df.columns)})")

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED
    )
    print(f"\nTraining RF on {len(X_train)} rows, testing on {len(X_test)} ...")

    model = RandomForestRegressor(
        n_estimators     = config.RF_N_ESTIMATORS,
        max_depth        = config.RF_MAX_DEPTH,
        min_samples_leaf = config.RF_MIN_SAMPLES,
        n_jobs           = -1,
        random_state     = config.RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"  MAE : {mae:.2f} Wh/km")
    print(f"  R²  : {r2:.4f}")

    importances = dict(zip(features, model.feature_importances_))
    print(f"\n  Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat:15s}: {imp:.4f}")

    return model, mae, r2


def sanity_check(model):
    """Spot-check predictions are physically correct."""
    features = config.ML_FEATURES
    print(f"\n── Physical spot-check (model predictions) ──")

    cases = [
        ("Steep uphill   (+5°, 30 km/h, residential)", [5.0,  30.0, 0]),
        ("Flat           ( 0°, 30 km/h, residential)", [0.0,  30.0, 0]),
        ("Steep downhill (-5°, 30 km/h, residential)", [-5.0, 30.0, 0]),
        ("Highway flat   ( 0°, 80 km/h, highway)",     [0.0,  80.0, 2]),
    ]

    preds = []
    for label, vals in cases:
        X = np.array([vals])
        pred = float(model.predict(X)[0])
        preds.append(pred)
        print(f"  {label} → {pred:.1f} Wh/km")

    uphill_ok   = preds[0] > preds[1]
    downhill_ok = preds[1] > preds[2]
    speed_ok    = preds[3] > preds[1]

    print()
    print(f"  Uphill > Flat     : {'✅' if uphill_ok   else '❌'}")
    print(f"  Flat > Downhill   : {'✅' if downhill_ok else '❌'}")
    print(f"  Highway > Flat    : {'✅' if speed_ok    else '❌'}")

    if uphill_ok and downhill_ok and speed_ok:
        print(f"\n  ✅ All physics checks passed")
    else:
        print(f"\n  ⚠  Some checks failed — review blending ratio or k_slope constant")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load real eVED data
    eved_df = load_eved_data(EVED_PATH)

    # 2. Generate slope augmentation anchored to eVED mean
    print(f"\nGenerating {N_SYNTHETIC} synthetic slope-augmentation rows ...")
    synth_df = generate_slope_augmentation(eved_df, N_SYNTHETIC)

    # 3. Blend: real data + synthetic
    combined = pd.concat([eved_df, synth_df], ignore_index=True).sample(
        frac=1, random_state=config.RANDOM_SEED
    )
    print(f"\n── Blended training set ──────────────────────────────")
    print(f"  eVED real rows  : {len(eved_df):,}")
    print(f"  Synthetic rows  : {len(synth_df):,}")
    print(f"  Total           : {len(combined):,}")
    print(f"  wh_per_km mean  : {combined['wh_per_km'].mean():.1f}")
    print(f"  wh_per_km std   : {combined['wh_per_km'].std():.1f}")

    # 4. Save blended dataset as the canonical EV_DATA_PATH
    combined.to_csv(EVED_PATH, index=False)
    print(f"\n  Saved blended dataset → {EVED_PATH}")

    # 5. Train
    model, mae, r2 = train(combined)

    # 6. Physical sanity check
    sanity_check(model)

    # 7. Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")
    print(f"\nNext step: python -m src.graph_builder  (rebuild graph with new RF)")


if __name__ == "__main__":
    main()