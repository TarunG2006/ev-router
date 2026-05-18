"""
src/preprocess_eved.py

Preprocess eVED (Extended Vehicle Energy Dataset) weekly CSVs
into ev_telemetry.csv for RF model retraining.

Citation: Zhang et al., "Extended Vehicle Energy Dataset (eVED)",
          IEEE VTC 2025 / IEEE ITS 2020.

Pipeline:
  eVED weekly CSVs
    → filter pure EV vehicles (Nissan Leaf, VehId in EV_VEHIDS)
    → compute per-row Wh from HV voltage × current × dt
    → compute per-row distance from GPS (haversine)
    → aggregate into ~0.3 km segments
    → compute wh_per_km per segment (net, signed current → regen aware)
    → filter 5 < wh_per_km < 500, speed > 2 km/h
    → physical sanity check: uphill > flat > downhill
    → save as data/raw/ev_telemetry.csv

Usage:
    cd C:\\Users\\Tarun\\Desktop\\ev_router
    venv\\Scripts\\activate
    python -m src.preprocess_eved
"""

import os
import glob
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ── Paths ─────────────────────────────────────────────────────────────────────
EVED_DIR    = r"C:\Users\Tarun\Downloads\eVED"      # folder with weekly CSVs
OUTPUT_PATH = r"C:\Users\Tarun\Desktop\ev_router\data\raw\ev_telemetry.csv"

# ── EV Vehicle IDs ────────────────────────────────────────────────────────────
# eVED pure EVs: 2013 Nissan Leaf (24 kWh).
# Script prints all unique VehIds from first file — update if different.
EV_VEHIDS = [10, 11, 12]

# ── Segment & filter config ───────────────────────────────────────────────────
SEGMENT_KM    = 0.1   # target segment length in km
MIN_SPEED_KMH = 2.0    # drop stationary rows
MIN_WH_PER_KM = 5.0    # drop near-zero / regen-dominated segments
MAX_WH_PER_KM = 500.0  # drop sensor outliers

# Gradient thresholds for sanity check
GRAD_UPHILL   =  0.01   # > 2% grade = uphill
GRAD_DOWNHILL = -0.01   # < -2% grade = downhill

# ── Key columns needed ────────────────────────────────────────────────────────
KEY_COLS = [
    'Timestamp(ms)',
    'HV Battery Voltage[V]',
    'HV Battery Current[A]',
    'Vehicle Speed[km/h]',
    'Latitude[deg]',
    'Longitude[deg]',
    'Gradient',
    'Class of Speed Limit',
]


# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two GPS points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2
         + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def process_trip(trip_df: pd.DataFrame) -> list[dict]:
    """
    Convert one trip's per-second OBD rows into ~SEGMENT_KM segments.

    Energy per row:
        Wh = V × I × dt / 3600
        Current is SIGNED: positive = discharge, negative = regen braking.
        Net segment Wh can be slightly negative on steep downhills —
        filtered out by MIN_WH_PER_KM > 0, keeping edge costs positive for A*.

    Returns list of dicts with keys:
        wh_per_km, gradient, speed_kmh, road_type
    """
    trip_df = trip_df.sort_values('Timestamp(ms)').reset_index(drop=True)

    # Drop stationary rows
    trip_df = trip_df[trip_df['Vehicle Speed[km/h]'] > MIN_SPEED_KMH]
    if len(trip_df) < 5:
        return []

    # ── Per-row dt (seconds) from timestamp diff ──────────────────────────────
    ts_ms = trip_df['Timestamp(ms)'].values.astype(float)
    dt = np.diff(ts_ms, prepend=ts_ms[0]) / 1000.0   # seconds; row 0 gets dt=0
    dt[0] = 1.0                                        # assume 1s for first row
    dt = np.clip(dt, 0.1, 10.0)                        # cap outliers (gaps, resets)

    # ── Per-row Wh (signed: positive = energy consumed) ──────────────────────
    V = trip_df['HV Battery Voltage[V]'].values
    I = trip_df['HV Battery Current[A]'].values        # signed
    wh_instant = V * I * dt / 3600.0                  # Wh per row

    # ── Per-row distance from GPS ─────────────────────────────────────────────
    lats = trip_df['Latitude[deg]'].values
    lons = trip_df['Longitude[deg]'].values
    dist_km = np.zeros(len(trip_df))
    for i in range(1, len(trip_df)):
        dist_km[i] = haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])

    gradients  = trip_df['Gradient'].values
    speeds     = trip_df['Vehicle Speed[km/h]'].values
    road_types = trip_df['Class of Speed Limit'].values

    # ── Aggregate into segments ───────────────────────────────────────────────
    segments  = []
    seg_wh    = 0.0
    seg_dist  = 0.0
    seg_grad  = []
    seg_spd   = []
    seg_road  = []

    for i in range(len(trip_df)):
        seg_wh   += wh_instant[i]
        seg_dist += dist_km[i]
        seg_grad.append(gradients[i])
        seg_spd.append(speeds[i])
        seg_road.append(road_types[i])

        if seg_dist >= SEGMENT_KM:
            if seg_dist > 0:
                wh_per_km = seg_wh / seg_dist
                if MIN_WH_PER_KM <= wh_per_km <= MAX_WH_PER_KM:
                    # road_type = mode of Class of Speed Limit in segment
                    road_mode = max(set(seg_road), key=seg_road.count)
                    segments.append({
                        'wh_per_km': round(wh_per_km, 4),
                        'gradient' : round(float(np.mean(seg_grad)), 6),
                        'speed_kmh': round(float(np.mean(seg_spd)),  2),
                        'road_type': road_mode,
                    })
            # Reset accumulator
            seg_wh   = 0.0
            seg_dist = 0.0
            seg_grad = []
            seg_spd  = []
            seg_road = []

    return segments


# ─────────────────────────────────────────────────────────────────────────────
def main():
    csv_files = sorted(glob.glob(os.path.join(EVED_DIR, "eVED_*_week.csv")))
    print(f"Found {len(csv_files)} weekly CSV files in: {EVED_DIR}")

    if not csv_files:
        print("\nERROR: No CSV files found. Check EVED_DIR path.")
        return

    # ── Peek at vehicle IDs so we can verify EV_VEHIDS ───────────────────────
    print(f"\nSampling VehIds from {os.path.basename(csv_files[0])}...")
    sample = pd.read_csv(csv_files[0], usecols=['VehId'], nrows=50_000)
    print(f"  All VehIds in sample : {sorted(sample['VehId'].unique())}")
    print(f"  Filtering for EV IDs : {EV_VEHIDS}")
    print("  ⚠  If Nissan Leaf IDs look different above, edit EV_VEHIDS in script.\n")

    all_segments = []

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        print(f"Processing {fname} ...", end="  ", flush=True)

        df = pd.read_csv(csv_path, usecols=KEY_COLS + ['VehId', 'Trip', 'DayNum'])

        # Filter EV vehicles
        ev_df = df[df['VehId'].isin(EV_VEHIDS)].copy()
        if ev_df.empty:
            print("no EV rows — skipping")
            continue

        # Drop rows with NaN in any key column
        ev_df = ev_df.dropna(subset=KEY_COLS)

        # Process each (vehicle, trip, day) independently
        file_segments = []
        groups = ev_df.groupby(['VehId', 'Trip', 'DayNum'])
        for _, trip_df in groups:
            file_segments.extend(process_trip(trip_df))

        all_segments.extend(file_segments)
        print(f"{len(file_segments):,} segments  "
              f"(EV rows: {len(ev_df):,})")

    if not all_segments:
        print("\nERROR: 0 segments extracted.")
        print("Likely cause: EV_VEHIDS don't match actual IDs in this dataset.")
        print("Fix: update EV_VEHIDS list at top of script and re-run.")
        return

    result_df = pd.DataFrame(all_segments)

    # ── Physical sanity check ─────────────────────────────────────────────────
    print("\n── Physical sanity check ────────────────────────────────")
    uphill   = result_df[result_df['gradient'] >  GRAD_UPHILL  ]['wh_per_km'].mean()
    flat     = result_df[result_df['gradient'].abs() <= abs(GRAD_UPHILL)]['wh_per_km'].mean()
    downhill = result_df[result_df['gradient'] <  GRAD_DOWNHILL]['wh_per_km'].mean()

    print(f"  Uphill   (gradient > {GRAD_UPHILL:+.0%}) mean Wh/km : {uphill:.1f}")
    print(f"  Flat     (|gradient| ≤ {abs(GRAD_UPHILL):.0%}) mean Wh/km : {flat:.1f}")
    print(f"  Downhill (gradient < {GRAD_DOWNHILL:+.0%}) mean Wh/km : {downhill:.1f}")

    if uphill > flat > downhill:
        print("  ✅ Physically correct: uphill > flat > downhill")
    else:
        print("  ⚠  WARNING: Physics check failed.")
        print("     Possible cause: HV Battery Current sign convention inverted.")
        print("     Try negating current: change  V * I  →  V * (-I)  in process_trip()")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\n── Dataset summary ──────────────────────────────────────")
    print(f"  Total segments  : {len(result_df):,}")
    print(f"  wh_per_km stats :\n{result_df['wh_per_km'].describe().round(2)}")
    print(f"  gradient  stats :\n{result_df['gradient'].describe().round(4)}")
    print(f"  road_type vals  : {sorted(result_df['road_type'].unique())}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(result_df):,} rows → {OUTPUT_PATH}")
    print(f"   Columns: {list(result_df.columns)}")
    print("\nNext step: python -m src.train_model  (retrain RF on this file)")


if __name__ == "__main__":
    main()