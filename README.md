# 🚗⚡ EV Delivery Router — Jaipur

**Battery-Aware LPA\* for Real-Time EV Delivery Routing under Dynamic Road Closures**

> IIT Guwahati · SDE Project · 2nd Year

---

## What This Project Does

Plans energy-optimal delivery routes for EVs across Jaipur's real road network.
When a road closes mid-delivery, the router replans instantly using LPA\* —
without restarting from scratch — while jointly ensuring the vehicle has enough
battery to reach the destination.

**Live demo:** `streamlit run app.py`

---

## Results (500-trial benchmark on Jaipur OSM network)

| Method | Success under closures | Avg replan time |
|---|---|---|
| Dijkstra (static) | **0.0%** ❌ | — |
| Full A\* replan | 94.6% | 193 ms |
| LPA\* incremental | 94.6% | 192 ms |
| **Battery-Aware LPA\*** | **99.0%** ✅ | 192 ms |

**Headline:** Battery-Aware LPA\* achieves 99% delivery success vs 94.6% for
standard replanning and 0% for Dijkstra, by jointly optimising route feasibility
and battery state in a single search.

**On replan speed:** LPA\* and A\* are equivalent in single-closure scenarios
(fresh LPA\* from current node = fresh A\*). LPA\*'s incremental advantage
accumulates across multiple sequential closures, which is the realistic
production scenario.

---

## System Architecture

```
OSM Jaipur graph (OSMnx)
        ↓
Edge cost prediction (Random Forest)
  └─ trained on eVED real OBD telemetry (Zhang et al., IEEE VTC 2025)
  └─ features: road gradient, speed, road type
  └─ output: Wh/km per edge
        ↓
Graph: 21,009 nodes · 55,553 edges · battery_cost on every edge
        ↓
Battery-Aware LPA* search
  └─ minimises total battery cost
  └─ incremental replan on road closure
  └─ bucket-discretised battery state (50 × 800 Wh = 40,000 Wh)
        ↓
Streamlit UI · Folium map · live rerouting visualisation
```

---

## Energy Model

**Dataset:** Extended Vehicle Energy Dataset (eVED)
— Zhang et al., IEEE VTC 2025.
Real OBD telemetry from Nissan Leaf vehicles (VehId 10, 11, 12)
driving real Michigan roads.

**Training data:** 2,141 real OBD segments + 6,000 synthetic
slope-augmented rows (anchored to eVED flat mean ~105 Wh/km)
to cover gradient extremes absent from flat Michigan roads.

**Model:** Random Forest Regressor
- R² = 0.41 (honest — real driving has genuine variability from
  traffic, driver behaviour, and conditions not captured in features)
- Feature importances: slope 76% · speed 22% · road type 1%
- Physics-correct outputs: uphill ~188 · flat ~85 · downhill ~41 Wh/km

**Vehicle:** Tata Nexon EV class, 40 kWh battery.
Model was trained on Nissan Leaf (24 kWh) OBD data;
energy patterns are scaled to the 40 kWh target vehicle.

---

## Algorithm — Battery-Aware LPA\*

Standard LPA\* (Koenig & Likhachev, 2002) minimises path cost
incrementally by reusing g-values from previous searches.

This implementation extends LPA\* with:

1. **Battery feasibility check** — routes that would exhaust the
   battery before reaching the destination are rejected, not just
   de-prioritised.

2. **Bucket-discretised battery state** — 50 buckets × 800 Wh =
   40,000 Wh total. Avoids floating-point g-value instability across
   replans.

3. **Admissible haversine heuristic** — h(n) = straight-line distance
   × 80 Wh/km (below eVED flat mean of ~85 Wh/km), guaranteeing
   the heuristic never overestimates.

4. **Parent pointer tracking** — parent pointers are recorded the
   moment g[node] = rhs[node] is committed, so path extraction after
   replan is always correct regardless of intermediate g-value staleness.

---

## Project Structure

```
ev_router/
├── app.py                        # Streamlit web app
├── config.py                     # Central constants
├── src/
│   ├── preprocess_eved.py        # eVED CSV → ev_telemetry.csv
│   ├── train_model.py            # RF training, saves battery_rf_model.pkl
│   ├── graph_builder.py          # OSMnx graph + RF edge costs → graphml
│   ├── lpa_star.py               # Battery-Aware LPA* implementation
│   └── benchmark.py              # 4-method comparison, 500 trials
├── data/
│   ├── raw/ev_telemetry.csv      # 2,141 real OBD segments
│   └── processed/
│       ├── jaipur_graph_with_attrs.graphml
│       └── benchmark_results.csv
└── models/
    └── battery_rf_model.pkl
```

---

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Build graph (run once)
python -m src.graph_builder

# Run benchmark
python -m src.benchmark

# Launch app
streamlit run app.py
```

---

## Key Configuration (`config.py`)

```python
BATTERY_CAPACITY_WH = 40_000   # Tata Nexon EV class
N_BUCKETS           = 50       # 50 × 800 Wh
N_DELIVERIES        = 500      # benchmark trials
JAIPUR_CENTER_LAT   = 26.9124
JAIPUR_CENTER_LON   = 75.7873
```

---

## Citations

- Zhang et al., *Extended Vehicle Energy Dataset (eVED)*, IEEE VTC 2025
- Koenig & Likhachev, *LPA\*: A Lifelong Planning Version of A\**, AAAI 2002
- Boeing, *OSMnx: New Methods for Acquiring, Constructing, Analyzing,
  and Visualizing Complex Street Networks*, 2017
