# EV Delivery Router with LPA* Real-Time Rerouting

A real-time routing system for electric vehicle delivery bikes that handles dynamic road closures and battery deviations using **LPA\* (Lifelong Planning A\*)** incremental replanning.

## Problem Statement

An EV delivery bike starts a route in Jaipur, India. Midway:
- Roads close dynamically (traffic, accidents, construction)
- Battery drains faster than predicted on slopes

The system must **reroute instantly** without recomputing the entire route from scratch.

## Key Features

| Feature | Implementation |
|---------|----------------|
| **Incremental Replanning** | LPA\* algorithm maintains g(n) and rhs(n), only updates inconsistent nodes |
| **Battery-Aware Routing** | State space is (node, battery_level), not just location |
| **ML Energy Prediction** | Random Forest model predicts Wh consumption from slope, distance, speed, road type |
| **Dynamic Disruptions** | Road closures (edge weight → ∞) and battery deviation >15% trigger replanning |
| **Real Data Support** | Supports Kaggle EV datasets with automatic preprocessing |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EV Delivery Router                        │
├─────────────────────────────────────────────────────────────┤
│  OSM Graph Parser                                            │
│  ├── Jaipur road network                                     │
│  ├── SRTM elevation data                                     │
│  └── Edge attributes: distance, slope, road_type            │
├─────────────────────────────────────────────────────────────┤
│  ML Battery Model (Random Forest)                           │
│  ├── Features: slope, distance_km, speed_kmh, road_type     │
│  ├── Target: energy_wh                                       │
│  ├── Supports real Kaggle EV datasets                       │
│  └── Synthetic data fallback for training                   │
├─────────────────────────────────────────────────────────────┤
│  LPA* Router (Two Variants)                                  │
│  ├── Standard LPA*: (node) state space                      │
│  ├── Battery-Aware LPA*: (node, battery_bucket) state space │
│  ├── Maintains g(n), rhs(n) for each state                  │
│  └── Incremental update on edge weight changes              │
├─────────────────────────────────────────────────────────────┤
│  Disruption Handlers                                         │
│  ├── Road closure → set edge weight to infinity             │
│  └── Battery deviation >15% → scale remaining edge weights  │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Python 3.10+**
- **OSMnx** - OpenStreetMap road network extraction
- **NetworkX** - Graph data structure
- **Scikit-learn** - Random Forest for battery prediction
- **SRTM** - Elevation data
- **Folium** - Interactive map visualization

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ev-delivery-router.git
cd ev-delivery-router
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

## Quick Start

### 1. Train Battery Model

**Option A: With Kaggle Dataset (Recommended)**
```bash
# Download EV dataset from Kaggle and place in data/raw/kaggle_ev_data.csv
# Supported datasets: Electric Vehicle Charging Sessions, VED, etc.
python -c "from src.battery_model import prepare_training_data, train_battery_model; prepare_training_data(); train_battery_model()"
```

**Option B: With Synthetic Data (Fallback)**
```bash
python -c "from src.battery_model import generate_synthetic_ev_data, train_battery_model; generate_synthetic_ev_data(); train_battery_model()"
```

### 2. Build Graph with Elevation
```bash
python -c "from src.battery_model import load_battery_model; from src.graph_builder import build_full_graph; build_full_graph(load_battery_model())"
```

### 3. Run Benchmark (500 Deliveries)
```bash
python src/benchmark.py
```

### 4. Generate Visualization
```bash
python visualizer/dashboard.py
```

## Benchmark Results (500 Simulated Deliveries)

| Method | Avg Replan Time | Nodes Expanded | Success Rate | Battery Failures |
|--------|-----------------|----------------|--------------|------------------|
| Dijkstra (static) | ~2 ms | N/A | ~95% | Cannot replan |
| Full A\* replan | ~80 ms | ~15,000 | ~92% | Some |
| **LPA\* incremental** | **~15 ms** | **~2,000** | ~92% | Some |
| **LPA\* Battery-Aware** | ~25 ms | ~5,000 | ~90% | **Guaranteed 0** |

**Key Insights:**
- LPA\* achieves **3-10x speedup** over full A\* replanning
- Battery-Aware LPA\* **guarantees** no battery failures by using (node, battery) state space
- Trade-off: Battery-aware explores more states but provides stronger guarantees

## Project Structure

```
ev_router/
├── config.py                 # All constants and paths
├── src/
│   ├── graph_builder.py      # OSM download + elevation + edge attributes
│   ├── battery_model.py      # RF training + Kaggle data loading
│   ├── lpa_star.py           # LPA* algorithm (node-only state space)
│   ├── lpa_star_battery.py   # LPA* with (node, battery_bucket) state space
│   ├── simulation.py         # Delivery simulation with disruptions
│   └── benchmark.py          # 500-trial benchmark (4 methods)
├── visualizer/
│   └── dashboard.py          # Folium interactive map
├── data/
│   ├── raw/                  # OSM graph, EV telemetry CSV, Kaggle data
│   └── processed/            # Graph with attributes, benchmark results
├── models/
│   └── battery_rf_model.pkl  # Trained Random Forest model
└── tests/
    ├── test_lpa_star.py      # LPA* + Battery-aware LPA* tests
    ├── test_battery_model.py # Battery model + Kaggle loading tests
    └── test_simulation.py    # Simulation tests
```

## Algorithm Details

### LPA* (Lifelong Planning A*)

LPA* maintains two values for each node:
- **g(n)**: Current best known cost from start to n
- **rhs(n)**: One-step lookahead value based on predecessors

A node is **consistent** when `g(n) = rhs(n)`.

On edge weight change (road closure or battery update):
1. Update `rhs` of affected nodes
2. Add inconsistent nodes to priority queue
3. Process queue until goal is consistent

This avoids recomputing the entire shortest path tree from scratch.

### Battery-Aware State Space

Instead of just tracking location, we track `(node_id, battery_bucket)`:
- 100 buckets (each = 10 Wh of the 1000 Wh capacity)
- Transitions only valid if remaining battery ≥ 0
- **Guarantees** found route is battery-feasible (no stranded vehicles!)

### Disruption Triggers

1. **Road Closure**: Edge weight set to infinity, affected nodes marked inconsistent
2. **Battery Deviation >15%**: Remaining edge weights scaled by actual/predicted ratio

## Using Kaggle Datasets

The system supports multiple Kaggle EV dataset formats:

1. **Trip-based datasets** (distance, energy, speed columns)
2. **Charging session datasets** (energy_kwh, duration columns)
3. **VED-style telemetry** (vehicle_speed, power columns)

Place your dataset at `data/raw/kaggle_ev_data.csv` and run:
```bash
python src/battery_model.py
```

The system auto-detects the format and preprocesses accordingly.

## Running Tests

```bash
python -m pytest tests/ -v --cov=src
```

## Future Improvements

- [x] ~~Integration with real Kaggle EV telemetry dataset~~
- [x] ~~Battery-aware LPA\* benchmarking~~
- [ ] Multi-vehicle fleet routing optimization
- [ ] Charging station waypoint integration
- [ ] Real-time traffic API integration
- [ ] D\* Lite comparison (backward search variant)

## References

- Koenig, S., & Likhachev, M. (2002). D* Lite. AAAI.
- Koenig, S., Likhachev, M., & Furcy, D. (2004). Lifelong Planning A*. AIJ.
- OpenStreetMap contributors

## Author

**Tarun** - IIT Guwahati  
SDE Intern Project | DSA + ML

## License

MIT License