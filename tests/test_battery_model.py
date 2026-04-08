# tests/test_battery_model.py
"""Unit tests for battery consumption ML model."""

import pytest
import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.battery_model import (
    load_battery_model,
    predict_edge_cost,
    generate_synthetic_ev_data,
    load_kaggle_ev_data,
    prepare_training_data,
    _preprocess_kaggle_data
)
import config


@pytest.fixture(scope="module")
def model():
    """Load the trained battery model."""
    return load_battery_model()


class TestModelLoading:
    """Tests for model loading and basic functionality."""

    def test_model_loads_successfully(self, model):
        """Model should load without errors."""
        assert model is not None, "Model should load"

    def test_model_has_predict_method(self, model):
        """Model should have a predict method."""
        assert hasattr(model, 'predict'), "Model should have predict method"

    def test_model_has_feature_importances(self, model):
        """Random Forest should have feature importances."""
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == len(config.ML_FEATURES)


class TestPredictions:
    """Tests for battery consumption predictions."""

    def test_prediction_is_positive(self, model):
        """Energy consumption should always be positive."""
        cost = predict_edge_cost(model, slope=0, distance_km=1.0, 
                                  speed_kmh=30, road_type=0)
        assert cost > 0, "Energy cost should be positive"

    def test_prediction_is_reasonable(self, model):
        """Prediction for 1km should be in reasonable range (5-50 Wh)."""
        cost = predict_edge_cost(model, slope=0, distance_km=1.0,
                                  speed_kmh=30, road_type=0)
        assert 5 < cost < 50, f"Cost {cost} Wh for 1km seems unreasonable"

    def test_uphill_costs_more_than_flat(self, model):
        """Uphill (positive slope) should consume more energy."""
        flat = predict_edge_cost(model, slope=0, distance_km=1.0,
                                  speed_kmh=30, road_type=0)
        uphill = predict_edge_cost(model, slope=5, distance_km=1.0,
                                    speed_kmh=30, road_type=0)
        assert uphill > flat, "Uphill should cost more energy than flat"

    def test_downhill_costs_less_than_flat(self, model):
        """Downhill (negative slope) should consume less energy."""
        flat = predict_edge_cost(model, slope=0, distance_km=1.0,
                                  speed_kmh=30, road_type=0)
        downhill = predict_edge_cost(model, slope=-5, distance_km=1.0,
                                      speed_kmh=30, road_type=0)
        assert downhill < flat, "Downhill should cost less energy than flat"

    def test_longer_distance_costs_more(self, model):
        """Longer distance should consume more energy."""
        short = predict_edge_cost(model, slope=0, distance_km=0.5,
                                   speed_kmh=30, road_type=0)
        long = predict_edge_cost(model, slope=0, distance_km=2.0,
                                  speed_kmh=30, road_type=0)
        assert long > short, "Longer distance should cost more"

    def test_higher_speed_costs_more(self, model):
        """Higher speed should consume more energy (drag increases)."""
        slow = predict_edge_cost(model, slope=0, distance_km=1.0,
                                  speed_kmh=20, road_type=0)
        fast = predict_edge_cost(model, slope=0, distance_km=1.0,
                                  speed_kmh=50, road_type=0)
        assert fast > slow, "Higher speed should cost more (aerodynamic drag)"

    def test_batch_prediction_works(self, model):
        """Model should handle batch predictions."""
        X = np.array([
            [0, 1.0, 30, 0],   # flat, 1km, 30kmh, residential
            [5, 1.0, 30, 0],   # uphill
            [-5, 1.0, 30, 0],  # downhill
        ])
        predictions = model.predict(X)
        assert len(predictions) == 3
        assert all(p > 0 for p in predictions)


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generates_correct_columns(self):
        """Generated data should have all required columns."""
        df = generate_synthetic_ev_data(n=100, save=False)
        
        required_cols = config.ML_FEATURES + [config.ML_TARGET]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_generates_correct_row_count(self):
        """Should generate requested number of rows."""
        df = generate_synthetic_ev_data(n=500, save=False)
        assert len(df) == 500

    def test_values_in_valid_ranges(self):
        """Generated values should be in valid physical ranges."""
        df = generate_synthetic_ev_data(n=1000, save=False)
        
        assert df['slope'].min() >= -6, "Slope too negative"
        assert df['slope'].max() <= 6, "Slope too positive"
        assert df['distance_km'].min() >= 0.05, "Distance too short"
        assert df['distance_km'].max() <= 2.5, "Distance too long"
        assert df['speed_kmh'].min() >= 10, "Speed too slow"
        assert df['speed_kmh'].max() <= 55, "Speed too fast"
        assert df['road_type'].isin([0, 1, 2]).all(), "Invalid road type"
        assert df['energy_wh'].min() > 0, "Energy should be positive"

    def test_energy_correlates_with_distance(self):
        """Energy should correlate positively with distance."""
        df = generate_synthetic_ev_data(n=5000, save=False)
        correlation = df['energy_wh'].corr(df['distance_km'])
        assert correlation > 0.5, "Energy should correlate with distance"


class TestConfigValues:
    """Tests for configuration values."""

    def test_battery_capacity_reasonable(self):
        """Battery capacity should be reasonable for EV bike."""
        assert 500 <= config.BATTERY_CAPACITY_WH <= 2000

    def test_deviation_threshold_is_percentage(self):
        """Deviation threshold should be between 0 and 1."""
        assert 0 < config.BATTERY_DEVIATION_THRESHOLD < 1

    def test_ml_features_match_model(self, model):
        """Config features should match model's expected features."""
        assert len(config.ML_FEATURES) == model.n_features_in_
