import numpy as np
import pandas as pd
import pytest

from keiba_place_ml.features import (
    add_relative_ability_features,
    augment_distance_history,
    engineer_features,
    validate_market_free,
)
from keiba_place_ml.synthetic import make_synthetic_panel


def test_synthetic_panel_has_exactly_three_top3_per_race():
    frame = make_synthetic_panel(races=30)
    assert frame.groupby("race_id").top3_label.sum().eq(3).all()
    assert frame.groupby("race_id").size().eq(
        frame.groupby("race_id").field_size.first()
    ).all()


def test_market_columns_are_rejected():
    with pytest.raises(ValueError, match="Forbidden market-like"):
        validate_market_free(["age", "win_odds"])


def test_shrinkage_handles_sparse_history():
    frame = make_synthetic_panel(races=20).head(3).copy()
    frame["career_starts"] = [0, 1, 10]
    frame["career_top3"] = [0, 1, 5]
    out = engineer_features(frame, prior_mean=0.2)
    assert out.loc[out.index[0], "career_top3_shrunk"] == pytest.approx(0.2)
    assert out.loc[out.index[1], "career_top3_shrunk"] < 1.0


def test_relative_features_use_complete_field_peers():
    frame = make_synthetic_panel(races=20)
    one_race = frame.loc[frame.race_id == frame.race_id.iloc[0]].copy()
    out = add_relative_ability_features(one_race, prior_mean=0.2)
    assert len(out) == len(one_race)
    assert np.isfinite(out["rel_career_top3_vs_others"]).all()


def test_future_and_same_day_labels_cannot_change_prior_history():
    base = pd.DataFrame(
        {
            "horse_id": ["a"] * 5,
            "race_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-03-01", "2020-04-01"]
            ),
            "surface": ["turf"] * 5,
            "distance_m": [1200, 1400, 1200, 1200, 1200],
            "racecourse": ["A"] * 5,
            "top3_label": [1, 0, 1, 0, 0],
        }
    )
    first = augment_distance_history(base)
    changed = base.copy()
    changed.loc[2:, "top3_label"] = 1 - changed.loc[2:, "top3_label"]
    second = augment_distance_history(changed)

    cols = [c for c in first if c.endswith(("_starts", "_top3"))]
    pd.testing.assert_frame_equal(first.loc[:2, cols], second.loc[:2, cols])
