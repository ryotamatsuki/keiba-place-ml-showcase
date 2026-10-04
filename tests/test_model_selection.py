import numpy as np

from keiba_place_ml.model_selection import (
    candidate_score,
    select_with_incumbent_one_se,
)
from keiba_place_ml.synthetic import make_synthetic_panel


def test_race_macro_score_is_finite():
    frame = make_synthetic_panel(races=30)
    p = np.full(len(frame), 3 / frame.field_size.to_numpy())
    score = candidate_score(frame, p, name="uniform")
    assert np.isfinite(score.race_macro_brier)
    assert score.races == frame.race_id.nunique()


def test_identical_incumbent_is_retained():
    frame = make_synthetic_panel(races=30)
    p = np.full(len(frame), 0.2)
    decision, table = select_with_incumbent_one_se(
        frame,
        {"incumbent": p, "same": p.copy()},
        incumbent="incumbent",
    )
    assert decision.winner == "incumbent"
    assert decision.incumbent_retained
    assert len(table) == 2
