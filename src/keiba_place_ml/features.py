"""Leakage-conscious, market-free feature engineering.

This module is adapted from author-written research code. It contains no source
acquisition logic and no third-party row-level data.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

HISTORY_PAIRS = {
    "career": ("career_top3", "career_starts"),
    "turf": ("turf_top3", "turf_starts"),
    "same_distance": ("same_distance_top3", "same_distance_starts"),
    "same_course": ("same_course_top3", "same_course_starts"),
}

NUMERIC_FEATURES = [
    "draw_pct",
    "age",
    "assigned_weight_kg",
    "assigned_weight_delta_from_prev_kg",
    "log_days_since_prev",
    "distance_change_from_prev_m",
    "surface_changed_from_prev",
    "career_top3_shrunk",
    "log_career_starts",
    "turf_top3_shrunk",
    "log_turf_starts",
    "same_distance_top3_shrunk",
    "log_same_distance_starts",
    "same_course_top3_shrunk",
    "log_same_course_starts",
    "recent3_finish_pct_mean",
    "recent3_top3_count",
    "recent3_open_plus_count",
    "recent3_graded_count",
    "recent3_relative_time_mean",
    "recent4_early_pos_pct_mean",
    "front_forward_share",
    "field_size",
    "distance_m",
]

CATEGORICAL_FEATURES = ["sex", "racecourse", "race_class"]

RELATIVE_FEATURES = (
    "rel_career_top3_vs_others",
    "rel_same_distance_top3_vs_others",
    "rel_same_course_top3_vs_others",
    "rel_recent3_finish_vs_others",
    "rel_recent3_time_vs_others",
)

SOURCE_RULES = (
    ("career_top3_shrunk", "rel_career_top3_vs_others", 1.0),
    ("same_distance_top3_shrunk", "rel_same_distance_top3_vs_others", 1.0),
    ("same_course_top3_shrunk", "rel_same_course_top3_vs_others", 1.0),
    ("recent3_finish_pct_mean", "rel_recent3_finish_vs_others", -1.0),
    ("recent3_relative_time_mean", "rel_recent3_time_vs_others", -1.0),
)

MARKET_TOKENS = (
    "odds",
    "market",
    "popularity",
    "favorite",
    "payout",
    "payoff",
    "return",
    "tansho",
    "fukusho",
)


def validate_market_free(columns: Iterable[str]) -> None:
    """Reject market/outcome-utility columns from the non-market feature path."""
    bad = sorted(
        str(col)
        for col in columns
        if any(token in str(col).lower() for token in MARKET_TOKENS)
    )
    if bad:
        raise ValueError(f"Forbidden market-like columns: {bad}")


def validate_training_frame(frame: pd.DataFrame) -> None:
    validate_market_free(frame.columns)
    required = {
        "top3_label",
        "race_id",
        "horse_id",
        "days_since_prev",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        *(column for pair in HISTORY_PAIRS.values() for column in pair),
    }
    derived = {
        "log_days_since_prev",
        *(f"{prefix}_top3_shrunk" for prefix in HISTORY_PAIRS),
        *(f"log_{prefix}_starts" for prefix in HISTORY_PAIRS),
    }
    missing = sorted((required - derived).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing training columns: {missing}")
    if frame["top3_label"].isna().any():
        raise ValueError("top3_label contains missing values")
    labels = pd.to_numeric(frame["top3_label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("top3_label must be binary")


def engineer_features(
    frame: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> pd.DataFrame:
    """Apply a common empirical-Bayes shrinkage rule to sparse history rates."""
    if not 0.0 < prior_mean < 1.0:
        raise ValueError("prior_mean must be in (0, 1)")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    validate_market_free(frame.columns)

    out = frame.copy()
    out["log_days_since_prev"] = np.log1p(
        pd.to_numeric(out["days_since_prev"], errors="coerce")
    )
    alpha = prior_mean * prior_strength
    beta = (1.0 - prior_mean) * prior_strength

    for prefix, (success_col, starts_col) in HISTORY_PAIRS.items():
        successes = pd.to_numeric(out[success_col], errors="coerce")
        starts = pd.to_numeric(out[starts_col], errors="coerce")
        invalid = (successes < 0) | (starts < 0) | (successes > starts)
        if invalid.fillna(False).any():
            raise ValueError(f"Invalid top3/start counts in {prefix}")
        out[f"{prefix}_top3_shrunk"] = (successes + alpha) / (
            starts + alpha + beta
        )
        out[f"log_{prefix}_starts"] = np.log1p(starts)
    return out


def assert_full_field_context(context: pd.DataFrame) -> None:
    """Require one row per starter before creating within-race relative features."""
    required = {"race_id", "horse_id", "field_size"}
    missing = required.difference(context.columns)
    if missing:
        raise ValueError(f"Missing full-field context columns: {sorted(missing)}")
    if context.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate race_id x horse_id rows")

    unique_sizes = context.groupby("race_id")["field_size"].nunique()
    if unique_sizes.gt(1).any():
        raise ValueError("field_size is not constant within race")
    observed = context.groupby("race_id").size()
    expected = context.groupby("race_id")["field_size"].first().astype(int)
    if observed.ne(expected).any():
        raise ValueError("Context does not contain every starter in the race")


def _leave_one_out_mean(frame: pd.DataFrame, source_col: str) -> pd.Series:
    values = pd.to_numeric(frame[source_col], errors="coerce")
    valid = values.notna().astype(int)
    sums = values.fillna(0.0).groupby(frame["race_id"]).transform("sum")
    counts = valid.groupby(frame["race_id"]).transform("sum")
    peer_sum = sums - values.fillna(0.0)
    peer_count = counts - valid
    return (peer_sum / peer_count.where(peer_count.gt(0))).where(values.notna())


def add_relative_ability_features(
    frame: pd.DataFrame,
    *,
    prior_mean: float,
    prior_strength: float = 6.0,
) -> pd.DataFrame:
    """Attach leave-one-out comparisons against the rest of each complete field."""
    assert_full_field_context(frame)
    engineered = engineer_features(
        frame, prior_mean=prior_mean, prior_strength=prior_strength
    )
    out = engineered.copy()
    for source_col, output_col, direction in SOURCE_RULES:
        own = pd.to_numeric(engineered[source_col], errors="coerce")
        peers = _leave_one_out_mean(engineered, source_col)
        out[output_col] = own - peers if direction > 0 else peers - own
    return out


def distance_regime(distance: np.ndarray) -> np.ndarray:
    """Simple predeclared distance buckets used by the research implementation."""
    return np.digitize(np.asarray(distance, dtype=float), [1400.0, 2000.0, 2600.0], right=True)


def augment_distance_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Create strictly-prior-date history counts without same-day/future leakage.

    This implementation intentionally uses a strict date comparison. Other races on
    the same calendar day are not treated as known information.
    """
    out = panel.reset_index(drop=True).copy()
    required = {"horse_id", "race_date", "surface", "distance_m", "racecourse"}
    if required.difference(out.columns) or out["horse_id"].isna().any():
        raise ValueError("Missing distance-history keys")

    dates = pd.to_datetime(out["race_date"], errors="raise").to_numpy()
    labels = pd.to_numeric(out.get("top3_label", pd.Series(np.nan, index=out.index)))
    if not labels.dropna().isin([0, 1]).all():
        raise ValueError("History labels must be binary")
    y = labels.to_numpy(dtype=float)
    distance = pd.to_numeric(out["distance_m"], errors="coerce").to_numpy(float)
    surface = out["surface"].fillna("").astype(str).to_numpy()
    course = out["racecourse"].fillna("").astype(str).to_numpy()

    keys = ("same_surface", "near", "regime", "course_distance")
    values = {key: np.full((len(out), 2), np.nan) for key in keys}

    for positions in out.groupby("horse_id", sort=False).indices.values():
        d = distance[positions]
        s = surface[positions]
        c = course[positions]
        t = dates[positions]
        target = y[positions]

        prior = t[:, None] > t[None, :]
        same_surface = (s[:, None] == s[None, :]) & prior
        same_distance = d[:, None] == d[None, :]
        regimes = distance_regime(d)
        rules = {
            "same_surface": same_surface,
            "near": same_surface & (np.abs(d[:, None] - d[None, :]) <= 200),
            "regime": same_surface & (regimes[:, None] == regimes[None, :]),
            "course_distance": same_surface & same_distance & (c[:, None] == c[None, :]),
        }

        for key, mask in rules.items():
            valid = s != ""
            if key != "same_surface":
                valid &= np.isfinite(d)
            if key == "course_distance":
                valid &= c != ""
            unknown = (mask & ~np.isfinite(target)[None, :]).any(axis=1)
            counts = mask.sum(axis=1).astype(float)
            successes = mask @ np.nan_to_num(target)
            counts[~valid | unknown] = np.nan
            successes[~valid | unknown] = np.nan
            values[key][positions, 0] = counts
            values[key][positions, 1] = successes

    for key, pair in values.items():
        out[f"{key}_starts"] = pair[:, 0]
        out[f"{key}_top3"] = pair[:, 1]
    return out
