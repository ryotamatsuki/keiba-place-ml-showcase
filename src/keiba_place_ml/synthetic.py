"""Synthetic, rights-clean race panel for runnable examples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _binomial_count(rng, starts: int, rate: float) -> int:
    return int(rng.binomial(max(starts, 0), float(np.clip(rate, 0.01, 0.99))))


def make_synthetic_panel(*, races: int = 600, seed: int = 20261003) -> pd.DataFrame:
    """Generate a toy panel with exactly three positive labels per race.

    Values are synthetic and do not correspond to real horses, races, odds, or results.
    """
    if races < 20:
        raise ValueError("Use at least 20 races for a meaningful chronological demo")

    rng = np.random.default_rng(seed)
    courses = np.array(["Course-A", "Course-B", "Course-C", "Course-D"])
    classes = np.array(["Allowance", "Open"])
    sexes = np.array(["M", "F", "G"])
    distances = np.array([1200, 1400, 1600, 1800, 2000, 2400])
    start = pd.Timestamp("2020-01-01")
    rows: list[dict[str, object]] = []

    for race_no in range(races):
        race_date = start + pd.Timedelta(days=race_no // 2)
        field_size = int(rng.integers(9, 17))
        distance = int(rng.choice(distances))
        racecourse = str(rng.choice(courses))
        race_class = str(rng.choice(classes, p=[0.7, 0.3]))
        latent = rng.normal(0.0, 1.0, field_size)
        draw = rng.permutation(np.arange(1, field_size + 1))
        noise = rng.normal(0.0, 0.8, field_size)
        top3_idx = np.argsort(-(latent + noise))[:3]

        early_style = np.clip(rng.beta(2, 4, field_size), 0, 1)
        front_share = float(np.mean(early_style <= 0.25))

        for i in range(field_size):
            ability_rate = 1.0 / (1.0 + np.exp(-(0.9 * latent[i] - 1.0)))
            career_starts = int(rng.integers(3, 31))
            turf_starts = int(rng.integers(0, career_starts + 1))
            same_distance_starts = int(rng.integers(0, min(career_starts, 10) + 1))
            same_course_starts = int(rng.integers(0, min(career_starts, 8) + 1))
            recent_finish = float(np.clip(0.55 - 0.12 * latent[i] + rng.normal(0, 0.12), 0, 1))
            recent_time = float(np.clip(0.025 - 0.007 * latent[i] + rng.normal(0, 0.008), -0.02, 0.08))

            rows.append(
                {
                    "race_id": f"SYN-{race_no:05d}",
                    "race_date": race_date,
                    "horse_id": f"SYN-{race_no:05d}-{i + 1:02d}",
                    "top3_label": int(i in top3_idx),
                    "field_size": field_size,
                    "draw_pct": float((draw[i] - 1) / max(field_size - 1, 1)),
                    "age": int(rng.integers(3, 9)),
                    "assigned_weight_kg": float(rng.choice([54, 55, 56, 57, 58])),
                    "assigned_weight_delta_from_prev_kg": float(rng.choice([-1, 0, 0, 0, 1])),
                    "days_since_prev": int(rng.integers(14, 120)),
                    "distance_change_from_prev_m": int(rng.choice([-400, -200, 0, 0, 200, 400])),
                    "surface_changed_from_prev": float(rng.random() < 0.1),
                    "career_starts": career_starts,
                    "career_top3": _binomial_count(rng, career_starts, ability_rate),
                    "turf_starts": turf_starts,
                    "turf_top3": _binomial_count(rng, turf_starts, ability_rate),
                    "same_distance_starts": same_distance_starts,
                    "same_distance_top3": _binomial_count(rng, same_distance_starts, ability_rate),
                    "same_course_starts": same_course_starts,
                    "same_course_top3": _binomial_count(rng, same_course_starts, ability_rate),
                    "recent3_finish_pct_mean": recent_finish,
                    "recent3_top3_count": int(rng.binomial(3, ability_rate)),
                    "recent3_open_plus_count": int(rng.binomial(3, 0.25 + 0.15 * (race_class == "Open"))),
                    "recent3_graded_count": int(rng.binomial(3, 0.15)),
                    "recent3_relative_time_mean": recent_time,
                    "recent4_early_pos_pct_mean": float(early_style[i]),
                    "front_forward_share": front_share,
                    "distance_m": distance,
                    "sex": str(rng.choice(sexes)),
                    "racecourse": racecourse,
                    "race_class": race_class,
                }
            )

    panel = pd.DataFrame(rows)
    panel["race_date"] = pd.to_datetime(panel["race_date"])
    return panel
