"""Race-macro proper scores and paired one-standard-error selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateScore:
    name: str
    race_macro_brier: float
    race_macro_log_loss: float
    runner_micro_brier: float
    runner_micro_log_loss: float
    races: int
    rows: int


@dataclass(frozen=True)
class SelectionDecision:
    winner: str
    point_brier_best: str
    incumbent: str | None
    incumbent_retained: bool
    reason: str


def _validate_probability(probability: np.ndarray, rows: int) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or len(p) != rows:
        raise ValueError("probability must be a 1D array aligned to frame rows")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability must be finite and lie in [0, 1]")
    return p


def _row_losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    return (p - y) ** 2, -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))


def race_loss_table(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    race_col: str = "race_id",
    block_col: str = "race_date",
    target_col: str = "top3_label",
) -> pd.DataFrame:
    required = {race_col, block_col, target_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing selection columns: {sorted(missing)}")

    p = _validate_probability(probability, len(frame))
    y = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("target must be binary 0/1")

    work = frame[[race_col, block_col]].copy()
    work["_brier"], work["_log_loss"] = _row_losses(y, p)
    if work.groupby(race_col)[block_col].nunique(dropna=False).gt(1).any():
        raise ValueError("block_col must be constant within each race")

    out = (
        work.groupby(race_col, sort=False, dropna=False)
        .agg(
            block=(block_col, "first"),
            rows=("_brier", "size"),
            brier=("_brier", "mean"),
            log_loss=("_log_loss", "mean"),
        )
        .reset_index()
    )
    if out["block"].isna().any():
        raise ValueError("block_col contains missing values")
    return out


def candidate_score(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    name: str,
) -> CandidateScore:
    p = _validate_probability(probability, len(frame))
    y = frame["top3_label"].astype(int).to_numpy(dtype=float)
    race = race_loss_table(frame, p)
    row_brier, row_logloss = _row_losses(y, p)
    return CandidateScore(
        name=name,
        race_macro_brier=float(race["brier"].mean()),
        race_macro_log_loss=float(race["log_loss"].mean()),
        runner_micro_brier=float(row_brier.mean()),
        runner_micro_log_loss=float(row_logloss.mean()),
        races=len(race),
        rows=len(frame),
    )


def _clustered_mean_and_se(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    g = np.asarray(clusters)
    if x.ndim != 1 or g.ndim != 1 or len(x) != len(g) or len(x) == 0:
        raise ValueError("values and clusters must be nonempty aligned 1D arrays")
    mean = float(x.mean())
    unique = pd.unique(g)
    if len(unique) < 2:
        return mean, float("inf")

    centered = x - mean
    cluster_scores = np.array([centered[g == label].sum() for label in unique], dtype=float)
    variance = (len(unique) / (len(unique) - 1.0)) * float(
        np.sum(cluster_scores**2)
    ) / (len(x) ** 2)
    return mean, sqrt(max(0.0, variance))


def paired_brier_delta(
    frame: pd.DataFrame,
    candidate_probability: np.ndarray,
    reference_probability: np.ndarray,
) -> tuple[float, float]:
    cand = race_loss_table(frame, candidate_probability).rename(
        columns={"brier": "cand_brier"}
    )
    ref = race_loss_table(frame, reference_probability).rename(
        columns={"brier": "ref_brier"}
    )
    merged = cand.merge(
        ref[["race_id", "block", "ref_brier"]],
        on=["race_id", "block"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(cand) or len(merged) != len(ref):
        raise ValueError("candidate and reference must cover identical races")
    delta = (merged["cand_brier"] - merged["ref_brier"]).to_numpy(dtype=float)
    return _clustered_mean_and_se(delta, merged["block"].to_numpy())


def select_with_incumbent_one_se(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    *,
    incumbent: str | None,
    one_se_multiplier: float = 1.0,
) -> tuple[SelectionDecision, pd.DataFrame]:
    """Select by race-macro Brier while regularizing against model churn."""
    if not predictions:
        raise ValueError("predictions must contain at least one candidate")
    if incumbent is not None and incumbent not in predictions:
        raise ValueError("incumbent must be present in predictions")

    scores = pd.DataFrame(
        [candidate_score(frame, p, name=name).__dict__ for name, p in predictions.items()]
    ).sort_values(["race_macro_brier", "race_macro_log_loss", "name"], kind="stable")
    best = str(scores.iloc[0]["name"])

    diagnostics = []
    for name, p in predictions.items():
        mean_delta, se_delta = paired_brier_delta(frame, p, predictions[best])
        diagnostics.append(
            {
                "name": name,
                "point_brier_best": name == best,
                "mean_brier_delta_vs_best": mean_delta,
                "se_brier_delta_vs_best": se_delta,
                "within_one_se_brier": (
                    mean_delta <= one_se_multiplier * se_delta + 1e-15
                ),
            }
        )
    table = scores.merge(pd.DataFrame(diagnostics), on="name", validate="one_to_one")

    retained = False
    if incumbent is not None:
        retained = bool(
            table.loc[table["name"] == incumbent, "within_one_se_brier"].iloc[0]
        )

    if retained:
        winner = incumbent
        reason = (
            "incumbent retained: race-macro Brier is within one paired "
            "date-clustered SE of the point-Brier-best"
        )
    else:
        winner = best
        reason = (
            "point-Brier-best selected: no incumbent was supplied or the incumbent "
            "fell outside the paired one-SE Brier retention band"
        )
    return (
        SelectionDecision(winner, best, incumbent, retained, reason),
        table.reset_index(drop=True),
    )
