"""Run a chronological, rights-clean demo using only synthetic data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from keiba_place_ml.model import (
    fit_logistic,
    fit_xgb_relative,
    predict_logistic,
    predict_xgb_relative,
)
from keiba_place_ml.model_selection import candidate_score, select_with_incumbent_one_se
from keiba_place_ml.synthetic import make_synthetic_panel


def chronological_split(frame, train_fraction=0.8):
    dates = np.array(sorted(frame["race_date"].unique()))
    cut = dates[max(1, int(len(dates) * train_fraction)) - 1]
    train = frame.loc[frame["race_date"] <= cut].copy()
    test = frame.loc[frame["race_date"] > cut].copy()
    if train.empty or test.empty:
        raise ValueError("chronological split produced an empty partition")
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--races", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--no-xgb", action="store_true")
    args = parser.parse_args()

    panel = make_synthetic_panel(races=args.races, seed=args.seed)
    train, test = chronological_split(panel)

    logistic, prior = fit_logistic(train)
    predictions = {
        "logistic_incumbent": predict_logistic(logistic, test, prior_mean=prior)
    }

    if not args.no_xgb:
        xgb, xgb_prior = fit_xgb_relative(train)
        predictions["xgb_relative"] = predict_xgb_relative(
            xgb, test, prior_mean=xgb_prior
        )

    decision, table = select_with_incumbent_one_se(
        test, predictions, incumbent="logistic_incumbent"
    )
    report = {
        "data": {
            "synthetic": True,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_max_date": str(train["race_date"].max().date()),
            "test_min_date": str(test["race_date"].min().date()),
        },
        "decision": asdict(decision),
        "scores": [
            asdict(candidate_score(test, p, name=name))
            for name, p in predictions.items()
        ],
    }
    print(json.dumps(report, indent=2))
    print("\nSelection table:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
