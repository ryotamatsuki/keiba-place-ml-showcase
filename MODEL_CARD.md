# Model card

## Task

Estimate marginal probability `P(top3)` for each starter.

## Public-demo models

1. L2-regularized logistic regression.
2. XGBoost with full-field leave-one-out relative-ability features.

The public demo is trained only on synthetic data. It is not a betting product and its
demo metrics do not represent real-world predictive performance.

## Feature principles

- no odds or market-derived columns in the non-market path;
- history rates use common empirical-Bayes shrinkage;
- relative features compare a runner with the rest of its complete field;
- chronology-sensitive histories use strictly earlier dates;
- same-day future information is not considered available.

## Evaluation

Primary comparison uses race-macro Brier score. Log loss is reported as a proper-score
companion diagnostic. Candidate replacement uses a paired, race-date-clustered
one-standard-error incumbent-retention rule.

## Known limitations

- synthetic data cannot validate real-world calibration;
- the public repo omits production data acquisition and production model artifacts;
- `P(top3)` is a marginal probability, not a full joint finishing-order distribution.
