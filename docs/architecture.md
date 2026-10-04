# Architecture

The public showcase isolates four ML concerns.

```text
synthetic / licensed input
        |
        v
market-free validation
        |
        v
leakage-conscious feature engineering
  - empirical-Bayes history rates
  - strictly-prior-date history
  - full-field leave-one-out comparisons
        |
        v
probability model
  - logistic incumbent
  - XGBoost relative candidate
        |
        v
proper-score evaluation
  - race-macro Brier
  - race-macro log loss
        |
        v
paired date-clustered one-SE gate
```

## Why race-macro scoring?

Runner-micro scoring gives larger fields more weight simply because they contain more
rows. The selection helper first computes a loss per race and then averages across
races, so each race receives equal weight.

## Why a paired one-SE gate?

Picking the numerically best validation score encourages model churn. The incumbent is
kept unless the point-best challenger beats it by more than one paired clustered
standard error. The comparison is paired because both candidates are evaluated on the
same races, and clustered by date because races on the same date can share conditions.

## Why full-field relative features?

A runner's absolute historical rate can mean something different depending on the
strength of the current field. Relative features compare each starter against the
leave-one-out mean of its peers. The implementation refuses incomplete fields so this
comparison cannot silently use a truncated context.

## What is not shown here?

The private project contains ingestion, source audits, production snapshots, immutable
pre-race locks, and post-race evaluation. Those operational pieces are excluded from
this public repository because the showcase is intentionally data-rights-clean.
