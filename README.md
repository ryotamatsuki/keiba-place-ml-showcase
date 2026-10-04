# keiba-place-ml-showcase

A public, data-rights-clean showcase of the machine-learning implementation behind a
Japanese horse-racing `P(top3)` probability experiment.

This repository is intentionally **not** a betting-data dump. It contains the
author-written ML and validation logic needed to demonstrate the engineering ideas:

- strictly chronological feature construction;
- market-free non-market features;
- empirical-Bayes shrinkage for sparse history;
- full-field leave-one-out relative ability features;
- probabilistic models for `P(top3)`;
- Brier score / log-loss evaluation;
- race-macro scoring;
- paired date-clustered one-standard-error model selection;
- reproducible CI tests;
- synthetic demo data so the code can run without redistributing third-party race data.

## What is deliberately excluded

The private research repository contains additional operational material. None of the
following is copied here:

- row-level historical race data obtained from third-party/public web sources;
- saved HTML/pages, caches, screenshots, or source captures;
- the 2026 live-history database;
- provider-specific scraping/parsing code;
- production prediction locks or outcome files;
- trained production model bundles;
- credentials, cookies, tokens, or private configuration.

See [DATA_POLICY.md](DATA_POLICY.md).

## Quick start

```bash
python -m pip install -e ".[dev,model]"
python scripts/train_demo.py --races 600
pytest -q
```

The demo generator creates synthetic races in memory. No real horse names, race records,
odds, or third-party row-level data are required.

## Why this project exists

The core problem is not "can a model fit horse-racing data?" It is whether a probability
model can be built and evaluated without chronology leakage, compared against an incumbent
with an explicit selection rule, frozen before future outcomes, and reproduced later.

The public code therefore emphasizes the **ML system** rather than the data source.

## Repository layout

```text
src/keiba_place_ml/
  features.py          # leakage-conscious feature engineering
  model.py             # logistic / XGBoost pipelines
  model_selection.py   # race-macro scores + paired one-SE gate
  synthetic.py         # rights-clean synthetic demo generator
scripts/
  train_demo.py
tests/
docs/
  architecture.md
```

## Relationship to the private research repository

The implementation here is a rights-clean extraction/adaptation of author-written code
from `keiba-place-probability-lab`. The private repository remains the canonical
research/audit environment. This public repository is the reproducible educational
showcase.

No license is granted beyond what is explicitly stated in this repository.
