# Data policy

This repository is designed to remain safe to publish even when the private research
project uses data collected from multiple public or third-party sources.

## Included

- author-written ML code;
- tests;
- documentation;
- synthetic demo data generated locally at runtime;
- aggregate methodological descriptions that do not reproduce source datasets.

## Excluded

- real row-level race histories;
- real horse-level training panels;
- HTML/page captures, caches, screenshots, or archived provider responses;
- provider-specific scraping and parsing code;
- production prediction inputs/outputs that reproduce third-party datasets;
- trained production bundles derived from non-redistributable data;
- credentials, cookies, account identifiers, or tokens.

"Publicly viewable" does not automatically mean "redistributable." The private research
repository therefore remains separate from this public showcase.

If a real dataset is added in the future, its redistribution license must be verified
before commit, and the repository should record the dataset name, source, license, and
allowed redistribution scope.
