# Signal Android modeling sample

Built from `signal_android_pilot_panel.csv`. Dataset was not rebuilt.

## Filters

- `smell_status_known_t == 1`
- `WMC_t` and `WMC_t1` numeric
- `smell_status_known_t1` is **not** required
- Unmatched PMD rows are excluded, not recoded as zero

## Counts

- Canonical longitudinal rows: **3495**
- `smell_status_known_t = 1`: **2396**
- Numeric WMC_t and WMC_t1 (full canonical file): **3495**
- Final complete-case modeling sample: **2396**
- Unique classes: **1018**
- Unknown-status rows excluded: **1099**

## Observations per transition

| version_t | version_t1 | n |
| --- | --- | ---: |
| 2016-1 | 2017-1 | 405 |
| 2017-1 | 2018-1 | 349 |
| 2018-1 | 2019-1 | 419 |
| 2019-1 | 2020-1 | 450 |
| 2020-1 | 2021-1 | 773 |

## Observations per class

mean=2.354; median=1.0; min=1; max=5

## Zero-fill check

Passed: among `smell_status_known_t=0` rows, all candidate count fields are NULL, never 0.

## Controls retained after variance/VIF rules

LOC_t, NOM_t

- Dropped CBO_t: 99.8331% of the modeling sample is 0 (threshold 95%); nunique=2.
- Retained LOC_t: VIF=1.72 (rule: drop only if VIF>10.0).
- Retained NOM_t: VIF=3.40 (rule: drop only if VIF>10.0).

Unit of analysis: class-transition row. Repeated classes are not treated as independent.
