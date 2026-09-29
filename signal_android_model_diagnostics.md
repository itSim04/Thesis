# Signal Android model diagnostics

## Outcome distribution (modeling sample)

- WMC_t1 is a non-negative integer; unique values=67; zeros=30.2588%.
- Unconditional variance/mean=23.879 (overdispersed unconditionally).
- corr(WMC_t, WMC_t1)=0.9690; share delta_WMC=0=79.0484%.
- Primary family = OLS on WMC_t1 with identity scale, because the lagged outcome is nearly linear and a log-link count model would exponentiate a large raw WMC_t. NegativeBinomial with log1p(WMC_t) is the robustness family, not the selection winner.

## Predictor representation

- DataClass, LongMethod, LongParameterList, MutableStaticState: **binary presence** (counts are almost 0/1; the thesis construct is presence/configuration).
- LawOfDemeter: **log1p(count)**. Binary LoD is banned (~86% present).
- GodClass: excluded from all fitted models.

## Residual diagnostics (primary OLS)

| Model | mean | sd | skew | kurtosis | min | max | share |resid|>10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_ols_cluster | 0.0000 | 2.8413 | -0.0785 | 107.2331 | -57.5030 | 36.6779 | 0.0121 |
| B_ols_cluster | 0.0000 | 2.8049 | -0.1217 | 97.7609 | -55.5209 | 37.0288 | 0.0125 |

OLS residuals inherit a long right tail from WMC. That is why NB is reported as robustness, not because it produced a stronger interaction result.

## VIF (Model A OLS RHS, including transition FE)

| variable | VIF |
| --- | ---: |
| WMC_t | 4.36 |
| LOC_t | 2.82 |
| NOM_t | 3.98 |
| DataClass_present | 1.13 |
| LongMethod_present | 1.36 |
| LongParameterList_present | 1.12 |
| MutableStaticState_present | 1.02 |
| LawOfDemeter_log1p | 1.66 |
| tr_2017-1->2018-1 | 1.60 |
| tr_2018-1->2019-1 | 1.69 |
| tr_2019-1->2020-1 | 1.73 |
| tr_2020-1->2021-1 | 1.99 |

## Control selection notes

- Dropped CBO_t: 99.8331% of the modeling sample is 0 (threshold 95%); nunique=2.
- Retained LOC_t: VIF=1.72 (rule: drop only if VIF>10.0).
- Retained NOM_t: VIF=3.40 (rule: drop only if VIF>10.0).

## Fit failures

None.

## Prespecified interaction support rule

- Binary × binary: both_present ≥ 20
- Binary × LoD intensity: n(A=1) ≥ 20 and both_with_LoD_present ≥ 20 and LoD count varies among A=1
- Terms failing the rule are unavailable, not tested-and-dropped by p-value.
