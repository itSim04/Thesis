# Signal Android Model A vs Model B

## 1. Research question

Within the Signal Android longitudinal sample for which PMD smell status is observed, does explicitly modeling smell interactions provide measurable incremental explanatory or predictive value beyond a model containing only individual-smell information?

This is not a test of whether smells are associated with WMC, and it is not a causal claim.

## 2. Sample construction

- Canonical rows: 3495
- Modeling sample: **2396** class-transition rows, **1018** classes
- Filter: `smell_status_known_t=1` and numeric `WMC_t`, `WMC_t1`
- Unmatched PMD classes excluded (counts remain NULL in the canonical file; zero-fill check passed)

| Transition | n |
| --- | ---: |
| 2016-1 → 2017-1 | 405 |
| 2017-1 → 2018-1 | 349 |
| 2018-1 → 2019-1 | 419 |
| 2019-1 → 2020-1 | 450 |
| 2020-1 → 2021-1 | 773 |

## 3. Statistical-unit definition

One row is a top-level class observed across a consecutive version pair. Classes can contribute up to five rows. Primary inference uses **class-clustered standard errors**. Robustness: GEE with exchangeable working correlation, and a class random-intercept MixedLM (ML/CG; REML/lbfgs is singular with singleton classes).

## 4. Outcome definition

`WMC_t1` (subsequent Weighted Method Count). Predictors are measured at t. `WMC_t` is the lagged baseline. `delta_WMC` is not the outcome and is not called software corrosion.

## 5. Predictor representation

| Construct | Representation in Model A/B | Why |
| --- | --- | --- |
| DataClass | binary presence | count is nearly 0/1; construct is presence |
| LongMethod | binary presence | same |
| LongParameterList | binary presence | same |
| MutableStaticState | binary presence | sparse; still encoded as presence, with a no-MS sensitivity |
| LawOfDemeter | log1p(count) | binary is ~86% present; intensity varies |
| GodClass | **excluded** | WMC-contaminated PMD definition |
| CBO_t | **dropped** | 99.8% zeros in the modeling sample |

Controls retained besides `WMC_t`: LOC_t, NOM_t.

- Dropped CBO_t: 99.8331% of the modeling sample is 0 (threshold 95%); nunique=2.
- Retained LOC_t: VIF=1.72 (rule: drop only if VIF>10.0).
- Retained NOM_t: VIF=3.40 (rule: drop only if VIF>10.0).

## 6. Model A specification (primary)

Family: **OLS** with class-clustered SE and transition fixed effects.

`WMC_t1 ~ WMC_t + LOC_t + NOM_t + DataClass_present + LongMethod_present + LongParameterList_present + MutableStaticState_present + LawOfDemeter_log1p + transition FE`

## 7. Model B specification (primary)

Same as Model A plus estimable interactions:

`DataClass_present + LongMethod_present + LongParameterList_present + MutableStaticState_present + LawOfDemeter_log1p + DataClass_x_LongParameterList + DataClass_x_LoDlog + LongMethod_x_LoDlog + LongParameterList_x_LoDlog + MutableStaticState_x_LoDlog`

Main effects remain in the model whenever an interaction is included.

## 8. Treatment of repeated classes

Primary: cluster-robust SE at `class`. Robustness 1: Gaussian GEE, exchangeable correlation by class, same transition FE as OLS. Robustness 2: `MixedLM` random intercept by class with `year_index` (ML, CG optimizer). REML/lbfgs MixedLM with transition FE is singular because 512 classes appear only once.

## 9. Treatment of transitions

In-sample: transition fixed effects (four dummies after dropping the first). Forward validation cannot identify a future dummy, so OOS refits replace FE with numeric `year_index` for **both** A and B.

## 10. Treatment of LawOfDemeter

Primary uses `log1p(LawOfDemeter_count_t)`. Binary LoD is not used. Sensitivity: drop LoD main effect and all LoD interactions.

## 11. Treatment of sparse constructs

MutableStaticState is present in **34** modeling-sample rows. It is kept in the primary specification. Sensitivity `A_noMS` / `B_noMS` drops it and its interactions. It is not deleted after looking at p-values.

## 12. Interaction-estimability analysis

| interaction | both | A only | B only | neither | estimable | reason |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| DataClass_present x LongMethod_present | 1 | 154 | 40 | 2201 | False | both_present=1 < 20; not forced into Model B |
| DataClass_present x LongParameterList_present | 29 | 126 | 43 | 2198 | True | both_present=29 >= 20 |
| DataClass_present x MutableStaticState_present | 5 | 150 | 29 | 2212 | False | both_present=5 < 20; not forced into Model B |
| LongMethod_present x LongParameterList_present | 2 | 39 | 70 | 2285 | False | both_present=2 < 20; not forced into Model B |
| LongMethod_present x MutableStaticState_present | 0 | 41 | 34 | 2321 | False | both_present=0 < 20; not forced into Model B |
| LongParameterList_present x MutableStaticState_present | 0 | 72 | 34 | 2290 | False | both_present=0 < 20; not forced into Model B |
| DataClass_present x LawOfDemeter_log1p | 68 | 87 | 1985 | 256 | True | n(DataClass=1)=155, both_with_LoD_present=68, LoD unique values among DataClass=24 |
| LongMethod_present x LawOfDemeter_log1p | 40 | 1 | 2013 | 342 | True | n(LongMethod=1)=41, both_with_LoD_present=40, LoD unique values among LongMethod=30; a_only=1 so LongMethod main effect in the interaction model is weakly identified |
| LongParameterList_present x LawOfDemeter_log1p | 47 | 25 | 2006 | 318 | True | n(LongParameterList=1)=72, both_with_LoD_present=47, LoD unique values among LongParameterList=25 |
| MutableStaticState_present x LawOfDemeter_log1p | 31 | 3 | 2022 | 340 | True | n(MutableStaticState=1)=34, both_with_LoD_present=31, LoD unique values among MutableStaticState=13; a_only=3 so MutableStaticState main effect in the interaction model is weakly identified |

LongMethod × LoD and MutableStaticState × LoD meet the both-present threshold but have `a_only` of 1 and 3. They stay in the locked Model B (the support rule was not changed after seeing coefficients). The Large LongMethod main-effect shift in Model B is a symptom of that weak cell, not a finding about LongMethod.

## 13. Fit comparison

| model | n | AIC | BIC | in-sample RMSE | in-sample MAE | in-sample R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lag_controls_only` | 2396 | 11823.5 | 11869.8 | 2.8436 | 1.0929 | 0.9429 |
| `A_ols_cluster` | 2396 | 11828.6 | 11903.7 | 2.8407 | 1.1017 | 0.9430 |
| `B_ols_cluster` | 2396 | 11776.8 | 11880.9 | 2.8043 | 1.0949 | 0.9445 |
| `A_nb_cluster` | 2396 | 8069.3 | 8150.3 | 3.5593 | 1.2721 | 0.9106 |
| `B_nb_cluster` | 2396 | 9943.2 | 10053.1 | 2.9622 | 1.1563 | 0.9381 |
| `A_ols_noloD` | 2396 | 11829.0 | 11898.4 | 2.8421 | 1.0931 | 0.9430 |
| `B_ols_noloD` | 2396 | 11830.8 | 11906.0 | 2.8420 | 1.0929 | 0.9430 |
| `A_ols_noMS` | 2396 | 11826.7 | 11896.1 | 2.8407 | 1.1013 | 0.9430 |
| `B_ols_noMS` | 2396 | 11772.9 | 11865.4 | 2.8043 | 1.0948 | 0.9445 |
| `A_ols_mixed` | 2396 | 11832.6 | 11902.0 | 2.8068 | 1.0638 | 0.9444 |
| `B_ols_mixed` | 2396 | 11780.8 | 11879.1 | 2.8028 | 1.0729 | 0.9446 |
| `A_ols_gee` | 2396 | 11828.6 | 11903.8 | 2.8407 | 1.1019 | 0.9430 |
| `B_ols_gee` | 2396 | 11776.8 | 11880.9 | 2.8043 | 1.0948 | 0.9445 |

## 14. Out-of-sample comparison (forward chaining)

Train on earlier transitions, test on the next later transition. A class's later row is never in the training set of a fold that tests an earlier period.

| model | mean OOS RMSE | mean OOS MAE | mean OOS R² |
| --- | ---: | ---: | ---: |
| `A_ols_cluster` | 3.1842 | 1.1295 | 0.9044 |
| `B_ols_cluster` | 3.1797 | 1.1380 | 0.9043 |
| `A_ols_noloD` | 3.1812 | 1.1312 | 0.9045 |
| `B_ols_noloD` | 3.1817 | 1.1296 | 0.9045 |
| `A_ols_noMS` | 3.1809 | 1.1240 | 0.9047 |
| `B_ols_noMS` | 3.1711 | 1.1281 | 0.9048 |
| `A_nb_cluster` | 3.9872 | 1.3568 | 0.8499 |
| `B_nb_cluster` | 3.6271 | 1.3596 | 0.8684 |

Fold-level numbers: `signal_android_oos_folds.csv`.

## 15. Robustness analyses

| comparison | ΔAIC (B−A) | Δ in-sample RMSE (B−A) | Δ mean OOS RMSE (B−A) | verdict |
| --- | ---: | ---: | ---: | --- |
| primary_ols | -51.74 | -0.0364 | -0.0045 | B_better_on_both_in_sample_AIC_and_OOS_RMSE |
| sensitivity_no_LoD | 1.86 | -0.0001 | 0.0005 | A_better_or_equal_on_both |
| sensitivity_no_MutableStaticState | -53.81 | -0.0364 | -0.0098 | B_better_on_both_in_sample_AIC_and_OOS_RMSE |
| robustness_negative_binomial | 1873.87 | -0.5972 | -0.3602 | mixed_in_sample_vs_OOS |
| sensitivity_class_random_intercept | -51.83 | -0.0040 | NA | incomplete_metrics |
| sensitivity_gee_exchangeable | -51.77 | -0.0364 | NA | incomplete_metrics |

p-values are in the coefficient file for completeness. They were **not** used to add or drop interactions.

## 16. Interaction interpretation

An interaction coefficient is the difference in the WMC_t1 association of one smell variable across values of another, conditional on the model. It is not synergy, not a configuration proof, and not a causal effect.

### Model A coefficients (primary OLS, clustered SE)

| term | coef | SE (clustered) | p | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `const` | -0.7203 | 0.1405 | 0.0000 | [-0.9957, -0.4449] |
| `WMC_t` | 0.9637 | 0.0362 | 0.0000 | [0.8928, 1.0345] |
| `LOC_t` | 0.0026 | 0.0016 | 0.0993 | [-0.0005, 0.0057] |
| `NOM_t` | 0.2114 | 0.0712 | 0.0030 | [0.0719, 0.3510] |
| `DataClass_present` | -0.0925 | 0.3134 | 0.7678 | [-0.7068, 0.5217] |
| `LongMethod_present` | -0.1816 | 1.2036 | 0.8800 | [-2.5407, 2.1774] |
| `LongParameterList_present` | 0.5224 | 0.5537 | 0.3454 | [-0.5628, 1.6077] |
| `MutableStaticState_present` | -0.1472 | 0.2770 | 0.5952 | [-0.6901, 0.3957] |
| `LawOfDemeter_log1p` | 0.1048 | 0.1252 | 0.4024 | [-0.1406, 0.3502] |
| transition FE (4 dummies) | (omitted from display) |  |  |  |

### Model B coefficients (primary OLS, clustered SE)

| term | coef | SE (clustered) | p | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `const` | -0.5641 | 0.1185 | 0.0000 | [-0.7963, -0.3319] |
| `WMC_t` | 0.9812 | 0.0331 | 0.0000 | [0.9164, 1.0461] |
| `LOC_t` | 0.0016 | 0.0015 | 0.2809 | [-0.0013, 0.0046] |
| `NOM_t` | 0.1844 | 0.0605 | 0.0023 | [0.0658, 0.3031] |
| `DataClass_present` | -0.0865 | 0.2532 | 0.7325 | [-0.5828, 0.4097] |
| `LongMethod_present` | -7.8814 | 4.7838 | 0.0995 | [-17.2575, 1.4946] |
| `LongParameterList_present` | -1.2243 | 0.6783 | 0.0711 | [-2.5537, 0.1052] |
| `MutableStaticState_present` | -0.0659 | 0.3616 | 0.8553 | [-0.7746, 0.6427] |
| `LawOfDemeter_log1p` | 0.0602 | 0.1142 | 0.5983 | [-0.1637, 0.2841] |
| `DataClass_x_LongParameterList` | 0.8785 | 0.7616 | 0.2487 | [-0.6143, 2.3712] |
| `DataClass_x_LoDlog` | -0.0694 | 0.3475 | 0.8416 | [-0.7504, 0.6116] |
| `LongMethod_x_LoDlog` | 2.4412 | 1.4312 | 0.0881 | [-0.3640, 5.2464] |
| `LongParameterList_x_LoDlog` | 0.7757 | 0.5580 | 0.1645 | [-0.3179, 1.8694] |
| `MutableStaticState_x_LoDlog` | 0.0052 | 0.2682 | 0.9845 | [-0.5204, 0.5308] |
| transition FE (4 dummies) | (omitted from display) |  |  |  |

## 17. Limitations

- LongMethod × LoD uses 41 LongMethod rows of which 40 also have LoD; the LongMethod main effect in Model B is weakly identified.
- Sample is PMD-observed top-level classes only, one project (Signal Android).
- Five transitions; forward validation is coarse.
- Most four-smell pairs are not estimable (cells of 0–5).
- LoD intensity is correlated with LOC (size).
- OLS residuals are heavy-tailed; NB robustness uses a different lag functional form.
- CBO in this extract is effectively unused (almost all zeros).
- No causal identification strategy.

## 18. Incremental value of interactions?

Primary OLS verdict (pre-specified dual criterion: lower AIC **and** lower mean forward RMSE for B): **B_better_on_both_in_sample_AIC_and_OOS_RMSE**.

In-sample ΔAIC (B−A) = -51.74; ΔRMSE (B−A) = -0.0364; Δ mean OOS RMSE (B−A) = -0.0045.

### Direct answer

Within the Signal Android longitudinal sample for which PMD smell status is observed, explicitly modeling estimable smell interactions produced only a **tiny, non-robust** increment over individual-smell information. It should not be treated as evidence that interaction/configuration terms are generally informative, nor as a causal finding.

Complementary measures (not used to retune the specification):

- Mechanical dual criterion (lower AIC and lower mean OOS RMSE): **B_better_on_both_in_sample_AIC_and_OOS_RMSE** (ΔAIC=-51.74; Δ mean OOS RMSE=-0.0045).
- Mean forward MAE moved the other way (Δ MAE B−A = 0.0085).
- Forward RMSE is lower for B in some folds and higher in others (`signal_android_oos_folds.csv`).
- Dropping LawOfDemeter (main effect and LoD interactions) removes the in-sample AIC gain; `sensitivity_no_LoD` does not favor B.
- The only estimable four-smell pair (DataClass × LongParameterList) is not enough, on its own, to improve Model B.
- NegativeBinomial robustness does not agree with OLS on AIC (B much worse AIC, mixed OOS).
- Lagged WMC already explains most of WMC_t1 (in-sample R² ≈ 0.94 even in Model A). Smell terms, including interactions, are a small residual layer.

Therefore the thesis-ready answer for this project and sample is: **no reliable incremental value of the interaction representation beyond individual-smell information**. A mechanical RMSE/AIC tick is not the same as a stable scientific increment.

This is not a claim about all software systems.

What was estimable: see Section 12.
What was not estimable: binary pairs with both_present < 20, including LongMethod×MutableStaticState (0) and LongParameterList×MutableStaticState (0).
What was too sparse: MutableStaticState (few dozen positives); several interaction cells.
What depended on LawOfDemeter: compare `primary_ols` vs `sensitivity_no_LoD`.
What depended on modeling family: compare `primary_ols` vs `robustness_negative_binomial`.
What depended on repeated-class treatment: compare clustered OLS vs GEE vs MixedLM.
Before a thesis-wide experiment: replicate the same locked rules on other CSIQ projects; do not expand until this Signal pipeline is accepted.

## Fit failures

None.
