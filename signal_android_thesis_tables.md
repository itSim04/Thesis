# Signal Android thesis tables

Generated from `run_signal_android_models.py`. Variable names match the CSV outputs.

## A. Modeling sample by transition

| version_t | version_t1 | n | unique classes |
| --- | --- | ---: | ---: |
| 2016-1 | 2017-1 | 405 | 405 |
| 2017-1 | 2018-1 | 349 | 349 |
| 2018-1 | 2019-1 | 419 | 419 |
| 2019-1 | 2020-1 | 450 | 450 |
| 2020-1 | 2021-1 | 773 | 773 |

Total n=2396; unique classes=1018.

## B. Predictor prevalence (modeling sample)

| variable | zero % | positive % | mean | median | SD | p75 | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `DataClass_count_t` | 93.53% | 6.47% | 0.069 | 0.000 | 0.281 | 0.000 | 0.000 | 1.000 | 4.0 |
| `LongMethod_count_t` | 98.29% | 1.71% | 0.022 | 0.000 | 0.191 | 0.000 | 0.000 | 0.000 | 5.0 |
| `LongParameterList_count_t` | 96.99% | 3.01% | 0.031 | 0.000 | 0.181 | 0.000 | 0.000 | 0.000 | 2.0 |
| `MutableStaticState_count_t` | 98.58% | 1.42% | 0.020 | 0.000 | 0.196 | 0.000 | 0.000 | 0.000 | 4.0 |
| `LawOfDemeter_count_t` | 14.32% | 85.68% | 9.390 | 4.000 | 20.091 | 9.000 | 21.000 | 35.250 | 419.0 |
| `WMC_t` | 31.68% | 68.32% | 5.450 | 2.000 | 10.778 | 7.000 | 14.000 | 21.000 | 192.0 |
| `WMC_t1` | 30.26% | 69.74% | 5.935 | 2.000 | 11.905 | 7.000 | 15.000 | 24.000 | 242.0 |
| `LOC_t` | 0.00% | 100.00% | 93.636 | 55.000 | 129.476 | 108.000 | 186.000 | 300.000 | 1898.0 |
| `CBO_t` | 99.83% | 0.17% | 0.002 | 0.000 | 0.041 | 0.000 | 0.000 | 0.000 | 1.0 |
| `NOM_t` | 34.77% | 65.23% | 2.927 | 2.000 | 5.039 | 3.000 | 7.000 | 11.000 | 80.0 |
| `LawOfDemeter_log1p` | 14.32% | 85.68% | 1.647 | 1.609 | 1.103 | 2.303 | 3.091 | 3.590 | 6.0 |

Binary presence in the modeling sample: DataClass=155; LongMethod=41; LongParameterList=72; MutableStaticState=34; LawOfDemeter present=2053 (85.68%).

## C. Model A coefficients (primary OLS, class-clustered SE)

| term | coef | SE | p | 95% CI |
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
| `tr_2017-1->2018-1` | 0.1640 | 0.1650 | 0.3202 | [-0.1594, 0.4874] |
| `tr_2018-1->2019-1` | 0.4821 | 0.1623 | 0.0030 | [0.1640, 0.8003] |
| `tr_2019-1->2020-1` | 0.1742 | 0.1702 | 0.3059 | [-0.1593, 0.5078] |
| `tr_2020-1->2021-1` | 0.6840 | 0.1373 | 0.0000 | [0.4150, 0.9531] |

## D. Model B coefficients (primary OLS, class-clustered SE)

| term | coef | SE | p | 95% CI |
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
| `tr_2017-1->2018-1` | 0.1673 | 0.1632 | 0.3053 | [-0.1525, 0.4870] |
| `tr_2018-1->2019-1` | 0.4869 | 0.1623 | 0.0027 | [0.1687, 0.8051] |
| `tr_2019-1->2020-1` | 0.1936 | 0.1680 | 0.2493 | [-0.1358, 0.5229] |
| `tr_2020-1->2021-1` | 0.6601 | 0.1278 | 0.0000 | [0.4095, 0.9106] |

## E. Model A vs Model B comparison

| comparison | n | ΔAIC (B−A) | ΔBIC (B−A) | Δ RMSE (B−A) | Δ MAE (B−A) | Δ OOS RMSE (B−A) | Δ OOS MAE (B−A) | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| primary_ols | 2396 | -51.74 | -22.83 | -0.0364 | -0.0067 | -0.0045 | 0.0085 | B_better_on_both_in_sample_AIC_and_OOS_RMSE |
| sensitivity_no_LoD | 2396 | 1.86 | 7.65 | -0.0001 | -0.0002 | 0.0005 | -0.0016 | A_better_or_equal_on_both |
| sensitivity_no_MutableStaticState | 2396 | -53.81 | -30.69 | -0.0364 | -0.0065 | -0.0098 | 0.0041 | B_better_on_both_in_sample_AIC_and_OOS_RMSE |
| robustness_negative_binomial | 2396 | 1873.87 | 1902.78 | -0.5972 | -0.1158 | -0.3602 | 0.0028 | mixed_in_sample_vs_OOS |
| sensitivity_class_random_intercept | 2396 | -51.83 | -22.93 | -0.0040 | 0.0091 | NA | NA | incomplete_metrics |
| sensitivity_gee_exchangeable | 2396 | -51.77 | -22.86 | -0.0364 | -0.0071 | NA | NA | incomplete_metrics |

Pre-specified dual criterion for claiming incremental value: lower AIC **and** lower mean forward RMSE. Complementary MAE/OOS-R²/LoD-off/family checks are reported and are not used to retune terms.

## F. Interaction support / available pairs

| interaction | type | both | A only | B only | neither | estimable | weak main-effect cell |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| DataClass_present x LongMethod_present | binary_x_binary | 1 | 154 | 40 | 2201 | False | False |
| DataClass_present x LongParameterList_present | binary_x_binary | 29 | 126 | 43 | 2198 | True | False |
| DataClass_present x MutableStaticState_present | binary_x_binary | 5 | 150 | 29 | 2212 | False | False |
| LongMethod_present x LongParameterList_present | binary_x_binary | 2 | 39 | 70 | 2285 | False | False |
| LongMethod_present x MutableStaticState_present | binary_x_binary | 0 | 41 | 34 | 2321 | False | False |
| LongParameterList_present x MutableStaticState_present | binary_x_binary | 0 | 72 | 34 | 2290 | False | False |
| DataClass_present x LawOfDemeter_log1p | binary_x_lod_intensity | 68 | 87 | 1985 | 256 | True | False |
| LongMethod_present x LawOfDemeter_log1p | binary_x_lod_intensity | 40 | 1 | 2013 | 342 | True | True |
| LongParameterList_present x LawOfDemeter_log1p | binary_x_lod_intensity | 47 | 25 | 2006 | 318 | True | False |
| MutableStaticState_present x LawOfDemeter_log1p | binary_x_lod_intensity | 31 | 3 | 2022 | 340 | True | True |

## G. Robustness results

| model | family | n | AIC | in-sample RMSE | mean OOS RMSE | mean OOS MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A_ols_cluster` | ols | 2396 | 11828.6 | 2.8407 | 3.1842 | 1.1295 |
| `B_ols_cluster` | ols | 2396 | 11776.8 | 2.8043 | 3.1797 | 1.1380 |
| `A_ols_noloD` | ols | 2396 | 11829.0 | 2.8421 | 3.1812 | 1.1312 |
| `B_ols_noloD` | ols | 2396 | 11830.8 | 2.8420 | 3.1817 | 1.1296 |
| `A_ols_noMS` | ols | 2396 | 11826.7 | 2.8407 | 3.1809 | 1.1240 |
| `B_ols_noMS` | ols | 2396 | 11772.9 | 2.8043 | 3.1711 | 1.1281 |
| `A_nb_cluster` | nb | 2396 | 8069.3 | 3.5593 | 3.9872 | 1.3568 |
| `B_nb_cluster` | nb | 2396 | 9943.2 | 2.9622 | 3.6271 | 1.3596 |
| `A_ols_mixed` | ols_mixed | 2396 | 11832.6 | 2.8068 | NA | NA |
| `B_ols_mixed` | ols_mixed | 2396 | 11780.8 | 2.8028 | NA | NA |
| `A_ols_gee` | ols_gee | 2396 | 11828.6 | 2.8407 | NA | NA |
| `B_ols_gee` | ols_gee | 2396 | 11776.8 | 2.8043 | NA | NA |
| `lag_controls_only` | ols | 2396 | 11823.5 | 2.8436 | NA | NA |
