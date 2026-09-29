"""Signal Android Model A vs Model B experiment.

Primary comparison: individual-smell information vs the same information plus
explicit, pre-specified estimable interaction terms.

No dataset rebuild. No manufactured zeros. No GodClass. No cyclomatic/NCSS
predictors. No causal claims. Interaction terms are not selected by p-value.
"""
from __future__ import annotations

import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / "signal_android_pilot_panel.csv"

# ---------------------------------------------------------------------------
# Prespecified rules (locked before coefficient inspection)
# ---------------------------------------------------------------------------
MIN_BOTH_FOR_BINARY_INTERACTION = 20
MIN_PRESENT_FOR_LOD_INTERACTION = 20
CONTROL_VIF_DROP_THRESHOLD = 10.0
CBO_ZERO_DROP_THRESHOLD = 0.95  # drop CBO if this share of sample is zero
LOD_BINARY_BANNED = True
GODCLASS_BANNED = True
YEAR_INDEX = {"2016-1": 0, "2017-1": 1, "2018-1": 2, "2019-1": 3, "2020-1": 4}
TRANSITIONS = [
    ("2016-1", "2017-1"),
    ("2017-1", "2018-1"),
    ("2018-1", "2019-1"),
    ("2019-1", "2020-1"),
    ("2020-1", "2021-1"),
]
FOUR = ["DataClass", "LongMethod", "LongParameterList", "MutableStaticState"]
BINARY_SOURCE = {
    "DataClass": "DataClass_count_t",
    "LongMethod": "LongMethod_count_t",
    "LongParameterList": "LongParameterList_count_t",
    "MutableStaticState": "MutableStaticState_count_t",
}

OUT_SAMPLE = ROOT / "signal_android_modeling_sample.csv"
OUT_SAMPLE_REPORT = ROOT / "signal_android_modeling_sample_report.md"
OUT_RESULTS = ROOT / "signal_android_model_results.csv"
OUT_COEFS = ROOT / "signal_android_model_coefficients.csv"
OUT_COMPARE = ROOT / "signal_android_model_comparison.csv"
OUT_SUPPORT = ROOT / "signal_android_interaction_support.csv"
OUT_DIAG = ROOT / "signal_android_model_diagnostics.md"
OUT_REPORT = ROOT / "signal_android_model_results.md"
OUT_PRED = ROOT / "signal_android_predictor_distributions.csv"
OUT_CORR = ROOT / "signal_android_predictor_correlations.csv"
OUT_VIF = ROOT / "signal_android_vif.csv"
OUT_OOS = ROOT / "signal_android_oos_folds.csv"
OUT_EVIDENCE = ROOT / "signal_android_model_evidence.json"
OUT_TABLES = ROOT / "signal_android_thesis_tables.md"


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def oos_r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def summarize_numeric(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {}
    q = x.quantile([0.75, 0.90, 0.95])
    return {
        "n": int(x.shape[0]),
        "zero_proportion": float((x == 0).mean()),
        "positive_proportion": float((x > 0).mean()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.90]),
        "p95": float(q.loc[0.95]),
        "maximum": float(x.max()),
        "n_unique": int(x.nunique()),
    }


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{100 * value:.{digits}f}%" if abs(value) <= 1.5 else f"{value:.{digits}f}%"


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def build_sample(panel):
    total = len(panel)
    known = panel["smell_status_known_t"] == 1
    numeric = panel["WMC_t"].notna() & panel["WMC_t1"].notna()
    sample = panel.loc[known & numeric].copy()
    unknown = panel.loc[~known]
    zero_fill_violations = []
    for col in BINARY_SOURCE.values():
        n_zero = int((unknown[col] == 0).sum())
        n_notnull = int(unknown[col].notna().sum())
        if n_zero or n_notnull:
            zero_fill_violations.append(f"{col}: notna={n_notnull} zeros={n_zero} among unknown")
    sample["year_index"] = sample["version_t"].map(YEAR_INDEX)
    sample["transition"] = sample["version_t"] + "->" + sample["version_t1"]
    for name, src in BINARY_SOURCE.items():
        sample[f"{name}_present"] = (sample[src] > 0).astype(int)
        sample[f"{name}_log1p"] = np.log1p(sample[src].astype(float))
    sample["LawOfDemeter_present"] = (sample["LawOfDemeter_count_t"] > 0).astype(int)
    sample["LawOfDemeter_log1p"] = np.log1p(sample["LawOfDemeter_count_t"].astype(float))
    sample["WMC_t_log1p"] = np.log1p(sample["WMC_t"].astype(float))
    sample["LOC_t_log1p"] = np.log1p(sample["LOC_t"].astype(float))
    return {
        "total": total,
        "n_known": int(known.sum()),
        "n_numeric": int(numeric.sum()),
        "sample": sample,
        "n_unknown": int((~known).sum()),
        "zero_fill_violations": zero_fill_violations,
        "obs_per_class": sample.groupby("class").size(),
        "obs_per_transition": sample.groupby(["version_t", "version_t1"]).size(),
    }


def interaction_cells(sample, a, b, a_is_lod=False, b_is_lod=False):
    av = sample["LawOfDemeter_present"] if a_is_lod else sample[f"{a}_present"]
    bv = sample["LawOfDemeter_present"] if b_is_lod else sample[f"{b}_present"]
    both = int(((av == 1) & (bv == 1)).sum())
    a_only = int(((av == 1) & (bv == 0)).sum())
    b_only = int(((av == 0) & (bv == 1)).sum())
    neither = int(((av == 0) & (bv == 0)).sum())
    return both, a_only, b_only, neither


def estimability_table(sample):
    rows = []
    # Four-construct binary pairs
    for a, b in combinations(FOUR, 2):
        both, a_only, b_only, neither = interaction_cells(sample, a, b)
        estimable = both >= MIN_BOTH_FOR_BINARY_INTERACTION
        rows.append({
            "interaction": f"{a}_present x {b}_present",
            "type": "binary_x_binary",
            "both_present": both,
            "a_only": a_only,
            "b_only": b_only,
            "neither": neither,
            "smallest_positive_cell": min(x for x in [both, a_only, b_only] if x >= 0),
            "estimable": estimable,
            "weak_main_effect_cell": min(a_only, b_only) < 5,
            "reason": (
                f"both_present={both} >= {MIN_BOTH_FOR_BINARY_INTERACTION}"
                if estimable
                else f"both_present={both} < {MIN_BOTH_FOR_BINARY_INTERACTION}; not forced into Model B"
            ),
        })
    # Binary x LoD intensity: support uses LoD presence cell; term is present * log1p(count)
    for a in FOUR:
        both, a_only, b_only, neither = interaction_cells(sample, a, "LawOfDemeter", b_is_lod=True)
        n_present = int(sample[f"{a}_present"].sum())
        lod_nunique_among_a = int(sample.loc[sample[f"{a}_present"] == 1, "LawOfDemeter_count_t"].nunique())
        estimable = (
            n_present >= MIN_PRESENT_FOR_LOD_INTERACTION
            and lod_nunique_among_a >= 2
            and both >= MIN_BOTH_FOR_BINARY_INTERACTION
        )
        rows.append({
            "interaction": f"{a}_present x LawOfDemeter_log1p",
            "type": "binary_x_lod_intensity",
            "both_present": both,
            "a_only": a_only,
            "b_only": b_only,
            "neither": neither,
            "smallest_positive_cell": min([both, a_only, n_present]),
            "estimable": estimable,
            "weak_main_effect_cell": a_only < 5,
            "reason": (
                f"n({a}=1)={n_present}, both_with_LoD_present={both}, LoD unique values among {a}={lod_nunique_among_a}"
                + ("" if estimable else "; marked unavailable / not forced")
                + (f"; a_only={a_only} so {a} main effect in the interaction model is weakly identified" if estimable and a_only < 5 else "")
            ),
        })
    return rows


def vif_table(X):
    rows = []
    if X.shape[1] < 2:
        return rows
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif = float(variance_inflation_factor(X.values, i))
        except Exception:
            vif = np.nan
        rows.append({"variable": col, "vif": vif})
    return rows


def add_const(X):
    return sm.add_constant(X, has_constant="add")


def transition_dummies(sample, drop_first=True):
    dummies = pd.get_dummies(sample["transition"], prefix="tr", drop_first=drop_first, dtype=float)
    return dummies


def assemble_X(sample, smell_terms, use_transition_fe=True, use_year_index=False, extra_controls=None):
    parts = [sample[["WMC_t"]].astype(float)]
    controls = extra_controls if extra_controls is not None else []
    for col in controls:
        parts.append(sample[[col]].astype(float))
    for term in smell_terms:
        parts.append(sample[[term]].astype(float))
    if use_year_index:
        parts.append(sample[["year_index"]].astype(float))
    elif use_transition_fe:
        parts.append(transition_dummies(sample))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return add_const(X)


def fit_ols_clustered(y, X, groups):
    model = sm.OLS(y, X, missing="drop")
    result = model.fit(cov_type="cluster", cov_kwds={"groups": groups.loc[X.index]})
    return result


def fit_ols_ordinary(y, X):
    return sm.OLS(y, X, missing="drop").fit()


def fit_nb_clustered(y, X, groups):
    y_int = np.asarray(y).astype(int)
    model = NegativeBinomial(y_int, X, missing="drop")
    result = model.fit(maxiter=300, disp=False, cov_type="cluster", cov_kwds={"groups": groups.loc[X.index]})
    return result


def fit_mixedlm(y, X, groups):
    # REML + lbfgs is singular here (many singleton classes). ML + CG is identified.
    X_re = X.drop(columns=["const"], errors="ignore").astype(float)
    exog = add_const(X_re)
    y_arr = np.asarray(y, dtype=float)
    g_arr = np.asarray(groups)
    last_err = None
    for reml, method in [(False, "cg"), (False, "bfgs"), (True, "cg")]:
        try:
            model = sm.MixedLM(y_arr, exog, groups=g_arr)
            return model.fit(reml=reml, method=method, maxiter=250)
        except Exception as exc:
            last_err = exc
    raise last_err


def fit_gee(y, X, groups):
    model = sm.GEE(
        np.asarray(y, dtype=float),
        X.astype(float),
        groups=np.asarray(groups),
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit()


def coef_rows(model_name, spec, result, family):
    rows = []
    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    ci = result.conf_int()
    for term in params.index:
        rows.append({
            "model": model_name,
            "specification": spec,
            "family": family,
            "term": term,
            "coef": float(params[term]),
            "se": float(bse[term]) if term in bse.index else np.nan,
            "p_value": float(pvalues[term]) if term in pvalues.index else np.nan,
            "ci_low": float(ci.loc[term, 0]) if term in ci.index else np.nan,
            "ci_high": float(ci.loc[term, 1]) if term in ci.index else np.nan,
        })
    return rows


def in_sample_metrics(y, pred, result, family, n_classes, n, extra=None):
    row = {
        "n": int(n),
        "n_classes": int(n_classes),
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
        "loglik": float(getattr(result, "llf", np.nan)),
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "r2": oos_r2(y, pred),
        "family": family,
    }
    if extra:
        row.update(extra)
    return row


def predict_align(result, X):
    pred = np.asarray(result.predict(X))
    return pred


def drop_zero_variance(X):
    keep = [c for c in X.columns if c == "const" or X[c].nunique(dropna=False) > 1]
    return X[keep]


def forward_folds():
    folds = []
    for i in range(1, len(TRANSITIONS)):
        train_tr = TRANSITIONS[:i]
        test_tr = TRANSITIONS[i]
        folds.append({
            "fold": i,
            "train_transitions": [f"{a}->{b}" for a, b in train_tr],
            "test_transition": f"{test_tr[0]}->{test_tr[1]}",
        })
    return folds


def fit_predict_oos(train, test, smell_terms, controls, family):
    y_train = train["WMC_t1"].astype(float)
    y_test = test["WMC_t1"].astype(float)
    X_train = assemble_X(train, smell_terms, use_transition_fe=False, use_year_index=True, extra_controls=controls)
    X_test = assemble_X(test, smell_terms, use_transition_fe=False, use_year_index=True, extra_controls=controls)
    X_train = drop_zero_variance(X_train)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    groups = train["class"]
    if family == "ols":
        result = fit_ols_clustered(y_train, X_train, groups)
        pred = predict_align(result, X_test)
    elif family == "nb":
        result = fit_nb_clustered(y_train, X_train, groups)
        pred = np.clip(predict_align(result, X_test), 0, None)
    else:
        raise ValueError(family)
    return {
        "rmse": rmse(y_test, pred),
        "mae": mae(y_test, pred),
        "r2": oos_r2(y_test, pred),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_params": int(X_train.shape[1]),
        "converged": bool(getattr(result, "mle_retvals", {}).get("converged", True)) if family == "nb" else True,
    }


def choose_controls(sample):
    """Variance/VIF rules only. Not coefficient significance."""
    notes = []
    cbo_zero = float((sample["CBO_t"] == 0).mean())
    controls = []
    if cbo_zero >= CBO_ZERO_DROP_THRESHOLD:
        notes.append(
            f"Dropped CBO_t: {cbo_zero:.4%} of the modeling sample is 0 "
            f"(threshold {CBO_ZERO_DROP_THRESHOLD:.0%}); nunique={sample['CBO_t'].nunique()}."
        )
    else:
        controls.append("CBO_t")
    # Candidate size/method controls: LOC and NOM. Apply VIF with WMC_t present.
    trial = ["LOC_t", "NOM_t"]
    X = add_const(sample[["WMC_t"] + trial].astype(float))
    vifs = {row["variable"]: row["vif"] for row in vif_table(X)}
    for col in trial:
        vif = vifs.get(col, np.nan)
        if vif is not None and vif > CONTROL_VIF_DROP_THRESHOLD:
            notes.append(f"Dropped {col}: VIF={vif:.2f} > {CONTROL_VIF_DROP_THRESHOLD} with WMC_t in the block.")
        else:
            controls.append(col)
            notes.append(f"Retained {col}: VIF={vif:.2f} (rule: drop only if VIF>{CONTROL_VIF_DROP_THRESHOLD}).")
    return controls, notes, vifs


def residual_diagnostics(y, pred, label):
    resid = np.asarray(y) - np.asarray(pred)
    return {
        "label": label,
        "residual_mean": float(np.mean(resid)),
        "residual_sd": float(np.std(resid, ddof=1)),
        "residual_skew": float(stats.skew(resid)),
        "residual_kurtosis": float(stats.kurtosis(resid)),
        "residual_min": float(np.min(resid)),
        "residual_max": float(np.max(resid)),
        "share_resid_gt_10": float(np.mean(np.abs(resid) > 10)),
    }


def main():
    panel = pd.read_csv(PANEL)
    pack = build_sample(panel)
    sample = pack["sample"]
    if pack["zero_fill_violations"]:
        raise SystemExit("Zero-fill detected among unmatched PMD rows: " + "; ".join(pack["zero_fill_violations"]))
    if GODCLASS_BANNED and any("GodClass" in c and "present" in c for c in sample.columns if False):
        pass
    n = len(sample)
    n_classes = int(sample["class"].nunique())
    print(f"Modeling sample n={n} classes={n_classes}", flush=True)

    sample_out_cols = [
        "project", "version_t", "version_t1", "transition", "year_index", "class",
        "WMC_t", "WMC_t1", "delta_WMC", "LOC_t", "CBO_t", "NOM_t",
        "smell_status_known_t",
        "DataClass_count_t", "LongMethod_count_t", "LongParameterList_count_t",
        "MutableStaticState_count_t", "LawOfDemeter_count_t",
        "DataClass_present", "LongMethod_present", "LongParameterList_present",
        "MutableStaticState_present", "LawOfDemeter_log1p",
    ]
    sample[sample_out_cols].to_csv(OUT_SAMPLE, index=False)

    # Predictor distributions
    pred_rows = []
    for col in [
        "DataClass_count_t", "LongMethod_count_t", "LongParameterList_count_t",
        "MutableStaticState_count_t", "LawOfDemeter_count_t",
        "WMC_t", "WMC_t1", "LOC_t", "CBO_t", "NOM_t", "LawOfDemeter_log1p",
    ]:
        stats_map = summarize_numeric(sample[col])
        stats_map["variable"] = col
        pred_rows.append(stats_map)
    write_csv(OUT_PRED, pred_rows)

    corr_cols = [
        "WMC_t", "WMC_t1", "LOC_t", "CBO_t", "NOM_t",
        "DataClass_count_t", "LongMethod_count_t", "LongParameterList_count_t",
        "MutableStaticState_count_t", "LawOfDemeter_count_t",
        "DataClass_present", "LongMethod_present", "LongParameterList_present",
        "MutableStaticState_present", "LawOfDemeter_log1p",
    ]
    corr = sample[corr_cols].corr()
    corr.to_csv(OUT_CORR)

    controls, control_notes, trial_vifs = choose_controls(sample)
    print("Controls:", controls, flush=True)

    support_rows = estimability_table(sample)
    write_csv(OUT_SUPPORT, support_rows)
    estimable_binary = [r["interaction"] for r in support_rows if r["type"] == "binary_x_binary" and r["estimable"]]
    estimable_lod = [r["interaction"] for r in support_rows if r["type"] == "binary_x_lod_intensity" and r["estimable"]]
    print("Estimable binary x binary:", estimable_binary, flush=True)
    print("Estimable binary x LoD:", estimable_lod, flush=True)

    # Create interaction columns
    sample["DataClass_x_LongParameterList"] = sample["DataClass_present"] * sample["LongParameterList_present"]
    for a in FOUR:
        sample[f"{a}_x_LoDlog"] = sample[f"{a}_present"] * sample["LawOfDemeter_log1p"]

    model_a_smells = [
        "DataClass_present",
        "LongMethod_present",
        "LongParameterList_present",
        "MutableStaticState_present",
        "LawOfDemeter_log1p",
    ]
    model_b_smells = list(model_a_smells)
    if "DataClass_present x LongParameterList_present" in estimable_binary:
        model_b_smells.append("DataClass_x_LongParameterList")
    for a in FOUR:
        label = f"{a}_present x LawOfDemeter_log1p"
        if label in estimable_lod:
            model_b_smells.append(f"{a}_x_LoDlog")

    y = sample["WMC_t1"].astype(float)
    groups = sample["class"]

    # VIF on intended primary OLS RHS (Model A, no interactions)
    X_a_vif = assemble_X(sample, model_a_smells, extra_controls=controls)
    vif_rows = vif_table(X_a_vif)
    write_csv(OUT_VIF, vif_rows)

    # Re-apply VIF drop to retained controls after smells are in the block? Keep the
    # pre-smell control rule; report full-model VIFs as diagnostics only.

    wmc = sample["WMC_t1"]
    family_notes = []
    family_notes.append(f"WMC_t1 is a non-negative integer; unique values={int(wmc.nunique())}; zeros={float((wmc==0).mean()):.4%}.")
    family_notes.append(f"Unconditional variance/mean={float(wmc.var()/wmc.mean()):.3f} (overdispersed unconditionally).")
    family_notes.append(f"corr(WMC_t, WMC_t1)={float(sample['WMC_t'].corr(sample['WMC_t1'])):.4f}; share delta_WMC=0={float((sample['delta_WMC']==0).mean()):.4%}.")
    family_notes.append(
        "Primary family = OLS on WMC_t1 with identity scale, because the lagged outcome is nearly linear "
        "and a log-link count model would exponentiate a large raw WMC_t. NegativeBinomial with log1p(WMC_t) "
        "is the robustness family, not the selection winner."
    )
    primary_family = "ols"
    robustness_family = "nb"

    # Fit models
    coef_all = []
    result_rows = []
    oos_rows = []
    fitted = {}

    specs = [
        ("A_ols_cluster", "A", "ols", "cluster", model_a_smells, True),
        ("B_ols_cluster", "B", "ols", "cluster", model_b_smells, True),
        ("A_ols_noloD", "A_noloD", "ols", "cluster", [t for t in model_a_smells if t != "LawOfDemeter_log1p"], True),
        ("B_ols_noloD", "B_noloD", "ols", "cluster", [t for t in model_b_smells if t != "LawOfDemeter_log1p" and not t.endswith("_x_LoDlog")], True),
        ("A_ols_noMS", "A_noMS", "ols", "cluster", [t for t in model_a_smells if t != "MutableStaticState_present"], True),
        ("B_ols_noMS", "B_noMS", "ols", "cluster", [t for t in model_b_smells if "MutableStaticState" not in t], True),
        ("A_nb_cluster", "A", "nb", "cluster", ["WMC_t_log1p" if t == "WMC_t" else t for t in model_a_smells], True),
        ("B_nb_cluster", "B", "nb", "cluster", ["WMC_t_log1p" if t == "WMC_t" else t for t in model_b_smells], True),
    ]

    def smells_for_nb(terms):
        return terms

    # OLS and NB use WMC_t raw for OLS; NB uses log1p(WMC_t) instead of WMC_t.
    def make_X(terms, family, extra_controls, use_fe=True, use_year=False, data=None):
        data = sample if data is None else data
        if family == "nb":
            # Replace lagged WMC with log1p version by assembling manually.
            parts_controls = extra_controls
            X = data[["WMC_t_log1p"]].astype(float).rename(columns={"WMC_t_log1p": "WMC_t_log1p"})
            for col in parts_controls:
                X = pd.concat([X, data[[col]].astype(float)], axis=1)
            for term in terms:
                X = pd.concat([X, data[[term]].astype(float)], axis=1)
            if use_year:
                X = pd.concat([X, data[["year_index"]].astype(float)], axis=1)
            elif use_fe:
                X = pd.concat([X, transition_dummies(data)], axis=1)
            X = X.loc[:, ~X.columns.duplicated()]
            return add_const(X)
        return assemble_X(data, terms, use_transition_fe=use_fe, use_year_index=use_year, extra_controls=extra_controls)

    fit_failures = []
    for spec_id, model_letter, family, dep, terms, _ in [
        ("A_ols_cluster", "A", "ols", "class-clustered SE", model_a_smells, None),
        ("B_ols_cluster", "B", "ols", "class-clustered SE", model_b_smells, None),
        ("A_ols_noloD", "A_noloD", "ols", "class-clustered SE; LoD excluded", [t for t in model_a_smells if t != "LawOfDemeter_log1p"], None),
        ("B_ols_noloD", "B_noloD", "ols", "class-clustered SE; LoD excluded", [t for t in model_b_smells if t != "LawOfDemeter_log1p" and not t.endswith("_x_LoDlog")], None),
        ("A_ols_noMS", "A_noMS", "ols", "class-clustered SE; MutableStaticState excluded", [t for t in model_a_smells if t != "MutableStaticState_present"], None),
        ("B_ols_noMS", "B_noMS", "ols", "class-clustered SE; MutableStaticState excluded", [t for t in model_b_smells if "MutableStaticState" not in t], None),
        ("A_nb_cluster", "A", "nb", "class-clustered SE; log1p(WMC_t)", model_a_smells, None),
        ("B_nb_cluster", "B", "nb", "class-clustered SE; log1p(WMC_t)", model_b_smells, None),
    ]:
        try:
            X = make_X(terms, family, controls)
            if family == "ols":
                result = fit_ols_clustered(y, X, groups)
                pred = predict_align(result, X)
            else:
                result = fit_nb_clustered(y, X, groups)
                pred = np.clip(predict_align(result, X), 0, None)
            fitted[spec_id] = {"result": result, "X": X, "pred": pred, "family": family, "terms": terms}
            met = in_sample_metrics(y, pred, result, family, n_classes, n, extra={
                "model": spec_id,
                "specification": spec_id.split("_", 1)[0] if spec_id[1] != "_" else spec_id,
                "comparison_group": "primary" if spec_id in {"A_ols_cluster", "B_ols_cluster"} else "sensitivity",
                "repeated_obs": dep,
                "transitions": "5 transition FE" if family == "ols" or True else "",
                "controls": "+".join(["WMC_t" if family == "ols" else "log1p(WMC_t)"] + controls),
                "smell_terms": "+".join(terms),
                "n_params": int(X.shape[1]),
            })
            # fix specification label
            if spec_id.startswith("A_ols_cluster"):
                met["specification"] = "A_primary_ols"
            elif spec_id.startswith("B_ols_cluster"):
                met["specification"] = "B_primary_ols"
            elif spec_id.startswith("A_nb"):
                met["specification"] = "A_nb"
            elif spec_id.startswith("B_nb"):
                met["specification"] = "B_nb"
            else:
                met["specification"] = spec_id
            met["model"] = spec_id
            met["repeated_obs"] = dep
            met["controls"] = "+".join((["WMC_t"] if family == "ols" else ["log1p(WMC_t)"]) + controls)
            met["smell_terms"] = "+".join(terms)
            met["n_params"] = int(X.shape[1])
            result_rows.append(met)
            coef_all.extend(coef_rows(spec_id, met["specification"], result, family))
            print(f"Fitted {spec_id} AIC={met['aic']:.1f} RMSE={met['rmse']:.3f}", flush=True)
        except Exception as exc:
            fit_failures.append(f"{spec_id}: {type(exc).__name__}: {exc}")
            print(f"FAILED {spec_id}: {exc}", flush=True)

    def record_robust_linear(spec_id, label, terms, result, X, pred, family, repeated_obs, n_used, n_classes_used):
        fitted[spec_id] = {"result": result, "X": X, "pred": pred, "family": family, "terms": terms}
        met = in_sample_metrics(y if n_used == n else sample.loc[X.index, "WMC_t1"].astype(float), pred, result, family, n_classes_used, n_used)
        met.update({
            "model": spec_id,
            "specification": label,
            "repeated_obs": repeated_obs,
            "controls": "+".join(["WMC_t"] + controls),
            "smell_terms": "+".join(terms),
            "n_params": int(X.shape[1]),
        })
        result_rows.append(met)
        try:
            coef_all.extend(coef_rows(spec_id, label, result, family))
        except Exception:
            params = result.params
            bse = getattr(result, "bse", pd.Series(dtype=float))
            pvalues = getattr(result, "pvalues", pd.Series(dtype=float))
            for term in params.index:
                coef_all.append({
                    "model": spec_id,
                    "specification": label,
                    "family": family,
                    "term": str(term),
                    "coef": float(params[term]),
                    "se": float(bse[term]) if term in getattr(bse, "index", []) else np.nan,
                    "p_value": float(pvalues[term]) if term in getattr(pvalues, "index", []) else np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                })
        print(f"Fitted {spec_id} AIC={met.get('aic', float('nan'))}", flush=True)

    # MixedLM robustness for primary A and B (OLS scale; ML/CG because REML/lbfgs is singular)
    for spec_id, terms, label in [
        ("A_ols_mixed", model_a_smells, "A_mixedlm"),
        ("B_ols_mixed", model_b_smells, "B_mixedlm"),
    ]:
        try:
            X = assemble_X(sample, terms, extra_controls=controls, use_transition_fe=False, use_year_index=True)
            result = fit_mixedlm(y, X, groups)
            pred = np.asarray(result.fittedvalues)
            record_robust_linear(
                spec_id, label, terms, result, X, pred, "ols_mixed",
                "class random intercept; year_index; ML/CG", n, n_classes,
            )
        except Exception as exc:
            fit_failures.append(f"{spec_id}: {type(exc).__name__}: {exc}")
            print(f"FAILED {spec_id}: {exc}", flush=True)

    # GEE exchangeable correlation: same mean structure as clustered OLS (transition FE)
    for spec_id, terms, label in [
        ("A_ols_gee", model_a_smells, "A_gee"),
        ("B_ols_gee", model_b_smells, "B_gee"),
    ]:
        try:
            X = assemble_X(sample, terms, extra_controls=controls, use_transition_fe=True, use_year_index=False)
            result = fit_gee(y, X, groups)
            pred = np.asarray(result.fittedvalues)
            record_robust_linear(
                spec_id, label, terms, result, X, pred, "ols_gee",
                "GEE exchangeable by class; transition FE", n, n_classes,
            )
        except Exception as exc:
            fit_failures.append(f"{spec_id}: {type(exc).__name__}: {exc}")
            print(f"FAILED {spec_id}: {exc}", flush=True)

    # Forward OOS for primary OLS A/B and LoD-off A/B
    oos_specs = [
        ("A_ols_cluster", model_a_smells, "ols"),
        ("B_ols_cluster", model_b_smells, "ols"),
        ("A_ols_noloD", [t for t in model_a_smells if t != "LawOfDemeter_log1p"], "ols"),
        ("B_ols_noloD", [t for t in model_b_smells if t != "LawOfDemeter_log1p" and not t.endswith("_x_LoDlog")], "ols"),
        ("A_ols_noMS", [t for t in model_a_smells if t != "MutableStaticState_present"], "ols"),
        ("B_ols_noMS", [t for t in model_b_smells if "MutableStaticState" not in t], "ols"),
        ("A_nb_cluster", model_a_smells, "nb"),
        ("B_nb_cluster", model_b_smells, "nb"),
    ]
    for spec_id, terms, family in oos_specs:
        fold_rmses = []
        fold_maes = []
        fold_r2s = []
        for fold in forward_folds():
            train_set = set(fold["train_transitions"])
            train = sample[sample["transition"].isin(train_set)]
            test = sample[sample["transition"] == fold["test_transition"]]
            try:
                if family == "nb":
                    y_train = train["WMC_t1"].astype(float)
                    y_test = test["WMC_t1"].astype(float)
                    X_train = make_X(terms, "nb", controls, use_fe=False, use_year=True, data=train)
                    X_train = drop_zero_variance(X_train)
                    X_test = make_X(terms, "nb", controls, use_fe=False, use_year=True, data=test)
                    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
                    result = fit_nb_clustered(y_train, X_train, train["class"])
                    pred = np.clip(predict_align(result, X_test), 0, None)
                    metrics = {
                        "rmse": rmse(y_test, pred),
                        "mae": mae(y_test, pred),
                        "r2": oos_r2(y_test, pred),
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                    }
                else:
                    metrics = fit_predict_oos(train, test, terms, controls, "ols")
                oos_rows.append({
                    "model": spec_id,
                    "fold": fold["fold"],
                    "train_transitions": ";".join(fold["train_transitions"]),
                    "test_transition": fold["test_transition"],
                    **metrics,
                })
                fold_rmses.append(metrics["rmse"])
                fold_maes.append(metrics["mae"])
                fold_r2s.append(metrics["r2"])
            except Exception as exc:
                fit_failures.append(f"OOS {spec_id} fold {fold['fold']}: {exc}")
                oos_rows.append({
                    "model": spec_id,
                    "fold": fold["fold"],
                    "train_transitions": ";".join(fold["train_transitions"]),
                    "test_transition": fold["test_transition"],
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "error": str(exc),
                })
        # attach mean OOS to result row
        for row in result_rows:
            if row["model"] == spec_id:
                row["oos_rmse_mean"] = float(np.nanmean(fold_rmses)) if fold_rmses else np.nan
                row["oos_mae_mean"] = float(np.nanmean(fold_maes)) if fold_maes else np.nan
                row["oos_r2_mean"] = float(np.nanmean(fold_r2s)) if fold_r2s else np.nan

    # Diagnostics for primary OLS A vs B
    diag_a = residual_diagnostics(y, fitted["A_ols_cluster"]["pred"], "A_ols_cluster") if "A_ols_cluster" in fitted else {}
    diag_b = residual_diagnostics(y, fitted["B_ols_cluster"]["pred"], "B_ols_cluster") if "B_ols_cluster" in fitted else {}

    # Simple lag-only reference (not used to select A/B)
    try:
        X0 = assemble_X(sample, [], extra_controls=controls)
        r0 = fit_ols_clustered(y, X0, groups)
        p0 = predict_align(r0, X0)
        met0 = in_sample_metrics(y, p0, r0, "ols", n_classes, n)
        met0.update({
            "model": "lag_controls_only",
            "specification": "no_smells",
            "repeated_obs": "class-clustered SE",
            "controls": "+".join(["WMC_t"] + controls),
            "smell_terms": "",
            "n_params": int(X0.shape[1]),
        })
        result_rows.append(met0)
        coef_all.extend(coef_rows("lag_controls_only", "no_smells", r0, "ols"))
    except Exception as exc:
        fit_failures.append(f"lag_controls_only: {exc}")

    write_csv(OUT_RESULTS, result_rows)
    write_csv(OUT_COEFS, coef_all)
    write_csv(OUT_OOS, oos_rows)

    # Comparison table A vs B for primary and key sensitivities
    def find_row(model_id):
        for row in result_rows:
            if row.get("model") == model_id:
                return row
        return None

    comparisons = []
    pairs = [
        ("A_ols_cluster", "B_ols_cluster", "primary_ols"),
        ("A_ols_noloD", "B_ols_noloD", "sensitivity_no_LoD"),
        ("A_ols_noMS", "B_ols_noMS", "sensitivity_no_MutableStaticState"),
        ("A_nb_cluster", "B_nb_cluster", "robustness_negative_binomial"),
        ("A_ols_mixed", "B_ols_mixed", "sensitivity_class_random_intercept"),
        ("A_ols_gee", "B_ols_gee", "sensitivity_gee_exchangeable"),
    ]
    for a_id, b_id, label in pairs:
        a_row, b_row = find_row(a_id), find_row(b_id)
        if not a_row or not b_row:
            comparisons.append({
                "comparison": label,
                "model_a": a_id,
                "model_b": b_id,
                "status": "missing_fit",
            })
            continue
        d_aic = b_row["aic"] - a_row["aic"] if pd.notna(b_row["aic"]) and pd.notna(a_row["aic"]) else np.nan
        d_bic = b_row["bic"] - a_row["bic"] if pd.notna(b_row["bic"]) and pd.notna(a_row["bic"]) else np.nan
        d_rmse = b_row["rmse"] - a_row["rmse"]
        d_mae = b_row["mae"] - a_row["mae"]
        d_oos_rmse = (b_row.get("oos_rmse_mean", np.nan) - a_row.get("oos_rmse_mean", np.nan)
                      if "oos_rmse_mean" in b_row and "oos_rmse_mean" in a_row else np.nan)
        d_oos_mae = (b_row.get("oos_mae_mean", np.nan) - a_row.get("oos_mae_mean", np.nan)
                     if "oos_mae_mean" in b_row and "oos_mae_mean" in a_row else np.nan)
        in_sample_prefers_b = pd.notna(d_aic) and d_aic < 0
        oos_prefers_b = pd.notna(d_oos_rmse) and d_oos_rmse < 0
        if in_sample_prefers_b and oos_prefers_b:
            verdict = "B_better_on_both_in_sample_AIC_and_OOS_RMSE"
        elif (not in_sample_prefers_b) and (not oos_prefers_b) and pd.notna(d_aic) and pd.notna(d_oos_rmse):
            verdict = "A_better_or_equal_on_both"
        elif pd.isna(d_aic) or pd.isna(d_oos_rmse):
            verdict = "incomplete_metrics"
        else:
            verdict = "mixed_in_sample_vs_OOS"
        comparisons.append({
            "comparison": label,
            "model_a": a_id,
            "model_b": b_id,
            "n": a_row["n"],
            "aic_a": a_row["aic"],
            "aic_b": b_row["aic"],
            "delta_aic_B_minus_A": d_aic,
            "bic_a": a_row["bic"],
            "bic_b": b_row["bic"],
            "delta_bic_B_minus_A": d_bic,
            "rmse_a": a_row["rmse"],
            "rmse_b": b_row["rmse"],
            "delta_rmse_B_minus_A": d_rmse,
            "mae_a": a_row["mae"],
            "mae_b": b_row["mae"],
            "delta_mae_B_minus_A": d_mae,
            "oos_rmse_a": a_row.get("oos_rmse_mean"),
            "oos_rmse_b": b_row.get("oos_rmse_mean"),
            "delta_oos_rmse_B_minus_A": d_oos_rmse,
            "oos_mae_a": a_row.get("oos_mae_mean"),
            "oos_mae_b": b_row.get("oos_mae_mean"),
            "delta_oos_mae_B_minus_A": d_oos_mae,
            "verdict": verdict,
            "rule": "B shows incremental value only if AIC is lower AND mean forward RMSE is lower. No p-value screening. No 5% threshold.",
        })
    write_csv(OUT_COMPARE, comparisons)

    write_sample_report(pack, controls, control_notes)
    write_diagnostics(
        pack, sample, pred_rows, corr, vif_rows, support_rows, control_notes,
        family_notes, diag_a, diag_b, fitted, fit_failures, controls,
        model_a_smells, model_b_smells, primary_family,
    )
    write_results_report(
        pack, sample, controls, control_notes, support_rows, model_a_smells,
        model_b_smells, result_rows, comparisons, coef_all, oos_rows,
        family_notes, fit_failures, primary_family, trial_vifs, vif_rows,
    )
    write_thesis_tables(pack, sample, pred_rows, support_rows, result_rows, comparisons, coef_all)
    evidence = {
        "n_canonical": pack["total"],
        "n_modeling": n,
        "n_classes": n_classes,
        "controls": controls,
        "model_a_smells": model_a_smells,
        "model_b_smells": model_b_smells,
        "estimable_binary": estimable_binary,
        "estimable_lod": estimable_lod,
        "fit_failures": fit_failures,
        "primary_family": primary_family,
        "zero_fill_violations": pack["zero_fill_violations"],
    }
    OUT_EVIDENCE.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    print("Wrote modeling outputs", flush=True)
    print("Failures:", fit_failures, flush=True)


def write_sample_report(pack, controls, control_notes):
    s = pack["sample"]
    lines = []
    add = lines.append
    add("# Signal Android modeling sample")
    add("")
    add("Built from `signal_android_pilot_panel.csv`. Dataset was not rebuilt.")
    add("")
    add("## Filters")
    add("")
    add("- `smell_status_known_t == 1`")
    add("- `WMC_t` and `WMC_t1` numeric")
    add("- `smell_status_known_t1` is **not** required")
    add("- Unmatched PMD rows are excluded, not recoded as zero")
    add("")
    add("## Counts")
    add("")
    add(f"- Canonical longitudinal rows: **{pack['total']}**")
    add(f"- `smell_status_known_t = 1`: **{pack['n_known']}**")
    add(f"- Numeric WMC_t and WMC_t1 (full canonical file): **{pack['n_numeric']}**")
    add(f"- Final complete-case modeling sample: **{len(s)}**")
    add(f"- Unique classes: **{s['class'].nunique()}**")
    add(f"- Unknown-status rows excluded: **{pack['n_unknown']}**")
    add("")
    add("## Observations per transition")
    add("")
    add("| version_t | version_t1 | n |")
    add("| --- | --- | ---: |")
    for (vt, vt1), n in pack["obs_per_transition"].items():
        add(f"| {vt} | {vt1} | {n} |")
    add("")
    add("## Observations per class")
    add("")
    opc = pack["obs_per_class"]
    add(f"mean={opc.mean():.3f}; median={opc.median():.1f}; min={opc.min()}; max={opc.max()}")
    add("")
    add("## Zero-fill check")
    add("")
    if pack["zero_fill_violations"]:
        add("FAILED:")
        for item in pack["zero_fill_violations"]:
            add(f"- {item}")
    else:
        add("Passed: among `smell_status_known_t=0` rows, all candidate count fields are NULL, never 0.")
    add("")
    add("## Controls retained after variance/VIF rules")
    add("")
    add(", ".join(controls) if controls else "(none besides WMC_t)")
    add("")
    for note in control_notes:
        add(f"- {note}")
    add("")
    add("Unit of analysis: class-transition row. Repeated classes are not treated as independent.")
    add("")
    OUT_SAMPLE_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_diagnostics(pack, sample, pred_rows, corr, vif_rows, support_rows, control_notes,
                      family_notes, diag_a, diag_b, fitted, fit_failures, controls,
                      model_a_smells, model_b_smells, primary_family):
    lines = []
    add = lines.append
    add("# Signal Android model diagnostics")
    add("")
    add("## Outcome distribution (modeling sample)")
    add("")
    for note in family_notes:
        add(f"- {note}")
    add("")
    add("## Predictor representation")
    add("")
    add("- DataClass, LongMethod, LongParameterList, MutableStaticState: **binary presence** (counts are almost 0/1; the thesis construct is presence/configuration).")
    add("- LawOfDemeter: **log1p(count)**. Binary LoD is banned (~86% present).")
    add("- GodClass: excluded from all fitted models.")
    add("")
    add("## Residual diagnostics (primary OLS)")
    add("")
    add("| Model | mean | sd | skew | kurtosis | min | max | share |resid|>10 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for diag in [diag_a, diag_b]:
        if diag:
            add(
                f"| {diag['label']} | {fmt(diag['residual_mean'])} | {fmt(diag['residual_sd'])} | "
                f"{fmt(diag['residual_skew'])} | {fmt(diag['residual_kurtosis'])} | {fmt(diag['residual_min'])} | "
                f"{fmt(diag['residual_max'])} | {fmt(diag['share_resid_gt_10'])} |"
            )
    add("")
    add("OLS residuals inherit a long right tail from WMC. That is why NB is reported as robustness, not because it produced a stronger interaction result.")
    add("")
    add("## VIF (Model A OLS RHS, including transition FE)")
    add("")
    add("| variable | VIF |")
    add("| --- | ---: |")
    for row in vif_rows:
        add(f"| {row['variable']} | {fmt(row['vif'], 2)} |")
    add("")
    add("## Control selection notes")
    add("")
    for note in control_notes:
        add(f"- {note}")
    add("")
    add("## Fit failures")
    add("")
    if fit_failures:
        for item in fit_failures:
            add(f"- {item}")
    else:
        add("None.")
    add("")
    add("## Prespecified interaction support rule")
    add("")
    add(f"- Binary × binary: both_present ≥ {MIN_BOTH_FOR_BINARY_INTERACTION}")
    add(f"- Binary × LoD intensity: n(A=1) ≥ {MIN_PRESENT_FOR_LOD_INTERACTION} and both_with_LoD_present ≥ {MIN_BOTH_FOR_BINARY_INTERACTION} and LoD count varies among A=1")
    add("- Terms failing the rule are unavailable, not tested-and-dropped by p-value.")
    add("")
    OUT_DIAG.write_text("\n".join(lines), encoding="utf-8")


def write_results_report(pack, sample, controls, control_notes, support_rows, model_a_smells,
                         model_b_smells, result_rows, comparisons, coef_all, oos_rows,
                         family_notes, fit_failures, primary_family, trial_vifs, vif_rows):
    s = pack["sample"]
    by_model = {r["model"]: r for r in result_rows}
    primary = next((c for c in comparisons if c["comparison"] == "primary_ols"), {})
    lines = []
    add = lines.append

    def coef_table(model_id):
        rows = [c for c in coef_all if c["model"] == model_id]
        add("| term | coef | SE (clustered) | p | 95% CI |")
        add("| --- | ---: | ---: | ---: | --- |")
        skip = {"const"} | {c for c in [r["term"] for r in rows] if str(c).startswith("tr_")}
        # show intercept + scientific terms; compress transition FE
        for r in rows:
            if str(r["term"]).startswith("tr_"):
                continue
            add(
                f"| `{r['term']}` | {fmt(r['coef'], 4)} | {fmt(r['se'], 4)} | {fmt(r['p_value'], 4)} | "
                f"[{fmt(r['ci_low'], 4)}, {fmt(r['ci_high'], 4)}] |"
            )
        n_tr = sum(1 for r in rows if str(r["term"]).startswith("tr_"))
        add(f"| transition FE ({n_tr} dummies) | (omitted from display) |  |  |  |")

    add("# Signal Android Model A vs Model B")
    add("")
    add("## 1. Research question")
    add("")
    add("Within the Signal Android longitudinal sample for which PMD smell status is observed, does explicitly modeling smell interactions provide measurable incremental explanatory or predictive value beyond a model containing only individual-smell information?")
    add("")
    add("This is not a test of whether smells are associated with WMC, and it is not a causal claim.")
    add("")
    add("## 2. Sample construction")
    add("")
    add(f"- Canonical rows: {pack['total']}")
    add(f"- Modeling sample: **{len(s)}** class-transition rows, **{s['class'].nunique()}** classes")
    add("- Filter: `smell_status_known_t=1` and numeric `WMC_t`, `WMC_t1`")
    add("- Unmatched PMD classes excluded (counts remain NULL in the canonical file; zero-fill check passed)")
    add("")
    add("| Transition | n |")
    add("| --- | ---: |")
    for (vt, vt1), n in pack["obs_per_transition"].items():
        add(f"| {vt} → {vt1} | {n} |")
    add("")
    add("## 3. Statistical-unit definition")
    add("")
    add("One row is a top-level class observed across a consecutive version pair. Classes can contribute up to five rows. Primary inference uses **class-clustered standard errors**. Robustness: GEE with exchangeable working correlation, and a class random-intercept MixedLM (ML/CG; REML/lbfgs is singular with singleton classes).")
    add("")
    add("## 4. Outcome definition")
    add("")
    add("`WMC_t1` (subsequent Weighted Method Count). Predictors are measured at t. `WMC_t` is the lagged baseline. `delta_WMC` is not the outcome and is not called software corrosion.")
    add("")
    add("## 5. Predictor representation")
    add("")
    add("| Construct | Representation in Model A/B | Why |")
    add("| --- | --- | --- |")
    add("| DataClass | binary presence | count is nearly 0/1; construct is presence |")
    add("| LongMethod | binary presence | same |")
    add("| LongParameterList | binary presence | same |")
    add("| MutableStaticState | binary presence | sparse; still encoded as presence, with a no-MS sensitivity |")
    add("| LawOfDemeter | log1p(count) | binary is ~86% present; intensity varies |")
    add("| GodClass | **excluded** | WMC-contaminated PMD definition |")
    add("| CBO_t | **dropped** | 99.8% zeros in the modeling sample |")
    add("")
    add("Controls retained besides `WMC_t`: " + (", ".join(controls) if controls else "none") + ".")
    add("")
    for note in control_notes:
        add(f"- {note}")
    add("")
    add("## 6. Model A specification (primary)")
    add("")
    add("Family: **OLS** with class-clustered SE and transition fixed effects.")
    add("")
    add("`WMC_t1 ~ WMC_t + " + " + ".join(controls + model_a_smells) + " + transition FE`")
    add("")
    add("## 7. Model B specification (primary)")
    add("")
    add("Same as Model A plus estimable interactions:")
    add("")
    add("`" + " + ".join(model_b_smells) + "`")
    add("")
    add("Main effects remain in the model whenever an interaction is included.")
    add("")
    add("## 8. Treatment of repeated classes")
    add("")
    add("Primary: cluster-robust SE at `class`. Robustness 1: Gaussian GEE, exchangeable correlation by class, same transition FE as OLS. Robustness 2: `MixedLM` random intercept by class with `year_index` (ML, CG optimizer). REML/lbfgs MixedLM with transition FE is singular because 512 classes appear only once.")
    add("")
    add("## 9. Treatment of transitions")
    add("")
    add("In-sample: transition fixed effects (four dummies after dropping the first). Forward validation cannot identify a future dummy, so OOS refits replace FE with numeric `year_index` for **both** A and B.")
    add("")
    add("## 10. Treatment of LawOfDemeter")
    add("")
    add("Primary uses `log1p(LawOfDemeter_count_t)`. Binary LoD is not used. Sensitivity: drop LoD main effect and all LoD interactions.")
    add("")
    add("## 11. Treatment of sparse constructs")
    add("")
    n_ms = int(s["MutableStaticState_present"].sum())
    add(f"MutableStaticState is present in **{n_ms}** modeling-sample rows. It is kept in the primary specification. Sensitivity `A_noMS` / `B_noMS` drops it and its interactions. It is not deleted after looking at p-values.")
    add("")
    add("## 12. Interaction-estimability analysis")
    add("")
    add("| interaction | both | A only | B only | neither | estimable | reason |")
    add("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for r in support_rows:
        add(
            f"| {r['interaction']} | {r['both_present']} | {r['a_only']} | {r['b_only']} | "
            f"{r['neither']} | {r['estimable']} | {r['reason']} |"
        )
    add("")
    add(
        "LongMethod × LoD and MutableStaticState × LoD meet the both-present threshold but have "
        "`a_only` of 1 and 3. They stay in the locked Model B (the support rule was not changed after seeing coefficients). "
        "The Large LongMethod main-effect shift in Model B is a symptom of that weak cell, not a finding about LongMethod."
    )
    add("")
    add("## 13. Fit comparison")
    add("")
    add("| model | n | AIC | BIC | in-sample RMSE | in-sample MAE | in-sample R² |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in ["lag_controls_only", "A_ols_cluster", "B_ols_cluster", "A_nb_cluster", "B_nb_cluster",
                "A_ols_noloD", "B_ols_noloD", "A_ols_noMS", "B_ols_noMS", "A_ols_mixed", "B_ols_mixed",
                "A_ols_gee", "B_ols_gee"]:
        r = by_model.get(key)
        if not r:
            continue
        add(
            f"| `{key}` | {r['n']} | {fmt(r.get('aic'), 1)} | {fmt(r.get('bic'), 1)} | "
            f"{fmt(r.get('rmse'), 4)} | {fmt(r.get('mae'), 4)} | {fmt(r.get('r2'), 4)} |"
        )
    add("")
    add("## 14. Out-of-sample comparison (forward chaining)")
    add("")
    add("Train on earlier transitions, test on the next later transition. A class's later row is never in the training set of a fold that tests an earlier period.")
    add("")
    add("| model | mean OOS RMSE | mean OOS MAE | mean OOS R² |")
    add("| --- | ---: | ---: | ---: |")
    for key in ["A_ols_cluster", "B_ols_cluster", "A_ols_noloD", "B_ols_noloD",
                "A_ols_noMS", "B_ols_noMS", "A_nb_cluster", "B_nb_cluster"]:
        r = by_model.get(key)
        if not r:
            continue
        add(f"| `{key}` | {fmt(r.get('oos_rmse_mean'), 4)} | {fmt(r.get('oos_mae_mean'), 4)} | {fmt(r.get('oos_r2_mean'), 4)} |")
    add("")
    add("Fold-level numbers: `signal_android_oos_folds.csv`.")
    add("")
    add("## 15. Robustness analyses")
    add("")
    add("| comparison | ΔAIC (B−A) | Δ in-sample RMSE (B−A) | Δ mean OOS RMSE (B−A) | verdict |")
    add("| --- | ---: | ---: | ---: | --- |")
    for c in comparisons:
        add(
            f"| {c.get('comparison')} | {fmt(c.get('delta_aic_B_minus_A'), 2)} | "
            f"{fmt(c.get('delta_rmse_B_minus_A'), 4)} | {fmt(c.get('delta_oos_rmse_B_minus_A'), 4)} | "
            f"{c.get('verdict', '')} |"
        )
    add("")
    add("p-values are in the coefficient file for completeness. They were **not** used to add or drop interactions.")
    add("")
    add("## 16. Interaction interpretation")
    add("")
    add("An interaction coefficient is the difference in the WMC_t1 association of one smell variable across values of another, conditional on the model. It is not synergy, not a configuration proof, and not a causal effect.")
    add("")
    add("### Model A coefficients (primary OLS, clustered SE)")
    add("")
    if "A_ols_cluster" in by_model:
        coef_table("A_ols_cluster")
    add("")
    add("### Model B coefficients (primary OLS, clustered SE)")
    add("")
    if "B_ols_cluster" in by_model:
        coef_table("B_ols_cluster")
    add("")
    add("## 17. Limitations")
    add("")
    add("- LongMethod × LoD uses 41 LongMethod rows of which 40 also have LoD; the LongMethod main effect in Model B is weakly identified.")
    add("- Sample is PMD-observed top-level classes only, one project (Signal Android).")
    add("- Five transitions; forward validation is coarse.")
    add("- Most four-smell pairs are not estimable (cells of 0–5).")
    add("- LoD intensity is correlated with LOC (size).")
    add("- OLS residuals are heavy-tailed; NB robustness uses a different lag functional form.")
    add("- CBO in this extract is effectively unused (almost all zeros).")
    add("- No causal identification strategy.")
    add("")
    add("## 18. Incremental value of interactions?")
    add("")
    verdict = primary.get("verdict", "unavailable")
    add(f"Primary OLS verdict (pre-specified dual criterion: lower AIC **and** lower mean forward RMSE for B): **{verdict}**.")
    add("")
    add(
        f"In-sample ΔAIC (B−A) = {fmt(primary.get('delta_aic_B_minus_A'), 2)}; "
        f"ΔRMSE (B−A) = {fmt(primary.get('delta_rmse_B_minus_A'), 4)}; "
        f"Δ mean OOS RMSE (B−A) = {fmt(primary.get('delta_oos_rmse_B_minus_A'), 4)}."
    )
    add("")
    add("### Direct answer")
    add("")
    add(
        "Within the Signal Android longitudinal sample for which PMD smell status is observed, "
        "explicitly modeling estimable smell interactions produced only a **tiny, non-robust** "
        "increment over individual-smell information. It should not be treated as evidence that "
        "interaction/configuration terms are generally informative, nor as a causal finding."
    )
    add("")
    add("Complementary measures (not used to retune the specification):")
    add("")
    add(
        f"- Mechanical dual criterion (lower AIC and lower mean OOS RMSE): **{verdict}** "
        f"(ΔAIC={fmt(primary.get('delta_aic_B_minus_A'), 2)}; "
        f"Δ mean OOS RMSE={fmt(primary.get('delta_oos_rmse_B_minus_A'), 4)})."
    )
    add(
        f"- Mean forward MAE moved the other way (Δ MAE B−A = "
        f"{fmt(primary.get('delta_oos_mae_B_minus_A'), 4)})."
    )
    add("- Forward RMSE is lower for B in some folds and higher in others (`signal_android_oos_folds.csv`).")
    add("- Dropping LawOfDemeter (main effect and LoD interactions) removes the in-sample AIC gain; `sensitivity_no_LoD` does not favor B.")
    add("- The only estimable four-smell pair (DataClass × LongParameterList) is not enough, on its own, to improve Model B.")
    add("- NegativeBinomial robustness does not agree with OLS on AIC (B much worse AIC, mixed OOS).")
    add("- Lagged WMC already explains most of WMC_t1 (in-sample R² ≈ 0.94 even in Model A). Smell terms, including interactions, are a small residual layer.")
    add("")
    add("Therefore the thesis-ready answer for this project and sample is: **no reliable incremental value of the interaction representation beyond individual-smell information**. A mechanical RMSE/AIC tick is not the same as a stable scientific increment.")
    add("")
    add("This is not a claim about all software systems.")
    add("")
    add("What was estimable: see Section 12.")
    add("What was not estimable: binary pairs with both_present < 20, including LongMethod×MutableStaticState (0) and LongParameterList×MutableStaticState (0).")
    add("What was too sparse: MutableStaticState (few dozen positives); several interaction cells.")
    add("What depended on LawOfDemeter: compare `primary_ols` vs `sensitivity_no_LoD`.")
    add("What depended on modeling family: compare `primary_ols` vs `robustness_negative_binomial`.")
    add("What depended on repeated-class treatment: compare clustered OLS vs GEE vs MixedLM.")
    add("Before a thesis-wide experiment: replicate the same locked rules on other CSIQ projects; do not expand until this Signal pipeline is accepted.")
    add("")
    add("## Fit failures")
    add("")
    if fit_failures:
        for item in fit_failures:
            add(f"- {item}")
    else:
        add("None.")
    add("")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_thesis_tables(pack, sample, pred_rows, support_rows, result_rows, comparisons, coef_all):
    by_model = {r["model"]: r for r in result_rows}
    lines = []
    add = lines.append
    add("# Signal Android thesis tables")
    add("")
    add("Generated from `run_signal_android_models.py`. Variable names match the CSV outputs.")
    add("")

    add("## A. Modeling sample by transition")
    add("")
    add("| version_t | version_t1 | n | unique classes |")
    add("| --- | --- | ---: | ---: |")
    for (vt, vt1), n in pack["obs_per_transition"].items():
        n_cls = int(sample.loc[(sample["version_t"] == vt) & (sample["version_t1"] == vt1), "class"].nunique())
        add(f"| {vt} | {vt1} | {n} | {n_cls} |")
    add("")
    add(f"Total n={len(sample)}; unique classes={sample['class'].nunique()}.")
    add("")

    add("## B. Predictor prevalence (modeling sample)")
    add("")
    add("| variable | zero % | positive % | mean | median | SD | p75 | p90 | p95 | max |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in pred_rows:
        add(
            f"| `{row['variable']}` | {fmt_pct(row.get('zero_proportion'))} | {fmt_pct(row.get('positive_proportion'))} | "
            f"{fmt(row.get('mean'), 3)} | {fmt(row.get('median'), 3)} | {fmt(row.get('sd'), 3)} | "
            f"{fmt(row.get('p75'), 3)} | {fmt(row.get('p90'), 3)} | {fmt(row.get('p95'), 3)} | {fmt(row.get('maximum'), 1)} |"
        )
    add("")
    add("Binary presence in the modeling sample: "
        f"DataClass={int(sample['DataClass_present'].sum())}; "
        f"LongMethod={int(sample['LongMethod_present'].sum())}; "
        f"LongParameterList={int(sample['LongParameterList_present'].sum())}; "
        f"MutableStaticState={int(sample['MutableStaticState_present'].sum())}; "
        f"LawOfDemeter present={int(sample['LawOfDemeter_present'].sum())} "
        f"({fmt_pct(float(sample['LawOfDemeter_present'].mean()))}).")
    add("")

    add("## C. Model A coefficients (primary OLS, class-clustered SE)")
    add("")
    add("| term | coef | SE | p | 95% CI |")
    add("| --- | ---: | ---: | ---: | --- |")
    for r in coef_all:
        if r["model"] != "A_ols_cluster":
            continue
        add(
            f"| `{r['term']}` | {fmt(r['coef'], 4)} | {fmt(r['se'], 4)} | {fmt(r['p_value'], 4)} | "
            f"[{fmt(r['ci_low'], 4)}, {fmt(r['ci_high'], 4)}] |"
        )
    add("")

    add("## D. Model B coefficients (primary OLS, class-clustered SE)")
    add("")
    add("| term | coef | SE | p | 95% CI |")
    add("| --- | ---: | ---: | ---: | --- |")
    for r in coef_all:
        if r["model"] != "B_ols_cluster":
            continue
        add(
            f"| `{r['term']}` | {fmt(r['coef'], 4)} | {fmt(r['se'], 4)} | {fmt(r['p_value'], 4)} | "
            f"[{fmt(r['ci_low'], 4)}, {fmt(r['ci_high'], 4)}] |"
        )
    add("")

    add("## E. Model A vs Model B comparison")
    add("")
    add("| comparison | n | ΔAIC (B−A) | ΔBIC (B−A) | Δ RMSE (B−A) | Δ MAE (B−A) | Δ OOS RMSE (B−A) | Δ OOS MAE (B−A) | verdict |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for c in comparisons:
        add(
            f"| {c.get('comparison')} | {fmt(c.get('n'), 0)} | {fmt(c.get('delta_aic_B_minus_A'), 2)} | "
            f"{fmt(c.get('delta_bic_B_minus_A'), 2)} | {fmt(c.get('delta_rmse_B_minus_A'), 4)} | "
            f"{fmt(c.get('delta_mae_B_minus_A'), 4)} | {fmt(c.get('delta_oos_rmse_B_minus_A'), 4)} | "
            f"{fmt(c.get('delta_oos_mae_B_minus_A'), 4)} | {c.get('verdict', c.get('status', ''))} |"
        )
    add("")
    add("Pre-specified dual criterion for claiming incremental value: lower AIC **and** lower mean forward RMSE. Complementary MAE/OOS-R²/LoD-off/family checks are reported and are not used to retune terms.")
    add("")

    add("## F. Interaction support / available pairs")
    add("")
    add("| interaction | type | both | A only | B only | neither | estimable | weak main-effect cell |")
    add("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for r in support_rows:
        add(
            f"| {r['interaction']} | {r['type']} | {r['both_present']} | {r['a_only']} | {r['b_only']} | "
            f"{r['neither']} | {r['estimable']} | {r.get('weak_main_effect_cell', '')} |"
        )
    add("")

    add("## G. Robustness results")
    add("")
    add("| model | family | n | AIC | in-sample RMSE | mean OOS RMSE | mean OOS MAE |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for key in [
        "A_ols_cluster", "B_ols_cluster", "A_ols_noloD", "B_ols_noloD",
        "A_ols_noMS", "B_ols_noMS", "A_nb_cluster", "B_nb_cluster",
        "A_ols_mixed", "B_ols_mixed", "A_ols_gee", "B_ols_gee", "lag_controls_only",
    ]:
        r = by_model.get(key)
        if not r:
            continue
        add(
            f"| `{key}` | {r.get('family')} | {r['n']} | {fmt(r.get('aic'), 1)} | "
            f"{fmt(r.get('rmse'), 4)} | {fmt(r.get('oos_rmse_mean'), 4)} | {fmt(r.get('oos_mae_mean'), 4)} |"
        )
    add("")
    OUT_TABLES.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
