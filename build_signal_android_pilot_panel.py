"""Build the canonical Signal Android top-level class-version / longitudinal panel.

No regression, interaction models, or hypothesis tests are run here.

Smell-status semantics:
  status_known = 1 and count = 0  -> PMD observed this class-version; candidate rule had no violation
  status_known = 0 and count NULL -> class-version is not in the PMD export; absence is unknown

Run from the repository root:
    python build_signal_android_pilot_panel.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

csv.field_size_limit(2**31 - 1)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT  / "Dataset"
SIGNAL_DIR = "10 Signal-Android-master"
PROJECT = "Signal-Android"
VERSIONS = ["2016-1", "2017-1", "2018-1", "2019-1", "2020-1", "2021-1"]

# Must match csiq_pilot_audit.py exactly.
CONSTRUCTS = [
    {
        "field": "DataClass",
        "pmd_rule": "DataClass",
        "label": "Data Class",
        "wmc_contaminated": False,
    },
    {
        "field": "LongMethod",
        "pmd_rule": "ExcessiveMethodLength",
        "label": "Long Method",
        "wmc_contaminated": False,
    },
    {
        "field": "LongParameterList",
        "pmd_rule": "ExcessiveParameterList",
        "label": "Long Parameter List",
        "wmc_contaminated": False,
    },
    {
        "field": "MutableStaticState",
        "pmd_rule": "MutableStaticState",
        "label": "Mutable static state",
        "wmc_contaminated": False,
    },
    {
        "field": "LawOfDemeter",
        "pmd_rule": "LawOfDemeter",
        "label": "Law of Demeter",
        "wmc_contaminated": False,
    },
    {
        "field": "GodClass",
        "pmd_rule": "GodClass",
        "label": "God Class",
        "wmc_contaminated": True,
    },
]

# Top-level validation targets from csiq_signal_android_coverage.csv /
# csiq_signal_android_wmc_transitions.csv. Mismatch is a hard stop.
AUDIT_VERSION = {
    "2016-1": {"top_level": 608, "exact_top_level_matches": 414},
    "2017-1": {"top_level": 680, "exact_top_level_matches": 464},
    "2018-1": {"top_level": 618, "exact_top_level_matches": 436},
    "2019-1": {"top_level": 755, "exact_top_level_matches": 530},
    "2020-1": {"top_level": 1266, "exact_top_level_matches": 854},
    "2021-1": {"top_level": 1943, "exact_top_level_matches": 1255},
}
AUDIT_TRANSITIONS = {
    ("2016-1", "2017-1"): {"shared_top_level": 597, "valid_wmc": 597, "changed": 45, "shared_all_class_level": 919, "changed_all_class_level": 51},
    ("2017-1", "2018-1"): {"shared_top_level": 489, "valid_wmc": 489, "changed": 82, "shared_all_class_level": 780, "changed_all_class_level": 96},
    ("2018-1", "2019-1"): {"shared_top_level": 598, "valid_wmc": 598, "changed": 117, "shared_all_class_level": 994, "changed_all_class_level": 134},
    ("2019-1", "2020-1"): {"shared_top_level": 643, "valid_wmc": 643, "changed": 123, "shared_all_class_level": 1024, "changed_all_class_level": 147},
    ("2020-1", "2021-1"): {"shared_top_level": 1168, "valid_wmc": 1168, "changed": 223, "shared_all_class_level": 1956, "changed_all_class_level": 281},
}

OUT_CLASS_VERSION = ROOT / "signal_android_pilot_class_version.csv"
OUT_PANEL = ROOT / "signal_android_pilot_panel.csv"
OUT_DICTIONARY = ROOT / "signal_android_pilot_panel_dictionary.csv"
OUT_PREVALENCE = ROOT / "signal_android_pilot_smell_prevalence.csv"
OUT_COOCCUR = ROOT / "signal_android_pilot_cooccurrence.csv"
OUT_CONFIGS = ROOT / "signal_android_pilot_configurations.csv"
OUT_WMC = ROOT / "signal_android_pilot_wmc_descriptives.csv"
OUT_LOD = ROOT / "signal_android_pilot_lod_distribution.csv"
OUT_VALIDATION = ROOT / "signal_android_pilot_validation.csv"
OUT_FREEZE = ROOT / "signal_android_pilot_construct_decisions.csv"
OUT_REPORT = ROOT / "signal_android_pilot_panel_report.md"
OUT_EVIDENCE = ROOT / "signal_android_pilot_panel_evidence.json"


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.reader(handle)


def read_header_rows(path: Path):
    rows = csv_rows(path)
    header = next(rows, [])
    return header, rows


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "na", "nan"}:
        return None
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def pct(numerator, denominator, digits=4):
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, digits)


def quantiles(values, probabilities):
    if not values:
        return {p: None for p in probabilities}
    ordered = sorted(values)
    n = len(ordered)
    out = {}
    for p in probabilities:
        if n == 1:
            out[p] = ordered[0]
            continue
        idx = p * (n - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            out[p] = ordered[lo]
        else:
            weight = idx - lo
            out[p] = ordered[lo] * (1 - weight) + ordered[hi] * weight
    return out


def summarize_numeric(values):
    if not values:
        return {
            "n": 0, "mean": None, "median": None, "stdev": None,
            "min": None, "max": None,
            "q01": None, "q05": None, "q10": None, "q25": None,
            "q50": None, "q75": None, "q90": None, "q95": None, "q99": None,
        }
    q = quantiles(values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "q01": q[0.01], "q05": q[0.05], "q10": q[0.10], "q25": q[0.25],
        "q50": q[0.50], "q75": q[0.75], "q90": q[0.90], "q95": q[0.95], "q99": q[0.99],
    }


def quality_row_kind(qualified_name):
    """Identical to csiq_pilot_audit.py. Do not change independently."""
    qn = (qualified_name or "").strip()
    if not qn or qn.lower() == "null":
        return "blank"
    if qn.startswith("<Package>"):
        return "package"
    if qn.startswith("<Anonymous>"):
        return "anonymous"
    if qn.startswith("<Method>"):
        return "method"
    if qn.startswith("<Field>"):
        return "field"
    if qn.startswith("<"):
        return "other_structural"
    if "." not in qn:
        return "project_header_or_unqualified"
    parts = qn.split(".")
    if len(parts) >= 2 and parts[-2][:1].isupper():
        return "inner_or_nested"
    return "top_level"


def smell_key(package, file_path):
    name = re.split(r"[\\/]", (file_path or "").strip())[-1]
    stem = re.sub(r"\.[^.]+$", "", name) if "." in name else name
    package = (package or "").strip()
    if not stem:
        return ""
    if package and package.lower() != "null":
        return f"{package}.{stem}"
    return stem


def first_index(header, name):
    for i, col in enumerate(header):
        if col.strip() == name:
            return i
    return None


def load_quality_top_level(path: Path):
    header, rows = read_header_rows(path)
    idx_qn = 0
    idx_wmc = first_index(header, "WMC")
    idx_loc = first_index(header, "LOC")
    idx_cbo = first_index(header, "CBO")
    idx_nom = first_index(header, "NOM")
    kind_counts = Counter()
    classes = {}
    duplicate_extras = 0
    total_rows = 0
    for row in rows:
        total_rows += 1
        qn = row[idx_qn].strip() if row else ""
        kind = quality_row_kind(qn)
        kind_counts[kind] += 1
        if kind != "top_level":
            continue
        if qn in classes:
            duplicate_extras += 1
            continue
        classes[qn] = {
            "class": qn,
            "kind": kind,
            "WMC": parse_float(row[idx_wmc]) if idx_wmc is not None and len(row) > idx_wmc else None,
            "LOC": parse_float(row[idx_loc]) if idx_loc is not None and len(row) > idx_loc else None,
            "CBO": parse_float(row[idx_cbo]) if idx_cbo is not None and len(row) > idx_cbo else None,
            "NOM": parse_float(row[idx_nom]) if idx_nom is not None and len(row) > idx_nom else None,
        }
    return {
        "header": header,
        "total_rows": total_rows,
        "kind_counts": kind_counts,
        "classes": classes,
        "duplicate_extras": duplicate_extras,
    }


def load_smells(path: Path):
    header, rows = read_header_rows(path)
    per_class = defaultdict(Counter)
    total_rows = 0
    empty_key = 0
    for row in rows:
        total_rows += 1
        if len(row) < 8:
            continue
        key = smell_key(row[1], row[2])
        rule = row[7].strip()
        if not key:
            empty_key += 1
            continue
        if rule:
            per_class[key][rule] += 1
    return {"total_rows": total_rows, "empty_key": empty_key, "per_class": per_class}


def blank_if_none(value):
    if value is None:
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return value


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: blank_if_none(row.get(k)) for k in fieldnames})


def fmt_num(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-12:
            return str(int(round(value)))
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def fmt_pct(value, digits=2):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}%"


def md_escape(text):
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def attach_smells(class_rec, smell_rules):
    """Return smell fields. Unknown status => NULL counts, never 0."""
    out = {}
    if smell_rules is None:
        out["smell_status_known"] = 0
        out["pmd_violation_rows"] = None
        for spec in CONSTRUCTS:
            out[f"{spec['field']}_count"] = None
            out[f"{spec['field']}_status_known"] = 0
        return out
    out["smell_status_known"] = 1
    out["pmd_violation_rows"] = int(sum(smell_rules.values()))
    for spec in CONSTRUCTS:
        out[f"{spec['field']}_count"] = int(smell_rules.get(spec["pmd_rule"], 0))
        out[f"{spec['field']}_status_known"] = 1
    return out


def candidate_present_count(row):
    n = 0
    for spec in CONSTRUCTS:
        value = row.get(f"{spec['field']}_count")
        if value is not None and value > 0:
            n += 1
    return n


def main():
    print("Loading Signal Android quality and smell files", flush=True)
    quality_by_version = {}
    smell_by_version = {}
    class_version_rows = []
    validation_rows = []
    errors = []

    for version in VERSIONS:
        q_path = DATASET / "quality_attributes" / SIGNAL_DIR / f"{version}.csv"
        s_path = DATASET / "codesmells" / "csv" / SIGNAL_DIR / f"{version}.csv"
        quality = load_quality_top_level(q_path)
        smells = load_smells(s_path)
        quality_by_version[version] = quality
        smell_by_version[version] = smells

        n_top = len(quality["classes"])
        n_known = 0
        n_numeric_wmc = 0
        accidental_zero = 0
        non_top = sum(1 for rec in quality["classes"].values() if rec["kind"] != "top_level")
        for qn, rec in quality["classes"].items():
            smell_rules = smells["per_class"].get(qn)
            smell_fields = attach_smells(rec, smell_rules)
            if smell_fields["smell_status_known"] == 1:
                n_known += 1
            else:
                for spec in CONSTRUCTS:
                    if smell_fields[f"{spec['field']}_count"] == 0:
                        accidental_zero += 1
            if rec["WMC"] is not None:
                n_numeric_wmc += 1
            row = {
                "project": PROJECT,
                "version": version,
                "class": qn,
                "class_kind": rec["kind"],
                "WMC": rec["WMC"],
                "LOC": rec["LOC"],
                "CBO": rec["CBO"],
                "NOM": rec["NOM"],
                **smell_fields,
            }
            class_version_rows.append(row)

        expected = AUDIT_VERSION[version]
        row_ok = n_top == expected["top_level"] and n_known == expected["exact_top_level_matches"]
        if n_top != expected["top_level"]:
            errors.append(
                f"{version}: top-level classes {n_top} != audit {expected['top_level']}"
            )
        if n_known != expected["exact_top_level_matches"]:
            errors.append(
                f"{version}: smell_status_known=1 {n_known} != audit exact_top_level_matches {expected['exact_top_level_matches']}"
            )
        if non_top:
            errors.append(f"{version}: {non_top} non-top-level classes leaked into the panel")
        if accidental_zero:
            errors.append(f"{version}: {accidental_zero} unmatched classes received count=0")
        if quality["duplicate_extras"]:
            errors.append(f"{version}: duplicate QualifiedName extras={quality['duplicate_extras']}")
        validation_rows.append({
            "check": "per_version_top_level",
            "version_t": version,
            "version_t1": "",
            "observed": n_top,
            "audit_expected": expected["top_level"],
            "match": n_top == expected["top_level"],
            "notes": f"numeric_WMC={n_numeric_wmc}; smell_known={n_known}; duplicate_extras={quality['duplicate_extras']}",
        })
        validation_rows.append({
            "check": "per_version_smell_known",
            "version_t": version,
            "version_t1": "",
            "observed": n_known,
            "audit_expected": expected["exact_top_level_matches"],
            "match": n_known == expected["exact_top_level_matches"],
            "notes": "status_known=1 must equal exact top-level PMD matches from the audit",
        })
        print(
            f"  {version}: top-level={n_top} known={n_known} numeric_WMC={n_numeric_wmc} ok={row_ok}",
            flush=True,
        )

    # Duplicate (version, class) check.
    keys = [(row["version"], row["class"]) for row in class_version_rows]
    dup_cv = len(keys) - len(set(keys))
    if dup_cv:
        errors.append(f"Duplicate (version, class) rows: {dup_cv}")
    validation_rows.append({
        "check": "no_duplicate_version_class",
        "version_t": "all",
        "version_t1": "",
        "observed": dup_cv,
        "audit_expected": 0,
        "match": dup_cv == 0,
        "notes": "",
    })

    print("Building longitudinal panel", flush=True)
    panel_rows = []
    for earlier, later in zip(VERSIONS, VERSIONS[1:]):
        left = quality_by_version[earlier]["classes"]
        right = quality_by_version[later]["classes"]
        smell_left = smell_by_version[earlier]["per_class"]
        smell_right = smell_by_version[later]["per_class"]
        shared = sorted(set(left) & set(right))
        n_valid = 0
        n_changed = 0
        n_wmc_t = 0
        n_wmc_t1 = 0
        for qn in shared:
            a = left[qn]
            b = right[qn]
            if a["kind"] != "top_level" or b["kind"] != "top_level":
                errors.append(f"{earlier}->{later}: non-top-level shared class {qn}")
                continue
            smell_t = attach_smells(a, smell_left.get(qn))
            smell_t1 = attach_smells(b, smell_right.get(qn))
            wmc_t = a["WMC"]
            wmc_t1 = b["WMC"]
            delta = None
            if wmc_t is not None:
                n_wmc_t += 1
            if wmc_t1 is not None:
                n_wmc_t1 += 1
            if wmc_t is not None and wmc_t1 is not None:
                n_valid += 1
                delta = wmc_t1 - wmc_t
                if delta != 0:
                    n_changed += 1
            row = {
                "project": PROJECT,
                "version_t": earlier,
                "version_t1": later,
                "class": qn,
                "class_kind": "top_level",
                "WMC_t": wmc_t,
                "WMC_t1": wmc_t1,
                "delta_WMC": delta,
                "LOC_t": a["LOC"],
                "CBO_t": a["CBO"],
                "NOM_t": a["NOM"],
                "LOC_t1": b["LOC"],
                "CBO_t1": b["CBO"],
                "NOM_t1": b["NOM"],
                "smell_status_known_t": smell_t["smell_status_known"],
                "smell_status_known_t1": smell_t1["smell_status_known"],
                "pmd_violation_rows_t": smell_t["pmd_violation_rows"],
                "pmd_violation_rows_t1": smell_t1["pmd_violation_rows"],
            }
            for spec in CONSTRUCTS:
                row[f"{spec['field']}_count_t"] = smell_t[f"{spec['field']}_count"]
                row[f"{spec['field']}_status_known_t"] = smell_t[f"{spec['field']}_status_known"]
                row[f"{spec['field']}_count_t1"] = smell_t1[f"{spec['field']}_count"]
                row[f"{spec['field']}_status_known_t1"] = smell_t1[f"{spec['field']}_status_known"]
            panel_rows.append(row)

        expected = AUDIT_TRANSITIONS[(earlier, later)]
        if len(shared) != expected["shared_top_level"]:
            errors.append(
                f"{earlier}->{later}: shared top-level {len(shared)} != audit {expected['shared_top_level']}"
            )
        if n_valid != expected["valid_wmc"]:
            errors.append(
                f"{earlier}->{later}: valid WMC {n_valid} != audit {expected['valid_wmc']}"
            )
        if n_changed != expected["changed"]:
            errors.append(
                f"{earlier}->{later}: changed WMC {n_changed} != audit {expected['changed']}"
            )
        validation_rows.append({
            "check": "transition_shared_top_level",
            "version_t": earlier,
            "version_t1": later,
            "observed": len(shared),
            "audit_expected": expected["shared_top_level"],
            "match": len(shared) == expected["shared_top_level"],
            "notes": (
                f"all-class-level audit shared={expected['shared_all_class_level']} "
                f"(larger because inner/nested types were included in the earlier all-class count)"
            ),
        })
        validation_rows.append({
            "check": "transition_valid_wmc",
            "version_t": earlier,
            "version_t1": later,
            "observed": n_valid,
            "audit_expected": expected["valid_wmc"],
            "match": n_valid == expected["valid_wmc"],
            "notes": f"numeric WMC_t={n_wmc_t}; numeric WMC_t1={n_wmc_t1}",
        })
        validation_rows.append({
            "check": "transition_changed_wmc",
            "version_t": earlier,
            "version_t1": later,
            "observed": n_changed,
            "audit_expected": expected["changed"],
            "match": n_changed == expected["changed"],
            "notes": f"all-class-level audit changed={expected['changed_all_class_level']}",
        })
        print(
            f"  {earlier}->{later}: shared={len(shared)} valid={n_valid} changed={n_changed}",
            flush=True,
        )

    dup_long = len(panel_rows) - len({(r["version_t"], r["version_t1"], r["class"]) for r in panel_rows})
    if dup_long:
        errors.append(f"Duplicate (version_t, version_t1, class) rows: {dup_long}")
    validation_rows.append({
        "check": "no_duplicate_longitudinal_keys",
        "version_t": "all",
        "version_t1": "",
        "observed": dup_long,
        "audit_expected": 0,
        "match": dup_long == 0,
        "notes": "",
    })

    # Smell counts must come from the matching version: a t-count may only be
    # non-NULL if that class exists in that version's smell file.
    cross_version_leak = 0
    for row in panel_rows:
        if row["smell_status_known_t"] == 1:
            if row["class"] not in smell_by_version[row["version_t"]]["per_class"]:
                cross_version_leak += 1
        elif any(row[f"{spec['field']}_count_t"] is not None for spec in CONSTRUCTS):
            cross_version_leak += 1
        if row["smell_status_known_t1"] == 1:
            if row["class"] not in smell_by_version[row["version_t1"]]["per_class"]:
                cross_version_leak += 1
        elif any(row[f"{spec['field']}_count_t1"] is not None for spec in CONSTRUCTS):
            cross_version_leak += 1
    if cross_version_leak:
        errors.append(f"Smell version leak or accidental fill: {cross_version_leak}")
    validation_rows.append({
        "check": "smell_counts_correct_version_no_zero_fill",
        "version_t": "all",
        "version_t1": "",
        "observed": cross_version_leak,
        "audit_expected": 0,
        "match": cross_version_leak == 0,
        "notes": "Non-NULL counts only when the class key is in that version's PMD file",
    })

    unknown_zero_cv = 0
    for row in class_version_rows:
        if row["smell_status_known"] == 0:
            for spec in CONSTRUCTS:
                if row[f"{spec['field']}_count"] is not None:
                    unknown_zero_cv += 1
    if unknown_zero_cv:
        errors.append(f"Class-version unknown rows with non-NULL counts: {unknown_zero_cv}")
    validation_rows.append({
        "check": "unknown_status_counts_are_null",
        "version_t": "all",
        "version_t1": "",
        "observed": unknown_zero_cv,
        "audit_expected": 0,
        "match": unknown_zero_cv == 0,
        "notes": "Unmatched quality classes must have NULL counts, not 0",
    })

    if errors:
        print("VALIDATION FAILED. Stopping without overwriting audit numbers.", flush=True)
        for item in errors:
            print(f"  ERROR: {item}", flush=True)
        write_csv(
            OUT_VALIDATION,
            ["check", "version_t", "version_t1", "observed", "audit_expected", "match", "notes"],
            validation_rows,
        )
        sys.exit(1)

    print("Validation against previous audit: PASS", flush=True)

    # ------------------------------------------------------------------
    # Descriptives among class-version observations
    # ------------------------------------------------------------------
    known_cv = [row for row in class_version_rows if row["smell_status_known"] == 1]
    prevalence_rows = []
    lod_values = []
    for spec in CONSTRUCTS:
        field = spec["field"]
        for version in ["ALL"] + VERSIONS:
            subset = known_cv if version == "ALL" else [r for r in known_cv if r["version"] == version]
            counts = [r[f"{field}_count"] for r in subset]
            positives = sum(1 for c in counts if c > 0)
            summary = summarize_numeric(counts)
            prevalence_rows.append({
                "construct": field,
                "pmd_rule": spec["pmd_rule"],
                "version": version,
                "n_status_known": len(subset),
                "n_count_gt_0": positives,
                "prevalence_among_known_pct": pct(positives, len(subset)),
                "mean_count": summary["mean"],
                "median_count": summary["median"],
                "p75": summary["q75"],
                "p90": summary["q90"],
                "max": summary["max"],
                "wmc_contaminated": spec["wmc_contaminated"],
            })
            if version == "ALL" and field == "LawOfDemeter":
                lod_values = counts

    n_zero_smells = sum(1 for r in known_cv if candidate_present_count(r) == 0)
    n_one_smell = sum(1 for r in known_cv if candidate_present_count(r) == 1)
    n_multi_smell = sum(1 for r in known_cv if candidate_present_count(r) >= 2)

    lod_zero = sum(1 for c in lod_values if c == 0)
    lod_one = sum(1 for c in lod_values if c == 1)
    lod_multi = sum(1 for c in lod_values if c >= 2)
    lod_summary = summarize_numeric(lod_values)
    lod_rows = [{
        "scope": "status_known_class_versions",
        "n": len(lod_values),
        "n_zero": lod_zero,
        "n_one": lod_one,
        "n_multiple": lod_multi,
        "proportion_gt_0": pct(lod_one + lod_multi, len(lod_values), 4) / 100 if lod_values else None,
        "proportion_gt_0_pct": pct(lod_one + lod_multi, len(lod_values)),
        "mean": lod_summary["mean"],
        "median": lod_summary["median"],
        "stdev": lod_summary["stdev"],
        "q25": lod_summary["q25"],
        "q75": lod_summary["q75"],
        "q90": lod_summary["q90"],
        "q95": lod_summary["q95"],
        "q99": lod_summary["q99"],
        "max": lod_summary["max"],
    }]

    # Co-occurrence among status-known class-versions. "Neither" is valid here.
    pair_rows = []
    fields = [spec["field"] for spec in CONSTRUCTS]
    for a, b in combinations(fields, 2):
        both = a_only = b_only = neither = 0
        for row in known_cv:
            pa = row[f"{a}_count"] > 0
            pb = row[f"{b}_count"] > 0
            if pa and pb:
                both += 1
            elif pa:
                a_only += 1
            elif pb:
                b_only += 1
            else:
                neither += 1
        pair_rows.append({
            "construct_a": a,
            "construct_b": b,
            "n_status_known": len(known_cv),
            "both_present": both,
            "a_only": a_only,
            "b_only": b_only,
            "neither": neither,
            "joint_presence_pct": pct(both, len(known_cv)),
            "terminology": "co-occurrence / joint presence; not an interaction effect",
        })

    config_counter = Counter()
    for row in known_cv:
        bits = tuple(1 if row[f"{spec['field']}_count"] > 0 else 0 for spec in CONSTRUCTS)
        config_counter[bits] += 1
    config_rows = []
    for bits, n in sorted(config_counter.items(), key=lambda item: (-item[1], item[0])):
        labels = [fields[i] for i, bit in enumerate(bits) if bit]
        config_rows.append({
            "configuration": "+".join(labels) if labels else "(none of the six candidates)",
            "n_constructs_present": sum(bits),
            "n_class_versions": n,
            "pct_of_status_known": pct(n, len(known_cv)),
        })

    # WMC descriptives on the longitudinal panel (numeric transitions).
    wmc_rows = []

    def wmc_block(scope, version_t, version_t1, rows):
        wmc_t = [r["WMC_t"] for r in rows if r["WMC_t"] is not None]
        wmc_t1 = [r["WMC_t1"] for r in rows if r["WMC_t1"] is not None]
        deltas = [r["delta_WMC"] for r in rows if r["delta_WMC"] is not None]
        s_t = summarize_numeric(wmc_t)
        s_t1 = summarize_numeric(wmc_t1)
        s_d = summarize_numeric(deltas)
        n_pos = sum(1 for d in deltas if d > 0)
        n_neg = sum(1 for d in deltas if d < 0)
        n_zero = sum(1 for d in deltas if d == 0)
        for name, summary, extra in [
            ("WMC_t", s_t, {}),
            ("WMC_t1", s_t1, {}),
            ("delta_WMC", s_d, {
                "proportion_eq_0": pct(n_zero, len(deltas)),
                "proportion_gt_0": pct(n_pos, len(deltas)),
                "proportion_lt_0": pct(n_neg, len(deltas)),
            }),
        ]:
            wmc_rows.append({
                "scope": scope,
                "version_t": version_t,
                "version_t1": version_t1,
                "variable": name,
                "n": summary["n"],
                "mean": summary["mean"],
                "median": summary["median"],
                "sd": summary["stdev"],
                "min": summary["min"],
                "max": summary["max"],
                "q01": summary["q01"],
                "q05": summary["q05"],
                "q25": summary["q25"],
                "q75": summary["q75"],
                "q95": summary["q95"],
                "q99": summary["q99"],
                "proportion_eq_0": extra.get("proportion_eq_0"),
                "proportion_gt_0": extra.get("proportion_gt_0"),
                "proportion_lt_0": extra.get("proportion_lt_0"),
            })

    wmc_block("overall", "all", "all", panel_rows)
    for earlier, later in zip(VERSIONS, VERSIONS[1:]):
        wmc_block(
            "transition",
            earlier,
            later,
            [r for r in panel_rows if r["version_t"] == earlier and r["version_t1"] == later],
        )

    # Freeze decisions from observed prevalence among known class-versions.
    freeze_rows = []
    agg_prev = {row["construct"]: row for row in prevalence_rows if row["version"] == "ALL"}
    for spec in CONSTRUCTS:
        stats = agg_prev[spec["field"]]
        n_pos = stats["n_count_gt_0"]
        prev = stats["prevalence_among_known_pct"]
        if spec["field"] == "GodClass":
            decision = "HOLD OUT"
            reason = (
                "Observed PMD construct and retained in the canonical file, but the "
                "GodClass rule is WMC-contaminated. Do not use GodClass_count as a "
                "predictor in the first WMC model. Sensitivity-only."
            )
        elif spec["field"] == "LawOfDemeter":
            if prev is not None and prev >= 80:
                decision = "KEEP"
                reason = (
                    f"Binary presence is near-constant ({fmt_pct(prev)} of status-known "
                    "class-versions). KEEP the COUNT only; DROP any binary LoD indicator. "
                    "Do not treat LoD co-occurrence as an interaction."
                )
            else:
                decision = "KEEP"
                reason = "Usable as a count; binary still not recommended."
        elif n_pos < 20:
            decision = "DROP"
            reason = (
                f"Too sparse for a first pilot: {n_pos} positives among "
                f"{stats['n_status_known']} status-known class-versions."
            )
        elif prev is not None and prev < 1:
            decision = "DROP"
            reason = f"Prevalence among status-known is {fmt_pct(prev)} (<1%)."
        else:
            decision = "KEEP"
            reason = (
                f"{n_pos} positives ({fmt_pct(prev)} of status-known). Distinct from "
                "the cyclomatic/NCSS family and not a manufactured zero."
            )
        freeze_rows.append({
            "construct": spec["field"],
            "pmd_rule": spec["pmd_rule"],
            "decision": decision,
            "n_status_known": stats["n_status_known"],
            "n_count_gt_0": n_pos,
            "prevalence_among_known_pct": prev,
            "mean_count": stats["mean_count"],
            "median_count": stats["median_count"],
            "max_count": stats["max"],
            "wmc_contaminated": spec["wmc_contaminated"],
            "reason": reason,
        })

    dictionary_rows = build_dictionary()

    class_version_fields = [
        "project", "version", "class", "class_kind",
        "WMC", "LOC", "CBO", "NOM",
        "smell_status_known", "pmd_violation_rows",
    ]
    for spec in CONSTRUCTS:
        class_version_fields.extend([f"{spec['field']}_count", f"{spec['field']}_status_known"])

    panel_fields = [
        "project", "version_t", "version_t1", "class", "class_kind",
        "WMC_t", "WMC_t1", "delta_WMC",
        "LOC_t", "CBO_t", "NOM_t",
        "LOC_t1", "CBO_t1", "NOM_t1",
        "smell_status_known_t", "smell_status_known_t1",
        "pmd_violation_rows_t", "pmd_violation_rows_t1",
    ]
    for spec in CONSTRUCTS:
        panel_fields.extend([
            f"{spec['field']}_count_t", f"{spec['field']}_status_known_t",
            f"{spec['field']}_count_t1", f"{spec['field']}_status_known_t1",
        ])

    write_csv(OUT_CLASS_VERSION, class_version_fields, class_version_rows)
    write_csv(OUT_PANEL, panel_fields, panel_rows)
    write_csv(OUT_DICTIONARY, [
        "field", "meaning", "source", "observed_or_derived",
        "valid_range_or_values", "missingness_meaning", "modeling_notes",
    ], dictionary_rows)
    write_csv(OUT_PREVALENCE, [
        "construct", "pmd_rule", "version", "n_status_known", "n_count_gt_0",
        "prevalence_among_known_pct", "mean_count", "median_count", "p75", "p90",
        "max", "wmc_contaminated",
    ], prevalence_rows)
    write_csv(OUT_COOCCUR, [
        "construct_a", "construct_b", "n_status_known", "both_present", "a_only",
        "b_only", "neither", "joint_presence_pct", "terminology",
    ], pair_rows)
    write_csv(OUT_CONFIGS, [
        "configuration", "n_constructs_present", "n_class_versions", "pct_of_status_known",
    ], config_rows)
    write_csv(OUT_WMC, [
        "scope", "version_t", "version_t1", "variable", "n", "mean", "median", "sd",
        "min", "max", "q01", "q05", "q25", "q75", "q95", "q99",
        "proportion_eq_0", "proportion_gt_0", "proportion_lt_0",
    ], wmc_rows)
    write_csv(OUT_LOD, list(lod_rows[0].keys()), lod_rows)
    write_csv(OUT_VALIDATION, [
        "check", "version_t", "version_t1", "observed", "audit_expected", "match", "notes",
    ], validation_rows)
    write_csv(OUT_FREEZE, [
        "construct", "pmd_rule", "decision", "n_status_known", "n_count_gt_0",
        "prevalence_among_known_pct", "mean_count", "median_count", "max_count",
        "wmc_contaminated", "reason",
    ], freeze_rows)

    evidence = {
        "n_class_version_rows": len(class_version_rows),
        "n_unique_classes_class_version": len({r["class"] for r in class_version_rows}),
        "n_longitudinal_rows": len(panel_rows),
        "n_unique_classes_longitudinal": len({r["class"] for r in panel_rows}),
        "n_status_known_class_versions": len(known_cv),
        "n_status_unknown_class_versions": len(class_version_rows) - len(known_cv),
        "n_zero_candidate_smells_among_known": n_zero_smells,
        "n_exactly_one_candidate_among_known": n_one_smell,
        "n_two_or_more_candidates_among_known": n_multi_smell,
        "n_distinct_configurations": len(config_counter),
        "sensitivity_file_created": False,
        "sensitivity_reason": (
            "Zero-filling unmatched top-level classes would manufacture negatives. "
            "PMD files are violation-only and do not prove that every top-level quality "
            "class was analyzed. No sensitivity file was written."
        ),
        "validation_errors": errors,
    }
    with OUT_EVIDENCE.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2, default=str)

    write_report(
        class_version_rows=class_version_rows,
        panel_rows=panel_rows,
        known_cv=known_cv,
        prevalence_rows=prevalence_rows,
        pair_rows=pair_rows,
        config_rows=config_rows,
        wmc_rows=wmc_rows,
        lod_rows=lod_rows,
        freeze_rows=freeze_rows,
        validation_rows=validation_rows,
        n_zero_smells=n_zero_smells,
        n_one_smell=n_one_smell,
        n_multi_smell=n_multi_smell,
    )
    print(f"Wrote {OUT_CLASS_VERSION} ({len(class_version_rows)} rows)")
    print(f"Wrote {OUT_PANEL} ({len(panel_rows)} rows)")
    print(f"Wrote {OUT_DICTIONARY}")
    print(f"Wrote {OUT_PREVALENCE}")
    print(f"Wrote {OUT_COOCCUR}")
    print(f"Wrote {OUT_CONFIGS}")
    print(f"Wrote {OUT_WMC}")
    print(f"Wrote {OUT_LOD}")
    print(f"Wrote {OUT_VALIDATION}")
    print(f"Wrote {OUT_FREEZE}")
    print(f"Wrote {OUT_REPORT}")
    print("Sensitivity file: not created")


def build_dictionary():
    rows = [
        {"field": "project", "meaning": "Project name", "source": "fixed from directory 10 Signal-Android-master", "observed_or_derived": "derived", "valid_range_or_values": "Signal-Android", "missingness_meaning": "never missing", "modeling_notes": "Constant in this pilot."},
        {"field": "version / version_t / version_t1", "meaning": "CSIQ annual snapshot label", "source": "quality and smell CSV filenames", "observed_or_derived": "observed", "valid_range_or_values": "2016-1 .. 2021-1", "missingness_meaning": "never missing", "modeling_notes": "version_t1 is the consecutive next snapshot, not a calendar interpolation."},
        {"field": "class", "meaning": "Top-level type identity", "source": "quality QualifiedName", "observed_or_derived": "observed", "valid_range_or_values": "Java-style qualified name with lowercase parent package segment", "missingness_meaning": "never missing in these files", "modeling_notes": "Exact string match across versions. Rename looks like death+birth."},
        {"field": "class_kind", "meaning": "Row-kind from QualifiedName shape", "source": "same classifier as csiq_pilot_audit.py", "observed_or_derived": "derived", "valid_range_or_values": "top_level only in these files", "missingness_meaning": "never missing", "modeling_notes": "Inner/nested, anonymous, method, field, package rows were excluded."},
        {"field": "WMC / WMC_t / WMC_t1", "meaning": "Weighted Method Count (first WMC column)", "source": "quality_attributes CSV", "observed_or_derived": "observed", "valid_range_or_values": "non-negative numeric", "missingness_meaning": "empty if the quality cell was non-numeric", "modeling_notes": "Viable lagged outcome. Do not put cyclomatic/NCSS PMD rules on the RHS."},
        {"field": "delta_WMC", "meaning": "WMC_t1 - WMC_t", "source": "derived", "observed_or_derived": "derived", "valid_range_or_values": "numeric difference", "missingness_meaning": "empty if either WMC is missing", "modeling_notes": "Not 'software corrosion'. Zero is common and is a real measured value, not missingness."},
        {"field": "LOC_t / CBO_t / NOM_t", "meaning": "Size, coupling, method-count controls at t", "source": "quality first LOC, CBO, NOM columns", "observed_or_derived": "observed", "valid_range_or_values": "numeric", "missingness_meaning": "empty if non-numeric", "modeling_notes": "Candidate controls. NOM collinear with TooManyMethods (not used). CBO collinear with CouplingBetweenObjects (not used)."},
        {"field": "LOC_t1 / CBO_t1 / NOM_t1", "meaning": "Same metrics at t+1", "source": "quality CSV of version_t1", "observed_or_derived": "observed", "valid_range_or_values": "numeric", "missingness_meaning": "empty if non-numeric", "modeling_notes": "Not default RHS. Stored so WMC_t1 can be checked against the subsequent version file."},
        {"field": "smell_status_known[_t/_t1]", "meaning": "1 if this class key appears in that version's PMD CSV", "source": "exact join to codesmells CSV", "observed_or_derived": "derived", "valid_range_or_values": "0 or 1", "missingness_meaning": "never missing; 0 means unknown, not clean", "modeling_notes": "Restrict first models to 1, or treat 0 as out-of-scope. Never recode 0 as 'no smell'."},
        {"field": "pmd_violation_rows[_t/_t1]", "meaning": "All design.xml violation rows for this class key, all rules", "source": "codesmells CSV", "observed_or_derived": "derived", "valid_range_or_values": "integer >= 1 when known", "missingness_meaning": "empty iff smell_status_known=0", "modeling_notes": "Diagnostic. Includes non-candidate rules such as cyclomatic thresholds."},
    ]
    for spec in CONSTRUCTS:
        missing = "empty iff status_known=0. Empty is unknown, not zero."
        notes = "Count is 0 only when PMD observed the class and this rule had no violation."
        if spec["wmc_contaminated"]:
            notes = (
                "WMC_contaminated=true. PMD GodClass uses size/complexity (including WMC-like) "
                "metrics. HOLD OUT of the first WMC model. Keep in the file for sensitivity."
            )
        if spec["field"] == "LawOfDemeter":
            notes = (
                "Do not binarize by default. ~86% of PMD-observed classes have count>0. "
                "If used, use the count. Co-occurrence with other constructs is not an interaction."
            )
        rows.append({
            "field": f"{spec['field']}_count[_t/_t1]",
            "meaning": f"PMD {spec['pmd_rule']} violation count ({spec['label']})",
            "source": f"codesmells CSV Rule={spec['pmd_rule']}",
            "observed_or_derived": "derived",
            "valid_range_or_values": "integer >= 0 when known; empty when unknown",
            "missingness_meaning": missing,
            "modeling_notes": notes,
        })
        rows.append({
            "field": f"{spec['field']}_status_known[_t/_t1]",
            "meaning": "Whether this construct's count is identified from a PMD class key",
            "source": "same as smell_status_known; one PMD export covers all six constructs",
            "observed_or_derived": "derived",
            "valid_range_or_values": "0 or 1; identical to smell_status_known in this dataset",
            "missingness_meaning": "never missing",
            "modeling_notes": "Redundant with smell_status_known given a single design.xml dump. Kept so a construct is never silently treated as observed.",
        })
    rows.append({
        "field": "GodClass WMC_contaminated",
        "meaning": "Metadata flag, not a panel column",
        "source": "PMD GodClass definition in codesmells.csv",
        "observed_or_derived": "derived",
        "valid_range_or_values": "true",
        "missingness_meaning": "n/a",
        "modeling_notes": "Explicitly true. Do not put GodClass_count on the RHS of the first WMC model.",
    })
    return rows


def write_report(
    class_version_rows,
    panel_rows,
    known_cv,
    prevalence_rows,
    pair_rows,
    config_rows,
    wmc_rows,
    lod_rows,
    freeze_rows,
    validation_rows,
    n_zero_smells,
    n_one_smell,
    n_multi_smell,
):
    n_cv = len(class_version_rows)
    n_unique_cv = len({r["class"] for r in class_version_rows})
    n_panel = len(panel_rows)
    n_unique_panel = len({r["class"] for r in panel_rows})
    n_known = len(known_cv)
    n_unknown = n_cv - n_known
    lod = lod_rows[0]
    freeze_lookup = {r["construct"]: r for r in freeze_rows}
    keep = [r for r in freeze_rows if r["decision"] == "KEEP"]
    hold = [r for r in freeze_rows if r["decision"] == "HOLD OUT"]
    drop = [r for r in freeze_rows if r["decision"] == "DROP"]

    per_version = []
    for version in VERSIONS:
        rows = [r for r in class_version_rows if r["version"] == version]
        known = [r for r in rows if r["smell_status_known"] == 1]
        per_version.append({
            "version": version,
            "n": len(rows),
            "known": len(known),
            "unknown": len(rows) - len(known),
            "numeric_wmc": sum(1 for r in rows if r["WMC"] is not None),
        })

    lines = []
    add = lines.append
    add("# Signal Android pilot panel report")
    add("")
    add("Generated by `build_signal_android_pilot_panel.py`. No regression or interaction model was estimated.")
    add("")
    add("This file constructs a canonical dataset. It does **not** establish that unmatched quality classes are clean.")
    add("")
    add("## A. Construction methodology")
    add("")
    add("- Universe: Signal Android CSIQ quality CSVs, six versions (`2016-1` … `2021-1`).")
    add("- One class-version row per top-level `QualifiedName` that appears in the quality file.")
    add("- One longitudinal row per top-level `QualifiedName` that exists in two consecutive versions.")
    add("- Smell counts are attached by exact key match to that **same** version's PMD CSV.")
    add("- Unmatched smell status is stored as `status_known=0` and **NULL counts**, never as zero.")
    add("- Candidate rules only: `DataClass`, `ExcessiveMethodLength`, `ExcessiveParameterList`, `MutableStaticState`, `LawOfDemeter`, `GodClass`.")
    add("- Cyclomatic/NCSS/Cognitive/NPath rules are not encoded as predictors.")
    add("- A sensitivity zero-filled file was **not** created.")
    add("")
    add("## B. Exact class join methodology")
    add("")
    add("```")
    add("smell_key = Package + '.' + basename(File)   # extension stripped")
    add("match iff smell_key == quality.QualifiedName")
    add("```")
    add("")
    add("No fuzzy match, no inner-class projection, no package-only match. Test-path PMD rows join if and only if the key equals a quality `QualifiedName`.")
    add("")
    add("## C. Top-level filtering methodology")
    add("")
    add("The classifier is identical to `csiq_pilot_audit.py`:")
    add("")
    add("- Exclude `<Package>`, `<Method>`, `<Field>`, `<Anonymous>`, other `<...>` rows.")
    add("- Exclude names with no `.` (project/commit header).")
    add("- Exclude inner/nested types whose parent segment starts with an uppercase letter (`Outer.Inner`).")
    add("- Retain only `top_level`.")
    add("")
    add("This is a QualifiedName-shape rule, not a guess about source files.")
    add("")
    add("## D. Longitudinal matching methodology")
    add("")
    add("Consecutive snapshots only: 2016-1→2017-1, …, 2020-1→2021-1.")
    add("Match = exact `QualifiedName` string equality on top-level classes present in both quality files.")
    add("`WMC_t1` is read from the later quality file. `delta_WMC = WMC_t1 - WMC_t` when both are numeric.")
    add("")
    add("## E. Smell-status-known semantics")
    add("")
    add("| status_known | count | Meaning |")
    add("| ---: | --- | --- |")
    add("| 1 | integer ≥ 0 | Class key is in that version's PMD export. 0 means this candidate rule had no violation. It does **not** mean the class has no other PMD rules. |")
    add("| 0 | NULL | Class key is not in the PMD export. Smell status is unknown. **Not clean. Not zero.** |")
    add("")
    add("Construct-specific `*_status_known` flags equal `smell_status_known` because one `design.xml` dump covers all six rules.")
    add("")
    add("Worked examples preserved in the file:")
    add("")
    add("- PMD observed, DataClass absent → `DataClass_count=0`, `DataClass_status_known=1`")
    add("- No PMD class record → `DataClass_count` empty, `DataClass_status_known=0`")
    add("")
    add(f"Class-version observations with fully known smell status: **{n_known}** of {n_cv} ({fmt_pct(pct(n_known, n_cv))}).")
    add("")
    add("## F. Per-version panel sizes")
    add("")
    add("| Version | Top-level class-versions | Numeric WMC | smell_status_known=1 | Unknown |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for row in per_version:
        add(f"| {row['version']} | {row['n']} | {row['numeric_wmc']} | {row['known']} | {row['unknown']} |")
    add("")
    add("| Transition | Longitudinal rows |")
    add("| --- | ---: |")
    for earlier, later in zip(VERSIONS, VERSIONS[1:]):
        n = sum(1 for r in panel_rows if r["version_t"] == earlier)
        add(f"| {earlier} → {later} | {n} |")
    add("")
    add(f"- Class-version rows: {n_cv}")
    add(f"- Unique classes in class-version file: {n_unique_cv}")
    add(f"- Longitudinal rows: {n_panel}")
    add(f"- Unique classes in longitudinal file: {n_unique_panel}")
    add("")
    add("## G. Smell prevalence")
    add("")
    add("Restricted to `smell_status_known=1`. Unmatched quality classes are not in the denominator and are not called clean.")
    add("")
    add("| Construct | PMD rule | Known | count>0 | Prevalence | Mean count | Median | p75 | p90 | Max | WMC contaminated |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in prevalence_rows:
        if row["version"] != "ALL":
            continue
        add(
            f"| {row['construct']} | `{row['pmd_rule']}` | {row['n_status_known']} | {row['n_count_gt_0']} | "
            f"{fmt_pct(row['prevalence_among_known_pct'])} | {fmt_num(row['mean_count'], 3)} | "
            f"{fmt_num(row['median_count'], 3)} | {fmt_num(row['p75'], 3)} | {fmt_num(row['p90'], 3)} | "
            f"{fmt_num(row['max'])} | {row['wmc_contaminated']} |"
        )
    add("")
    add("Per-version prevalence is in `signal_android_pilot_smell_prevalence.csv`.")
    add("")
    add(f"Among the {n_known} status-known class-versions, number of the six candidates with count>0:")
    add("")
    add(f"- zero candidate constructs: {n_zero_smells}")
    add(f"- exactly one: {n_one_smell}")
    add(f"- two or more: {n_multi_smell}")
    add("")
    add("Zero candidate constructs still means the class had some other PMD design.xml violation (otherwise it would not be in the smell file). It is **not** a clean class.")
    add("")
    add("### Law of Demeter count distribution (status-known only)")
    add("")
    add(f"- n = {lod['n']}; zero = {lod['n_zero']}; one = {lod['n_one']}; multiple = {lod['n_multiple']}")
    add(f"- proportion > 0 = {fmt_pct(lod['proportion_gt_0_pct'])}")
    add(f"- mean = {fmt_num(lod['mean'], 3)}; median = {fmt_num(lod['median'], 3)}; sd = {fmt_num(lod['stdev'], 3)}")
    add(f"- q25 = {fmt_num(lod['q25'])}; q75 = {fmt_num(lod['q75'])}; q90 = {fmt_num(lod['q90'])}; q95 = {fmt_num(lod['q95'])}; q99 = {fmt_num(lod['q99'])}; max = {fmt_num(lod['max'])}")
    add("")
    add("## H. Co-occurrence statistics")
    add("")
    add("These are **joint presence / configurations**, not interaction effects. `neither` is counted only inside `status_known=1`.")
    add("")
    add("| A | B | Both | A only | B only | Neither | Joint presence % |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in pair_rows:
        add(
            f"| {row['construct_a']} | {row['construct_b']} | {row['both_present']} | "
            f"{row['a_only']} | {row['b_only']} | {row['neither']} | {fmt_pct(row['joint_presence_pct'])} |"
        )
    add("")
    add(f"Distinct observed multi-label configurations of the six candidates: **{len(config_rows)}** (including the all-zero-candidate pattern).")
    add("Full configuration table: `signal_android_pilot_configurations.csv`.")
    add("")
    add("Top configurations:")
    add("")
    add("| Configuration | n constructs | n class-versions | % of known |")
    add("| --- | ---: | ---: | ---: |")
    for row in config_rows[:12]:
        add(f"| {md_escape(row['configuration'])} | {row['n_constructs_present']} | {row['n_class_versions']} | {fmt_pct(row['pct_of_status_known'])} |")
    add("")
    add("## I. WMC distributions")
    add("")
    add("Longitudinal panel, numeric values only.")
    add("")
    add("| Scope | Variable | n | Mean | Median | SD | Min | Max | q01 | q05 | q25 | q75 | q95 | q99 | =0 | >0 | <0 |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in wmc_rows:
        if row["scope"] == "overall" or row["variable"] == "delta_WMC":
            scope = row["scope"] if row["scope"] == "overall" else f"{row['version_t']}→{row['version_t1']}"
            add(
                f"| {scope} | {row['variable']} | {row['n']} | {fmt_num(row['mean'], 3)} | {fmt_num(row['median'], 3)} | "
                f"{fmt_num(row['sd'], 3)} | {fmt_num(row['min'])} | {fmt_num(row['max'])} | {fmt_num(row['q01'], 3)} | "
                f"{fmt_num(row['q05'], 3)} | {fmt_num(row['q25'], 3)} | {fmt_num(row['q75'], 3)} | {fmt_num(row['q95'], 3)} | "
                f"{fmt_num(row['q99'], 3)} | {fmt_pct(row['proportion_eq_0'])} | {fmt_pct(row['proportion_gt_0'])} | "
                f"{fmt_pct(row['proportion_lt_0'])} |"
            )
    add("")
    add("Complete WMC_t / WMC_t1 / delta tables, including per-transition levels: `signal_android_pilot_wmc_descriptives.csv`.")
    add("")
    add("## J. Validation against previous audit")
    add("")
    add("All checks in `signal_android_pilot_validation.csv` matched. Hard-stop errors: none.")
    add("")
    add("Compared against the **top-level** columns of the previous audit, which is the universe of this panel:")
    add("")
    add("| Transition | Shared top-level | Valid WMC | Changed WMC | Match |")
    add("| --- | ---: | ---: | ---: | --- |")
    for (earlier, later), expected in AUDIT_TRANSITIONS.items():
        add(
            f"| {earlier} → {later} | {expected['shared_top_level']} | {expected['valid_wmc']} | "
            f"{expected['changed']} | yes |"
        )
    add("")
    add("The earlier audit also reported larger all-class-level shared counts (e.g. 919 vs 597 for 2016→2017). Those extra rows are inner/nested types. They are **not** in this panel by design. Those numbers were not overwritten.")
    add("")
    add("Also verified: no duplicate `(version, class)`; no duplicate `(version_t, version_t1, class)`; no non-top-level leakage; no accidental zero-fill; smell counts attached from the matching version; `WMC_t1` read from the subsequent quality file.")
    add("")
    add("## K. Anomalies discovered")
    add("")
    add("1. PMD-observed classes are a selected sample: they had at least one `design.xml` violation of any rule. A class with zero of the six candidates is not clean.")
    add("2. LawOfDemeter dominates joint-presence tables. That is co-occurrence, not an interaction.")
    add("3. Early-year PMD paths often lack `src/main`, so test/main tagging remains coarse. Tests were not dropped.")
    add("4. `ConversationAdapter`-style unmatched PMD keys from the previous audit are outside this quality-driven panel and stay out.")
    add("5. No sensitivity zero-fill file. Command.txt shows PMD was aimed at the repo, but the CSV does not list analyzed-but-clean files. That is not enough to recode unknown as 0.")
    add("")
    add("## L. Recommended modeling-ready variables")
    add("")
    add("If modeling proceeds later, on rows with `smell_status_known_t=1` and numeric `WMC_t`/`WMC_t1`:")
    add("")
    add("- Outcome: `WMC_t1` (or `delta_WMC` as a descriptive companion, not as 'corrosion')")
    add("- Lag: `WMC_t`")
    add("- Controls to consider: `LOC_t`, `CBO_t`, `NOM_t`")
    add("- Individual-smell information:")
    keep_rhs = [r["construct"] for r in keep if r["construct"] != "GodClass"]
    for name in keep_rhs:
        add(f"  - `{name}_count_t`")
    add("")
    add("Construct freeze:")
    add("")
    add("| Construct | Decision | Positives among known | Prevalence | Reason (short) |")
    add("| --- | --- | ---: | ---: | --- |")
    for row in freeze_rows:
        add(
            f"| {row['construct']} | **{row['decision']}** | {row['n_count_gt_0']} | "
            f"{fmt_pct(row['prevalence_among_known_pct'])} | {md_escape(row['reason'][:180])} |"
        )
    add("")
    add("KEEP / HOLD OUT / DROP counts: "
        f"KEEP={len(keep)}, HOLD OUT={len(hold)}, DROP={len(drop)}.")
    add("")
    add("## M. Variables that MUST NOT be used for the first WMC model")
    add("")
    add("- `GodClass_count_*` — WMC-contaminated (`WMC_contaminated=true`).")
    add("- Binary `LawOfDemeter` — near-constant among PMD-observed classes.")
    add("- Any cyclomatic / NCSS / CognitiveComplexity / NPath / AvoidDeeplyNestedIfStmts rule.")
    add("- `CouplingBetweenObjects` / `ExcessiveImports` together with `CBO_t`.")
    add("- Manufactured zeros for `smell_status_known=0`.")
    add("- Configuration dummies interpreted as 'interaction effects' without a Model A vs Model B comparison.")
    add("- `delta_WMC` labeled as software corrosion.")
    add("")
    add("Remaining issue before modeling: the estimand must be restricted to PMD-observed top-level classes (`smell_status_known_t=1`), or a separately justified (currently unjustified) complete-universe assumption is required. That is a sample-definition choice, not a license to fill zeros.")
    add("")
    add("## Generated files")
    add("")
    add("- `signal_android_pilot_class_version.csv`")
    add("- `signal_android_pilot_panel.csv`")
    add("- `signal_android_pilot_panel_dictionary.csv`")
    add("- `signal_android_pilot_smell_prevalence.csv`")
    add("- `signal_android_pilot_cooccurrence.csv`")
    add("- `signal_android_pilot_configurations.csv`")
    add("- `signal_android_pilot_wmc_descriptives.csv`")
    add("- `signal_android_pilot_lod_distribution.csv`")
    add("- `signal_android_pilot_validation.csv`")
    add("- `signal_android_pilot_construct_decisions.csv`")
    add("- `signal_android_pilot_panel_report.md`")
    add("- `signal_android_pilot_panel_evidence.json`")
    add("")
    add("Not generated: `signal_android_pilot_panel_sensitivity.csv` (zero-fill not defensible).")
    add("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
