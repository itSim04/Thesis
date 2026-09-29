"""Direct, non-modeling inventory of the supplied CSIQ-style data folder."""
import csv
import itertools
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2**31 - 1)

ROOT = Path(__file__).parent / "Dataset"
OUTPUT = Path(__file__).parent / "csiq_archaeology_evidence.json"


def csv_rows(path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.reader(handle)


def header_and_count(path):
    reader = csv_rows(path)
    header = next(reader, [])
    return header, sum(1 for _ in reader)


def project_version(path, section):
    relative = path.relative_to(ROOT / section)
    return relative.parent.name, path.stem


def valid_quality_identifier(identifier):
    identifier = identifier.strip()
    return bool(identifier and not identifier.startswith("<") and identifier.lower() != "null")


def load_quality(path):
    reader = csv_rows(path)
    header = next(reader)
    classes = {}
    row_count = 0
    structural_or_blank = 0
    for row in reader:
        row_count += 1
        identifier = row[0].strip() if row else ""
        if not valid_quality_identifier(identifier):
            structural_or_blank += 1
            continue
        # Keep the first occurrence: duplicates are reported separately below.
        classes.setdefault(identifier, row)
    return header, classes, row_count, structural_or_blank


def smell_key(package, file_path):
    name = re.split(r"[\\/]", file_path.strip())[-1]
    stem = re.sub(r"\.[^.]+$", "", name)
    package = package.strip()
    if not stem:
        return ""
    return f"{package}.{stem}" if package and package.lower() != "null" else stem


def scan_smells(path, quality_identifiers, raw_events, class_presence, pair_presence):
    reader = csv_rows(path)
    header = next(reader)
    per_class = defaultdict(Counter)
    row_count = 0
    for row in reader:
        row_count += 1
        if len(row) < 8:
            continue
        key = smell_key(row[1], row[2])
        rule = row[7].strip()
        if key and rule:
            per_class[key][rule] += 1
            raw_events[rule] += 1

    matched = set(per_class).intersection(quality_identifiers)
    for key in matched:
        rules = sorted(per_class[key])
        class_presence.update(rules)
        pair_presence.update(itertools.combinations(rules, 2))
    return header, row_count, per_class, matched


quality_paths = sorted((ROOT / "quality_attributes").rglob("*.csv"))
smell_paths = sorted((ROOT / "codesmells" / "csv").rglob("*.csv"))
quality_by_version = {project_version(path, "quality_attributes"): path for path in quality_paths}
smell_by_version = {project_version(path, "codesmells/csv"): path for path in smell_paths}
paired_versions = sorted(set(quality_by_version) & set(smell_by_version))

all_csv = sorted(ROOT.rglob("*.csv"))
section_counts = Counter(
    path.relative_to(ROOT).parts[0] if len(path.relative_to(ROOT).parts) > 1 else "root"
    for path in all_csv
)

evidence = {
    "source_root": str(ROOT),
    "archive_materialization": {
        "top_level_entries": sorted(path.name for path in ROOT.iterdir()),
        "files_total": sum(1 for path in ROOT.rglob("*") if path.is_file()),
        "csv_files_total": len(all_csv),
        "csv_files_by_top_level_section": dict(section_counts),
        "csv_bytes_total": sum(path.stat().st_size for path in all_csv),
    },
    "root_tables": {},
    "schemas": {},
    "join": {},
    "longitudinal": {},
    "smells": {},
}

for relative in [
    "repositories.csv",
    "versions.csv",
    "codesmells.csv",
    "attribute-details.csv",
    "synthesized/class_level_quality_code_smell_mdi.csv",
]:
    header, count = header_and_count(ROOT / relative)
    evidence["root_tables"][relative] = {"rows": count, "columns": len(header), "header": header}

issue_paths = sorted((ROOT / "issues").rglob("*.csv"))
issue_counts = [header_and_count(path)[1] for path in issue_paths]
evidence["issues"] = {
    "files": len(issue_paths),
    "rows_total": sum(issue_counts),
    "rows_min": min(issue_counts),
    "rows_max": max(issue_counts),
    "header": header_and_count(issue_paths[0])[0],
}

repositories_header, repositories_rows = next(csv_rows(ROOT / "repositories.csv")), None
versions_reader = csv_rows(ROOT / "versions.csv")
versions_header = next(versions_reader)
version_registry = defaultdict(list)
for row in versions_reader:
    if len(row) >= 3:
        version_registry[row[1].strip()].append(row[2].strip())
evidence["version_registry"] = dict(version_registry)

quality_cache = {}
quality_schema_counts = Counter()
smell_schema_counts = Counter()
per_version = []
raw_events = Counter()
class_presence = Counter()
pair_presence = Counter()
per_project = defaultdict(lambda: {
    "quality_rows": 0,
    "quality_classes": 0,
    "smell_rows": 0,
    "smell_file_classes": 0,
    "exact_matched_classes": 0,
    "versions": [],
    "pair_presence": Counter(),
})

for key in paired_versions:
    project, version = key
    quality_header, quality_classes, quality_rows, structural_rows = load_quality(quality_by_version[key])
    quality_cache[key] = (quality_header, quality_classes)
    smell_header, smell_rows, per_class, matched = scan_smells(
        smell_by_version[key], quality_classes, raw_events, class_presence, pair_presence
    )
    quality_schema_counts[tuple(quality_header)] += 1
    smell_schema_counts[tuple(smell_header)] += 1
    project_summary = per_project[project]
    project_summary["quality_rows"] += quality_rows
    project_summary["quality_classes"] += len(quality_classes)
    project_summary["smell_rows"] += smell_rows
    project_summary["smell_file_classes"] += len(per_class)
    project_summary["exact_matched_classes"] += len(matched)
    project_summary["versions"].append(version)
    for class_key in matched:
        project_summary["pair_presence"].update(itertools.combinations(sorted(per_class[class_key]), 2))
    per_version.append({
        "project_directory": project,
        "version": version,
        "quality_rows": quality_rows,
        "quality_structural_or_blank_rows": structural_rows,
        "quality_unique_class_identifiers": len(quality_classes),
        "smell_rows": smell_rows,
        "smell_file_classes": len(per_class),
        "exact_identifier_matches": len(matched),
        "smell_class_match_rate_pct": round(100 * len(matched) / len(per_class), 2) if per_class else None,
    })

evidence["schemas"]["quality_attributes"] = [
    {"files": count, "columns": len(header), "header": list(header)}
    for header, count in quality_schema_counts.items()
]
evidence["schemas"]["codesmells_csv"] = [
    {"files": count, "columns": len(header), "header": list(header)}
    for header, count in smell_schema_counts.items()
]
evidence["join"] = {
    "paired_version_files": len(paired_versions),
    "unpaired_quality_files": sorted("/".join(key) for key in set(quality_by_version) - set(smell_by_version)),
    "unpaired_smell_files": sorted("/".join(key) for key in set(smell_by_version) - set(quality_by_version)),
    "derivation": "smell key = Package + '.' + basename(File) without extension; exact match to quality QualifiedName",
    "per_version": per_version,
}

def version_order(value):
    match = re.match(r"(\d+)-(\d+)$", value)
    return tuple(map(int, match.groups())) if match else (999999, 999999)

for project, summary in per_project.items():
    summary["versions"].sort(key=version_order)
    summary["smell_class_match_rate_pct"] = round(
        100 * summary["exact_matched_classes"] / summary["smell_file_classes"], 2
    ) if summary["smell_file_classes"] else None
    pairs = summary.pop("pair_presence")
    summary["nonzero_pair_types"] = len(pairs)
    summary["pair_instances"] = sum(pairs.values())
    summary["pairs_ge_10"] = sum(value >= 10 for value in pairs.values())

longitudinal = {}
for project, summary in per_project.items():
    transitions = []
    total_shared = total_valid_wmc = total_changed_wmc = 0
    versions = summary["versions"]
    for earlier, later in zip(versions, versions[1:]):
        earlier_header, earlier_classes = quality_cache[(project, earlier)]
        later_header, later_classes = quality_cache[(project, later)]
        earlier_wmc = earlier_header.index("WMC")
        later_wmc = later_header.index("WMC")
        shared = set(earlier_classes) & set(later_classes)
        valid_wmc = changed_wmc = valid_complexity = changed_complexity = 0
        for identifier in shared:
            before, after = earlier_classes[identifier], later_classes[identifier]
            try:
                before_wmc = float(before[earlier_wmc])
                after_wmc = float(after[later_wmc])
                valid_wmc += 1
                changed_wmc += before_wmc != after_wmc
            except (ValueError, IndexError):
                pass
            if len(before) > 3 and len(after) > 3 and before[2].strip() and after[2].strip():
                valid_complexity += 1
                changed_complexity += before[2].strip() != after[2].strip()
        transitions.append({
            "from": earlier,
            "to": later,
            "shared_exact_qualified_names": len(shared),
            "valid_numeric_WMC": valid_wmc,
            "changed_WMC": changed_wmc,
            "valid_complexity_category": valid_complexity,
            "changed_complexity_category": changed_complexity,
        })
        total_shared += len(shared)
        total_valid_wmc += valid_wmc
        total_changed_wmc += changed_wmc
    longitudinal[project] = {
        "transitions": transitions,
        "shared_exact_qualified_names_total": total_shared,
        "valid_numeric_WMC_total": total_valid_wmc,
        "changed_WMC_total": total_changed_wmc,
    }
evidence["longitudinal"] = longitudinal

pair_values = sorted(pair_presence.values())
evidence["smells"] = {
    "raw_rule_events": sum(raw_events.values()),
    "raw_event_top_30": raw_events.most_common(30),
    "rules_with_matched_class_presence": len(class_presence),
    "matched_class_version_presence_top_30": class_presence.most_common(30),
    "nonzero_pair_types": len(pair_presence),
    "pair_instances": sum(pair_presence.values()),
    "pair_count_distribution": {
        "min": min(pair_values),
        "median": statistics.median(pair_values),
        "p75": statistics.quantiles(pair_values, n=4)[2],
        "p90": statistics.quantiles(pair_values, n=10)[8],
        "max": max(pair_values),
        "pairs_ge_2": sum(value >= 2 for value in pair_values),
        "pairs_ge_5": sum(value >= 5 for value in pair_values),
        "pairs_ge_10": sum(value >= 10 for value in pair_values),
        "pairs_ge_30": sum(value >= 30 for value in pair_values),
        "pairs_ge_100": sum(value >= 100 for value in pair_values),
    },
    "pair_top_40": [
        {"rule_a": a, "rule_b": b, "matched_class_versions": count}
        for (a, b), count in pair_presence.most_common(40)
    ],
    "per_project": per_project,
}

with OUTPUT.open("w", encoding="utf-8") as handle:
    json.dump(evidence, handle, ensure_ascii=False, indent=2)

print(f"Wrote {OUTPUT}")
