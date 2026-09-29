"""CSIQ V5 dataset archaeology for the Signal Android smell-interaction pilot.

This script does not fit or evaluate regression models. It:
  1. Reads the 47 PMD/CSIQ rules from codesmells.csv
  2. Classifies each rule from in-file definitions (not as 47 independent smells)
  3. Validates Signal Android smell/quality join coverage per version
  4. Tests whether zero-smell classes can be identified
  5. Inspects longitudinal WMC transitions
  6. Writes classification, coverage, WMC, schema, and Markdown audit files

Run from the repository root:
    python csiq_pilot_audit.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(2**31 - 1)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"
SIGNAL_DIR = "10 Signal-Android-master"
VERSIONS = ["2016-1", "2017-1", "2018-1", "2019-1", "2020-1", "2021-1"]

OUT_RULES = ROOT / "csiq_smell_rule_classification.csv"
OUT_COVERAGE = ROOT / "csiq_signal_android_coverage.csv"
OUT_WMC = ROOT / "csiq_signal_android_wmc_transitions.csv"
OUT_SCHEMA = ROOT / "csiq_pilot_proposed_schema.csv"
OUT_AUDIT = ROOT / "csiq_pilot_audit.md"
OUT_JSON = ROOT / "csiq_pilot_audit_evidence.json"


# ---------------------------------------------------------------------------
# Conceptual classification of the 47 catalog rules.
# Representation types follow the thesis brief:
#   a) design/code smell
#   b) complexity/metric threshold
#   c) structural/property rule
#   d) something else
# Independent-construct decisions are conceptual. Prevalence may still
# exclude a conceptually valid construct from the first pilot.
# ---------------------------------------------------------------------------
RULE_CLASSIFICATION = {
    "LawOfDemeter": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "CouplingBetweenObjects; ExcessiveImports; quality CBO",
        "independent": "conditional",
        "construct_group": "coupling_lod",
        "wmc_risk": "low",
        "reason": (
            "Classic coupling smell (only talk to friends). Distinct from size/"
            "complexity constructs. Conditional because catalog frequency is"
            " extreme: if nearly every analyzed class violates it, the indicator"
            " has no useful variance. Do not treat LoD co-occurrence with other"
            " rules as an interaction effect."
        ),
    },
    "SignatureDeclareThrowsException": {
        "type_code": "c",
        "type_label": "structural/property rule",
        "overlap": "AvoidUncheckedExceptionsInSignatures; AvoidThrowingRawExceptionTypes",
        "independent": "no",
        "construct_group": "exception_api_convention",
        "wmc_risk": "low",
        "reason": (
            "API signature convention (do not throw java.lang.Exception), not a"
            " design smell of class responsibility or structure. Group with other"
            " exception-signature rules if studied at all; not a pilot construct."
        ),
    },
    "StdCyclomaticComplexity": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "CyclomaticComplexity; ModifiedCyclomaticComplexity; CognitiveComplexity; NPathComplexity; quality WMC/MCC",
        "independent": "no",
        "construct_group": "cyclomatic_complexity_family",
        "wmc_risk": "high",
        "reason": (
            "Threshold on standard cyclomatic complexity. Near-duplicate of"
            " CyclomaticComplexity and ModifiedCyclomaticComplexity. Using any"
            " member of this family to explain WMC is tautological: WMC is the"
            " weighted/McCabe complexity of the class."
        ),
    },
    "CyclomaticComplexity": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "StdCyclomaticComplexity; ModifiedCyclomaticComplexity; CognitiveComplexity; NPathComplexity; quality WMC/MCC",
        "independent": "no",
        "construct_group": "cyclomatic_complexity_family",
        "wmc_risk": "high",
        "reason": (
            "Catalog maps this onto 'Complex Class', but the rule is a method"
            " cyclomatic-complexity threshold, not an independent smell"
            " construct. Do not use as a predictor of WMC. If complexity smells"
            " are studied later, keep a single representative, not all three"
            " cyclomatic variants."
        ),
    },
    "ModifiedCyclomaticComplexity": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "CyclomaticComplexity; StdCyclomaticComplexity; SwitchDensity",
        "independent": "no",
        "construct_group": "cyclomatic_complexity_family",
        "wmc_risk": "high",
        "reason": (
            "Same construct as CyclomaticComplexity except switch is one decision"
            " point. Different operationalization, not a different smell."
        ),
    },
    "ImmutableField": {
        "type_code": "d",
        "type_label": "something else (immutability suggestion; not a defect)",
        "overlap": "FinalFieldCouldBeStatic; MutableStaticState; SingularField",
        "independent": "no",
        "construct_group": "immutability_suggestion",
        "wmc_risk": "low",
        "reason": (
            "PMD reports private fields that never change after initialization so"
            " they could be declared final. That is a style/immutability"
            " opportunity, not a smell. Treating presence as a smell would invert"
            " the construct."
        ),
    },
    "CognitiveComplexity": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "CyclomaticComplexity family; AvoidDeeplyNestedIfStmts; CollapsibleIfStatements",
        "independent": "no",
        "construct_group": "cognitive_nesting_complexity",
        "wmc_risk": "high",
        "reason": (
            "Human-readability complexity with nesting penalties. Related to but"
            " not identical to cyclomatic complexity. Still a complexity metric"
            " threshold and still contaminated as a WMC predictor. Do not treat"
            " as a separate smell from the cyclomatic family in the first pilot."
        ),
    },
    "AvoidCatchingGenericException": {
        "type_code": "d",
        "type_label": "something else (exception-handling convention)",
        "overlap": "AvoidThrowingRawExceptionTypes; SignatureDeclareThrowsException; ExceptionAsFlowControl",
        "independent": "no",
        "construct_group": "exception_handling_convention",
        "wmc_risk": "low",
        "reason": (
            "Catch-clause convention, not a class-level design smell. Exception"
            " rules in this catalog are many overlapping operationalizations of"
            " exception-API style, not distinct smells."
        ),
    },
    "TooManyMethods": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "GodClass; ExcessiveClassLength; ExcessivePublicCount; NcssTypeCount; quality NOM",
        "independent": "conditional",
        "construct_group": "large_class_responsibility",
        "wmc_risk": "medium",
        "reason": (
            "Responsibility/size smell (too many methods). Conceptually close to"
            " GodClass and Large Class. Conditional: include only if GodClass is"
            " dropped because of WMC contamination, and treat as one 'large class'"
            " construct rather than three. Correlated with WMC/NOM."
        ),
    },
    "AvoidThrowingRawExceptionTypes": {
        "type_code": "d",
        "type_label": "something else (exception-handling convention)",
        "overlap": "SignatureDeclareThrowsException; AvoidThrowingNullPointerException; AvoidUncheckedExceptionsInSignatures",
        "independent": "no",
        "construct_group": "exception_handling_convention",
        "wmc_risk": "low",
        "reason": "Raw throw-type convention; not an independent design-smell construct.",
    },
    "NPathComplexity": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "CyclomaticComplexity family; CognitiveComplexity",
        "independent": "no",
        "construct_group": "cyclomatic_complexity_family",
        "wmc_risk": "high",
        "reason": (
            "Counts acyclic paths rather than decision points. Same underlying"
            " method-complexity construct as cyclomatic complexity, different"
            " formula. Not independent; not usable as a WMC predictor."
        ),
    },
    "ExcessiveImports": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "CouplingBetweenObjects; LawOfDemeter; quality CBO",
        "independent": "no",
        "construct_group": "coupling_metric_threshold",
        "wmc_risk": "low",
        "reason": (
            "Import-count threshold used as a coupling proxy. Overlaps CBO and"
            " CouplingBetweenObjects. Prefer LawOfDemeter if a coupling smell is"
            " needed; do not include both import-count and CBO-like rules."
        ),
    },
    "ExcessiveMethodLength": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "NcssCount; NcssMethodCount; CognitiveComplexity; AvoidDeeplyNestedIfStmts",
        "independent": "yes",
        "construct_group": "long_method",
        "wmc_risk": "medium",
        "reason": (
            "Catalog label is Long Method, a standard code smell. Distinct from"
            " class-level GodClass. Correlated with WMC but not the same variable."
            " Use this one representative; do not also include NcssCount/"
            "NcssMethodCount as separate smells."
        ),
    },
    "NcssCount": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "ExcessiveMethodLength; ExcessiveClassLength; NcssMethodCount; NcssTypeCount; NcssConstructorCount; quality LOC",
        "independent": "no",
        "construct_group": "ncss_size_family",
        "wmc_risk": "medium",
        "reason": (
            "NCSS statement-count threshold at class/method/constructor grain."
            " Size metric, not a distinct smell. Overlaps Long Method and Large"
            " Class."
        ),
    },
    "MutableStaticState": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "ImmutableField; FinalFieldCouldBeStatic",
        "independent": "yes",
        "construct_group": "mutable_global_state",
        "wmc_risk": "low",
        "reason": (
            "Non-private non-final static fields: global mutable state / broken"
            " encapsulation. Distinct from size, complexity, and LoD. Not a"
            " rename of ImmutableField (which is a 'could be final' suggestion)."
        ),
    },
    "GodClass": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "TooManyMethods; ExcessiveClassLength; TooManyFields; ExcessivePublicCount; quality WMC, ATFD, LTCC; synthesized mdi_godclass",
        "independent": "conditional",
        "construct_group": "god_class",
        "wmc_risk": "high",
        "reason": (
            "Classic design smell, metric strategy from Lanza/Marinescu-style"
            " detection (reported against the class). Conditional for a WMC"
            " outcome: PMD GodClass is itself a function of complexity/size/"
            "cohesion/ATFD, so it is conceptually contaminated as a predictor of"
            " WMC. Keep as a construct only with that caveat, or hold it out of"
            " WMC models and use DataClass/Long Method/LoD instead."
        ),
    },
    "SimplifyBooleanExpressions": {
        "type_code": "d",
        "type_label": "something else (micro-simplification)",
        "overlap": "SimplifyBooleanReturns; SimplifyConditional; SimplifiedTernary; LogicInversion",
        "independent": "no",
        "construct_group": "boolean_micro_simplification",
        "wmc_risk": "low",
        "reason": "Local boolean cleanup. Not a design smell. Do not split this family into five constructs.",
    },
    "UseUtilityClass": {
        "type_code": "c",
        "type_label": "structural/property rule",
        "overlap": "ClassWithOnlyPrivateConstructorsShouldBeFinal; AbstractClassWithoutAnyMethod",
        "independent": "no",
        "construct_group": "utility_class_shape",
        "wmc_risk": "low",
        "reason": (
            "Suggestion to add a private constructor when all methods are static."
            " Class-shape property, not a smell of poor abstraction in the GodClass/"
            "DataClass sense."
        ),
    },
    "CollapsibleIfStatements": {
        "type_code": "d",
        "type_label": "something else (micro-simplification)",
        "overlap": "AvoidDeeplyNestedIfStmts; SimplifyConditional; CognitiveComplexity",
        "independent": "no",
        "construct_group": "boolean_micro_simplification",
        "wmc_risk": "low",
        "reason": "Consecutive-if merge suggestion. Micro-style, not an independent smell.",
    },
    "DataClass": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "TooManyFields (weak); quality LCOM/LTCC (weak)",
        "independent": "yes",
        "construct_group": "data_class",
        "wmc_risk": "low",
        "reason": (
            "Classic design smell (data holder with little behavior). Distinct"
            " from GodClass and from complexity thresholds. Particularly useful"
            " beside a WMC outcome because DataClass is not a WMC threshold."
        ),
    },
    "AvoidUncheckedExceptionsInSignatures": {
        "type_code": "c",
        "type_label": "structural/property rule",
        "overlap": "SignatureDeclareThrowsException; AvoidThrowingRawExceptionTypes",
        "independent": "no",
        "construct_group": "exception_api_convention",
        "wmc_risk": "low",
        "reason": "Exception-signature convention; overlapping operationalization, not a distinct smell.",
    },
    "SingularField": {
        "type_code": "c",
        "type_label": "structural/property rule",
        "overlap": "ImmutableField; TooManyFields",
        "independent": "no",
        "construct_group": "field_scope_property",
        "wmc_risk": "low",
        "reason": "Field used in a single method could be local. Property/refactor hint, not a named design smell.",
    },
    "AvoidDeeplyNestedIfStmts": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "CognitiveComplexity; CollapsibleIfStatements; CyclomaticComplexity family",
        "independent": "no",
        "construct_group": "cognitive_nesting_complexity",
        "wmc_risk": "medium",
        "reason": (
            "Nesting-depth threshold. Close to CognitiveComplexity. Do not add as"
            " a seventh complexity smell. If nesting is studied, pick one of"
            " CognitiveComplexity or AvoidDeeplyNestedIfStmts, and not with WMC"
            " as the outcome."
        ),
    },
    "SimplifyBooleanAssertion": {
        "type_code": "d",
        "type_label": "something else (test-assertion style)",
        "overlap": "SimplifyBooleanExpressions; SimplifyBooleanReturns",
        "independent": "no",
        "construct_group": "boolean_micro_simplification",
        "wmc_risk": "low",
        "reason": "Test-assertion simplification. Not a production design smell; may concentrate in test files.",
    },
    "ExcessiveClassLength": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "GodClass; TooManyMethods; NcssTypeCount; NcssCount; quality LOC",
        "independent": "conditional",
        "construct_group": "large_class_responsibility",
        "wmc_risk": "medium",
        "reason": (
            "Catalog label is Large Class. Same responsibility/size family as"
            " GodClass and TooManyMethods. Use at most one representative of this"
            " family."
        ),
    },
    "ExcessivePublicCount": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "TooManyMethods; TooManyFields; GodClass",
        "independent": "no",
        "construct_group": "large_class_responsibility",
        "wmc_risk": "medium",
        "reason": (
            "Catalog maps this to 'Class Data Should be Private', which is a"
            " different Fowler smell (public fields). PMD ExcessivePublicCount is"
            " a threshold on the number of public members. Do not treat the"
            " catalog alias as Fowler CDSBP, and do not treat it as independent"
            " of TooManyMethods/GodClass."
        ),
    },
    "TooManyFields": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "GodClass; ExcessiveClassLength; DataClass; quality NOF",
        "independent": "conditional",
        "construct_group": "large_class_responsibility",
        "wmc_risk": "medium",
        "reason": (
            "Field-count size smell. Overlaps GodClass/Large Class. Could be a"
            " DataClass cousin but is not the same construct. Do not include"
            " beside GodClass and ExcessiveClassLength as a third independent"
            " smell."
        ),
    },
    "FinalFieldCouldBeStatic": {
        "type_code": "d",
        "type_label": "something else (immutability/static suggestion)",
        "overlap": "ImmutableField; MutableStaticState",
        "independent": "no",
        "construct_group": "immutability_suggestion",
        "wmc_risk": "low",
        "reason": "Suggestion that a final instance field could be static. Not a smell.",
    },
    "NcssMethodCount": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "NcssCount; ExcessiveMethodLength",
        "independent": "no",
        "construct_group": "ncss_size_family",
        "wmc_risk": "medium",
        "reason": "Method NCSS threshold. Duplicate grain of NcssCount / Long Method.",
    },
    "ExcessiveParameterList": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "UseObjectForClearerAPI",
        "independent": "yes",
        "construct_group": "long_parameter_list",
        "wmc_risk": "low",
        "reason": (
            "Catalog label is Long Parameter List, a standard smell. Distinct from"
            " GodClass, DataClass, LoD, and Long Method. UseObjectForClearerAPI is"
            " a related suggestion, not a second construct."
        ),
    },
    "SimplifyBooleanReturns": {
        "type_code": "d",
        "type_label": "something else (micro-simplification)",
        "overlap": "SimplifyBooleanExpressions; SimplifyConditional; LogicInversion",
        "independent": "no",
        "construct_group": "boolean_micro_simplification",
        "wmc_risk": "low",
        "reason": "Micro boolean-return cleanup. Same family as SimplifyBooleanExpressions.",
    },
    "UselessOverridingMethod": {
        "type_code": "d",
        "type_label": "something else (useless/redundant code)",
        "overlap": "none obvious among the 47 beyond general redundancy",
        "independent": "no",
        "construct_group": "redundant_code",
        "wmc_risk": "low",
        "reason": "Redundant override. Not a standard design-smell construct for this pilot.",
    },
    "CouplingBetweenObjects": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "ExcessiveImports; LawOfDemeter; quality CBO",
        "independent": "no",
        "construct_group": "coupling_metric_threshold",
        "wmc_risk": "low",
        "reason": (
            "CBO threshold. The quality file already contains CBO. Using this"
            " rule plus CBO (as control or outcome-adjacent metric) is circular."
            " Prefer the continuous CBO metric or LawOfDemeter, not both this"
            " rule and CBO."
        ),
    },
    "ClassWithOnlyPrivateConstructorsShouldBeFinal": {
        "type_code": "c",
        "type_label": "structural/property rule",
        "overlap": "UseUtilityClass",
        "independent": "no",
        "construct_group": "utility_class_shape",
        "wmc_risk": "low",
        "reason": "Final-on-class modifier property. Not a design smell.",
    },
    "UseObjectForClearerAPI": {
        "type_code": "d",
        "type_label": "something else (API-shape suggestion)",
        "overlap": "ExcessiveParameterList",
        "independent": "no",
        "construct_group": "long_parameter_list",
        "wmc_risk": "low",
        "reason": "Suggestion to bundle many parameters into an object. Same construct as Long Parameter List.",
    },
    "AvoidThrowingNullPointerException": {
        "type_code": "d",
        "type_label": "something else (exception-handling convention)",
        "overlap": "AvoidThrowingRawExceptionTypes; AvoidCatchingGenericException",
        "independent": "no",
        "construct_group": "exception_handling_convention",
        "wmc_risk": "low",
        "reason": "Specific throw-type convention. Not an independent smell construct.",
    },
    "AvoidRethrowingException": {
        "type_code": "d",
        "type_label": "something else (exception-handling convention)",
        "overlap": "AvoidThrowingNewInstanceOfSameException; ExceptionAsFlowControl",
        "independent": "no",
        "construct_group": "exception_handling_convention",
        "wmc_risk": "low",
        "reason": "Rethrow convention. Overlaps AvoidThrowingNewInstanceOfSameException.",
    },
    "SimplifyConditional": {
        "type_code": "d",
        "type_label": "something else (micro-simplification)",
        "overlap": "SimplifyBooleanExpressions; SimplifiedTernary; LogicInversion",
        "independent": "no",
        "construct_group": "boolean_micro_simplification",
        "wmc_risk": "low",
        "reason": "Micro conditional cleanup. Same boolean-simplification family.",
    },
    "AbstractClassWithoutAnyMethod": {
        "type_code": "c",
        "type_label": "structural/property rule",
        "overlap": "UseUtilityClass",
        "independent": "no",
        "construct_group": "utility_class_shape",
        "wmc_risk": "low",
        "reason": "Abstract class with no methods. Structural property, rare, not a pilot smell.",
    },
    "ExceptionAsFlowControl": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "AvoidRethrowingException; AvoidCatchingGenericException",
        "independent": "conditional",
        "construct_group": "exception_as_flow_control",
        "wmc_risk": "low",
        "reason": (
            "Using exceptions for ordinary control flow is a recognized design"
            " issue and is distinct from signature conventions. Conditional on"
            " having enough positive classes in Signal; catalog-wide count is low."
        ),
    },
    "SimplifiedTernary": {
        "type_code": "d",
        "type_label": "something else (micro-simplification)",
        "overlap": "SimplifyConditional; SimplifyBooleanExpressions",
        "independent": "no",
        "construct_group": "boolean_micro_simplification",
        "wmc_risk": "low",
        "reason": "Ternary micro-cleanup. Not a design smell.",
    },
    "LogicInversion": {
        "type_code": "d",
        "type_label": "something else (micro-simplification)",
        "overlap": "SimplifyBooleanExpressions; SimplifyBooleanReturns; SimplifyConditional",
        "independent": "no",
        "construct_group": "boolean_micro_simplification",
        "wmc_risk": "low",
        "reason": "Negation cleanup. Same micro-simplification family.",
    },
    "SwitchDensity": {
        "type_code": "a",
        "type_label": "design/code smell",
        "overlap": "ModifiedCyclomaticComplexity; CyclomaticComplexity family",
        "independent": "conditional",
        "construct_group": "switch_statement",
        "wmc_risk": "medium",
        "reason": (
            "Catalog label is Switch Statement. Distinct-ish from GodClass, but"
            " switch density is also a complexity operationalization and catalog"
            " count is very low. Include only if Signal has usable positives;"
            " otherwise drop."
        ),
    },
    "NcssTypeCount": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "ExcessiveClassLength; NcssCount; GodClass",
        "independent": "no",
        "construct_group": "ncss_size_family",
        "wmc_risk": "medium",
        "reason": "Class-level NCSS threshold. Duplicate of Large Class / NcssCount.",
    },
    "DoNotExtendJavaLangError": {
        "type_code": "c",
        "type_label": "structural/property rule",
        "overlap": "AvoidThrowingRawExceptionTypes",
        "independent": "no",
        "construct_group": "exception_api_convention",
        "wmc_risk": "low",
        "reason": "Inheritance restriction on java.lang.Error. Rare structural rule, not a smell construct.",
    },
    "AvoidThrowingNewInstanceOfSameException": {
        "type_code": "d",
        "type_label": "something else (exception-handling convention)",
        "overlap": "AvoidRethrowingException",
        "independent": "no",
        "construct_group": "exception_handling_convention",
        "wmc_risk": "low",
        "reason": "Low-count rethrow wrapping convention. Not independent of AvoidRethrowingException.",
    },
    "NcssConstructorCount": {
        "type_code": "b",
        "type_label": "complexity/metric threshold",
        "overlap": "NcssCount; NcssMethodCount; ExcessiveMethodLength",
        "independent": "no",
        "construct_group": "ncss_size_family",
        "wmc_risk": "medium",
        "reason": "Constructor NCSS threshold. Same NCSS size family; catalog count is tiny.",
    },
}


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.reader(handle)


def read_header_rows(path: Path):
    rows = csv_rows(path)
    header = next(rows, [])
    return header, rows


def pct(numerator, denominator, digits=2):
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, digits)


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
            "n": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
            "q01": None,
            "q05": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q99": None,
        }
    q = quantiles(values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "q01": q[0.01],
        "q05": q[0.05],
        "q10": q[0.10],
        "q25": q[0.25],
        "q50": q[0.50],
        "q75": q[0.75],
        "q90": q[0.90],
        "q95": q[0.95],
        "q99": q[0.99],
    }


def smell_key(package, file_path):
    name = re.split(r"[\\/]", (file_path or "").strip())[-1]
    stem = re.sub(r"\.[^.]+$", "", name) if "." in name else name
    package = (package or "").strip()
    if not stem:
        return ""
    if package and package.lower() != "null":
        return f"{package}.{stem}"
    return stem


def file_extension(file_path):
    name = re.split(r"[\\/]", (file_path or "").strip())[-1]
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def path_kind(file_path):
    text = (file_path or "").replace("\\", "/").lower()
    if "/androidtest/" in text or "/androidtest" in text:
        return "android_test"
    if "/src/test/" in text or "/test/" in text:
        return "test"
    if "/src/main/" in text:
        return "main"
    if "/generated/" in text or "/build/" in text:
        return "generated_or_build"
    return "other"


def quality_row_kind(qualified_name):
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


def load_catalog():
    header, rows = read_header_rows(DATASET / "codesmells.csv")
    header = [col.strip() for col in header]
    records = []
    for row in rows:
        if len(row) < 5:
            continue
        record = {
            "catalog_index": row[0].strip(),
            "rule": row[1].strip(),
            "catalog_code_smell_label": row[2].strip(),
            "catalog_details": row[3].strip(),
            "catalog_count": parse_float(row[4]),
        }
        if record["rule"]:
            records.append(record)
    return header, records


def collect_rule_metadata(catalog_rules):
    """Rule set + example violation text from actual PMD CSV exports.

    Signal Android is scanned fully. Other projects are scanned only until every
    catalog rule has a sample Description, smallest files first, so Hadoop/Lucene
    dumps are touched only if a rare rule is still missing.
    """
    needed = {row["rule"] for row in catalog_rules}
    meta = {rule: {"rule_sets": Counter(), "sample_description": ""} for rule in needed}
    smell_files = list((DATASET / "codesmells" / "csv").rglob("*.csv"))
    signal_files = [p for p in smell_files if SIGNAL_DIR in p.parts]
    other_files = sorted(
        (p for p in smell_files if SIGNAL_DIR not in p.parts),
        key=lambda p: p.stat().st_size,
    )
    # Skip multi-hundred-MB dumps unless a rare rule is still missing after
    # smaller projects. 40 MB still includes JUnit/Dropwizard/Checkstyle-scale files.
    other_files = [p for p in other_files if p.stat().st_size <= 40 * 1024 * 1024]
    remaining = set(needed)

    def harvest(path, stop_when_described=False, count_rule_sets=True):
        header, rows = read_header_rows(path)
        try:
            idx_desc = header.index("Description")
            idx_set = header.index("Rule set")
            idx_rule = header.index("Rule")
        except ValueError:
            return
        for row in rows:
            if stop_when_described and not remaining:
                return
            if len(row) <= max(idx_desc, idx_set, idx_rule):
                continue
            rule = row[idx_rule].strip()
            if rule not in meta:
                continue
            if stop_when_described and rule not in remaining:
                continue
            rule_set = row[idx_set].strip()
            if rule_set and (count_rule_sets or not meta[rule]["rule_sets"]):
                meta[rule]["rule_sets"][rule_set] += 1
            if not meta[rule]["sample_description"]:
                desc = row[idx_desc].strip()
                if desc:
                    meta[rule]["sample_description"] = desc
                    remaining.discard(rule)

    for path in signal_files:
        harvest(path, stop_when_described=False, count_rule_sets=True)
    for path in other_files:
        if not remaining:
            break
        harvest(path, stop_when_described=True, count_rule_sets=False)
    return meta, sorted(remaining)


def load_quality(path: Path):
    header, rows = read_header_rows(path)
    wmc_indexes = [i for i, name in enumerate(header) if name.strip() == "WMC"]
    loc_indexes = [i for i, name in enumerate(header) if name.strip() == "LOC"]
    kind_counts = Counter()
    duplicates = Counter()
    records_by_qn = {}
    duplicate_qn = 0
    total_rows = 0
    wmc_mismatch = 0
    class_kinds = {"top_level", "inner_or_nested", "project_header_or_unqualified"}
    for row in rows:
        total_rows += 1
        qn = row[0].strip() if row else ""
        kind = quality_row_kind(qn)
        kind_counts[kind] += 1
        if kind in class_kinds:
            if qn in records_by_qn:
                duplicate_qn += 1
                duplicates[qn] += 1
            else:
                wmc_primary = parse_float(row[wmc_indexes[0]]) if wmc_indexes else None
                wmc_secondary = parse_float(row[wmc_indexes[1]]) if len(wmc_indexes) > 1 else None
                if (
                    wmc_primary is not None
                    and wmc_secondary is not None
                    and wmc_primary != wmc_secondary
                ):
                    wmc_mismatch += 1
                records_by_qn[qn] = {
                    "qualified_name": qn,
                    "name": row[1].strip() if len(row) > 1 else "",
                    "kind": kind,
                    "wmc": wmc_primary,
                    "wmc_secondary": wmc_secondary,
                    "loc": parse_float(row[loc_indexes[0]]) if loc_indexes else None,
                    "row": row,
                }
    return {
        "header": header,
        "total_rows": total_rows,
        "kind_counts": kind_counts,
        "classes": records_by_qn,
        "duplicate_qualified_name_extras": duplicate_qn,
        "wmc_column_mismatch_classes": wmc_mismatch,
        "wmc_indexes": wmc_indexes,
    }


def load_smells(path: Path):
    header, rows = read_header_rows(path)
    idx = {name: header.index(name) for name in ["Package", "File", "Rule set", "Rule", "Description", "Problem"] if name in header}
    total_rows = 0
    malformed = 0
    per_class = defaultdict(lambda: {"rules": Counter(), "rows": 0, "files": set(), "exts": Counter(), "path_kinds": Counter(), "rule_sets": Counter(), "packages": set()})
    rule_rows = Counter()
    rule_sets = Counter()
    extensions = Counter()
    path_kinds = Counter()
    problem_values = []
    empty_rule = 0
    for row in rows:
        total_rows += 1
        if len(row) < 8:
            malformed += 1
            continue
        package = row[idx["Package"]] if "Package" in idx else row[1]
        file_path = row[idx["File"]] if "File" in idx else row[2]
        rule = (row[idx["Rule"]] if "Rule" in idx else row[7]).strip()
        rule_set = (row[idx["Rule set"]] if "Rule set" in idx else row[6]).strip()
        if "Problem" in idx:
            problem = parse_float(row[idx["Problem"]])
            if problem is not None:
                problem_values.append(problem)
        if not rule:
            empty_rule += 1
            continue
        key = smell_key(package, file_path)
        ext = file_extension(file_path)
        kind = path_kind(file_path)
        extensions[ext] += 1
        path_kinds[kind] += 1
        rule_rows[rule] += 1
        if rule_set:
            rule_sets[rule_set] += 1
        if not key:
            continue
        rec = per_class[key]
        rec["rules"][rule] += 1
        rec["rows"] += 1
        rec["files"].add(file_path)
        rec["exts"][ext] += 1
        rec["path_kinds"][kind] += 1
        rec["rule_sets"][rule_set] += 1
        rec["packages"].add(package.strip())
    return {
        "header": header,
        "total_rows": total_rows,
        "malformed_rows": malformed,
        "empty_rule_rows": empty_rule,
        "per_class": per_class,
        "rule_rows": rule_rows,
        "rule_sets": rule_sets,
        "extensions": extensions,
        "path_kinds": path_kinds,
        "problem_min": min(problem_values) if problem_values else None,
        "problem_max": max(problem_values) if problem_values else None,
        "problem_n_unique": len(set(problem_values)),
        "problem_is_global_1_to_n": (
            bool(problem_values)
            and min(problem_values) == 1
            and max(problem_values) == total_rows
            and len(problem_values) == total_rows
            and len(set(problem_values)) == total_rows
        ),
    }


def load_versions_registry():
    header, rows = read_header_rows(DATASET / "versions.csv")
    records = []
    for row in rows:
        if len(row) < 3 or not row[1].strip():
            continue
        records.append({
            "repository_name": row[1].strip(),
            "version": row[2].strip(),
            "n_classes": parse_float(row[5]) if len(row) > 5 else None,
            "n_packages": parse_float(row[6]) if len(row) > 6 else None,
            "n_problematic_classes": parse_float(row[9]) if len(row) > 9 else None,
            "n_highly_problematic_classes": parse_float(row[10]) if len(row) > 10 else None,
        })
    return records


def inspect_synthesized():
    path = DATASET / "synthesized" / "class_level_quality_code_smell_mdi.csv"
    header, rows = read_header_rows(path)
    filenames = Counter()
    signal_rows = 0
    godclass_pos = 0
    total = 0
    filename_idx = header.index("filename") if "filename" in header else None
    god_idx = header.index("GodClass") if "GodClass" in header else None
    mdi_idx = header.index("mdi_godclass") if "mdi_godclass" in header else None
    qn_idx = header.index("QualifiedName") if "QualifiedName" in header else 1
    signal_kinds = Counter()
    signal_god = Counter()
    for row in rows:
        total += 1
        filename = row[filename_idx].strip() if filename_idx is not None and len(row) > filename_idx else ""
        if filename:
            filenames[filename] += 1
        is_signal = "signal" in filename.lower()
        if is_signal:
            signal_rows += 1
            qn = row[qn_idx].strip() if qn_idx is not None and len(row) > qn_idx else ""
            signal_kinds[quality_row_kind(qn)] += 1
            if god_idx is not None and len(row) > god_idx:
                signal_god[row[god_idx].strip() or "empty"] += 1
        if god_idx is not None and len(row) > god_idx:
            if parse_float(row[god_idx]) not in {None, 0.0}:
                godclass_pos += 1
    return {
        "path": str(path.relative_to(ROOT)),
        "header": header,
        "n_rows": total,
        "n_distinct_filename_values": len(filenames),
        "filename_values": sorted(filenames),
        "signal_rows": signal_rows,
        "signal_row_kinds": dict(signal_kinds),
        "signal_godclass_value_counts": dict(signal_god),
        "godclass_nonzero_rows": godclass_pos,
        "has_godclass": god_idx is not None,
        "has_mdi_godclass": mdi_idx is not None,
        "note": (
            "Class-level quality extract plus a derived GodClass/MDI indicator. "
            "Signal Android is included. This is not a complete PMD smell-by-class universe. "
            "GodClass=0 means not GodClass under the MDI formula, not 'no PMD smells'."
        ),
    }


def inspect_command_and_html():
    command = (DATASET / "codesmells" / "Command.txt").read_text(encoding="utf-8", errors="replace")
    html_files = sorted((DATASET / "codesmells" / "html").rglob("*.html"))
    return {
        "command_excerpt": "pmd.bat ... -R category/java/design.xml -f csv",
        "command_contains_design_xml": "category/java/design.xml" in command,
        "command_contains_signal": "Signal-Android" in command,
        "html_report_files": [str(p.relative_to(DATASET)) for p in html_files],
        "html_is_per_version": all(len(p.relative_to(DATASET / "codesmells" / "html").parts) == 2 for p in html_files),
        "note": (
            "HTML reports are one design.html per project directory, not per"
            " version, and they list 'Problems found' (violations only)."
        ),
    }


def investigate_other_universe_tables():
    issues_dir = DATASET / "issues"
    issue_files = sorted(issues_dir.rglob("*.csv")) if issues_dir.exists() else []
    issue_header = None
    if issue_files:
        issue_header, _ = read_header_rows(issue_files[0])
    return {
        "root_tables": [
            "repositories.csv",
            "versions.csv",
            "codesmells.csv",
            "attribute-details.csv",
            "synthesized/class_level_quality_code_smell_mdi.csv",
        ],
        "issues_files": len(issue_files),
        "issues_header": issue_header,
        "issues_are_github_tickets": bool(issue_header) and "Issue Title" in issue_header,
        "attribute_details_are_quality_metrics": True,
        "complete_smell_class_universe_table_found": False,
    }


def flag_wmc_discontinuity(transition):
    flags = []
    pct_changed = transition.get("pct_wmc_changed")
    mean_delta = transition.get("delta_mean")
    median_delta = transition.get("delta_median")
    mean_t = transition.get("mean_wmc_t")
    mean_t1 = transition.get("mean_wmc_t1")
    if pct_changed is not None and pct_changed >= 40:
        flags.append("high_share_of_classes_changed_wmc_ge_40pct")
    if median_delta is not None and abs(median_delta) >= 1:
        flags.append("median_delta_wmc_not_zero")
    if mean_delta is not None and abs(mean_delta) >= 5:
        flags.append("mean_abs_delta_wmc_ge_5")
    if mean_t and mean_t1 and mean_t > 0:
        ratio = mean_t1 / mean_t
        if ratio >= 2 or ratio <= 0.5:
            flags.append("mean_wmc_level_shifted_by_factor_of_2")
    return flags or ["none"]


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def md_escape(text):
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def fmt_num(value, digits=2):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if abs(value) >= 100 or value == int(value):
            if value == int(value):
                return str(int(value))
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value):
    if value is None:
        return "NA"
    return f"{value:.2f}%"


def main():
    catalog_header, catalog = load_catalog()
    print(f"Catalog rules: {len(catalog)}", flush=True)
    rule_meta, rules_without_desc = collect_rule_metadata(catalog)
    print(
        f"Rule metadata collected; missing sample descriptions: {len(rules_without_desc)}",
        flush=True,
    )
    versions_registry = load_versions_registry()
    synthesized = inspect_synthesized()
    command_info = inspect_command_and_html()
    universe_tables = investigate_other_universe_tables()
    signal_version_meta = [
        row for row in versions_registry if row["repository_name"].lower().startswith("signal")
    ]

    quality_by_version = {}
    smell_by_version = {}
    coverage_rows = []
    rule_signal_rows = Counter()
    rule_signal_class_versions = Counter()
    matched_rule_signal_class_versions = Counter()
    pair_counter = Counter()
    path_kind_unmatched_smell = Counter()
    unmatched_smell_examples = []
    unmatched_quality_examples = []
    all_quality_classes = {}
    persistence = Counter()

    for version in VERSIONS:
        print(f"Loading Signal Android {version}", flush=True)
        q_path = DATASET / "quality_attributes" / SIGNAL_DIR / f"{version}.csv"
        s_path = DATASET / "codesmells" / "csv" / SIGNAL_DIR / f"{version}.csv"
        quality = load_quality(q_path)
        smells = load_smells(s_path)
        quality_by_version[version] = quality
        smell_by_version[version] = smells

        quality_classes = quality["classes"]
        top_level = {k: v for k, v in quality_classes.items() if v["kind"] == "top_level"}
        inner = {k: v for k, v in quality_classes.items() if v["kind"] == "inner_or_nested"}
        smell_classes = set(smells["per_class"])
        matched = sorted(smell_classes & set(quality_classes))
        matched_top = sorted(smell_classes & set(top_level))
        smell_only = sorted(smell_classes - set(quality_classes))
        quality_only = sorted(set(quality_classes) - smell_classes)
        quality_top_only = sorted(set(top_level) - smell_classes)

        for key, rec in smells["per_class"].items():
            for rule, count in rec["rules"].items():
                rule_signal_rows[rule] += count
                rule_signal_class_versions[rule] += 1
                if key in quality_classes:
                    matched_rule_signal_class_versions[rule] += 1
            if key in quality_classes:
                rules = sorted(rec["rules"])
                pair_counter.update(
                    (rules[i], rules[j]) for i in range(len(rules)) for j in range(i + 1, len(rules))
                )

        for key in smell_only[:20]:
            rec = smells["per_class"][key]
            kinds = rec["path_kinds"].most_common(1)
            path_kind_unmatched_smell.update(rec["path_kinds"])
            if len(unmatched_smell_examples) < 12:
                unmatched_smell_examples.append({
                    "version": version,
                    "key": key,
                    "path_kind": kinds[0][0] if kinds else "",
                    "ext": rec["exts"].most_common(1)[0][0] if rec["exts"] else "",
                })
        for key in smell_only:
            path_kind_unmatched_smell.update(smells["per_class"][key]["path_kinds"])

        for key in quality_top_only[:8]:
            unmatched_quality_examples.append({
                "version": version,
                "key": key,
                "kind": quality_classes[key]["kind"],
                "wmc": quality_classes[key]["wmc"],
            })

        numeric_wmc_quality = sum(1 for rec in quality_classes.values() if rec["wmc"] is not None)
        numeric_wmc_top = sum(1 for rec in top_level.values() if rec["wmc"] is not None)
        classes_with_any_violation_in_quality = len(matched)
        # Absence from smell CSV cannot be treated as a known zero unless a
        # complete analyzed-class universe exists. Record both interpretations.
        coverage_rows.append({
            "project": SIGNAL_DIR,
            "version": version,
            "quality_rows": quality["total_rows"],
            "quality_package_rows": quality["kind_counts"].get("package", 0),
            "quality_method_rows": quality["kind_counts"].get("method", 0),
            "quality_field_rows": quality["kind_counts"].get("field", 0),
            "quality_anonymous_rows": quality["kind_counts"].get("anonymous", 0),
            "quality_other_structural_rows": quality["kind_counts"].get("other_structural", 0),
            "quality_project_header_or_unqualified_rows": quality["kind_counts"].get("project_header_or_unqualified", 0),
            "quality_inner_or_nested_classes": len(inner),
            "quality_top_level_classes": len(top_level),
            "unique_class_level_quality_observations": len(quality_classes),
            "quality_duplicate_qn_extra_rows": quality["duplicate_qualified_name_extras"],
            "quality_numeric_wmc_classes": numeric_wmc_quality,
            "quality_numeric_wmc_top_level": numeric_wmc_top,
            "smell_violation_rows": smells["total_rows"],
            "smell_malformed_rows": smells["malformed_rows"],
            "unique_classes_in_smell_file": len(smell_classes),
            "exact_class_name_matches": len(matched),
            "exact_top_level_matches": len(matched_top),
            "smell_classes_not_in_quality": len(smell_only),
            "quality_classes_not_in_smell": len(quality_only),
            "top_level_quality_classes_not_in_smell": len(quality_top_only),
            "pct_smell_classes_matching_quality": pct(len(matched), len(smell_classes)),
            "pct_quality_classes_matching_smell": pct(len(matched), len(quality_classes)),
            "pct_top_level_quality_matching_smell": pct(len(matched_top), len(top_level)),
            "pct_quality_classes_with_known_positive_smell": pct(classes_with_any_violation_in_quality, len(quality_classes)),
            "pct_quality_classes_with_known_smell_status_if_absence_is_unknown": pct(len(matched), len(quality_classes)),
            "pct_quality_classes_with_known_smell_status_if_absence_were_zero": 100.0 if quality_classes else None,
            "absence_equals_zero_justified": "no",
            "smell_file_contains_zero_violation_classes": "no",
            "smell_problem_is_global_1_to_n": smells["problem_is_global_1_to_n"],
            "smell_problem_n_unique": smells["problem_n_unique"],
            "smell_problem_min": smells["problem_min"],
            "smell_problem_max": smells["problem_max"],
            "smell_rule_sets": ";".join(f"{k}:{v}" for k, v in smells["rule_sets"].most_common()),
            "smell_file_extensions": ";".join(f"{k}:{v}" for k, v in smells["extensions"].most_common()),
            "smell_path_kinds": ";".join(f"{k}:{v}" for k, v in smells["path_kinds"].most_common()),
            "versions_csv_n_classes": next((r["n_classes"] for r in signal_version_meta if r["version"] == version), None),
            "versions_csv_n_problematic_classes": next((r["n_problematic_classes"] for r in signal_version_meta if r["version"] == version), None),
        })

        for qn in quality_classes:
            all_quality_classes.setdefault(qn, set()).add(version)

    for versions_present in all_quality_classes.values():
        persistence[len(versions_present)] += 1

    # Longitudinal WMC on exact QualifiedName.
    wmc_rows = []
    for earlier, later in zip(VERSIONS, VERSIONS[1:]):
        left = quality_by_version[earlier]["classes"]
        right = quality_by_version[later]["classes"]
        shared = sorted(set(left) & set(right))
        n_wmc_t = n_wmc_t1 = n_valid = n_changed = 0
        deltas = []
        wmc_t_vals = []
        wmc_t1_vals = []
        top_deltas = []
        top_wmc_t = []
        top_wmc_t1 = []
        shared_top = 0
        valid_top = 0
        changed_top = 0
        for qn in shared:
            a = left[qn]
            b = right[qn]
            is_top = a["kind"] == "top_level" and b["kind"] == "top_level"
            if is_top:
                shared_top += 1
            if a["wmc"] is not None:
                n_wmc_t += 1
            if b["wmc"] is not None:
                n_wmc_t1 += 1
            if a["wmc"] is not None and b["wmc"] is not None:
                n_valid += 1
                delta = b["wmc"] - a["wmc"]
                deltas.append(delta)
                wmc_t_vals.append(a["wmc"])
                wmc_t1_vals.append(b["wmc"])
                if delta != 0:
                    n_changed += 1
                if is_top:
                    valid_top += 1
                    top_deltas.append(delta)
                    top_wmc_t.append(a["wmc"])
                    top_wmc_t1.append(b["wmc"])
                    if delta != 0:
                        changed_top += 1
        delta_summary = summarize_numeric(deltas)
        row = {
            "project": SIGNAL_DIR,
            "version_t": earlier,
            "version_t1": later,
            "shared_exact_qualified_names": len(shared),
            "shared_top_level": shared_top,
            "n_numeric_wmc_t": n_wmc_t,
            "n_numeric_wmc_t1": n_wmc_t1,
            "n_valid_wmc_transition": n_valid,
            "n_wmc_changed": n_changed,
            "n_wmc_unchanged": n_valid - n_changed,
            "pct_wmc_changed": pct(n_changed, n_valid),
            "pct_delta_wmc_eq_0": pct(n_valid - n_changed, n_valid),
            "n_valid_top_level_transition": valid_top,
            "n_top_level_wmc_changed": changed_top,
            "pct_top_level_wmc_changed": pct(changed_top, valid_top),
            "mean_wmc_t": statistics.fmean(wmc_t_vals) if wmc_t_vals else None,
            "mean_wmc_t1": statistics.fmean(wmc_t1_vals) if wmc_t1_vals else None,
            "mean_wmc_t_top_level": statistics.fmean(top_wmc_t) if top_wmc_t else None,
            "mean_wmc_t1_top_level": statistics.fmean(top_wmc_t1) if top_wmc_t1 else None,
            "delta_mean": delta_summary["mean"],
            "delta_median": delta_summary["median"],
            "delta_stdev": delta_summary["stdev"],
            "delta_min": delta_summary["min"],
            "delta_max": delta_summary["max"],
            "delta_q01": delta_summary["q01"],
            "delta_q05": delta_summary["q05"],
            "delta_q10": delta_summary["q10"],
            "delta_q25": delta_summary["q25"],
            "delta_q50": delta_summary["q50"],
            "delta_q75": delta_summary["q75"],
            "delta_q90": delta_summary["q90"],
            "delta_q95": delta_summary["q95"],
            "delta_q99": delta_summary["q99"],
        }
        row["discontinuity_flags"] = ";".join(flag_wmc_discontinuity(row))
        wmc_rows.append(row)

    # Classification table joined to Signal frequencies.
    classification_rows = []
    for item in catalog:
        rule = item["rule"]
        spec = RULE_CLASSIFICATION.get(rule)
        if spec is None:
            spec = {
                "type_code": "d",
                "type_label": "something else (unlisted in script map)",
                "overlap": "unknown",
                "independent": "no",
                "construct_group": "unclassified",
                "wmc_risk": "unknown",
                "reason": "Rule present in catalog but missing from the script classification map.",
            }
        meta = rule_meta.get(rule, {})
        rule_sets = meta.get("rule_sets", Counter())
        description = item["catalog_details"] or meta.get("sample_description") or ""
        description_source = (
            "codesmells.csv Details" if item["catalog_details"]
            else ("PMD violation Description field" if meta.get("sample_description") else "unavailable in dataset")
        )
        classification_rows.append({
            "rule": rule,
            "description": description,
            "description_source": description_source,
            "rule_set": "; ".join(f"{name} (n={count})" for name, count in rule_sets.most_common()) or "not observed in scanned PMD CSVs",
            "catalog_index": item["catalog_index"],
            "catalog_code_smell_label": item["catalog_code_smell_label"],
            "catalog_event_count": int(item["catalog_count"]) if item["catalog_count"] is not None else "",
            "representation_type_code": spec["type_code"],
            "representation_type": spec["type_label"],
            "potential_conceptual_overlap": spec["overlap"],
            "independent_smell_construct_for_pilot": spec["independent"],
            "construct_group": spec["construct_group"],
            "wmc_outcome_contamination_risk": spec["wmc_risk"],
            "reason": spec["reason"],
            "sample_violation_description": meta.get("sample_description", ""),
            "signal_violation_rows": rule_signal_rows.get(rule, 0),
            "signal_class_version_units_with_rule": rule_signal_class_versions.get(rule, 0),
            "signal_matched_class_version_units_with_rule": matched_rule_signal_class_versions.get(rule, 0),
        })

    schema_rows = [
        {"field": "project", "status": "derivable", "source": "directory name (10 Signal-Android-master)", "notes": "Constant for this pilot."},
        {"field": "version", "status": "observed", "source": "quality_attributes and codesmells csv filenames (YYYY-1)", "notes": "Six annual snapshots 2016-1..2021-1."},
        {"field": "class", "status": "observed", "source": "quality QualifiedName", "notes": "Use exact QualifiedName. Inner/nested types are separate rows and generally do not join to PMD file-level keys."},
        {"field": "class_kind", "status": "derivable", "source": "QualifiedName shape", "notes": "top_level vs inner_or_nested vs project_header_or_unqualified. Anonymous/package rows should be excluded."},
        {"field": "WMC_t", "status": "observed", "source": "first WMC column in quality CSV at version t", "notes": "Class-level Weighted Method Count / McCabe. Duplicate later WMC column exists; audit checks equality."},
        {"field": "WMC_t1", "status": "observed", "source": "first WMC column in quality CSV at version t+1", "notes": "Only for classes with exact QualifiedName in both versions."},
        {"field": "delta_WMC", "status": "derivable", "source": "WMC_t1 - WMC_t", "notes": "Defined only for valid numeric transitions. Do not call this software corrosion."},
        {"field": "LOC_t", "status": "observed", "source": "quality LOC", "notes": "Candidate control; collinear with size smells."},
        {"field": "CBO_t", "status": "observed", "source": "quality CBO", "notes": "Candidate control; collinear with CouplingBetweenObjects / ExcessiveImports."},
        {"field": "NOM_t", "status": "observed", "source": "quality NOM", "notes": "Candidate control; collinear with TooManyMethods."},
        {"field": "smell_<rule>_count", "status": "derivable", "source": "codesmells CSV aggregated by join key", "notes": "Observed only for classes that appear in the smell file. File is violation-level, not class-level."},
        {"field": "smell_<construct>_present", "status": "uncertain", "source": "aggregated PMD rule(s) for a construct group", "notes": "Positive presence is observed for matched classes. Absence in smell CSV is NOT a reliable zero unless a complete analyzed-class universe is assumed and documented."},
        {"field": "smell_status_known", "status": "derivable", "source": "class key present in smell CSV", "notes": "1 if the class appears in the PMD export (has at least one design.xml violation), else 0/unknown."},
        {"field": "join_key", "status": "derivable", "source": "Package + '.' + basename(File without extension)", "notes": "Validated independently. Matches top-level types whose filename stem equals the class name."},
        {"field": "zero_smell_class_flag", "status": "unavailable/uncertain", "source": "none found", "notes": "No CSIQ table lists analyzed classes with zero PMD violations. Do not fabricate zeros."},
        {"field": "interaction_term", "status": "unavailable", "source": "not in dataset", "notes": "Co-occurrence can be derived; an interaction effect cannot. Interaction is a modeling choice, not a field."},
        {"field": "software_corrosion", "status": "unavailable", "source": "not operationalized", "notes": "Do not add unless a measurable definition is specified."},
        {"field": "mdi_godclass / synthesized GodClass", "status": "observed for Signal, but a different construct", "source": "synthesized/class_level_quality_code_smell_mdi.csv", "notes": "Signal is present. GodClass=0/1 is an MDI metric formula, not a PMD class inventory. It can identify non-GodClass classes for that formula only. It cannot identify zero-PMD-smell classes."},
        {"field": "issues_*", "status": "observed but not class-joined", "source": "issues/10. signal-android", "notes": "GitHub issue exports. No class-level smell universe and no QualifiedName."},
    ]

    write_csv(OUT_RULES, [
        "rule",
        "description",
        "description_source",
        "rule_set",
        "catalog_index",
        "catalog_code_smell_label",
        "catalog_event_count",
        "representation_type_code",
        "representation_type",
        "potential_conceptual_overlap",
        "independent_smell_construct_for_pilot",
        "construct_group",
        "wmc_outcome_contamination_risk",
        "reason",
        "sample_violation_description",
        "signal_violation_rows",
        "signal_class_version_units_with_rule",
        "signal_matched_class_version_units_with_rule",
    ], classification_rows)

    write_csv(OUT_COVERAGE, list(coverage_rows[0].keys()), coverage_rows)
    write_csv(OUT_WMC, list(wmc_rows[0].keys()), wmc_rows)
    write_csv(OUT_SCHEMA, ["field", "status", "source", "notes"], schema_rows)

    evidence = {
        "signal_directory": SIGNAL_DIR,
        "n_catalog_rules": len(catalog),
        "coverage": coverage_rows,
        "wmc_transitions": wmc_rows,
        "class_version_persistence": dict(sorted(persistence.items())),
        "synthesized": {k: v for k, v in synthesized.items() if k != "header"} | {"header_n": len(synthesized["header"])},
        "command_info": command_info,
        "universe_tables": universe_tables,
        "unmatched_smell_examples": unmatched_smell_examples,
        "unmatched_quality_top_level_examples": unmatched_quality_examples[:12],
        "path_kind_unmatched_smell_class_rows": dict(path_kind_unmatched_smell),
        "top_cooccurring_rule_pairs_matched_classes": [
            {"rule_a": a, "rule_b": b, "matched_class_versions": n}
            for (a, b), n in pair_counter.most_common(15)
        ],
        "decision_log": [
            "Quality universe for class-level counts includes top-level, inner/nested, and rare unqualified names; excludes <Package>, <Method>, <Field>, <Anonymous>, other <...> rows.",
            "Smell files are PMD CSV violation dumps; every class in a smell file has at least one violation by construction.",
            "Join key = Package + '.' + basename(File) without extension, exact-matched to quality QualifiedName.",
            "Absence from smell CSV is not encoded as zero.",
            "No regression models were estimated.",
        ],
        "rules_without_sample_description": rules_without_desc,
    }
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2, default=str)

    write_audit(
        catalog=catalog,
        classification_rows=classification_rows,
        coverage_rows=coverage_rows,
        wmc_rows=wmc_rows,
        schema_rows=schema_rows,
        synthesized=synthesized,
        command_info=command_info,
        universe_tables=universe_tables,
        persistence=persistence,
        pair_counter=pair_counter,
        unmatched_smell_examples=unmatched_smell_examples,
        signal_version_meta=signal_version_meta,
        quality_by_version=quality_by_version,
    )
    print(f"Wrote {OUT_RULES}")
    print(f"Wrote {OUT_COVERAGE}")
    print(f"Wrote {OUT_WMC}")
    print(f"Wrote {OUT_SCHEMA}")
    print(f"Wrote {OUT_AUDIT}")
    print(f"Wrote {OUT_JSON}")


def write_audit(
    catalog,
    classification_rows,
    coverage_rows,
    wmc_rows,
    schema_rows,
    synthesized,
    command_info,
    universe_tables,
    persistence,
    pair_counter,
    unmatched_smell_examples,
    signal_version_meta,
    quality_by_version,
):
    by_group = defaultdict(list)
    for row in classification_rows:
        by_group[row["construct_group"]].append(row)

    n_rules = len(classification_rows)
    n_yes = sum(1 for r in classification_rows if r["independent_smell_construct_for_pilot"] == "yes")
    n_cond = sum(1 for r in classification_rows if r["independent_smell_construct_for_pilot"] == "conditional")
    n_no = sum(1 for r in classification_rows if r["independent_smell_construct_for_pilot"] == "no")
    type_counts = Counter(r["representation_type_code"] for r in classification_rows)

    total_quality_rows = sum(r["quality_rows"] for r in coverage_rows)
    total_quality_classes = sum(r["unique_class_level_quality_observations"] for r in coverage_rows)
    total_smell_rows = sum(r["smell_violation_rows"] for r in coverage_rows)
    total_smell_classes = sum(r["unique_classes_in_smell_file"] for r in coverage_rows)
    total_matches = sum(r["exact_class_name_matches"] for r in coverage_rows)

    candidate_yes = [r for r in classification_rows if r["independent_smell_construct_for_pilot"] == "yes"]
    candidate_cond = [r for r in classification_rows if r["independent_smell_construct_for_pilot"] == "conditional"]

    lines = []
    add = lines.append

    add("# CSIQ V5 Signal Android Pilot Audit")
    add("")
    add("Generated by `csiq_pilot_audit.py`. No regression or predictive model was estimated.")
    add("")
    add("Research question under exploration (not tested here):")
    add("")
    add("> Does modeling interactions/configurations among software smells provide information beyond modeling individual software smells independently?")
    add("")
    add("This audit only prepares data decisions. A later negative result on incremental value remains fully possible and is not treated as a failure of the dataset.")
    add("")
    add("## A. Dataset structure")
    add("")
    add(f"- Dataset root: `Dataset`")
    add("- Top-level tables: `repositories.csv`, `versions.csv`, `codesmells.csv`, `attribute-details.csv`, `synthesized/class_level_quality_code_smell_mdi.csv`.")
    add("- Per-project, per-version folders: `quality_attributes/<project>/<YYYY-1>.csv` and `codesmells/csv/<project>/<YYYY-1>.csv`.")
    add("- Smell HTML reports exist under `codesmells/html/<project>/design.html` (one file per project, not per version).")
    add("- GitHub issue dumps exist under `issues/` and are ticket-level, not class-level.")
    add("- `codesmells/Command.txt` shows PMD invoked as `pmd.bat -d <repo> -R category/java/design.xml -f csv`.")
    add(f"- Command file points at Signal Android and uses design.xml: `{command_info['command_contains_signal']}`, `{command_info['command_contains_design_xml']}`.")
    add(f"- `codesmells.csv` contains **{n_rules} rules** (catalog event totals, not unique classes).")
    add("- Quality files mix several row kinds in one CSV: a project/commit header, `<Package>` rows, `<Method>` rows, `<Field>` rows, `<Anonymous>` rows, inner/nested types, and top-level classes. Method and field rows are not class-level observations.")
    add("- Smell files are PMD violation-level records with fields `Problem, Package, File, Priority, Line, Description, Rule set, Rule`.")
    add("- `attribute-details.csv` documents quality metrics (WMC, CBO, LCOM, …), not PMD smells.")
    add("- `versions.csv` reports `Number of problematic classes` / `highly problematic classes`. For Signal Android those counts are 1–3 per year. That is a metric-based 'problematic class' flag (high coupling/complexity/low cohesion), **not** a PMD smell universe.")
    add("")
    add("### Filtering / interpretation decisions")
    add("")
    add("1. Class-level quality observations = unique `QualifiedName` values that are not `<Package>`, `<Method>`, `<Field>`, `<Anonymous>`, or other `<...>` structural rows. Project header rows without a dot are counted but are not usable WMC units.")
    add("2. Smell class key = `Package + '.' + basename(File without extension)`. Validated from the raw files, not taken on faith from prior notes.")
    add("3. Exact match = smell key equals quality `QualifiedName`. No fuzzy matching, no inner-class projection onto the enclosing type, no package-only match.")
    add("4. Inner/nested quality types are retained in counts but are reported separately because they almost never join to file-stem smell keys.")
    add("5. Absence from a smell CSV is **not** recoded as 'no smell'.")
    add("6. All 47 catalog rules are retained in the classification table. None were dropped for being inconveniently rare, noisy, or collinear; exclusion from the *pilot construct list* is a separate, documented decision.")
    add("")
    add("## B. Complete list / classification of the 47 rules")
    add("")
    add(f"Representation-type counts: (a) design/code smell = {type_counts.get('a', 0)}; (b) complexity/metric threshold = {type_counts.get('b', 0)}; (c) structural/property = {type_counts.get('c', 0)}; (d) something else = {type_counts.get('d', 0)}.")
    add("")
    add(f"Independent-construct decisions: yes = {n_yes}; conditional = {n_cond}; no = {n_no}.")
    add("")
    add("These are **not** 47 software smells. They are PMD `design.xml` rules. CSIQ `codesmells.csv` maps a few of them onto Fowler-style names; at least one mapping is misleading (`ExcessivePublicCount` → 'Class Data Should be Private').")
    add("")
    add("| Rule | Type | Catalog smell label | Independent for pilot | Signal matched class-versions | Construct group |")
    add("| --- | --- | --- | --- | ---: | --- |")
    for row in classification_rows:
        add(
            f"| `{md_escape(row['rule'])}` | {row['representation_type_code']} {md_escape(row['representation_type'])} | "
            f"{md_escape(row['catalog_code_smell_label']) or '—'} | {row['independent_smell_construct_for_pilot']} | "
            f"{row['signal_matched_class_version_units_with_rule']} | `{md_escape(row['construct_group'])}` |"
        )
    add("")
    add("Full descriptions, overlap notes, and reasons are in `csiq_smell_rule_classification.csv`.")
    add("")
    add("Rules whose catalog `Details` field was empty had descriptions filled from the PMD `Description` column when that text exists in the smell CSVs. That fill is marked in `description_source`.")
    add("")
    add("## C. Conceptual overlap among rules")
    add("")
    add("Do not treat every PMD rule as a distinct smell construct. The overlapping families are:")
    add("")
    add("| Construct group | Rules | How to treat |")
    add("| --- | --- | --- |")
    group_order = [
        "cyclomatic_complexity_family",
        "cognitive_nesting_complexity",
        "god_class",
        "large_class_responsibility",
        "long_method",
        "ncss_size_family",
        "long_parameter_list",
        "data_class",
        "coupling_lod",
        "coupling_metric_threshold",
        "mutable_global_state",
        "immutability_suggestion",
        "exception_as_flow_control",
        "exception_api_convention",
        "exception_handling_convention",
        "boolean_micro_simplification",
        "switch_statement",
        "utility_class_shape",
        "field_scope_property",
        "redundant_code",
    ]
    for group in group_order:
        members = by_group.get(group, [])
        if not members:
            continue
        names = ", ".join(f"`{m['rule']}`" for m in members)
        decision = Counter(m["independent_smell_construct_for_pilot"] for m in members)
        add(f"| `{group}` | {names} | {', '.join(f'{k}={v}' for k, v in decision.items())} |")
    add("")
    add("Especially important overlaps:")
    add("")
    add("1. **CyclomaticComplexity / StdCyclomaticComplexity / ModifiedCyclomaticComplexity / NPathComplexity** are alternative formulas for method complexity, not four smells. **CognitiveComplexity** and **AvoidDeeplyNestedIfStmts** are nesting-aware relatives of the same family.")
    add("2. That family is the same conceptual variable as **WMC** (Weighted Method Count / McCabe). Using it to explain WMC is circular.")
    add("3. **GodClass**, **TooManyMethods**, **ExcessiveClassLength**, **TooManyFields**, **ExcessivePublicCount**, and the **NCSS** rules are overlapping size/responsibility operationalizations. GodClass is additionally a composite of complexity/cohesion/ATFD-style metrics.")
    add("4. **ExcessiveMethodLength**, **NcssCount**, and **NcssMethodCount** are overlapping Long Method / size thresholds. Keep at most one.")
    add("5. **ExcessiveParameterList** and **UseObjectForClearerAPI** are one Long Parameter List construct.")
    add("6. **LawOfDemeter**, **CouplingBetweenObjects**, and **ExcessiveImports** overlap coupling; CBO is already a quality column.")
    add("7. **ImmutableField** and **FinalFieldCouldBeStatic** are suggestions, not smells. **MutableStaticState** is the actual encapsulation/global-state issue.")
    add("8. Exception rules split into signature conventions, catch/throw conventions, and the one design-like rule **ExceptionAsFlowControl**.")
    add("9. Boolean simplify/logic/ternary/assertion rules are micro-refactor hints.")
    add("")
    add("Observed co-occurrence is **not** an interaction effect. The pairs below are descriptive only (matched Signal class-version units with both rules present):")
    add("")
    add("| Rule A | Rule B | Matched class-versions |")
    add("| --- | --- | ---: |")
    for (a, b), n in pair_counter.most_common(12):
        add(f"| `{a}` | `{b}` | {n} |")
    add("")
    add("## D. Signal Android smell/quality matching methodology")
    add("")
    add("Join (validated from raw files):")
    add("")
    add("```")
    add("smell_key = Package + '.' + basename(File)  # extension stripped")
    add("match if smell_key == quality.QualifiedName")
    add("```")
    add("")
    add("This is a file-stem key. It can match a top-level type named like its `.java` file. It does not identify:")
    add("")
    add("- inner/nested types (`Outer.Inner` in quality vs `pkg.Outer` from `Outer.java`)")
    add("- anonymous classes")
    add("- classes whose filename stem differs from the type name")
    add("- non-Java files (PMD was run with Java `design.xml`)")
    add("")
    add("Two different coverage percentages are reported and must not be mixed:")
    add("")
    add("1. **Smell-file → quality join rate**: `exact matches / unique classes in the smell file`. This answers: of PMD-violating file keys, how many land on a quality class?")
    add("2. **Quality-class smell-status coverage**: `exact matches / unique class-level quality observations`. This answers: for what fraction of quality classes do we observe at least one PMD violation record? It is **not** the fraction with known presence *or* absence.")
    add("")
    add("If absence were (incorrectly) treated as zero, (2) would be 100% by definition. The files do not justify that.")
    add("")
    add("PMD CSV structure evidence that the export is violation-only:")
    add("")
    add("- Every smell row has a `Rule`. There are no clean-class placeholder rows.")
    add("- The `Problem` column restarts (it is not a global 1..N index of analyzed classes). In Signal 2016-1 there are 4357 violation rows but only 142 distinct Problem values.")
    add("- HTML reports are titled 'Problems found'.")
    add("- `Command.txt` uses `-f csv`, which emits violations, not analyzed-but-clean files.")
    add("")
    add("Unmatched smell-key examples (first few):")
    add("")
    for ex in unmatched_smell_examples[:8]:
        add(f"- {ex['version']}: `{ex['key']}` (path_kind={ex['path_kind']}, ext={ex['ext']})")
    add("")
    add("## E. Per-version coverage results")
    add("")
    add("| Version | Quality rows | Method rows | Field rows | Unique class-level quality obs. | Top-level | Inner/nested | Smell violation rows | Unique smell classes | Exact matches | Smell→quality match % | Quality classes with observed violation % | Top-level with observed violation % |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in coverage_rows:
        add(
            f"| {row['version']} | {row['quality_rows']} | {row['quality_method_rows']} | {row['quality_field_rows']} | "
            f"{row['unique_class_level_quality_observations']} | "
            f"{row['quality_top_level_classes']} | {row['quality_inner_or_nested_classes']} | "
            f"{row['smell_violation_rows']} | {row['unique_classes_in_smell_file']} | {row['exact_class_name_matches']} | "
            f"{fmt_pct(row['pct_smell_classes_matching_quality'])} | {fmt_pct(row['pct_quality_classes_matching_smell'])} | "
            f"{fmt_pct(row['pct_top_level_quality_matching_smell'])} |"
        )
    add("")
    add(f"Aggregated across six Signal versions (class-version units, not distinct classes): quality rows = {total_quality_rows}; unique class-level quality observations = {total_quality_classes}; smell violation rows = {total_smell_rows}; unique smell classes = {total_smell_classes}; exact matches = {total_matches}; smell→quality = {fmt_pct(pct(total_matches, total_smell_classes))}; quality with observed violation = {fmt_pct(pct(total_matches, total_quality_classes))}.")
    add("")
    add("`versions.csv` class counts vs quality unique class-level observations:")
    add("")
    add("| Version | versions.csv n classes | Quality unique class-level obs. | versions.csv problematic classes |")
    add("| --- | ---: | ---: | ---: |")
    for row in coverage_rows:
        add(
            f"| {row['version']} | {fmt_num(row['versions_csv_n_classes'])} | "
            f"{row['unique_class_level_quality_observations']} | {fmt_num(row['versions_csv_n_problematic_classes'])} |"
        )
    add("")
    add("The `versions.csv` class count is close to the quality unique-class count but is not identical. Anonymous rows are numerous in quality files and are excluded from class-level observations here. Do not use `Number of problematic classes` as a smell denominator (Signal values are 1–3).")
    add("")
    add("## F. Whether zero-smell classes can be identified reliably")
    add("")
    add("**No. Not from the tables present in this dataset.**")
    add("")
    add("Checked and insufficient:")
    add("")
    add(f"1. Smell CSVs: violation-only. `smell_file_contains_zero_violation_classes = no` for every Signal version.")
    add("2. Smell HTML: problems-found lists, and not even per version.")
    add("3. `versions.csv`: problematic-class counts are metric-based and tiny; they are not PMD coverage.")
    add("4. `attribute-details.csv`: metric dictionary only.")
    add(f"5. Synthesized MDI/GodClass table: {synthesized['n_rows']} rows across {synthesized['n_distinct_filename_values']} filename values. **Signal Android rows = {synthesized['signal_rows']}** (kinds: {synthesized.get('signal_row_kinds')}). `GodClass` value counts for Signal: {synthesized.get('signal_godclass_value_counts')}. This table **can** label non-GodClass classes under the MDI formula (`GodClass=0`). It **cannot** identify classes with zero PMD design.xml violations. PMD `GodClass` and synthesized `GodClass` are different operationalizations (PMD matched class-versions with GodClass in Signal = 257; synthesized Signal `GodClass=1` is much larger).")
    add("6. Issues CSVs: GitHub tickets (`Issue Title`, `State`, `Labels`). No QualifiedName, no PMD rules.")
    add("7. Repositories.csv: project metadata only.")
    add("")
    add("Therefore:")
    add("")
    add("- A class present in the smell CSV has ≥1 design.xml violation (known positive for at least one rule).")
    add("- A quality class absent from the smell CSV may be clean, unscanned, non-Java, filename-mismatched, inner/anonymous, or outside PMD's actual walk. Those alternatives cannot be separated with the files on disk.")
    add("- Encoding `absent = 0` would manufacture most of the 'no smell' class universe and is not licensed by the file structure.")
    add("")
    add("A later sensitivity analysis *may* assume that PMD walked every `.java` file in the repo (Command.txt is consistent with that intent) and that top-level types whose names equal file stems form the analyzed universe. That would still be an assumption, not a proof, and must be labeled as such.")
    add("")
    add("## G. Recommended candidate smell constructs for the pilot")
    add("")
    add("The dataset does **not** support using all 47 rules as independent smells. It also does not require forcing a count of 5–8 if prevalence is too thin. Based on definitions, overlap, WMC contamination, and Signal matched class-version frequencies, the appropriate first-pilot set is **six primary constructs**, plus **two optional/conditional** ones.")
    add("")
    add("### Primary (conceptually distinct; usable beside a WMC outcome if prevalence is adequate)")
    add("")
    add("| Construct | Rule(s) to encode | Signal matched class-versions | Why |")
    add("| --- | --- | ---: | --- |")
    primary = ["DataClass", "ExcessiveMethodLength", "ExcessiveParameterList", "MutableStaticState", "LawOfDemeter", "GodClass"]
    reasons = {
        "DataClass": "Classic design smell, not a WMC threshold.",
        "ExcessiveMethodLength": "Long Method. Keep this one size-of-method indicator; drop NCSS duplicates.",
        "ExcessiveParameterList": "Long Parameter List. Drop UseObjectForClearerAPI as a second variable.",
        "MutableStaticState": "Global mutable state. Do not include ImmutableField.",
        "LawOfDemeter": "Coupling smell. Among matched Signal class-versions it is near-saturated as a binary flag (~86%). Prefer violation counts, or drop the binary indicator.",
        "GodClass": "Classic smell, but high WMC contamination. Include only as a labeled, contaminated construct or hold out of WMC models.",
    }
    lookup = {r["rule"]: r for r in classification_rows}
    for name in primary:
        row = lookup[name]
        add(f"| {row['construct_group']} | `{name}` | {row['signal_matched_class_version_units_with_rule']} | {reasons[name]} |")
    add("")
    lod_matched = lookup["LawOfDemeter"]["signal_matched_class_version_units_with_rule"]
    add(
        f"LawOfDemeter binary saturation among matched Signal class-versions: "
        f"{lod_matched} / {total_matches} = {fmt_pct(pct(lod_matched, total_matches))}. "
        f"A binary LoD indicator is therefore a near-constant among PMD-positive classes. "
        f"If LoD is kept, encode violation counts, not presence."
    )
    add("")
    add("### Optional / do not add as extra independent smells")
    add("")
    add("- **TooManyMethods** or **ExcessiveClassLength**: same large-class family as GodClass. Add at most one, and only if GodClass is excluded.")
    add("- **ExceptionAsFlowControl** (5 matched class-versions) and **SwitchDensity** (19): too sparse for the first pilot.")
    add("- **Cyclomatic / NCSS / Cognitive / NPath family**: do not use with WMC as the outcome.")
    add("- **ImmutableField**: not a smell.")
    add("- Boolean micro-rules and exception-signature conventions: not pilot constructs.")
    add("")
    add("This is 6 constructs, not 47 and not a forced 8. Expanding to 8 by splitting cyclomatic variants or NCSS grains would manufacture independent variables that are the same construct.")
    add("")
    add("Yes-rated independent rules in the classification file:")
    add("")
    for row in candidate_yes:
        add(f"- `{row['rule']}` ({row['construct_group']}), Signal matched class-versions = {row['signal_matched_class_version_units_with_rule']}")
    add("")
    add("Conditional rules:")
    add("")
    for row in candidate_cond:
        add(f"- `{row['rule']}` ({row['construct_group']}), Signal matched class-versions = {row['signal_matched_class_version_units_with_rule']}")
    add("")
    add("## H. Risks / limitations discovered")
    add("")
    add("1. **Violation-only smell export.** Zeroes are not observed.")
    add("2. **Grain mismatch.** PMD events are file/line/rule; quality is class (and inner/anonymous). File-stem join under-identifies inner types.")
    add("3. **LawOfDemeter saturation.** Catalog-wide this one rule dominates event counts. Class-level binary LoD may still be common enough to be uninformative.")
    add("4. **WMC tautology** if complexity-threshold rules or GodClass are used as predictors of WMC.")
    add("5. **Misleading catalog aliases.** ExcessivePublicCount is not Fowler 'Class Data Should be Private'. CyclomaticComplexity is not by itself 'Complex Class' as a distinct smell.")
    add("6. **Test vs main.** Smell paths include androidTest/test/main. Quality classes do not carry a test flag. Mixing them silently is a validity threat.")
    add("7. **Java-only PMD.** design.xml will not score Kotlin. Quality metrics may still include non-Java types. Unmatched keys can reflect language mix rather than cleanliness.")
    add("8. **Synthesized GodClass ≠ PMD universe.** Signal is in the MDI table, but `GodClass=0` is not 'no PMD smells'. PMD GodClass and MDI GodClass disagree in scale.")
    add("9. **Panel attrition.** Class-version persistence is uneven because the project grew (see below). A balanced six-year panel will be much smaller than 2021 headcount.")
    add("10. **Measurement discontinuities.** WMC transitions are inspected below; a high share of ΔWMC = 0 is expected and is not by itself evidence of stability of the *tool*.")
    add("11. **Co-occurrence ≠ interaction.** Pair counts in this audit are not Model B.")
    add("12. **Selected-on-smells bias** if the panel is restricted to classes that appear in smell CSVs.")
    add("")
    add("## Longitudinal WMC feasibility (exact QualifiedName, consecutive versions)")
    add("")
    add("| t → t+1 | Shared classes | Numeric WMC_t | Numeric WMC_t+1 | Valid transitions | Changed | Δ=0 % | Mean Δ | Median Δ | SD Δ | Mean WMC_t (all / top-level) | Flags |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in wmc_rows:
        add(
            f"| {row['version_t']} → {row['version_t1']} | {row['shared_exact_qualified_names']} | "
            f"{row['n_numeric_wmc_t']} | {row['n_numeric_wmc_t1']} | {row['n_valid_wmc_transition']} | "
            f"{row['n_wmc_changed']} | {fmt_pct(row['pct_delta_wmc_eq_0'])} | {fmt_num(row['delta_mean'], 3)} | "
            f"{fmt_num(row['delta_median'], 3)} | {fmt_num(row['delta_stdev'], 3)} | "
            f"{fmt_num(row['mean_wmc_t'], 2)} / {fmt_num(row.get('mean_wmc_t_top_level'), 2)} | {row['discontinuity_flags']} |"
        )
    add("")
    add("Quantiles of ΔWMC are in `csiq_signal_android_wmc_transitions.csv` (q01..q99).")
    add("")
    add("Class-version persistence (how many distinct QualifiedNames appear in k of the 6 Signal quality snapshots):")
    add("")
    add("| k versions | Distinct QualifiedNames |")
    add("| ---: | ---: |")
    for k in sorted(persistence):
        add(f"| {k} | {persistence[k]} |")
    add("")
    add("Matching is exact `QualifiedName`. Rename/move of a class looks like death + birth, not a WMC change.")
    add("")
    add("## I. Proposed pilot schema (only what the files support)")
    add("")
    add("| Field | Status | Source | Notes |")
    add("| --- | --- | --- | --- |")
    for row in schema_rows:
        add(f"| `{md_escape(row['field'])}` | {row['status']} | {md_escape(row['source'])} | {md_escape(row['notes'])} |")
    add("")
    add("Observed vs derivable vs unavailable is the binding constraint. Do not fill unavailable fields with zeros or with literature defaults.")
    add("")
    add("## J. Explicit recommendations for the NEXT step")
    add("")
    add("Next step is **dataset building**, still not modeling:")
    add("")
    add("1. Build a Signal Android class-version panel of top-level types with numeric WMC.")
    add("2. Attach PMD construct counts for the six primary constructs via the validated join key.")
    add("3. Store `smell_status_known` = class appeared in that version's smell CSV. Do **not** recode the complement as zero in the canonical file.")
    add("4. Produce two analysis views later, without discarding the canonical file: (i) known-positive-only; (ii) optional sensitivity file where top-level java-like keys missing from PMD are set to zero **with an assumption flag**.")
    add("5. Compute consecutive-version `WMC_t`, `WMC_t1`, `delta_WMC` on exact QualifiedName.")
    add("6. Do not create interaction terms yet. Co-occurrence tables can be added as diagnostics only.")
    add("7. After the panel exists, check construct prevalence and LoD saturation, then freeze the construct list before any Model A / Model B comparison.")
    add("")
    add("## K. Answers to the six decision questions")
    add("")
    add("1. **Is Signal Android suitable for the first pilot?** Conditionally yes for a *descriptive / panel-construction* pilot: six paired versions, high smell→quality join rate relative to other CSIQ projects, and usable WMC transitions. It is not yet suitable for a causal interaction model, and it is not suitable for treating missing smell rows as zeros.")
    add("2. **Which smell constructs should be tested?** DataClass, Long Method (`ExcessiveMethodLength`), Long Parameter List (`ExcessiveParameterList`), MutableStaticState, LawOfDemeter (if not saturated), and GodClass only with a WMC-contamination label. Not all 47 rules.")
    add("3. **Can we reliably define smell presence/absence?** Presence of a violation: yes, for keys in the smell CSV. Absence: **no**, not reliably. Presence/absence as a complete binary universe is not supported.")
    add("4. **Is WMC a viable first longitudinal outcome?** Yes as a numeric class-level outcome with lagged `WMC_t`, subject to attrition, inner-class noise, and the prohibition on using cyclomatic/GodClass-style predictors without a contamination discussion. ΔWMC = 0 is common; that is an outcome distribution fact, not a model result.")
    add("5. **Biggest methodological threat?** Encoding unknown smell status as 'no smell', compounded by using complexity-threshold PMD rules (or GodClass) to explain WMC, and interpreting co-occurrence as interaction.")
    add("6. **Exact next dataset-building step?** Write a Signal top-level class-version panel with observed WMC and observed PMD construct counts, plus a `smell_status_known` flag, without fabricating zeros and without running Model A/B.")
    add("")
    add("## Methodological distinctions preserved")
    add("")
    add("- Smell co-occurrence is not an interaction effect.")
    add("- Correlation is not causation.")
    add("- Multiple smell variables are not evidence of interaction.")
    add("- ML feature interaction is not necessarily a meaningful smell interaction.")
    add("- Smell prioritization is not configuration prioritization.")
    add("- The thesis is not trying to prove that smell interactions matter; it asks whether configuration information adds incremental value. A negative result is a valid scientific outcome.")
    add("")
    add("## Generated files")
    add("")
    add("- `csiq_pilot_audit.py`")
    add("- `csiq_smell_rule_classification.csv`")
    add("- `csiq_signal_android_coverage.csv`")
    add("- `csiq_signal_android_wmc_transitions.csv`")
    add("- `csiq_pilot_proposed_schema.csv`")
    add("- `csiq_pilot_audit.md`")
    add("- `csiq_pilot_audit_evidence.json`")
    add("")

    OUT_AUDIT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
