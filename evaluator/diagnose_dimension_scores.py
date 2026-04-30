#!/usr/bin/env python3
"""
diagnose_dimension_scores.py

Full diagnostics for GPT dimension-score files used to train the FiLM evaluator.

Usage:
  python evaluator/diagnose_dimension_scores.py --scores data/gpt4_dimension_scores_smooth.json
  python evaluator/diagnose_dimension_scores.py --scores data/gpt4_dimension_scores.json --compare data/gpt4_dimension_scores_smooth.json
"""

import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


WIKI_DIM_NAMES = ["Discussion", "History", "Event", "Person", "Location"]
ECOM_DIM_NAMES = ["Price", "Quality", "Brand", "Features", "Ethical"]


def load_json(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_domain(topic: str, ecommerce_topics: set) -> str:
    return "ecommerce" if topic in ecommerce_topics else "wikipedia"


def load_ecommerce_topics(data_dir: str):
    p = Path(data_dir) / "topics_ecommerce.json"
    if p.exists():
        try:
            return set(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def entropy_normalized(scores):
    arr = np.array(scores, dtype=float)
    if arr.sum() <= 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(arr)))


def validate_rows(d):
    bad = []
    total = 0

    for topic, qs in d.items():
        if not isinstance(qs, dict):
            bad.append((topic, None, "topic_value_not_dict", qs))
            continue

        for q, s in qs.items():
            total += 1
            if not isinstance(s, list) or len(s) != 5:
                bad.append((topic, q, "scores_not_len5", s))
                continue

            for x in s:
                if not isinstance(x, int) or x < 1 or x > 10:
                    bad.append((topic, q, "score_not_int_1_to_10", s))
                    break

    return total, bad


def compute_report(path: str, data_dir: str = "data", n_alpha_samples: int = 10, seed: int = 42):
    d = load_json(path)
    ecommerce_topics = load_ecommerce_topics(data_dir)
    rng = np.random.default_rng(seed)

    total_rows, bad_rows = validate_rows(d)

    vals = []
    entropies = []
    margins = []
    multi_relevant = 0
    total_valid = 0
    dominant = Counter()
    domain_dominant = defaultdict(Counter)
    domain_counts = Counter()
    topic_reports = []
    scalar_targets = []

    per_dim_values = defaultdict(list)

    for topic, qs in d.items():
        if not isinstance(qs, dict):
            continue

        domain = infer_domain(topic, ecommerce_topics)
        topic_vectors = []
        topic_dominant = Counter()

        for q, s in qs.items():
            if not isinstance(s, list) or len(s) != 5:
                continue
            if any((not isinstance(x, int)) or x < 1 or x > 10 for x in s):
                continue

            arr = np.array(s, dtype=float)
            topic_vectors.append(arr)

            vals.extend(s)
            total_valid += 1
            domain_counts[domain] += 1

            entropies.append(entropy_normalized(arr))

            sorted_s = np.sort(arr)[::-1]
            margins.append(float(sorted_s[0] - sorted_s[1]))

            if (arr >= 5).sum() >= 2:
                multi_relevant += 1

            dom = int(np.argmax(arr))
            dominant[dom] += 1
            domain_dominant[domain][dom] += 1
            topic_dominant[dom] += 1

            for i, x in enumerate(arr):
                per_dim_values[i].append(float(x))

            v = arr / 10.0
            for _ in range(n_alpha_samples):
                alpha = rng.dirichlet(np.ones(5))
                scalar_targets.append(float(alpha @ v))

        if topic_vectors:
            mat = np.stack(topic_vectors, axis=0)
            maj_dim, maj_count = topic_dominant.most_common(1)[0]
            topic_reports.append({
                "topic": topic,
                "domain": domain,
                "n_questions": int(mat.shape[0]),
                "majority_dim": int(maj_dim),
                "majority_share": float(maj_count / mat.shape[0]),
                "mean_entropy": float(np.mean([entropy_normalized(row) for row in mat])),
                "mean_std_across_dims": float(mat.std(axis=0).mean()),
                "mean_score": float(mat.mean()),
                "std_score": float(mat.std()),
                "multi_relevant_ratio": float(((mat >= 5).sum(axis=1) >= 2).mean()),
            })

    vals_counter = Counter(vals)
    n_values = len(vals)
    scalar_targets = np.array(scalar_targets, dtype=float) if scalar_targets else np.array([])

    if total_valid > 0:
        max_dom_share = max(dominant.values()) / total_valid
    else:
        max_dom_share = 0.0

    collapsed_topics = [
        r for r in topic_reports
        if r["majority_share"] >= 0.85 or r["mean_std_across_dims"] < 0.75
    ]

    report = {
        "path": str(path),
        "topics": len(d),
        "total_rows": total_rows,
        "valid_rows": total_valid,
        "bad_rows_count": len(bad_rows),
        "bad_rows_sample": bad_rows[:5],

        "score_counts": {str(k): vals_counter[k] for k in range(1, 11)},
        "extreme_1_or_10_ratio": float((vals_counter[1] + vals_counter[10]) / n_values) if n_values else 0.0,
        "middle_3_to_8_ratio": float(sum(vals_counter[k] for k in range(3, 9)) / n_values) if n_values else 0.0,

        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "median_entropy": float(np.median(entropies)) if entropies else 0.0,
        "mean_top1_top2_margin": float(np.mean(margins)) if margins else 0.0,
        "median_top1_top2_margin": float(np.median(margins)) if margins else 0.0,
        "multi_relevant_ratio": float(multi_relevant / total_valid) if total_valid else 0.0,

        "dominant_dimension_counts": {str(i): dominant[i] for i in range(5)},
        "dominant_dimension_share": {str(i): float(dominant[i] / total_valid) if total_valid else 0.0 for i in range(5)},
        "max_dominant_share": float(max_dom_share),

        "domain_counts": dict(domain_counts),
        "domain_dominant_share": {
            domain: {
                str(i): float(domain_dominant[domain][i] / domain_counts[domain]) if domain_counts[domain] else 0.0
                for i in range(5)
            }
            for domain in domain_counts
        },

        "per_dimension_mean": {
            str(i): float(np.mean(per_dim_values[i])) if per_dim_values[i] else 0.0
            for i in range(5)
        },
        "per_dimension_std": {
            str(i): float(np.std(per_dim_values[i])) if per_dim_values[i] else 0.0
            for i in range(5)
        },

        "scalar_target_mean": float(scalar_targets.mean()) if scalar_targets.size else 0.0,
        "scalar_target_std": float(scalar_targets.std()) if scalar_targets.size else 0.0,
        "scalar_target_min": float(scalar_targets.min()) if scalar_targets.size else 0.0,
        "scalar_target_max": float(scalar_targets.max()) if scalar_targets.size else 0.0,
        "scalar_target_p10": float(np.percentile(scalar_targets, 10)) if scalar_targets.size else 0.0,
        "scalar_target_p90": float(np.percentile(scalar_targets, 90)) if scalar_targets.size else 0.0,

        "collapsed_topics_count": len(collapsed_topics),
        "collapsed_topics_sample": collapsed_topics[:30],

        "topic_reports": topic_reports,
    }

    return report


def verdict(report):
    failures = []
    warnings = []

    if report["bad_rows_count"] > 0:
        failures.append("Bad rows exist: malformed score vectors must be fixed.")

    if report["middle_3_to_8_ratio"] < 0.10:
        failures.append("Very low middle-score usage: labels are mostly hard 1/10.")
    elif report["middle_3_to_8_ratio"] < 0.25:
        warnings.append("Middle-score usage is modest: labels may still be somewhat hard.")

    if report["mean_entropy"] < 0.35:
        failures.append("Low entropy: dimension vectors are too one-hot.")
    elif report["mean_entropy"] < 0.45:
        warnings.append("Entropy is acceptable but still hard/classifier-like.")

    if report["multi_relevant_ratio"] < 0.10:
        failures.append("Very few multi-relevant questions: weak multi-dimensional signal.")
    elif report["multi_relevant_ratio"] < 0.20:
        warnings.append("Multi-relevant ratio is low but usable.")

    if report["max_dominant_share"] > 0.75:
        failures.append("One dominant dimension accounts for >75% of questions.")
    elif report["max_dominant_share"] > 0.60:
        warnings.append("Dominant dimension imbalance is high.")

    if report["scalar_target_std"] < 0.08:
        failures.append("Scalar target spread is too low: little training signal.")
    elif report["scalar_target_std"] < 0.12:
        warnings.append("Scalar target spread is modest.")

    if failures:
        status = "NOT IDEAL — consider improving prompt/regenerating before final training."
    elif warnings:
        status = "USABLE — train on it, but report limitations / inspect results."
    else:
        status = "GOOD — label distribution is suitable for training."

    return status, warnings, failures


def print_report(report, name="REPORT"):
    dim_names = WIKI_DIM_NAMES  # same indices; names differ for ecommerce but okay for general report

    status, warnings, failures = verdict(report)

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"File: {report['path']}")
    print(f"Topics: {report['topics']}")
    print(f"Rows: valid={report['valid_rows']} / total={report['total_rows']} | bad={report['bad_rows_count']}")

    print("\nScore counts:")
    for k in range(1, 11):
        print(f"  {k:2d}: {report['score_counts'].get(str(k), 0)}")

    print("\nDistribution:")
    print(f"  extreme_1_or_10_ratio: {report['extreme_1_or_10_ratio']:.3f}")
    print(f"  middle_3_to_8_ratio:   {report['middle_3_to_8_ratio']:.3f}")
    print(f"  mean_entropy:          {report['mean_entropy']:.3f}")
    print(f"  median_entropy:        {report['median_entropy']:.3f}")
    print(f"  mean_top1_top2_margin: {report['mean_top1_top2_margin']:.3f}")
    print(f"  multi_relevant_ratio:  {report['multi_relevant_ratio']:.3f}")

    print("\nDominant dimension share:")
    for i in range(5):
        print(f"  D{i} ({dim_names[i]}): {report['dominant_dimension_share'].get(str(i), 0.0):.3f}")

    print(f"  max_dominant_share:    {report['max_dominant_share']:.3f}")

    print("\nPer-dimension mean ± std:")
    for i in range(5):
        mean = report["per_dimension_mean"].get(str(i), 0.0)
        std = report["per_dimension_std"].get(str(i), 0.0)
        print(f"  D{i}: {mean:.3f} ± {std:.3f}")

    print("\nScalar target distribution under random alphas:")
    print(f"  mean: {report['scalar_target_mean']:.3f}")
    print(f"  std:  {report['scalar_target_std']:.3f}")
    print(f"  min:  {report['scalar_target_min']:.3f}")
    print(f"  max:  {report['scalar_target_max']:.3f}")
    print(f"  p10:  {report['scalar_target_p10']:.3f}")
    print(f"  p90:  {report['scalar_target_p90']:.3f}")

    print("\nCollapsed topics:")
    print(f"  count: {report['collapsed_topics_count']}")
    for r in report["collapsed_topics_sample"][:10]:
        print(
            f"  - {r['topic']} | domain={r['domain']} | "
            f"majority_dim={r['majority_dim']} | "
            f"majority_share={r['majority_share']:.2f} | "
            f"mean_std={r['mean_std_across_dims']:.2f}"
        )

    print("\nVerdict:")
    print(f"  {status}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")

    if report["bad_rows_count"] > 0:
        print("\nBad row sample:")
        for row in report["bad_rows_sample"]:
            print(" ", row)


def compare_reports(a, b, name_a="A", name_b="B"):
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    keys = [
        "valid_rows",
        "extreme_1_or_10_ratio",
        "middle_3_to_8_ratio",
        "mean_entropy",
        "multi_relevant_ratio",
        "max_dominant_share",
        "scalar_target_std",
        "collapsed_topics_count",
    ]

    print(f"{'metric':32s} {name_a:>14s} {name_b:>14s} {'delta(B-A)':>14s}")
    print("-" * 80)
    for k in keys:
        va = a[k]
        vb = b[k]
        if isinstance(va, int):
            print(f"{k:32s} {va:14d} {vb:14d} {vb-va:14d}")
        else:
            print(f"{k:32s} {va:14.4f} {vb:14.4f} {vb-va:14.4f}")


def save_json(report, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to dimension scores JSON")
    parser.add_argument("--compare", default=None, help="Optional second scores JSON to compare")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-alpha-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Optional path to save full report JSON")
    parser.add_argument("--compare-out", default=None, help="Optional path to save compare report JSON")
    args = parser.parse_args()

    report = compute_report(
        args.scores,
        data_dir=args.data_dir,
        n_alpha_samples=args.n_alpha_samples,
        seed=args.seed,
    )
    print_report(report, name="DIMENSION SCORE DIAGNOSTIC")

    if args.out:
        save_json(report, args.out)
        print(f"\nSaved report JSON to {args.out}")

    if args.compare:
        report_b = compute_report(
            args.compare,
            data_dir=args.data_dir,
            n_alpha_samples=args.n_alpha_samples,
            seed=args.seed,
        )
        print_report(report_b, name="COMPARISON FILE DIAGNOSTIC")
        compare_reports(report, report_b, name_a=Path(args.scores).name, name_b=Path(args.compare).name)

        if args.compare_out:
            save_json({"first": report, "second": report_b}, args.compare_out)
            print(f"\nSaved comparison JSON to {args.compare_out}")


if __name__ == "__main__":
    main()