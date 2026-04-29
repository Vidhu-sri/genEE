#!/usr/bin/env python3
"""
Diagnose quality of pairs.jsonl (3.1k dataset from GPT-5.2 web UI).
Checks for the same issues that killed the v2 dataset.

Usage:
  python diagnose_v1.py --input genEE/evaluator/pairs.jsonl
  python diagnose_v1.py --input /path/to/pairs.jsonl
"""

import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to pairs.jsonl")
    args = parser.parse_args()

    rows = []
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"{'='*60}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"  Total pairs: {len(rows)}")

    topics = Counter(r["topic"] for r in rows)
    domains = Counter(r["domain"] for r in rows)
    print(f"  Topics: {len(topics)}")
    print(f"  Domains: {dict(domains)}")
    print(f"  Pairs per topic: min={min(topics.values())} max={max(topics.values())} "
          f"avg={sum(topics.values())/len(topics):.1f}")

    # ── Winner balance ──
    print(f"\n{'='*60}")
    print(f"WINNER BALANCE")
    print(f"{'='*60}")
    w = Counter(r["winner"] for r in rows)
    total = w["A"] + w["B"]
    print(f"  Overall: A={w['A']} ({100*w['A']/total:.1f}%)  "
          f"B={w['B']} ({100*w['B']/total:.1f}%)")

    for dom in sorted(domains):
        subset = [r for r in rows if r["domain"] == dom]
        ww = Counter(r["winner"] for r in subset)
        t = ww["A"] + ww["B"]
        print(f"  {dom}: A={ww['A']} ({100*ww['A']/t:.1f}%)  "
              f"B={ww['B']} ({100*ww['B']/t:.1f}%)")

    # ── Difficulty distribution ──
    print(f"\n{'='*60}")
    print(f"DIFFICULTY DISTRIBUTION")
    print(f"{'='*60}")
    d = Counter(r["difficulty"] for r in rows)
    for diff in ["easy", "medium", "hard"]:
        n = d.get(diff, 0)
        print(f"  {diff:8s}: {n:4d} ({100*n/len(rows):.1f}%)")

    # ── Slice distribution ──
    s = Counter(r["slice"] for r in rows)
    print(f"\n  Slice: {dict(s)}")

    # ── Alpha diversity ──
    print(f"\n{'='*60}")
    print(f"ALPHA DIVERSITY")
    print(f"{'='*60}")
    maxes = [max(r["alpha"]) for r in rows]
    peaky = sum(1 for m in maxes if m > 0.5)
    moderate = sum(1 for m in maxes if 0.3 < m <= 0.5)
    mixed = sum(1 for m in maxes if m <= 0.3)
    print(f"  Peaky (max>0.5):    {peaky:4d} ({100*peaky/len(rows):.1f}%)")
    print(f"  Moderate (0.3-0.5): {moderate:4d} ({100*moderate/len(rows):.1f}%)")
    print(f"  Mixed (max≤0.3):    {mixed:4d} ({100*mixed/len(rows):.1f}%)")
    print(f"  Alpha max range: [{min(maxes):.3f}, {max(maxes):.3f}]")

    # Show a few alpha samples
    import random
    random.seed(42)
    samples = random.sample(rows, min(5, len(rows)))
    print(f"\n  Sample alphas:")
    for r in samples:
        dom_idx = max(range(5), key=lambda i: r["alpha"][i])
        print(f"    {r['alpha']}  dominant=dim{dom_idx}  winner={r['winner']}")

    # ── CRITICAL: Dominant dim → winner bias ──
    # This is what killed v2: dim 0 → A wins 63%, dim 4 → A wins 43%
    print(f"\n{'='*60}")
    print(f"CRITICAL: DOMINANT DIM → WINNER BIAS")
    print(f"(v2 dataset failed here: dim0→A=63%, dim4→A=43%)")
    print(f"(healthy = all dims ~50% A)")
    print(f"{'='*60}")

    DIM_NAMES = {
        "wikipedia": ["Discussion", "History", "Event", "Person", "Location"],
        "ecommerce": ["Price", "Quality", "Brand", "Features", "Ethical"],
    }

    for dom in sorted(domains):
        subset = [r for r in rows if r["domain"] == dom]
        dim_names = DIM_NAMES.get(dom, [f"dim{i}" for i in range(5)])
        print(f"\n  {dom} ({len(subset)} pairs):")
        for d_idx in range(5):
            dr = [r for r in subset
                  if max(range(5), key=lambda i: r["alpha"][i]) == d_idx]
            if not dr:
                print(f"    dim {d_idx} ({dim_names[d_idx]:15s}): no pairs")
                continue
            a_wins = sum(1 for r in dr if r["winner"] == "A")
            a_pct = 100 * a_wins / len(dr)
            bias = "⚠️ BIASED" if abs(a_pct - 50) > 10 else "✓ OK"
            print(f"    dim {d_idx} ({dim_names[d_idx]:15s}): n={len(dr):3d}  "
                  f"A={a_pct:.1f}%  B={100-a_pct:.1f}%  {bias}")

    # ── Per-difficulty winner balance ──
    print(f"\n{'='*60}")
    print(f"PER-DIFFICULTY WINNER BALANCE")
    print(f"(healthy = each difficulty ~50% A)")
    print(f"{'='*60}")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in rows if r["difficulty"] == diff]
        if not subset:
            continue
        a_wins = sum(1 for r in subset if r["winner"] == "A")
        a_pct = 100 * a_wins / len(subset)
        bias = "⚠️" if abs(a_pct - 50) > 10 else "✓"
        print(f"  {diff:8s}: n={len(subset):4d}  A={a_pct:.1f}%  B={100-a_pct:.1f}%  {bias}")

    # ── Keyword heuristic test ──
    # For peaky alphas, does the winning question's text match the dominant dim?
    print(f"\n{'='*60}")
    print(f"KEYWORD HEURISTIC (peaky alphas only, max>0.5)")
    print(f"(v2 got 53.7% here — barely above random)")
    print(f"(healthy = >65%)")
    print(f"{'='*60}")

    DIM_KEYWORDS = {
        "wikipedia": {
            0: {"discuss", "what", "concept", "define", "meaning", "philosophy", "theory", "view", "role", "idea"},
            1: {"history", "historical", "origin", "founded", "ancient", "century", "when", "timeline", "era"},
            2: {"event", "battle", "war", "revolution", "crisis", "incident", "influence", "impact", "turning"},
            3: {"who", "person", "biography", "thinker", "philosopher", "leader", "known", "famous", "epictetus"},
            4: {"where", "place", "geography", "city", "country", "region", "location", "practiced", "today", "modern"},
        },
        "ecommerce": {
            0: {"price", "cost", "cheap", "expensive", "budget", "affordable", "value", "deal", "discount"},
            1: {"quality", "durable", "reliable", "material", "sturdy", "premium", "well-made", "build"},
            2: {"brand", "reputation", "popular", "trusted", "review", "rating", "recommend", "top-rated"},
            3: {"feature", "function", "spec", "capability", "performance", "technology", "design", "versatile"},
            4: {"ethical", "sustainable", "eco", "organic", "fair", "environmental", "responsible", "green"},
        },
    }

    correct = 0
    tested = 0
    for r in rows:
        if max(r["alpha"]) < 0.5:
            continue
        dom = r["domain"]
        if dom not in DIM_KEYWORDS:
            continue
        dom_dim = max(range(5), key=lambda i: r["alpha"][i])
        keywords = DIM_KEYWORDS[dom][dom_dim]

        q_win = r["q_a"].lower() if r["winner"] == "A" else r["q_b"].lower()
        q_lose = r["q_b"].lower() if r["winner"] == "A" else r["q_a"].lower()

        win_hits = sum(1 for kw in keywords if kw in q_win)
        lose_hits = sum(1 for kw in keywords if kw in q_lose)

        if win_hits > lose_hits:
            correct += 1
        tested += 1

    if tested > 0:
        print(f"  Tested: {tested} peaky pairs")
        print(f"  Keyword heuristic accuracy: {correct}/{tested} = {100*correct/tested:.1f}%")
    else:
        print(f"  No peaky pairs found")

    # ── GPT-4 agreement check (using cached scores if available) ──
    print(f"\n{'='*60}")
    print(f"GPT-4 EVALUATOR AGREEMENT (if cache available)")
    print(f"{'='*60}")

    cache_paths = [
        "gpt4_score_cache.json",
        "genEE/evaluator/gpt4_score_cache.json",
        "./data/gpt4_score_cache.json",
    ]
    cache = {}
    for cp in cache_paths:
        if Path(cp).exists():
            cache = json.loads(Path(cp).read_text())
            print(f"  Loaded cache from {cp} ({len(cache)} entries)")
            break

    if cache:
        agree = 0
        checked = 0
        for r in rows:
            dom = r["domain"]
            topic = r["topic"]
            ck_a = f"{dom}|{topic}|{r['q_a']}"
            ck_b = f"{dom}|{topic}|{r['q_b']}"
            if ck_a in cache and ck_b in cache:
                ra = cache[ck_a]
                rb = cache[ck_b]
                # Skip fallback [5,5,5,5,5] entries
                if ra == [5,5,5,5,5] or rb == [5,5,5,5,5]:
                    continue
                alpha = r["alpha"]
                sa = sum(a * s / 10.0 for a, s in zip(alpha, ra))
                sb = sum(a * s / 10.0 for a, s in zip(alpha, rb))
                gpt_winner = "A" if sa > sb else "B"
                if gpt_winner == r["winner"]:
                    agree += 1
                checked += 1

        if checked > 0:
            print(f"  Checked: {checked} pairs with cached GPT-4 scores")
            print(f"  Agreement: {agree}/{checked} = {100*agree/checked:.1f}%")
        else:
            print(f"  No overlapping cached scores found")
    else:
        print(f"  No cache file found — skip")

    # ── Unique questions per topic ──
    print(f"\n{'='*60}")
    print(f"QUESTION COVERAGE")
    print(f"{'='*60}")
    for topic in sorted(topics)[:5]:
        subset = [r for r in rows if r["topic"] == topic]
        qs = set()
        for r in subset:
            qs.add(r["q_a"])
            qs.add(r["q_b"])
        print(f"  {topic:30s}: {len(subset):3d} pairs, {len(qs):2d} unique questions")
    if len(topics) > 5:
        print(f"  ... ({len(topics) - 5} more topics)")

    # ── VERDICT ──
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    issues = []

    # Check winner balance
    a_pct_overall = 100 * w["A"] / total
    if abs(a_pct_overall - 50) > 8:
        issues.append(f"Winner imbalance: A={a_pct_overall:.1f}%")

    # Check dim bias (worst case)
    worst_bias = 0
    for dom in domains:
        subset = [r for r in rows if r["domain"] == dom]
        for d_idx in range(5):
            dr = [r for r in subset
                  if max(range(5), key=lambda i: r["alpha"][i]) == d_idx]
            if len(dr) >= 10:
                a_pct = 100 * sum(1 for r in dr if r["winner"] == "A") / len(dr)
                worst_bias = max(worst_bias, abs(a_pct - 50))
    if worst_bias > 12:
        issues.append(f"Dim→winner bias: worst deviation = {worst_bias:.1f}pp from 50%")

    # Check keyword heuristic
    if tested > 0:
        kw_acc = 100 * correct / tested
        if kw_acc < 55:
            issues.append(f"Keyword heuristic only {kw_acc:.1f}% (labels may not follow alpha)")

    if not issues:
        print(f"  ✓ Dataset looks CLEAN. No major issues detected.")
        print(f"  ✓ Safe to use for training or as ground truth.")
    else:
        print(f"  Issues found:")
        for iss in issues:
            print(f"    ⚠️  {iss}")
        if len(issues) <= 1 and worst_bias < 15:
            print(f"\n  Minor issues only — dataset is probably usable.")
        else:
            print(f"\n  Significant issues — consider regenerating or filtering.")


if __name__ == "__main__":
    main()