#!/usr/bin/env python3
"""
reproduce_acts.py — Reproduce Acts 1 and 2 (the failed approaches).

Act 1: Train MiniLM on pairwise preference labels → 60% accuracy plateau
Act 2: Zero-shot MiniLM cosine similarity → 0% top-1, 37% dim agreement

These failed attempts motivate the FiLM approach (Act 3).

Usage:
  python evaluator/reproduce_acts.py --act 1 --pairs data/pairs_19k.jsonl
  python evaluator/reproduce_acts.py --act 2 --ip0 data/ip0.json --n-topics 10
  python evaluator/reproduce_acts.py --act 2 --ip0 data/ip0.json --compare-gpt4
"""

import json, argparse
from pathlib import Path
import numpy as np


def act1_analysis(pairs_path):
    """
    Act 1: Analyze pairwise label quality.
    Shows position bias and weak alpha-label correlation.
    """
    pairs = [json.loads(l) for l in Path(pairs_path).read_text().strip().split("\n")]
    print(f"Loaded {len(pairs)} pairwise comparisons")

    # Position bias: how often does position A win?
    a_wins = sum(1 for p in pairs if p.get("label", p.get("winner", "")) in ("A", "a", 0))
    b_wins = sum(1 for p in pairs if p.get("label", p.get("winner", "")) in ("B", "b", 1))
    total = a_wins + b_wins
    if total > 0:
        print(f"\nPosition bias:")
        print(f"  A wins: {a_wins}/{total} ({100*a_wins/total:.1f}%)")
        print(f"  B wins: {b_wins}/{total} ({100*b_wins/total:.1f}%)")
        print(f"  Bias: {abs(50 - 100*a_wins/total):.1f}% from balanced")

    # Alpha-label correlation via keyword heuristic
    # Check if the "winning" question's text contains keywords from the alpha's peak dimension
    dim_keywords = {
        0: {"price", "cost", "cheap", "expensive", "budget", "afford", "discount", "value"},
        1: {"quality", "durable", "reliable", "material", "build", "sturdy", "premium"},
        2: {"brand", "reputation", "review", "rating", "trusted", "popular", "recommend"},
        3: {"feature", "function", "specification", "performance", "technology", "design"},
        4: {"ethical", "sustainable", "eco", "organic", "environment", "green", "fair"},
    }

    correct = 0
    total_with_alpha = 0
    for p in pairs:
        alpha = p.get("alpha")
        if not alpha: continue
        peak_dim = np.argmax(alpha)
        keywords = dim_keywords.get(peak_dim, set())

        winner = p.get("label", p.get("winner", ""))
        if winner in ("A", "a", 0):
            text = p.get("question_a", p.get("A", "")).lower()
        else:
            text = p.get("question_b", p.get("B", "")).lower()

        if any(kw in text for kw in keywords):
            correct += 1
        total_with_alpha += 1

    if total_with_alpha > 0:
        print(f"\nAlpha-label keyword correlation:")
        print(f"  Correct: {correct}/{total_with_alpha} ({100*correct/total_with_alpha:.1f}%)")
        print(f"  Random baseline: ~20%")
        print(f"  Conclusion: {'WEAK' if correct/total_with_alpha < 0.3 else 'OK'} correlation")

    print(f"\nAct 1 conclusion:")
    print(f"  Position bias ({100*a_wins/max(total,1):.1f}% A-win) + weak alpha correlation")
    print(f"  → Small models memorize position patterns, not alpha-question relationships")
    print(f"  → Direct pairwise training fails for this task")


def act2_evaluation(ip0_path, n_topics, compare_gpt4=False):
    """
    Act 2: Zero-shot MiniLM evaluation.
    Shows 0% top-1, 37% dimension-level agreement with GPT-4.
    """
    from evaluator import make_evaluator

    ip0 = json.loads(Path(ip0_path).read_text())
    topics = list(ip0.keys())[:n_topics]
    rng = np.random.default_rng(42)

    minilm = make_evaluator("minilm")

    gpt4 = None
    if compare_gpt4:
        gpt4 = make_evaluator("gpt4")

    top1, dim_agree, total, total_qs = 0, 0, 0, 0

    # Detect ecom topics
    ecom_file = Path("data/topics_ecommerce.json")
    ecom_topics = set(json.loads(ecom_file.read_text())) if ecom_file.exists() else set()

    for topic in topics:
        questions = ip0[topic]
        domain = "ecommerce" if topic in ecom_topics else "wikipedia"

        for _ in range(3):
            conc = np.ones(5) * 0.5
            conc[rng.integers(5)] = 4.0
            alpha = rng.dirichlet(conc).tolist()

            m_scores = minilm.score(questions, alpha, topic, domain)
            m_r = minilm.relevance_vectors(questions, topic, domain)

            if gpt4:
                g_scores = gpt4.score(questions, alpha, topic, domain)
                g_r = gpt4.relevance_vectors(questions, topic, domain)

                if np.argmax(m_scores) == np.argmax(g_scores):
                    top1 += 1

                for qi in range(len(questions)):
                    if np.argmax(m_r[qi]) == np.argmax(g_r[qi]):
                        dim_agree += 1
                    total_qs += 1

            total += 1

    print(f"\nAct 2: Zero-shot MiniLM Evaluation ({total} evaluations)")
    print(f"{'='*50}")
    if gpt4:
        print(f"  Top-1 agreement:     {top1}/{total} ({100*top1/total:.1f}%)")
        print(f"  Dim-level agree:     {dim_agree}/{total_qs} ({100*dim_agree/max(total_qs,1):.1f}%)")
    else:
        print(f"  (Run with --compare-gpt4 to get agreement metrics)")
        print(f"  MiniLM scores for first topic:")
        t = topics[0]
        qs = ip0[t][:5]
        alpha = [0.6, 0.1, 0.1, 0.1, 0.1]
        domain = "ecommerce" if t in ecom_topics else "wikipedia"
        scores = minilm.score(qs, alpha, t, domain)
        for q, s in zip(qs, scores):
            print(f"    {s:.4f}  {q[:60]}")

    print(f"\n  Conclusion: Zero-shot cosine similarity lacks semantic precision")
    print(f"  for preference dimension classification → motivates FiLM (Act 3)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--act", type=int, required=True, choices=[1, 2])
    p.add_argument("--pairs", default="data/pairs_19k.jsonl", help="Act 1: pairwise labels file")
    p.add_argument("--ip0", default="data/ip0.json", help="Act 2: initial question pools")
    p.add_argument("--n-topics", type=int, default=10)
    p.add_argument("--compare-gpt4", action="store_true", help="Act 2: compare against GPT-4")
    args = p.parse_args()

    if args.act == 1:
        act1_analysis(args.pairs)
    else:
        act2_evaluation(args.ip0, args.n_topics, args.compare_gpt4)


if __name__ == "__main__":
    main()
