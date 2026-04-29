#!/usr/bin/env python3
"""
Generate ~19,000 pairwise preference pairs from ip0.json (v2).

Key improvements over v1:
  - Alphas pre-generated via Dirichlet (40% peaky, 40% moderate, 20% mixed)
  - A/B position randomized to eliminate index-order bias
  - Question index numbers stripped from prompt text
  - GPT only decides winner/slice/difficulty (no alpha generation)
  - Supports gpt-5-mini (default) or gpt-5
  - Resume-safe with state file
  - --sample-pairs flag to reduce pairs per topic and save cost
  - --dry-run to preview prompts without API calls

Usage:
  pip install openai python-dotenv numpy
  # set OPENAI_API_KEY in .env or environment

  python generate_pairs.py                                  # default: gpt-5-mini, all 190 pairs/topic
  python generate_pairs.py --model gpt-5 --effort low       # higher quality, ~$40-55
  python generate_pairs.py --sample-pairs 100               # ~10k pairs total, saves ~50%
  python generate_pairs.py --dry-run                        # preview prompt, no API calls
"""

import json
import os
import time
import itertools
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IP0_PATH   = "./data/ip0.json"
OUT_PATH   = "./data/pairs_v2.jsonl"
STATE_PATH = "./data/pairs_v2_state.json"

MAX_ROUNDS_PER_CHUNK = 4    # repair retries per chunk
MAX_RETRIES = 3             # API retries per call
RETRY_SLEEP = 8
CHUNK_SIZE  = 80            # pairs per API call

ECOMMERCE_TOPICS = {
    "Spray Bottles", "Cookware Sets", "TV Antennas", "DVI",
    "Measuring Tools & Scales", "Coaxial Cables", "Area Rugs",
    "Lighting Cables", "Kitchen Sinks", "Wall Plates", "Clips",
    "Hair Treatment Oils", "Temporary Tattoos", "Spoons",
    "Boot & Shoe Covers", "Fuses", "Computers & Accessories",
    "Wireless Access Points", "Safety Work Gloves", "Tea Accessories",
    "Bathroom Vanities", "Bookshelf Albums", "Telescopes",
    "Dining Chair Slipcovers", "Single Rods", "Home Bar Furniture",
    "Lighting & Ceiling Fans", "Vehicle Backup Cameras", "Drills",
    "Tablet Accessories", "Champagne Glasses", "Window Treatments",
    "Diamond Blades", "Hair Combs", "Bath Bombs", "Table Saw Accessories",
    "Speaker", "Item Finders", "Computer Cases", "Racks, Shelves & Drawers",
    "Surveillance Video Recorders", "Over-Ear Headphones", "Garage Storage",
    "Refillable Containers", "Camera & Photo", "Specialty Tools & Gadgets",
    "Lash Enhancers & Primers", "Conditioners", "Electrical", "Vacuums",
}

ECOM_DIMS = ["Price", "Quality", "Brand Reputation", "Features/Functionality", "Ethical Considerations"]
WIKI_DIMS = ["Discussion", "History", "Event", "Person", "Location"]


# ─────────────────────────────────────────────
# System prompt — GPT only picks winner, slice, difficulty
# ─────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You label pairwise question preferences for training a recommender evaluator.

DOMAIN: {domain}
PREFERENCE DIMENSIONS: {dims}

For each pair you receive:
  idx   — pair identifier (return unchanged)
  alpha — [a0..a4] user preference weights over the 5 dimensions
  q_a   — question A
  q_b   — question B

YOUR TASK per pair:
  1. "winner": "A" or "B" — which question better serves a user with that alpha.
     High weight on dimension i means the user strongly wants questions about dimension i.
     For hard/close pairs the winner can go either way — use your judgment.
  2. "slice": "exploit" (directly addresses dominant alpha dimension)
              or "explore" (broadens into less-weighted dimensions).
  3. "difficulty": "easy" (obvious), "medium" (reasonable), "hard" (genuinely close).

OUTPUT FORMAT:
- ONLY valid JSONL (one JSON object per line). No markdown, no commentary.
- Keys per line: idx, winner, slice, difficulty
- Exactly N lines for N input pairs.

AVOID THESE BIASES:
- Do NOT prefer longer questions over shorter ones.
- Do NOT prefer q_a over q_b or vice versa — position is randomized.
- Do NOT assume questions with higher original indices are better.
- Decide ONLY based on alpha-question alignment."""


# ─────────────────────────────────────────────
# Alpha generation via Dirichlet
# ─────────────────────────────────────────────
def generate_alpha(rng: np.random.Generator) -> List[float]:
    """
    Sample one alpha vector with controlled diversity:
      40% peaky   — one dominant dim (concentration 5.0 vs 0.3)
      40% moderate — one elevated dim (concentration 3.0 vs 1.0)
      20% mixed   — roughly uniform (all concentrations 2.0)
    """
    r = rng.random()
    if r < 0.4:
        conc = np.ones(5) * 0.3
        conc[rng.integers(5)] = 5.0
    elif r < 0.8:
        conc = np.ones(5) * 1.0
        conc[rng.integers(5)] = 3.0
    else:
        conc = np.ones(5) * 2.0

    a = rng.dirichlet(conc)
    a = np.round(a, 4)
    a[0] += round(1.0 - a.sum(), 4)
    a = np.round(a, 4)
    return a.tolist()


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────
def domain_for_topic(topic: str) -> str:
    return "ecommerce" if topic in ECOMMERCE_TOPICS else "wikipedia"


def dims_for_domain(domain: str) -> List[str]:
    return ECOM_DIMS if domain == "ecommerce" else WIKI_DIMS


def all_pairs_indices(nq: int) -> List[Tuple[int, int]]:
    return list(itertools.combinations(range(nq), 2))


def sample_pairs(
    all_p: List[Tuple[int, int]], n: int, nq: int, rng: random.Random
) -> List[Tuple[int, int]]:
    """Sample n pairs ensuring every question appears in >= min_per_q pairs."""
    if n >= len(all_p):
        return list(all_p)

    min_per_q = max(4, n // nq)
    selected: Set[Tuple[int, int]] = set()
    q_counts = {i: 0 for i in range(nq)}

    shuffled = list(all_p)
    rng.shuffle(shuffled)

    for ia, ib in shuffled:
        if q_counts[ia] < min_per_q or q_counts[ib] < min_per_q:
            selected.add((ia, ib))
            q_counts[ia] += 1
            q_counts[ib] += 1
        if len(selected) >= n:
            break

    for ia, ib in shuffled:
        if len(selected) >= n:
            break
        if (ia, ib) not in selected:
            selected.add((ia, ib))

    return sorted(selected)


def pair_key(ia: int, ib: int) -> str:
    a, b = min(ia, ib), max(ia, ib)
    return f"{a}-{b}"


def load_state(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {"done": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(path: str, state: Dict) -> None:
    Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")


def chunked(items: list, k: int) -> List[list]:
    return [items[i:i + k] for i in range(0, len(items), k)]


# ─────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────
def build_system_prompt(domain: str, dims: List[str]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        domain=domain,
        dims=", ".join(f"{i}={d}" for i, d in enumerate(dims)),
    )


def build_user_prompt(
    topic: str,
    questions: List[str],
    chunk_data: List[dict],
) -> str:
    """Build user prompt. Questions listed neutrally, pairs include pre-assigned alpha."""
    q_block = "\n".join([f"  Q{i + 1}: {q}" for i, q in enumerate(questions)])

    pair_lines = []
    for d in chunk_data:
        pair_lines.append(json.dumps({
            "idx":   d["idx"],
            "alpha": d["alpha"],
            "q_a":   questions[d["shown_a"]],
            "q_b":   questions[d["shown_b"]],
        }, ensure_ascii=False))

    return f"""TOPIC: {topic}

ALL QUESTIONS (for context — ordering is arbitrary):
{q_block}

INPUT PAIRS (output one JSONL decision line per pair):
{chr(10).join(pair_lines)}

N={len(chunk_data)}"""


# ─────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────
def load_client() -> OpenAI:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment/.env")
    return OpenAI(api_key=key)


def call_api(client: OpenAI, model: str, effort: str, system: str, user: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=model,
                reasoning={"effort": effort},
                input=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.output_text
        except Exception as e:
            last_err = e
            print(f"    [api] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(RETRY_SLEEP * attempt)
    raise RuntimeError(f"API failed after {MAX_RETRIES} attempts: {last_err}")


# ─────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────
def parse_response(text: str, expected_idxs: Set[str]) -> Dict[str, dict]:
    """Parse JSONL response. Returns {idx: {winner, slice, difficulty}}."""
    results = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        idx = str(obj.get("idx", ""))
        if idx not in expected_idxs or idx in results:
            continue

        winner = obj.get("winner")
        slc    = obj.get("slice")
        diff   = obj.get("difficulty")

        if winner not in ("A", "B"):
            continue
        if slc not in ("exploit", "explore"):
            continue
        if diff not in ("easy", "medium", "hard"):
            continue

        results[idx] = {"winner": winner, "slice": slc, "difficulty": diff}
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate pairwise preference dataset v2")
    parser.add_argument("--model", default="gpt-5-mini",
                        choices=["gpt-5-mini", "gpt-5"],
                        help="Model (default: gpt-5-mini)")
    parser.add_argument("--effort", default="low",
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort (default: low)")
    parser.add_argument("--sample-pairs", type=int, default=0,
                        help="Sample this many pairs per topic (0 = all C(n,2))")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ip0", default=IP0_PATH)
    parser.add_argument("--output", default=OUT_PATH)
    parser.add_argument("--state", default=STATE_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompt + cost estimate, no API calls")
    args = parser.parse_args()

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    ip0 = json.loads(Path(args.ip0).read_text(encoding="utf-8"))
    topics = list(ip0.keys())
    print(f"Loaded {len(topics)} topics from {args.ip0}")

    state = load_state(args.state)
    done = state.get("done", {})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = None if args.dry_run else load_client()

    # ── Cost estimate ──
    total_pairs = 0
    for topic in topics:
        nq = len(ip0[topic])
        n_all = len(all_pairs_indices(nq))
        total_pairs += min(args.sample_pairs, n_all) if args.sample_pairs > 0 else n_all

    est_calls = (total_pairs + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total pairs: {total_pairs}")
    print(f"Estimated API calls: ~{est_calls} (chunks of {CHUNK_SIZE})")
    print(f"Model: {args.model}, Effort: {args.effort}")
    print()

    out_file = out_path.open("a", buffering=1, encoding="utf-8")
    grand_wrote = 0

    for ti, topic in enumerate(topics, 1):
        questions = ip0[topic]
        nq = len(questions)
        if not isinstance(questions, list) or nq < 2:
            print(f"[{ti:03d}] {topic} — skip ({nq} questions)")
            continue

        domain = domain_for_topic(topic)
        dims = dims_for_domain(domain)

        all_p = all_pairs_indices(nq)
        if args.sample_pairs > 0:
            topic_pairs = sample_pairs(all_p, args.sample_pairs, nq, py_rng)
        else:
            topic_pairs = all_p

        done_keys = set(done.get(topic, []))
        needed = [(ia, ib) for ia, ib in topic_pairs
                  if pair_key(ia, ib) not in done_keys]
        n_total = len(topic_pairs)

        if not needed:
            print(f"[{ti:03d}/{len(topics)}] {topic} — complete ({n_total}/{n_total})")
            continue

        print(f"[{ti:03d}/{len(topics)}] {topic} ({domain}) — "
              f"need {len(needed)}/{n_total}")

        sys_prompt = build_system_prompt(domain, dims)

        # ── Pre-generate alphas + randomize A/B ──
        pair_meta: Dict[str, dict] = {}
        for ia, ib in needed:
            alpha = generate_alpha(np_rng)
            swapped = py_rng.random() < 0.5
            pk = pair_key(ia, ib)
            pair_meta[pk] = {
                "idx":     pk,
                "ia":      ia,
                "ib":      ib,
                "alpha":   alpha,
                "swapped": swapped,
                "shown_a": ib if swapped else ia,
                "shown_b": ia if swapped else ib,
            }

        needed_keys = [pair_key(ia, ib) for ia, ib in needed]
        chunks = chunked(needed_keys, CHUNK_SIZE)

        for ci, chunk_keys in enumerate(chunks, 1):
            remaining_keys = set(chunk_keys) - done_keys

            for round_i in range(1, MAX_ROUNDS_PER_CHUNK + 1):
                if not remaining_keys:
                    break

                chunk_data = [pair_meta[k] for k in sorted(remaining_keys)]
                user_prompt = build_user_prompt(topic, questions, chunk_data)

                # ── Dry run ──
                if args.dry_run:
                    print("\n" + "=" * 60)
                    print("SYSTEM PROMPT")
                    print("=" * 60)
                    print(sys_prompt)
                    print("\n" + "=" * 60)
                    print("USER PROMPT (first chunk, truncated)")
                    print("=" * 60)
                    print(user_prompt[:4000])
                    if len(user_prompt) > 4000:
                        print(f"\n... ({len(user_prompt)} chars total)")

                    print(f"\nPairs this chunk: {len(chunk_data)}")
                    print(f"Total pairs all topics: {total_pairs}")
                    print(f"Estimated API calls: ~{est_calls}")

                    sample_a = [d["alpha"] for d in chunk_data[:10]]
                    print("\nSample alphas (first 10):")
                    for a in sample_a:
                        mx = max(a)
                        kind = "peaky" if mx > 0.5 else ("moderate" if mx > 0.3 else "mixed")
                        print(f"  {a}  ({kind}, max={mx:.3f})")

                    n_sw = sum(1 for d in chunk_data if d["swapped"])
                    print(f"\nA/B swaps: {n_sw}/{len(chunk_data)} ({100*n_sw/len(chunk_data):.0f}%)")

                    out_file.close()
                    return

                # ── Call API ──
                raw = call_api(client, args.model, args.effort, sys_prompt, user_prompt)
                parsed = parse_response(raw, remaining_keys)

                wrote = 0
                for idx_key, decision in parsed.items():
                    meta = pair_meta[idx_key]

                    # Un-swap winner
                    raw_winner = decision["winner"]
                    if meta["swapped"]:
                        actual_winner = "B" if raw_winner == "A" else "A"
                    else:
                        actual_winner = raw_winner

                    row = {
                        "topic":      topic,
                        "domain":     domain,
                        "alpha":      meta["alpha"],
                        "q_a":        questions[meta["ia"]],
                        "q_b":        questions[meta["ib"]],
                        "winner":     actual_winner,
                        "slice":      decision["slice"],
                        "difficulty": decision["difficulty"],
                    }
                    out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done_keys.add(idx_key)
                    wrote += 1

                remaining_keys -= set(parsed.keys())
                done[topic] = sorted(done_keys)
                state["done"] = done
                save_state(args.state, state)

                print(f"  chunk {ci}/{len(chunks)} round {round_i}: "
                      f"wrote {wrote}, remaining={len(remaining_keys)}")

                if remaining_keys:
                    time.sleep(0.5)

            grand_wrote += (len(chunk_keys) - len(remaining_keys))

            if remaining_keys:
                print(f"  WARNING: {topic} chunk {ci} — "
                      f"{len(remaining_keys)} unfilled")

            time.sleep(0.2)

    out_file.close()

    # ── Summary + quality stats ──
    print(f"\n{'=' * 50}")
    print(f"Done. Newly written: {grand_wrote}")
    print(f"Output: {args.output}")
    print(f"State:  {args.state}")

    if out_path.exists() and out_path.stat().st_size > 0:
        winners = {"A": 0, "B": 0}
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        slices = {"exploit": 0, "explore": 0}
        alpha_maxes = []
        total_lines = 0

        with out_path.open(encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                try:
                    obj = json.loads(line)
                    winners[obj["winner"]] += 1
                    difficulties[obj["difficulty"]] += 1
                    slices[obj["slice"]] += 1
                    alpha_maxes.append(max(obj["alpha"]))
                except Exception:
                    pass

        print(f"Total lines: {total_lines}")

        if alpha_maxes:
            tw = winners["A"] + winners["B"]
            print(f"\nQuality stats:")
            print(f"  Winner:  A={winners['A']} ({100*winners['A']/tw:.1f}%)  "
                  f"B={winners['B']} ({100*winners['B']/tw:.1f}%)")
            print(f"  Difficulty: {dict(difficulties)}")
            print(f"  Slice: {dict(slices)}")

            peaky = sum(1 for m in alpha_maxes if m > 0.5)
            moderate = sum(1 for m in alpha_maxes if 0.3 < m <= 0.5)
            mixed = sum(1 for m in alpha_maxes if m <= 0.3)
            print(f"  Alpha types: peaky={peaky} ({100*peaky/len(alpha_maxes):.0f}%), "
                  f"moderate={moderate} ({100*moderate/len(alpha_maxes):.0f}%), "
                  f"mixed={mixed} ({100*mixed/len(alpha_maxes):.0f}%)")
            print(f"  Alpha max: [{min(alpha_maxes):.3f}, {max(alpha_maxes):.3f}]")


if __name__ == "__main__":
    main()