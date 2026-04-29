#!/usr/bin/env python3
"""
generate_dimension_scores.py — Create FiLM training data from GPT-4.

For each question in ip0.json, GPT-4o-mini rates its relevance to each
of the 5 persona dimensions on a scale of 1-10.

This is the "what is this question about" step — the hard part that
requires an LLM. The FiLM model then learns the easy part:
"how does a user with alpha value it."

Input:  data/ip0.json
Output: data/gpt4_dimension_scores.json

Cost: ~$0.05 for ~1000 questions
Time: ~2-3 minutes

Usage:
  python evaluator/generate_dimension_scores.py
  python evaluator/generate_dimension_scores.py --model gpt-4o-mini --ip0 data/ip0.json
"""

import json, os, time, argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ECOM_DIM_NAMES = ["Price", "Quality", "Brand Reputation", "Features", "Ethical"]
WIKI_DIM_NAMES = ["Discussion", "History", "Event", "Person", "Location"]

ECOM_DIM_DESCRIPTIONS = [
    "Price, cost, affordable, budget, expensive, cheap, value, deal, discount, pricing",
    "Quality, durable, reliable, materials, craftsmanship, build, sturdy, premium, well-made",
    "Brand, reputation, popular, trusted, reviews, ratings, well-known, recommended, top-rated",
    "Features, functionality, specifications, capabilities, performance, technology, design, versatile",
    "Ethical, sustainable, eco-friendly, fair trade, organic, environmental, responsible, green",
]
WIKI_DIM_DESCRIPTIONS = [
    "Discussion, debate, definition, explanation, overview, concepts, ideas, meaning, philosophy, theory",
    "History, historical, origins, timeline, founded, ancient, century, era, development, evolution",
    "Events, battles, wars, revolutions, incidents, occurrences, crises, turning points, milestones",
    "People, persons, biography, thinker, philosopher, leader, figure, life, born, known for, role",
    "Locations, places, geography, city, country, region, where, practiced, spread, modern, today",
]

# Add your ecommerce topics here to auto-detect domain
ECOMMERCE_TOPICS = set()  # Will be populated from personas.json if available

PROMPT = """You are evaluating questions suggested for the topic "{topic}" in the {domain} domain.

For each question below, rate its relevance to EACH of the following preference dimensions on a scale of 1-10:
{dim_descriptions}

Rate how well the question serves a user who cares about that specific dimension.
- 1 = completely irrelevant to that dimension
- 5 = somewhat relevant
- 10 = perfectly targets that dimension

Output ONLY a JSON array. Each element: {{"idx": <int>, "scores": [<5 ints>]}}.
No markdown, no commentary, no extra text.

Questions:
{questions_block}
"""


def detect_domain(topic, data_path=None):
    """Detect domain from topic name. Uses topics files if available."""
    if data_path:
        ecom_file = Path(data_path) / "topics_ecommerce.json"
        if ecom_file.exists():
            ecom_topics = set(json.loads(ecom_file.read_text()))
            if topic in ecom_topics:
                return "ecommerce"
            return "wikipedia"
    # Fallback: simple heuristic
    return "ecommerce" if topic in ECOMMERCE_TOPICS else "wikipedia"


def score_batch(client, model, topic, domain, questions):
    if domain == "ecommerce":
        dim_names, dim_descs = ECOM_DIM_NAMES, ECOM_DIM_DESCRIPTIONS
    else:
        dim_names, dim_descs = WIKI_DIM_NAMES, WIKI_DIM_DESCRIPTIONS

    dim_block = "\n".join(f"  Dim {i} ({dim_names[i]}): {dim_descs[i]}" for i in range(5))
    q_block = "\n".join(f"  idx={i}: {q}" for i, q in enumerate(questions))
    prompt = PROMPT.format(topic=topic, domain=domain, dim_descriptions=dim_block, questions_block=q_block)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            results = {}
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    for obj in arr:
                        idx, scores = obj.get("idx"), obj.get("scores", [])
                        if idx is not None and len(scores) == 5:
                            results[int(idx)] = [max(1, min(10, int(s))) for s in scores]
                    if results:
                        return results
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            # Fallback JSONL
            for line in text.splitlines():
                line = line.strip().rstrip(",")
                if not line or line in ("[]", "[", "]"): continue
                try:
                    obj = json.loads(line)
                    idx, scores = obj.get("idx"), obj.get("scores", [])
                    if idx is not None and len(scores) == 5:
                        results[int(idx)] = [max(1, min(10, int(s))) for s in scores]
                except: continue
            if results: return results
            print(f"  [warn] parse failed for {topic}, attempt {attempt+1}: {text[:200]}")
        except Exception as e:
            print(f"  [error] attempt {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip0", default="data/ip0.json")
    parser.add_argument("--output", default="data/gpt4_dimension_scores.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--data-dir", default="data", help="Path to data dir for domain detection")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI()

    ip0 = json.loads(Path(args.ip0).read_text())
    print(f"Loaded {len(ip0)} topics, {sum(len(v) for v in ip0.values())} total questions")

    output_path = Path(args.output)
    all_scores = json.loads(output_path.read_text()) if output_path.exists() else {}
    print(f"Cache: {sum(len(v) for v in all_scores.values())} already scored")

    api_calls = 0
    for topic, questions in ip0.items():
        domain = detect_domain(topic, args.data_dir)
        cached = all_scores.get(topic, {})
        uncached = [q for q in questions if q not in cached]
        if not uncached: continue

        print(f"  {topic} ({domain}): {len(uncached)} to score")
        for start in range(0, len(uncached), args.batch_size):
            batch = uncached[start:start + args.batch_size]
            results = score_batch(client, args.model, topic, domain, batch)
            if topic not in all_scores: all_scores[topic] = {}
            for li, q in enumerate(batch):
                all_scores[topic][q] = results.get(li, [5, 5, 5, 5, 5])
            api_calls += 1
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(all_scores, indent=2))
            if start + args.batch_size < len(uncached): time.sleep(0.3)

    print(f"\nDone. {sum(len(v) for v in all_scores.values())} questions scored. API calls: {api_calls}")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
