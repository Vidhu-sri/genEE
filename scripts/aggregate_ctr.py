"""Aggregate per-model CTR across topics for the small-models study.

Scans results/ecommerce__full_ctr__*__user* run dirs, reads each topic's
topic_summaries/*.json (initial_ctr, final_ctr, improvement), and prints a
per-model table: mean initial -> mean final -> mean delta (percentage points).
"""
import json
import glob
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Fixed 10-topic comparison set (matches the gpt-4o-mini ceiling run).
COMPARISON_TOPICS = {
    "Area Rugs", "Coaxial Cables", "Cookware Sets", "DVI", "Kitchen Sinks",
    "Lighting Cables", "Measuring Tools & Scales", "Spray Bottles",
    "TV Antennas", "Wall Plates",
}


def summarize(run_dir):
    ts_dir = os.path.join(run_dir, "topic_summaries")
    if not os.path.isdir(ts_dir):
        return None
    inits, finals, deltas = [], [], []
    for f in glob.glob(os.path.join(ts_dir, "*.json")):
        if os.path.basename(f)[:-5] not in COMPARISON_TOPICS:
            continue  # ignore extra topics so all models average over the same 10
        d = json.load(open(f, encoding="utf-8"))
        inits.append(d.get("initial_ctr", 0.0))
        finals.append(d.get("final_ctr", 0.0))
        deltas.append(d.get("improvement", d.get("final_ctr", 0.0) - d.get("initial_ctr", 0.0)))
    if not inits:
        return None
    n = len(inits)
    return (sum(inits) / n * 100, sum(finals) / n * 100,
            sum(deltas) / n * 100, n)


def model_from_dir(name):
    # ecommerce__full_ctr__<model>__<eval>__user[...]
    parts = name.split("__")
    return parts[2] if len(parts) > 2 else name


def main():
    runs = sorted(glob.glob(os.path.join(RESULTS, "ecommerce__full_ctr__*")))
    rows = []
    for r in runs:
        if r.endswith("_bak") or "CORRUPTED" in r:
            continue  # skip quarantined/corrupted runs
        s = summarize(r)
        if s:
            tag = os.path.basename(r)
            verb = "verbalized" in tag
            rows.append((model_from_dir(tag), verb, *s))
    if not rows:
        print("No full_ctr runs found yet.")
        return
    # sort: numeric before verbalized, by final CTR
    rows.sort(key=lambda x: (x[1], -x[4]))
    print(f"\n{'model':28s} {'feedback':11s} {'IP0%':>7s} {'IP14%':>7s} {'delta_pp':>9s} {'topics':>6s}")
    print("-" * 72)
    for model, verb, i0, i14, dl, n in rows:
        print(f"{model:28s} {'verbalized' if verb else 'numeric':11s} "
              f"{i0:7.3f} {i14:7.3f} {dl:+9.3f} {n:6d}")
    print()


if __name__ == "__main__":
    main()
