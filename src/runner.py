"""
runner.py — GenEE experiment runner with MiniLM evaluator.

Everything is SYNC now — no API calls for scoring.
Only LLM API calls are for question generation (explore/exploit).

Usage:
  python src/runner.py --domain wikipedia --method explore_exploit
  python src/runner.py --domain ecommerce --method all
  python src/runner.py --domain wikipedia --method explore_exploit --user-level
  python src/runner.py --domain ecommerce --method all --resume
  python src/runner.py --domain wikipedia --method all --dry-run
"""

import os, json, yaml, argparse, time, random
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

import numpy as np

config_path = Path(__file__).resolve().parent.parent / "config.yaml"
config = yaml.safe_load(open(config_path, "r"))
load_dotenv()

data_path = Path(config["data_dir"])
results_base = Path(config["results_dir"])

from src.utils import (
    load_topics, load_ip,
    eval_ip_all_personas,             # sync, MiniLM-based
    eval_ip_all_personas_user_level,  # sync, Dirichlet user-level
    simulate_ctr,
    generate_explore_ip,   # sync, LLM API
    generate_exploit_ip,   # sync, LLM API
    merge_pool,
)

# ─── Personas ───
with open(data_path / "personas.json") as f:
    _all = json.load(f)
ECOM_P = [p["name"] for p in _all if p.get("domain") == "ecommerce"]
WIKI_P = [p["name"] for p in _all if p.get("domain") == "wikipedia"]

def personas_for(d): return ECOM_P if d == "ecommerce" else WIKI_P

# ─── Methods ───
METHODS = ["no_drop", "random_ctr", "partial_ctr", "full_ctr", "explore_exploit"]

def _drop_worst(ip, ctrs, keep_k):
    n_drop = max(0, len(ip) - keep_k)
    if n_drop == 0: return [], list(ip)
    ranked = sorted(ctrs.items(), key=lambda x: x[1])
    dropped = [q for q, _ in ranked[:n_drop]]
    keep = [q for q in ip if q not in set(dropped)]
    return dropped, keep

def _dedup(qs):
    seen = set(); out = []
    for q in qs:
        k = q.strip().lower()
        if k not in seen: seen.add(k); out.append(q)
    return out

def _score(ip, topic, domain, personas, cfg, user_level, rng):
    """Score questions. Returns persona_scores dict."""
    if user_level:
        return eval_ip_all_personas_user_level(
            ip, topic, personas, domain,
            n_users_per_persona=cfg.get("users_per_persona", 10),
            rng=rng,
        )
    else:
        return eval_ip_all_personas(ip, topic, personas, domain)

# ── Method implementations ──
# All return: (new_ip, ctrs, pscores, dropped, explore, exploit, t_eval, t_gen, t_sim)

def m_no_drop(ip, topic, domain, personas, iteration, cfg, user_level, rng):
    t0 = time.time()
    ps = _score(ip, topic, domain, personas, cfg, user_level, rng)
    te = time.time() - t0

    t0 = time.time()
    ctrs = simulate_ctr(ip, ps, K=cfg["sim_K"], S=cfg["sim_S"],
                        RS=cfg["sim_RS"], T=cfg["sim_T"], seed=cfg["seed"]+iteration)
    ts = time.time() - t0

    t0 = time.time()
    explore = generate_explore_ip(ip, topic, domain)
    tg = time.time() - t0

    new_ip = _dedup(list(ip) + explore)
    return new_ip, ctrs, ps, [], explore, [], te, tg, ts

def m_random_ctr(ip, topic, domain, personas, iteration, cfg, user_level, rng):
    t0 = time.time()
    ps = _score(ip, topic, domain, personas, cfg, user_level, rng)
    te = time.time() - t0

    t0 = time.time()
    real_ctrs = simulate_ctr(ip, ps, K=cfg["sim_K"], S=cfg["sim_S"],
                             RS=cfg["sim_RS"], T=cfg["sim_T"], seed=cfg["seed"]+iteration)
    ts = time.time() - t0

    n_drop = max(0, len(ip) - cfg["keep_k"])
    fake = {q: rng.uniform(0, 0.15) for q in ip}
    ranked = sorted(fake.items(), key=lambda x: x[1])
    dropped = [q for q, _ in ranked[:n_drop]]
    keep = [q for q in ip if q not in set(dropped)]

    t0 = time.time()
    explore = generate_explore_ip(ip, topic, domain)
    tg = time.time() - t0

    new_ip = merge_pool(ip, keep, explore, [], cfg["pool_size"])
    return new_ip, real_ctrs, ps, dropped, explore, [], te, tg, ts

def m_partial_ctr(ip, topic, domain, personas, iteration, cfg, user_level, rng):
    t0 = time.time()
    ps = _score(ip, topic, domain, personas, cfg, user_level, rng)
    te = time.time() - t0

    t0 = time.time()
    ctrs = simulate_ctr(ip, ps, K=cfg["sim_K"], S=cfg["sim_S"],
                        RS=cfg["sim_RS"], T=cfg["sim_T"], seed=cfg["seed"]+iteration)
    ts = time.time() - t0

    dropped, keep = _drop_worst(ip, ctrs, cfg["keep_k"])

    t0 = time.time()
    explore = generate_explore_ip(ip, topic, domain)
    tg = time.time() - t0

    new_ip = merge_pool(ip, keep, explore, [], cfg["pool_size"])
    return new_ip, ctrs, ps, dropped, explore, [], te, tg, ts

def m_full_ctr(ip, topic, domain, personas, iteration, cfg, user_level, rng):
    t0 = time.time()
    ps = _score(ip, topic, domain, personas, cfg, user_level, rng)
    te = time.time() - t0

    t0 = time.time()
    ctrs = simulate_ctr(ip, ps, K=cfg["sim_K"], S=cfg["sim_S"],
                        RS=cfg["sim_RS"], T=cfg["sim_T"], seed=cfg["seed"]+iteration)
    ts = time.time() - t0

    dropped, keep = _drop_worst(ip, ctrs, cfg["keep_k"])

    t0 = time.time()
    exploit = generate_exploit_ip(ctrs, topic, domain)
    tg = time.time() - t0

    new_ip = merge_pool(ip, keep, [], exploit, cfg["pool_size"])
    return new_ip, ctrs, ps, dropped, [], exploit, te, tg, ts

def m_explore_exploit(ip, topic, domain, personas, iteration, cfg, user_level, rng):
    t0 = time.time()
    ps = _score(ip, topic, domain, personas, cfg, user_level, rng)
    te = time.time() - t0

    t0 = time.time()
    ctrs = simulate_ctr(ip, ps, K=cfg["sim_K"], S=cfg["sim_S"],
                        RS=cfg["sim_RS"], T=cfg["sim_T"], seed=cfg["seed"]+iteration)
    ts = time.time() - t0

    dropped, keep = _drop_worst(ip, ctrs, cfg["keep_k"])

    t0 = time.time()
    explore = generate_explore_ip(ip, topic, domain)
    exploit = generate_exploit_ip(ctrs, topic, domain)
    tg = time.time() - t0

    new_ip = merge_pool(ip, keep, explore, exploit, cfg["pool_size"])
    return new_ip, ctrs, ps, dropped, explore, exploit, te, tg, ts

METHOD_FNS = {
    "no_drop": m_no_drop, "random_ctr": m_random_ctr,
    "partial_ctr": m_partial_ctr, "full_ctr": m_full_ctr,
    "explore_exploit": m_explore_exploit,
}

# ─── Logging ───
CSV_COLS = [
    "iteration", "topic", "domain", "method", "generator_model", "evaluator",
    "user_level", "avg_ctr", "best_ctr", "worst_ctr", "median_ctr", "std_ctr",
    "pool_size", "n_dropped", "n_added_explore", "n_added_exploit",
    "eval_time_sec", "gen_time_sec", "sim_time_sec", "total_time_sec",
]

def build_log(run_id, domain, topic, method, gen_model, eval_label, user_level,
              iteration, ip, ctrs, pscores, personas,
              dropped, explore, exploit, te, tg, ts, tt, cfg):
    ctr_vals = [ctrs.get(q, 0.0) for q in ip]
    dropped_s, explore_s, exploit_s = set(dropped or []), set(explore or []), set(exploit or [])

    questions = []
    # For user-level, pscores has many users. Aggregate to persona-level for logging.
    # For cohort-level, pscores keys are persona names directly.
    all_scorer_keys = list(pscores.keys())

    for q in ip:
        ps = {}
        for p in all_scorer_keys:
            if q in pscores[p]:
                ps[p] = pscores[p][q]
        status = "kept"
        if q in dropped_s: status = "dropped"
        elif q in explore_s: status = "new_explore"
        elif q in exploit_s: status = "new_exploit"
        questions.append({"text": q, "ctr": ctrs.get(q, 0.0), "persona_scores": ps, "status": status})

    # Aggregate persona averages
    persona_avgs = {}
    for p in all_scorer_keys:
        v = [pscores[p].get(q, 0) for q in ip]
        persona_avgs[p] = float(np.mean(v)) if v else 0.0

    return {
        "run_id": run_id, "domain": domain, "topic": topic,
        "method": method, "generator_model": gen_model, "evaluator": eval_label,
        "user_level": user_level,
        "iteration": iteration, "timestamp": datetime.now().isoformat(),
        "pool_size": len(ip), "questions": questions,
        "avg_ctr": float(np.mean(ctr_vals)) if ctr_vals else 0.0,
        "best_ctr": float(np.max(ctr_vals)) if ctr_vals else 0.0,
        "worst_ctr": float(np.min(ctr_vals)) if ctr_vals else 0.0,
        "median_ctr": float(np.median(ctr_vals)) if ctr_vals else 0.0,
        "std_ctr": float(np.std(ctr_vals)) if ctr_vals else 0.0,
        "persona_avg_scores": persona_avgs,
        "n_dropped": len(dropped or []), "n_added_explore": len(explore or []),
        "n_added_exploit": len(exploit or []),
        "dropped_questions": dropped or [],
        "added_explore_questions": explore or [],
        "added_exploit_questions": exploit or [],
        "eval_time_sec": round(te, 3), "gen_time_sec": round(tg, 3),
        "sim_time_sec": round(ts, 3), "total_time_sec": round(tt, 3),
        "sim_K": cfg["sim_K"], "sim_S": cfg["sim_S"],
        "sim_T": cfg["sim_T"], "sim_RS": cfg["sim_RS"], "seed": cfg["seed"],
    }

def append_csv(path, log, header):
    with open(path, "a") as f:
        if header: f.write(",".join(CSV_COLS) + "\n")
        f.write(",".join(str(log.get(c, "")) for c in CSV_COLS) + "\n")

# ─── Resume ───
def last_iter(rd, topic):
    ld = rd / "logs"
    if not ld.exists(): return -1
    mx = -1
    for f in ld.glob(f"{topic}_iter_*.json"):
        try: mx = max(mx, int(f.stem.split("_iter_")[1]))
        except: pass
    return mx

def load_snap(rd, topic, it):
    p = rd / "pool_snapshots" / f"{topic}_iter_{it}.json"
    return json.loads(p.read_text()) if p.exists() else None

# ─── Main loop (fully sync) ───
def run_experiment(domain, method, user_level=False, resume=False, topics_subset=None):
    gen_model = config.get("gen_model", "gpt-3.5-turbo")
    eval_label = "minilm"  # always MiniLM now
    ul_tag = "user" if user_level else "cohort"

    run_id = f"{domain}__{method}__{gen_model}__{eval_label}__{ul_tag}"
    rd = results_base / run_id
    for sub in ["logs", "pool_snapshots", "topic_summaries"]:
        (rd / sub).mkdir(parents=True, exist_ok=True)

    topics = load_topics(domain)
    if topics_subset:
        topics = [t for t in topics if t in topics_subset]
    personas = personas_for(domain)

    cfg = {
        "pool_size": int(config.get("pool_size", 20)),
        "keep_k": int(config.get("keep_k", 10)),
        "sim_K": int(config.get("sim_K", 3)),
        "sim_S": int(config.get("sim_S", 5000)),
        "sim_T": float(config.get("sim_T", 1.5)),
        "sim_RS": float(config.get("sim_RS", 11)),
        "seed": int(config.get("seed", 42)),
        "iterations": int(config.get("iterations", 15)),
        "users_per_persona": int(config.get("users_per_persona", 10)),
    }

    (rd / "run_config.json").write_text(json.dumps({
        "run_id": run_id, "domain": domain, "method": method,
        "generator_model": gen_model, "evaluator": eval_label,
        "user_level": user_level, "personas": personas, "topics": topics,
        **cfg, "started_at": datetime.now().isoformat(),
    }, indent=2))

    csv_path = rd / "results.csv"
    fn = METHOD_FNS[method]
    rng = np.random.default_rng(cfg["seed"])
    cum = {"eval": 0.0, "gen": 0.0, "sim": 0.0, "total": 0.0}

    for topic in topics:
        print(f"\n{'='*60}\n[{run_id}] {topic}\n{'='*60}")

        start = 0
        ip = load_ip(topic)
        if resume:
            li = last_iter(rd, topic)
            if li >= 0:
                s = load_snap(rd, topic, li)
                if s: ip, start = s, li + 1; print(f"  Resuming from iter {start}")
        if start >= cfg["iterations"]:
            print("  Done, skipping."); continue

        ctr_hist = []
        for i in tqdm(range(start, cfg["iterations"]), desc=f"  {topic}", leave=False):
            t0 = time.time()
            new_ip, ctrs, pscores, dropped, expl, expt, te, tg, ts = \
                fn(ip, topic, domain, personas, i, cfg, user_level, rng)
            tt = time.time() - t0

            log = build_log(
                run_id, domain, topic, method, gen_model, eval_label, user_level,
                i, ip, ctrs, pscores, personas,
                dropped, expl, expt, te, tg, ts, tt, cfg,
            )
            ctr_hist.append(log["avg_ctr"])

            (rd / "logs" / f"{topic}_iter_{i}.json").write_text(json.dumps(log, indent=2))
            (rd / "pool_snapshots" / f"{topic}_iter_{i}.json").write_text(json.dumps(new_ip, indent=2))

            hdr = not csv_path.exists() or csv_path.stat().st_size == 0
            append_csv(csv_path, log, hdr)

            ip = new_ip
            cum["eval"] += te; cum["gen"] += tg; cum["sim"] += ts; cum["total"] += tt

            tqdm.write(
                f"    i={i:2d} avg_ctr={log['avg_ctr']:.4f} best={log['best_ctr']:.4f} "
                f"pool={log['pool_size']} eval={te:.1f}s gen={tg:.1f}s"
            )

        (rd / "topic_summaries" / f"{topic}.json").write_text(json.dumps({
            "topic": topic, "method": method, "ctr_trajectory": ctr_hist,
            "initial_ctr": ctr_hist[0] if ctr_hist else 0,
            "final_ctr": ctr_hist[-1] if ctr_hist else 0,
            "improvement": ctr_hist[-1] - ctr_hist[0] if len(ctr_hist) > 1 else 0,
            "final_pool": ip,
        }, indent=2))

    (rd / "run_summary.json").write_text(json.dumps({
        "run_id": run_id, "completed_at": datetime.now().isoformat(),
        "cumulative_time": cum,
    }, indent=2))

    print(f"\nDONE: {run_id} | eval={cum['eval']:.0f}s gen={cum['gen']:.0f}s total={cum['total']:.0f}s")

# ─── CLI ───
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["wikipedia", "ecommerce"])
    p.add_argument("--method", required=True, choices=METHODS + ["all"])
    p.add_argument("--user-level", action="store_true",
                   help="Use Dirichlet-sampled per-user alphas instead of fixed cohort alphas")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--topics", nargs="*", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        personas = personas_for(args.domain)
        topics = load_topics(args.domain)
        if args.topics: topics = [t for t in topics if t in args.topics]
        methods = METHODS if args.method == "all" else [args.method]
        iters = config.get("iterations", 15)
        pool = config.get("pool_size", 20)
        ups = config.get("users_per_persona", 10) if args.user_level else 1
        n_gen = len(topics) * iters * len(methods)
        print(f"=== DRY RUN ===")
        print(f"Domain:      {args.domain}")
        print(f"Methods:     {methods}")
        print(f"Generator:   {config.get('gen_model')}")
        print(f"Evaluator:   MiniLM (local, no API)")
        print(f"User-level:  {args.user_level} ({ups} users/persona)" if args.user_level else f"User-level:  False (cohort)")
        print(f"Topics:      {len(topics)}")
        print(f"Personas:    {personas}")
        print(f"Iterations:  {iters}")
        print(f"Pool size:   {pool}")
        print(f"Scoring:     FREE (MiniLM local)")
        print(f"Gen LLM calls: ~{n_gen:,}")
        return

    methods = METHODS if args.method == "all" else [args.method]
    for method in methods:
        print(f"\n{'#'*60}")
        print(f"# {method} | {args.domain} | gen={config.get('gen_model')} | eval=minilm | {'user-level' if args.user_level else 'cohort'}")
        print(f"{'#'*60}")
        run_experiment(args.domain, method, args.user_level, args.resume, args.topics)

if __name__ == "__main__":
    main()