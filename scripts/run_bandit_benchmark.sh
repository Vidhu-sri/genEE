#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULTS_DIR="results/bandit_10x10"
mkdir -p "$RESULTS_DIR"

if [[ -x "venv/bin/python" ]]; then
    PYTHON="venv/bin/python"
elif [[ -x "venv/Scripts/python.exe" ]]; then
    PYTHON="venv/Scripts/python.exe"
else
    PYTHON="${PYTHON:-python}"
fi

"$PYTHON" - <<'PY'
from pathlib import Path

import yaml

expected = {
    "generator_model": "gpt-4o-mini",
    "evaluator_backend": "film",
    "film_checkpoint": "evaluator/checkpoints/best_dimensions_mpnet.pt",
    "iterations": 15,
    "sim_K": 3,
    "sim_S": 5000,
    "sim_T": 1.5,
    "sim_RS": 11,
    "users_per_persona": 10,
    "initial_ip_size": 5,
    "evaluator_device": "cpu",
    "bandit_ucb_c": 0.002,
}

config = yaml.safe_load(Path("config.yaml").read_text())
mismatches = [
    f"{key}: expected {value!r}, found {config.get(key)!r}"
    for key, value in expected.items()
    if config.get(key) != value
]
if mismatches:
    raise SystemExit("Benchmark config mismatch:\n  " + "\n  ".join(mismatches))

checkpoint = Path(config["film_checkpoint"])
if not checkpoint.is_file():
    raise SystemExit(f"Missing evaluator checkpoint: {checkpoint}")

print("Benchmark config verified.")
PY

mapfile -d '' -t TOPICS_WIKI < <("$PYTHON" - <<'PY'
import json
import sys

for topic in json.load(open("data/topics_wikipedia.json", encoding="utf-8"))[:10]:
    sys.stdout.write(topic + "\0")
PY
)

mapfile -d '' -t TOPICS_ECOM < <("$PYTHON" - <<'PY'
import json
import sys

for topic in json.load(open("data/topics_ecommerce.json", encoding="utf-8"))[:10]:
    sys.stdout.write(topic + "\0")
PY
)

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
    shift
fi

if (($#)); then
    METHODS=("$@")
else
    METHODS=(bandit_ucb bandit_epsilon bandit_thompson)
fi

benchmark_complete() {
    local domain="$1"
    local method="$2"
    local run_tag="$3"

    "$PYTHON" - "$domain" "$method" "$run_tag" <<'PY'
import json
import sys
from pathlib import Path

domain, method, run_tag = sys.argv[1:]
topics = json.loads(Path(f"data/topics_{domain}.json").read_text())[:10]
run_dir = Path(
    f"results/{domain}__{method}__gpt-4o-mini__"
    f"film_best_dimensions_mpnet__user{run_tag}"
)

complete = all(
    (run_dir / "logs" / f"{topic}_iter_14.json").is_file()
    and (run_dir / "pool_snapshots" / f"{topic}_iter_14.json").is_file()
    and (run_dir / "topic_summaries" / f"{topic}.json").is_file()
    for topic in topics
)
raise SystemExit(0 if complete else 1)
PY
}

run_benchmark() {
    local domain="$1"
    local method="$2"
    local run_tag=""
    local -a tag_args=()
    if [[ "$method" == "bandit_ucb" ]]; then
        run_tag="__ucb_c002"
        tag_args=(--run-tag ucb_c002)
    fi
    local log_path="$RESULTS_DIR/${domain}_${method}${run_tag}.log"
    shift 2

    if benchmark_complete "$domain" "$method" "$run_tag"; then
        echo "SKIP: $domain $method is already complete (10 topics x 15 iterations)."
        return
    fi

    if [[ "$CHECK_ONLY" == true ]]; then
        echo "PENDING: $domain $method"
        return
    fi

    {
        printf '\n=== %s %s | %s ===\n' "${domain^^}" "$method" "$(date --iso-8601=seconds)"
        "$PYTHON" -m src.runner \
            --domain "$domain" \
            --method "$method" \
            --topics "$@" \
            --user-level \
            --resume \
            "${tag_args[@]}"
    } 2>&1 | tee -a "$log_path"
}

for method in "${METHODS[@]}"; do
    case "$method" in
        bandit_ucb|bandit_epsilon|bandit_thompson) ;;
        *)
            echo "Unsupported benchmark method: $method" >&2
            exit 2
            ;;
    esac

    run_benchmark wikipedia "$method" "${TOPICS_WIKI[@]}"
    run_benchmark ecommerce "$method" "${TOPICS_ECOM[@]}"
done

if [[ "$CHECK_ONLY" == true ]]; then
    echo "Benchmark status check complete."
else
    echo "Bandit 10x10 benchmark complete."
fi
