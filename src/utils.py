import os, re, json, math, random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import numpy as np

from .llm import LLM

# ─── Config ───
config_path = Path(__file__).resolve().parent.parent / "config.yaml"
config = yaml.safe_load(open(config_path, "r"))

data_path = Path(config["data_dir"])
prompts_path = Path(config["prompts_dir"])
results_path = Path(config["results_dir"])

# ─── LLM (for generation only now) ───
llm = LLM(
    qs_model=config.get("qs_model", "gpt-4-turbo"),
    gen_model=config.get("gen_model", "gpt-3.5-turbo"),
    qs_temperature=config.get("qs_temperature", 0.0),
    gen_temperature=config.get("gen_temperature", 0.8),
    batch_size=config.get("batch_size", 20),
    rate_limit_rps=config.get("rate_limit_rps", 5),
)

# ─── MiniLM Evaluator (replaces LLM-based scoring) ───
import sys
evaluator_dir = str(Path(__file__).resolve().parent.parent / "evaluator")
if evaluator_dir not in sys.path:
    sys.path.insert(0, evaluator_dir)

from evaluator import make_evaluator, to_original_score_scale



# _evaluator = make_evaluator("film", checkpoint = 'evaluator/checkpoints/best.pt)
_evaluator = make_evaluator(
    config.get("evaluator_backend", "minilm"),
    checkpoint=config.get("film_checkpoint", "evaluator/checkpoints/best.pt"),
    device=config.get("evaluator_device", "cpu"),
)

# ─── Persona → Alpha mapping ───
# Each persona maps to a peaked alpha on its dimension.
# Dimension order must match evaluator.py's DOMAIN_DIMS:
#   ecommerce: [Price, Quality, Brand, Features, Ethical]
#   wikipedia: [Discussion, History, Event, Person, Location]

PERSONA_TO_DIM = {
    # ecommerce (indices into ECOM_DIM_DESCRIPTIONS)
    "Price": 0,
    "Quality": 1,
    "Brand Reputation": 2,
    "Features & Functionality": 3,
    "Ethical Considerations": 4,
    # wikipedia (indices into WIKI_DIM_DESCRIPTIONS)
    "Discussion-Focused": 0,
    "History-Focused": 1,
    "Event-Focused": 2,
    "Person-Focused": 3,
    "Location-Focused": 4,
}

def persona_to_alpha(persona_name: str, concentration: float = 10.0) -> List[float]:
    """
    Fixed peaked alpha for a persona.
    concentration controls how peaked: higher = more focused on that dimension.
    """
    dim = PERSONA_TO_DIM[persona_name]
    alpha = [1.0] * 5
    alpha[dim] = concentration
    total = sum(alpha)
    return [a / total for a in alpha]


def sample_user_alpha(persona_name: str, rng: np.random.Generator,
                      concentration: float = 5.0) -> List[float]:
    """
    Sample a user-level alpha from Dirichlet centered on the persona's dimension.
    Each call gives a different user within this persona cohort.
    
    concentration: how tightly users cluster around the persona center.
      - 5.0 = moderate spread (realistic)
      - 20.0 = tight cluster
      - 1.0 = very diverse within cohort
    """
    dim = PERSONA_TO_DIM[persona_name]
    # Dirichlet concentration vector: peaked on the persona's dimension
    conc = np.ones(5) * 1.0
    conc[dim] = concentration
    return rng.dirichlet(conc).tolist()


# ─── Template rendering ───
import regex
PLACEHOLDER_PAT = regex.compile(r"\{\{(\w+)\}\}|<(\w+)>")

def render_template(template: str, mapping: dict) -> str:
    def _repl(m):
        k = m.group(1) or m.group(2)
        return str(mapping.get(k, m.group(0)))
    return PLACEHOLDER_PAT.sub(_repl, template)


# ─── IO ───
def load_topics(domain):
    return json.load(open(data_path / f"topics_{domain}.json", "r"))

def load_ip(topic, ip_file="ip0.json"):
    all_ips = json.load(open(data_path / ip_file, "r"))
    return all_ips.get(topic, [])

def init_ips(domain):
    topics = load_topics(domain)
    prompt_file = "initial_prompt_wikipedia.txt" if domain == "wikipedia" else "initial_prompt.txt"
    template = (prompts_path / prompt_file).read_text()
    all_ips = {}
    N0 = int(config.get("n", 20))
    for topic in topics:
        prompt = render_template(template, {"TOPIC": topic, "N": str(N0)})
        questions = llm.generate_list(prompt, N0)
        all_ips[topic] = questions
        print(f"{topic} questions made")
    with open(data_path / "ip0.json", "w") as f:
        json.dump(all_ips, f, indent=2)


# ─── Scoring (MiniLM, replaces LLM API calls) ───

def _persona_map():
    objs = json.load(open(data_path / "personas.json", "r"))
    return {p["name"]: p.get("prompt", "") for p in objs}


def eval_ip_all_personas(ip, topic, selected_personas, domain):
    """
    Score all questions for all personas using MiniLM evaluator.
    
    Returns: dict[persona_name -> dict[question -> score(float 1..10)]]
    
    This is now SYNC (no API calls). The MiniLM evaluator runs locally.
    """
    persona_scores = {}
    
    # Get relevance vectors once for all questions (shared across personas)
    # Shape: [n_questions, 5], values in [0, 1]
    r = _evaluator.relevance_vectors(ip,topic, domain)
    
    for persona_name in selected_personas:
        alpha = persona_to_alpha(persona_name)
        alpha_arr = np.array(alpha, dtype=np.float32)
        
        # score = alpha · r(q), gives [0, 1] range
        # raw_scores = (r * alpha_arr[None, :]).sum(axis=-1)  # [n_questions]
        raw_scores = _evaluator.score(ip, alpha, topic, domain)  
        
        # Scale to 1-10 to match what simulate_ctr expects
        scaled = to_original_score_scale(raw_scores)
        
        persona_scores[persona_name] = {
            q: float(s) for q, s in zip(ip, scaled)
        }
    
    return persona_scores


def eval_ip_all_personas_user_level(
    ip, topic, selected_personas, domain,
    n_users_per_persona: int = 10,
    rng: np.random.Generator = None,
):
    """
    Score questions for user-level personas (Dirichlet-sampled alphas).
    
    Instead of 5 personas, creates n_users_per_persona users per persona type,
    each with a different alpha sampled from Dirichlet.
    
    Returns: dict[user_id -> dict[question -> score(float 1..10)]]
    
    user_id format: "Price_user_3", "Quality_user_7", etc.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    
    
    user_scores = {}
    for persona_name in selected_personas:
        for uid in range(n_users_per_persona):
            user_id = f"{persona_name}_user_{uid}"
            alpha = sample_user_alpha(persona_name, rng)
            alpha_arr = np.array(alpha, dtype=np.float32)
            
            raw_scores = _evaluator.score(ip, alpha, topic, domain)  
            scaled = to_original_score_scale(raw_scores)
            
            user_scores[user_id] = {
                q: float(s) for q, s in zip(ip, scaled)
            }
    
    return user_scores


# ─── Simulator (softmax CTR) ───

def softmax_click_probs(relevances, RS=11.0, T=1.5):
    exps = [math.exp(r / T) for r in relevances]
    exp_none = math.exp(RS / T)
    z = sum(exps) + exp_none
    probs = [e / z for e in exps]
    p_none = exp_none / z
    return probs, p_none


def simulate_ctr(ip, persona_scores, K=3, S=5000, RS=11.0, T=1.5, seed=None):
    """
    Works with both cohort-level and user-level persona_scores.
    persona_scores: {persona_or_user_id: {question: score_1_to_10}}
    """
    if not ip:
        return {}
    if seed is not None:
        random.seed(seed)
    ctr_counts = Counter({q: 0 for q in ip})
    personas = list(persona_scores.keys())
    n_ip = len(ip)
    score_lookup = {
        p: {q: float(persona_scores[p].get(q, 1.0)) for q in ip}
        for p in personas
    }

    for _ in range(S):
        pj = random.choice(personas)
        shown = random.sample(ip, K) if K <= n_ip else [random.choice(ip) for _ in range(K)]
        rels = [score_lookup[pj].get(q, 1.0) for q in shown]
        probs, p_none = softmax_click_probs(rels, RS=RS, T=T)
        u = random.random()
        cdf = 0.0
        clicked = None
        for q, p_prob in zip(shown, probs):
            cdf += p_prob
            if u < cdf:
                clicked = q
                break
        if clicked:
            ctr_counts[clicked] += 1
    return {q: ctr_counts[q] / float(S) for q in ip}


# ─── Generators (still use LLM) ───

def generate_explore_ip(ip, topic, domain):
    prompt_template = (prompts_path / f"explore_prompt_{domain}.txt").read_text()
    N = int(config.get("explore_n", 5))
    prompt = render_template(prompt_template, {"TITLE": topic, "CATEGORY": topic, "N": str(N)})
    return llm.generate_list(prompt, N)


def generate_exploit_ip(ctrs_or_scores, topic, domain):
    prompt_template = (prompts_path / f"exploit_prompt_{domain}.txt").read_text()
    N = int(config.get("exploit_n", 5))
    top_k = sorted(ctrs_or_scores.items(), key=lambda x: x[1], reverse=True)[:max(1, N)]
    questions_str = "\n".join(f"Question: {q}\nCTR: {round(p * 100, 1)}%" for q, p in top_k)
    prompt = render_template(prompt_template, {
        "TITLE": topic, "CATEGORY": topic, "N": str(N), "QUESTIONS_AND_CTR": questions_str
    })
    return llm.generate_list(prompt, N)


# ─── Pool merge ───

def merge_pool(old_ip, keep, explore, exploit, N_pool):
    new = []
    seen = set()
    for q in (keep + exploit + explore + old_ip):
        if q not in seen:
            seen.add(q)
            new.append(q)
        if len(new) >= N_pool:
            break
    return new