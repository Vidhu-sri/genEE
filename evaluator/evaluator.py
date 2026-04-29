"""
evaluator.py — Evaluator factory for the GenEE loop.

Three backends:
  1. "minilm"  — Zero-shot cosine similarity (Act 2 baseline)
  2. "film"    — Trained FiLM model (Act 3, needs checkpoint)
  3. "gpt4"    — GPT-4 API scoring (original paper)

All backends use consistent signature:
  ev.score(questions, alpha, topic, domain) → [float]
  ev.relevance_vectors(questions, topic, domain) → [n, 5]
"""

import json, os, time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ─── Dimensions ───
WIKI_DIM_DESCRIPTIONS = [
    "Discussion, debate, definition, explanation, overview, concepts, ideas, meaning, philosophy, theory",
    "History, historical, origins, timeline, founded, ancient, century, era, development, evolution",
    "Events, battles, wars, revolutions, incidents, occurrences, crises, turning points, milestones",
    "People, persons, biography, thinker, philosopher, leader, figure, life, born, known for, role",
    "Locations, places, geography, city, country, region, where, practiced, spread, modern, today",
]
ECOM_DIM_DESCRIPTIONS = [
    "Price, cost, affordable, budget, expensive, cheap, value, deal, discount, pricing",
    "Quality, durable, reliable, materials, craftsmanship, build, sturdy, premium, well-made",
    "Brand, reputation, popular, trusted, reviews, ratings, well-known, recommended, top-rated",
    "Features, functionality, specifications, capabilities, performance, technology, design, versatile",
    "Ethical, sustainable, eco-friendly, fair trade, organic, environmental, responsible, green",
]
WIKI_DIM_NAMES = ["Discussion", "History", "Event", "Person", "Location"]
ECOM_DIM_NAMES = ["Price", "Quality", "Brand Reputation", "Features", "Ethical"]
DOMAIN_DIMS = {"wikipedia": WIKI_DIM_DESCRIPTIONS, "ecommerce": ECOM_DIM_DESCRIPTIONS}
DOMAIN_DIM_NAMES = {"wikipedia": WIKI_DIM_NAMES, "ecommerce": ECOM_DIM_NAMES}



def normalize_alpha(alpha: List[float]) -> np.ndarray:
    alpha = np.array(alpha, dtype=np.float32)

    if alpha.shape != (5,):
        raise ValueError(f"alpha must have shape (5,), got {alpha.shape}")

    if np.any(alpha < 0):
        raise ValueError(f"alpha values must be non-negative, got {alpha}")
    
    total = float(alpha.sum())

    if total <= 0:
        raise ValueError(f"alpha sum must be positive, got {total}")

    return alpha / total


to_original_score_scale = lambda scores_01: 1 + 9.0*np.array(scores_01, dtype = np.float32)
    


class Evaluator(ABC):
    @abstractmethod
    def score(self, questions: List[str], alpha: List[float],
              topic: str, domain: str) -> List[float]: ...

    @abstractmethod
    def relevance_vectors(self, questions: List[str],
                          topic: str, domain: str) -> np.ndarray: ...


# ═══════════════════════════════════════════════
# Backend 1: Zero-shot MiniLM (Act 2 — baseline)
# ═══════════════════════════════════════════════

def _make_minilm_evaluator(model_name="sentence-transformers/all-MiniLM-L6-v2",
                            tau=10.0, device="cpu"):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    class _MiniLM(Evaluator):
        def __init__(self):
            self.device = torch.device(device)
            self.tau = tau
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self._dim_embs = {}
            for dom, descs in DOMAIN_DIMS.items():
                embs = self._encode(descs)
                self._dim_embs[dom] = F.normalize(embs, dim=-1)

        @torch.no_grad()
        def _encode(self, texts):
            enc = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=128, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

        @torch.no_grad()
        def relevance_vectors(self, questions, topic, domain):
            texts = [f"Domain: {domain}. Topic: {topic}. Question: {q}" for q in questions]
            q_embs = F.normalize(self._encode(texts), dim=-1)
            sims = q_embs @ self._dim_embs[domain].T
            r = F.softmax(sims * self.tau, dim=-1).cpu().numpy()
            if len(questions) > 1:
                mn, mx = r.min(axis=0, keepdims=True), r.max(axis=0, keepdims=True)
                r = (r - mn) / np.maximum(mx - mn, 1e-8)
            return r

        def score(self, questions, alpha, topic, domain):
            alpha = normalize_alpha(alpha)
            r = self.relevance_vectors(questions, topic, domain)
            return (r * np.array(alpha, dtype=np.float32)[None, :]).sum(axis=-1).tolist()

    return _MiniLM()


# ═══════════════════════════════════════════════
# Backend 2: Trained FiLM (Act 3)
# ═══════════════════════════════════════════════

def _make_film_evaluator(checkpoint, device="cpu"):
    import torch
    import sys
    # Ensure model.py is importable
    eval_dir = str(Path(__file__).resolve().parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from model import FiLMEvaluator as FiLMModel

    dev = torch.device(device)

    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    head_mode = ckpt.get("head_mode", "dimensions")  # fallback for old ckpts

    model = FiLMModel(freeze_encoder=True, head_mode=head_mode).to(dev)

    state = model.state_dict()
    for k, v in ckpt["model_state_dict"].items():
        if k in state:
            state[k] = v
    model.load_state_dict(state)
    model.eval()
    print(f"[FiLM] Loaded {checkpoint} (head={head_mode}, epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")

    class _FiLM(Evaluator):
        def __init__(self):
            self.model = model
            self.device = dev

        def relevance_vectors(self, questions, topic, domain):
            """
            Per-dimension scores using one-hot alphas.
            This correctly isolates each dimension's contribution.
            """
            with torch.no_grad():
                texts = self.model.format_input(questions, topic, domain)
                q_emb = self.model.encode(texts, self.device)
                cols = []
                for d in range(5):
                    alpha = torch.zeros(len(questions), 5, device=self.device)
                    alpha[:, d] = 1.0
                    _, scores = self.model(q_emb, alpha)
                    cols.append(scores.cpu().numpy())
            return np.stack(cols, axis=1)  # [n, 5]

        def score(self, questions, alpha, topic, domain):

            alpha= normalize_alpha(alpha)
            with torch.no_grad():
                texts = self.model.format_input(questions, topic, domain)
                q_emb = self.model.encode(texts, self.device)
                alpha_t = torch.tensor(
                    [alpha] * len(questions), dtype=torch.float32, device=self.device
                )
                _, scores = self.model(q_emb, alpha_t)
            return scores.cpu().numpy().tolist()

    return _FiLM()


# ═══════════════════════════════════════════════
# Backend 3: GPT-4 (original paper)
# ═══════════════════════════════════════════════

class GPT4Evaluator(Evaluator):
    def __init__(self, model="gpt-4o-mini", cache_path="./data/gpt4_score_cache.json"):
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and Path(cache_path).exists():
            self._cache = json.loads(Path(cache_path).read_text())

    def _ck(self, t, d, q): return f"{d}|{t}|{q}"

    def _save(self):
        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.cache_path).write_text(json.dumps(self._cache, indent=2))

    def _call(self, topic, domain, questions):
        dn, dd = DOMAIN_DIM_NAMES[domain], DOMAIN_DIMS[domain]
        db = "\n".join(f"  Dim {i} ({dn[i]}): {dd[i]}" for i in range(5))
        qb = "\n".join(f"  idx={i}: {q}" for i, q in enumerate(questions))
        prompt = f"""Rate each question for topic "{topic}" ({domain}) on 5 dimensions (1-10).
{db}
Output ONLY JSON array: [{{"idx":0,"scores":[1,2,3,4,5]}}, ...]
Questions:
{qb}"""
        for att in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                txt = r.choices[0].message.content.strip()
                txt = txt.replace("```json", "").replace("```", "").strip()
                res = {}
                for obj in json.loads(txt):
                    idx, sc = obj.get("idx"), obj.get("scores", [])
                    if idx is not None and len(sc) == 5:
                        res[int(idx)] = [max(1, min(10, int(s))) for s in sc]
                if res: return res
            except Exception as e:
                time.sleep(2 * (att + 1))
        return {}

    def relevance_vectors(self, questions, topic, domain):
        uncached, results = [], {}
        for i, q in enumerate(questions):
            ck = self._ck(topic, domain, q)
            if ck in self._cache: results[i] = self._cache[ck]
            else: uncached.append(i)
        if uncached:
            for s in range(0, len(uncached), 20):
                bi = uncached[s:s + 20]
                bq = [questions[i] for i in bi]
                gpt = self._call(topic, domain, bq)
                for li, gi in enumerate(bi):
                    sc = gpt.get(li, [5] * 5)
                    results[gi] = sc
                    self._cache[self._ck(topic, domain, questions[gi])] = sc
                self._save()
        r = np.zeros((len(questions), 5), dtype=np.float32)
        for i in range(len(questions)):
            r[i] = np.array(results.get(i, [5] * 5), dtype=np.float32) / 10.0
        return r

    def score(self, questions, alpha, topic, domain):
        alpha = normalize_alpha(alpha)
        r = self.relevance_vectors(questions, topic, domain)
        return (r * np.array(alpha, dtype=np.float32)[None, :]).sum(axis=-1).tolist()


# ═══════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════

def make_evaluator(backend: str = "minilm", **kwargs) -> Evaluator:
    """
    make_evaluator("minilm")                              → Act 2, zero-shot
    make_evaluator("film", checkpoint="path/best.pt")     → Act 3, trained FiLM
    make_evaluator("gpt4")                                → original paper
    """
    if backend == "minilm":
        return _make_minilm_evaluator(
            model_name=kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            tau=kwargs.get("tau", 10.0),
            device=kwargs.get("device", "cpu"),
        )
    elif backend == "film":
        return _make_film_evaluator(
            checkpoint=kwargs.get("checkpoint", "evaluator/checkpoints/best.pt"),
            device=kwargs.get("device", "cpu"),
        )
    elif backend == "gpt4":
        return GPT4Evaluator(
            model=kwargs.get("model", "gpt-4o-mini"),
            cache_path=kwargs.get("cache_path", "./data/gpt4_score_cache.json"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")