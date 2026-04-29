import os, re, json, time, hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import asyncio
import os, re, json, time, hashlib, random
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv


async def safe_api_call(acall, retries=5, base_delay=1.0):
    """Retry with exponential backoff on rate limits or transient failures."""
    for attempt in range(retries):
        try:
            return await acall()
        except (RateLimitError, APIError, APIConnectionError) as e:
            wait = base_delay * (2 ** attempt) * (1.0 + random.random())
            print(f"[warn] API throttled ({type(e).__name__}), retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)
    print("[error] giving up after retries; returning None")
    return None


load_dotenv()
client = OpenAI()
aclient = AsyncOpenAI()




load_dotenv()
client = OpenAI()
aclient = AsyncOpenAI()

class DiskCache:
    """Tiny JSON cache: key=str -> value (score)."""
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self.mem = json.loads(self.path.read_text())
            except Exception:
                self.mem = {}
        else:
            self.mem = {}

    def get(self, k, default=None):
        return self.mem.get(k, default)

    def set(self, k, v):
        self.mem[k] = v

    def persist(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.mem, indent=2))
        tmp.replace(self.path)

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

class LLM:
    def __init__(self,
                 qs_model=os.getenv("QS_MODEL", "gpt-4-turbo"),
                 gen_model=os.getenv("GEN_MODEL", "gpt-3.5-turbo"),
                 qs_temperature=0.0,
                 gen_temperature=0.8,
                 max_tokens_score=16,
                 max_tokens_gen=512,
                 cache_path=".cache/qs_cache.json",
                 batch_size=20,
                 rate_limit_rps=5):
        self.qs_model = qs_model
        self.gen_model = gen_model
        self.qs_temperature = float(qs_temperature)
        self.gen_temperature = float(gen_temperature)
        self.max_tokens_score = int(max_tokens_score)
        self.max_tokens_gen = int(max_tokens_gen)
        self.batch_size = int(batch_size)
        self.rate_limit_interval = 1.0 / max(1, int(rate_limit_rps))
        self.cache = DiskCache(cache_path)

    # ---------- GENERATION ----------
    def generate_list(self, prompt: str, k: int = 10):
        r = client.chat.completions.create(
            model=self.gen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.gen_temperature,
            max_tokens=self.max_tokens_gen,
        )
        txt = r.choices[0].message.content
        # robust line extraction
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        # strip common prefixes
        cleaned = []
        for ln in lines:
            ln = re.sub(r"^(New Question:\s*)", "", ln, flags=re.I).strip()
            ln = re.sub(r"^(Question\s*\d+\s*:\s*)", "", ln, flags=re.I).strip()
            ln = ln.strip("-• ").strip()
            if ln:
                cleaned.append(ln)
        return cleaned[:k]

    # ---------- SCORING (single) ----------
    def relevance_score_1to10(self, prompt: str) -> float:
        """JSON-returning scorer with cache; numeric 1..10."""
        key = _hash("QS|" + self.qs_model + "|" + prompt)
        hit = self.cache.get(key)
        if hit is not None:
            return float(hit)
        sys = 'Return ONLY a JSON like {"score":7} where score is integer 1..10.'
        r = client.chat.completions.create(
            model=self.qs_model,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":prompt}],
            temperature=self.qs_temperature,
            max_tokens=self.max_tokens_score,
        )
        txt = r.choices[0].message.content.strip()
        try:
            obj = json.loads(txt)
            s = int(obj.get("score", 5))
        except Exception:
            m = re.search(r'(?:^|\D)(10|[1-9])(?:\D|$)', txt)
            s = int(m.group(1)) if m else 5
        s = max(1, min(10, s))
        self.cache.set(key, s)
        self.cache.persist()
        return float(s)

    # ---------- SCORING (batched async) ----------
    async def score_many(self, prompts: list[str]) -> list[float]:
        """Return 1..10 floats in the same order; cached + chunked + safe retries."""
        results: list[float] = [None] * len(prompts)
        to_fetch = []
        for i, p in enumerate(prompts):
            key = _hash("QS|" + self.qs_model + "|" + p)
            hit = self.cache.get(key)
            if hit is not None:
                results[i] = float(hit)
            else:
                to_fetch.append((i, p, key))
        if not to_fetch:
            return results

        sys = 'Return ONLY a JSON like {"score":7} where score is integer 1..10.'
        for start in range(0, len(to_fetch), self.batch_size):
            chunk = to_fetch[start:start + self.batch_size]

            async def _one(idx, prompt, key):
                async def call():
                    return await aclient.chat.completions.create(
                        model=self.qs_model,
                        messages=[{"role": "system", "content": sys},
                                {"role": "user", "content": prompt}],
                        temperature=self.qs_temperature,
                        max_tokens=self.max_tokens_score,
                    )
                resp = await safe_api_call(call)
                if resp is None:
                    results[idx] = 5.0
                    return
                txt = resp.choices[0].message.content.strip()
                try:
                    obj = json.loads(txt)
                    s = int(obj.get("score", 5))
                except Exception:
                    m = re.search(r'(?:^|\D)(10|[1-9])(?:\D|$)', txt)
                    s = int(m.group(1)) if m else 5
                s = max(1, min(10, s))
                self.cache.set(key, s)
                results[idx] = float(s)

            await asyncio.gather(*[_one(i, p, k) for (i, p, k) in chunk])
            self.cache.persist()
            await asyncio.sleep(self.rate_limit_interval)
        return results

