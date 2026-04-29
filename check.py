
# check integrity of dataset

import json

REQUIRED_KEYS = {
    "topic": str,
    "domain": str,
    "alpha": list,
    "user_profile_text": str,
    "q_a": str,
    "q_b": str,
    "winner": str,
    "slice": str,
    "difficulty": str,
}

def check(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                raise ValueError(f"Blank line at {i}")

            obj = json.loads(line)

            # keys
            if set(obj.keys()) != set(REQUIRED_KEYS.keys()):
                raise ValueError(f"Bad keys at line {i}: {obj.keys()}")

            # types
            for k, t in REQUIRED_KEYS.items():
                if not isinstance(obj[k], t):
                    raise ValueError(f"Bad type for {k} at line {i}")

            # alpha sanity
            if len(obj["alpha"]) != 5:
                raise ValueError(f"alpha len != 5 at line {i}")
            if any(a < 0 for a in obj["alpha"]):
                raise ValueError(f"negative alpha at line {i}")
            if not (0.99 <= sum(obj["alpha"]) <= 1.01):
                raise ValueError(f"alpha not normalized at line {i}")

            # enums
            if obj["winner"] not in ("A", "B"):
                raise ValueError(f"bad winner at line {i}")
            if obj["slice"] not in ("explore", "exploit"):
                raise ValueError(f"bad slice at line {i}")
            if obj["difficulty"] not in ("easy", "medium", "hard"):
                raise ValueError(f"bad difficulty at line {i}")

    print("✓ Schema OK")

check(r'C:\Users\Lenovo\Desktop\research\genEE\data\pairs.jsonl')
