import json
from src.llm import LLM
import yaml
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os



env_path = Path(__file__).resolve().parent.parent/".env"
load_dotenv(dotenv_path=env_path)


from pathlib import Path

config_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)



data_path = Path(config["data_dir"])
prompts_path = Path(config["prompts_dir"])
results_path = Path(config["results_dir"])

llm = LLM(model="Qwen/Qwen3-32B", temperature=0.7, max_tokens=200)

def load_topics(domain):
    file_path = data_path / f"topics_{domain}.json"
    with open(file_path, "r") as f:
        return json.load(f)

def load_ip(topic, ip_file="ip0.json"):
    file_path = data_path / ip_file
    with open(file_path, "r") as f:
        all_ips = json.load(f)
    return all_ips.get(topic, [])

def eval_ip(ip, topic, persona_name, domain):
   
    with open(data_path / "personas.json", "r") as f:
        personas = json.load(f)

    for p in personas:
        if p["name"] == persona_name:
            persona = p["name"]
            

    
    prompt_file = prompts_path / f"persona_prompt_{domain}.txt"
    with open(prompt_file, "r") as f:
        template = f.read()

    scores = {}
    for q in ip:
        prompt = template.replace("{{TOPIC}}", topic)\
                         .replace("{{PERSONA}}", persona)\
                         .replace("{{QUESTION}}", q)

        score = llm.relevance_score_1to10(prompt)
        scores[q] = score

    return scores




def generate_explore_ip(ip, topic, domain, persona_name):
    prompt_file = prompts_path / "explore_prompt.txt"
    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    questions = "\n".join(f"- {q}" for q in ip)
    prompt = prompt_template.replace("{{QUESTIONS}}", questions)
    return llm.generate_list(prompt, config["k"] // 2)

def generate_exploit_ip(scores, topic, domain, persona_name):
    prompt_file = prompts_path / "exploit_prompt.txt"
    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:config["k"]]
    questions_str = "\n".join(f"Question: {q}\nCTR: {round(score * 100, 1)}%" for q, score in top_k)
    prompt = prompt_template.replace("{{QUESTIONS_AND_CTR}}", questions_str)
    return llm.generate_list(prompt, config["k"] // 2)

def init_ips(domain):
    topics = load_topics(domain)
    
    
    prompt_file = prompts_path / ("initial_prompt_wikipedia.txt" if domain == "wikipedia" else "initial_prompt.txt")

    with open(prompt_file, "r") as f:
        template = f.read()

    all_ips = {}
    for topic in topics:
        prompt = template.replace("{{TOPIC}}", topic).replace("{{N}}", str(config["n"]))
        #print(prompt)
        questions = llm.generate_list(prompt, config["n"])
        #print(questions)
        all_ips[topic] = questions
        print(f'{topic} questions made')

    with open(data_path / "ip0.json", "a") as f:
        json.dump(all_ips, f, indent=2)
