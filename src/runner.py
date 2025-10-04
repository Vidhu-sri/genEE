import os
import json
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path

from src.utils import (
    load_topics,
    load_ip,
    eval_ip,
    generate_explore_ip,
    generate_exploit_ip,
    init_ips,
)


config_path = Path(__file__).resolve().parent.parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


data_path = Path(config["data_dir"])
prompts_path = Path(config["prompts_dir"])
results_path = Path(config["results_dir"])
k = config["k"]
iterations = config["iterations"]


with open(data_path / "personas.json", "r") as file:
    personas = json.load(file)
    personas = [p['name'] for p in personas]

def func(domain: str):
    topics = load_topics(domain)

    #print('topics obtained')


    #for topic in tqdm(topics, desc=f"[{domain}] Processing topics"):
    for topic in topics:

        print(f'working on {topic}')
        for persona_name in personas:
            ip = load_ip(topic)

            print(f'ip retrieved: {ip}')

            for i in tqdm(range(iterations), desc=f"[{domain}] {topic} - {persona_name}", leave=False):
                scores = eval_ip(ip, topic, persona_name, domain)
                print(scores)
                high_ctr = sorted(scores.keys(), key=lambda x: scores[x])[-k:]

                explore = generate_explore_ip(ip, topic, domain, persona_name)
                exploit = generate_exploit_ip(scores, topic, domain, persona_name)

                ip = high_ctr + explore + exploit
                avg_ctr = sum(scores.values()) / len(scores)

                os.makedirs(results_path / "pool_snapshots", exist_ok=True)
                os.makedirs(results_path / "logs", exist_ok=True)

                snapshot_file = results_path / "pool_snapshots" / f"{topic}_{persona_name}_ip_iter_{i}.json"
                with open(snapshot_file, "w") as f:
                    json.dump(ip, f, indent=2)

                ctr_file = results_path / "ctr_results.csv"
                write_header = not ctr_file.exists()
                with open(ctr_file, "a") as f:
                    if write_header:
                        f.write("iteration,avg_ctr,topic,persona\n")
                    f.write(f"{i},{avg_ctr},{topic},{persona_name}\n")


print("starting up")
#init_ips("wikipedia")
#init_ips("ecommerce")
func("wikipedia")
func("ecommerce")
