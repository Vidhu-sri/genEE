


import os
from dotenv import load_dotenv
import openai
from openai import OpenAI




load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()



class LLM:
    def __init__(self, model="gpt-4.1", temperature=0.7, max_tokens=512):

        self.model = model
        self.temperature = temperature
    


    def generate(self, prompt):
    
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            print(f"[LLM ERROR]: {e}")
            return ""
        
        except Exception:
            return ""
    
    def relevance_score_1to10(self, prompt: str) -> float:
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        try:
            
            for line in content.splitlines():
                if line.lower().startswith("score:"):
                    val = line.split(":")[1].strip()
                    return min(max(float(val), 1.0), 10.0)  
        except:
            pass
        return 5.0  
    def generate_list(self, prompt, k=10):
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        text = response.choices[0].message.content.strip()
        lines = [line.strip("- ").strip() for line in text.split("\n") if line.strip()]
        return lines[:k]

        
        





