A generative recommender system for question generation. 

lowest CTR questions are dropped, and replaced in part with CTR exploited questions and in part with 
high temperature exploratory questions. 

The exploration vs exploitation is handled by bandits, 
user represented by user level embeddings, 

custom evaluator with dirichlet distribution to simulate user vectors as of now. 


Reference:
- https://arxiv.org/abs/2406.05255 
- https://arxiv.org/abs/2401.04858  
- https://www.sciencedirect.com/science/article/abs/pii/S0957417422001543


![App Diagram](results\bandit_10x10\bandit_vs_fixed_ecommerce.png)
![App Diagram](results\bandit_10x10\bandit_vs_fixed_wikipedia.png)
