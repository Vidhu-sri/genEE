"""
Architecture:
  "Domain: {domain}. Topic: {topic}. Question: {question}"
    → frozen MiniLM encoder → 384-dim embedding
  Alpha vector → FiLM generator → gamma, beta
  Modulated = embedding * (1 + gamma) + beta

  Main mode: scalar
    Modulated → Scalar head → personalized relevance score r_ij

  Legacy mode: dimensions
    Modulated → Dimension head → 5 scores
    Final score = alpha · dimension_scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from transformers import AutoModel, AutoTokenizer

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HIDDEN_DIM = 384
N_DIMS = 5


class FiLMGenerator(nn.Module):
    """Generate scale (gamma) and shift (beta) from alpha vector."""
    def __init__(self, alpha_dim=N_DIMS, feature_dim=HIDDEN_DIM, hidden=128):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(alpha_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(alpha_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, alpha):
        return self.gamma_net(alpha), self.beta_net(alpha)


class DimensionHead(nn.Module):
    """Predict per-dimension relevance scores from FiLM-modulated embedding."""
    def __init__(self, in_dim=HIDDEN_DIM, hidden=128, n_dims=N_DIMS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_dims),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)  # [B, 5] in [0, 1]


class ScalarHead(nn.Module):
    """
    Predict a single relevance score directly from FiLM-modulated embedding.
    This matches the formulation: r_ij = f_theta(q_i, t, alpha_j) directly.
    Alpha enters ONLY via FiLM, not via final dot product.
    """
    def __init__(self, in_dim=HIDDEN_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B] in [0, 1]


class FiLMEvaluator(nn.Module):
    """
    Two head modes:
      head_mode="scalar"     (default, matches Image 1 formulation)
        → predicts scalar directly: r_ij = f_theta(q_i, t, alpha_j)
        → alpha enters ONLY through FiLM
        → cleanest, most defensible
      
      head_mode="dimensions" (older formulation)
        → predicts 5 dim scores, then score = alpha · dims
        → alpha enters through both FiLM AND final dot product
        → useful for interpretability (per-dimension scores)
    """
    def __init__(self, freeze_encoder=True, model_name=MINILM_NAME, head_mode="scalar"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.head_mode = head_mode

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.film = FiLMGenerator(alpha_dim=N_DIMS, feature_dim=HIDDEN_DIM)
        if head_mode == "scalar":
            self.head = ScalarHead(in_dim=HIDDEN_DIM)
        elif head_mode == "dimensions":
            self.head = DimensionHead(in_dim=HIDDEN_DIM, n_dims=N_DIMS)
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

    def encode(self, texts: List[str], device, no_grad: bool = None) -> torch.Tensor:
        """
        Encode texts with MiniLM -> [B, 384].

        If encoder parameters are frozen, this automatically runs without gradients.
        If encoder is unfrozen, gradients are allowed during training.
        """
        if no_grad is None:
            no_grad = not any(p.requires_grad for p in self.encoder.parameters())

        def _encode_inner():
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = self.encoder(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

        if no_grad:
            with torch.no_grad():
                return _encode_inner()
        return _encode_inner()

    @staticmethod
    def format_input(questions: List[str], topic: str, domain: str) -> List[str]:
        """Format input text with topic+domain context for encoding."""
        return [f"Domain: {domain}. Topic: {topic}. Question: {q}" for q in questions]

    def forward(self, q_emb, alpha):
        """
        q_emb: [B, 384] pre-encoded embeddings
        alpha: [B, 5]
        
        Returns: (dim_scores or None, score)
          - scalar head:    (None, score [B])
          - dimension head: (dim_scores [B, 5], score [B])
        """
        gamma, beta = self.film(alpha)
        modulated = q_emb * (1 + gamma) + beta

        if self.head_mode == "scalar":
            score = self.head(modulated)  # [B]
            return None, score
        else:
            dim_scores = self.head(modulated)  # [B, 5]
            score = (alpha * dim_scores).sum(dim=-1)  # [B]
            return dim_scores, score

    def predict(self, questions: List[str], alpha: List[float],
                topic: str, domain: str, device=None):
        """Convenience method for inference."""
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            texts = self.format_input(questions, topic, domain)
            q_emb = self.encode(texts, device)
            alpha_t = torch.tensor(
                [alpha] * len(questions), dtype=torch.float32, device=device
            )
            return self(q_emb, alpha_t)
