#!/usr/bin/env python3
"""
train.py — Train the FiLM evaluator on GPT-4 dimension scores.

Fixes applied:
  1. Input text includes "Domain: {domain}. Topic: {topic}. Question: {q}"
  2. Train/val split is by TOPIC, not by expanded sample (no leakage)
  3. Alpha sampled via Dirichlet for data augmentation
  Loss = MSE(predicted_score, alpha · GPT_dimension_scores)

Usage:
  python evaluator/train.py
  python evaluator/train.py --epochs 50 --device cuda
  python evaluator/train.py --val-fraction 0.2
"""

import json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import FiLMEvaluator


# ─── Domain detection ───

def load_ecom_topics(data_dir="data"):
    p = Path(data_dir) / "topics_ecommerce.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


# ─── Dataset ───

class DimScoreDataset(Dataset):
    """
    Each sample: (formatted_text, alpha, dim_targets, scalar_target)
    
    formatted_text = "Domain: {domain}. Topic: {topic}. Question: {question}"
    alpha is re-sampled each call (Dirichlet augmentation)
    dim_targets = GPT-4 scores normalized to [0, 1]
    scalar_target = alpha · dim_targets
    """
    def __init__(self, texts, dim_scores, n_alphas_per_q=20):
        """
        texts: list of formatted strings
        dim_scores: np.array [N, 5] in [0, 1]
        """
        self.texts = texts
        self.dim_scores = dim_scores
        self.n_alphas = n_alphas_per_q
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return len(self.texts) * self.n_alphas

    def __getitem__(self, idx):
        q_idx = idx // self.n_alphas
        text = self.texts[q_idx]
        dims = self.dim_scores[q_idx]

        # Random alpha: sometimes peaked, sometimes flat
        conc = self.rng.uniform(0.5, 5.0, size=5)
        conc[self.rng.integers(5)] = self.rng.uniform(3.0, 15.0)
        alpha = self.rng.dirichlet(conc).astype(np.float32)

        target = (alpha * dims).sum()
        return text, alpha, dims, target


def collate(batch):
    texts, alphas, dims, targets = zip(*batch)
    return (
        list(texts),
        torch.tensor(np.array(alphas), dtype=torch.float32),
        torch.tensor(np.array(dims), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32),
    )


def build_datasets(scores_path, data_dir="data", val_fraction=0.2, n_alphas=20):
    """
    Build train/val datasets split BY TOPIC (no question leakage).
    """
    data = json.loads(Path(scores_path).read_text())
    ecom_topics = load_ecom_topics(data_dir)

    # Group by topic
    topics = list(data.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(topics)

    n_val = max(1, int(len(topics) * val_fraction))
    val_topics = set(topics[:n_val])
    train_topics = set(topics[n_val:])

    print(f"Topics: {len(topics)} total | {len(train_topics)} train | {len(val_topics)} val")
    print(f"Val topics: {list(val_topics)[:5]}{'...' if len(val_topics) > 5 else ''}")

    def make_dataset(topic_set):
        texts = []
        dims = []
        for topic in topic_set:
            domain = "ecommerce" if topic in ecom_topics else "wikipedia"
            for question, scores in data[topic].items():
                texts.append(f"Domain: {domain}. Topic: {topic}. Question: {question}")
                dims.append([s / 10.0 for s in scores])
        dims = np.array(dims, dtype=np.float32)
        return DimScoreDataset(texts, dims, n_alphas_per_q=n_alphas)

    train_ds = make_dataset(train_topics)
    val_ds = make_dataset(val_topics)

    print(f"Train: {len(train_ds.texts)} questions × {n_alphas} = {len(train_ds)} samples")
    print(f"Val:   {len(val_ds.texts)} questions × {n_alphas} = {len(val_ds)} samples")

    return train_ds, val_ds, val_topics


# ─── Training ───

def train(args):
    device = torch.device(args.device)
    freeze_encoder = not args.unfreeze_encoder

    train_ds, val_ds, val_topics = build_datasets(
        args.scores, args.data_dir, args.val_fraction, args.n_alphas
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model = FiLMEvaluator(freeze_encoder=freeze_encoder, head_mode=args.head_mode).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_total:,} total | {n_trainable:,} trainable ({100*n_trainable/n_total:.1f}%)")

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        if freeze_encoder:
            model.encoder.eval()
        tl, nb = 0.0, 0

        for texts, alphas, dim_gt, target_gt in train_dl:
            alphas = alphas.to(device)
            target_gt = target_gt.to(device)

            q_emb = model.encode(texts, device)
            pred_dims, pred_score = model(q_emb, alphas)

            # Scalar loss only: forces model to USE alpha via FiLM.
            # Dimension-level loss removed because it would push
            # predicted_dims toward GPT's dims regardless of alpha,
            # making FiLM bypass-able.
            loss = F.mse_loss(pred_score, target_gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            tl += loss.item()
            nb += 1

        scheduler.step()

        # ── Validate ──
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for texts, alphas, dim_gt, target_gt in val_dl:
                alphas = alphas.to(device)
                target_gt = target_gt.to(device)

                q_emb = model.encode(texts, device)
                pred_dims, pred_score = model(q_emb, alphas)

                loss = F.mse_loss(pred_score, target_gt)
                vl += loss.item()
                vn += 1

        avg_tl = tl / max(nb, 1)
        avg_vl = vl / max(vn, 1)

        print(f"  epoch {epoch+1:3d}/{args.epochs} | "
              f"train={avg_tl:.5f} | val={avg_vl:.5f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        if avg_vl < best_val:
            best_val = avg_vl
            model_state_dict = model.state_dict()
            if freeze_encoder:
                model_state_dict = {
                    k: v for k, v in model_state_dict.items()
                    if not k.startswith("encoder.")
                }
            torch.save({
                "model_state_dict": model_state_dict,
                "head_mode": args.head_mode,
                "freeze_encoder": freeze_encoder,
                "epoch": epoch,
                "val_loss": best_val,
                "val_topics": list(val_topics),
            }, save_path)
            print(f"    ✓ saved (val={best_val:.5f})")

    print(f"\nDone. Best val loss: {best_val:.5f} → {save_path}")
    print(f"Held-out val topics: {list(val_topics)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="data/gpt4_dimension_scores.json")
    p.add_argument("--output", default="evaluator/checkpoints/best.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-alphas", type=int, default=20)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--head-mode", default="scalar", choices=["scalar", "dimensions"],
                   help="scalar: r_ij = f(q,t,alpha) directly (Image 1 formulation, recommended). "
                        "dimensions: predict 5 dims, then alpha · dims (legacy, weaker).")
    p.add_argument("--unfreeze-encoder", action="store_true",
                   help="Fine-tune the MiniLM encoder instead of keeping it frozen.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()
