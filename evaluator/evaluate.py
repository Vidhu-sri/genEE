    #!/usr/bin/env python3
    """
    evaluate.py — Compare trained FiLM vs GPT-4 on HELD-OUT topics.

    Key: only evaluates on topics NOT seen during training.
    The checkpoint stores which topics were used for validation.

    Usage:
    python evaluator/evaluate.py
    python evaluator/evaluate.py --checkpoint evaluator/checkpoints/best.pt
    python evaluator/evaluate.py --eval-on held_out   # only val topics (default)
    python evaluator/evaluate.py --eval-on all        # all topics (for comparison)
    """

    import json, argparse
    from pathlib import Path

    import numpy as np
    import torch
    from scipy.stats import spearmanr

    from model import FiLMEvaluator


    def load_ecom_topics(data_dir="data"):
        p = Path(data_dir) / "topics_ecommerce.json"
        return set(json.loads(p.read_text())) if p.exists() else set()


    def load_model(checkpoint, device="cpu"):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        head_mode = ckpt.get("head_mode", "dimensions")  # fallback for old ckpts

        model = FiLMEvaluator(freeze_encoder=True, head_mode=head_mode).to(device)

        state = model.state_dict()
        for k, v in ckpt["model_state_dict"].items():
            if k in state:
                state[k] = v
        model.load_state_dict(state)
        model.eval()

        val_topics = set(ckpt.get("val_topics", []))
        print(f"Loaded FiLM (head={head_mode}, epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")
        print(f"Held-out val topics: {len(val_topics)}")
        return model, val_topics


    def film_scores(model, questions, alpha, topic, domain, device):
        """
        Score questions with FiLM. Returns (scores, dim_scores).
        
        For dim_scores, we always use one-hot alphas so this works for
        both scalar-head and dimension-head models consistently.
        """
        with torch.no_grad():
            texts = model.format_input(questions, topic, domain)
            q_emb = model.encode(texts, device)

            # Personalized score
            alpha_t = torch.tensor([alpha] * len(questions), dtype=torch.float32, device=device)
            _, scores = model(q_emb, alpha_t)

            # Per-dim via one-hot
            cols = []
            for d in range(5):
                a_oh = torch.zeros(len(questions), 5, device=device)
                a_oh[:, d] = 1.0
                _, s = model(q_emb, a_oh)
                cols.append(s.cpu().numpy())
            dim_scores = np.stack(cols, axis=1)

        return scores.cpu().numpy(), dim_scores


    def film_relevance_vectors(model, questions, topic, domain, device):
        """
        Get per-dimension scores using one-hot alphas.
        This gives "how relevant is q for dimension d" independent of user preferences.
        """
        with torch.no_grad():
            texts = model.format_input(questions, topic, domain)
            q_emb = model.encode(texts, device)
            cols = []
            for d in range(5):
                alpha = torch.zeros(len(questions), 5, device=device)
                alpha[:, d] = 1.0
                _, scores = model(q_emb, alpha)
                cols.append(scores.cpu().numpy())
        return np.stack(cols, axis=1)  # [n_questions, 5]


    def gpt4_scores(gpt4_data, questions, alpha, topic):
        alpha_arr = np.array(alpha, dtype=np.float32)
        topic_data = gpt4_data.get(topic, {})
        dims = []
        for q in questions:
            d = np.array(topic_data.get(q, [5]*5), dtype=np.float32) / 10.0
            dims.append(d)
        dims = np.array(dims)
        scores = (dims * alpha_arr[None, :]).sum(axis=-1)
        return scores, dims


    def evaluate(args):
        device = torch.device(args.device)
        ip0 = json.loads(Path(args.ip0).read_text())
        gpt4_data = json.loads(Path(args.scores).read_text())
        ecom_topics = load_ecom_topics(args.data_dir)

        model, val_topics = load_model(args.checkpoint, device)

        # Decide which topics to evaluate on
        all_topics = list(ip0.keys())
        if args.eval_on == "held_out":
            if not val_topics:
                print("WARNING: checkpoint has no val_topics stored. Evaluating on all.")
                topics = all_topics
            else:
                topics = [t for t in all_topics if t in val_topics]
                print(f"Evaluating on {len(topics)} held-out topics")
        else:
            topics = all_topics
            print(f"Evaluating on ALL {len(topics)} topics (includes training topics)")

        if args.n_topics:
            topics = topics[:args.n_topics]

        rng = np.random.default_rng(42)

        rank_corrs, pearson_corrs = [], []
        maes, rmses = [], []
        top1, top3_sum, dim_agree = 0, 0.0, 0
        total_evals, total_qs = 0, 0

        for ti, topic in enumerate(topics):
            questions = ip0[topic]
            domain = "ecommerce" if topic in ecom_topics else "wikipedia"

            if topic not in gpt4_data:
                continue

            for ai in range(args.n_alphas):
                conc = np.ones(5) * 0.5
                conc[rng.integers(5)] = 4.0
                alpha = rng.dirichlet(conc).tolist()

                fs, fd = film_scores(model, questions, alpha, topic, domain, device)
                gs, gd = gpt4_scores(gpt4_data, questions, alpha, topic)

                maes.append(float(np.mean(np.abs(fs - gs))))
                rmses.append(float(np.sqrt(np.mean((fs - gs) ** 2))))

                # Spearman
                if len(set(np.round(fs, 6))) > 1 and len(set(np.round(gs, 6))) > 1:
                    rho, _ = spearmanr(fs, gs)
                    if not np.isnan(rho): rank_corrs.append(rho)
                    pear = np.corrcoef(fs, gs)[0, 1]
                    if not np.isnan(pear): pearson_corrs.append(pear)

                # Top-1
                if np.argmax(fs) == np.argmax(gs): top1 += 1

                # Top-3
                f3 = set(np.argsort(fs)[-3:])
                g3 = set(np.argsort(gs)[-3:])
                top3_sum += len(f3 & g3) / 3.0

                # Dimension agreement
                for qi in range(len(questions)):
                    if np.argmax(fd[qi]) == np.argmax(gd[qi]):
                        dim_agree += 1
                    total_qs += 1

                total_evals += 1

                # Print first example
                if ti == 0 and ai == 0:
                    print(f"\nExample: {topic} ({domain})")
                    print(f"  Alpha: {[f'{a:.2f}' for a in alpha]}")
                    print(f"  {'Question':<55s} {'FiLM':>7s} {'GPT-4':>7s}")
                    print(f"  {'─'*72}")
                    for i, q in enumerate(questions[:8]):
                        qs = q[:52] + "..." if len(q) > 55 else q
                        print(f"  {qs:<55s} {fs[i]:7.4f} {gs[i]:7.4f}")

        # Report
        eval_type = "HELD-OUT" if args.eval_on == "held_out" and val_topics else "ALL"
        print(f"\n{'='*70}")
        print(f"FiLM vs GPT-4 — {eval_type} topics ({total_evals} evaluations)")
        print(f"{'='*70}")
        if rank_corrs:
            print(f"  Spearman ρ:          {np.mean(rank_corrs):.3f} (±{np.std(rank_corrs):.3f})")
        if pearson_corrs:
            print(f"  Pearson r:           {np.mean(pearson_corrs):.3f} (±{np.std(pearson_corrs):.3f})")
        print(f"  Top-1 agreement:     {top1}/{total_evals} ({100*top1/max(total_evals,1):.1f}%)")
        print(f"  Top-3 overlap:       {top3_sum/max(total_evals,1):.2f} ({100*top3_sum/max(total_evals,1):.1f}%)")
        print(f"  Dim top-dim agree:   {dim_agree}/{total_qs} ({100*dim_agree/max(total_qs,1):.1f}%)")
        print(f"  MAE:                 {np.mean(maes):.4f} (±{np.std(maes):.4f})")
        print(f"  RMSE:                {np.mean(rmses):.4f} (±{np.std(rmses):.4f})")
        print(f"\n  Baselines:")
        # print(f"    Zero-shot MiniLM (Act 2):  0% top-1, 37% dim-agree")
        # print(f"    Random:                    ~5% top-1, ~20% dim-agree")


    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", default="evaluator/checkpoints/best.pt")
        p.add_argument("--ip0", default="data/ip0.json")
        p.add_argument("--scores", default="data/gpt4_dimension_scores.json")
        p.add_argument("--data-dir", default="data")
        p.add_argument("--n-topics", type=int, default=None)
        p.add_argument("--n-alphas", type=int, default=5)
        p.add_argument("--eval-on", default="held_out", choices=["held_out", "all"])
        p.add_argument("--device", default="cpu")
        args = p.parse_args()
        evaluate(args)

    if __name__ == "__main__":
        main()