"""
Cross-Model DIF Robustness — Colab GPU script.

Runs the same DIF v1 pipeline on an alternate LM to test whether the
kernel shape is a property of language statistics (model-invariant) or
specific to Mistral-7B.

Models tested:
  - Baseline: mistralai/Mistral-7B-v0.1 (existing results)
  - Alt 1: meta-llama/Llama-3.1-8B-Instruct (similar size, different training)
  - Alt 2: EleutherAI/pythia-2.8b (smaller, different arch, no auth needed)
  - Alt 3: microsoft/phi-2 (2.7B, compact)

Strategy: Run on a SUBSET of ECSC or PERSUADE (e.g., 50 docs) to keep
compute manageable. Compare DIF scalars and kernel shape to Mistral baseline.

Requires:
  - Google Colab with T4/A100 GPU
  - !pip install transformers bitsandbytes accelerate pandas matplotlib scipy

Usage:
  1. Upload this script + data to Colab
  2. Set MODELS, INPUT_FILE, etc. below
  3. Run all cells
"""

# ======================================================================
# CONFIG
# ======================================================================

# Models to compare (first is reference, rest are alternates)
MODELS = [
    {
        "name": "mistral-7b",
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "use_4bit": True,
        "needs_auth": False,
    },
    {
        "name": "pythia-2.8b",
        "hf_id": "EleutherAI/pythia-2.8b",
        "use_4bit": False,  # small enough for fp16
        "needs_auth": False,
    },
    # Uncomment for additional models:
    # {
    #     "name": "llama-3.1-8b",
    #     "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
    #     "use_4bit": True,
    #     "needs_auth": True,  # needs HF_TOKEN
    # },
    # {
    #     "name": "phi-2",
    #     "hf_id": "microsoft/phi-2",
    #     "use_4bit": False,
    #     "needs_auth": False,
    # },
]

INPUT_FILE = "ecsc_texts.jsonl"  # or .csv
CORPUS_NAME = "ecsc"
TEXT_COL = "text"
DOC_ID_COL = "doc_id"
MAX_DOCS = 50  # subset size for tractability
OUTPUT_DIR = "results_cross_model_dif"

K_GRID = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

TARGET_LENGTH = 32
N_WINDOWS = 10  # fewer for speed
MIN_CONTEXT = 64  # ECSC has shorter docs
END_BUFFER = 32
MIN_TOKENS = MIN_CONTEXT + TARGET_LENGTH + END_BUFFER

SEED = 42
N_BOOT = 2000

# ======================================================================
# IMPORTS
# ======================================================================

import gc
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ======================================================================
# MODEL MANAGEMENT
# ======================================================================

def load_model(model_cfg):
    """Load a single model, return (model, tokenizer)."""
    hf_id = model_cfg["hf_id"]
    print(f"\nLoading {model_cfg['name']} ({hf_id})...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_cfg["use_4bit"] and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_config, device_map="auto",
        )
    elif torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float16, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_id)
        model = model.to("cpu")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loaded. Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tokenizer, device


def unload_model(model):
    """Free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ======================================================================
# GAIN CURVE COMPUTATION
# ======================================================================

@torch.no_grad()
def compute_nll_at_lag(model, token_ids, target_start, target_end, k, device):
    """Compute mean NLL on target region with k tokens of context."""
    if k == 0:
        chunk = token_ids[target_start:target_end]
        if len(chunk) < 2:
            return float("nan")
        input_ids = torch.tensor([chunk], device=device)
        logits = model(input_ids).logits[0]
        nlls = []
        for i in range(len(chunk) - 1):
            lp = torch.log_softmax(logits[i], dim=-1)
            nlls.append(-lp[chunk[i + 1]].item())
        return np.mean(nlls) if nlls else float("nan")
    else:
        ctx_start = max(0, target_start - k)
        actual_k = target_start - ctx_start
        chunk = token_ids[ctx_start:target_end]
        input_ids = torch.tensor([chunk], device=device)
        logits = model(input_ids).logits[0]
        nlls = []
        for i in range(actual_k, len(chunk) - 1):
            lp = torch.log_softmax(logits[i], dim=-1)
            nlls.append(-lp[chunk[i + 1]].item())
        return np.mean(nlls) if nlls else float("nan")


def compute_gain_curves(model, tokenizer, texts, doc_ids, device, rng):
    """Compute gain curves for all documents with one model."""
    results = []
    for idx, (text, doc_id) in enumerate(zip(texts, doc_ids)):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n = len(tokens)
        print(f"    [{idx + 1}/{len(texts)}] {doc_id} ({n} tok)...", end=" ", flush=True)

        if n < MIN_TOKENS:
            print("SKIP")
            continue

        # Sample windows
        min_start = MIN_CONTEXT
        max_start = n - TARGET_LENGTH - END_BUFFER
        if max_start <= min_start:
            print("SKIP")
            continue

        positions = np.linspace(min_start, max_start, N_WINDOWS, dtype=int)
        jitter = rng.integers(-5, 6, size=N_WINDOWS)
        positions = np.clip(positions + jitter, min_start, max_start)

        nll_by_k = {k: [] for k in K_GRID}
        for ts in positions:
            te = ts + TARGET_LENGTH
            for k in K_GRID:
                nll = compute_nll_at_lag(model, tokens, ts, te, k, device)
                if not math.isnan(nll):
                    nll_by_k[k].append(nll)

        nll_0 = np.mean(nll_by_k[0]) if nll_by_k[0] else float("nan")
        rec = {"doc_id": doc_id, "n_tokens": n}
        for k in K_GRID:
            mean_nll = np.mean(nll_by_k[k]) if nll_by_k[k] else float("nan")
            rec[f"mean_gain_{k}"] = nll_0 - mean_nll if not math.isnan(mean_nll) else float("nan")
        results.append(rec)
        print(f"gain_128={rec['mean_gain_128']:.3f}")

    return pd.DataFrame(results)


# ======================================================================
# DIF COMPUTATION
# ======================================================================

N_BINS = len(K_GRID) - 1

def compute_dif_for_df(df):
    """Compute kernel + DIF scalars from gain curves."""
    k_grid = np.array(K_GRID)
    rows = []
    for _, row in df.iterrows():
        g128 = row["mean_gain_128"]
        if g128 <= 0 or math.isnan(g128):
            continue
        gains = {k: row[f"mean_gain_{k}"] for k in K_GRID}
        shares = np.array([
            max(0, (gains[k_grid[i + 1]] - gains[k_grid[i]]) / g128)
            for i in range(N_BINS)
        ])
        s = shares.sum()
        if s > 0:
            shares /= s

        # Scalars
        def lq(q):
            cum = 0.0
            for i in range(N_BINS):
                prev = cum
                cum += shares[i]
                if cum >= q:
                    frac = (q - prev) / (shares[i] + 1e-15)
                    lo = k_grid[i] + 1 if k_grid[i] > 0 else 1
                    hi = k_grid[i + 1]
                    return lo + frac * (hi - lo)
            return float(k_grid[-1])

        def mean_lag():
            total = 0.0
            for i in range(N_BINS):
                lo = k_grid[i] + 1 if k_grid[i] > 0 else 1
                hi = k_grid[i + 1]
                total += shares[i] * (lo + hi) / 2
            return total

        rec = {"doc_id": row["doc_id"],
               "L50": lq(0.5), "L80": lq(0.8), "L90": lq(0.9),
               "tail_33_128": sum(shares[i] for i in range(N_BINS) if k_grid[i] >= 32),
               "tail_65_128": sum(shares[i] for i in range(N_BINS) if k_grid[i] >= 64),
               "mean_lag": mean_lag(), "gain128": g128}
        for i in range(N_BINS):
            rec[f"p_{i}"] = shares[i]
        rows.append(rec)

    return pd.DataFrame(rows)


def bootstrap_summary(doc_df, rng):
    """Bootstrap CIs on DIF scalars."""
    metrics = ["L50", "L80", "L90", "tail_33_128", "tail_65_128", "mean_lag", "gain128"]
    rows = []
    for m in metrics:
        vals = doc_df[m].dropna().values
        if len(vals) < 3:
            continue
        means = np.array([
            vals[rng.integers(0, len(vals), len(vals))].mean()
            for _ in range(N_BOOT)
        ])
        rows.append({"metric": m, "mean": vals.mean(),
                      "ci_lo": np.percentile(means, 2.5),
                      "ci_hi": np.percentile(means, 97.5)})
    return pd.DataFrame(rows)


# ======================================================================
# CROSS-MODEL COMPARISON
# ======================================================================

def compare_models(all_results, out_dir):
    """Compare DIF across models: table + kernel overlay + distances."""
    out_dir = Path(out_dir)

    model_names = list(all_results.keys())
    kernels = {}
    summaries = {}

    rng = np.random.default_rng(SEED)
    for name, gain_df in all_results.items():
        doc_df = compute_dif_for_df(gain_df)
        summary = bootstrap_summary(doc_df, rng)
        kernel = np.array([doc_df[f"p_{i}"].mean() for i in range(N_BINS)])
        kernels[name] = kernel
        summaries[name] = summary
        doc_df.to_csv(out_dir / f"{name}_dif_doc_level.csv", index=False)
        summary.to_csv(out_dir / f"{name}_dif_summary.csv", index=False)

    # Comparison table
    comp_rows = []
    for name in model_names:
        s = summaries[name]
        rec = {"model": name}
        for _, r in s.iterrows():
            rec[f"{r['metric']}_mean"] = r["mean"]
            rec[f"{r['metric']}_ci_lo"] = r["ci_lo"]
            rec[f"{r['metric']}_ci_hi"] = r["ci_hi"]
        comp_rows.append(rec)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_dir / "cross_model_comparison.csv", index=False)

    print("\n=== Cross-Model DIF Comparison ===")
    for name in model_names:
        s = summaries[name]
        print(f"\n{name}:")
        for _, r in s.iterrows():
            m = r["metric"]
            v = r["mean"]
            if "tail" in m:
                print(f"  {m:14s}: {v:.1%}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
            else:
                print(f"  {m:14s}: {v:.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    # Kernel distances
    dist_rows = []
    for i, n1 in enumerate(model_names):
        for j, n2 in enumerate(model_names):
            k1, k2 = kernels[n1], kernels[n2]
            l2 = np.linalg.norm(k1 - k2)
            k1s = np.maximum(k1, 0); k1s /= k1s.sum()
            k2s = np.maximum(k2, 0); k2s /= k2s.sum()
            js = jensenshannon(k1s, k2s)
            dist_rows.append({"model_a": n1, "model_b": n2, "L2": l2, "JensenShannon": js})
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(out_dir / "cross_model_kernel_distances.csv", index=False)

    print("\n=== Kernel Distances ===")
    for _, r in dist_df.iterrows():
        if r["model_a"] != r["model_b"]:
            print(f"  {r['model_a']} vs {r['model_b']}: L2={r['L2']:.4f}, JS={r['JensenShannon']:.4f}")

    # Overlay plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]
    x = np.arange(N_BINS)

    k_grid = np.array(K_GRID)
    bin_labels = []
    for i in range(N_BINS):
        lo = k_grid[i] + 1 if k_grid[i] > 0 else 1
        hi = k_grid[i + 1]
        bin_labels.append(f"lag {lo}-{hi}")

    # Bar chart
    ax = axes[0]
    n_models = len(model_names)
    width = 0.7 / n_models
    for idx, name in enumerate(model_names):
        offset = (idx - n_models / 2 + 0.5) * width
        ax.bar(x + offset, kernels[name], width, color=colors[idx % len(colors)],
               alpha=0.8, label=name, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of gain(128)")
    ax.set_title("Cross-Model Influence Kernel Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Cumulative
    ax2 = axes[1]
    bin_rights = k_grid[1:]
    for idx, name in enumerate(model_names):
        cum = np.cumsum(kernels[name])
        ax2.plot(bin_rights, cum, "o-", color=colors[idx % len(colors)],
                 label=name, linewidth=2, markersize=5)

    for q, lbl in [(0.5, "L50"), (0.8, "L80"), (0.9, "L90")]:
        ax2.axhline(q, color="gray", linestyle=":", alpha=0.5)
        ax2.annotate(lbl, xy=(1.5, q + 0.01), fontsize=9, color="gray")

    ax2.set_xlabel("Lag (tokens)")
    ax2.set_ylabel("Cumulative share")
    ax2.set_title("Cumulative Kernel: Cross-Model")
    ax2.set_xscale("log")
    ax2.set_xlim(1, 150)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "cross_model_kernel_overlay.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved overlay: {out_dir / 'cross_model_kernel_overlay.png'}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Load data
    path = Path(INPUT_FILE)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in open(path)]
        data = pd.DataFrame(records)
    else:
        data = pd.read_csv(path)

    # Subsample
    if len(data) > MAX_DOCS:
        data = data.sample(n=MAX_DOCS, random_state=SEED).reset_index(drop=True)
    print(f"Using {len(data)} documents")

    texts = data[TEXT_COL].tolist()
    doc_ids = data[DOC_ID_COL].tolist()

    # Run each model
    all_results = {}
    for model_cfg in MODELS:
        name = model_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")

        model, tokenizer, device = load_model(model_cfg)
        gain_df = compute_gain_curves(model, tokenizer, texts, doc_ids, device, rng)
        gain_df.to_csv(out_dir / f"{name}_gain_curves.csv", index=False)
        all_results[name] = gain_df

        print(f"  Computed {len(gain_df)} gain curves")
        unload_model(model)

    # Compare
    compare_models(all_results, out_dir)
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
