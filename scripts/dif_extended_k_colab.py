"""
Extended DIF: Push k-grid to 512 tokens — Colab GPU script.

Extends the gain curve from k_max=128 to k_max=512, enabling:
  - New bins: lag 129-192, 193-256, 257-384, 385-512
  - New tail shares: 129-512, 257-512
  - New DIF scalars: L95, L99

Requires:
  - Google Colab with T4/A100 GPU
  - !pip install transformers bitsandbytes accelerate pandas matplotlib

Input: JSONL file with doc_id + text fields (or CSV with text column)
Output: extended_gain_curves.csv, extended_dif_results.csv, extended_kernel.png

Usage:
  1. Upload this script + data to Colab
  2. Set INPUT_FILE, CORPUS_NAME, TEXT_COL, DOC_ID_COL below
  3. Run all cells
"""

# ======================================================================
# CONFIG — edit these for your corpus
# ======================================================================

INPUT_FILE = "persuade_texts.jsonl"  # or .csv
CORPUS_NAME = "persuade"
TEXT_COL = "text"
DOC_ID_COL = "essay_id"
OUTPUT_DIR = "results_extended_dif"

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Extended k-grid: original 12 points + 4 new long-range points
K_GRID = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

# Window params
TARGET_LENGTH = 32      # tokens to score per window
N_WINDOWS = 20          # windows per document
MIN_CONTEXT = 512       # need at least k_max tokens of context
END_BUFFER = 64         # skip last 64 tokens
MIN_TOKENS = MIN_CONTEXT + TARGET_LENGTH + END_BUFFER  # 608

SEED = 42

# ======================================================================
# SETUP
# ======================================================================

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name):
    """Load model with 4-bit quantization."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return model, tokenizer


def load_data(path, text_col, doc_id_col):
    """Load texts from JSONL or CSV."""
    path = Path(path)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in open(path)]
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(path)
    print(f"Loaded {len(df)} documents from {path}")
    return df


# ======================================================================
# GAIN CURVE COMPUTATION
# ======================================================================

@torch.no_grad()
def compute_nll_at_lag(model, token_ids, target_start, target_end, k, device):
    """Compute mean NLL on target region with k tokens of context.

    Parameters
    ----------
    token_ids : full document token list
    target_start, target_end : indices of scoring region in full sequence
    k : number of context tokens before target_start
    """
    if k == 0:
        # No context: feed only target tokens, score from position 1
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

        # Score only target tokens
        nlls = []
        for i in range(actual_k, len(chunk) - 1):
            lp = torch.log_softmax(logits[i], dim=-1)
            nlls.append(-lp[chunk[i + 1]].item())
        return np.mean(nlls) if nlls else float("nan")


def compute_gain_curve_single_doc(model, tokenizer, text, device, rng):
    """Compute gain curve at all k in K_GRID for one document."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)

    if n < MIN_TOKENS:
        return None

    # Sample window positions
    min_start = MIN_CONTEXT
    max_start = n - TARGET_LENGTH - END_BUFFER
    if max_start <= min_start:
        return None

    positions = np.linspace(min_start, max_start, N_WINDOWS, dtype=int)
    jitter = rng.integers(-5, 6, size=N_WINDOWS)
    positions = np.clip(positions + jitter, min_start, max_start)

    # Compute NLL at each (window, k)
    nll_by_k = {k: [] for k in K_GRID}

    for target_start in positions:
        target_end = target_start + TARGET_LENGTH
        for k in K_GRID:
            nll = compute_nll_at_lag(model, tokens, target_start, target_end, k, device)
            if not math.isnan(nll):
                nll_by_k[k].append(nll)

    # Average across windows, compute gain = nll_0 - nll_k
    result = {"n_tokens": n, "n_windows": len(positions)}
    nll_0 = np.mean(nll_by_k[0]) if nll_by_k[0] else float("nan")
    for k in K_GRID:
        mean_nll = np.mean(nll_by_k[k]) if nll_by_k[k] else float("nan")
        result[f"mean_nll_{k}"] = mean_nll
        result[f"mean_gain_{k}"] = nll_0 - mean_nll if not math.isnan(mean_nll) else float("nan")

    return result


# ======================================================================
# DIF COMPUTATION (extended)
# ======================================================================

def compute_extended_dif(df, k_grid):
    """Compute DIF on extended k-grid."""
    k_grid = np.array(k_grid)
    n_bins = len(k_grid) - 1
    bin_labels = []
    for i in range(n_bins):
        lo = k_grid[i] + 1 if k_grid[i] > 0 else 1
        hi = k_grid[i + 1]
        bin_labels.append(f"lag {lo}-{hi}")

    k_max = k_grid[-1]
    gain_max_col = f"mean_gain_{k_max}"

    rows = []
    for _, row in df.iterrows():
        g_max = row[gain_max_col]
        if g_max <= 0 or math.isnan(g_max):
            continue

        gains = {k: row[f"mean_gain_{k}"] for k in k_grid}
        shares = np.array([
            max(0, (gains[k_grid[i + 1]] - gains[k_grid[i]]) / g_max)
            for i in range(n_bins)
        ])
        s = shares.sum()
        if s > 0:
            shares /= s

        # DIF scalars
        def lq(q):
            cum = 0.0
            for i in range(n_bins):
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
            for i in range(n_bins):
                lo = k_grid[i] + 1 if k_grid[i] > 0 else 1
                hi = k_grid[i + 1]
                total += shares[i] * (lo + hi) / 2
            return total

        rec = {
            "doc_id": row.get("doc_id", row.get("essay_id", "")),
            "L50": lq(0.5), "L80": lq(0.8), "L90": lq(0.9),
            "L95": lq(0.95), "L99": lq(0.99),
            "tail_33_k": sum(shares[i] for i in range(n_bins) if k_grid[i] >= 32),
            "tail_65_k": sum(shares[i] for i in range(n_bins) if k_grid[i] >= 64),
            "tail_129_k": sum(shares[i] for i in range(n_bins) if k_grid[i] >= 128),
            "tail_257_k": sum(shares[i] for i in range(n_bins) if k_grid[i] >= 256),
            "mean_lag": mean_lag(),
            f"gain_{k_max}": g_max,
        }
        for i in range(n_bins):
            rec[f"p_{i}"] = shares[i]
        rows.append(rec)

    doc_df = pd.DataFrame(rows)

    # Bootstrap summary
    rng = np.random.default_rng(SEED)
    metrics = ["L50", "L80", "L90", "L95", "L99",
               "tail_33_k", "tail_65_k", "tail_129_k", "tail_257_k",
               "mean_lag", f"gain_{k_max}"]
    summary = []
    for m in metrics:
        vals = doc_df[m].dropna().values
        if len(vals) < 3:
            continue
        means = np.array([
            vals[rng.integers(0, len(vals), len(vals))].mean()
            for _ in range(2000)
        ])
        summary.append({
            "metric": m, "mean": vals.mean(),
            "ci_lo": np.percentile(means, 2.5),
            "ci_hi": np.percentile(means, 97.5),
        })

    summary_df = pd.DataFrame(summary)

    # Kernel summary
    kernel_rows = []
    for i in range(n_bins):
        vals = doc_df[f"p_{i}"].values
        means = np.array([
            vals[rng.integers(0, len(vals), len(vals))].mean()
            for _ in range(2000)
        ])
        kernel_rows.append({
            "bin": bin_labels[i],
            "share_mean": vals.mean(),
            "ci_lo": np.percentile(means, 2.5),
            "ci_hi": np.percentile(means, 97.5),
        })
    kernel_df = pd.DataFrame(kernel_rows)

    return doc_df, summary_df, kernel_df


def plot_extended_kernel(kernel_df, summary_df, corpus_name, out_dir, k_grid):
    """Plot extended kernel with new bins highlighted."""
    k_grid = np.array(k_grid)
    n_bins = len(k_grid) - 1
    x = np.arange(n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Bar chart
    ax = axes[0]
    colors = ["#3498db"] * 11 + ["#e74c3c"] * (n_bins - 11)  # red for new bins
    ax.bar(x, kernel_df["share_mean"].values, 0.7,
           yerr=kernel_df["ci_hi"].values - kernel_df["share_mean"].values,
           capsize=2, color=colors, alpha=0.8, edgecolor="white")

    # Divider between original and extended
    if n_bins > 11:
        ax.axvline(10.5, color="black", linestyle="--", alpha=0.5)
        ax.annotate("k > 128\n(new)", xy=(12, ax.get_ylim()[1] * 0.85),
                    fontsize=9, color="#e74c3c", fontweight="bold", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(kernel_df["bin"].values, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Share of total gain")
    ax.set_title(f"{corpus_name.upper()}: Extended Influence Kernel (k=0..{k_grid[-1]})")
    ax.grid(True, alpha=0.3, axis="y")

    # Cumulative
    ax2 = axes[1]
    cum = np.cumsum(kernel_df["share_mean"].values)
    bin_rights = k_grid[1:]
    ax2.plot(bin_rights, cum, "o-", color="#3498db", linewidth=2, markersize=5)

    for q, lbl in [(0.5, "L50"), (0.8, "L80"), (0.9, "L90"), (0.95, "L95"), (0.99, "L99")]:
        ax2.axhline(q, color="gray", linestyle=":", alpha=0.4)
        ax2.annotate(lbl, xy=(1.5, q + 0.008), fontsize=8, color="gray")

    # Mark boundary
    ax2.axvline(128, color="black", linestyle="--", alpha=0.4)
    ax2.annotate("k=128", xy=(130, 0.05), fontsize=8, color="black", alpha=0.6)

    ax2.set_xlabel("Lag (tokens)")
    ax2.set_ylabel("Cumulative share")
    ax2.set_title(f"{corpus_name.upper()}: Cumulative Extended Kernel")
    ax2.set_xscale("log")
    ax2.set_xlim(1, 600)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{corpus_name}_extended_kernel.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Load model
    model, tokenizer = load_model(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data = load_data(INPUT_FILE, TEXT_COL, DOC_ID_COL)

    # Compute extended gain curves
    results = []
    for idx, row in data.iterrows():
        doc_id = row[DOC_ID_COL]
        text = row[TEXT_COL]
        print(f"  [{idx + 1}/{len(data)}] {doc_id}...", end=" ", flush=True)

        res = compute_gain_curve_single_doc(model, tokenizer, text, device, rng)
        if res is None:
            print("SKIP (too short)")
            continue

        res[DOC_ID_COL] = doc_id
        # Carry through metadata columns
        for col in data.columns:
            if col not in [TEXT_COL, DOC_ID_COL] and col not in res:
                res[col] = row[col]
        results.append(res)
        print(f"OK ({res['n_tokens']} tok)")

    gain_df = pd.DataFrame(results)
    # Standardize doc_id column
    if DOC_ID_COL != "doc_id":
        gain_df["doc_id"] = gain_df[DOC_ID_COL]
    gain_df.to_csv(out_dir / f"{CORPUS_NAME}_extended_gain_curves.csv", index=False)
    print(f"\nSaved gain curves: {len(gain_df)} docs")

    # Compute extended DIF
    doc_df, summary_df, kernel_df = compute_extended_dif(gain_df, K_GRID)
    doc_df.to_csv(out_dir / f"{CORPUS_NAME}_extended_dif_doc_level.csv", index=False)
    summary_df.to_csv(out_dir / f"{CORPUS_NAME}_extended_dif_summary.csv", index=False)
    kernel_df.to_csv(out_dir / f"{CORPUS_NAME}_extended_kernel_summary.csv", index=False)

    # Print results
    print(f"\n=== Extended DIF Results: {CORPUS_NAME.upper()} (n={len(doc_df)}) ===")
    for _, r in summary_df.iterrows():
        m = r["metric"]
        v = r["mean"]
        if "tail" in m:
            print(f"  {m:14s}: {v:.1%}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
        else:
            print(f"  {m:14s}: {v:.2f}  [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")

    # Plot
    plot_extended_kernel(kernel_df, summary_df, CORPUS_NAME, out_dir, K_GRID)
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
