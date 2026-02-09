"""
DIF Token-Shuffle Control — Colab template for proper re-inference.

This script is meant to run on Google Colab with GPU.
It takes the original text, shuffles tokens within each document,
re-runs Mistral-7B inference to get new gain curves, then computes DIF
on the shuffled data.

This is the "gold standard" shuffle control that destroys real linguistic
structure while preserving token-level statistics.

NOTE: Requires GPU + transformers + bitsandbytes installed.

Usage (Colab):
  1. Upload this script and your text data
  2. Install deps: !pip install transformers bitsandbytes accelerate pandas
  3. Run each section in order
"""

# =============================================================
# SECTION 1: Setup (run on Colab)
# =============================================================

TEMPLATE = '''
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Config ---
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
K_GRID = [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
WINDOW_SIZE = 256
STRIDE = 128  # sliding window stride
N_SHUFFLE_REPS = 10  # number of shuffle repetitions per document
SEED = 42

# --- Load model ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config, device_map="auto"
)
model.eval()


# =============================================================
# SECTION 2: Token shuffle function
# =============================================================

def shuffle_tokens(token_ids, rng):
    """Shuffle token IDs within a document (preserving BOS if present)."""
    ids = list(token_ids)
    # Keep BOS token in place if present
    if ids and ids[0] == tokenizer.bos_token_id:
        body = ids[1:]
        rng.shuffle(body)
        return [ids[0]] + body
    else:
        rng.shuffle(ids)
        return ids


def compute_gain_curve(token_ids, k_grid=K_GRID, window_size=WINDOW_SIZE):
    """Compute gain curve for a sequence of token IDs using sliding windows."""
    n = len(token_ids)
    if n < window_size:
        return None

    input_ids = torch.tensor([token_ids], device=model.device)
    windows = []
    for start in range(0, n - window_size + 1, STRIDE):
        windows.append((start, start + window_size))

    gains_by_k = {k: [] for k in k_grid}

    with torch.no_grad():
        for w_start, w_end in windows:
            chunk = input_ids[:, w_start:w_end]
            outputs = model(chunk)
            logits = outputs.logits  # (1, window_size, vocab)

            # NLL at each position
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = chunk[:, 1:]
            nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            nll = nll.squeeze(0).cpu().numpy()  # (window_size - 1,)

            # For each k, compute mean NLL using only positions with >= k context
            for k in k_grid:
                # Positions with at least k tokens of context within this window
                valid_pos = slice(max(k, 1) - 1, None)
                mean_nll_k = nll[valid_pos].mean()
                # gain_k = NLL_0 - NLL_k (approximation)
                if k == 0:
                    gains_by_k[0].append(0.0)
                else:
                    nll_0 = nll[0]  # NLL with minimal context
                    gains_by_k[k].append(nll_0 - mean_nll_k)

    # Average across windows
    result = {}
    for k in k_grid:
        vals = gains_by_k[k]
        result[k] = np.mean(vals) if vals else 0.0

    return result


# =============================================================
# SECTION 3: Run shuffle control
# =============================================================

def run_token_shuffle_control(texts, doc_ids, n_reps=N_SHUFFLE_REPS):
    """Run token-shuffle control on a list of texts.

    For each document:
      1. Tokenize original text, compute gain curve (if not already available)
      2. Shuffle tokens N_SHUFFLE_REPS times
      3. Re-run inference on each shuffle
      4. Average gain curves across shuffles
      5. Compute DIF on shuffled gain curves

    Returns DataFrame with real and shuffled DIF per document.
    """
    rng = np.random.default_rng(SEED)
    results = []

    for doc_idx, (text, doc_id) in enumerate(zip(texts, doc_ids)):
        print(f"  [{doc_idx+1}/{len(texts)}] {doc_id}")
        tokens = tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) < WINDOW_SIZE:
            print(f"    Skipping (too short: {len(tokens)} tokens)")
            continue

        # Real gain curve
        real_gains = compute_gain_curve(tokens)
        if real_gains is None:
            continue

        # Shuffled gain curves
        shuffled_gains_all = {k: [] for k in K_GRID}
        for rep in range(n_reps):
            shuffled_tokens = shuffle_tokens(tokens.copy(), rng)
            shuf_gains = compute_gain_curve(shuffled_tokens)
            if shuf_gains is not None:
                for k in K_GRID:
                    shuffled_gains_all[k].append(shuf_gains[k])

        # Average shuffled gains
        avg_shuf = {k: np.mean(shuffled_gains_all[k]) for k in K_GRID}

        results.append({
            "doc_id": doc_id,
            "n_tokens": len(tokens),
            **{f"real_gain_{k}": real_gains[k] for k in K_GRID},
            **{f"shuf_gain_{k}": avg_shuf[k] for k in K_GRID},
        })

    return pd.DataFrame(results)


# =============================================================
# SECTION 4: Example usage
# =============================================================

# --- Load your data ---
# For ECSC:
#   texts = [...]  # list of transcript texts
#   doc_ids = [...]
# For PERSUADE:
#   df = pd.read_csv("persuade_texts.csv")
#   texts = df["text"].tolist()
#   doc_ids = df["essay_id"].tolist()

# results = run_token_shuffle_control(texts, doc_ids, n_reps=10)
# results.to_csv("token_shuffle_results.csv", index=False)
# print("Done! Now compute DIF on the shuffled gain columns.")
'''

if __name__ == "__main__":
    print("This is a Colab template. Copy to a Colab notebook to run.")
    print("The template is stored in the TEMPLATE variable.")
    print()
    print(TEMPLATE)
