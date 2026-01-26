# LRTIA - Long-Range Token Influence Analyzer

## Project Overview

LRTIA is a tool for measuring how earlier text influences predictions in language models. The core question: **How far back does context matter for prediction?**

The tool produces "memory curves" showing how the benefit of context decays with distance, and extracts summary metrics like **half-life** (context length to achieve 50% of total benefit).

## Key Insight

Long-range context effects are **subtle and distributed** across all tokens, not dramatic effects on specific tokens. Like a conversation that builds on earlier topics - every sentence is slightly more predictable because of what came before, but no single word has a massive dependency on distant context.

This insight led us away from span-masking approaches toward **context ablation**.

## The Context Ablation Method

Instead of masking specific spans and measuring delta-NLL, we:

1. **Pick a target region** (e.g., last 40 tokens of a document)
2. **Progressively truncate context** (test with 8, 16, 32, 64, 128, 256... tokens before target)
3. **Measure perplexity on the target** at each context length
4. **Plot the "memory curve"** showing how perplexity improves as context increases

### Why This Works Better Than Span Masking

- Span masking assumes localized dependencies (masking position X hurts prediction at position Y)
- In reality, coherence provides a diffuse benefit spread across all predictions
- Context ablation captures this cumulative, distributed benefit naturally

## Key Metrics

### Half-Life
The primary scalar metric. Context length (in tokens) at which 50% of the total perplexity benefit is achieved.

- **Lower half-life** = most benefit comes from local context
- **Higher half-life** = benefits from longer-range context

### Expected Patterns

For **intact (coherent) text**:
- Steep drop in perplexity with first 32-64 tokens
- Then plateaus (local coherence already provides most of the signal)
- Half-life typically 20-40 tokens

For **shuffled (sentence-randomized) text**:
- Gradual improvement across all context lengths
- Needs more context to find any useful patterns
- Half-life typically 100-200 tokens

The **ratio of half-lives** (shuffled/intact) quantifies the long-range coherence structure.

## Sanity Check Results

Using Mistral-7B on 5 extended passages (~500 tokens each):

```
Perplexity by context length:
Context     Intact    Shuffled    Ratio
8           5.10      8.52        1.67x
32          4.15      7.74        1.87x
64          3.49      7.12        2.04x
128         3.18      6.29        1.98x
256         3.13      5.41        1.73x
384         2.90      4.48        1.54x
```

Key findings:
- Intact always more predictable than shuffled (as expected)
- Intact gets most benefit in first 64 tokens, then plateaus
- Shuffled keeps improving throughout (searching for signal)
- Half-life ratio ~3-5x differentiates the conditions

## Repository Structure

```
lrtia/
├── lrtia/                      # Main package (original span-masking approach)
│   ├── config.py               # Pydantic configuration
│   ├── cli.py                  # Typer CLI
│   ├── data/                   # Data ingestion and windowing
│   ├── intervention/           # Span selection and transforms
│   ├── model/                  # HuggingFace backend
│   ├── metrics/                # Delta-NLL, KL, etc.
│   └── ...
├── notebooks/                  # Colab notebooks for context ablation
│   ├── LRTIA_Context_Ablation.ipynb        # Basic version
│   ├── LRTIA_Context_Ablation_Dense.ipynb  # Dense sampling with half-life
│   ├── LRTIA_Perplexity_Test.ipynb         # Simple intact vs shuffled
│   └── LRTIA_Colab_v2.ipynb                # Earlier span-masking attempt
├── scripts/                    # Corpus generation scripts
├── configs/                    # YAML configurations
└── tests/                      # Unit tests
```

## Current State

### What Works
- Context ablation method clearly differentiates intact from shuffled text
- Half-life metric provides meaningful scalar summary
- Notebooks run on Colab with Mistral-7B (4-bit quantized on T4 GPU)
- Dense sampling (22 context lengths from 4-448 tokens) captures curve shape

### What's Next
1. **Integrate context ablation into main LRTIA framework** - Currently only in notebooks
2. **Test on real population comparisons** - Different authors, genres, etc.
3. **Longer documents** - Current passages are ~500 tokens, limiting max context to ~460
4. **Multiple target positions** - Currently only measures last 40 tokens
5. **Statistical comparison framework** - Bootstrap CIs for half-life differences

## Technical Notes

### Model Loading (Colab)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Perplexity Computation
```python
def compute_perplexity_on_region(token_ids, target_start, target_end):
    """Compute perplexity only on tokens in [target_start, target_end]."""
    input_ids = torch.tensor([token_ids], device=model.device)
    outputs = model(input_ids)
    logits = outputs.logits[0]

    total_loss = 0.0
    for i in range(target_start, target_end - 1):
        log_probs = torch.log_softmax(logits[i], dim=-1)
        token_loss = -log_probs[token_ids[i + 1]].item()
        total_loss += token_loss

    return np.exp(total_loss / (target_end - target_start - 1))
```

### Half-Life Computation
```python
def compute_half_life(contexts, perplexities):
    """Context length where 50% of benefit is achieved."""
    total_benefit = perplexities[0] - perplexities[-1]
    target_ppl = perplexities[0] - 0.5 * total_benefit

    # Linear interpolation to find crossing point
    for i in range(len(perplexities) - 1):
        if perplexities[i] >= target_ppl >= perplexities[i+1]:
            frac = (perplexities[i] - target_ppl) / (perplexities[i] - perplexities[i+1])
            return contexts[i] + frac * (contexts[i+1] - contexts[i])
    return contexts[-1]
```

## Test Passages

The notebooks use 5 extended passages (~500 tokens each) that "build on themselves":
1. **Neural Networks** - Technical explanation with forward references
2. **Detective Story** - Narrative with plot development
3. **Climate Change** - Scientific argument building to conclusion
4. **Risotto Recipe** - Sequential cooking instructions
5. **Roman Empire** - Historical narrative with cause and effect

Each passage is tested intact and with 5 sentence-shuffled versions.

## Key Files for Continuation

1. **`notebooks/LRTIA_Context_Ablation_Dense.ipynb`** - Most complete notebook with half-life analysis
2. **`lrtia/config.py`** - Configuration schema (may need updating for context ablation)
3. **`scripts/generate_sanity_check_corpus.py`** - Generates test corpora

## Original Plan File

See `.claude/plans/adaptive-churning-hellman.md` for the original implementation plan (focused on span-masking approach, now superseded by context ablation for the core analysis).

## Contact

This project was developed iteratively through conversation. The key breakthrough was recognizing that long-range effects are subtle and distributed, not localized - leading to the context ablation approach.
