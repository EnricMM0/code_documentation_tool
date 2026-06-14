"""
eval_23.py  –  Section 2.3 parameter tuning evaluation
-------------------------------------------------------
Evaluates zero-shot, one-shot, and three-shot prompting strategies
on 500 stratified functions from the CodeSearchNet Python test split.

Requirements:
    pip install datasets sacrebleu rouge-score bert-score groq tqdm numpy pandas

Usage:
    python eval_23.py --api-key <GROQ_API_KEY>
    python eval_23.py --api-key <GROQ_API_KEY> --workers 8
"""

import argparse
import json
import math
import os
import random
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from datasets import load_dataset
from groq import Groq
from rouge_score import rouge_scorer as rouge_lib
from sacrebleu import sentence_bleu
from bert_score import score as bert_score_fn
from tqdm import tqdm

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

MODEL        = "llama-3.1-8b-instant"
N_SAMPLE     = 500          # total functions to evaluate
N_STRATA     = 3            # short / medium / long by prompt token count
N_WORKERS    = 5            # parallel Groq API threads
RANDOM_SEED  = 42
OUTPUT_CSV   = "eval_23_results.csv"
OUTPUT_JSON  = "eval_23_summary.json"

SYSTEM_PROMPT = (
    "You are an expert Python documentation generator. "
    "Write concise, accurate docstrings."
)
OUTPUT_RULES = (
    "Rules:\n"
    "- Return ONLY the docstring text, no quotes, no markdown.\n"
    "- Do NOT repeat the function name or signature.\n"
    "- Be concise and accurate."
)


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def remove_comments_and_docstrings(code: str) -> str:
    """Strip all comments and docstrings from a Python code snippet."""
    code = textwrap.dedent(code)
    # Remove triple-quoted strings
    code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        # Remove standalone string literals (one-line docstrings)
        if (stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 1) or \
           (stripped.startswith("'") and stripped.endswith("'") and len(stripped) > 1):
            continue
        # Remove inline # comments
        if '#' in line:
            line = line.split('#', 1)[0]
        lines.append(line.rstrip())
    return "\n".join(lines)


def preprocess_split(split, desc="Preprocessing"):
    """Remove comments/docstrings and filter samples where docstring is still in code."""
    cleaned = []
    skipped = 0
    for item in tqdm(split, desc=desc, unit="fn"):
        code = remove_comments_and_docstrings(item["code"])
        doc  = item["docstring"].strip()
        if not code.strip() or not doc:
            skipped += 1
            continue
        if doc in code:
            skipped += 1
            continue
        cleaned.append({"code": code, "docstring": doc})
    tqdm.write(f"  → kept {len(cleaned):,}  |  skipped {skipped:,}")
    return cleaned


def filter_percentile(items, lo=1, hi=99):
    """Remove docstring-length outliers outside the given percentile range."""
    lengths = [len(x["docstring"].split()) for x in items]
    p_lo = np.percentile(lengths, lo)
    p_hi = np.percentile(lengths, hi)
    filtered = [x for x in tqdm(items, desc="  Filtering outliers", unit="fn", leave=False)
                if p_lo <= len(x["docstring"].split()) <= p_hi]
    tqdm.write(f"  → after percentile filter [{lo}–{hi}%]: {len(filtered):,}")
    return filtered


# ──────────────────────────────────────────────
# Stratified sampling by prompt token count
# ──────────────────────────────────────────────

def estimate_tokens(code: str) -> int:
    """Rough token count: characters / 4 (GPT-style approximation)."""
    return max(1, len(code) // 4)


def stratified_sample(items, n_total, n_strata, seed=RANDOM_SEED):
    """
    Split items into n_strata equal-width buckets by prompt token count,
    then sample n_total // n_strata from each bucket.
    """
    rng = random.Random(seed)
    counts = [estimate_tokens(x["code"]) for x in items]
    min_c, max_c = min(counts), max(counts)
    boundaries = [min_c + i * (max_c - min_c) / n_strata for i in range(n_strata + 1)]

    buckets = [[] for _ in range(n_strata)]
    for item, c in zip(items, counts):
        for s in range(n_strata):
            if boundaries[s] <= c < boundaries[s + 1] or (s == n_strata - 1 and c == boundaries[-1]):
                buckets[s].append(item)
                break

    per_stratum = n_total // n_strata
    sample = []
    for b in buckets:
        take = min(per_stratum, len(b))
        sample.extend(rng.sample(b, take))
    rng.shuffle(sample)
    return sample


# ──────────────────────────────────────────────
# Groq API generation
# ──────────────────────────────────────────────

_RETRY_PATTERN = re.compile(r'try again in ([\d.]+)(ms|s)', re.IGNORECASE)

def _groq_call(client: Groq, messages: list, max_tokens: int = 200,
               max_retries: int = 8) -> str:
    """Call the Groq API with automatic retry on 429 rate-limit errors."""
    delay = 1.0  # fallback starting delay in seconds
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" not in msg:
                raise  # non-rate-limit errors bubble up immediately
            # Parse the suggested wait time from the error message if present
            m = _RETRY_PATTERN.search(msg)
            if m:
                value = float(m.group(1))
                unit  = m.group(2).lower()
                delay = (value / 1000.0) if unit == "ms" else value
                delay += 0.2   # small safety buffer
            else:
                delay = min(delay * 2, 60.0)  # exponential backoff, cap at 60s
            if attempt < max_retries - 1:
                tqdm.write(f"  [429] rate limit – retrying in {delay:.2f}s "
                           f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def generate_0shot(client: Groq, code: str) -> str:
    return _groq_call(client, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Write a docstring for the following Python function.\n{OUTPUT_RULES}\n\n"
            f"```python\n{code}\n```"
        )},
    ])


def generate_1shot(client: Groq, code: str, train_items: list) -> str:
    ex = random.choice(train_items)
    return _groq_call(client, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Write a docstring for the following Python function.\n{OUTPUT_RULES}\n\n"
            f"### EXAMPLE\n"
            f"Code:\n```python\n{ex['code']}\n```\n"
            f"Docstring: {ex['docstring']}\n\n"
            f"### YOUR TURN\n"
            f"Code:\n```python\n{code}\n```"
        )},
    ])


def generate_3shot(client: Groq, code: str, train_items: list) -> str:
    examples = random.sample(train_items, 3)
    ex_block = ""
    for i, ex in enumerate(examples, 1):
        ex_block += (
            f"### EXAMPLE {i}\n"
            f"Code:\n```python\n{ex['code']}\n```\n"
            f"Docstring: {ex['docstring']}\n\n"
        )
    return _groq_call(client, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Write a docstring for the following Python function.\n{OUTPUT_RULES}\n\n"
            f"{ex_block}"
            f"### YOUR TURN\n"
            f"Code:\n```python\n{code}\n```"
        )},
    ])


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str) -> float:
    return sentence_bleu(hypothesis, [reference]).score


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


# BERTScore is batched at the end for efficiency.


# ──────────────────────────────────────────────
# Per-function evaluation (runs all 3 strategies)
# ──────────────────────────────────────────────

def evaluate_item(idx: int, item: dict, client: Groq, train_items: list) -> dict:
    code = item["code"]
    ref  = item["docstring"]
    row  = {"idx": idx, "reference": ref}

    for strategy, fn in [
        ("0shot", lambda: generate_0shot(client, code)),
        ("1shot", lambda: generate_1shot(client, code, train_items)),
        ("3shot", lambda: generate_3shot(client, code, train_items)),
    ]:
        try:
            gen = fn()
        except Exception as e:
            print(f"  [WARN] idx={idx} strategy={strategy} error: {e}")
            gen = ""

        row[f"{strategy}_gen"]   = gen
        row[f"{strategy}_bleu"]  = compute_bleu(ref, gen)  if gen else 0.0
        row[f"{strategy}_rouge"] = compute_rouge_l(ref, gen) if gen else 0.0
        # BERTScore filled in later in batch
        row[f"{strategy}_bert"]  = None

    return row


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Groq API key")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--n",       type=int, default=N_SAMPLE)
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)

    client = Groq(api_key=args.api_key)

    # ── 1. Load dataset ──────────────────────────────────────
    print("Loading CodeSearchNet Python dataset...")
    dataset = load_dataset("google/code_x_glue_ct_code_to_text", "python")
    train_raw = list(dataset["train"])
    test_raw  = list(dataset["test"])
    print(f"  Train: {len(train_raw):,}  |  Test: {len(test_raw):,}")

    # ── 2. Preprocess ────────────────────────────────────────
    print("\nPreprocessing train split (for few-shot examples)...")
    train_clean = preprocess_split(train_raw, desc="Train – removing comments")
    train_clean = filter_percentile(train_clean)

    print("\nPreprocessing test split...")
    test_clean = preprocess_split(test_raw, desc="Test  – removing comments")
    test_clean = filter_percentile(test_clean)

    # ── 3. Stratified sample from test ───────────────────────
    print(f"\nStratified sampling {args.n} functions from test split ({N_STRATA} strata by token count)...")
    sample = stratified_sample(test_clean, args.n, N_STRATA, seed=args.seed)
    per_stratum = args.n // N_STRATA
    print(f"  Sampled {len(sample):,} functions (~{per_stratum} per stratum)")

    # ── 4. Run evaluation in parallel ────────────────────────
    print(f"\nRunning evaluation ({len(sample)} functions × 3 strategies, {args.workers} workers)...")
    rows = [None] * len(sample)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_item, i, item, client, train_clean): i
            for i, item in enumerate(sample)
        }
        with tqdm(total=len(futures), desc="Evaluating", unit="fn",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    rows[i] = future.result()
                except Exception as e:
                    tqdm.write(f"  [ERROR] idx={i}: {e}")
                    rows[i] = {"idx": i, "reference": sample[i]["docstring"]}
                pbar.update(1)

    rows = [r for r in rows if r is not None]

    # ── 5. BERTScore (batched) ────────────────────────────────
    # Truncate by character count: 800 chars ≈ 160 English words ≈ 200 tokens,
    # well under BERT-base's 512-token hard limit regardless of tokenisation.
    print(f"\nComputing BERTScore for {len(rows)} functions × 3 strategies...")
    MAX_CHARS = 800

    for strategy in tqdm(["0shot", "1shot", "3shot"],
                         desc="BERTScore strategies", unit="strategy"):
        refs = [(r.get("reference", "") or "")[:MAX_CHARS] for r in rows]
        hyps = [(r.get(f"{strategy}_gen", "") or "")[:MAX_CHARS] for r in rows]
        _, _, F = bert_score_fn(
            hyps, refs,
            model_type="bert-base-uncased",
            batch_size=16,
            verbose=False,
            device="cpu",
        )
        for r, f_val in zip(rows, F.tolist()):
            r[f"{strategy}_bert"] = f_val

    # ── 6. Save results ───────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # ── 7. Summary ────────────────────────────────────────────
    summary = {}
    print("\n" + "="*55)
    print(f"{'Strategy':<12} {'BLEU':>8} {'ROUGE-L':>10} {'BERTScore':>12}")
    print("-"*55)
    for strategy in ["0shot", "1shot", "3shot"]:
        bleu  = df[f"{strategy}_bleu"].mean()
        rouge = df[f"{strategy}_rouge"].mean()
        bert  = df[f"{strategy}_bert"].mean()
        summary[strategy] = {"BLEU": round(bleu, 4), "ROUGE-L": round(rouge, 4), "BERTScore": round(bert, 4)}
        print(f"{strategy:<12} {bleu:>8.4f} {rouge:>10.4f} {bert:>12.4f}")
    print("="*55)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
