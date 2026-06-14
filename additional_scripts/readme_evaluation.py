"""
Evaluates a single generated README against an original using:
chunked semantic similarity, ROUGE-1/2 recall, BERTScore recall,
key-term coverage, and Flesch-Kincaid grade delta.

Requirements:
    pip install sentence-transformers rouge-score bert-score textstat
"""

import re
import json
import numpy as np
import textstat
from rouge_score import rouge_scorer as rouge_lib
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_fn

# ─── File paths ───────────────────────────────────────────────────────────────

GENERATED_README = "GENERATED_README_PATH"   
ORIGINAL_README  = "ORIGINAL_README_PATH"   

PROJECT_NAME     = "PROJECT_NAME"                             

# ─── Helpers ──────────────────────────────────────────────────────────────────

def read(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Cannot read {path}")


# ─── Metric 1: Chunked semantic similarity ────────────────────────────────────

print("Loading sentence transformer model...")
_sent_model = SentenceTransformer("all-mpnet-base-v2")

def chunked_similarity(text_a: str, text_b: str, chunk_size: int = 500) -> float:
    """Average max cosine similarity over fixed-size character chunks."""
    def chunks(text):
        lines = text.replace("\r", "").split("\n")
        buf, out = [], []
        for line in lines:
            buf.append(line)
            if sum(len(x) for x in buf) >= chunk_size:
                out.append(" ".join(buf))
                buf = []
        if buf:
            out.append(" ".join(buf))
        return out or [text]

    ca, cb = chunks(text_a), chunks(text_b)
    emb_a = _sent_model.encode(ca, convert_to_tensor=True, show_progress_bar=False)
    emb_b = _sent_model.encode(cb, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(emb_a, emb_b).cpu().numpy()
    return float(np.mean(sim_matrix.max(axis=1)))


# ─── Metric 2: ROUGE-1 / ROUGE-2 recall ──────────────────────────────────────

_rouge = rouge_lib.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

def rouge_recall(reference: str, hypothesis: str) -> tuple[float, float]:
    scores = _rouge.score(reference, hypothesis)
    return scores["rouge1"].recall, scores["rouge2"].recall


# ─── Metric 3: BERTScore recall ───────────────────────────────────────────────

def bertscore_recall(reference: str, hypothesis: str) -> float:
    hyp = hypothesis[:800]
    ref = reference[:800]
    _, R, _ = bert_score_fn(
        [hyp], [ref],
        lang="en",
        model_type="roberta-large",
        verbose=False,
    )
    return float(R.mean().item())


# ─── Metric 4: Key-term coverage ──────────────────────────────────────────────

def key_term_coverage(reference: str, hypothesis: str) -> float:
    def terms(text):
        return set(re.findall(r"\b[A-Za-z_]\w*(?:\.\w+)*\b", text))
    orig_terms = terms(reference)
    gen_terms  = terms(hypothesis)
    if not orig_terms:
        return 0.0
    return len(orig_terms & gen_terms) / len(orig_terms)


# ─── Metric 5: Flesch-Kincaid grade delta ─────────────────────────────────────

def fk_delta(reference: str, hypothesis: str) -> float:
    return textstat.flesch_kincaid_grade(hypothesis) - textstat.flesch_kincaid_grade(reference)


# ─── Evaluation ───────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"Evaluating: {PROJECT_NAME}")
print(f"  Generated : {GENERATED_README}")
print(f"  Original  : {ORIGINAL_README}")

original  = read(ORIGINAL_README)
generated = read(GENERATED_README)

print("  Computing semantic similarity...", end=" ", flush=True)
sem_sim = chunked_similarity(original, generated)
print(f"{sem_sim:.4f}")

print("  Computing ROUGE...", end=" ", flush=True)
r1, r2 = rouge_recall(original, generated)
print(f"R1={r1:.4f}  R2={r2:.4f}")

print("  Computing BERTScore...", end=" ", flush=True)
bert_r = bertscore_recall(original, generated)
print(f"{bert_r:.4f}")

print("  Computing key-term coverage...", end=" ", flush=True)
ktc = key_term_coverage(original, generated)
print(f"{ktc:.4f}")

print("  Computing FK delta...", end=" ", flush=True)
fk = fk_delta(original, generated)
print(f"{fk:+.2f}")

# ─── Summary ──────────────────────────────────────────────────────────────────

result = {
    "project":     PROJECT_NAME,
    "sem_sim":     round(sem_sim, 4),
    "rouge1_r":    round(r1, 4),
    "rouge2_r":    round(r2, 4),
    "bertscore_r": round(bert_r, 4),
    "keyterm_cov": round(ktc, 4),
    "fk_delta":    round(fk, 2),
}

print("\n" + "=" * 85)
print(f"{'Project':<18} {'Sem.Sim':>8} {'R1-R':>8} {'R2-R':>8} {'BERT-R':>8} {'KT-Cov':>8} {'FK Δ':>7}")
print("-" * 85)
print(f"{result['project']:<18} {result['sem_sim']:>8.4f} {result['rouge1_r']:>8.4f} "
      f"{result['rouge2_r']:>8.4f} {result['bertscore_r']:>8.4f} "
      f"{result['keyterm_cov']:>8.4f} {result['fk_delta']:>+7.2f}")
print("=" * 85)

# ─── Save to JSON ─────────────────────────────────────────────────────────────

out_path = f"readme_eval_{PROJECT_NAME}_results.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {out_path}")
