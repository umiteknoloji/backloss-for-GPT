#!/usr/bin/env python3
"""
PRO: Sparse BackLoss + confidence-based routing + n-gram mixture
Kaggle-friendly reference implementation (single-file).

This script is designed for large-vocab next-token modeling workflows where
n-gram priors dominate and Sparse BackLoss contributes only where confident.

Key defaults (from research notes):
- raw embedding
- emb_dim=64
- min_samples/min_target=50/8
- n-gram weights: 4/3/2 = 0.40/0.35/0.25
- confidence = |w| * n (not sqrt(n))
- global alpha base around 0.95 (favor n-gram)
- sparse BL prediction from softmax(W @ f)
"""

from __future__ import annotations

import argparse
import sys
import json
import math
import os
import random
import time
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class ProConfig:
    # Data / run
    train_path: Optional[str] = None
    eval_path: Optional[str] = None
    output_dir: str = "./pro_outputs"
    seed: int = 42
    max_train_tokens: Optional[int] = None
    max_eval_tokens: Optional[int] = 30000

    # Vocabulary
    vocab_size: int = 50257

    # Embedding
    emb_dim: int = 64
    embedding_type: str = "raw"  # only raw in this script

    # Sparse expert thresholds
    min_samples: int = 50
    min_target: int = 8

    # BackLoss
    eps: float = 1e-6

    # N-gram mix: (4g, 3g, 2g)
    w4: float = 0.40
    w3: float = 0.35
    w2: float = 0.25

    # Global blend (ngram-heavy)
    alpha_base: float = 0.95

    # Confidence routing
    conf_ref_quantile: float = 0.95

    # Runtime
    topk_bl: int = 2048  # compute BL only over top-k frequent candidate tokens


# -----------------------------
# Utility
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_token_ids(path: str, max_tokens: Optional[int] = None) -> np.ndarray:
    """
    Expected file format:
    - JSON list of ints, OR
    - whitespace-separated ints text file.
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if not txt:
        return np.array([], dtype=np.int32)

    if txt[0] == "[":
        arr = np.array(json.loads(txt), dtype=np.int32)
    else:
        vals = txt.split()
        arr = np.array([int(v) for v in vals], dtype=np.int32)

    if max_tokens is not None:
        arr = arr[:max_tokens]
    return arr


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x)
    z = np.exp(x - m)
    s = z.sum()
    if s <= 0:
        return np.full_like(z, 1.0 / len(z))
    return z / s


def perplexity_from_logprob_sum(logprob_sum: float, n: int) -> float:
    if n <= 0:
        return float("inf")
    return float(math.exp(-logprob_sum / n))


# -----------------------------
# N-gram model
# -----------------------------

class NGramTables:
    def __init__(self) -> None:
        self.uni = Counter()
        self.bi = defaultdict(Counter)          # (t-1) -> next counts
        self.tri = defaultdict(Counter)         # (t-2, t-1) -> next
        self.four = defaultdict(Counter)        # (t-3, t-2, t-1) -> next

        self.uni_total = 0
        self.vocab_size = 0

    def fit(self, toks: np.ndarray, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        for i, tok in enumerate(toks.tolist()):
            self.uni[tok] += 1
            if i >= 1:
                self.bi[toks[i - 1]][tok] += 1
            if i >= 2:
                self.tri[(toks[i - 2], toks[i - 1])][tok] += 1
            if i >= 3:
                self.four[(toks[i - 3], toks[i - 2], toks[i - 1])][tok] += 1
        self.uni_total = int(len(toks))

    def _prob_from_counter(self, cnt: Counter, tok: int, alpha: float = 0.1) -> float:
        # tiny additive smoothing
        denom = sum(cnt.values()) + alpha * self.vocab_size
        return (cnt.get(tok, 0) + alpha) / denom

    def p2(self, t1: int, tok: int) -> float:
        return self._prob_from_counter(self.bi[t1], tok)

    def p3(self, t2: int, t1: int, tok: int) -> float:
        return self._prob_from_counter(self.tri[(t2, t1)], tok)

    def p4(self, t3: int, t2: int, t1: int, tok: int) -> float:
        return self._prob_from_counter(self.four[(t3, t2, t1)], tok)

    def interpolated_prob(self, ctx: Tuple[int, int, int], tok: int, w4: float, w3: float, w2: float) -> float:
        t3, t2, t1 = ctx
        return w4 * self.p4(t3, t2, t1, tok) + w3 * self.p3(t2, t1, tok) + w2 * self.p2(t1, tok)


# -----------------------------
# Sparse BackLoss experts
# -----------------------------

@dataclass
class Expert:
    # dense matrix over candidate subset: [n_cands, 2*emb_dim]
    W: np.ndarray
    cand_ids: np.ndarray
    # per-candidate confidence proxy |w| * n
    cand_conf: np.ndarray


class SparseBackLoss:
    def __init__(self, cfg: ProConfig, token_emb: np.ndarray, freq_topk: np.ndarray):
        self.cfg = cfg
        self.token_emb = token_emb
        self.freq_topk = freq_topk

        self.pair_experts: Dict[Tuple[int, int], Expert] = {}
        self.unigram_experts: Dict[int, Expert] = {}
        self.global_expert: Optional[Expert] = None

        self.conf_ref: float = 1.0

    def _build_expert(self, X: np.ndarray, y: np.ndarray, cand_ids: np.ndarray) -> Optional[Expert]:
        # X: [n, d], y: [n]
        n, d = X.shape
        if n < self.cfg.min_samples:
            return None

        # global stats
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma < self.cfg.eps, self.cfg.eps, sigma)

        N = math.log(d + 1.0)
        denom = (N * sigma).astype(np.float64)

        # map y to candidate index
        y_counter = Counter(y.tolist())

        rows = []
        confidences = []
        kept_ids = []

        for c in cand_ids.tolist():
            n_c = y_counter.get(int(c), 0)
            if n_c < self.cfg.min_target:
                continue

            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            mu_c = X[idx].mean(axis=0)

            w = (mu_c - mu) / denom
            conf = float(np.abs(w).mean() * n_c)  # |w| * n

            rows.append(w.astype(np.float32))
            confidences.append(conf)
            kept_ids.append(c)

        if not rows:
            return None

        W = np.stack(rows, axis=0)
        cand_conf = np.array(confidences, dtype=np.float32)
        kept = np.array(kept_ids, dtype=np.int32)
        return Expert(W=W, cand_ids=kept, cand_conf=cand_conf)

    def fit(self, toks: np.ndarray) -> None:
        # Build training tuples: context (t-2, t-1) -> target t
        # Need at least 3 tokens for context and target
        if len(toks) < 4:
            return

        d = self.cfg.emb_dim * 2

        # Pre-collect by pair and unigram
        pair_Xy = defaultdict(lambda: {"X": [], "y": []})
        uni_Xy = defaultdict(lambda: {"X": [], "y": []})
        global_X = []
        global_y = []

        for i in range(2, len(toks)):
            p2 = int(toks[i - 2])
            p1 = int(toks[i - 1])
            y = int(toks[i])

            x = np.concatenate([self.token_emb[p2], self.token_emb[p1]], axis=0)
            if x.shape[0] != d:
                raise ValueError("Feature dimension mismatch.")

            pair_Xy[(p2, p1)]["X"].append(x)
            pair_Xy[(p2, p1)]["y"].append(y)

            uni_Xy[p1]["X"].append(x)
            uni_Xy[p1]["y"].append(y)

            global_X.append(x)
            global_y.append(y)

        # Build global expert first (fallback)
        Xg = np.array(global_X, dtype=np.float32)
        yg = np.array(global_y, dtype=np.int32)
        self.global_expert = self._build_expert(Xg, yg, self.freq_topk)

        # Pair experts
        for key, data in pair_Xy.items():
            X = np.array(data["X"], dtype=np.float32)
            y = np.array(data["y"], dtype=np.int32)
            ex = self._build_expert(X, y, self.freq_topk)
            if ex is not None:
                self.pair_experts[key] = ex

        # Unigram fallback experts
        for key, data in uni_Xy.items():
            X = np.array(data["X"], dtype=np.float32)
            y = np.array(data["y"], dtype=np.int32)
            ex = self._build_expert(X, y, self.freq_topk)
            if ex is not None:
                self.unigram_experts[key] = ex

        # Calibrate confidence reference
        all_conf = []
        for ex in self.pair_experts.values():
            all_conf.append(float(np.max(ex.cand_conf)))
        for ex in self.unigram_experts.values():
            all_conf.append(float(np.max(ex.cand_conf)))
        if self.global_expert is not None:
            all_conf.append(float(np.max(self.global_expert.cand_conf)))

        if all_conf:
            self.conf_ref = float(np.quantile(np.array(all_conf), self.cfg.conf_ref_quantile))
            self.conf_ref = max(self.conf_ref, 1e-6)

    def _select_expert(self, p2: int, p1: int) -> Optional[Expert]:
        ex = self.pair_experts.get((p2, p1))
        if ex is not None:
            return ex
        ex = self.unigram_experts.get(p1)
        if ex is not None:
            return ex
        return self.global_expert

    def predict_bl_dist(self, p2: int, p1: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns:
        - cand_ids
        - p_bl over cand_ids
        - routing confidence scalar r in [0,1]
        """
        ex = self._select_expert(p2, p1)
        if ex is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32), 0.0

        x = np.concatenate([self.token_emb[p2], self.token_emb[p1]], axis=0).astype(np.float32)
        logits = ex.W @ x
        p = stable_softmax(logits).astype(np.float32)

        conf_pair = float(np.max(ex.cand_conf)) if len(ex.cand_conf) else 0.0
        r = conf_pair / self.conf_ref if self.conf_ref > 0 else 0.0
        r = float(np.clip(r, 0.0, 1.0))
        return ex.cand_ids, p, r


# -----------------------------
# PRO Model
# -----------------------------

class ProSparseBLModel:
    def __init__(self, cfg: ProConfig):
        self.cfg = cfg
        self.ng = NGramTables()
        self.token_emb: Optional[np.ndarray] = None
        self.bl: Optional[SparseBackLoss] = None
        self.freq_topk: Optional[np.ndarray] = None

    def _build_raw_embeddings(self, train_toks: np.ndarray) -> np.ndarray:
        # Frequency profile + random sign projection flavor (deterministic seed)
        V = self.cfg.vocab_size
        D = self.cfg.emb_dim
        rng = np.random.default_rng(self.cfg.seed)

        # Build lightweight co-occurrence profile with previous token
        co = np.zeros((V, D), dtype=np.float32)
        proj = rng.standard_normal((V, D), dtype=np.float32) * 0.01

        for i in range(1, len(train_toks)):
            prev = int(train_toks[i - 1])
            cur = int(train_toks[i])
            co[cur] += proj[prev]

        # Normalize rows; unseen rows stay near zero then get tiny noise
        norms = np.linalg.norm(co, axis=1, keepdims=True) + 1e-6
        emb = co / norms
        emb += rng.standard_normal((V, D), dtype=np.float32) * 1e-4
        return emb.astype(np.float32)

    def fit(self, train_toks: np.ndarray) -> None:
        self.ng.fit(train_toks, self.cfg.vocab_size)

        # Candidate BL classes = most frequent tokens
        freqs = np.zeros(self.cfg.vocab_size, dtype=np.int64)
        for t, c in self.ng.uni.items():
            freqs[int(t)] = int(c)
        topk_ids = np.argsort(-freqs)[: self.cfg.topk_bl]
        self.freq_topk = topk_ids.astype(np.int32)

        self.token_emb = self._build_raw_embeddings(train_toks)

        self.bl = SparseBackLoss(cfg=self.cfg, token_emb=self.token_emb, freq_topk=self.freq_topk)
        self.bl.fit(train_toks)

    def _ng_prob(self, t3: int, t2: int, t1: int, tok: int) -> float:
        return self.ng.interpolated_prob((t3, t2, t1), tok, self.cfg.w4, self.cfg.w3, self.cfg.w2)

    def predict_next_prob(self, t3: int, t2: int, t1: int, tok: int) -> float:
        p_ng = self._ng_prob(t3, t2, t1, tok)

        if self.bl is None:
            return p_ng

        cand_ids, p_bl, r = self.bl.predict_bl_dist(t2, t1)
        if len(cand_ids) == 0:
            return p_ng

        # alpha_eff = 0.95 + 0.05*(1-r), i.e. trust n-gram more when confidence low
        alpha_eff = self.cfg.alpha_base + (1.0 - self.cfg.alpha_base) * (1.0 - r)
        alpha_eff = float(np.clip(alpha_eff, 0.0, 1.0))

        # BL prob only defined on candidate subset
        idx = np.where(cand_ids == tok)[0]
        p_bl_tok = float(p_bl[idx[0]]) if len(idx) else 0.0

        p = alpha_eff * p_ng + (1.0 - alpha_eff) * p_bl_tok
        return max(p, 1e-12)

    def evaluate(self, eval_toks: np.ndarray) -> Dict[str, float]:
        if len(eval_toks) < 4:
            return {"ppl": float("inf"), "acc": 0.0, "n": 0}

        logprob_sum = 0.0
        correct = 0
        total = 0

        # For accuracy, we pick argmax over union of:
        # - top-k BL candidates
        # - local ngram support set (tokens seen in p4/p3/p2 counters)
        for i in range(3, len(eval_toks)):
            t3 = int(eval_toks[i - 3])
            t2 = int(eval_toks[i - 2])
            t1 = int(eval_toks[i - 1])
            y = int(eval_toks[i])

            p_true = self.predict_next_prob(t3, t2, t1, y)
            logprob_sum += math.log(p_true)

            # candidate set for argmax
            cand = set()
            cand.update(self.ng.bi[t1].keys())
            cand.update(self.ng.tri[(t2, t1)].keys())
            cand.update(self.ng.four[(t3, t2, t1)].keys())
            if self.freq_topk is not None:
                cand.update(self.freq_topk[:256].tolist())

            if not cand:
                yhat = int(np.argmax(self.freq_topk)) if self.freq_topk is not None else 0
            else:
                best_tok, best_p = None, -1.0
                for tok in cand:
                    p = self.predict_next_prob(t3, t2, t1, int(tok))
                    if p > best_p:
                        best_p = p
                        best_tok = int(tok)
                yhat = best_tok if best_tok is not None else 0

            correct += int(yhat == y)
            total += 1

        ppl = perplexity_from_logprob_sum(logprob_sum, total)
        acc = 100.0 * correct / max(total, 1)
        return {"ppl": float(ppl), "acc": float(acc), "n": int(total)}


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> ProConfig:
    ap = argparse.ArgumentParser(description="PRO Sparse BackLoss + n-gram trainer/evaluator")
    ap.add_argument("--train_path", type=str, default=None)
    ap.add_argument("--eval_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="./pro_outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab_size", type=int, default=50257)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--min_samples", type=int, default=50)
    ap.add_argument("--min_target", type=int, default=8)
    ap.add_argument("--w4", type=float, default=0.40)
    ap.add_argument("--w3", type=float, default=0.35)
    ap.add_argument("--w2", type=float, default=0.25)
    ap.add_argument("--alpha_base", type=float, default=0.95)
    ap.add_argument("--conf_ref_quantile", type=float, default=0.95)
    ap.add_argument("--topk_bl", type=int, default=2048)
    ap.add_argument("--max_train_tokens", type=int, default=0)
    ap.add_argument("--max_eval_tokens", type=int, default=30000)

    # In notebook environments argv can contain unknown IPython flags.
    # parse_known_args prevents hard-crash (SystemExit) in those cases.
    if argv is None:
        argv = sys.argv[1:]
    a, _unknown = ap.parse_known_args(argv)
    cfg = ProConfig(
        train_path=a.train_path,
        eval_path=a.eval_path,
        output_dir=a.output_dir,
        seed=a.seed,
        max_train_tokens=None if a.max_train_tokens <= 0 else a.max_train_tokens,
        max_eval_tokens=None if a.max_eval_tokens <= 0 else a.max_eval_tokens,
        vocab_size=a.vocab_size,
        emb_dim=a.emb_dim,
        min_samples=a.min_samples,
        min_target=a.min_target,
        w4=a.w4,
        w3=a.w3,
        w2=a.w2,
        alpha_base=a.alpha_base,
        conf_ref_quantile=a.conf_ref_quantile,
        topk_bl=a.topk_bl,
    )
    return cfg


def _resolve_path(cli_value: Optional[str], env_name: str) -> Optional[str]:
    if cli_value:
        return cli_value
    env_val = os.environ.get(env_name, "").strip()
    return env_val or None


def _autodiscover_kaggle_paths() -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort path discovery for Kaggle notebooks if user runs script with no args.
    Looks for common file names under /kaggle/input.
    """
    root = Path("/kaggle/input")
    if not root.exists():
        return None, None

    train_candidates = []
    eval_candidates = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name in {"train_tokens.txt", "train_tokens.json", "train_ids.txt", "train_ids.json"}:
            train_candidates.append(str(p))
        if name in {"eval_tokens.txt", "eval_tokens.json", "valid_tokens.txt", "val_tokens.txt", "eval_ids.txt"}:
            eval_candidates.append(str(p))

    train_path = train_candidates[0] if train_candidates else None
    eval_path = eval_candidates[0] if eval_candidates else None
    return train_path, eval_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    cfg.train_path = _resolve_path(cfg.train_path, "TRAIN_PATH")
    cfg.eval_path = _resolve_path(cfg.eval_path, "EVAL_PATH")
    if not cfg.train_path or not cfg.eval_path:
        auto_train, auto_eval = _autodiscover_kaggle_paths()
        cfg.train_path = cfg.train_path or auto_train
        cfg.eval_path = cfg.eval_path or auto_eval
    if not cfg.train_path or not cfg.eval_path:
        raise ValueError(
            "Missing input paths. Provide --train_path and --eval_path, "
            "or set TRAIN_PATH and EVAL_PATH environment variables."
        )

    t0 = time.time()
    train_toks = load_token_ids(cfg.train_path, cfg.max_train_tokens)
    eval_toks = load_token_ids(cfg.eval_path, cfg.max_eval_tokens)

    if len(train_toks) < 4 or len(eval_toks) < 4:
        raise ValueError("Need at least 4 tokens in train/eval.")

    model = ProSparseBLModel(cfg)

    t_fit0 = time.time()
    model.fit(train_toks)
    t_fit = time.time() - t_fit0

    t_eval0 = time.time()
    metrics = model.evaluate(eval_toks)
    t_eval = time.time() - t_eval0

    out = {
        "config": asdict(cfg),
        "metrics": metrics,
        "timing_sec": {
            "fit": t_fit,
            "eval": t_eval,
            "total": time.time() - t0,
        },
        "counts": {
            "train_tokens": int(len(train_toks)),
            "eval_tokens": int(len(eval_toks)),
            "pair_experts": int(len(model.bl.pair_experts) if model.bl else 0),
            "unigram_experts": int(len(model.bl.unigram_experts) if model.bl else 0),
            "has_global_expert": bool(model.bl.global_expert is not None if model.bl else False),
        },
    }

    out_json = os.path.join(cfg.output_dir, "pro_run_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
