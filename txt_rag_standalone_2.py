# -*- coding: utf-8 -*-
"""
Build & query a RAG index directly from plain-text files exported by PDF->TXT step.
Plus: lightweight knowledge-guided rule suggestion for physiologic thresholds.

Usage:

  # 1) Build index
  python txt_rag_standalone_2.py build --txt_dir txt_out --store store_txt_rag --dense

  # 2) Ad-hoc query
  python txt_rag_standalone_2.py query --store store_txt_rag "single ventricle infant low NIRS threshold"

  # 3) Suggest candidate knowledge rules
  python txt_rag_standalone_2.py rules --store store_txt_rag --out knowledge_rules.json

Dependencies:
  pip install numpy scipy scikit-learn
  pip install sentence-transformers
"""

from __future__ import annotations
import os, re, json, argparse, time, pickle, math, dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# -------- dense embeddings --------
_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# ============================= utilities =============================
def _now_year() -> int:
    return 2025

def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================= config =============================
@dataclass
class RetrievalConfig:
    # TF-IDF
    token_pattern: str = r"(?u)\b[\w/%.+\-]+\b"  # keep PaO2/FiO2, SpO2%.
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.9
    max_features: int = 60000
    sublinear_tf: bool = True
    norm: str = "l2"

    # Dense embeddings
    use_dense: bool = True
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_batch_size: int = 16
    normalize_dense: bool = True

    # Reciprocal RAG fusion
    rrf_k: int = 60
    variants_topk: int = 50

    # Adaptive-k cutoff
    adaptive_gamma: float = 0.20
    min_keep: int = 5
    max_keep: int = 20

    # MMR compression
    mmr_lambda: float = 0.7
    mmr_top_m: int = 8

    # Score fusion
    alpha_lex: float = 0.60
    beta_dense: float = 0.30
    lambda_prior: float = 0.10

    # Evidence priors
    recency_tau: float = 6.0  # years; exponential decay for older docs
    doc_type_weight: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "guideline": 1.0,
            "consensus": 1.0,
            "review": 0.7,
            "trial": 0.6,
            "cohort": 0.5,
            "case": 0.3,
            "other": 0.4,
        }
    )
    pediatric_bonus: float = 0.20
    sv_bonus: float = 0.20

    # Query expansion (domain synonyms)
    synonyms: Dict[str, List[str]] = dataclasses.field(
        default_factory=lambda: {
            "oxygenation index": ["OI", "PaO2/FiO2", "PF ratio", "oxygenation"],
            "ecmo weaning": ["decannulation", "wean from ECMO", "discontinuation"],
            "indications": ["criteria", "thresholds", "initiation"],
            "lactate": ["serum lactate"],
            "nirs": ["near-infrared spectroscopy", "cerebral oximetry"],
            "single ventricle": ["single-ventricle physiology", "SV physiology"],
            "neonate": ["neonatal", "newborn", "infant"],
        }
    )

    # Knowledge-rule suggestion: what to scan for
    knowledge_terms: Dict[str, Dict[str, Any]] = dataclasses.field(
        default_factory=lambda: {
            # rule_name: config
            "NIRS_low": {
                "keywords": ["nirs", "near-infrared", "cerebral oximetry"],
                "direction": "low",
                "unit": "%",
                "min_allowed": 30,
                "max_allowed": 80,
            },
            "O2_sat_low": {
                "keywords": ["spo2", "oxygen saturation", "o2 sat"],
                "direction": "low",
                "unit": "%",
                "min_allowed": 60,
                "max_allowed": 100,
            },
            "Lactate_high": {
                "keywords": ["lactate", "serum lactate"],
                "direction": "high",
                "unit": "mmol/L",
                "min_allowed": 1.0,
                "max_allowed": 15.0,
            },
            "SBP_low": {
                "keywords": ["systolic blood pressure", "sbp"],
                "direction": "low",
                "unit": "mmHg",
                "min_allowed": 30,
                "max_allowed": 90,
            },
            "MVO2_sats_low": {
                "keywords": ["svo2", "mixed venous", "mvo2"],
                "direction": "low",
                "unit": "%",
                "min_allowed": 30,
                "max_allowed": 90,
            },
        }
    )


# ============================= retriever =============================
class Retriever:
    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=cfg.token_pattern,
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            max_features=cfg.max_features,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=cfg.sublinear_tf,
            norm=cfg.norm,
        )
        self.X: Optional[sparse.csr_matrix] = None
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self._use_dense = bool(cfg.use_dense and _HAS_ST)
        self._dense_model: Optional["SentenceTransformer"] = None
        self.E: Optional[np.ndarray] = None  # (N, D)

    # ---------- Fit / Index ----------
    def fit(self, docs: List[str], metas: List[Dict[str, Any]]) -> "Retriever":
        t0 = time.time()
        self.texts = list(docs)
        self.metas = [dict(m) for m in metas]
        self.X = self.vectorizer.fit_transform(self.texts)
        print(
            f"[fit] TF-IDF docs={len(self.texts)} "
            f"vocab={len(self.vectorizer.vocabulary_)} "
            f"time={time.time()-t0:.2f}s"
        )

        if self._use_dense:
            try:
                t1 = time.time()
                self._dense_model = SentenceTransformer(self.cfg.dense_model)
                self.E = (
                    self._dense_model.encode(
                        self.texts,
                        normalize_embeddings=self.cfg.normalize_dense,
                        batch_size=self.cfg.dense_batch_size,
                        show_progress_bar=False,
                    )
                    .astype("float32")
                )
                print(
                    f"[fit] Dense embeddings shape={self.E.shape} "
                    f"time={time.time()-t1:.2f}s"
                )
            except Exception as e:
                print("[fit] dense disabled:", e)
                self._use_dense = False
                self.E = None
        return self

    # ---------- Priors ----------
    def _compute_priors(self) -> np.ndarray:
        now = _now_year()
        w = np.zeros(len(self.metas), dtype=float)
        for i, m in enumerate(self.metas):
            dt = str(m.get("doc_type", "other")).lower()
            w_dt = self.cfg.doc_type_weight.get(dt, self.cfg.doc_type_weight["other"])
            yr = m.get("year")
            if isinstance(yr, (int, float)) and yr > 0:
                age = max(0.0, now - float(yr))
                w_rec = math.exp(-age / max(1e-6, self.cfg.recency_tau))
            else:
                w_rec = 0.5

            bonus = 0.0
            if bool(m.get("pediatric")):
                bonus += self.cfg.pediatric_bonus
            if bool(m.get("single_ventricle")):
                bonus += self.cfg.sv_bonus

            w[i] = w_dt * w_rec + bonus
        return _minmax01(w)

    # ---------- Single variant scoring ----------
    def _score_single_query(
        self, query: str, topk: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.X is not None

        # Lexical (TF-IDF)
        q_vec = self.vectorizer.transform([query])
        slex = (self.X @ q_vec.T).toarray().ravel()
        slex = _minmax01(slex)

        # Dense
        if self._use_dense and self._dense_model is not None and self.E is not None:
            qvec = self._dense_model.encode(
                [query],
                normalize_embeddings=self.cfg.normalize_dense,
                show_progress_bar=False,
            )[0]
            sdense = np.clip(
                self.E @ qvec / (np.linalg.norm(qvec) + 1e-9), -1, 1
            )
            sdense = _minmax01(sdense)
        else:
            sdense = np.zeros_like(slex)

        # Prior
        sprior = self._compute_priors()

        # Fused score
        fused = (
            self.cfg.alpha_lex * slex
            + self.cfg.beta_dense * sdense
            + self.cfg.lambda_prior * sprior
        )
        idx = np.argsort(-fused)[: max(topk, self.cfg.max_keep)]
        return idx, fused[idx]

    # ---------- Query variants + RRF fusion ----------
    def search(self, query: str, max_return: Optional[int] = None) -> List[Dict[str, Any]]:
        variants = self._query_variants(query)
        rrf_scores = np.zeros(len(self.texts), dtype=float)

        for qv in variants:
            idx, _ = self._score_single_query(qv, topk=self.cfg.variants_topk)
            for rank, doc_id in enumerate(idx, start=1):
                rrf_scores[doc_id] += 1.0 / (self.cfg.rrf_k + rank)

        order = np.argsort(-rrf_scores)
        keep = self._adaptive_cut(rrf_scores[order])
        order = order[:keep]
        final_ids = self._mmr_compress(order, target_m=self.cfg.mmr_top_m)

        rows: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(final_ids, start=1):
            rows.append(
                {
                    "rank": i,
                    "doc_id": int(doc_id),
                    "text": self.texts[doc_id],
                    "meta": dict(self.metas[doc_id]),
                    "score_rrf": float(rrf_scores[doc_id]),
                }
            )
        if max_return is not None:
            rows = rows[:max_return]
        return rows

    # ---------- Adaptive-k ----------
    def _adaptive_cut(self, sorted_scores: np.ndarray) -> int:
        arr = np.asarray(sorted_scores, dtype=float)
        n = len(arr)
        if n == 0:
            return 0
        if np.all(arr <= 0):
            return max(self.cfg.min_keep, 1)

        # Look for first large relative drop
        for i in range(min(n - 1, self.cfg.max_keep * 2)):
            a, b = arr[i], arr[i + 1]
            if a <= 0:
                break
            rel = (a - b) / max(a, 1e-9)
            if rel >= self.cfg.adaptive_gamma and i + 1 >= self.cfg.min_keep:
                return min(i + 1, self.cfg.max_keep)
        return min(max(self.cfg.min_keep, n), self.cfg.max_keep)

    # ---------- MMR ----------
    def _mmr_compress(self, doc_ids: np.ndarray, target_m: int) -> List[int]:
        ids = list(map(int, doc_ids))
        if len(ids) <= target_m:
            return ids

        # choose similarity backend
        if self._use_dense and self.E is not None:
            M = self.E[ids]

            def sim_i_j(i, j):
                return _cosine(M[i], M[j])
        else:
            Xsub = self.X[ids].astype("float32")

            def sim_i_j(i, j):
                vi = Xsub[i]
                vj = Xsub[j]
                num = (vi @ vj.T).toarray().ravel()[0]
                den = (
                    np.linalg.norm(vi.toarray())
                    * np.linalg.norm(vj.toarray())
                    + 1e-9
                )
                return float(num / den)

        # approximate relevance: decaying weights
        rrf = np.linspace(1.0, 0.7, num=len(ids))

        selected: List[int] = []
        cand = list(range(len(ids)))
        selected.append(cand.pop(0))
        while cand and len(selected) < target_m:
            best, best_score = None, -1e9
            for ci in cand:
                max_sim = (
                    max(sim_i_j(ci, sj) for sj in selected)
                    if selected
                    else 0.0
                )
                score = (
                    self.cfg.mmr_lambda * rrf[ci]
                    - (1 - self.cfg.mmr_lambda) * max_sim
                )
                if score > best_score:
                    best_score, best = score, ci
            selected.append(best)
            cand.remove(best)

        return [ids[i] for i in selected]

    # ---------- Query variants ----------
    def _query_variants(self, query: str) -> List[str]:
        q = query.strip()
        v = [q]
        q_low = q.lower()

        # synonym-based expansions
        for k, exps in self.cfg.synonyms.items():
            if k in q_low:
                for e in exps:
                    v.append(q_low.replace(k, e))

        # step-down (remove stop-like words)
        v.append(self._stepdown(q_low))

        # deduplicate
        seen, uniq = set(), []
        for s in v:
            s = re.sub(r"\s+", " ", s).strip()
            if s and s not in seen:
                uniq.append(s)
                seen.add(s)
        return uniq[:6]

    @staticmethod
    def _stepdown(q: str) -> str:
        toks = re.findall(r"[\w/%.+\-]+", q)
        drop = {
            "the",
            "a",
            "an",
            "about",
            "for",
            "of",
            "to",
            "and",
            "or",
            "in",
            "on",
            "with",
            "without",
            "from",
            "by",
        }
        keep = [t for t in toks if t not in drop and len(t) > 1]
        return " ".join(keep)


# ============================= TXT corpus building =============================
YEAR_RX = re.compile(r"(19|20)\d{2}")
DOC_TYPE_WEIGHTED_KEYS = {
    "guideline": ["guideline", "consensus", "practice", "recommendation"],
    "consensus": ["consensus"],
    "review": ["review", "systematic"],
    "trial": ["randomized", "trial", "rct"],
    "cohort": ["cohort", "registry", "retrospective", "prospective"],
    "case": ["case report", "case series"],
}
PEDIATRIC_KEYS = [
    "pediatric",
    "paediatric",
    "neonat",
    "newborn",
    "infant",
    "children",
    "child",
]
SV_KEYS = [
    "single-ventricle",
    "single ventricle",
    "sv physiology",
    "hlhs",
    "glenn",
    "fontan",
]
PAGE_RX = re.compile(r"^\[PAGE\s+(\d+)\]\s*$")


def guess_year(name_or_text: str) -> Optional[int]:
    m = YEAR_RX.search(name_or_text or "")
    return int(m.group(0)) if m else None


def guess_doc_type(name: str, text_head: str) -> str:
    s = (name + " " + text_head).lower()
    for dt, keys in DOC_TYPE_WEIGHTED_KEYS.items():
        if any(k in s for k in keys):
            return dt
    return "other"


def has_pediatric(text: str) -> bool:
    s = text.lower()
    return any(k in s for k in PEDIATRIC_KEYS)


def has_sv(text: str) -> bool:
    s = text.lower()
    return any(k in s for k in SV_KEYS)


def split_by_pages(txt: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    cur_num, cur_buf = None, []
    for line in txt.splitlines():
        m = PAGE_RX.match(line.strip())
        if m:
            if cur_num is not None:
                pages.append((cur_num, "\n".join(cur_buf).strip()))
            cur_num = int(m.group(1))
            cur_buf = []
        else:
            cur_buf.append(line)
    if cur_num is not None:
        pages.append((cur_num, "\n".join(cur_buf).strip()))
    if not pages:
        pages = [(1, txt)]
    return pages


def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z(])", text)
    if len(parts) <= 1:
        parts = text.split(". ")
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(
    sentences: List[str], target_chars: int = 600, overlap_chars: int = 100
) -> List[str]:
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= target_chars:
            buf = buf + " " + s
        else:
            chunks.append(buf)
            prefix = buf[-overlap_chars:] if overlap_chars > 0 else ""
            buf = (prefix + " " + s).strip()
    if buf:
        chunks.append(buf)
    return chunks


def build_corpus_from_txt(
    txt_dir: Path, target_chars: int = 600, overlap_chars: int = 100
) -> Tuple[List[str], List[Dict[str, Any]]]:
    txt_paths = sorted(p for p in txt_dir.glob("*.txt"))
    assert txt_paths, f"No .txt files found in {txt_dir}"
    all_texts: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    print(f"[scan] found {len(txt_paths)} txt files.")
    for path in txt_paths:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        pages = split_by_pages(raw)
        head_text = " ".join(pg for _, pg in pages[:3])[:4000]

        meta_year = guess_year(path.name) or guess_year(head_text)
        meta_doc_type = guess_doc_type(path.name, head_text)
        meta_ped = has_pediatric(head_text)
        meta_sv = has_sv(head_text)

        chunk_count = 0
        for (pgno, pgtext) in pages:
            if not (pgtext and pgtext.strip()):
                continue
            sents = sentence_split(pgtext)
            chunks = chunk_sentences(
                sents,
                target_chars=target_chars,
                overlap_chars=overlap_chars,
            )
            for ch in chunks:
                all_texts.append(ch)
                all_metas.append(
                    {
                        "source": path.name,
                        "file_name": path.name,
                        "page": pgno,
                        "year": meta_year,
                        "section": "",
                        "doc_type": meta_doc_type,
                        "pediatric": meta_ped,
                        "single_ventricle": meta_sv,
                    }
                )
                chunk_count += 1
        print(
            f"[parse] {path.name}: pages={len(pages)} chunks={chunk_count}"
        )
    print(f"[build] total chunks={len(all_texts)}")
    return all_texts, all_metas


# ============================= knowledge-rule suggestion =============================

NUM_RX = re.compile(r"(?<![A-Za-z0-9])([-+]?\d+(?:\.\d+)?)(?![A-Za-z0-9])")

def _extract_numbers_near_keyword(
    text: str,
    keyword: str,
    window: int = 40,
) -> List[float]:
    out: List[float] = []
    tl = text.lower()
    kw = keyword.lower()
    start = 0
    while True:
        idx = tl.find(kw, start)
        if idx == -1:
            break
        a = max(0, idx - window)
        b = min(len(text), idx + len(kw) + window)
        seg = text[a:b]
        for m in NUM_RX.finditer(seg):
            try:
                val = float(m.group(1))
                out.append(val)
            except Exception:
                pass
        start = idx + len(kw)
    return out


def suggest_knowledge_rules(
    retriever: Retriever,
    topn: int = 400,
    focus_pediatric_sv: bool = True,
) -> Dict[str, Any]:
    cfg = retriever.cfg
    texts = retriever.texts
    metas = retriever.metas

    # rank chunks roughly by prior (guideline/peds/SV more important)
    priors = retriever._compute_priors()
    order = np.argsort(-priors)
    cand_ids = order[: min(topn, len(order))]

    rules_out: Dict[str, Any] = {}

    for rule_name, rule_cfg in cfg.knowledge_terms.items():
        direction = rule_cfg.get("direction", "low")
        keywords = rule_cfg.get("keywords", [])
        vmin = float(rule_cfg.get("min_allowed", -1e9))
        vmax = float(rule_cfg.get("max_allowed", 1e9))

        collected: List[float] = []

        for idx in cand_ids:
            m = metas[idx]
            if focus_pediatric_sv:
                if not (m.get("pediatric") or m.get("single_ventricle")):
                    continue

            # prioritize guideline / consensus / review
            dt = str(m.get("doc_type", "other")).lower()
            if dt not in ("guideline", "consensus", "review", "trial", "cohort"):
                continue

            txt = texts[idx]
            t_low = txt.lower()

            # quick filter: must contain at least one keyword
            if not any(kw in t_low for kw in keywords):
                continue

            # extract numbers near each keyword
            for kw in keywords:
                nums = _extract_numbers_near_keyword(txt, kw, window=64)
                for val in nums:
                    if vmin <= val <= vmax:
                        collected.append(val)

        if not collected:
            continue

        collected = sorted(collected)
        # Robust summary (median + IQR)
        arr = np.array(collected, dtype=float)
        med = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))

        if direction == "low":
            # interpret as "values below ~X concerning"
            thr = med
        elif direction == "high":
            thr = med
        else:
            thr = med

        rules_out[rule_name] = {
            "suggested_threshold": thr,
            "direction": direction,
            "unit": rule_cfg.get("unit", ""),
            "q1": q1,
            "q3": q3,
            "candidates_sorted": collected[:80],  # cap for readability
            "n_candidates": int(len(collected)),
        }

    return rules_out


# ============================= persist / load =============================
def save_index(store_dir: Path, retriever: Retriever):
    store_dir.mkdir(parents=True, exist_ok=True)

    with (store_dir / "texts.jsonl").open("w", encoding="utf-8") as f:
        for t in retriever.texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    with (store_dir / "metas.jsonl").open("w", encoding="utf-8") as f:
        for m in retriever.metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with open(store_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(retriever.vectorizer, f)

    if retriever.X is not None:
        sparse.save_npz(store_dir / "X_tfidf.npz", retriever.X)

    if getattr(retriever, "E", None) is not None:
        np.save(store_dir / "E_dense.npy", retriever.E)

    with open(store_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(retriever.cfg.__dict__, f, ensure_ascii=False, indent=2)

    print(f"[persist] saved to {store_dir}")


def load_index(store_dir: Path) -> Retriever:
    texts, metas = [], []

    with open(store_dir / "texts.jsonl", "r", encoding="utf-8") as f:
        for ln in f:
            texts.append(json.loads(ln)["text"])

    with open(store_dir / "metas.jsonl", "r", encoding="utf-8") as f:
        for ln in f:
            metas.append(json.loads(ln))

    with open(store_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    X = sparse.load_npz(store_dir / "X_tfidf.npz")

    cfg = RetrievalConfig()
    r = Retriever(cfg)
    r.texts = texts
    r.metas = metas
    r.vectorizer = vectorizer
    r.X = X

    e_path = store_dir / "E_dense.npy"
    if e_path.exists():
        try:
            r.E = np.load(e_path)
            r._use_dense = True
        except Exception:
            r.E = None
            r._use_dense = False

    print(
        f"[load] {store_dir} | docs={len(texts)} "
        f"| dense={'yes' if r.E is not None else 'no'}"
    )
    return r


# ============================= CLI =============================
def cmd_build(
    txt_dir: Path,
    store_dir: Path,
    chunk_chars: int = 600,
    overlap_chars: int = 100,
    use_dense: bool = False,
):
    t0 = time.time()
    docs, metas = build_corpus_from_txt(
        txt_dir,
        target_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )
    cfg = RetrievalConfig()
    cfg.use_dense = bool(use_dense and _HAS_ST)

    r = Retriever(cfg).fit(docs, metas)
    save_index(store_dir, r)
    print(f"[done] build in {time.time()-t0:.1f}s")


def cmd_query(store_dir: Path, query: str, k: int = 8):
    r = load_index(store_dir)
    hits = r.search(query, max_return=k)
    print(f"\nQuery: {query}\nTop-{len(hits)} results:")
    for h in hits:
        m = h["meta"]
        cite = (
            f"{m.get('file_name')} | y={m.get('year')} | "
            f"page={m.get('page')} | type={m.get('doc_type')} | "
            f"ped={m.get('pediatric')} | sv={m.get('single_ventricle')}"
        )
        print(f"\n#{h['rank']} [{cite}] (rrf={h['score_rrf']:.4f})")
        print((h["text"] or "").replace("\n", " ").strip()[:700])


def cmd_rules(store_dir: Path, out_path: Path):
    r = load_index(store_dir)
    rules = suggest_knowledge_rules(r, topn=400, focus_pediatric_sv=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    print(f"[rules] suggested knowledge rules saved to {out_path}")
    if not rules:
        print("[rules] WARNING: no candidate rules found; check corpus / filters.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    # build
    ap_b = sub.add_parser("build")
    ap_b.add_argument(
        "--txt_dir", required=True, help="folder containing .txt files"
    )
    ap_b.add_argument(
        "--store", required=True, help="output store folder"
    )
    ap_b.add_argument(
        "--chunk", type=int, default=600, help="chunk target chars"
    )
    ap_b.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="chunk overlap chars",
    )
    ap_b.add_argument(
        "--dense",
        action="store_true",
        help="enable dense embeddings if sentence-transformers is available",
    )

    # query
    ap_q = sub.add_parser("query")
    ap_q.add_argument(
        "--store", required=True, help="store folder"
    )
    ap_q.add_argument(
        "query", nargs="+", help="your query sentence"
    )
    ap_q.add_argument(
        "-k", type=int, default=8, help="top-k results to display"
    )

    # rules (knowledge-guided threshold suggestion)
    ap_r = sub.add_parser("rules")
    ap_r.add_argument(
        "--store",
        required=True,
        help="existing store folder (built via 'build')",
    )
    ap_r.add_argument(
        "--out",
        default="knowledge_rules.json",
        help="output JSON path for suggested rules",
    )

    args = ap.parse_args()

    if args.mode == "build":
        cmd_build(
            Path(args.txt_dir),
            Path(args.store),
            chunk_chars=args.chunk,
            overlap_chars=args.overlap,
            use_dense=args.dense,
        )
    elif args.mode == "query":
        q = " ".join(args.query)
        cmd_query(Path(args.store), q, k=args.k)
    elif args.mode == "rules":
        cmd_rules(Path(args.store), Path(args.out))


if __name__ == "__main__":
    main()


