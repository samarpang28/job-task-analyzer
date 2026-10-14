"""
retrieval.py - HYBRID RAG retrieval over the case-study chunks.

Three ways to search (config.RETRIEVAL_MODE picks the one the agent uses):

  "vector"  - semantic search: embeddings + cosine similarity in Chroma (Sessions 7-8)
  "keyword" - BM25 keyword search: rewards chunks that contain the query's exact words
  "hybrid"  - both, merged with Reciprocal Rank Fusion (RRF):
                 score(chunk) = sum over the two lists of 1 / (60 + rank)
              so a chunk ranked high by EITHER method rises, and one ranked high by BOTH wins.

Why hybrid? Pure cosine similarity sometimes returns chunks that are "about the same topic"
but not relevant (e.g. a contract-extraction case for "financial ratios"). Exact keyword
matching catches specific terms (ratios, delinquent, triage) that embeddings blur.
"""
import math
import pickle
import re
from collections import Counter, defaultdict

import config

STOPWORDS = set("""a an the and or of to in on for with by from at as is are was were be been being this that
these those it its into over under than then so such not no can could will would should may might must do does
did done using use used via per about across within without their them they we our you your i ai""".split())

RRF_K = 60          # standard constant for Reciprocal Rank Fusion
POOL = 20           # how many results each method contributes before fusion


def tokenize(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in STOPWORDS and len(w) > 1]


class BM25:
    """Classic Okapi BM25 keyword ranking (k1 = 1.5, b = 0.75), built from scratch."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = defaultdict(list)          # term -> [(doc index, term frequency)]
        for i, doc in enumerate(docs):
            toks = tokenize(doc)
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.n = len(docs)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, top_n: int):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda x: -x[1])[:top_n]


_bm25 = None


def _corpus(collection):
    """All chunk ids, texts and metadata, read once from Chroma and cached to disk."""
    cache = config.DATA_DIR / "bm25_corpus.pkl"
    if cache.exists():
        try:
            data = pickle.loads(cache.read_bytes())
            if len(data["ids"]) == collection.count():
                return data
        except Exception:
            pass
    ids, docs, metas = [], [], []
    total, step = collection.count(), 5000
    for offset in range(0, total, step):
        got = collection.get(limit=step, offset=offset, include=["documents", "metadatas"])
        ids += got["ids"]
        docs += got["documents"]
        metas += got["metadatas"]
    data = {"ids": ids, "docs": docs, "metas": metas}
    try:
        cache.write_bytes(pickle.dumps(data))
    except Exception:
        pass
    return data


def _keyword_index(collection):
    global _bm25
    if _bm25 is None:
        data = _corpus(collection)
        _bm25 = (BM25(data["docs"]), data)
    return _bm25


def vector_search(collection, query: str, n: int = POOL):
    res = collection.query(query_texts=[query], n_results=n)
    return [{"id": i, "doc": d, "meta": m, "similarity": round(1 - float(dist), 3)}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def keyword_search(collection, query: str, n: int = POOL):
    bm25, data = _keyword_index(collection)
    return [{"id": data["ids"][i], "doc": data["docs"][i], "meta": data["metas"][i], "bm25": round(s, 2)}
            for i, s in bm25.search(query, n)]


def search(collection, query: str, k: int = config.TOP_K, mode: str = None):
    """Top-k chunks for a query, at most one per company case."""
    mode = mode or config.RETRIEVAL_MODE
    if mode == "vector":
        ranked = vector_search(collection, query)
    elif mode == "keyword":
        ranked = keyword_search(collection, query)
    else:
        vec, kw = vector_search(collection, query), keyword_search(collection, query)
        fused, info = defaultdict(float), {}
        for source in (vec, kw):
            for rank, hit in enumerate(source, start=1):
                fused[hit["id"]] += 1.0 / (RRF_K + rank)
                info.setdefault(hit["id"], {}).update(hit)
        ranked = [dict(info[i], rrf=round(s, 4), found_by="+".join(
                      m for m, src in (("vector", vec), ("keyword", kw)) if any(h["id"] == i for h in src)))
                  for i, s in sorted(fused.items(), key=lambda x: -x[1])]
    seen, hits = set(), []
    for h in ranked:
        case = h["meta"].get("case_id")
        if case in seen:
            continue
        seen.add(case)
        hits.append(h)
        if len(hits) == k:
            break
    return hits
