"""检索三路对照：向量（bge-m3）/ BM25 / RRF 混合，同一金标准与分片策略（heading）。"""
import json
import time

import numpy as np
from rank_bm25 import BM25Okapi

import run_eval as R
from run_bm25_baseline import rank_scores, tokenize


def evaluate(rank_fn, chunks, gold):
    r1 = r3 = cov = 0
    for i, item in enumerate(gold):
        idx = rank_fn(i)[:3]
        docs = [chunks[j]["doc"] for j in idx]
        ok_docs = {item["expected_doc"], *item.get("acceptable_docs", [])}
        if docs[0] in ok_docs:
            r1 += 1
        if ok_docs & set(docs):
            r3 += 1
        joined = " ".join(chunks[j]["text"] for j in idx).lower()
        kws = item["answer_keywords"]
        cov += sum(1 for kw in kws if kw.lower() in joined) / len(kws)
    n = len(gold)
    return {"recall@1": round(r1 / n, 3), "recall@3": round(r3 / n, 3), "answer_coverage": round(cov / n, 3), "n": n}


def rrf_scores(rankings, k=60):
    """按标准的一基排名计算 Reciprocal Rank Fusion 分数。"""
    scores = {}
    for ordering in rankings:
        for rank, chunk_id in enumerate(ordering, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def main():
    R.init_embedding_client()
    gold = R.load_json("gold_qa.json")["items"]
    chunks = R.load_kb_chunks("heading")

    mat = R.embed([c["text"] for c in chunks])
    qmat = R.embed([it["question"] for it in gold])
    vec_sims = qmat @ mat.T

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    bm_scores = np.array([bm25.get_scores(tokenize(it["question"])) for it in gold])

    def vec_rank(i):
        return list(np.argsort(-vec_sims[i]))

    def bm_rank(i):
        return rank_scores(bm_scores[i])

    K = 60

    def rrf_rank(i):
        v, b = vec_rank(i), bm_rank(i)
        score = rrf_scores([v, b], k=K)
        return sorted(score, key=lambda c: (-score[c], c))

    result = {
        "strategy": "heading",
        "vector": evaluate(vec_rank, chunks, gold),
        "bm25": evaluate(bm_rank, chunks, gold),
        "hybrid_rrf": evaluate(rrf_rank, chunks, gold),
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "_meta": {
            "embed_model": R.EMBED_MODEL,
            "chunks": len(chunks),
            "bm25_tokenizer": "jieba.cut_for_search",
            "rrf_k": K,
            "rrf_rank_base": 1,
        },
    }
    for k in ["vector", "bm25", "hybrid_rrf"]:
        print(f"{k:10s}: {result[k]}")
    out = R.ROOT / "results_hybrid.json"
    out.write_text(json.dumps({"retrieval_hybrid": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out.name}")


if __name__ == "__main__":
    main()
