"""BM25 检索基线：与实验 A 同一金标准、同两种分片策略，用于与向量检索对照。纯本地计算。"""
import json
import time

import jieba
from rank_bm25 import BM25Okapi

from run_eval import KB_DIR, ROOT, load_json, load_kb_chunks


def tokenize(text):
    return [t for t in jieba.cut_for_search(text) if t.strip()]


def rank_scores(scores):
    """按分数降序排名；同分时保留语料中的原始顺序。"""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def main():
    gold = load_json("gold_qa.json")["items"]
    result = {}
    for strategy in ["whole", "heading"]:
        chunks = load_kb_chunks(strategy)
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        r1 = r3 = cov = 0
        for item in gold:
            scores = bm25.get_scores(tokenize(item["question"]))
            idx = rank_scores(scores)[:3]
            docs = [chunks[i]["doc"] for i in idx]
            ok_docs = {item["expected_doc"], *item.get("acceptable_docs", [])}
            if docs[0] in ok_docs:
                r1 += 1
            if ok_docs & set(docs):
                r3 += 1
            joined = " ".join(chunks[i]["text"] for i in idx).lower()
            kws = item["answer_keywords"]
            cov += sum(1 for kw in kws if kw.lower() in joined) / len(kws)
        n = len(gold)
        result[strategy] = {
            "chunks": len(chunks),
            "recall@1": round(r1 / n, 3),
            "recall@3": round(r3 / n, 3),
            "answer_coverage": round(cov / n, 3),
            "n": n,
        }
        print(f"BM25 {strategy}: {result[strategy]}")
    result["run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = ROOT / "results_bm25.json"
    out.write_text(json.dumps({"retrieval_bm25": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out.name}")


if __name__ == "__main__":
    main()
