import json, os, time
from mas_survey.agents import State, RetrievalAgent
from mas_survey.retriever import Retriever
from mas_survey.llm_client import LLMClient
from utils import set_seed

set_seed(42)

import csv, json

with open("probes/ground_truth_retrieval.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

with open("probes/probes_retrieval.json", "r", encoding="utf-8") as f:
    probes = json.load(f)


client = LLMClient(api_key= os.getenv("TOGETHER_API_KEY"))
meta_path="artifacts/index/doc_id_to_meta.json"
retriever = Retriever(
        vec_store_path="data/id_to_embedding.npz",
        index_dir="artifacts/index/whoosh",
        faiss_index_path="artifacts/index/faiss.index",
        faiss_ids_path="artifacts/index/ids.json"
    )

def precision_at_k(retrieved, labels, k):
    return sum(labels[:k]) / k

def average_precision(retrieved, labels):
    score = 0.0
    num_relevant = 0
    for i, label in enumerate(labels, start=1):
        if label == 1:
            num_relevant += 1
            score += num_relevant / i
    return score / max(1, sum(labels))

def evaluate_method(method):
    retrieval_agent = RetrievalAgent(client, 10, retriever, meta_path, top_k=10, method=method)

    results = {}

    for probe in probes:
        qid = probe["id"]
        query = probe["query"]
        state = State(query=query)
        state = retrieval_agent.run(state)
        retrieved_ids = [doc["rid"] for doc in state.evidence]

        # Get labels for retrieved docs based on current query
        labels = [1 if rid in ground_truth[str(qid)]["relevant"] else 0 for rid in retrieved_ids]
        ap = average_precision(retrieved_ids, labels)
        prec_k = precision_at_k(retrieved_ids, labels, k=10)

        results[qid] = {
            "query": query,
            "precision@10": prec_k,
            "average_precision": ap,
            "retrieved_ids": retrieved_ids,
            "labels": labels
        }

    return results

method = "fused"
start_time = time.time()
results = evaluate_method(method)
end_time = time.time()
map_score = sum(r["average_precision"] for r in results.values()) / len(results)
print(f"MAP for method {method}: {map_score:.4f}")
print("Retrieval Time:", end_time-start_time)

with open(f"reports/ir_results/retrieval_results_{method}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)