import json, os, csv

from mas_survey.agents import RetrievalAgent, State
from mas_survey.retriever import Retriever
from mas_survey.llm_client import LLMClient

from typing import DefaultDict

retrieval_probes = [
    {"id": 1, "query": "Do you have separate accounts or joint accounts?"},
    {"id": 2, "query": "Do you like sandwich?"},
    {"id": 3, "query": "What kind of impacts do you think immigrants made to the country?"},
    {"id": 4, "query": "Have you ever experience something bad in the country?"},
    {"id": 5, "query": "Are you interested in other cultures from immigrants?"},
    {"id": 6, "query": "Which country or places would you like to travel?"},
    {"id": 7, "query": "What benefits do doing sports have?"},
    {"id": 8, "query": "Do you like eating fast food?"},
    {"id": 9, "query": "Do you agree to give priorities to this group of people?"},
    {"id": 10, "query": "How do you think about the public transport system?"},
    {"id": 11, "query": "Do you face any financial problems?"},
    {"id": 12, "query": "Do you like your job?"},
    {"id": 13, "query": "What do you think about tipping culture"},
    {"id": 14, "query": "How is your relationship with family?"},
    {"id": 15, "query": "What is your thoughts of people from Asia?"},
]

client = LLMClient(api_key= os.getenv("TOGETHER_API_KEY"))
meta_path="artifacts/index/doc_id_to_meta.json"
retriever = Retriever(
        vec_store_path="data/id_to_embedding.npz",
        index_dir="artifacts/index/whoosh",
        faiss_index_path="artifacts/index/faiss.index",
        faiss_ids_path="artifacts/index/ids.json"
    )

methods = ["hybrid", "bm25", "vector"]

evidence = {}

for probe in retrieval_probes:
    qid = probe["id"]
    query = probe["query"]
    evidence[qid] = {"query": query, "docs": []}

    for method in methods:
        retrieval_agent = RetrievalAgent(client, 10, retriever, meta_path, top_k=10, method=method)
        state = State(query=query)
        state = retrieval_agent.run(state)

        existing_ids = {doc["rid"] for doc in evidence[qid]["docs"]}
        for doc in state.evidence:
            if doc["rid"] not in existing_ids:
                evidence[qid]["docs"].append(doc)


with open("probes/retrieval_probes_unlabel.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(["query_id", "query", "doc_id", "title", "content", "score", "relevance"])
    
    # Write rows
    for qid, data in evidence.items():
        query_text = data["query"]
        for doc in data["docs"]:
            writer.writerow([
                qid,
                query_text,
                doc.get("rid", ""),
                doc.get("title", ""),
                doc.get("content", ""),
                doc.get("score", 0),
                ""  
            ])

ground_truth = DefaultDict(lambda: {"relevant": []})

with open("probes/retrieval_probes_unlabel.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row["relevance"]) == 1:
            ground_truth[row["query_id"]]["relevant"].append(row["doc_id"])

with open("probes/ground_truth_retrieval.json", "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

with open("probes/probes_retrieval.json", "w", encoding="utf-8") as f:
    json.dump(retrieval_probes, f, indent=2, ensure_ascii=False)