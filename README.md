[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/JG5U42VD)
# Multi-Agent System for Automatic Survey

## Overview

This project is to design a multi-agent system (MAS) that answers survey questions with evidence that support the answers. The system was optimized for multiple constraints: maximizing classification accuracy, ensuring highquality supporting evidence, and minimizing both runtime and computational cost to make the system practical for
large-scale usage. The designed approach is using 4 agents: Query Expansion Agent, Retrieval Agent, Classification Agent and Aggregate Agent, where each agent handles expanding query, retrieve relevant documents from corpus,
classify documents to different options that answer the survey and aggregate to normalized and readable output respectively. The final MAS achieved JS score 0.25 and MAP 0.14 according to Kaggle.

## MAS Design
**Architecture:**  
Question → Query Expansion Agent → Retrieval Agent → Classification Agent → Aggregate Agent → Outputs
The system progress is tracked with state, which stores all intermediate and final outputs. The Query Expansion Agent
uses an LLM to generate and clean an expanded natural language query, updating State.query to improve recall and
better phrasing. The Retrieval Agent preprocesses this query and retrieves relevant documents into State.evidence
from pre-built indices. The Classification Agent constructs context from description and content fields (up to 400
tokens), including headers with title, source, and URL, and marks documents lacking sufficient evidence as
“INSUFFICIENT EVIDENCE” to reduce hallucination. Classification occurs in batches of 10 passages, producing
structured JSON results stored in State.classifications. The Aggregate Agent computes a normalized distribution over
options, ignoring insufficient-evidence passages, and stores it in State.distribution; all supporting document IDs are
saved in State.supports, while State.messages logs each agent’s progress.
Design Intuition:
Query expansion agent is designed to not only expand but also process questions that are not complete in one sentence,
query expansion can also help rephrase the question for higher quality information retrieval. The retrieval agent is to
extract all the candidate documents that are relevant to the topic of the question, which are the documents that are
more likely to be able to answer the survey questions. The classification agent is designed to analyze the opinions of
each document and classify it. The aggregation agent is to help produce the normalized distribution.

Full Details: `./reports/report.pdf`
---

## Quick start (Python 3.9, CPU)

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
````

Edit `config.yaml` if needed, then:

```bash
# 1) Build (validate / prepare artifacts)
python -m index.build config.yaml --api_key dummy_api

# 2) Run (write the CSV submission)
python -m mas_survey.run --config config.yaml --api_key dummy_api
```

**Output:** `./artifacts/submission_js.csv`

---

## File layout

This repo contains the project code, datasets, evaluation scripts, generated indexes/artifacts, and reports. The tracked layout is roughly:

```
.
├── .github/
├── artifacts/
│   ├── full_submission_js.csv
│   ├── submission_js.csv
│   ├── test_submission_js.csv
│   └── index/
│       ├── doc_id_to_meta.json
│       ├── faiss.index
│       ├── ids.json
│       ├── meta.json
│       └── whoosh/
│           ├── MAIN_71hfe86y5r215dew.seg
│           ├── MAIN_WRITELOCK
│           └── _MAIN_1.toc
├── data/
│   ├── dev/
│   │   ├── dev.json
│   │   ├── dev_groundtruth.json
│   │   └── mini_dev.json
│   ├── documents.jsonl
│   ├── id_to_embedding.npz
│   ├── mini_documents.jsonl
│   ├── mini_id_to_embeddings.npz
│   ├── test.json
│   └── ...
├── eval/
│   ├── eval_end_to_end.py
│   ├── eval_llm.py
│   ├── eval_retrieval.py
│   ├── llm_generate_probes.py
│   └── retrieve_generate_probes.py
├── index/
│   ├── __init__.py
│   ├── build.py
│   ├── index.py
│   └── __pycache__/
├── mas_survey/
│   ├── __init__.py
│   ├── agents.py
│   ├── llm_client.py
│   ├── retriever.py
│   └── run.py
├── probes/
│   ├── ground_truth_retrieval.json
│   ├── llm_probes.json
│   ├── probes_retrieval.json
│   ├── retrieval_evidence.json
│   └── retrieval_probes_label.csv
├── reports/
│   ├── end_to_end_results/
│   │   ├── test_llm_few_shot.png
│   │   ├── test_llm_max_retry.png
│   │   ├── test_llm_with_option_parsing.png
│   │   ├── test_llm_without_option_parse.png
│   │   └── test_retrieval_bm25.png
│   ├── ir_results/
│   │   ├── bm25_log.png
│   │   ├── hybrid_with_mmr_log.png
│   │   ├── hybrid_without_mmr_log.png
│   │   ├── retrieval_results_bm25.json
│   │   ├── retrieval_results_fused.json
│   │   ├── retrieval_results_hybrid.json
│   │   ├── retrieval_results_vector.json
│   │   └── vector_log.png
│   └── llm_results/
│       ├── base.json
│       ├── base_rate.png
│       ├── few_shot.json
│       ├── few_shot.png
│       ├── option_parsing.json
│       ├── option_parsing_rate.png
│       ├── retry.json
│       └── retry_rate.png
├── .gitignore
├── config.yaml
├── prompts.py
├── README.md
├── requirements.txt
├── submit.py
├── utils.py
```

You provide the data files (mini or full):

* `data/mini_documents.jsonl` — one JSON per line with fields like `"id"`, `"title"`, `"description"`, `"post_content"`, `"content"`, …
* `data/dev/mini_dev.json` — a single JSON object keyed by the **full question string**; each value has a `distribution` stub (option keys with zeros) and an empty `supports` list.
* Full `documents.jsonl` (and embeddings) are on the Kaggle dataset;  
  they are **too large for this repo**.  
  **Do not commit large artifacts** (keep repo size <10 MB).  
  Always add them to `.gitignore`.

---

## CSV schema reminder

* Header **exactly**: `question,distribution,supports`
* UTF-8; follow CSV quoting (double quotes inside a quoted cell are doubled).
* Probabilities **≥ 0** and sum to **1.0 ± 1e-6**.

---

## FAQ

**Can I use GPUs?**
No. CPU-only.

**Can I add libraries?**
Yes, within course constraints. Start here, then add BM25 / FAISS / scikit-learn as needed.
