from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from collections import Counter
from pathlib import Path

from .llm_client import LLMClient
from .retriever import Retriever
from prompts import CLASSIFY_SYSTEM, CLASSIFY_SCHEMA, CLASSIFY_SCHEMA_PARSED_OPTION, CLASSIFY_SYSTEM_PARSED_OPTION, CLASSIFY_SYSTEM_FEW_SHOT
from utils import split_sentences, clean_text, estimate_tokens, set_seed

set_seed(42)

import numpy as np
import json, re

# -------------------- Shared State --------------------
@dataclass
class State:
    query: str
    options: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list) 
    classifications: List[Dict[str, Any]] = field(default_factory=list)
    distribution: Dict[str, float] = field(default_factory=dict)
    supports: List[str] = field(default_factory=list)  
    messages: List[Dict[str, Any]] = field(default_factory=list)

class Agent:
    name: str = "Agent"
    def __init__(self, client: LLMClient, support_k: int = 100):
        self.client = client
        self.support_k = support_k
    def run(self, state: State):
        raise NotImplementedError

# -----------------------------Expand Query Agent ----------------------------------------------------------------------------
class ExpandQueryAgent(Agent):
    name = "ExpandQuery"
    def __init__(self, client: LLMClient, support_k):
        super().__init__(client, support_k) 

    def run(self, state: State):
        query = state.query
        
        # prompts
        system = """You are a helpful assistant that helps expand queries for high quality retrieval. Given the query below, 
            expand and rephrase the query to a natural language query. Return ONLY the the expanded query. NO explaination, 
            commentary, markdown, or any extra text\n\n"""
        user = f"Query:\n{query}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # expanded query
        expanded = self.client.chat(messages)

        # strip in case model returns text other than query
        if isinstance(expanded, str):
            expanded = expanded.strip()
            expanded = expanded.split("\n")[0]
            expanded = expanded.strip('"“”')
            
        # if nothing return, set expanded to original query
        if not expanded:
            expanded = query

        # replace query to expanded query
        state.query = expanded

        state.messages.append({
            "agent": self.name,
            "msg": f"Expanded query from '{query}' to '{expanded}'"
        })
        
        return state


# -----------------------------Retrieval Agent ----------------------------------------------------------------------------

class RetrievalAgent(Agent):
    name = "Retrieval"
    def __init__(self, client: LLMClient, support_k, retriever: Retriever, meta_path: str, top_k=100, method="bm25"):
        super().__init__(client, support_k) 
        self.top_k = top_k 
        with open(meta_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)  
        self.meta_lookup = {str(k): v for k, v in raw_meta.items()}
        self.retriever = retriever
        self.method = method

    def retrieve(self, query, top_pool=200, k=120):
        if self.method == "bm25":
            results = self.retriever.retrieve_bm25(query, k)
        elif self.method == "vector":
            results = self.retriever.vector_search_topk(self.client, query, k)
        elif self.method == "fused":
            results = self.retriever.retrive_rrf_fuse(self.client, query, k=150, topn=k)
        elif self.method == "hybrid":
            results = self.retriever.retrieve_hybrid(self.client, query, top_pool, k)
        else:
            raise ValueError(f"Unknown method {self.method}")

        return results

    def run(self, state: State) -> State:
        query = state.query
        top_docs = self.retrieve(query, top_pool=200, k=self.top_k)
        # deduplicate
        top_docs = list(dict(top_docs).items())
        # get 100 candidate docs only
        top_docs = top_docs[:self.support_k] 
        # pad doc id if not enough 100
        if len(top_docs) < self.support_k:
            padding = self.support_k - len(top_docs)
            selected_ids = {id for id, _ in top_docs}

            padding_docs = [
                (id, 0.0) 
                for id in self.meta_lookup 
                if id not in selected_ids
            ][:padding]  

            top_docs.extend(padding_docs)

        state.supports = [id for id, _ in top_docs]

        # Build evidence list - retrieved doc_id and get its contents
        evidence = []
        for rid, score in top_docs:
            doc = self.meta_lookup[rid]
            evidence.append({
                "rid": rid,
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "description": doc.get("description", ""),
                "post_content": doc.get("post_content", ""),
                "content": doc.get("content", ""),
                "score": score
            })

        # Store results
        state.evidence = evidence
        state.messages.append({
            "agent": self.name,
            "msg": f"Retrieved {len(top_docs)} docs using {self.method} for query: {query}"
        })

        return state


# -----------------------------Classification Agent ----------------------------------------------------------------------------

class ClassifyAgent(Agent):
    name = "Classify"
    def __init__(self, client: LLMClient, support_k):
        super().__init__(client, support_k)

    def parse_options(self, options: List[str]) -> Dict[str, str]:
        return {str(i+1): opt for i, opt in enumerate(options, start=1)}

    def build_context(self, doc: dict, max_tokens: int = 400) -> str:
        if doc.get("description"):
            text = doc["description"] + " " + doc.get("content", "")
        else:
            text = doc.get("content", "")
            
        text = clean_text(text)
        header = f"{doc.get('title', '')} | {doc.get('source', 'n/a')} | {doc.get('url', 'n/a')}"
        return f"{header}\n{text[:max_tokens]}"
        
    def run(self, state: State, batch_size: int = 10, option_parsing: bool = True, max_retry:int = 3, few_shot:bool = True):
        query = state.query
        options = state.options
        docs = state.evidence

        # parse options with numbers
        if option_parsing:
            parsed_options = self.parse_options(options)
            options_text = "\n".join([f"{i}. {opt}" for i, opt in parsed_options.items()])
            schema = CLASSIFY_SCHEMA_PARSED_OPTION
            if few_shot:
                system = CLASSIFY_SYSTEM_FEW_SHOT
            else: 
                system = CLASSIFY_SYSTEM_PARSED_OPTION
        else:
            options_text = "\n".join(options)
            system = CLASSIFY_SYSTEM
            schema = CLASSIFY_SCHEMA

        classifications = []

        # ask model by batches
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]

            passages = "\n\n".join(
                [f"{j+1}. [ID: {doc['rid']}]\n{self.build_context(doc)}" for j, doc in enumerate(batch)]
            )

            payload = f"Question:{query}\nOptions:\n{options_text}\n\nPassages:\n{passages}"
            out = self.client.chat_json(
                system,
                payload,
                schema_hint=schema,
                max_tokens=3500,
                max_retry=max_retry
            )

            results = out.get("results", [])
            if results:
                for r in results:
                    option = r.get("option")
                    if option_parsing:
                        if option is not None:
                            match = re.search(r"\d+", option)
                            option_num = match.group() if match else option
                        else:
                            # missing option
                            option_key = r.get("option_text", "unknown")
                        # get actual option text
                        r["option_text"] = parsed_options.get(option_num, option_num)
                    else:
                        # 
                        if "option_text" not in r and option is not None:
                            r["option_text"] = option
                        elif "option_text" not in r:
                            r["option_text"] = "unknown" # missing option
                classifications.extend(results)
            else:
                print("Warning: model returned invalid JSON:", out)
                        
        state.classifications = classifications
        state.messages.append({
            "agent": self.name,
            "msg": f"Classified {len(docs)} passages with {len(classifications)} valid JSON results"
        })
        return state
    
# ----------------------------- Aggregate Agent ----------------------------------------------------------------------------
#      
class AggregateAgent(Agent):
    name = "Aggregate"
    def __init__(self, client, support_k):
        super().__init__(client, support_k)

    def run(self, state: State):
        options = state.options
        classifications = state.classifications
        
        option_count = Counter(c["option_text"] for c in classifications)
        distribution = {option: option_count.get(option, 0) for option in options}
        total = sum(distribution.values())
        if total == 0:
            distribution = {option: 1/len(options) for option in options}
        else:
            # normalize
            distribution = {option: v/total for option, v in distribution.items()}

        state.distribution = distribution
        state.messages.append({
            "agent": self.name,
            "msg": f"Aggregated classification into normalized distribution: {state.distribution}"
        })

        return state


