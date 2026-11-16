from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import re, json, asyncio
import numpy as np
import os

from utils import query_processor, set_seed
from .llm_client import LLMClient
try:
    from retry.api import retry_call
except Exception:
    retry_call = None

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

set_seed(42)

VEC_BACKEND = "faiss" if HAS_FAISS else "numpy"

class Retriever:
    def __init__(self, vec_store_path, index_dir, faiss_index_path, faiss_ids_path):
        self.vec_store_path = Path(vec_store_path)
        self.index_dir = Path(index_dir)
        self.faiss_index_path = Path(faiss_index_path)
        self.faiss_ids_path = Path(faiss_ids_path)

    def retrieve_bm25(self, query: str, k: int = 8) -> List[Tuple[str, float]]:
        try:
            from whoosh import scoring, index
            from whoosh.qparser import MultifieldParser, OrGroup
            ix = index.open_dir(str(self.index_dir))
            with ix.searcher(weighting=scoring.BM25F(B=0.75, K1=1.2)) as searcher:
                # Lenient OR query across fields; strip punctuation to avoid parser surprises
                parser = MultifieldParser(["text"], schema=ix.schema, group=OrGroup.factory(0.0))
                cleaned = re.sub(r"[^\w\s]", " ", query).lower()
                q = parser.parse(cleaned)
                results = searcher.search(q, limit=k)
                out = []
                for r in results:
                    rid = f"{r['doc_id']}"
                    out.append((rid, float(r.score)))
                return out
        except Exception as e:
            print("Whoosh search failed.", e)
            return []

    def retrive_emb(self, query_vec: np.ndarray,k: int = 50) -> List[Tuple[str, float]]:
        try:
            import faiss
        except Exception as e:
            raise RuntimeError("FAISS not installed. `pip install faiss-cpu`") from e
        if not self.faiss_index_path.exists() or not self.faiss_ids_path.exists():
            raise RuntimeError("FAISS index or ids mapping not found. Build it first.")
        index = faiss.read_index(str(self.faiss_index_path))
        ids = json.loads(self.faiss_ids_path.read_text(encoding="utf-8"))
        q = query_vec.astype(np.float32)[None, :]
        scores, idxs = index.search(q, k)
        idxs = idxs[0]; scores = scores[0]
        out = []
        for i, s in zip(idxs, scores):
            if i == -1:
                continue
            out.append((ids[i], float(s)))
        return out

    def vector_search_topk(self, client: LLMClient, query_text: str, k: int = 50, model: str = "togethercomputer/m2-bert-80M-32k-retrieval") -> List[Tuple[str, float]]:
        qv = asyncio.run(client.embed_one_async(query_text))
        qv = np.array(qv, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        # Prefer FAISS if available; fallback to NumPy if FAISS import/index is missing
        if VEC_BACKEND == "faiss":
            try:
                return self.retrive_emb(qv, k=k)
            except Exception:
                pass
        vec_store = np.load(self.vec_store_path, allow_pickle=True)
        vecs = vec_store["embeddings"]
        ids  = vec_store["ids"]
        if vecs.size == 0 or len(ids) == 0:
            return []
        sims = (vecs @ qv.reshape(-1, 1)).ravel()
        top_idx = np.argpartition(-sims, min(k, len(sims)-1))[:k]
        return sorted(((ids[i], float(sims[i])) for i in top_idx), key=lambda x: -x[1])

    def retrive_rrf_fuse(self, client: LLMClient, query: str, k: int = 60, topn: int = 50) -> List[Tuple[str, float]]:
        bm25 = self.retrieve_bm25(query, k=k)
        vec = self.vector_search_topk(client, query, k=k)
        rank: Dict[str, float] = {}
        for i, (rid, _) in enumerate(bm25):
            rank[rid] = rank.get(rid, 0.0) + 1.0 / (k + i + 1)
        for i, (rid, _) in enumerate(vec):
            rank[rid] = rank.get(rid, 0.0) + 1.0 / (k + i + 1)
        return sorted(rank.items(), key=lambda x: -x[1])[:topn]

    def mmr_select(self, fused_ids: List[str], id_to_vec: Dict[str, np.ndarray], lambda_: float = 0.7, k: int = 5) -> List[str]:
        selected: List[str] = []
        candidates = fused_ids[:]
        while candidates and len(selected) < k:
            best_id = None
            best_score = -1e9
            for rid in candidates:
                rel = 1.0
                if not selected:
                    div_penalty = 0.0
                else:
                    sims = []
                    for sid in selected:
                        v1 = id_to_vec.get(rid)
                        v2 = id_to_vec.get(sid)
                        if v1 is None or v2 is None:
                            continue
                        sims.append(float(np.dot(v1, v2)))
                    div_penalty = max(sims) if sims else 0.0
                score = lambda_ * rel - (1.0 - lambda_) * div_penalty
                if score > best_score:
                    best_score = score
                    best_id = rid
            if best_id is None:
                break
            selected.append(best_id)
            candidates.remove(best_id)
        return selected

    def retrieve_hybrid(self, client: LLMClient, query: str, top_pool: int = 40, final_k: int = 4, mmr_lambda: float = 0.7):
        vec_store = np.load(self.vec_store_path, allow_pickle=True) 
        fused = self.retrive_rrf_fuse(client, query, 60, top_pool)
        fused_ids = [rid for rid,_ in fused]
        vecs = vec_store["embeddings"] if self.vec_store_path.exists() else np.zeros((0, 768), dtype=np.float32)
        ids  = json.loads(self.faiss_ids_path.read_text(encoding="utf-8")) if self.faiss_ids_path.exists() else []
        id_to_idx = {rid:i for i,rid in enumerate(ids)}
        id_to_vec = {rid: vecs[id_to_idx[rid]] for rid in ids} if len(ids)>0 else {}
        keep = self.mmr_select(fused_ids, id_to_vec, lambda_=mmr_lambda, k=final_k)
        scored_ids = [(doc_id, float(len(keep) - rank)) for rank, doc_id in enumerate(keep)]
        return scored_ids
    