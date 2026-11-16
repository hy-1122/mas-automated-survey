import numpy as np
import json, time, asyncio
from typing import List
from pathlib import Path
from utils import get_combine_text, clean_text
from typing import Dict, Any
from utils import set_seed

set_seed(42)

def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:  # loop already running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)

def build_whoosh_index(docs: List[dict], index_dir: Path , reset: bool = True):
    try:
        from whoosh import fields, index
        from whoosh.analysis import StemmingAnalyzer
        import shutil

        schema = fields.Schema(
            doc_id=fields.ID(stored=True, unique=True),                             
            text=fields.TEXT(stored=True, analyzer=StemmingAnalyzer())   
        )
        
        if index_dir.exists():
            shutil.rmtree(index_dir, ignore_errors=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        ix = index.create_in(index_dir, schema)
        writer = ix.writer(limitmb=256)
        for d in docs:
            text = get_combine_text(d)
            writer.add_document(
                doc_id=d.get("id",""),
                text=text,
            )
        writer.commit()
        print(f"Whoosh index built at {index_dir} (reset={reset})")
    except Exception as e:
        print("Whoosh not available. Install with `pip install whoosh-reloaded`.", e)


def build_vector_store(docs: List[Dict], client: Any, out_path: Path, batch_size: int = 16) -> Dict[str, Any]:
    texts = [get_combine_text(d) for d in docs]
    texts = [clean_text(t) for t in texts]
    ids = [f"{d.get('id', '')}" for d in docs]

    vecs_all: List[List[float]] = []
    if texts:
        vecs_all = run_async(client.embed_many_async(texts, batch_size=batch_size))

    if not vecs_all:
        emb_arr = np.zeros((0, 768), dtype=np.float32)
    else:
        emb_arr = np.array(vecs_all, dtype=np.float32)

    np.savez_compressed(out_path, 
                        embeddings=emb_arr,
                        ids=np.array(ids),
                        model=np.array(client.embed_model),
                        created_at=np.array(int(time.time())))
    print(f"Vector store saved: {emb_arr.shape[0]} vectors, dim={emb_arr.shape[1] if emb_arr.size else 0}")
    return {"embeddings": emb_arr, "ids": ids, "model": client.embed_model, "created_at": int(time.time())}

 
def build_faiss_flat_index(vecs: np.ndarray, ids: List[str], index_path: Path, ids_path: Path) -> None:
    try:
        import faiss
    except Exception as e:
        raise RuntimeError("FAISS not installed. `pip install faiss-cpu`") from e
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs.astype(np.float32))
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(ids.tolist(), ensure_ascii=False), encoding='utf-8')
    print(f"FAISS Flat index saved: {len(ids)} vectors, dim={d}")

def build_meta(docs: List[dict], out_path: Path) -> None:
    meta = {}
    for d in docs:
        doc_id=d.get("id","")
        meta[doc_id] = {
            "url": d.get("url",""),
            "title": d.get("title",""),
            "source": d.get("source",""),
            "language": d.get("language",""),
            "date": d.get("date",""),
            "group": d.get("group",""),
            "description": d.get("description",""),
            "post_content": d.get("post_content",""),
            "content": d.get("content",""),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Document metadata saved: {len(meta)} entries to {out_path}")