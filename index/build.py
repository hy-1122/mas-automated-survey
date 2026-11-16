# index/build.py
"""
Minimal placeholder for the index builder.

Keeps the CLI contract only (arguments parsing). No indexing logic.
Usage:
    python -m index.build --config config.yaml --api_key <YOUR_KEY>
"""

import argparse
from together import Together
import json, yaml
from index.index import build_faiss_flat_index, build_whoosh_index, build_meta, build_vector_store
from mas_survey.llm_client import LLMClient
from pathlib import Path
import numpy as np
from utils import set_seed

set_seed(42)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="(placeholder) Build indexes from documents.jsonl"
    )
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    p.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Together (or other) API key; accepted but unused in placeholder.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Placeholder: just acknowledge arguments.
    print(f"[index.build] config={args.config}  api_key={'<provided>' if args.api_key else '<none>'}")

    config_path = Path(args.config)  # this is a string path
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    FULL_DATASET = cfg["index"]["full_dataset"]

    # Get index output directory
    faiss_index_path = Path(cfg["index"]["faiss_index_path"])
    faiss_ids_path = Path(cfg["index"]["faiss_ids_path"])
    whoosh_dir = Path(cfg["index"]["whoosh_output_dir"])
    whoosh_dir.mkdir(exist_ok=True)
    meta_path = Path(cfg["index"]["meta_path"])
    
    docs_path = cfg["index"]["documents"]["full" if FULL_DATASET else "mini"]
    vec_store = cfg["index"]["vec_store"]["full" if FULL_DATASET else "mini"]

    # load documents
    with open(docs_path,"r",encoding="utf-8") as f:
        docs = [json.loads(line) for line in f] 

    if args.api_key:
        client = LLMClient(api_key=args.api_key)
    else:
        client = None

    if not FULL_DATASET:
        print("[index.build] Building vector store ...")
        build_vector_store(docs, client, out_path=Path(vec_store))

    # load embeddings from vector store
    print(f"[index.build] Loading vector embeddings ...")
    try:
        vec_store = np.load(vec_store, allow_pickle=True)
        emb = vec_store["embeddings"]
        ids = vec_store["ids"]
    except Exception as e:
        print("[index.build] Failed to load embeddings.", e)
        return

    # build index and meta
    print(f"[index.build] Building bm25 whoosh index ...")
    build_whoosh_index(docs, index_dir=whoosh_dir)
    print(f"[index.build] Building vector faiss index ...")
    build_faiss_flat_index(emb, ids, faiss_index_path, faiss_ids_path)
    print(f"[index.build] Saving meta ...")
    build_meta(docs, meta_path)

if __name__ == "__main__":
    main()
