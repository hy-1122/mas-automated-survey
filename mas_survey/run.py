# mas_survey/run.py
"""
Minimal placeholder for the MAS runner (two arguments only).

CLI:
    python -m mas_survey.run --config config.yaml --api_key <YOUR_KEY>
"""

import argparse, json, yaml, csv
from pathlib import Path
from .llm_client import LLMClient
from .retriever import Retriever
from .agents import State, RetrievalAgent, ClassifyAgent,  AggregateAgent, ExpandQueryAgent
from utils import set_seed

set_seed(42)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="(placeholder) Run MAS pipeline")
    p.add_argument("--config", type=str, default="config.yaml",
                   help="Path to config.yaml (default: config.yaml)")
    p.add_argument("--api_key", type=str, default=None,
                   help="API key (accepted but unused in placeholder).")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    print(f"[mas_survey.run] config={args.config}  api_key={'<provided>' if args.api_key else '<none>'}")

    config_path = Path(args.config)  # this is a string path
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    FULL_DATASET = cfg["index"]["full_dataset"]

    # Get index directory and path
    faiss_index_path = Path(cfg["index"]["faiss_index_path"])
    faiss_ids_path = Path(cfg["index"]["faiss_ids_path"])
    whoosh_dir = Path(cfg["index"]["whoosh_output_dir"])
    whoosh_dir.mkdir(exist_ok=True)
    meta_path = Path(cfg["index"]["meta_path"])
    vec_store_path = cfg["index"]["vec_store"]["full" if FULL_DATASET else "mini"]

    # get output path and required config
    output_path = cfg["run"]["output_csv"]
    test_input_path = cfg["run"]["test_input"]
    test_output_path = cfg["run"]["test_output"]
    support_k = cfg["run"]["supports_k"]
    dev_path = cfg["run"]["dev"]["full" if FULL_DATASET else "mini"]
    expand_query = cfg["run"]["query_expansion"]
    options_parsing = cfg["run"]["option_parsing"]
    max_retry = cfg["run"]["max_retry_prompt"]
    retrieve_method = cfg["run"]["retrieve_method"]
    few_shot = cfg["run"]["few_shot"]

    if args.api_key:
        client = LLMClient(api_key=args.api_key)
    else:
        client = None

    retriever = Retriever(
        vec_store_path=vec_store_path,
        index_dir=whoosh_dir,
        faiss_index_path=faiss_index_path,
        faiss_ids_path=faiss_ids_path
    )

    expand_query_agent = ExpandQueryAgent(client, support_k)
    retrieval_agent = RetrievalAgent(client, support_k, retriever, meta_path=meta_path, method=retrieve_method)
    classify_agent = ClassifyAgent(client, support_k)
    aggregate_agent = AggregateAgent(client, support_k)


    with open(dev_path, "r", encoding="utf-8") as f:
        survey = json.load(f)

    with open(test_input_path, "r", encoding="utf-8") as f:
        test = json.load(f)

    all_outputs = []

    for query, values in test.items():
        options = []
        for value in values["distribution"]:
            options.append(value)

        print(f"[mas_survey.run] Processing {query}")
        state = State(query=query)
        state.options = options
        state.evidence = []
        state.classifications = []

        if expand_query:
            state = expand_query_agent.run(state)
            for m in state.messages:
                if m["agent"] == expand_query_agent.name:
                    print(f"[max_survey.run]-[{m['agent']}] {m['msg']}")

        state = retrieval_agent.run(state)
        for m in state.messages:
            if m["agent"] == retrieval_agent.name:
                print(f"[max_survey.run]-[{m['agent']}] {m['msg']}")

        state = classify_agent.run(state, option_parsing=options_parsing, max_retry=max_retry, few_shot=few_shot)
        for m in state.messages:
            if m["agent"] == classify_agent.name:
                print(f"[max_survey.run]-[{m['agent']}] {m['msg']}")

        state = aggregate_agent.run(state)
        for m in state.messages:
            if m["agent"] == aggregate_agent.name:
                print(f"[max_survey.run]-[{m['agent']}] {m['msg']}")

        output = {
            "question": query,
            "distribution": state.distribution,
            "supports": state.supports
        }

        print("[mas_survey.run] Completed!")
        
        all_outputs.append(output)


    # with open(output_path, mode="w", newline="", encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=["question", "distribution", "supports"])
    #     writer.writeheader()
    #     for row in all_outputs:
    #         writer.writerow({
    #             "question": row["question"],
    #             "distribution": json.dumps(row["distribution"]),  
    #             "supports": json.dumps(row["supports"])     
    #         })   

    with open(test_output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "distribution", "supports"])
        writer.writeheader()
        for row in all_outputs:
            writer.writerow({
                "question": row["question"],
                "distribution": json.dumps(row["distribution"]),  
                "supports": json.dumps(row["supports"])     
            })    


if __name__ == "__main__":
    main()
