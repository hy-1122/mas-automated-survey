from prompts import CLASSIFY_SCHEMA, CLASSIFY_SYSTEM, CLASSIFY_SCHEMA_PARSED_OPTION, CLASSIFY_SYSTEM_PARSED_OPTION, CLASSIFY_SYSTEM_FEW_SHOT
from utils import clean_text
from mas_survey.llm_client import LLMClient
from typing import List, Dict
import random, json, os, re
from utils import set_seed

set_seed(42)

def is_valid_json(output, valid_options):
    if not isinstance(output, dict):
        return False
    if "results" not in output or not isinstance(output["results"], list):
        return False
    
    for r in output["results"]:
        if not isinstance(r, dict):
            return False
        if "id" not in r or "option" not in r:
            return False
        if not isinstance(r["id"], str) or not isinstance(r["option"], str):
            return False
        if valid_options:
            val = r.get("option_text", r["option"]).strip().lower()
            if val != "insufficient evidence" and val not in [o.lower() for o in valid_options]:
                return False

    return True

def test_probe(client, probe, meta, option_parsing, few_shot, max_retry=3):
    query = probe["query"]
    options = probe["options"]
    passages = sample_passages(meta)

    if option_parsing:
        parsed_options = parse_options(options)
        options_text = "\n".join([f"{i}. {opt}" for i, opt in parsed_options.items()])
        schema = CLASSIFY_SCHEMA_PARSED_OPTION
        if few_shot:
            system = CLASSIFY_SYSTEM_FEW_SHOT
        else:
            system = CLASSIFY_SYSTEM_PARSED_OPTION
    else:
        schema = CLASSIFY_SCHEMA
        system = CLASSIFY_SYSTEM
        options_text = options

    payload = f"Question:{query}\nOptions:\n{options_text}\n\nPassages:\n{passages}"

    out = client.chat_json(system, payload, schema_hint=schema, max_tokens=3500, max_retry=max_retry)

    results = out.get("results", [])
    if results:
        for r in results:
            if option_parsing:
                match = re.search(r"\d+", r["option"])
                option = match.group() if match else r["option"]
                r["option_text"] = parsed_options.get(option, r["option"])
            else:
                r["option_text"] = r["option"]

    valid = is_valid_json(out, options)
    return valid, out

def parse_options(options: List[str]) -> Dict[str, str]:
    return {str(i+1): opt for i, opt in enumerate(options, start=1)}

def sample_passages(meta, k=5):
    meta = list(meta.items())     
    sample = random.sample(meta, k)  
    passages_list = []
    for i, (rid, s) in enumerate(sample):
        if s.get("description"):
            text = s["description"] + " " + s.get("content", "")
        else:
            text = s.get("content", "")
        text = clean_text(text)
        header = f"{s.get('title', '')} | {s.get('source', 'n/a')} | {s.get('url', 'n/a')}"
        passages_list.append(f"{i+1}. [ID: {rid}]\n{header}\n{text[:400]}")
    return "\n\n".join(passages_list)

meta_path="artifacts/index/doc_id_to_meta.json"
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)  

with open("probes/llm_probes.json", "r", encoding="utf-8") as f:
    probes = json.load(f)

client = LLMClient(api_key= os.getenv("TOGETHER_API_KEY"))

method = "few_shot"
if method == "base":
    OPTION_PARSING = False
    MAX_RETRY = 1
    FEW_SHOT = False
elif method == "option_parsing":
    OPTION_PARSING = True
    MAX_RETRY = 1
    FEW_SHOT = False
elif method == "retry":
    MAX_RETRY = 3
    OPTION_PARSING = False
    FEW_SHOT = False
elif method == "few_shot":
    MAX_RETRY = 1
    OPTION_PARSING = True
    FEW_SHOT = True


results = []
for probe in probes:
    valid, out = test_probe(client, probe, meta, OPTION_PARSING, FEW_SHOT, MAX_RETRY)
    result = {
        "probe_id": probe["id"],
        "query": probe["query"],
        "options": probe["options"],
        "valid_json": valid,
    }
    results.append(result)

total = len(results)
success = sum(r["valid_json"] for r in results)
json_success_rate = success / total
print(f"JSON Parse Success Rate - {method}: {json_success_rate:.2%}")

with open(f"reports/llm_results/{method}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)