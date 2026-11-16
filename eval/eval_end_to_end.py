import csv, json
import numpy as np
from scipy.spatial.distance import jensenshannon
from utils import set_seed

set_seed(42)

submission = {}
with open("artifacts\submission_js.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = row["question"]
        dist = json.loads(row["distribution"])  
        supports = json.loads(row["supports"])
        submission[question] = {
            "distribution": dist,
            "supports": supports
        }

with open("data\dev\dev_groundtruth.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

js_scores = []
for question, gt in ground_truth.items():
    if question not in submission:
        continue
    
    gt_dist = gt["distribution"]
    sub_dist = submission[question].get("distribution", {})

    options = list(gt_dist.keys())
    p = np.array([gt_dist[o] for o in options], dtype=np.float64)
    q = np.array([sub_dist.get(o, 0.0) for o in options], dtype=np.float64)

    p /= p.sum()
    q /= q.sum()

    js = jensenshannon(p, q, base=2)  
    js_scores.append(js)


avg_js = np.mean(js_scores)
print(f"JS: {avg_js:.4f}")


def average_precision(predicted, relevant):
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    return sum_precisions / len(relevant)

ap_scores = []
for question, gt in ground_truth.items():
    if question not in submission:
        continue
    
    predicted = submission[question].get("supports", [])
    relevant = set(gt.get("supports", [])) 
    ap = average_precision(predicted, relevant)
    ap_scores.append(ap)

map_score = np.mean(ap_scores)
print(f"MAP: {map_score:.4f}")
