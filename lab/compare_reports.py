"""Compare reports from the same task/split; prints strict per-example outcomes."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("baseline")
parser.add_argument("candidate")
args = parser.parse_args()
with open(args.baseline, encoding="utf-8") as f:
    base = json.load(f)
with open(args.candidate, encoding="utf-8") as f:
    candidate = json.load(f)
expected = lambda r: [(p["id"], p["expected"]) for p in r["predictions"]]
if base["split"] != candidate["split"] or expected(base) != expected(candidate):
    raise SystemExit("Reports use different evaluation examples; comparison refused.")
for metric in ["accuracy", "schema_valid_rate", "median_request_seconds", "aggregate_tokens_per_second"]:
    print(f"{metric}: baseline={base[metric]:.4f}; candidate={candidate[metric]:.4f}")
print("Timing is descriptive; control hardware load and output lengths for speed claims.")
for old, new in zip(base["predictions"], candidate["predictions"]):
    if old["correct"] != new["correct"]:
        print(old["id"], "improved" if new["correct"] else "regressed")
