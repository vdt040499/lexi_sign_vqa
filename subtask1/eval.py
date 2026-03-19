"""
Evaluate Sub Task 1 results — F2-score, Precision, Recall.

Usage:
    python subtask1/eval.py
    python subtask1/eval.py --results data/public_test/results.json
"""

import argparse
import json


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_sample(ground_truth: list[dict], predicted: list[dict]) -> tuple[float, float, float]:
    gt_ids   = {a["law_id"] + "#" + a["article_id"] for a in ground_truth}
    pred_ids = {a["law_id"] + "#" + a["article_id"] for a in predicted}

    if not pred_ids:
        return 0.0, 0.0, 0.0

    precision = len(gt_ids & pred_ids) / len(pred_ids)
    recall    = len(gt_ids & pred_ids) / len(gt_ids) if gt_ids else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f2


def evaluate(samples: list[dict]) -> dict:
    precisions, recalls, f2s = [], [], []
    skipped = 0

    for sample in samples:
        if "predicted_articles" not in sample:
            skipped += 1
            continue
        if "relevant_articles" not in sample:
            skipped += 1
            continue

        p, r, f2 = score_sample(sample["relevant_articles"], sample["predicted_articles"])
        precisions.append(p)
        recalls.append(r)
        f2s.append(f2)

    n = len(f2s)
    return {
        "evaluated":  n,
        "skipped":    skipped,
        "precision":  sum(precisions) / n if n else 0.0,
        "recall":     sum(recalls)    / n if n else 0.0,
        "f2":         sum(f2s)        / n if n else 0.0,
    }


def main() -> None:
    from subtask1.config import OUTPUT_JSON

    parser = argparse.ArgumentParser(description="Evaluate Sub Task 1 F2-score")
    parser.add_argument("--results", "-r", type=str, default=OUTPUT_JSON,
                        help="Path to results JSON file")
    args = parser.parse_args()

    samples = load_json(args.results)
    results = evaluate(samples)

    print(f"\n{'─'*40}")
    print(f"  File      : {args.results}")
    print(f"  Evaluated : {results['evaluated']} samples  (skipped {results['skipped']})")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F2-score  : {results['f2']:.4f}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()
