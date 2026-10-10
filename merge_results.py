"""Merge result files from several runs and list what still needs (re-)running.

Run: python merge_results.py prompts_china30.tsv results_china30.tsv china30/*.tsv runs/*.jsonl

Rows are keyed by (responder, Prompt ID, language). A finished row ('success' or 'blocked')
beats a failed one, so re-running only the failed prompts and merging again fills the gaps.
For each responder with gaps, writes todo_<responder>.tsv with just the prompts to re-run.
"""

import csv
import json
import sys
from implementations.tab_delimited_exporter import TabDelimitedExporter

DONE = ("success", "blocked")
csv.field_size_limit(10**9)


def load(path):
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in open(path, encoding="utf-8")]
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


def main(prompts_path, out_path, sources):
    prompts = load(prompts_path)
    ids = {p["Prompt ID"] for p in prompts}
    best = {}
    for src in sources:
        for row in load(src):
            if row.get("Prompt ID") not in ids:
                continue  # rows from other prompt sets
            key = (row["responder"], row["Prompt ID"], row["language"])
            if key not in best or (row["status"] in DONE) >= (best[key]["status"] in DONE):
                best[key] = row

    rows = sorted(best.values(), key=lambda r: (r["responder"], r["Prompt ID"], r["language"]))
    TabDelimitedExporter().export_results(out_path, rows)

    for responder in sorted({r["responder"] for r in rows}):
        todo = [p for p in prompts
                if any(best.get((responder, p["Prompt ID"], lang), {}).get("status") not in DONE
                       for lang in ("English", "Mandarin"))]
        done = sum(1 for r in rows if r["responder"] == responder and r["status"] in DONE)
        print(f"{responder:24} {done:4}/{2 * len(prompts)} done, {len(todo)} prompts to re-run")
        if todo:
            with open(f"todo_{responder}.tsv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(prompts[0].keys()), delimiter="\t")
                w.writeheader()
                w.writerows(todo)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
