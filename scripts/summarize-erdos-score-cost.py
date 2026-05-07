#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_RESULTS = Path("blog/open-erdos-problems/llm-scored-problems.jsonl")
GPT55_INPUT_PER_MILLION = 5.00
GPT55_OUTPUT_PER_MILLION = 30.00


def usage_value(usage, *keys):
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Summarize GPT-5.5 usage and estimated cost for Erdős scoring JSONL.")
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--input-price", type=float, default=GPT55_INPUT_PER_MILLION)
    parser.add_argument("--output-price", type=float, default=GPT55_OUTPUT_PER_MILLION)
    args = parser.parse_args()

    records = []
    for line in args.results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    with_usage = [r for r in records if r.get("usage")]
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for record in with_usage:
        usage = record["usage"]
        input_tokens += usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens += usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens += usage_value(usage, "total_tokens")

    estimated = (input_tokens / 1_000_000 * args.input_price) + (output_tokens / 1_000_000 * args.output_price)
    print(f"records: {len(records)}")
    print(f"records_with_usage: {len(with_usage)}")
    print(f"input_tokens: {input_tokens:,}")
    print(f"output_tokens: {output_tokens:,}")
    print(f"total_tokens: {total_tokens:,}")
    print(f"estimated_cost_usd: ${estimated:.4f}")


if __name__ == "__main__":
    main()
