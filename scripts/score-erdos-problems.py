#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_INPUT = Path("blog/open-erdos-problems/open-problems.json")
DEFAULT_OUTPUT = Path("blog/open-erdos-problems/llm-scored-problems.jsonl")
DEFAULT_ENV = Path.home() / ".codex" / "gpt-pro-oracle.env"


SYSTEM_PROMPT = """You are a careful mathematical research assistant.
You are scoring open problems of Paul Erdos for a public blog table.
Use web search to check the linked Erdős Problems page, likely references, and recent literature.
Do not claim a problem is solved unless you find strong evidence from reliable sources.
Return only valid JSON. No markdown, no prose outside the JSON object."""


def load_env_file(path):
    values = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return values

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def get_config():
    env_path = Path(os.environ.get("GPT_PRO_ORACLE_ENV", str(DEFAULT_ENV)))
    file_values = load_env_file(env_path)
    api_key = os.environ.get("OPENAI_API_KEY") or file_values.get("OPENAI_API_KEY")
    base_url = (os.environ.get("OPENAI_BASE_URL") or file_values.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    return api_key, base_url, env_path


def api_request(method, url, api_key, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {api_key}")
    if body is not None:
        request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API returned HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OpenAI API request failed: {error}") from error


def extract_output_text(response):
    if response.get("output_text"):
        return response["output_text"]

    chunks = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def extract_sources(response):
    sources = []
    seen = set()

    def visit(value):
        if isinstance(value, dict):
            url = value.get("url")
            title = value.get("title")
            if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in seen:
                seen.add(url)
                sources.append({"url": url, "title": title or ""})
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(response.get("output", []))
    visit(response.get("sources", []))
    return sources


def parse_json_object(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def build_prompt(problem):
    url = f"https://www.erdosproblems.com/{problem['id']}"
    tags = ", ".join(problem.get("tags") or [])
    prize = f"${problem.get('prize', 0):,}" if problem.get("prize", 0) else "none"
    return f"""Score this open Erdős problem.

Problem URL: {url}
Problem id: {problem['id']}
Prize: {prize}
Tags: {tags}
Statement:
{problem['statement']}

Task:
1. Do a targeted literature review. Search the problem page, its remarks/references, and recent web/arXiv literature for relevant progress, partial results, and context.
2. Spend some effort thinking about plausible solution approaches and barriers. Do not write a proof attempt unless it is genuinely useful for scoring.
3. Score:
   - hardness: integer 0-100, where 100 means an extremely hard frontier-level open problem and 50 means a serious but narrower research problem.
   - interestingness: integer 0-100, where 100 means a solution would likely be highly influential or conceptually important.

Return only a JSON object with exactly these keys:
{{
  "problem_id": {problem['id']},
  "hardness": <integer 0-100>,
  "interestingness": <integer 0-100>,
  "hardness_rationale": "<concise but substantive explanation>",
  "interestingness_rationale": "<concise but substantive explanation>",
  "literature_review": "<brief summary of what you found, including partial results and recent status>",
  "solve_attempt_notes": "<brief notes on plausible approaches and barriers>",
  "confidence": <number 0-1 measuring confidence in the two scores, considering source quality, amount of literature found, and clarity of the problem's current status>,
  "status_caveat": "<mention if the open status may be stale or uncertain, otherwise say it appears open according to the checked sources>"
}}"""


def build_payload(args, problem):
    payload = {
        "model": args.model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(problem)},
        ],
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": args.max_output_tokens,
        "background": True,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
    }
    return payload


def create_and_wait(args, api_key, base_url, payload):
    created = api_request("POST", f"{base_url}/responses", api_key, payload, timeout=args.request_timeout)
    response_id = created.get("id")
    if not response_id:
        return created

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        current = api_request("GET", f"{base_url}/responses/{response_id}", api_key, timeout=args.request_timeout)
        status = current.get("status")
        if status in {"completed", "failed", "cancelled", "incomplete"}:
            return current
        time.sleep(args.poll_interval)

    raise RuntimeError(f"Timed out waiting for background response {response_id}")


def load_done_ids(output_path):
    done = set()
    if not output_path.exists():
        return done
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("ok") and record.get("problem_id") is not None:
            done.add(int(record["problem_id"]))
    return done


def score_problem(args, api_key, base_url, problem):
    response = create_and_wait(args, api_key, base_url, build_payload(args, problem))
    if response.get("status") in {"failed", "cancelled", "incomplete"}:
        return {
            "ok": False,
            "problem_id": problem["id"],
            "status": response.get("status"),
            "error": response.get("error") or response.get("incomplete_details") or response,
            "usage": response.get("usage") or {},
            "response_id": response.get("id"),
            "model": args.model,
        }

    output_text = extract_output_text(response)
    parsed = parse_json_object(output_text)
    parsed["problem_id"] = int(parsed.get("problem_id", problem["id"]))
    parsed["source_url"] = f"https://www.erdosproblems.com/{problem['id']}"
    parsed["sources"] = extract_sources(response)
    parsed["usage"] = response.get("usage") or {}
    parsed["response_id"] = response.get("id")
    parsed["model"] = args.model
    parsed["ok"] = True
    return parsed


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def select_problems(args):
    data = json.loads(args.input.read_text(encoding="utf-8"))
    problems = data["problems"]
    if args.problem_id is not None:
        problems = [p for p in problems if p["id"] == args.problem_id]
        if not problems:
            raise SystemExit(f"Problem #{args.problem_id} not found in {args.input}")
    if args.paid:
        problems = [p for p in problems if int(p.get("prize") or 0) > 0]
    if args.limit is not None:
        problems = problems[: args.limit]
    return problems


def main():
    parser = argparse.ArgumentParser(description="Score open Erdős problems with the OpenAI Responses API.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--problem-id", type=int)
    parser.add_argument("--paid", action="store_true", help="Only score problems with a nonzero dollar prize.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Rescore problems already present in the output JSONL.")
    args = parser.parse_args()

    api_key, base_url, env_path = get_config()
    if not api_key:
        raise SystemExit(f"Missing OPENAI_API_KEY. Run ~/.codex/skills/gpt-pro-oracle/scripts/setup.sh. Expected env file: {env_path}")

    problems = select_problems(args)
    if not args.force:
        done = load_done_ids(args.output)
        problems = [p for p in problems if p["id"] not in done]

    print(f"Scoring {len(problems)} problems with {args.model}; concurrency={args.concurrency}; output={args.output}", file=sys.stderr)
    if not problems:
        return 0

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(score_problem, args, api_key, base_url, problem): problem for problem in problems}
        for future in concurrent.futures.as_completed(futures):
            problem = futures[future]
            try:
                record = future.result()
            except Exception as error:
                failures += 1
                record = {"ok": False, "problem_id": problem["id"], "error": str(error)}
            append_jsonl(args.output, record)
            status = "ok" if record.get("ok") else "failed"
            print(f"{status}: #{problem['id']}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
