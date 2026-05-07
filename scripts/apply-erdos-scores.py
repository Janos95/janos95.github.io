#!/usr/bin/env python3
import argparse
import html
import json
import re
from pathlib import Path


DEFAULT_PROBLEMS = Path("blog/open-erdos-problems/open-problems.json")
DEFAULT_RESULTS = Path("blog/open-erdos-problems/llm-scored-problems.jsonl")
DEFAULT_PAGE = Path("blog/open-erdos-problems/index.html")


def load_results(path):
    results = {}
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("ok") and record.get("problem_id") is not None:
            results[int(record["problem_id"])] = record
    return results


def money(value):
    return f"${int(value):,}" if int(value or 0) else ""


def table_rows(problems, results, include_rationales=False):
    rows = []
    for problem in problems:
        result = results.get(problem["id"])
        hardness = result.get("hardness") if result else problem["hardness"]
        interestingness = result.get("interestingness") if result else problem["interestingness"]
        score_class = " score-cell scored" if result else " score-cell placeholder"
        details = f'<a href="https://www.erdosproblems.com/{problem["id"]}" target="_blank" rel="noopener noreferrer">Problem page</a>'
        if include_rationales:
            details = (
                f'<a href="#rationale-{problem["id"]}">Rationale</a>'
                if result
                else '<span class="placeholder-label">Pending</span>'
            )
        rows.append(f"""                <tr data-id="{problem['id']}" data-hardness="{hardness}" data-interestingness="{interestingness}" data-prize="{problem['prize']}">
                    <td class="problem-number"><a href="https://www.erdosproblems.com/{problem['id']}" target="_blank" rel="noopener noreferrer">#{problem['id']}</a></td>
                    <td class="{score_class.strip()}">{hardness}</td>
                    <td class="{score_class.strip()}">{interestingness}</td>
                    <td class="prize-cell">{money(problem['prize'])}</td>
                    <td class="rationale-cell">{details}</td>
                </tr>""")
    return "\n".join(rows)


def rationale_sections(problems, results):
    sections = []
    for problem in problems:
        result = results.get(problem["id"])
        if not result:
            continue
        sources = []
        for source in result.get("sources", [])[:6]:
            url = source.get("url")
            if not url:
                continue
            title = source.get("title") or url
            sources.append(f'<li><a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(title)}</a></li>')
        source_html = f"<ul>{''.join(sources)}</ul>" if sources else "<p>No sources returned.</p>"
        sections.append(f"""                <section id="rationale-{problem['id']}" class="rationale-entry">
                    <h3><a href="https://www.erdosproblems.com/{problem['id']}" target="_blank" rel="noopener noreferrer">Problem #{problem['id']}</a></h3>
                    <p><strong>Hardness {result.get('hardness')}.</strong> {html.escape(result.get('hardness_rationale', ''))}</p>
                    <p><strong>Interest {result.get('interestingness')}.</strong> {html.escape(result.get('interestingness_rationale', ''))}</p>
                    <p><strong>Literature review.</strong> {html.escape(result.get('literature_review', ''))}</p>
                    <p><strong>Solve-attempt notes.</strong> {html.escape(result.get('solve_attempt_notes', ''))}</p>
                    <p><strong>Status caveat.</strong> {html.escape(result.get('status_caveat', ''))}</p>
                    <div class="rationale-sources"><strong>Sources</strong>{source_html}</div>
                </section>""")
    if not sections:
        return ""
    return "\n                <section class=\"rationale-list\" aria-labelledby=\"rationales-heading\">\n                    <h2 id=\"rationales-heading\">Rationales</h2>\n" + "\n".join(sections) + "\n                </section>"


def main():
    parser = argparse.ArgumentParser(description="Apply LLM Erdős scores to the static blog page.")
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--include-rationales", action="store_true", help="Embed full model rationales in the public HTML page.")
    args = parser.parse_args()

    problems = json.loads(args.problems.read_text(encoding="utf-8"))["problems"]
    results = load_results(args.results)
    page = args.page.read_text(encoding="utf-8")

    page = re.sub(
        r'\s*<tbody id="problem-table-body">.*?\s*</tbody>',
        "\n                        <tbody id=\"problem-table-body\">\n" + table_rows(problems, results, args.include_rationales) + "\n                        </tbody>",
        page,
        flags=re.S,
    )

    rationales = rationale_sections(problems, results) if args.include_rationales else ""
    if re.search(r'\s*<section class="rationale-list" aria-labelledby="rationales-heading">.*?</section>\s*</section>\s*</article>', page, re.S):
        page = re.sub(
            r'\s*<section class="rationale-list" aria-labelledby="rationales-heading">.*?</section>\s*</section>\s*</article>',
            lambda match: "\n" + rationales + "\n            </section>\n        </article>",
            page,
            flags=re.S,
        )
    else:
        page = page.replace("\n            </section>\n        </article>", "\n" + rationales + "\n            </section>\n        </article>")

    args.page.write_text(page, encoding="utf-8")
    print(f"Applied {len(results)} scored records to {args.page}")


if __name__ == "__main__":
    main()
