import re
import json
import argparse

# Strict env id from the "Results for" line (clean logs only).
_TASK_RE = re.compile(r"Results for\s+(gr1_unified/[A-Za-z0-9_/]+_Env)\s*:?")
# Tee-merged logs often break the Results line; the next stable token is from the harness.
_EXECUTED_RE = re.compile(r"Successfully executed:\s*(gr1_unified/[A-Za-z0-9_/]+_Env)")
_SUCCESS_RE = re.compile(r"Success rate:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _resolve_task_id(chunk: str) -> str | None:
    """Prefer ``Successfully executed:`` (authoritative) when the Results line is corrupted."""
    m_exec = _EXECUTED_RE.search(chunk)
    m_task = _TASK_RE.match(chunk)
    strict = m_task.group(1) if m_task else None
    executed = m_exec.group(1) if m_exec else None
    if executed is not None:
        return executed
    return strict


def extract_success_rates(log_content):
    """Parse per-episode blocks that start with ``Results for gr1_unified/..._Env``.

    The log is often interleaved by ``tee``; a single regex from ``Results for`` to
    ``Success rate:`` can absorb the next task's stderr. We split on each
    ``Results for gr1_unified/`` boundary and take the first ``Success rate:`` *within* each chunk.
    """
    chunks = re.split(r"(?=Results for gr1_unified/)", log_content)

    # Every resolved env id gets an entry; ``None`` means tee-interleaved log dropped ``Success rate:``.
    tasks: dict[str, float | None] = {}
    num_result_blocks = 0

    for chunk in chunks:
        chunk = chunk.lstrip()
        if not chunk.startswith("Results for gr1_unified/"):
            continue

        num_result_blocks += 1
        task = _resolve_task_id(chunk)
        if not task:
            continue

        sm = _SUCCESS_RE.search(chunk)
        tasks[task] = float(sm.group(1)) if sm else None

    numeric = {k: v for k, v in tasks.items() if v is not None}
    if not numeric:
        print("No success rates found in the log content.")
        return None

    average_rate = sum(numeric.values()) / len(numeric)

    pnpclose_success_rates = {k: v for k, v in numeric.items() if "Close" in k}
    if len(pnpclose_success_rates) > 0:
        pnpclose_avg_rate = sum(pnpclose_success_rates.values()) / len(pnpclose_success_rates)
    else:
        pnpclose_avg_rate = -9999

    pnponly_success_rates = {k: v for k, v in numeric.items() if "Close" not in k}
    if len(pnponly_success_rates) > 0:
        pnponly_avg_rate = sum(pnponly_success_rates.values()) / len(pnponly_success_rates)
    else:
        pnponly_avg_rate = -9999

    missing = sorted(k for k, v in tasks.items() if v is None)
    out: dict = {
        "tasks": tasks,
        "average_success_rate": {
            "overall": average_rate,
            "PnPClose": pnpclose_avg_rate,
            "PnPOnly": pnponly_avg_rate,
        },
        "num_tasks_parsed": len(numeric),
        "num_tasks_total": len(tasks),
        "num_result_blocks": num_result_blocks,
    }
    if missing:
        out["tasks_without_success_rate"] = missing
    return out


def save_to_json(data, output_file):
    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Extract task success rates from log file and calculate average.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", "--input", required=True, help="Path to the input log file")

    parser.add_argument("-o", "--output", default="success_rates.json", help="Path to the output JSON file")

    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output to console")

    args = parser.parse_args()

    try:
        with open(args.input, "r") as file:
            log_content = file.read()

        success_data = extract_success_rates(log_content)

        if success_data:
            save_to_json(success_data, args.output)

            if args.verbose:
                print(f"Success rates extracted and saved to {args.output}")
                print("Results:")
                print(json.dumps(success_data, indent=4))
            else:
                n = success_data.get("num_tasks_parsed", 0)
                total = success_data.get("num_tasks_total", n)
                print(f"Success rates extracted ({n}/{total} tasks with numeric rate) and saved to {args.output}")
        else:
            print("No success rates were extracted.")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
