# Log summarizer agent (dashboard-only, for the logs part only).
# When the log file grows or a human wants a quick review, this agent reads
# the recent entries and produces a short plain-language digest: what was
# asked, how it was routed, whether it worked, anything suspicious
# (e.g. a summary view used where base tables look more direct).
# It only summarizes what is in the log - it never invents content.
import json

from src.llm import llm
from .logging_utils import read_recent


def summarize_recent_logs(last=5):
    # read the most recent records, compact them (drop long SQL), and let
    # the LLM turn them into a few readable lines per run
    records = read_recent(last)
    if not records:
        return "No log entries found."

    lines = []
    for record in records:
        compact = {
            "ts": record.get("ts"),
            "kind": record.get("kind"),
            "description": (record.get("description") or "")[:200],
            "question": (record.get("question") or "")[:200],
            "workspaces": record.get("workspaces"),
            "attempts": [
                {
                    "attempt": a.get("attempt"),
                    "outcome": (a.get("outcome") or "")[:120],
                }
                for a in (record.get("attempts") or [])
            ],
            "final_status": record.get("final_status"),
        }
        lines.append(json.dumps(compact, ensure_ascii=False))

    prompt = (
        "Summarize this QueryLLM request log for a data engineer.\n"
        "For each run, state in 1-3 short lines: what the analyst asked, "
        "how it was routed, whether it succeeded, and anything suspicious "
        "(e.g. a summary view chosen where base tables look more direct).\n"
        "Plain language only; do not invent facts that are not in the log.\n\n"
        + "\n".join(lines)
    )
    return llm(
        prompt,
        system="You summarize request logs.",
        session="ses_queryllm_logs",
        temperature=0.2,
    )
