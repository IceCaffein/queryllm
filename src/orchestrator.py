# Orchestrator (first defined in notebook/06_queryllm.ipynb).
# query_llm(question, catalog, rules) is THE analyst-facing entry point:
# intent -> SQL writer -> validator (repair loop) -> explanation.
import time

from src.llm import llm
from src.intent import intent_agent
from src.writer import sql_writer
from src.validator import validator_agent


def explanation_agent(question, sql):
    prompt = (
        "Explain what this SQL query does in 2-3 plain sentences for a data "
        "analyst. Mention any column that appears anonymized/masked or "
        "aggregated in the query.\n\n"
        f"Question: {question}\n\n"
        f"SQL:\n{sql}\n\n"
        "Explanation:"
    )
    return llm(prompt, system="You explain SQL queries simply.", session="ses_queryllm_explain")


def query_llm(question, catalog, rules, max_retries=3, explain=True):
    # per-stage durations are recorded in the result (and the logs) so every
    # speed optimization can be measured before/after
    t_start = time.perf_counter()
    workspaces = intent_agent(question, catalog)
    t_intent = time.perf_counter() - t_start
    if workspaces is None:
        # honest refusal: the question is outside every catalog workspace
        return {
            "question": question,
            "workspaces": None,
            "status": "no_route",
            "answer": "This question cannot be answered with the data available in the catalog.",
        }

    print(f"routed to: {workspaces}")
    t_write_start = time.perf_counter()
    draft = sql_writer(question, workspaces, catalog, rules)
    t_write = time.perf_counter() - t_write_start
    t_validate_start = time.perf_counter()
    outcome = validator_agent(question, draft, workspaces, catalog, rules, max_retries=max_retries)
    t_validate = time.perf_counter() - t_validate_start

    durations = {
        "intent_s": round(t_intent, 2),
        "write_s": round(t_write, 2),
        "validate_s": round(t_validate, 2),
        "total_s": round(time.perf_counter() - t_start, 2),
    }

    if outcome["status"] != "ok":
        return {
            "question": question,
            "workspaces": workspaces,
            "status": "failed",
            "sql": outcome["sql"],
            "history": outcome["history"],
            "durations": durations,
        }

    result = {
        "question": question,
        "workspaces": workspaces,
        "status": "ok",
        "sql": outcome["sql"],
        "columns": outcome["columns"],
        "rows": outcome["rows"],
        "attempts": outcome["attempts"],
        "history": outcome["history"],
        "durations": durations,
    }
    # the dashboard skips the explanation (explain=False): it shows the SQL
    # only, so no explanation LLM call is spent
    if explain:
        result["explanation"] = explanation_agent(question, outcome["sql"])
    return result
