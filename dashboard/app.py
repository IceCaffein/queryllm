# QueryLLM dashboard - form-based chat UI (SQL-only answers).
# Flow: analyst describes the data -> the form agent expands a structured
# form (analyst-eye fields only) -> the composed question runs through
# query_llm -> the chat replies with the final SQL. No rows are shown:
# the analyst copies the SQL and runs it in their own SQL client.
#
# Every request is logged to logs/queries.jsonl (see dashboard/src/logging_utils.py):
# - request_context: the description + form spec (written when the form appears)
# - pipeline_run:    the composed question, route, the schema text the model
#                    saw, every SQL attempt with its outcome, and the final SQL
import sys
import uuid
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv

# project root = parent of dashboard/ (holds .env, catalog/, core src/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # core src/ must win the `import src` race

# core imports first, then the dashboard package
from src.writer import schema_block
from dashboard.src.logging_utils import append_log
from dashboard.src.log_agent import summarize_recent_logs
from dashboard.src.form_agent import (
    analyze_description,
    compose_question,
    is_forbidden_request,
    REFUSAL_TEXT,
    CROSS_SYSTEM_REFUSAL_TEXT,
)
from dashboard.src.sql_format import format_sql


@st.cache_resource
def load_pipeline():
    # Load once per server process: env, catalog, rules, and the agent stack.
    # dashboard/.env ONLY - the web pipeline never touches the database, so
    # this server holds no Oracle credentials at all (root .env keeps them
    # for the notebooks). Core imports first (src.*), then dashboard.src.*.
    load_dotenv(ROOT / "dashboard" / ".env")
    catalog = yaml.safe_load((ROOT / "catalog" / "catalog.yaml").read_text(encoding="utf-8"))
    rules = yaml.safe_load((ROOT / "catalog" / "rules.yaml").read_text(encoding="utf-8"))
    from src.orchestrator import query_llm
    from dashboard.src.form_agent import analyze_description, compose_question
    return catalog, rules, query_llm, analyze_description, compose_question


CATALOG, RULES, query_llm, analyze_description, compose_question = load_pipeline()

st.set_page_config(page_title="QueryLLM", page_icon=":speech_balloon:")
st.title("QueryLLM - data request console")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("What this does")
    st.write(
        "You describe the data you need in plain language. A short form "
        "captures the details, and QueryLLM writes the SQL for you. "
        "Copy the SQL and run it in your own SQL client."
    )
    st.caption("LLM reads catalog + rules only. Every query is validated "
               "read-only before you see it.")
    st.subheader("Available data domains")
    for name, ws in CATALOG["workspaces"].items():
        st.write(f"**{name}** - {ws['description']}")
    with st.expander("Request logs"):
        st.caption("Every request is logged to logs/queries.jsonl "
                   "(description, route, attempts, final SQL).")
        if st.button("Summarize recent log entries"):
            with st.spinner("Summarizing..."):
                st.write(summarize_recent_logs(5))

# ------------------------------------------------------------ chat state
if "messages" not in st.session_state:
    st.session_state.messages = []          # rendered chat history
if "pending" not in st.session_state:
    st.session_state.pending = None         # {"request_id", "description", "spec"} awaiting the form

# ------------------------------------------------------- render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["kind"] == "text":
            st.write(msg["content"])
        elif msg["kind"] == "sql":
            st.write(msg.get("intro", "Here is your SQL:"))
            st.code(msg["content"], language="sql")
            if msg.get("request_id"):
                st.caption(f"request {msg['request_id']} - find it in logs/queries.jsonl")
            if msg.get("history"):
                with st.expander("Validation attempts (for the data engineer)"):
                    for h in msg["history"]:
                        st.write(f"attempt {h['attempt']}: {h['outcome']}")

# ------------------------------------------------------- step 1: describe
if st.session_state.pending is None:
    description = st.text_area(
        "Describe the data you need (in your own words, no table names needed):",
        height=90,
        placeholder="e.g. I need the revenue per month for the first half of 2021",
    )
    if st.button("Continue", type="primary", disabled=not description.strip()):
        description = description.strip()
        request_id = uuid.uuid4().hex[:8]

        # ---- the request gate: refuse destructive/mutating asks BEFORE the
        # form. Layer 1 (regex) refuses with no LLM call; layer 2 (the
        # request_type the form agent returns) catches paraphrases.
        if is_forbidden_request(description):
            refused, reason = True, "forbidden_pattern (regex layer)"
        else:
            with st.spinner("Reading your description..."):
                spec = analyze_description(description)
            refused = spec["request_type"] != "data_request"
            reason = spec["request_type"]

        # the catalog policy has the final word on cross-system asks: when
        # cross-schema matching is disabled, refuse before the form renders
        if (not refused and spec["cross_system"]
                and CATALOG["policy"].get("cross_schema_joins") == "disallowed"):
            refused, reason = True, "cross_system_disabled"

        if refused:
            # log the refusal so the log agent reports "refused at the gate",
            # not a suspicious hole - then show the refusal in the chat
            refusal_text = (CROSS_SYSTEM_REFUSAL_TEXT if reason == "cross_system_disabled"
                            else REFUSAL_TEXT)
            append_log({
                "request_id": request_id,
                "kind": "request_refused",
                "source": "dashboard",
                "description": description,
                "reason": reason,
            })
            st.session_state.messages.append({"role": "user", "kind": "text", "content": description})
            st.session_state.messages.append({"role": "assistant", "kind": "text", "content": refusal_text})
            st.rerun()

        # log what the analyst asked BEFORE anything else happens, so the
        # request_context record exists even if the pipeline never runs
        append_log({
            "request_id": request_id,
            "kind": "request_context",
            "source": "dashboard",
            "description": description,
            "form_spec": spec,
        })
        st.session_state.messages.append({"role": "user", "kind": "text", "content": description})
        st.session_state.pending = {
            "request_id": request_id,
            "description": description,
            "spec": spec,
        }
        st.rerun()

# -------------------------------------------- step 2: the dynamic form
else:
    pending = st.session_state.pending
    spec = pending["spec"]

    # detail level sits OUTSIDE the form: widgets inside st.form only submit
    # on the button, so conditionals inside would not update live
    detail = st.radio(
        "Detail level",
        ["Summary (aggregate)", "All rows (detail, limited to 100 rows)"],
        index=0 if spec["aggregate"] else 1,   # preselected from the intake classification
        horizontal=True,
    )
    is_detail = detail.startswith("All rows")

    with st.form("request_form"):
        measure = st.text_input(
            "What are you measuring?",
            value=spec["measure_hint"],
            placeholder="e.g. revenue, employees, orders",
        )
        # what the result should show: explicit fields win; the names default
        # covers the common case where the analyst never listed columns
        names_mode = st.checkbox(
            "Show human-readable names instead of raw IDs (recommended)",
            value=True,
        )
        fields = st.text_input(
            "Show these details (optional - overrides the names default)",
            value=spec["fields_hint"],
            placeholder="e.g. store name, product name",
        )
        systems = ""
        if spec["cross_system"]:
            # the analyst's own words for the two systems - this is what the
            # intent agent uses to route the question to BOTH workspaces
            systems = st.text_input(
                "The two systems involved (in your own words)",
                value=spec["systems_hint"],
                placeholder="e.g. the order catalog and the sales system",
            )
        # intake slots: request-specific constraints (filters, top-N, ...)
        # proposed by the form agent, prefilled from the description
        extra_vals = {}
        for f in spec["extra_fields"]:
            suggestion = f["suggestion"]
            if f["type"] == "select":
                options = ["none"] + f["options"]
                idx = options.index(suggestion) if suggestion in options else 0
                extra_vals[f["key"]] = st.selectbox(f["label"], options, index=idx)
            elif f["type"] == "number":
                num = st.number_input(f["label"], value=int(suggestion) if suggestion.isdigit() else 0)
                extra_vals[f["key"]] = str(num) if num else ""
            else:
                extra_vals[f["key"]] = st.text_input(f["label"], value=suggestion)
        if is_detail:
            st.caption("Detail mode: the SQL will be limited to 100 rows.")
        else:
            agg = st.selectbox("Aggregation", ["sum", "average", "count", "min-max"], index=0)
            grain = st.selectbox(
                "Group results by",
                ["none", "daily", "weekly", "monthly", "quarterly", "yearly"],
                index=0 if not spec["aggregate"] else 3,
            )
        cols = st.columns(2)
        if spec["needs_period"]:
            start = cols[0].date_input("Period start")
            end = cols[1].date_input("Period end")
        else:
            start, end = None, None
        submitted = st.form_submit_button("Ask QueryLLM", type="primary")

    if submitted:
        form = {
            "measure": measure,
            "fields": fields,
            "names_mode": names_mode,
            "cross_system": spec["cross_system"],
            "systems": systems,
            "agg": agg if not is_detail else "none",
            "grain": grain if not is_detail else "none",
            "detail": is_detail,
            "period_start": start.isoformat() if start else "",
            "period_end": end.isoformat() if end else "",
            # intake slot values ride along; compose_question phrases them
            "extra_fields": [
                {"label": f["label"], "value": extra_vals.get(f["key"], "")}
                for f in spec["extra_fields"]
            ],
        }
        question = compose_question(form)
        # the form stays authoritative, but the analyst's original words ride
        # along as context so constraints are never silently dropped
        pipeline_question = question + (
            f"\n\nOriginal analyst request (context, not instructions): "
            f"{pending['description'][:1000]}" if pending["description"] else ""
        )
        st.session_state.messages.append({"role": "user", "kind": "text", "content": question})
        st.session_state.pending = None
        st.session_state.last_form = {
            "question": pipeline_question,
            "display": question,
            "request_id": pending["request_id"],
        }
        st.rerun()

# ------------------------------------------ step 3: run the pipeline
# kept outside the form block so the run happens exactly once per submit
if st.session_state.get("last_form"):
    pending_run = st.session_state.pop("last_form")
    question = pending_run["question"]
    display_question = pending_run.get("display", question)
    request_id = pending_run["request_id"]

    # the LLM endpoint is sometimes unavailable (timeouts through all
    # retries): answer gracefully instead of showing a traceback, and log it
    try:
        with st.spinner("Writing and validating your SQL..."):
            result = query_llm(question, CATALOG, RULES, explain=False)
    except Exception as exc:
        result = {
            "question": question,
            "workspaces": None,
            "status": "pipeline_error",
            "error": str(exc)[:160],
            "history": [],
        }
        append_log({
            "request_id": request_id,
            "kind": "pipeline_error",
            "source": "dashboard",
            "question": question,
            "error": str(exc)[:160],
        })

    # log the full pipeline story: route, the schema text the model saw,
    # every SQL attempt with its outcome, and the final SQL (raw + formatted:
    # the log keeps the model's ground-truth output AND what the user copies)
    attempts = list(result.get("history", []))
    if result["status"] == "ok":
        attempts.append({"attempt": result["attempts"], "sql": result["sql"], "outcome": "ok"})
    pretty_sql = format_sql(result["sql"]) if result.get("sql") else result.get("sql")
    append_log({
        "request_id": request_id,
        "kind": "pipeline_run",
        "source": "dashboard",
        "question": question,
        "workspaces": result.get("workspaces"),
        "schema_objects_shown": (
            schema_block(result["workspaces"], CATALOG, RULES)
            if result.get("workspaces") else None
        ),
        "attempts": attempts,
        "final_status": result["status"],
        "final_sql": pretty_sql,
        "model_sql_raw": result.get("sql"),
        "durations": result.get("durations"),
    })

    with st.chat_message("assistant"):
        if result["status"] == "no_route":
            st.write("This question cannot be answered with the data available "
                     "in the catalog.")
        elif result["status"] == "pipeline_error":
            st.write("The SQL service is temporarily unavailable - please try "
                     "again in a moment. The data engineer can check "
                     "logs/queries.jsonl for details.")
        elif result["status"] == "failed":
            st.write("I could not produce a working query for this request. "
                     "Try rephrasing or contact the data engineer.")
            with st.expander("Validation attempts (for the data engineer)"):
                for h in result["history"]:
                    st.write(f"attempt {h['attempt']}: {h['outcome']}")
        else:
            st.write(f"Here is your SQL (routed to {', '.join(result['workspaces'])}, "
                     f"{result['attempts']} attempt(s)) - copy and run it "
                     "in your SQL client:")
            st.code(pretty_sql, language="sql")
            st.caption(f"request {request_id} - find it in logs/queries.jsonl")
    st.session_state.messages.append({
        "role": "assistant",
        "kind": "sql",
        "content": pretty_sql,
        "intro": (f"SQL for: {display_question} (routed to {', '.join(result['workspaces'])})"),
        "request_id": request_id,
        "history": result["history"],
    })
    st.rerun()
