# Form agent (dashboard-only).
# Turns the analyst's free-text description into a small structured form
# spec, and turns the filled form back into ONE analyst-language question.
# This lives in dashboard/src - it is a UI concern, not part of the core
# pipeline; it only imports the shared LLM wrapper from the core src/.
import json
import re

from src.llm import llm

# ---------------------------------------------------------------------------
# Request gate, layer 1: deterministic forbidden-pattern check.
# These are destructive/mutating INTENT phrases, matched as combinations so
# benign lookalikes ("create a monthly report") pass. Layer 2 (the LLM
# request_type inside analyze_description) catches paraphrases the regex misses.
_FORBIDDEN_PATTERNS = [
    r"\bdrop\s+\S",
    r"\btruncate\b",
    r"\bdelete\b",
    r"\binsert\s+into\b",
    r"\bmerge\s+into\b",
    r"\bupdate\s+(the\s+)?\w*table\b",
    r"\balter\s+(table|view|user|schema)\b",
    r"\brecreate\b",
    r"\brebuild\b",
    r"\bgrant\b",
    r"\brevoke\b",
    r"\bclean(ing)?\s+up\b",
    r"\bwipe\b",
]
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in _FORBIDDEN_PATTERNS]

# shown in the chat when the gate refuses a request
REFUSAL_TEXT = (
    "This tool only answers read-only data questions. Schema or data changes "
    "(drop, recreate, insert, update, delete) are not possible here - please "
    "contact the data engineer."
)

# shown when the analyst asks a cross-system question while the catalog
# policy disables cross-schema matching (wording set by the data engineer)
CROSS_SYSTEM_REFUSAL_TEXT = (
    "This request compares data across two systems, but cross-system matching "
    "is currently disabled: no reliable connection exists between the schemas. "
    "Ask about one data domain at a time"
)


def is_forbidden_request(description):
    # True when any forbidden pattern matches - refuse WITHOUT an LLM call
    return any(rx.search(description) for rx in _FORBIDDEN_RE)


def analyze_description(description):
    # Intake pattern: ONE llm call classifies the request AND proposes 3-6
    # request-specific form slots (filters, top-N, ...) so constraints in the
    # description ("only for New York") reach the pipeline instead of being
    # dropped by a fixed form. Strict validation: malformed slots are dropped
    # and a filters slot is always offered; parse failure -> fixed form.
    prompt = (
        "Act as an intake agent for a text-to-SQL dashboard. Read the data "
        "analyst's description, classify it, and design the follow-up form.\n\n"
        "Return ONLY a JSON object with exactly these keys:\n"
        '{"request_type": "data_request"|"schema_change"|"data_change", '
        '"needs_period": true/false, "aggregate": true/false, '
        '"cross_system": true/false, '
        '"measure_hint": "<2-4 words in the analyst\'s own language>", '
        '"fields_hint": "<comma-separated list of what the result should show, '
        "in the analyst's words>\", "
        '"systems_hint": "<the analyst\'s own words for the two systems involved>", '
        '"extra_fields": [<0-6 slots, see rules>]}\n\n'
        "Rules:\n"
        "- request_type: 'data_request' when the analyst asks to read/select "
        "data; 'schema_change' for DDL (drop, recreate, alter, create, rename "
        "objects); 'data_change' for mutating rows (insert, update, delete, "
        "truncate, merge, wipe)\n"
        "- needs_period: true when the description mentions any time span "
        "(last month, this year, between dates, per month, ...)\n"
        "- aggregate: false ONLY when the analyst clearly wants detail "
        "(all rows, every record, list each ...)\n"
        "- cross_system: true when the description compares, matches or links "
        "data across TWO systems/catalogs/databases\n"
        "- measure_hint: the thing being measured in the analyst's words, "
        "empty string if unclear\n"
        "- fields_hint: only what the analyst explicitly asks the result to "
        "show; empty string otherwise\n"
        "- systems_hint: the analyst's own words for the two systems; empty "
        "string unless cross_system is true\n"
        "- extra_fields: slots for what the fixed keys do NOT capture - "
        "filters/conditions (city, status, category, exclusions), top-N, "
        "sorting, minimums. Each slot: {\"key\", \"label\", \"type\": "
        "\"text\"|\"select\"|\"number\", \"options\": [only for select], "
        "\"suggestion\"}. ALWAYS include first: {\"key\": \"filters\", "
        '"label": "Any filters or conditions?", "type": "text", '
        '"options": [], "suggestion": "<value from the description, or empty>"}. '
        "Labels and suggestions in the analyst's own words - NEVER schema, "
        "table or column names.\n\n"
        f"Description: {description}"
    )
    answer = llm(
        prompt,
        system="You structure data requests.",
        session="ses_queryllm_form",
        temperature=0,
    )
    spec = {}
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if match:
        try:
            spec = json.loads(match.group(0))
        except json.JSONDecodeError:
            spec = {}
    # conservative defaults: read-only data request, summary view, no period,
    # empty hints, single system - unknown never blocks the form (layer 1
    # regex still guards)
    request_type = str(spec.get("request_type", "data_request") or "data_request")
    if request_type not in ("data_request", "schema_change", "data_change"):
        request_type = "data_request"
    return {
        "request_type": request_type,
        "needs_period": bool(spec.get("needs_period", False)),
        "aggregate": bool(spec.get("aggregate", True)),
        "cross_system": bool(spec.get("cross_system", False)),
        "measure_hint": str(spec.get("measure_hint", "") or ""),
        "fields_hint": str(spec.get("fields_hint", "") or ""),
        "systems_hint": str(spec.get("systems_hint", "") or ""),
        "extra_fields": _valid_extra_fields(spec.get("extra_fields")),
    }


def _valid_extra_fields(raw):
    # whitelist slot types, drop malformed entries, cap at 6, and guarantee
    # the filters slot exists (the most common lossy-constraint class)
    fields = []
    for entry in (raw or [])[:6]:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", "") or "").strip()
        ftype = str(entry.get("type", "text") or "text")
        if not label or ftype not in ("text", "select", "number"):
            continue
        fields.append({
            "key": (str(entry.get("key") or label.lower().replace(" ", "_")))[:40],
            "label": label,
            "type": ftype,
            "options": [str(o) for o in (entry.get("options") or []) if str(o).strip()][:8]
                       if ftype == "select" else [],
            "suggestion": str(entry.get("suggestion", "") or ""),
        })
    if not any(f["key"] == "filters" for f in fields):
        fields.insert(0, {
            "key": "filters",
            "label": "Any filters or conditions?",
            "type": "text",
            "options": [],
            "suggestion": "",
        })
    return fields[:6]


# aggregation wordings: form value -> analyst-language phrase
_AGG_PHRASES = {
    "sum": "the sum of",
    "average": "the average",
    "count": "the count of",
    "min-max": "the minimum and maximum",
}

# grain form values are adjectives (monthly) - "per" needs the noun (month)
_GRAIN_NOUNS = {
    "daily": "day",
    "weekly": "week",
    "monthly": "month",
    "quarterly": "quarter",
    "yearly": "year",
}


def compose_question(form):
    # Build ONE analyst-eye sentence from the filled form. This is the exact
    # text the core pipeline agents receive - no schema names appear here.
    # Detail mode bakes the 100-row cap into the request so the generated
    # SQL comes out with FETCH FIRST 100 ROWS ONLY.
    #
    # Suffix order (applies in BOTH detail and summary mode):
    #   period -> cross-system scope -> what the result should show
    measure = (form.get("measure") or "").strip() or "requested data"

    showing = ""
    fields = (form.get("fields") or "").strip()
    if fields:
        showing = f", showing {fields}"
    elif form.get("names_mode", True):
        showing = ", showing descriptive names instead of raw IDs where available"

    start = (form.get("period_start") or "").strip()
    end = (form.get("period_end") or "").strip()
    period = f" between {start} and {end}" if start and end else ""

    # cross-system scope: the analyst's own words for the two systems - this
    # is what lets the intent agent route to BOTH workspaces (it only routes
    # a pair when the catalog records a connection between them)
    cross = ""
    if form.get("cross_system"):
        systems = (form.get("systems") or "").strip()
        if systems:
            cross = f", matching data across the two systems: {systems}"
        else:
            cross = ", matching data across the two systems involved"

    # intake slots (filters, top-N, ...): every filled slot becomes one
    # readable phrase - this is how constraints like "only New York" survive
    extras = ""
    for f in form.get("extra_fields") or []:
        value = str(f.get("value") or "").strip()
        if value and value.lower() != "none":
            extras += f", {str(f.get('label', 'condition')).strip().rstrip('?')}: {value}"

    if form.get("detail"):
        return (f"Show me the {measure} as detail rows, limited to 100 rows"
                f"{period}{cross}{showing}{extras}")

    agg = (form.get("agg") or "none").strip().lower()
    if agg in _AGG_PHRASES:
        sentence = f"Show me {_AGG_PHRASES[agg]} {measure}"
    else:
        sentence = f"Show me the {measure}"

    grain = (form.get("grain") or "none").strip().lower()
    # skip the grain phrase when the measure already says it ("revenue per month")
    if grain != "none" and _GRAIN_NOUNS.get(grain, grain) not in measure.lower():
        sentence += f" per {_GRAIN_NOUNS.get(grain, grain)}"

    return f"{sentence}{period}{cross}{showing}{extras}"
