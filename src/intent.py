# Intent agent (first defined in notebook/05_intent_agent.ipynb).
# Routes an analyst question to ONE workspace, or TWO only when the
# catalog records a connection between the pair. Never trusts the model:
# every answer is re-validated against catalog.yaml.
import json
import re

from src.llm import llm


def parse_json_object(text):
    # tolerate fences or extra prose: extract the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def connected_pair_exists(workspaces, catalog):
    # a two-workspace route is valid only if a catalog connection links them
    wanted = set(workspaces)
    for conn in catalog.get("connections", []):
        sides = {t.split(".", 1)[0] for t in conn["tables"]}
        if sides == wanted:
            return True
    return False


def intent_agent(question, catalog):
    # the prompt template lives INSIDE the function so this cell exports
    # verbatim to src/intent.py (no module-level constants to carry along)
    intent_prompt = """You route analyst questions to the right data workspace.

Workspaces (name: what it contains):
{workspace_list}

Connections between workspaces (pairs that may be combined):
{connection_list}

Rules:
- Pick ONE workspace for a normal question.
- Pick TWO only if the question genuinely needs data from both AND the pair
  is listed above as connected.
- If no workspace can answer the question, pick null.
Answer with ONLY a JSON object, no other text:
{{"workspaces": ["CO"]}}   or   {{"workspaces": ["HR", "SH"]}}   or   {{"workspaces": null}}"""
    workspace_list = "\n".join(
        f"- {name}: {ws['description']}" for name, ws in catalog["workspaces"].items()
    )
    # the catalog policy decides whether two-workspace routing is allowed at
    # all: when cross-schema joins are disabled the model must never see the
    # candidate pairs, so it cannot be tempted into a two-workspace route
    if catalog["policy"].get("cross_schema_joins") == "disallowed":
        connection_list = "- none (cross-schema joins are disabled)"
    else:
        pairs = [
            f"- {' <-> '.join(conn['tables'])} (join key: {conn['join_key']})"
            for conn in catalog.get("connections", [])
        ]
        connection_list = "\n".join(pairs) if pairs else "- none"
    prompt = (
        intent_prompt.format(workspace_list=workspace_list, connection_list=connection_list)
        + f"\n\nQuestion: {question}"
    )
    answer = llm(
        prompt,
        system="You are a routing agent.",
        session="ses_queryllm_intent",
        temperature=0,
        # routing keeps thinking ON: its reasoning chain is short (~1-2s)
        # and measurably improves routing accuracy; the writer/validator
        # run with thinking OFF because their chains cost 40s+
        enable_thinking=True,
    )
    parsed = parse_json_object(answer) or {}
    workspaces = parsed.get("workspaces")

    # normalize + validate: every name must exist in the catalog
    if isinstance(workspaces, str):
        workspaces = [workspaces]
    if not isinstance(workspaces, list):
        workspaces = []
    workspaces = [w for w in workspaces if w in catalog["workspaces"]]
    workspaces = list(dict.fromkeys(workspaces))          # keep order, drop dupes

    # enforce the catalog policy maximum (2 when connections exist)
    max_ws = catalog["policy"].get("max_workspaces_per_question", 1)
    workspaces = workspaces[:max_ws]

    if len(workspaces) == 2 and not connected_pair_exists(workspaces, catalog):
        # never route blind: fall back to the single strongest workspace
        print(f"  note: {workspaces} not connected in catalog -> falling back to first")
        workspaces = workspaces[:1]

    return workspaces or None
