# SQL writer agent (first defined in notebook/03_sql_writer.ipynb).
# schema_block renders workspace(s) + PDP tags + FK/connection join hints;
# sql_rules renders the structural SQL rules; sql_writer is the agent.
import re

from src.llm import llm
from src.rules import pdp_class, render_rules


def schema_block(workspace_names, catalog, rules):
    names = [workspace_names] if isinstance(workspace_names, str) else list(workspace_names)
    lines = []
    for ws_name in names:
        ws = catalog["workspaces"][ws_name]
        # always qualify: SCHEMA.TABLE makes the generated SQL portable to any
        # SQL client (the analyst copies it out of the dashboard)
        prefix = f"{ws_name}."
        for table_name, table in ws["tables"].items():
            lines.append(f"{prefix}{table_name} ({table['object_type']}): {table['description'] or '-'}")
            for c in table["columns"]:
                cls = pdp_class(c["name"], rules)
                tag = f" [{cls}]" if cls else ""
                part = f"{c['name']} {c['type']}{tag}" + ("" if c["description"] == "" else f" -- {c['description']}")
                lines.append(f"    {part}")
            if table["primary_key"]:
                lines.append(f"    primary key: {', '.join(table['primary_key'])}")
            for fk in table["foreign_keys"]:
                if not fk["reference_resolved"]:
                    continue
                if len(names) == 1 or fk["references_schema"] == ws_name:
                    ref_prefix = "" if fk["references_schema"] == ws_name else fk["references_schema"] + "."
                    lines.append(
                        f"    FK {','.join(fk['columns'])} -> "
                        f"{ref_prefix}{fk['references_table']}.({','.join(fk['references_columns'])})"
                    )
    if len(names) > 1:
        for ws_name in names:
            for tname, table in catalog["workspaces"][ws_name]["tables"].items():
                for fk in table["foreign_keys"]:
                    if fk["reference_resolved"] and fk["references_schema"] != ws_name and fk["references_schema"] in names:
                        lines.append(
                            f"verified cross-workspace join: "
                            f"{ws_name}.{tname}.({','.join(fk['columns'])}) = "
                            f"{fk['references_schema']}.{fk['references_table']}.({','.join(fk['references_columns'])})"
                        )
        # candidate connections recorded in the catalog (never invented by the
        # model). 'unresolved' means the two schemas use different key names,
        # so the model must compare via descriptive columns instead of joining.
        for conn in catalog.get("connections", []):
            sides = {t.split(".", 1)[0] for t in conn["tables"]}
            if len(sides) == 2 and sides <= set(names):
                key = conn.get("join_key")
                if key and key != "unresolved":
                    lines.append(
                        f"candidate connection: {conn['tables'][0]} = {conn['tables'][1]} "
                        f"on {key} (candidate - the join values must match on both sides)"
                    )
                else:
                    lines.append(
                        f"candidate connection: {conn['tables'][0]} <-> {conn['tables'][1]} "
                        "has NO reliable join key - do not invent one; compare via descriptive columns only"
                    )
    return "\n".join(lines)


def preselect_tables(question, workspace_name, catalog, max_tables=6):
    # the helpers live INSIDE the function so this cell exports verbatim to
    # src/writer.py (no module-level constants to carry along)
    def _tokens(text):
        return set(re.findall(r"[a-z]+", (text or "").lower()))

    def _hit(token, word):
        # plural-tolerant keyword match: store ~ stores, order ~ orders,
        # and prefix matches from 4+ chars (product ~ products)
        t, w = token.rstrip("s"), word.rstrip("s")
        return t == w or (len(t) >= 4 and (w.startswith(t) or t.startswith(w)))
    # Deterministic table preselection (NO LLM call): keep the tables the
    # question plausibly needs plus their FK-linked neighbors, so the writer
    # prompt shrinks and generation gets faster. Skipped entirely when the
    # whole workspace already fits (CO/HR keep all their tables).
    # Returns (pruned_tables, ranked_names) - (None, None) when nothing to prune.
    tables = catalog["workspaces"][workspace_name]["tables"]
    if len(tables) <= max_tables:
        return None, None
    tokens = _tokens(question)
    scores = {}
    for tname, table in tables.items():
        score = 0
        for t in tokens:
            if any(_hit(t, w) for w in _tokens(tname)):
                score += 3
            elif any(_hit(t, w) for w in _tokens(" ".join(c["name"] for c in table["columns"]))):
                score += 2
            elif any(_hit(t, w) for w in _tokens(table["description"])):
                score += 1
        scores[tname] = score
    ranked = sorted(scores, key=scores.get, reverse=True)
    keep = set(ranked[:max_tables])
    # FK neighbors of kept tables count as reachable context when they scored > 0
    for tname in list(keep):
        for fk in tables[tname]["foreign_keys"]:
            ref = fk["references_table"]
            if fk["reference_resolved"] and ref in tables and scores.get(ref, 0) > 0:
                keep.add(ref)
    pruned = {t: tables[t] for t in sorted(keep)}
    return pruned, ranked


def sql_rules(policy):
    if policy["cross_schema_joins"] in ("verified_fk_only", "catalog_connections_only"):
        # catalog_connections_only: joins come from candidate connections
        # recorded in catalog.yaml (never invented by the model)
        cross = ("Cross-schema joins are allowed, but ONLY using the joins listed "
                 "in the schema (verified cross-workspace joins or candidate "
                 "connections with their recorded key), and table names must be "
                 "qualified as SCHEMA.TABLE.")
    else:
        cross = "Cross-schema joins are NOT allowed: answer within ONE workspace."
    return f"""Rules:
- Write ONE query in Oracle SQL (12c+ syntax, FETCH FIRST for limits).
- SELECT or WITH only. Never INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE.
- Use only tables and columns listed in the schema. Never invent names.
- Always qualify table names as SCHEMA.TABLE (e.g. CO.ORDERS) so the SQL runs unchanged in any SQL client.
- {cross}
- Return ONLY the SQL query, no explanations, no markdown fences."""


def sql_writer(question, workspace_names, catalog, rules):
    names = [workspace_names] if isinstance(workspace_names, str) else list(workspace_names)
    # respect the policy: multiple workspaces only when the catalog allows it
    if len(names) > 1 and catalog["policy"]["cross_schema_joins"] == "disallowed":
        raise ValueError("policy disallows cross-schema joins: pass a single workspace")
    # deterministic preselection: shrink the prompt to the tables the
    # question plausibly needs (single workspace only; multi-workspace is
    # policy-gated and dormant). When everything fits, the view is unchanged.
    used_catalog = catalog
    if len(names) == 1:
        pruned, _ = preselect_tables(question, names[0], catalog)
        if pruned is not None:
            used_catalog = {
                **catalog,
                "workspaces": {
                    names[0]: {**catalog["workspaces"][names[0]], "tables": pruned}
                },
            }
    schema = schema_block(names, used_catalog, rules)
    examples = "\n\n".join(
        f"Example: {ex['question']}\n{ex['sql']}"
        for ex in catalog["workspaces"][names[0]].get("examples", [])
    )
    label = ", ".join(names) if len(names) > 1 else names[0]
    rules_text = render_rules(rules)
    prompt = (
        f"{rules_text}\n\n"
        f"{sql_rules(catalog['policy'])}\n\n"
        f"Schema ({label}):\n{schema}\n\n"
        + (f"Few-shot examples:\n{examples}\n\n" if examples else "")
        + f"Question: {question}\n"
        "SQL:"
    )
    answer = llm(prompt, system="You are an expert SQL writer.", session="ses_queryllm_sql")
    # strip markdown fences if the model added them anyway
    answer = re.sub(r"^```(?:sql)?\s*|\s*```$", "", answer.strip(), flags=re.IGNORECASE)
    return answer.strip()
