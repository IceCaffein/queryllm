# Validator agent (first defined in notebook/04_validator.ipynb).
# is_safe_sql + execute_readonly are the deterministic guard every SQL
# attempt must pass; repair_sql + validator_agent are the repair loop.
import os
import re

import sqlglot
from sqlglot import exp

import oracledb

from src.llm import llm
from src.rules import render_rules
from src.writer import schema_block, sql_rules


def is_safe_sql(sql):
    # must start with SELECT or WITH and contain no write/DDL keywords.
    # the pattern lives INSIDE the function so this cell exports verbatim to
    # src/validator.py (no module-level constants to carry along).
    if not re.match(r"\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return False
    blocked = re.compile(
        r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
        re.IGNORECASE,
    )
    return not blocked.search(sql)


def static_check(sql, workspace_names, catalog):
    # DB-free validation: guard, Oracle parse, and a catalog identifier check.
    # Catches the invalid-identifier error class (e.g. ORA-00904 on a column
    # the catalog never mentioned) without touching the database.
    # the builtin set lives INSIDE the function so this cell exports verbatim
    _BUILTIN_IDENTIFIERS = {"ROWNUM", "SYSDATE", "SYSTIMESTAMP", "LEVEL", "NULL", "DUAL"}
    sql = sql.strip().rstrip(";").strip()
    if not is_safe_sql(sql):
        return "not a SELECT/WITH statement (or contains write/DDL keywords)"
    try:
        tree = sqlglot.parse_one(sql, read="oracle")
    except Exception as exc:
        return f"parse error: {str(exc)[:140]}"
    names = [workspace_names] if isinstance(workspace_names, str) else list(workspace_names)
    known_tables, known_columns = {"DUAL"}, set()
    for ws in names:
        for tname, table in catalog["workspaces"][ws]["tables"].items():
            known_tables.add(tname.upper())
            for c in table["columns"]:
                known_columns.add(c["name"].upper())
    cte_aliases = {c.alias_or_name.upper() for c in tree.find_all(exp.CTE)}
    output_aliases = {a.alias_or_name.upper() for a in tree.find_all(exp.Alias)}
    used_tables = {t.name.upper().split(".")[-1] for t in tree.find_all(exp.Table)}
    unknown = used_tables - known_tables - cte_aliases
    if unknown:
        return f"unknown tables: {', '.join(sorted(unknown))}"

    # per-table column availability: real tables -> their catalog columns;
    # subquery/CTE references are skipped (the table check covers their names)
    tables_cols = {}
    for ws in names:
        for tname, table in catalog["workspaces"][ws]["tables"].items():
            tables_cols[tname.upper()] = {c["name"].upper() for c in table["columns"]}
    alias_map = {}
    for t in tree.find_all(exp.Table):
        tname = t.name.upper().split(".")[-1]
        if t.alias:
            alias_map[t.alias.upper()] = tname
        alias_map.setdefault(tname, tname)

    # qualified columns must exist on the table their alias resolves to
    for col in tree.find_all(exp.Column):
        if col.table:
            tname = alias_map.get(col.table.upper())
            if tname and tname in tables_cols and col.name.upper() not in tables_cols[tname]:
                return f"unknown column {col.name} for table {tname}"

    # unqualified columns inside single-source selects must belong to that
    # source table - this is what catches made-up columns in CTE bodies
    for sel in tree.find_all(exp.Select):
        # sqlglot 30.x stores the FROM under the key "from_"
        src_tables = [t for t in (sel.args.get("from_") or exp.select("1")).find_all(exp.Table)]
        for j in (sel.args.get("joins") or []):
            src_tables += [t for t in j.find_all(exp.Table)]
        src_names = {t.name.upper().split(".")[-1] for t in src_tables}
        if len(src_names) != 1:
            continue
        src = next(iter(src_names))
        if src not in tables_cols:
            continue
        allowed = (tables_cols[src]
                   | {a.alias_or_name.upper() for a in sel.find_all(exp.Alias)}
                   | output_aliases | cte_aliases | _BUILTIN_IDENTIFIERS)
        for col in sel.find_all(exp.Column):
            if col.name.upper() not in allowed:
                return f"unknown column {col.name} for table {src}"

    used_columns = {c.name.upper() for c in tree.find_all(exp.Column)}
    unknown = used_columns - known_columns - cte_aliases - output_aliases - _BUILTIN_IDENTIFIERS
    if unknown:
        return f"unknown columns: {', '.join(sorted(unknown))}"
    return None


def execute_readonly(sql, workspace_names, catalog, max_rows=20):
    # a trailing semicolon is not allowed inside the wrapper subquery
    sql = sql.strip().rstrip(";").strip()
    if not is_safe_sql(sql):
        raise ValueError("blocked: not a SELECT/WITH statement (or contains write/DDL keywords)")
    names = [workspace_names] if isinstance(workspace_names, str) else list(workspace_names)
    wrapped = f"SELECT * FROM ({sql}) WHERE ROWNUM <= {max_rows}"
    with oracledb.connect(user=os.getenv("ORA_USER"),
                          password=os.getenv("ORA_PASSWORD"),
                          dsn=os.getenv("ORA_DSN")) as conn:
        with conn.cursor() as cur:
            # single-workspace queries use unqualified names: set the session
            # schema. Multi-workspace queries are SCHEMA.TABLE qualified.
            if len(names) == 1:
                ws = names[0]
                if not re.fullmatch(r"[A-Za-z0-9_]+", ws) or ws not in catalog["workspaces"]:
                    raise ValueError(f"blocked: invalid workspace {ws!r}")
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {ws}")
            cur.execute(wrapped)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return columns, rows


def repair_sql(question, sql, problem, workspace_names, catalog, rules):
    schema = schema_block(workspace_names, catalog, rules)
    rules_text = render_rules(rules)
    prompt = (
        "You fix SQL queries.\n\n"
        f"{rules_text}\n\n"
        f"{sql_rules(catalog['policy'])}\n\n"
        f"Question: {question}\n\n"
        f"Schema:\n{schema}\n\n"
        f"Previous SQL:\n{sql}\n\n"
        f"Problem:\n{problem}\n\n"
        "Return ONLY the corrected SQL query."
    )
    answer = llm(prompt, system="You fix SQL queries.", session="ses_queryllm_validator")
    answer = re.sub(r"^```(?:sql)?\s*|\s*```$", "", answer.strip(), flags=re.IGNORECASE)
    return answer.strip().rstrip(";").strip()


def validator_agent(question, sql, workspace_names, catalog, rules, max_retries=3, execute=True):
    # Two modes:
    # - execute=False (the web pipeline): DB-free - static checks only, so no
    #   query path can ever touch the database
    # - execute=True (notebook testing): static checks first, then a read-only
    #   DB round-trip for runtime errors and empty-result detection
    history = []
    for attempt in range(1, max_retries + 1):
        problem = static_check(sql, workspace_names, catalog)
        if problem is None and execute:
            try:
                columns, rows = execute_readonly(sql, workspace_names, catalog)
                if rows:
                    return {
                        "status": "ok",
                        "sql": sql,
                        "columns": columns,
                        "rows": rows,
                        "attempts": attempt,
                        "history": history,
                        "validated": "executed",
                    }
                problem = ("The query executed but returned 0 rows. Reconsider the "
                           "filters, joins and date logic so the question is answered "
                           "with the data that exists.")
                history.append({"attempt": attempt, "sql": sql, "outcome": "empty"})
            except Exception as exc:
                problem = str(exc).strip() or type(exc).__name__
                history.append({"attempt": attempt, "sql": sql, "outcome": f"error: {problem[:160]}"})
        elif problem is None:
            return {
                "status": "ok",
                "sql": sql,
                "attempts": attempt,
                "history": history,
                "validated": "static",
            }
        else:
            history.append({"attempt": attempt, "sql": sql, "outcome": f"static: {problem[:160]}"})
        if attempt == max_retries:
            break
        print(f"  attempt {attempt} -> {history[-1]['outcome'][:60]} | asking the LLM to repair")
        sql = repair_sql(question, sql, problem, workspace_names, catalog, rules)
    return {"status": "failed", "sql": sql, "history": history}
