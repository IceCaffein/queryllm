# Validator agent (first defined in notebook/04_validator.ipynb).
# is_safe_sql + execute_readonly are the deterministic guard every SQL
# attempt must pass; repair_sql + validator_agent are the repair loop.
import os
import re

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


def validator_agent(question, sql, workspace_names, catalog, rules, max_retries=3):
    history = []
    for attempt in range(1, max_retries + 1):
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
                }
            problem = ("The query executed but returned 0 rows. Reconsider the "
                       "filters, joins and date logic so the question is answered "
                       "with the data that exists.")
            history.append({"attempt": attempt, "sql": sql, "outcome": "empty"})
        except Exception as exc:
            problem = str(exc).strip() or type(exc).__name__
            history.append({"attempt": attempt, "sql": sql, "outcome": f"error: {problem[:160]}"})
        if attempt == max_retries:
            break
        print(f"  attempt {attempt} -> {history[-1]['outcome'][:60]} | asking the LLM to repair")
        sql = repair_sql(question, sql, problem, workspace_names, catalog, rules)
    return {"status": "failed", "sql": sql, "history": history}
