# SQL presentation formatter (dashboard-only).
# The chat displays formatted SQL; the log keeps the raw model output as
# ground truth. sqlglot parses with the Oracle dialect and pretty-prints -
# it may normalize trivial wording (e.g. 'x IS NOT NULL' -> 'NOT x IS NULL'),
# which is semantically identical, and it FAILS OPEN: if it cannot parse the
# SQL, the original text is returned unchanged so an answer is never blocked
# on cosmetics.
import sqlglot


def format_sql(sql):
    try:
        return sqlglot.transpile(sql, read="oracle", write="oracle", pretty=True)[0]
    except Exception:
        return sql
