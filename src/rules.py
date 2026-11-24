# PDP rules helpers (first defined in notebook/03_sql_writer.ipynb).
# render_rules turns catalog/rules.yaml into prompt text; pdp_class
# classifies a column by word-boundary name patterns.
import re


def render_rules(rules):
    pdp = rules["pdp"]
    hints = pdp["classification_hints"]
    lines = ["PDP rules (columns are classified by name patterns):"]
    for cls in ("specific", "direct", "indirect"):
        pats = ", ".join(hints.get(cls, []))
        lines.append(f"- {cls.upper()} = column names matching: {pats}")
        lines.append(f"    handling -> {pdp['handling'][cls]}")
    lines.append("- everything else is NON_PERSONAL: no restriction")
    lines.append("Query rules:")
    lines.extend(f"- {rule}" for rule in rules["query_rules"])
    return "\n".join(lines)


def pdp_class(column_name, rules):
    # word-boundary match on the lowercased column name
    hints = rules["pdp"]["classification_hints"]
    name = column_name.lower()
    for cls in ("specific", "direct", "indirect"):
        for pattern in hints.get(cls, []):
            if re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", name):
                return cls.upper()
    return None
