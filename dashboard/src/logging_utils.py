# Dashboard logging helpers (dashboard-only).
# Every analyst request is logged to <project root>/logs/queries.jsonl as a
# pretty-printed JSON record. Records are separated by a "---" line so the
# file is readable by humans AND parseable by scripts.
# Never logged: row data, credentials, full result sets.
import json
import time
from pathlib import Path

# dashboard/src/../../ -> project root (self-contained, no cwd dependency)
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_PATH = LOG_DIR / "queries.jsonl"
SEPARATOR = "\n---\n"


def append_log(record):
    # stamp a timestamp and append one pretty-printed record + separator
    stamped = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
    LOG_DIR.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stamped, indent=2, ensure_ascii=False, default=str))
        fh.write("\n---\n")
    return stamped


def read_recent(last=5, path=None):
    # return the most recent records (oldest -> newest), parsed back to dicts;
    # undecodable chunks are skipped instead of crashing the reader
    path = Path(path) if path else LOG_PATH
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    records = []
    for chunk in text.split("\n---\n"):
        if not chunk.strip():
            continue
        try:
            records.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return records[-last:]
