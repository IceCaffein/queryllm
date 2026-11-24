# QueryLLM LLM wrapper (first defined in notebook/03_sql_writer.ipynb).
# Reads credentials from .env at call time: LLM_URL, SQL_AGENT (API key),
# LLM_MODEL. 3 attempts with growing backoff - endpoints are sometimes slow.
import os
import time

import requests


def llm(prompt, system="You are an expert SQL writer.", session="ses_queryllm",
        temperature=0.2, enable_thinking=False):
    payload = {
        "model": os.getenv("LLM_MODEL", "qwen3.8-flash"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        # qwen3.8-flash THINKS by default: ~12k hidden reasoning tokens were
        # 90% of the latency (55s -> 4-6s with thinking off). SQL generation
        # does not need it; the validator repair loop is the safety net.
        # EXCEPTION: the intent agent keeps thinking ON (short chain, better
        # routing accuracy) - it passes enable_thinking=True explicitly.
        "enable_thinking": enable_thinking,
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('SQL_AGENT')}",
        "x-opencode-session": session,
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(
                os.getenv("LLM_URL"), headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            print(f"  llm call failed (attempt {attempt + 1}/3), retrying...")
            time.sleep(5 * (attempt + 1))
    raise last_error
