#!/usr/bin/env python3
import json
import pathlib
import re

import requests

text = pathlib.Path("scripts/LLM_request.py").read_text(encoding="utf-8")
m = re.search(r'ak\s*=\s*"([^"]+)"', text)
assert m and m.group(1) != "****", "AK not found"
ak = m.group(1)
pathlib.Path("revision/artifacts/.bytedance_ak").write_text(ak, encoding="utf-8")
print("ak_len", len(ak))

url = "https://search.bytedance.net/gpt/openapi/online/v2/crawl"
data = {
    "messages": [
        {"role": "system", "content": "Reply with only one letter."},
        {
            "role": "user",
            "content": "What is 2+2? Options:\nA. 3\nB. 4\nC. 5\nD. 6\nOnly tell me the final answer.",
        },
    ],
    "model": "gpt-4o-2024-11-20",
    "max_tokens": 8,
    "temperature": 0,
    "top_p": 0,
    "n": 1,
    "stream": False,
}
try:
    r = requests.post(
        f"{url}?ak={ak}",
        headers={"Content-Type": "application/json"},
        json=data,
        timeout=60,
    )
    print("status", r.status_code)
    print("body_head", r.text[:400])
except Exception as e:
    print("ERR", type(e).__name__, e)
