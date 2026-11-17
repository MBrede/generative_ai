"""Precompute data/chunking_demo.json for the interactive chunking figure in content/embeddings.qmd.
Uses the default text and sentence splitting of chunking_app.py. Run from script/:
uv run --env-file example_solutions/session10/.env --with openai,numpy _src/chunking_demo_data.py
"""
import ast
import json
import os
import re

import numpy as np
from openai import OpenAI

EMB_MODEL = "qwen3-embedding-0-6b"
APP = os.path.join(os.path.dirname(__file__), "chunking_app.py")
tree = ast.parse(open(APP).read())
text = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "DEFAULT")

rows = []  # same splitting as parse() in chunking_app.py
for i, para in enumerate(p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()):
    m = re.match(r"\s*([\w ]{1,30}):\s+(.*)", para, re.S)
    topic, body = (m[1], m[2]) if m else (f"paragraph {i + 1}", para)
    rows += [dict(topic=topic, sentence=s) for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s]

client = OpenAI(base_url=os.environ["URL"], api_key=os.environ["KEY"])
e = np.array([d.embedding for d in client.embeddings.create(model=EMB_MODEL, input=[r["sentence"] for r in rows]).data])
e /= np.linalg.norm(e, axis=1, keepdims=True)
out = dict(model=EMB_MODEL, sentences=rows, similarity=np.round(e @ e.T, 4).tolist())
dst = os.path.join(os.path.dirname(__file__), "..", "data", "chunking_demo.json")
json.dump(out, open(dst, "w"), indent=1)
print(f"{len(rows)} sentences -> {os.path.normpath(dst)}")
