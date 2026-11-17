# /// script
# requires-python = ">=3.11"
# dependencies = ["streamlit", "streamlit-keyup", "openai", "umap-learn", "plotly", "pandas"]
# ///
"""Live next-token explorer: type a prompt, see its tokens and the next-token distribution.
Needs an OpenAI-compatible /completions endpoint with echo + logprobs (vLLM). Env: URL (incl. /v1), KEY. Run from script/:
uv run --env-file example_solutions/session10/.env --with streamlit,streamlit-keyup,openai,umap-learn,plotly,pandas \
  streamlit run _src/prompt_focus_app.py
"""
import html
import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from st_keyup import st_keyup

GEN_MODEL = "qwen3-8-27b"  # always warm on the course endpoint
EMB_MODEL = "qwen3-embedding-0-6b"
ANCHORS = """regression, residuals, variance, heteroscedasticity, outlier, correlation, coefficient, distribution
Python, function, variable, error, library, install, debug, code
power, battery, cable, voltage, switch, plug, device, charger
doctor, symptoms, pain, fever, diagnosis, medicine, patient, blood
recipe, oven, salt, flour, bake, taste, kitchen, sauce
bank, money, interest, loan, account, stock, price, budget
engine, oil, tire, brake, fuel, car, mechanic, warning
tree, river, mountain, weather, rain, animal, forest, sun"""
st.set_page_config(layout="wide", page_title="Next-token explorer")
client = OpenAI(base_url=os.environ.get("URL", "http://localhost:11434/v1"), api_key=os.environ.get("KEY", "ollama"))
ss = st.session_state
ss.setdefault("text", "I am fitting a linear regression in Python.\\nQ: What should I check first?\\nA:")
ss.setdefault("ver", 0)
ss.setdefault("history", [])  # list of DataFrames: embedded candidates per press


def show(t):
    return t.replace("\n", "⏎").replace(" ", "·")


@st.cache_data(max_entries=200)
def analyse(model, prompt, k):
    """One call: prompt tokens with their logprobs + top-k distribution for the next token."""
    lp = client.completions.create(model=model, prompt=prompt, max_tokens=1, temperature=0, logprobs=k, echo=True).choices[0].logprobs
    n = len(lp.tokens) - 1  # last entry is the generated token
    toks = list(zip(lp.tokens[:n], lp.token_logprobs[:n]))
    nxt = pd.DataFrame(sorted(((t, math.exp(v)) for t, v in lp.top_logprobs[n].items()), key=lambda x: -x[1]), columns=["token", "p"])
    return toks, nxt


@st.cache_data
def embed(model, texts):
    return np.array([d.embedding for d in client.embeddings.create(model=model, input=list(texts)).data])


with st.sidebar:
    k = st.slider("Top-k candidates (max 20)", 5, 20, 20)

st.title("What does the model expect next?")
st.caption("Type freely; `\\n` = line break. Updates after a short pause.")
raw = st_keyup("Prompt", value=ss.text, key=f"prompt{ss.ver}", debounce=500)
ss.text = raw or ""
prompt = ss.text.replace("\\n", "\n")
if not prompt.strip():
    st.stop()
toks, nxt = analyse(GEN_MODEL, prompt, k)

# typed text as tokens, coloured by how expected each token was (green = expected, red = surprising)
chips = []
for t, lp in toks:
    p = 1.0 if lp is None else math.exp(lp)
    hue = 120 * min(1, max(0, (math.log10(p) + 4) / 4))  # p=1e-4 red ... p=1 green
    chips.append(f'<span title="p={p:.4f}" style="background:hsl({hue:.0f},70%,80%);color:#111;border-radius:4px;'
                 f'padding:2px 4px;margin:2px;display:inline-block;font-family:monospace">{html.escape(show(t))}</span>'
                 + ("<br>" if "\n" in t else ""))
st.markdown(f"**{len(toks)} tokens** (hover for probability given the tokens before)<br>" + "".join(chips), unsafe_allow_html=True)


def append(tok):
    ss.text = ss.text + tok.replace("\n", "\\n")
    ss.ver += 1  # new widget key so the input shows the appended text


c1, c2 = st.columns([3, 2])
with c1:
    rest = max(1 - nxt.p.sum(), 1e-12)
    entropy = -(nxt.p * np.log(nxt.p)).sum() - rest * math.log(rest)
    st.subheader(f"Next token  ·  entropy ≈ {entropy:.2f} nats  ·  top-{k} mass {1 - rest:.0%}")
    fig = px.bar(nxt.assign(tok=nxt.token.map(show)), x="p", y="tok", orientation="h", range_x=[0, 1])
    fig.update_layout(height=60 + 24 * len(nxt), yaxis=dict(autorange="reversed", title=None), margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Continue")
    st.button(f"Next token → `{show(nxt.token[0])}`", on_click=append, args=(nxt.token[0],), type="primary")
    a1, a2, a3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    x = a1.number_input("x tokens", 1, 100, 10)
    if a2.button(f"Proceed {x} tokens"):
        ss.auto = x
    if a3.button("Stop"):
        ss.auto = 0
    if ss.get("auto"):
        st.caption(f"proceeding… {ss.auto} left")
    st.caption("or pick a candidate:")
    for i, t in enumerate(nxt.token[1:8], 1):
        st.button(f"`{show(t)}`  {nxt.p[i]:.1%}", key=f"cand{i}", on_click=append, args=(t,))

st.subheader("Where does the next token point? (embedding)")
b1, b2 = st.columns([1, 5])
live = b1.toggle("Embed every step", False)
if b2.button("Clear embedding history"):
    ss.history, ss.embedded = [], None
anchors = tuple(w.strip() for w in st.sidebar.text_area("Background terms (map)", ANCHORS, height=220)
                .replace("\n", ",").split(",") if w.strip())


@st.cache_resource
def anchor_map(words):
    """UMAP fitted once on the background terms; prompts and candidates are projected into this fixed map."""
    import umap

    return umap.UMAP(n_neighbors=10, min_dist=0.3, metric="cosine", random_state=0).fit(embed(EMB_MODEL, words))


if live and ss.get("embedded") != prompt:
    ss.embedded = prompt
    words = [t.strip() or show(t) for t in nxt.token]
    vecs = embed(EMB_MODEL, tuple(words + [prompt]))
    ss.history.append(dict(words=words, p=nxt.p.to_numpy(), vecs=vecs))

if live and len(anchors) < 12:
    st.info("Need at least 12 background terms.")
elif live and ss.history:
    red = anchor_map(anchors)
    A = red.embedding_
    xy = red.transform(np.vstack([h["vecs"] for h in ss.history]))
    steps_xy, i = [], 0
    for h in ss.history:  # split back per step: candidates, then prompt
        n = len(h["words"])
        steps_xy.append((xy[i:i + n], xy[i + n]))
        i += n + 1
    prompts = np.array([pr for _, pr in steps_xy])
    cents = np.array([np.average(c, axis=0, weights=h["p"]) for (c, _), h in zip(steps_xy, ss.history)])
    cand, pr_now, cur = steps_xy[-1][0], prompts[-1], ss.history[-1]

    # illustrative density: Gaussian blobs at candidate positions, weighted by probability
    pts = np.vstack([A, xy])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.1 * (hi - lo)
    gx, gy = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 120), np.linspace(lo[1] - pad[1], hi[1] + pad[1], 120)
    X, Y = np.meshgrid(gx, gy)
    bw = 0.07 * (hi - lo).mean()
    dens = sum(p * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * bw ** 2)) for (x, y), p in zip(cand, cur["p"]))

    fig = go.Figure(go.Contour(x=gx, y=gy, z=dens, showscale=False, contours_coloring="heatmap", line_width=0,
                               colorscale=[[0, "rgba(255,120,40,0)"], [1, "rgba(255,90,20,0.75)"]], hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=A[:, 0], y=A[:, 1], mode="text", text=list(anchors), name="background terms",
                             textfont=dict(color="gray", size=11)))
    fig.add_trace(go.Scatter(x=prompts[:, 0], y=prompts[:, 1], mode="lines+markers", name="prompt (history)",
                             marker=dict(symbol="star", size=9, color="rgba(0,0,0,0.35)"), line=dict(dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=cents[:, 0], y=cents[:, 1], mode="lines+markers", name="tendency (history)",
                             marker=dict(size=7, color="rgba(220,60,20,0.4)"), line=dict(dash="dot", color="rgba(220,60,20,0.4)")))
    fig.add_trace(go.Scatter(x=cand[:, 0], y=cand[:, 1], mode="markers+text", text=cur["words"], textposition="top center",
                             marker=dict(size=6 + 40 * np.sqrt(cur["p"]), color="rgba(200,50,10,0.8)"), name="next-token candidates",
                             customdata=cur["p"], hovertemplate="%{text}<br>p=%{customdata:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[pr_now[0]], y=[pr_now[1]], mode="markers", name="prompt now",
                             marker=dict(symbol="star", size=20, color="black")))
    fig.add_annotation(x=cents[-1, 0], y=cents[-1, 1], ax=pr_now[0], ay=pr_now[1], xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor="crimson", text="")
    fig.update_layout(height=650, xaxis_visible=False, yaxis_visible=False, legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Grey: fixed background terms (UMAP fitted on them). Orange cloud: probability-weighted density of the "
               "next-token candidates. Red arrow: from the prompt to the probability-weighted centre of the candidates. "
               "Illustrative only: a 2D projection of single-token embeddings, not the model's internal state.")

# auto-proceed: one greedy token per rerun, so every step is rendered (and embedded, if the toggle is on)
if ss.get("auto", 0) > 0:
    ss.auto -= 1
    append(nxt.token[0])
    st.rerun()
