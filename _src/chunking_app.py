"""Chunking strategies side by side: fixed-size vs. the two semantic chunkers (breakpoint, clustering)
compared by Qu, Tu & Bao (2025), "Is Semantic Chunking Worth the Computational Cost?", Findings of NAACL.
Start with chunking_app.sh (needs URL/KEY of an OpenAI-compatible endpoint with an embedding model).
"""
import html
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from sklearn.cluster import DBSCAN, AgglomerativeClustering

EMB_MODEL = "qwen3-embedding-0-6b"
COLORS = ["#d9c8f0", "#cdebc5", "#fbe3a8", "#f6c6c6", "#c6e0f6", "#e8e8e8"]
# Introduction of Qu, Tu & Bao (2025), CC BY 4.0, citation markers removed; paragraphs act as "topics"
DEFAULT = """chunking in RAG: In Retrieval-Augmented Generation (RAG) systems, cutting documents into smaller units called "chunks" has a crucial effect on the quality of both retrieval and generation tasks. By retrieving the most relevant chunks for a given query and feeding them into a generative language model, these systems aim to produce accurate and contextually appropriate answers. However, the effectiveness of chunking strategies remains a significant challenge in optimizing retrieval quality and computational efficiency.

fixed vs semantic: Known as fixed-size chunking, the traditional way to chunk is to cut documents into chunks of a fixed length such as 200 tokens. While computationally simple, this approach can fragment semantically related content across multiple chunks, leading to suboptimal retrieval performance. Recently, there has been a surge of interest in semantic chunking, where documents are segmented based on semantic similarity, with some industry applications suggesting promising improvements in performance. However, there is no systematic evidence that semantic chunking yields a performance gain in downstream tasks, and if there is, the gain is significant enough to justify the computational overhead than fixed-size chunking.

evaluation: Such a systematic evaluation is not trivial due to the lack of data that can be directly used to compare chunking strategies. Therefore, we design an indirect evaluation using three proxy tasks: (1) document retrieval, measuring the ability to identify relevant documents; (2) evidence retrieval, measuring the ability to locate ground-truth evidence; and (3) answer generation, testing the quality of answers produced by a generative model using retrieved chunks. Our findings challenge prevailing assumptions about the benefits of semantic chunking, suggesting that its advantages are highly task-dependent and often insufficient to justify the added computational costs. This study lays the groundwork for future exploration of more efficient and adaptive chunking strategies in RAG systems.

contributions: We present a novel, large-scale evaluation framework comparing semantic and fixed-size chunking across diverse tasks. We demonstrate that while semantic chunking shows some benefits in certain scenarios, these are inconsistent and often insufficient to justify the computational cost."""

st.set_page_config(layout="wide", page_title="Chunking strategies")
client = OpenAI(base_url=os.environ.get("URL", "http://localhost:11434/v1"), api_key=os.environ.get("KEY", "ollama"))


def parse(text):
    """Paragraphs = source documents (optional 'topic: ' prefix), split into sentences."""
    rows = []
    for i, para in enumerate(p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()):
        m = re.match(r"\s*([\w ]{1,30}):\s+(.*)", para, re.S)
        topic, body = (m[1], m[2]) if m else (f"paragraph {i + 1}", para)
        rows += [dict(topic=topic, sentence=s) for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s]
    return pd.DataFrame(rows)


@st.cache_data
def embed(sentences):
    e = np.array([d.embedding for d in client.embeddings.create(model=EMB_MODEL, input=list(sentences)).data])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def fixed(n_sent, size, overlap):
    starts = range(0, max(n_sent - overlap, 1), size - overlap)
    return [list(range(s, min(s + size, n_sent))) for s in starts]


def split_at(n_sent, breaks):
    edges = [0] + [b + 1 for b in breaks] + [n_sent]
    return [list(range(a, b)) for a, b in zip(edges, edges[1:]) if b > a]


def combined_distance(emb, lam):
    """Eq. 1-3 in Qu et al.: lambda * positional distance + (1 - lambda) * cosine distance (negative cos -> 0)."""
    n = len(emb)
    idx = np.arange(n)
    d_pos = np.abs(idx[:, None] - idx[None, :]) / n
    d_cos = 1 - np.maximum(emb @ emb.T, 0)
    return np.maximum(lam * d_pos + (1 - lam) * d_cos, 0)  # clip float noise (-1e-16) on the diagonal


def clusters(labels):
    """Cluster labels -> chunks (members in document order); DBSCAN noise (-1) becomes singletons."""
    out = {}
    for i, l in enumerate(labels):
        out.setdefault(l if l >= 0 else f"noise{i}", []).append(i)
    return sorted(out.values(), key=lambda c: c[0])


def purity(chunks, topics):
    return np.mean([pd.Series([topics[i] for i in c]).value_counts().iloc[0] / len(c) for c in chunks])


def show_chunks(chunks, df, color):
    boxes = []
    for c in chunks:
        lines = "".join(
            f'<div title="{html.escape(df.sentence[i])}" style="background:{color[df.topic[i]]};color:#111;'
            f'padding:1px 6px;margin:2px 0;border-radius:3px;font-size:0.82em;white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis"><b>{i}</b> {html.escape(df.sentence[i])}</div>' for i in c)
        boxes.append(f'<div style="border:2px solid #555;border-radius:6px;padding:3px;margin:6px 0">{lines}</div>')
    st.markdown("".join(boxes), unsafe_allow_html=True)


st.title("Chunking: fixed-size vs. semantic")
st.caption("Default text: introduction of Qu, Tu & Bao (2025), *Is Semantic Chunking Worth the Computational Cost?*, "
           "Findings of NAACL, CC BY 4.0. Paste your own text in the sidebar.")
text = st.sidebar.text_area("Document (blank line = new paragraph, optional `topic:` prefix)", DEFAULT, height=500)
df = parse(text)
if len(df) < 4:
    st.info("Need at least 4 sentences.")
    st.stop()
topics = list(dict.fromkeys(df.topic))
color = {t: COLORS[i % len(COLORS)] for i, t in enumerate(topics)}
st.markdown("Topics: " + " ".join(f'<span style="background:{color[t]};color:#111;padding:2px 8px;border-radius:4px">'
                                   f'{html.escape(t)}</span>' for t in topics) + "  ·  hover a sentence for full text",
            unsafe_allow_html=True)
emb = embed(tuple(df.sentence))
n = len(df)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("(a) Fixed-size")
    size = st.slider("Sentences per chunk", 2, 8, 3)
    overlap = st.slider("Overlap (sentences)", 0, size - 1, 0)
    ch_a = fixed(n, size, overlap)
    st.caption(f"{len(ch_a)} chunks · topic purity {purity(ch_a, df.topic):.0%}")
    show_chunks(ch_a, df, color)
    st.caption("Cuts every n sentences, blind to content. Overlap repeats sentences to keep some context.")

with c2:
    st.subheader("(b) Breakpoint-based")
    gaps = 1 - np.sum(emb[:-1] * emb[1:], axis=1)  # cosine distance between neighbours
    mode = st.radio("Threshold", ["percentile", "absolute"], horizontal=True)
    if mode == "percentile":
        q = st.slider("Percentile of distances", 50, 99, 80)
        thr = np.percentile(gaps, q)
    else:
        thr = st.slider("Cosine distance", 0.0, 1.0, float(np.round(np.median(gaps), 2)), 0.01)
    ch_b = split_at(n, np.flatnonzero(gaps > thr))
    st.caption(f"{len(ch_b)} chunks · topic purity {purity(ch_b, df.topic):.0%}")
    show_chunks(ch_b, df, color)
    g = pd.DataFrame({"between": [f"{i}|{i + 1}" for i in range(n - 1)], "distance": gaps, "cut": gaps > thr})
    fig = px.bar(g, x="between", y="distance", color="cut", color_discrete_map={True: "crimson", False: "gray"})
    fig.add_hline(y=thr, line_dash="dash")
    fig.update_layout(height=280, margin=dict(t=10, b=10), showlegend=False, xaxis_title="neighbouring sentences")
    st.plotly_chart(fig, width="stretch")
    st.caption("Walks through the text and cuts wherever two neighbouring sentences are further apart than the "
               "threshold. Only ever looks at two sentences at a time.")

with c3:
    st.subheader("(c) Clustering-based")
    lam = st.slider("λ (0 = meaning only, 1 = position only)", 0.0, 1.0, 0.2, 0.05)
    algo = st.radio("Algorithm", ["single linkage", "DBSCAN"], horizontal=True)
    D = combined_distance(emb, lam)
    if algo == "single linkage":
        t = st.slider("Distance threshold", 0.05, 0.6, 0.3, 0.01)
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=t, metric="precomputed",
                                         linkage="single").fit_predict(D)
    else:
        eps = st.slider("eps", 0.05, 0.6, 0.3, 0.01)
        labels = DBSCAN(eps=eps, min_samples=st.slider("min_samples", 1, 4, 1), metric="precomputed").fit_predict(D)
    ch_c = clusters(labels)
    st.caption(f"{len(ch_c)} chunks · topic purity {purity(ch_c, df.topic):.0%}")
    show_chunks(ch_c, df, color)
    fig = px.imshow(D, color_continuous_scale="Viridis_r", labels=dict(color="distance"), aspect="equal")
    fig.update_layout(height=320, margin=dict(t=10, b=10), xaxis_title="sentence", yaxis_title="sentence")
    st.plotly_chart(fig, width="stretch")
    st.caption("Groups sentences by a mix of meaning and position, so chunks can collect sentences from all over "
               "the document. λ trades the two off.")

st.caption("Topic purity = share of the majority topic per chunk, averaged over chunks (only meaningful with topic "
           "labels). Qu et al. (2025) find that the semantic chunkers rarely beat fixed-size chunking consistently, "
           "especially on real (non-stitched) documents, and cost noticeably more compute.")
