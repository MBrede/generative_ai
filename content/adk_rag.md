# ADK Rag


ADK does not ship its own vector-store wrappers. This is less of a
drawback than it sounds: the retrieval part stays exactly the plain
qdrant and sentence-transformers code from above, and ADK only enters
the picture when an LLM is supposed to use it.

The first step is again the setup of a vector database and an embedding
model:

``` python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="paper",
    vectors_config=VectorParams(size=embedder.get_embedding_dimension(),
                                distance=Distance.COSINE),
)
```

    True

As an example, we will add the “Attention is all you need” paper. This
is how the head of our txt-file looks like:

             Attention Is All You Need
    arXiv:1706.03762v7 [cs.CL] 2 Aug 2023




                                                 Ashish Vaswani∗                Noam Shazeer∗               Niki Parmar∗             Jakob Uszkoreit∗
                                                  Google Brain                   Google Brain             Google Research            Google Research
                                              avaswani@google.com             noam@google.com            nikip@google.com            usz@google.com

                               

Since we can not just dump the document at once, we will chunk it (more
about that later). For now, a naive chunking into windows of 100 words
with an overlap of 20 words will do:

``` python
def chunk(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)]

chunks = chunk(text)

client.upsert(
    collection_name="paper",
    points=[
        PointStruct(id=i, vector=vector.tolist(), payload={"text": c})
        for i, (c, vector) in enumerate(zip(chunks, embedder.encode(chunks)))
    ],
)
len(chunks)
```

    78

The retrieval itself is a plain Python function. In ADK, any function
with type hints and a docstring can be handed to an agent as a tool; the
docstring becomes the tool description the model sees.

``` python
def find_references(question: str) -> str:
    """Query a database containing the paper "Attention is all you Need" in parts.
    This paper introduced the mechanism of self-attention to the NLP-literature.
    Returns a collection of scored text-snippets that are relevant to your question."""
    hits = client.query_points(
        collection_name="paper",
        query=embedder.encode(question).tolist(),
        limit=5,
    ).points
    return "\n".join(f"{round(h.score, 2)} - {h.payload['text']}" for h in hits)

print(find_references("What do the terms Key, Value and Query stand for in self-attention?")[:800])
```

    0.55 - 3.2 Attention An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum 3 Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The
    0.51 - query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (

To use the tool, we need an agent. We use Google’s Agent Development Kit
(ADK) here, which we will look at in more detail in the session on
agents (`uv add google-adk litellm`). ADK talks to local models through
a `LiteLlm` wrapper, whose model string starts with a provider prefix
(`ollama_chat/` for Ollama). The helper below keeps that string in one
place, so swapping models is a one-word change:

``` python
import os
from google.adk.models.lite_llm import LiteLlm

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

def model(name: str) -> LiteLlm:
    return LiteLlm(model=f"ollama_chat/{name}")
```

Since we are going to ask several agents several questions, we also wrap
the Runner-boilerplate into a small function. It prints every tool call,
so we can see whether the agent actually used the database, and it skips
the model’s reasoning (“thinking”) parts, so only the answer is
returned:

``` python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

def ask(agent: LlmAgent, question: str) -> str:
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="rag", session_service=session_service)
    session = session_service.create_session_sync(app_name="rag", user_id="student")
    answer = ""
    for event in runner.run(
        user_id="student",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=question)]),
    ):
        for call in event.get_function_calls():  # a model may request several calls at once
            print(f"[tool call] {call.name}({call.args})")
        if event.is_final_response() and event.content:
            answer = "".join(p.text for p in event.content.parts if p.text and not p.thought)
    return answer
```

Now we can build the agent and give it our tool:

``` python
agent = LlmAgent(
    name="paper_assistant",
    model=model("qwen3:4b"),
    instruction="You answer questions about scientific papers.",
    tools=[find_references],
)

print(ask(agent, "What is the meaning of Query, Key and Value in the context of self-attention?"))
```

    [tool call] find_references({'question': 'What is the meaning of Query, Key and Value in the context of self-attention?'})


    In the context of self-attention, **Query**, **Key**, and **Value** are vector representations derived from the input sequence, each serving a specific role:

    1. **Query (Q)**: Represents the **focus** of attention. It determines what parts of the sequence the model should attend to. The query is used to compute compatibility scores with keys.

    2. **Key (K)**: Represents the **features** used to compute compatibility with queries. The key encodes information about each position in the sequence, allowing the model to measure how relevant a particular position is to the query.

    3. **Value (V)**: Represents the **content** that is weighted based on the compatibility scores. The value is the actual information that is aggregated (via weighted sum) to form the output.

    ### How they work together:
    - The **attention weights** are computed by comparing the query (Q) to all keys (K) using a dot-product (or other compatibility function), normalized by a scaling factor (e.g., √dk).
    - These weights are then applied to the corresponding values (V) to produce a weighted sum, which becomes the output for that position.

    ### Example:
    In self-attention, all queries, keys, and values come from the same sequence. For each position, the model computes attention weights to determine how much to focus on other positions, enabling it to capture dependencies across the sequence. 

    This mechanism allows the model to dynamically prioritize relevant parts of the input when processing information.

The agent does call the tool, but have a close look at the answer: it
reads like general knowledge about attention, and nothing in it points
back to the retrieved passages. We cannot tell which parts come from the
paper and which from the model’s training data. Nothing in the
instruction told the model that the retrieved text should be the basis
of its answer.

One way to steer the agent is to be more explicit in the system prompt.
In ADK, the system prompt is simply the `instruction` of the agent:

``` python
strict_agent = LlmAgent(
    name="paper_assistant",
    model=model("qwen3:4b"),
    instruction="""You are designed to help answer questions based on a collection of paper-excerpts.
You have access to tools that allow you to query paper-content.
Do not answer without tool-usage if a tool can be used to answer a question.
Always back up your claims with a text passage you retrieved and quote it.
If the tools do not return relevant passages, say so instead of guessing.""",
    tools=[find_references],
)

print(ask(strict_agent, "How does the paper 'Attention is all you need' define the term self attention?"))
```

    [tool call] find_references({'question': "How does the paper 'Attention is all you need' define the term self attention?"})


    The paper "Attention is all you need" defines **self-attention** as an "attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence" (score: 0.58). It emphasizes that self-attention, also termed **intra-attention**, allows every position in a sequence to attend to all others, enabling the model to capture dependencies between arbitrary positions within the same sequence. This mechanism is critical for tasks like reading comprehension and summarization, as it allows the model to weigh the importance of different elements in the sequence dynamically. 

    Key passage:  
    *"Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."*  

    This is elaborated further in section 3.2 of the paper.

Now the answer is short, quotes the retrieved definition and stays close
to the source.

Whether an agent uses its tools reliably depends heavily on the model.
Very small models (around 1 billion parameters) frequently skip the tool
and answer from their own, possibly hallucinated, knowledge, or produce
malformed tool calls. Try swapping in a smaller model in the
`model(...)` calls above and compare. Using a bigger model is not always
feasible though.

Another way to use the retrieval-pipeline is to not give a weak model
the opportunity to mess up the tool calling at all. Instead of letting
the model decide whether to retrieve, Python always retrieves and hands
the snippets to the model, whose only job is to summarize them. This is
the deterministic-routing pipeline idea we will meet again in the
session on LLM pipelines: the control flow is fixed in code, the LLM
fills in one well-defined step.

``` python
summarizer = LlmAgent(
    name="summarizer",
    model=model("qwen3:4b"),
    instruction="""Answer the question using only the provided text snippets.
Quote the snippets you use. If they do not contain the answer, say so.""",
)

def rag(question: str) -> str:
    snippets = find_references(question)
    return ask(summarizer, f"Question: {question}\n\nSnippets:\n{snippets}")

print(rag("What is the meaning of Query, Key and Value in the context of self-attention?"))
```

    In the context of self-attention, **Query**, **Key**, and **Value** are components of an attention mechanism:  
    - **Query** is a vector that determines what information to focus on.  
    - **Key** is a vector used to compute compatibility with the Query, determining relevance.  
    - **Value** is the information that is weighted and summed based on the compatibility between the Query and Key.  

    The output is a weighted sum of the Values, where the weights are computed by a compatibility function (e.g., dot-product) between the Query and Keys. In self-attention, all Queries, Keys, and Values originate from the same sequence, enabling each position to attend to others in the sequence.  

    **Snippets used**:  
    0.55: "Attention function maps a query and a set of key-value pairs to an output... the output is a weighted sum of the values, where the weight is computed by a compatibility function of the query with the corresponding key."  
    0.52: "Queries and keys of dimension dk... values of dimension dv... dot products of the query with all keys... softmax function to obtain the weights on the values."  
    0.53: "Self-attention... relates different positions of a single sequence... all queries, keys, and values come from the same place."

Since the retrieval no longer depends on the model’s decision, this also
works with models that are too small for reliable tool calling.
