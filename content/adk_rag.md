# RAG with ADK


ADK has no ready-made integration for local vector stores like Qdrant.
This is less of a drawback than it sounds: the retrieval part stays
exactly the plain Qdrant and sentence-transformers code from above, and
ADK only enters the picture when an LLM is supposed to use it.

The first step is again the setup of a vector database and an embedding
model. Note that an in-memory Qdrant client (`location=":memory:"`)
loses all data when the Python kernel restarts, which is fine for this
example:

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

    Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]

    True

As an example, we will add the paper “Attention Is All You Need”. Here
is a short excerpt from our plain-text version of the paper:

             Attention Is All You Need
    arXiv:1706.03762v7 [cs.CL] 2 Aug 2023




                                                 Ashish Vaswani∗                Noam Shazeer∗               Niki Parmar∗             Jakob Uszkoreit∗
                                                  Google Brain                   Google Brain             Google Research            Google Research
                                              avaswani@google.com             noam@google.com            nikip@google.com            usz@google.com

                               

One embedding for the whole paper would average over every topic in it,
and the embedding model truncates long inputs anyway. We therefore chunk
the text (more about that later). For now, a naive chunking into windows
of 100 words with an overlap of 20 words will do. The last line shows
how many chunks this gives us:

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
docstring becomes the tool description the model sees. Each returned
snippet is prefixed with its cosine similarity to the question, higher
means more similar:

``` python
def find_references(question: str) -> str:
    """Query a database containing the paper "Attention Is All You Need" in parts.
    This paper introduced the transformer architecture and its self-attention mechanism.
    Returns a collection of scored text snippets that are relevant to your question."""
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

To use the tool, we need an agent: an LLM that can decide to call tools
and read their results before it answers. We use Google’s Agent
Development Kit (ADK) here, which we will look at in more detail in the
session on agents. It is installed with `uv add google-adk litellm`, and
the model used below has to be available in Ollama
(`ollama pull qwen3:4b`). ADK talks to local models through a `LiteLlm`
wrapper, whose model string starts with a provider prefix
(`ollama_chat/` for Ollama). The helper below keeps that string in one
place, so swapping models is a one-word change:

``` python
import os
from google.adk.models.lite_llm import LiteLlm

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

def model(name: str) -> LiteLlm:
    return LiteLlm(model=f"ollama_chat/{name}")
```

To run an agent, ADK needs a runner, which takes care of the session
(the conversation state) for us. `InMemoryRunner.run_debug` is the
shortest way to send a question and collect everything that happened as
a list of events. As the name says, it is meant for experiments like
this one, not for production code. With `quiet=True` it does not print
the events itself, since our helper below prints what we want to see.
`run_debug` is asynchronous: it waits for the model’s response over the
network, and `await` lets Python pause at this point until the response
has arrived. In a Jupyter notebook, `await` can be used directly in a
cell; in a plain Python script, you would wrap the call as
`asyncio.run(ask(...))`.

Since we are going to compare several agents, we wrap all of this into a
small function. It creates a new runner, and thereby a fresh session,
for every call, so the agent remembers nothing between questions and
each test is independent. It prints every tool call, so we can see
whether the agent actually used the database, and it skips the model’s
reasoning (“thinking”) parts if the model returns them as separate
parts, so only the answer is returned:

``` python
from google.adk import Agent
from google.adk.runners import InMemoryRunner

async def ask(agent, question: str) -> str:
    events = await InMemoryRunner(agent=agent).run_debug(question, quiet=True)
    for event in events:
        for call in event.get_function_calls():  # a model may request several calls at once
            print(f"[tool call] {call.name}({call.args})")
    final = [e for e in events if e.is_final_response() and e.content][-1]
    return "".join(p.text for p in final.content.parts if p.text and not p.thought)
```

To compare different ways of building the system, we ask all of them the
same question:

``` python
q = "What is the meaning of Query, Key and Value in the context of self-attention?"
```

Now we can build the agent and give it our tool:

``` python
agent = Agent(
    name="paper_assistant",
    model=model("qwen3:4b"),
    instruction="You answer questions about scientific papers.",
    tools=[find_references],
)

print(await ask(agent, q))
```

    The **Query (Q)**, **Key (K)**, and **Value (V)** are three transformed vectors derived from the input tokens in the self-attention mechanism:

    1. **Query**: Represents the "question" or "focus" of the current token. It determines what information the token is seeking from other tokens.
    2. **Key**: Acts as a "tag" or "identifier" for the tokens. It helps measure the relevance of other tokens to the current token.
    3. **Value**: Contains the actual information or content of the tokens. It is used to compute the weighted sum of relevant information based on the attention scores.

    In self-attention, the attention score between two tokens is computed as the dot product of the Query and Key vectors, followed by a softmax normalization. This score determines how much weight the Value of one token should contribute to the output of another. This mechanism allows the model to dynamically focus on relevant parts of the input sequence. 

    This concept is central to the transformer architecture introduced in the paper *Attention Is All You Need*.

LLM answers vary from run to run, so your output will differ from ours.
In our run, no `[tool call]` line appears: the agent did not consult the
database at all and answered from its training data. Even in runs where
it does call the tool, look closely at the answer: does it quote the
retrieved passages, or does it read like general knowledge about
attention? If nothing in the answer points back to the snippets printed
above, we cannot tell which parts come from the paper and which from the
model’s training data. Nothing in the instruction told the model to use
the tool, or that the retrieved text should be the basis of its answer.

One way to steer the agent is to be more explicit in the system prompt.
In ADK, the system prompt is simply the `instruction` of the agent:

``` python
strict_agent = Agent(
    name="strict_paper_assistant",
    model=model("qwen3:4b"),
    instruction="""You are designed to help answer questions based on a collection of paper excerpts.
You have access to tools that allow you to query the paper content.
Do not answer without using a tool if a tool can be used to answer a question.
Always back up your claims with a text passage you retrieved and quote it.
If the tools do not return relevant passages, say so instead of guessing.""",
    tools=[find_references],
)

print(await ask(strict_agent, q))
```

    [tool call] find_references({'question': 'What is the meaning of Query, Key and Value in the context of self-attention?'})


    The **Query**, **Key**, and **Value** are fundamental components of the self-attention mechanism in transformers. Here's their role based on the paper:

    1. **Query (Q)**: Represents the current position's "question" or "focus" in the sequence. It is used to determine how much attention (weight) to assign to other elements in the sequence.  
    2. **Key (K)**: Represents the "features" or "information" of other positions in the sequence. It is used to compute compatibility scores with the Query.  
    3. **Value (V)**: Represents the actual information (e.g., word embeddings) that is weighted and summed based on the attention scores.  

    The attention mechanism computes weights by comparing the Query to the Keys (e.g., via dot-product), then uses these weights to compute a weighted sum of the Values. This allows each position to "attend" to others, capturing dependencies within the sequence.  

    **Key passage**:  
    > "The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key."  
    > "We compute the dot products of the query with all keys, divide each by dk, and apply a softmax function to obtain the weights on the values."  

    This process enables self-attention to dynamically prioritize relevant parts of the sequence for each position.

Compare the two answers: in our run, the stricter agent called the tool
and quoted the paper. In general, the stricter instruction should lead
to quotes that you can find in the retrieved snippets, and to less
content that the snippets do not support.

Whether an agent uses its tools reliably depends heavily on the model.
Very small models (around 1 billion parameters) frequently skip the tool
and answer from their own, possibly hallucinated, knowledge, or produce
malformed tool calls. Try swapping in a smaller model such as
`qwen3:0.6b` or `llama3.2:1b` in the `model(...)` calls above and
compare. Since a bigger model is not always an option, another approach
is to take the decision about retrieval away from the model entirely.
Instead of letting the model decide whether to retrieve, Python always
retrieves and hands the snippets to the model, whose only job is to
answer from them. This is the idea of an LLM pipeline with deterministic
routing, which we will meet again in the session on LLM pipelines
(**?@sec-orchestration**): the control flow is fixed in code, and the
LLM fills in one well-defined step.

ADK expresses such a fixed control flow as a `Workflow`. A workflow is a
graph of steps, called nodes, and its `edges` define which node runs
after which. Graphs allow branches and loops, which we will use in the
session on LLM pipelines; our pipeline is just a straight line.

Each node can be a plain Python function or an agent. Whatever a node
returns is handed to the next node: a function receives it through a
parameter that must be named `node_input` (other parameter names are
looked up in the workflow’s state instead), and an agent receives it as
its user message. A node sees nothing else, which is why `retrieve`
below repeats the question in its output: otherwise the summarizer would
get the snippets, but not the question. The tuple
`("START", retrieve, summarizer)` describes a chain: `"START"` is the
entry point that receives the user’s message, which then flows through
`retrieve` into `summarizer`.

``` python
from google.adk import Workflow

def retrieve(node_input: str) -> str:
    return f"Question: {node_input}\n\nSnippets:\n{find_references(node_input)}"

summarizer = Agent(
    name="summarizer",
    model=model("qwen3:4b"),
    instruction="""Answer the question using only the provided text snippets.
Quote the snippets you use. If they do not contain the answer, say so.""",
)

rag = Workflow(name="rag", edges=[("START", retrieve, summarizer)])

print(await ask(rag, q))
```

    The meaning of Query, Key, and Value in the context of self-attention is as follows:  
    - **Query**: Represents the current position's need for information.  
    - **Key**: Used to compute the compatibility (or similarity) between the current position and others, determining how much each value should contribute.  
    - **Value**: Holds the actual information that is weighted based on the compatibility between the query and keys.  

    The attention mechanism computes weights by matching the query with keys, then uses these weights to compute a weighted sum of the values (quoted from snippet 0.55 and 0.52). In self-attention, all positions in a sequence act as both query, key, and value, enabling the model to relate different positions within the same sequence (snippet 0.53).

`InMemoryRunner` accepts a workflow just like an agent, so our `ask`
helper works unchanged. Note that no `[tool call]` line appears:
retrieval is now a step of the graph, not a decision of the model. This
makes the pipeline work with models that are too small for reliable tool
calling, but it comes at a cost. The workflow always retrieves, even for
questions that need no retrieval, and it retrieves exactly once with the
raw question, so it cannot rephrase the query or search again when the
results are poor, as a tool-calling agent can. And “answer using only
the snippets” is still just a request: a small summarizer can drift away
from the snippets all the same.

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead>
<tr>
<th></th>
<th>who decides to retrieve?</th>
<th>model size needed</th>
<th>traceability</th>
</tr>
</thead>
<tbody>
<tr>
<td>agent with tool</td>
<td>the model</td>
<td>large enough for reliable tool calls</td>
<td>tool calls visible, use of the snippets unclear</td>
</tr>
<tr>
<td>agent with strict instruction</td>
<td>the model</td>
<td>large enough for reliable tool calls</td>
<td>quotes can be checked against the snippets</td>
</tr>
<tr>
<td>retrieve-then-summarize workflow</td>
<td>the code, always once</td>
<td>small models suffice</td>
<td>fixed path, quotes can be checked</td>
</tr>
</tbody>
</table>

To see how the three variants handle missing information, ask all of
them a question the paper does not answer, for example about GPT-4, and
compare whether they admit that the snippets do not contain the answer.
