## Llamaindex RAG


The first thing in both the Llamaindex and the manual way of creating a
retrieval pipeline is the setup of a vector database:

``` python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Batch
DIMENSIONS = 384
client = QdrantClient(location=":memory:")
```

To store data and query the database, we have to load a embedding-model.
As in the manual way of creating a retrieval pipeline discussed before,
we can use a huggingface-SentenceTranformer model. But instead of using
the SentenceTransformer class from the sentence_transformers library, we
have to use the HuggingFaceEmbedding class from Llamaindex. This model
is entered into the Llamaindex-Settings.

``` python
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")
Settings.embed_model = embed_model
```

The next step is to wrap the vector-store into a
Llamaindex-VectorStoreIndex. This index can be used to add our documents
to the database.

``` python
from llama_index.vector_stores.qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(client=client, collection_name="paper")
```

As an example, we will add the “Attention is all you need” paper. This
is how the head of our txt-file looks like:

             Attention Is All You Need
    arXiv:1706.03762v7 [cs.CL] 2 Aug 2023




                                                 Ashish Vaswani∗                Noam Shazeer∗               Niki Parmar∗             Jakob Uszkoreit∗
                                                  Google Brain                   Google Brain             Google Research            Google Research
                                              avaswani@google.com             noam@google.com            nikip@google.com            usz@google.com

                               

Since we can not just dump the document at once, we will chunk it in
sentences (more about that later). This can be done like this (ignore
the parameters by now, we will look at them later):

``` python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

node_parser = SentenceSplitter(chunk_size=20, chunk_overlap=5)

nodes = node_parser.get_nodes_from_documents(
    [Document(text=text)], show_progress=False
)
```

    Metadata length (0) is close to chunk size (20). Resulting chunks are less than 50 tokens. Consider increasing the chunk size or decreasing the size of your metadata to avoid this.

These documents are then added to our database and transformed in an
*index* llamaindex can use:

``` python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex(
    nodes=nodes,
    vector_store=vector_store,
)
```

This index can already be used to retrieve documents from the database
(by converting it to a *retriever*).

``` python
retriever = index.as_retriever(similarity_top_k=10)
retriever.retrieve('What do the terms Key, Value and Query stand for in self-attention?')
```

    [NodeWithScore(node=TextNode(id_='f179163e-b751-462b-bc33-878bd1d4e8e8', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='a91dc640-1e5f-4ae0-9b12-2d299c219be9', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='2d5b4fe94c35c42fb9f0851b43257320ec9ad5c63b127e463fa2215993f117fc'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='01a0f046-3e0f-457c-a37e-7ba8ea82cca5', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='b8e28f60c384f88a38758a9b8df2ec53ef4316e372031c3bbd60f545ae0ae44f')}, text='attention function can be described as mapping a query and a set of key-value pairs to an output,', mimetype='text/plain', start_char_idx=11228, end_char_idx=11325, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.6689261429690226),
     NodeWithScore(node=TextNode(id_='2b89faef-3d46-401f-b3b0-26006401feef', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='e1d5b4df-5d1c-439d-9775-5088c63eb9f3', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='b39aebf8977b80cbe0d8dd5fa04a19223f1e818c235371d6af215d0556100bc0'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='0aa4a8f3-4af2-4fe0-9358-1ae72fe8786f', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='d6c4dee83e22dc8559d7a39700c28e907b50c8e479fd5af1941db4408208cf69')}, text='In a self-attention layer all of the keys,', mimetype='text/plain', start_char_idx=16067, end_char_idx=16109, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.6140070285724392),
     NodeWithScore(node=TextNode(id_='44972890-4447-4cb5-852f-e1486eed0bdd', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='055b5c0c-7837-405b-b65a-5dd5057880e3', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='aa669bf3b7256481cc8755c1c78352a123179dc767de25754ca48fbf23b689bc'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='747a4331-c2ea-4a00-b419-0b08d418ddcd', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='a642151581faab17008e1e3c622d5eec5ed6e9cbea16178e5ac24dafefa4acd8')}, text='of performing a single attention function with dmodel -dimensional keys, values and queries,', mimetype='text/plain', start_char_idx=13777, end_char_idx=13869, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.6003160602014549),
     NodeWithScore(node=TextNode(id_='18e49238-7647-4909-bfe4-23a35d4c5ca3', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='95f73bbf-b989-433a-b1a6-ebd858241187', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='f6cbc386d726e01d6d2a7c175368263f644c5ce71b85c105929e485dcf0b9203'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='efc76fc4-7ee5-494a-935d-06a9efd571d9', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='5439b8ee1db45977a39995f337377707d150ef741a4180c98a55ce744ac13692')}, text='we vary the number of attention heads and the attention key and value dimensions,', mimetype='text/plain', start_char_idx=32587, end_char_idx=32668, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5878171880241171),
     NodeWithScore(node=TextNode(id_='a0d046f8-7251-4b55-8457-78e688ef82d1', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='b4b6a870-d5e7-43b8-a98b-e5660829fa7e', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='7f9189e6569cc76c3e4c706c4a298ce83039920dda5ff4b64ec3d2a2ecd17d16'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='dad4c0dd-260f-43cb-8997-8668ac1ca95e', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='5cf958d495be43203293644ac0c78bb1ae632d95f5eac85a487a134eeb8f83e1')}, text='keys and values we then perform the attention function in parallel,', mimetype='text/plain', start_char_idx=14085, end_char_idx=14152, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5599990554034037),
     NodeWithScore(node=TextNode(id_='5aeffddb-5e61-4bd2-b7c7-0908b617345c', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='20d4b9c4-38cb-4c61-9ae9-2d79fa770f8b', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='8be71b432f8f0abcddbd647735091a419597e3c6b174d427210c39d82d3357db'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='85add884-1b17-487d-8136-3faed207f79a', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='8e3333b3547e738a6d3fffa8d21dcbd4a8d6b56b6f8cd8f1ccf263541439d80e')}, text='2.\nSelf-attention, sometimes called intra-attention is an attention mechanism relating different', mimetype='text/plain', start_char_idx=8000, end_char_idx=8096, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5510363892301526),
     NodeWithScore(node=TextNode(id_='e14b73d1-ee56-46c4-974b-6d6f2401304b', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='2ceaf2a5-d16b-46a8-bd1a-6ab3eb650c52', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='d99e4e0b99cb6641047a17c6e9da3a59b0d4ea0df2f0b213ca47d30d333ebac1'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='c23c3273-7cf2-47c8-9fd5-b47f2cf6e10f', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='4aa20349e1dd87216ce6467799ecd9f96716d0de4be21cdd0d33d83cd860c3f1')}, text='of the sequence. Self-attention has been\nused successfully in a variety of tasks including reading comprehension,', mimetype='text/plain', start_char_idx=8165, end_char_idx=8278, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5396821423481503),
     NodeWithScore(node=TextNode(id_='7ca1930d-752c-4f19-aa10-0f1b26a06f3b', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='10642476-b727-4743-b01b-52096a29fce9', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='a153765fee9f0efcfb6804fd5bc8312f9f13f9c282fd03dd49247f24dbc0c25b'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='fd1e0285-6636-428c-be1d-d29707f22d2f', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='7b536fd47292343852aa48f5586a9d0c1290c7c340aeb49708c8f7ebc2b2da5a')}, text='In practice, we compute the attention function on a set of queries simultaneously,', mimetype='text/plain', start_char_idx=12231, end_char_idx=12313, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5200972369180269),
     NodeWithScore(node=TextNode(id_='747a4331-c2ea-4a00-b419-0b08d418ddcd', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='44972890-4447-4cb5-852f-e1486eed0bdd', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='5bfecc13dc676c3effef63180d8e345a59aee3b00971ee632c28a5beb3e936d4'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='35eabe69-ac0e-4d8f-af59-d8c3a75e0394', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='5190a75cc39636a77d1a3db884b7cb86f0afe3c8c5fba1701abbb266740d5bdd')}, text='values and queries,\nwe found it beneficial to linearly project the queries,', mimetype='text/plain', start_char_idx=13850, end_char_idx=13925, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5128181572746763),
     NodeWithScore(node=TextNode(id_='c310de50-a2a5-4642-be1c-cdea5d9afcf1', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='7fda0fb0-6e63-4560-b1f6-e5bc9722ab1a', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='69ab46ef913b8936b3669fb47d6b591b43a5dd0ca37a3e2dbf0ce04d8c703765'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='ff0f74c7-ddfe-461b-aa5f-ed4e49cb9e8f', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='8d64e319bfdf6f04a0917a395eada4c0a280701c07b475000ff2de68606bca76'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='72169dc7-1426-4e59-a91e-b6ddd4c850d6', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='78736791801c79f892ed19fd6161ad7c8524bc7a9c57b4c0c01121be99dddb18')}, text='Operations\n      Self-Attention', mimetype='text/plain', start_char_idx=18618, end_char_idx=18649, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5069105137924549)]

The retriever can then directly be use as a tool to answer questions
about our documents:

``` python
from llama_index.core.tools import BaseTool, FunctionTool

def find_references(question: str) -> str:
    """Query a database containing the paper "Attention is all you Need" in parts.
    This paper introduced the mechanism of self-attention to the NLP-literature.
    Returns a collection of scored text-snippets that are relevant to your question."""
    return '\n'.join([f'{round(n.score,2)} - {n.node.text}' for n in retriever.retrieve(q)])


find_references_tool = FunctionTool.from_defaults(fn=find_references)
```

This tool can then be added to an agent as we discussed before:

``` python
from llama_index.core.agent import ReActAgent

from llama_index.llms.lmstudio import LMStudio


llm = LMStudio(model_name="llama-3.2-1b-instruct",
        base_url="http://localhost:1234/v1",
    temperature=0.5,
    request_timeout=600)


agent = ReActAgent.from_tools(tools=[find_references_tool],llm=llm, verbose=True)
```

    /home/brede/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/pydantic/_internal/_fields.py:132: UserWarning: Field "model_name" in LMStudio has conflict with protected namespace "model_".

    You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
      warnings.warn(

Which can then be used to answer chat-requests:

``` python
response = agent.chat("What is the meaning of Query, Key and Value in the context of self-attention?")
print(str(response))
```

    > Running step e29ee90e-3a6a-4863-8356-f24382cd4f7b. Step input: What is the meaning of Query, Key and Value in the context of self-attention?
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step a49b74d8-09cf-4a51-87c0-e193bb82c5d9. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step e9da34c8-e36f-4ba0-93aa-2871e5f5cb81. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 6f6775ff-5c83-46e0-b130-53726789cb61. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 20563141-4fb7-42dc-8ad7-6d7113468fa9. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 280d7afb-c916-4b54-91aa-0764560b0f14. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step e61a4039-d100-4ce6-9043-ef10c85bf8f0. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 6cb20084-238d-4822-90d9-3d5429482ef2. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 6eab1685-2ed3-4144-9d3d-6c7b871e60c9. Step input: None
    Thought: In the context of self-attention, Query, Key and Value refer to three important components used in the process of computing attention scores for a given input sequence.
    Action: I'll
    Action Input: {'input': 'hello world', 'num_beams': 5}
    Observation: Error: No such tool named `I'll`.

    ValueError: Reached max iterations.
    [0;31m---------------------------------------------------------------------------[0m
    [0;31mValueError[0m                                Traceback (most recent call last)
    Cell [0;32mIn[10], line 1[0m
    [0;32m----> 1[0m response [38;5;241m=[39m [43magent[49m[38;5;241;43m.[39;49m[43mchat[49m[43m([49m[38;5;124;43m"[39;49m[38;5;124;43mWhat is the meaning of Query, Key and Value in the context of self-attention?[39;49m[38;5;124;43m"[39;49m[43m)[49m
    [1;32m      2[0m [38;5;28mprint[39m([38;5;28mstr[39m(response))

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py:311[0m, in [0;36mDispatcher.span.<locals>.wrapper[0;34m(func, instance, args, kwargs)[0m
    [1;32m    308[0m             _logger[38;5;241m.[39mdebug([38;5;124mf[39m[38;5;124m"[39m[38;5;124mFailed to reset active_span_id: [39m[38;5;132;01m{[39;00me[38;5;132;01m}[39;00m[38;5;124m"[39m)
    [1;32m    310[0m [38;5;28;01mtry[39;00m:
    [0;32m--> 311[0m     result [38;5;241m=[39m [43mfunc[49m[43m([49m[38;5;241;43m*[39;49m[43margs[49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[38;5;241;43m*[39;49m[43mkwargs[49m[43m)[49m
    [1;32m    312[0m     [38;5;28;01mif[39;00m [38;5;28misinstance[39m(result, asyncio[38;5;241m.[39mFuture):
    [1;32m    313[0m         [38;5;66;03m# If the result is a Future, wrap it[39;00m
    [1;32m    314[0m         new_future [38;5;241m=[39m asyncio[38;5;241m.[39mensure_future(result)

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/callbacks/utils.py:41[0m, in [0;36mtrace_method.<locals>.decorator.<locals>.wrapper[0;34m(self, *args, **kwargs)[0m
    [1;32m     39[0m callback_manager [38;5;241m=[39m cast(CallbackManager, callback_manager)
    [1;32m     40[0m [38;5;28;01mwith[39;00m callback_manager[38;5;241m.[39mas_trace(trace_id):
    [0;32m---> 41[0m     [38;5;28;01mreturn[39;00m [43mfunc[49m[43m([49m[38;5;28;43mself[39;49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[43margs[49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[38;5;241;43m*[39;49m[43mkwargs[49m[43m)[49m

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/agent/runner/base.py:647[0m, in [0;36mAgentRunner.chat[0;34m(self, message, chat_history, tool_choice)[0m
    [1;32m    642[0m     tool_choice [38;5;241m=[39m [38;5;28mself[39m[38;5;241m.[39mdefault_tool_choice
    [1;32m    643[0m [38;5;28;01mwith[39;00m [38;5;28mself[39m[38;5;241m.[39mcallback_manager[38;5;241m.[39mevent(
    [1;32m    644[0m     CBEventType[38;5;241m.[39mAGENT_STEP,
    [1;32m    645[0m     payload[38;5;241m=[39m{EventPayload[38;5;241m.[39mMESSAGES: [message]},
    [1;32m    646[0m ) [38;5;28;01mas[39;00m e:
    [0;32m--> 647[0m     chat_response [38;5;241m=[39m [38;5;28;43mself[39;49m[38;5;241;43m.[39;49m[43m_chat[49m[43m([49m
    [1;32m    648[0m [43m        [49m[43mmessage[49m[38;5;241;43m=[39;49m[43mmessage[49m[43m,[49m
    [1;32m    649[0m [43m        [49m[43mchat_history[49m[38;5;241;43m=[39;49m[43mchat_history[49m[43m,[49m
    [1;32m    650[0m [43m        [49m[43mtool_choice[49m[38;5;241;43m=[39;49m[43mtool_choice[49m[43m,[49m
    [1;32m    651[0m [43m        [49m[43mmode[49m[38;5;241;43m=[39;49m[43mChatResponseMode[49m[38;5;241;43m.[39;49m[43mWAIT[49m[43m,[49m
    [1;32m    652[0m [43m    [49m[43m)[49m
    [1;32m    653[0m     [38;5;28;01massert[39;00m [38;5;28misinstance[39m(chat_response, AgentChatResponse)
    [1;32m    654[0m     e[38;5;241m.[39mon_end(payload[38;5;241m=[39m{EventPayload[38;5;241m.[39mRESPONSE: chat_response})

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py:311[0m, in [0;36mDispatcher.span.<locals>.wrapper[0;34m(func, instance, args, kwargs)[0m
    [1;32m    308[0m             _logger[38;5;241m.[39mdebug([38;5;124mf[39m[38;5;124m"[39m[38;5;124mFailed to reset active_span_id: [39m[38;5;132;01m{[39;00me[38;5;132;01m}[39;00m[38;5;124m"[39m)
    [1;32m    310[0m [38;5;28;01mtry[39;00m:
    [0;32m--> 311[0m     result [38;5;241m=[39m [43mfunc[49m[43m([49m[38;5;241;43m*[39;49m[43margs[49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[38;5;241;43m*[39;49m[43mkwargs[49m[43m)[49m
    [1;32m    312[0m     [38;5;28;01mif[39;00m [38;5;28misinstance[39m(result, asyncio[38;5;241m.[39mFuture):
    [1;32m    313[0m         [38;5;66;03m# If the result is a Future, wrap it[39;00m
    [1;32m    314[0m         new_future [38;5;241m=[39m asyncio[38;5;241m.[39mensure_future(result)

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/agent/runner/base.py:579[0m, in [0;36mAgentRunner._chat[0;34m(self, message, chat_history, tool_choice, mode)[0m
    [1;32m    576[0m dispatcher[38;5;241m.[39mevent(AgentChatWithStepStartEvent(user_msg[38;5;241m=[39mmessage))
    [1;32m    577[0m [38;5;28;01mwhile[39;00m [38;5;28;01mTrue[39;00m:
    [1;32m    578[0m     [38;5;66;03m# pass step queue in as argument, assume step executor is stateless[39;00m
    [0;32m--> 579[0m     cur_step_output [38;5;241m=[39m [38;5;28;43mself[39;49m[38;5;241;43m.[39;49m[43m_run_step[49m[43m([49m
    [1;32m    580[0m [43m        [49m[43mtask[49m[38;5;241;43m.[39;49m[43mtask_id[49m[43m,[49m[43m [49m[43mmode[49m[38;5;241;43m=[39;49m[43mmode[49m[43m,[49m[43m [49m[43mtool_choice[49m[38;5;241;43m=[39;49m[43mtool_choice[49m
    [1;32m    581[0m [43m    [49m[43m)[49m
    [1;32m    583[0m     [38;5;28;01mif[39;00m cur_step_output[38;5;241m.[39mis_last:
    [1;32m    584[0m         result_output [38;5;241m=[39m cur_step_output

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py:311[0m, in [0;36mDispatcher.span.<locals>.wrapper[0;34m(func, instance, args, kwargs)[0m
    [1;32m    308[0m             _logger[38;5;241m.[39mdebug([38;5;124mf[39m[38;5;124m"[39m[38;5;124mFailed to reset active_span_id: [39m[38;5;132;01m{[39;00me[38;5;132;01m}[39;00m[38;5;124m"[39m)
    [1;32m    310[0m [38;5;28;01mtry[39;00m:
    [0;32m--> 311[0m     result [38;5;241m=[39m [43mfunc[49m[43m([49m[38;5;241;43m*[39;49m[43margs[49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[38;5;241;43m*[39;49m[43mkwargs[49m[43m)[49m
    [1;32m    312[0m     [38;5;28;01mif[39;00m [38;5;28misinstance[39m(result, asyncio[38;5;241m.[39mFuture):
    [1;32m    313[0m         [38;5;66;03m# If the result is a Future, wrap it[39;00m
    [1;32m    314[0m         new_future [38;5;241m=[39m asyncio[38;5;241m.[39mensure_future(result)

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/agent/runner/base.py:412[0m, in [0;36mAgentRunner._run_step[0;34m(self, task_id, step, input, mode, **kwargs)[0m
    [1;32m    408[0m [38;5;66;03m# TODO: figure out if you can dynamically swap in different step executors[39;00m
    [1;32m    409[0m [38;5;66;03m# not clear when you would do that by theoretically possible[39;00m
    [1;32m    411[0m [38;5;28;01mif[39;00m mode [38;5;241m==[39m ChatResponseMode[38;5;241m.[39mWAIT:
    [0;32m--> 412[0m     cur_step_output [38;5;241m=[39m [38;5;28;43mself[39;49m[38;5;241;43m.[39;49m[43magent_worker[49m[38;5;241;43m.[39;49m[43mrun_step[49m[43m([49m[43mstep[49m[43m,[49m[43m [49m[43mtask[49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[38;5;241;43m*[39;49m[43mkwargs[49m[43m)[49m
    [1;32m    413[0m [38;5;28;01melif[39;00m mode [38;5;241m==[39m ChatResponseMode[38;5;241m.[39mSTREAM:
    [1;32m    414[0m     cur_step_output [38;5;241m=[39m [38;5;28mself[39m[38;5;241m.[39magent_worker[38;5;241m.[39mstream_step(step, task, [38;5;241m*[39m[38;5;241m*[39mkwargs)

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py:311[0m, in [0;36mDispatcher.span.<locals>.wrapper[0;34m(func, instance, args, kwargs)[0m
    [1;32m    308[0m             _logger[38;5;241m.[39mdebug([38;5;124mf[39m[38;5;124m"[39m[38;5;124mFailed to reset active_span_id: [39m[38;5;132;01m{[39;00me[38;5;132;01m}[39;00m[38;5;124m"[39m)
    [1;32m    310[0m [38;5;28;01mtry[39;00m:
    [0;32m--> 311[0m     result [38;5;241m=[39m [43mfunc[49m[43m([49m[38;5;241;43m*[39;49m[43margs[49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[38;5;241;43m*[39;49m[43mkwargs[49m[43m)[49m
    [1;32m    312[0m     [38;5;28;01mif[39;00m [38;5;28misinstance[39m(result, asyncio[38;5;241m.[39mFuture):
    [1;32m    313[0m         [38;5;66;03m# If the result is a Future, wrap it[39;00m
    [1;32m    314[0m         new_future [38;5;241m=[39m asyncio[38;5;241m.[39mensure_future(result)

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/callbacks/utils.py:41[0m, in [0;36mtrace_method.<locals>.decorator.<locals>.wrapper[0;34m(self, *args, **kwargs)[0m
    [1;32m     39[0m callback_manager [38;5;241m=[39m cast(CallbackManager, callback_manager)
    [1;32m     40[0m [38;5;28;01mwith[39;00m callback_manager[38;5;241m.[39mas_trace(trace_id):
    [0;32m---> 41[0m     [38;5;28;01mreturn[39;00m [43mfunc[49m[43m([49m[38;5;28;43mself[39;49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[43margs[49m[43m,[49m[43m [49m[38;5;241;43m*[39;49m[38;5;241;43m*[39;49m[43mkwargs[49m[43m)[49m

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/agent/react/step.py:818[0m, in [0;36mReActAgentWorker.run_step[0;34m(self, step, task, **kwargs)[0m
    [1;32m    815[0m [38;5;129m@trace_method[39m([38;5;124m"[39m[38;5;124mrun_step[39m[38;5;124m"[39m)
    [1;32m    816[0m [38;5;28;01mdef[39;00m [38;5;21mrun_step[39m([38;5;28mself[39m, step: TaskStep, task: Task, [38;5;241m*[39m[38;5;241m*[39mkwargs: Any) [38;5;241m-[39m[38;5;241m>[39m TaskStepOutput:
    [1;32m    817[0m [38;5;250m    [39m[38;5;124;03m"""Run step."""[39;00m
    [0;32m--> 818[0m     [38;5;28;01mreturn[39;00m [38;5;28;43mself[39;49m[38;5;241;43m.[39;49m[43m_run_step[49m[43m([49m[43mstep[49m[43m,[49m[43m [49m[43mtask[49m[43m)[49m

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/agent/react/step.py:576[0m, in [0;36mReActAgentWorker._run_step[0;34m(self, step, task)[0m
    [1;32m    572[0m reasoning_steps, is_done [38;5;241m=[39m [38;5;28mself[39m[38;5;241m.[39m_process_actions(
    [1;32m    573[0m     task, tools, output[38;5;241m=[39mchat_response
    [1;32m    574[0m )
    [1;32m    575[0m task[38;5;241m.[39mextra_state[[38;5;124m"[39m[38;5;124mcurrent_reasoning[39m[38;5;124m"[39m][38;5;241m.[39mextend(reasoning_steps)
    [0;32m--> 576[0m agent_response [38;5;241m=[39m [38;5;28;43mself[39;49m[38;5;241;43m.[39;49m[43m_get_response[49m[43m([49m
    [1;32m    577[0m [43m    [49m[43mtask[49m[38;5;241;43m.[39;49m[43mextra_state[49m[43m[[49m[38;5;124;43m"[39;49m[38;5;124;43mcurrent_reasoning[39;49m[38;5;124;43m"[39;49m[43m][49m[43m,[49m[43m [49m[43mtask[49m[38;5;241;43m.[39;49m[43mextra_state[49m[43m[[49m[38;5;124;43m"[39;49m[38;5;124;43msources[39;49m[38;5;124;43m"[39;49m[43m][49m
    [1;32m    578[0m [43m[49m[43m)[49m
    [1;32m    579[0m [38;5;28;01mif[39;00m is_done:
    [1;32m    580[0m     task[38;5;241m.[39mextra_state[[38;5;124m"[39m[38;5;124mnew_memory[39m[38;5;124m"[39m][38;5;241m.[39mput(
    [1;32m    581[0m         ChatMessage(content[38;5;241m=[39magent_response[38;5;241m.[39mresponse, role[38;5;241m=[39mMessageRole[38;5;241m.[39mASSISTANT)
    [1;32m    582[0m     )

    File [0;32m~/MEGA/Honorar/Generative AI/script/.venv/lib/python3.10/site-packages/llama_index/core/agent/react/step.py:437[0m, in [0;36mReActAgentWorker._get_response[0;34m(self, current_reasoning, sources)[0m
    [1;32m    435[0m     [38;5;28;01mraise[39;00m [38;5;167;01mValueError[39;00m([38;5;124m"[39m[38;5;124mNo reasoning steps were taken.[39m[38;5;124m"[39m)
    [1;32m    436[0m [38;5;28;01melif[39;00m [38;5;28mlen[39m(current_reasoning) [38;5;241m==[39m [38;5;28mself[39m[38;5;241m.[39m_max_iterations:
    [0;32m--> 437[0m     [38;5;28;01mraise[39;00m [38;5;167;01mValueError[39;00m([38;5;124m"[39m[38;5;124mReached max iterations.[39m[38;5;124m"[39m)
    [1;32m    439[0m [38;5;28;01mif[39;00m [38;5;28misinstance[39m(current_reasoning[[38;5;241m-[39m[38;5;241m1[39m], ResponseReasoningStep):
    [1;32m    440[0m     response_step [38;5;241m=[39m cast(ResponseReasoningStep, current_reasoning[[38;5;241m-[39m[38;5;241m1[39m])

    [0;31mValueError[0m: Reached max iterations.

As you can see, the model request ends up with errors. The model is not
powerful enough to answer in the structured manner we need for the
function-calling of the tool. To circumvent this, we can try a
function-calling-finetuned model:

We can try to solve this issue by using a language model that is
finetuned on function calling:

``` python
fc_llm = LMStudio(model_name="phi-3-mini-4k-instruct-function-calling",
        base_url="http://localhost:1234/v1",
    temperature=0.2,
    request_timeout=600)

agent = ReActAgent.from_tools(tools=[find_references_tool],llm=fc_llm, verbose=True)
response = agent.chat("What is the meaning of Query, Key and Value in the context of self-attention?")
print(str(response))
```

    > Running step 717acece-81ed-4bda-9cad-23a7184fc33c. Step input: What is the meaning of Query, Key and Value in the context of self-attention?
    Thought: (Implicit) I can answer without any more tools!
    Answer:  In the context of self-attention, "Query" refers to a vector that represents the current input or query. It acts as a filter for determining which parts of the input should be attended to more heavily.

    "Key" refers to another set of vectors associated with each element in the input sequence. These keys are used by the self-attention mechanism to determine how much attention should be given to each part of the input. The output of the query-key computation is a weighting factor that determines the relative importance of each element in the input.

    Finally, "Value" refers to the actual content associated with each element in the input sequence. These values are combined with the weighted keys using a specified operation (usually some form of matrix multiplication) to produce an output vector that represents the attended-to information from the input.
     In the context of self-attention, "Query" refers to a vector that represents the current input or query. It acts as a filter for determining which parts of the input should be attended to more heavily.

    "Key" refers to another set of vectors associated with each element in the input sequence. These keys are used by the self-attention mechanism to determine how much attention should be given to each part of the input. The output of the query-key computation is a weighting factor that determines the relative importance of each element in the input.

    Finally, "Value" refers to the actual content associated with each element in the input sequence. These values are combined with the weighted keys using a specified operation (usually some form of matrix multiplication) to produce an output vector that represents the attended-to information from the input.

This model does not run into an issue with the structured output, it
does not try to use the tool anymore though.

One way to try to solve this issue is to adapt the agent-prompt:

``` python
print(agent.get_prompts()['agent_worker:system_prompt'].template)
```

    You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

    ## Tools

    You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
    This may require breaking the task into subtasks and using different tools to complete each subtask.

    You have access to the following tools:
    {tool_desc}


    ## Output Format

    Please answer in the same language as the question and use the following format:

    ```
    Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
    Action: tool name (one of {tool_names}) if using a tool.
    Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
    ```

    Please ALWAYS start with a Thought.

    NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

    Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

    If this format is used, the user will respond in the following format:

    ```
    Observation: tool response
    ```

    You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

    ```
    Thought: I can answer without using any more tools. I'll use the user's language to answer
    Answer: [your answer here (In the same language as the user's question)]
    ```

    ```
    Thought: I cannot answer the question with the provided tools.
    Answer: [your answer here (In the same language as the user's question)]
    ```

    ## Current Conversation

    Below is the current conversation consisting of interleaving human and assistant messages.

This we can adapt in the following way:

``` python
from llama_index.core import PromptTemplate
new_agent_template_str = """
You are designed to help answer questions based on a collection of paper-excerpts.

## Tools

You have access to tools that allow you to query paper-content. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask. Do not answer without tool-usage if a tool can be used to answer a question. Do try to find a text passage to back up your claims whenever possible. Do not answer without reference if the appropriate text is available in the tools you have access to.

You have access to the following tools:
{tool_desc}


## Output Format

Please answer in the same language as the question and use the following format:

\`\`\`
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
\`\`\`


Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.
...
## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
"""
new_agent_template = PromptTemplate(new_agent_template_str)
agent.update_prompts(
    {"agent_worker:system_prompt": new_agent_template}
)
```

We can test this new prompt with the same question:

``` python
response = agent.chat("What is the meaning of Query, Key and Value in the context of self-attention?")
print(str(response))
```

    > Running step 6d31cdb7-9acb-4322-8b13-42e869d76a8d. Step input: What is the meaning of Query, Key and Value in the context of self-attention?
    Thought: (Implicit) I can answer without any more tools!
    Answer:  In the context of self-attention, "Query" refers to a vector that represents the current input or query. It acts as a filter for determining which parts of the input should be attended to more heavily.

    "Key" refers to another set of vectors associated with each element in the input sequence. These keys are used by the self-attention mechanism to determine how much attention should be given to each part of the input. The output of the query-key computation is a weighting factor that determines the relative importance of each element in the input.

    Finally, "Value" refers to the actual content associated with each element in the input sequence. These values are combined with the weighted keys using a specified operation (usually some form of matrix multiplication) to produce an output vector that represents the attended-to information from the input.
     In the context of self-attention, "Query" refers to a vector that represents the current input or query. It acts as a filter for determining which parts of the input should be attended to more heavily.

    "Key" refers to another set of vectors associated with each element in the input sequence. These keys are used by the self-attention mechanism to determine how much attention should be given to each part of the input. The output of the query-key computation is a weighting factor that determines the relative importance of each element in the input.

    Finally, "Value" refers to the actual content associated with each element in the input sequence. These values are combined with the weighted keys using a specified operation (usually some form of matrix multiplication) to produce an output vector that represents the attended-to information from the input.

The model still tries to answer without the tool.

Let’s try to ask a more specific question:

``` python
response = agent.chat("How does the paper 'Attention is all you need' define the term self attention?")
print(str(response))
```

    > Running step 550e01a4-d723-4c9b-ab96-44f163f1284f. Step input: How does the paper 'Attention is all you need' define the term self attention?
    Thought: (Implicit) I can answer without any more tools!
    Answer:  The paper "Attention Is All You Need" defines self-attention as a mechanism for computing a weighted sum of values, where the weights are computed by performing a dot product between keys and queries, followed by applying a softmax function to obtain normalized probabilities. This allows each element in the input sequence to attend selectively to other elements based on their relative importance, which is determined by the similarity between the key and query vectors. The resulting weighted sum of values forms the output representation for that particular position in the sequence.
     The paper "Attention Is All You Need" defines self-attention as a mechanism for computing a weighted sum of values, where the weights are computed by performing a dot product between keys and queries, followed by applying a softmax function to obtain normalized probabilities. This allows each element in the input sequence to attend selectively to other elements based on their relative importance, which is determined by the similarity between the key and query vectors. The resulting weighted sum of values forms the output representation for that particular position in the sequence.

Still no dice.

One solution to this problem is to just use a bigger model:

``` python
llm = LMStudio(model_name="llama-3.2-3b-instruct", #3 Billion instead of 1
        base_url="http://localhost:1234/v1",
    temperature=0.2,
    request_timeout=600)


agent = ReActAgent.from_tools(tools=[find_references_tool],llm=llm, verbose=True)

response = agent.chat("How does the paper 'Attention is all you need' define the term self attention?")
print(str(response))
```

    > Running step ca2821e4-0d40-4619-9291-dc8594d537c8. Step input: How does the paper 'Attention is all you need' define the term self attention?
    Thought: The current language of the user is English. I need to use a tool to help me answer the question.
    Action: find_references
    Action Input: {'question': 'What defines self-attention in the context of Attention is all you Need', 'properties': AttributedDict([('title', 'Question'), ('type', 'string')])}
    Observation: Error: find_references() got an unexpected keyword argument 'properties'
    > Running step 99f9538c-7098-44b5-b2f8-700a6cd08da8. Step input: None
    Thought: The tool find_references does not accept a 'properties' argument. I should have checked the documentation before using the tool.
    Action: research
    Action Input: {'query': 'self-attention in Attention is all you need paper definition', 'num_results': 3}
    Observation: Error: No such tool named `research`.
    > Running step 49191a4e-3189-4c4a-a313-82f437e44cf4. Step input: None
    Thought: I don't have a tool called 'research'. I should have used find_references again to get the definition of self-attention.
    Action: find_references
    Action Input: {'question': 'What defines self-attention in the context of Attention is all you need'}
    Observation: Error: name 'q' is not defined
    > Running step 170262a7-dacd-4b04-b583-735f06972b4d. Step input: None
    Thought: The tool find_references expects a JSON object with a 'question' key, but I passed it as a keyword argument. I should have used the 'question' string directly.
    Action: find_references
    Action Input: {'question': 'What defines self-attention in the context of Attention is all you need'}
    Observation: Error: name 'q' is not defined
    > Running step c3dc0f39-6af7-45f0-a68f-8c22a7c3a570. Step input: None
    Thought: The error message still indicates that 'q' is not defined, but I passed a valid 'question' string. This suggests that there might be an issue with the tool's implementation.
    Answer: Self-attention in the context of the paper "Attention is all you Need" refers to a mechanism where the model attends to all positions in an input sequence simultaneously and weighs their importance. It's a key component of the Transformer architecture, introduced by Vaswani et al. in their 2017 paper, not just "Attention is all you Need".
    Self-attention in the context of the paper "Attention is all you Need" refers to a mechanism where the model attends to all positions in an input sequence simultaneously and weighs their importance. It's a key component of the Transformer architecture, introduced by Vaswani et al. in their 2017 paper, not just "Attention is all you Need".

This is not always feasible though.

Another way to use the retrieval-pipeline is to not give a weak model
the opportunity to mess up the tool calling. This can be implemented by
using a query-engine instead of the retriever. This directly wraps the
retrieval in a LLM-Summarization-Module that only returns summaries.

Doing this, we can use two separate models for each part of the task -
one for the planning and answering and one for the structured
summarization:

``` python
query_engine = index.as_query_engine(use_async=False, llm=fc_llm, verbose=True)
response = query_engine.query("What is the meaning of Query, Key and Value in the context of self-attention?")
print(str(response))
```

     In the context of self-attention, a "query" represents the input to be processed by the attention mechanism, while "keys" and "values" are derived from the same input. The keys correspond to specific features or aspects of the input data, while the values represent the corresponding outputs that result from applying those key-value pairs through an attention function. Essentially, these three components help facilitate selective focus on different parts of the input data during processing by assigning varying weights to different elements based on their importance in achieving a desired outcome.

Finally an answer we can work with!
