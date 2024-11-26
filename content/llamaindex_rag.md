# Llamaindex Rag


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

node_parser = SentenceSplitter(chunk_size=100, chunk_overlap=20)

nodes = node_parser.get_nodes_from_documents(
    [Document(text=text)], show_progress=False
)
```

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

    [NodeWithScore(node=TextNode(id_='04c12537-5f33-4d41-a4d4-df30d2aed6e4', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='22d5c0dc-d921-4790-ac6e-4f6a6d5f336f', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='772c092906000e119c69ad2e5cb90148a6c8b113d54a20fb9d5984d6a9695ee8'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='893d077f-a8ab-4a3f-9765-69ef72d46ec4', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='df269253fe4504ec666a0a40380f9399466c5bd366c7ce6c853ee45b31d4bc84')}, text='of the values, where the weight assigned to each value is computed by a compatibility function of the\nquery with the corresponding key.\n\n3.2.1   Scaled Dot-Product Attention\nWe call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of\nqueries and keys of dimension dk , and\n                                    √ values of dimension dv .', mimetype='text/plain', start_char_idx=11715, end_char_idx=12088, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.588352239002419),
     NodeWithScore(node=TextNode(id_='c42d8e8c-24ac-447a-8058-d62d198ce9eb', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='e961df5f-04be-4bf8-bba0-b30b346e6e3e', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='944203475caa494a68b2ca15140cea2278792db8546209bcc538388bf227b57d'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='12962f1d-060f-49d3-9ff9-be2dceb23736', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='46773d9899458459b747af4980832a961621033663b11cb056304074633c0f14')}, text='Self-attention, sometimes called intra-attention is an attention mechanism relating different positions\nof a single sequence in order to compute a representation of the sequence. Self-attention has been\nused successfully in a variety of tasks including reading comprehension, abstractive summarization,\ntextual entailment and learning task-independent sentence representations [4, 27, 28, 22].', mimetype='text/plain', start_char_idx=8003, end_char_idx=8396, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5581949233902119),
     NodeWithScore(node=TextNode(id_='893d077f-a8ab-4a3f-9765-69ef72d46ec4', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='04c12537-5f33-4d41-a4d4-df30d2aed6e4', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='4dc2893909c949675d444324e091b9dcae176eafe0faeb456e4f571f79863ac8'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e48f428a-1d0f-4830-8aca-82cbf4cd4b67', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='4a7481ff7440b3355d18a8f77fdbcf637903e138a37a44c74d4fd287baf610f2')}, text='We compute the dot products of the\nquery with all keys, divide each by dk , and apply a softmax function to obtain the weights on the\nvalues.\nIn practice, we compute the attention function on a set of queries simultaneously, packed together\ninto a matrix Q. The keys and values are also packed together into matrices K and V .', mimetype='text/plain', start_char_idx=12089, end_char_idx=12415, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5557579023667499),
     NodeWithScore(node=TextNode(id_='0146f53a-f1b1-4d80-a333-26746920ab9d', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='c0f333cd-8860-48e5-b177-649855617c5a', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='c5cea5e4a2c19b51c1912e3fbb06fd9f445f2ab46a888146c9540685c513a907'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='5d433fd9-785b-4f25-b3b0-5cd206b0ca37', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='e52e557964f178c114303403bfab945ce6fc6bc18fbc723bc2c110071beaf965')}, text='• The encoder contains self-attention layers. In a self-attention layer all of the keys, values\n           and queries come from the same place, in this case, the output of the previous layer in the\n           encoder. Each position in the encoder can attend to all positions in the previous layer of the\n           encoder.', mimetype='text/plain', start_char_idx=16021, end_char_idx=16345, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5531707169222685),
     NodeWithScore(node=TextNode(id_='22d5c0dc-d921-4790-ac6e-4f6a6d5f336f', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='71788dae-10dc-4341-8ebd-250a8836bce5', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='f1c9e10879cdc5796376d70528c5ccd9d988818269ef633ea539e6d2df1922d1'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='04c12537-5f33-4d41-a4d4-df30d2aed6e4', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='4dc2893909c949675d444324e091b9dcae176eafe0faeb456e4f571f79863ac8')}, text='3.2   Attention\n\nAn attention function can be described as mapping a query and a set of key-value pairs to an output,\nwhere the query, keys, values, and output are all vectors. The output is computed as a weighted sum\n\n\n                                                  3\n\x0c           Scaled Dot-Product Attention                                  Multi-Head Attention\n\n\n\n\nFigure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several\nattention layers running in parallel.', mimetype='text/plain', start_char_idx=11208, end_char_idx=11712, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.5503383930857552),
     NodeWithScore(node=TextNode(id_='55481635-fcaa-4e90-9625-9b0c3bfa3109', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='923d6eec-1ba9-4972-b457-47cc1cb5e5a7', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='534fa8133845bae34a1c58d14d5fe840710190a12c4951fa24b1acaaa4ed8e35'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='ea0b511f-4179-4f64-8e5b-1cf5f6d76404', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='a958edeb1ca826ae9eb259fb9846f6fe7d822b9462583eea56914ae0383170e5')}, text='.                       .              .\n                                                                                                                 <EOS>       <EOS>            <EOS>                 <EOS>\n                                                                                                                  <pad>      <pad>             <pad>                <pad>\n\n\n\n\n     Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5\n     Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution.', mimetype='text/plain', start_char_idx=55980, end_char_idx=56574, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.46287885047540767),
     NodeWithScore(node=TextNode(id_='04b195bd-26e4-4d8c-afdc-780e96bdd345', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='c28b6b26-7bbf-4682-9399-a7804be460ae', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='c3ad5697d4d156dd0b4c85c17741ee433c10899ddffbd3575904ce08cd6736de'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='c0f333cd-8860-48e5-b177-649855617c5a', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='c5cea5e4a2c19b51c1912e3fbb06fd9f445f2ab46a888146c9540685c513a907')}, text='3.2.3    Applications of Attention in our Model\nThe Transformer uses multi-head attention in three different ways:\n\n         • In "encoder-decoder attention" layers, the queries come from the previous decoder layer,\n           and the memory keys and values come from the output of the encoder. This allows every\n           position in the decoder to attend over all positions in the input sequence.', mimetype='text/plain', start_char_idx=15478, end_char_idx=15877, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.4550194901912972),
     NodeWithScore(node=TextNode(id_='d93b8e55-28cb-417e-838a-a22abf7cfbc9', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='398e22c4-5cd8-42ed-ba1d-43f213413bc2', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='cd837bc3b60f4cff2ab7f296f85515886d65e8ca7c2a3fb9c7b10fb1c6904949'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='e9ffed0b-00f1-4408-bd5d-512f5d05138d', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='d0057b6da67faef5766281c2cae5a165b6e5396059cd7c09222a6d9e77ca985c')}, text='On each of these projected versions of\nqueries, keys and values we then perform the attention function in parallel, yielding dv -dimensional\n   4\n     To illustrate why the dot products get large, assume that the components of q and k are independent random\nvariables with mean 0 and variance 1. Then their dot product, q · k = di=1\n                                                                        P k\n                                                                              qi ki , has mean 0 and variance dk .', mimetype='text/plain', start_char_idx=14037, end_char_idx=14560, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.45141889186813816),
     NodeWithScore(node=TextNode(id_='158309a7-9a7a-47e6-ac58-1a4e98eee41b', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='e4e96748-8e42-4c45-a1b3-3e0b2a179475', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='162e546ee2aace8fdcf9330a044ed33bc46d32219ec57c876b93a1fad69425e7'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='8229d93a-1fb8-492f-9227-2b13658180f7', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='e9431f13405886d724857fa8ba6e9d0bd84affbaf2d35beedeeda36e79d95de8')}, text='4     Why Self-Attention\nIn this section we compare various aspects of self-attention layers to the recurrent and convolu-\ntional layers commonly used for mapping one variable-length sequence of symbol representations\n(x1 , ..., xn ) to another sequence of equal length (z1 , ..., zn ), with xi , zi ∈ Rd , such as a hidden\nlayer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we\nconsider three desiderata.', mimetype='text/plain', start_char_idx=20488, end_char_idx=20939, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.4348473100243987),
     NodeWithScore(node=TextNode(id_='721c5981-90a9-4046-a757-4593a362ddf7', embedding=None, metadata={}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='847f2be4-3799-41b5-80c0-b390298eba24', node_type=<ObjectType.DOCUMENT: '4'>, metadata={}, hash='74e64008cffed21d58edef5058f6cf6b3bc853bf936b83eefb70563168b73c5a'), <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='a07f95e3-64fb-4637-ac18-4a928541df80', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='b11620062050474b2e5a6317e981c8ad07b227f032ebe169b1cb4f87c8994aa6'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='165241f9-efb1-433a-896b-b6ea61168d3f', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='559f529f69207d17371f20407b6f1b4691910f9c8a90c9cefbb741e95fbf5de9')}, text='Operations\n      Self-Attention                      O(n2 · d)             O(1)                O(1)\n      Recurrent', mimetype='text/plain', start_char_idx=18618, end_char_idx=18733, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'), score=0.4276254505797798)]

The retriever can then directly be use as a tool to answer questions
about our documents:

``` python
from llama_index.core.tools import BaseTool, FunctionTool

def find_references(question: str) -> str:
    """Query a database containing the paper "Attention is all you Need" in parts.
    This paper introduced the mechanism of self-attention to the NLP-literature.
    Returns a collection of scored text-snippets that are relevant to your question."""
    return '\n'.join([f'{round(n.score,2)} - {n.node.text}' for n in retriever.retrieve(question)])


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

    > Running step 062240ab-0d21-4fdb-a603-fb386970c32f. Step input: What is the meaning of Query, Key and Value in the context of self-attention?
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 2a291a80-5090-4373-945d-3a647ac2b758. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 908425d7-8f06-4830-8585-4ff312b43c45. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 543ab12f-e5e7-4a59-b103-b7fc7bd0a3fe. Step input: None
    Observation: Error: Could not parse output. Please follow the thought-action-input format. Try again.
    > Running step 1b3cb4e3-e976-4420-a489-906b8f6c5776. Step input: None
    Thought: Let's break down what Query, Key, and Value mean in the context of self-attention.
    Action: Use
    Action Input: {'input': "What are the most relevant words for the sentence 'The quick brown fox jumps over the lazy dog'?", 'num_beams': 5}
    Observation: Error: No such tool named `Use`.
    > Running step 1101520e-54ff-42db-b327-d9d902acb957. Step input: None
    Thought: I need to find a way to input the query and parameters into a tool.
    Action: Use
    Action Input: {'input': "What are the most relevant words for the sentence 'The quick brown fox jumps over the lazy dog'?", 'num_beams': 5}
    Observation: Error: No such tool named `Use`.
    > Running step 47c5a9f6-5055-4f8c-9a3b-49f1db40abcb. Step input: None
    Thought: I'm using a different tool to find references. Let me check if it supports finding relevant text snippets for the given query.
    Action: find_references
    Action Input: {'properties': AttributedDict([('question', "What are the most relevant words for the sentence 'The quick brown fox jumps over the lazy dog'?"), ('num_beams', 5)]), 'required': ['query', 'parameters']}
    Observation: Error: find_references() got an unexpected keyword argument 'properties'

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

    > Running step 78c0a52b-55fa-4241-ade5-67c0b92b9bf3. Step input: What is the meaning of Query, Key and Value in the context of self-attention?
    Thought: (Implicit) I can answer without any more tools!
    Answer:  In the context of self-attention, "Query", "Key" and "Value" are terms used to describe different components of a neural network architecture. Here's what they mean:
    1. Query - The query component is used to retrieve information from memory banks during attention computation. It represents a set of learned parameters that enable the model to focus on specific parts of an input sequence when processing it. In other words, the query function defines how much importance we should give to each part of our input data while computing self-attention weights.
    2. Key - The key component is used to determine which parts of the input sequence are most relevant for a particular output location in the model's memory bank. It represents another set of learned parameters that help us identify important features in an input sequence during attention computation. In other words, the key function helps us decide what we should focus on when computing self-attention weights.
    3. Value - The value component is used to store the actual data corresponding to each memory bank location in a neural network architecture. It represents our stored knowledge or "memory" that can be retrieved later during attention computation. In other words, the value function holds all of the information we need to compute an output based on self-attention weights.
    In summary, query, key and value are components of a neural network architecture used in self-attention that help us focus on specific parts of our input sequence, identify important features within it, and retrieve relevant stored knowledge/memory to compute outputs.
     In the context of self-attention, "Query", "Key" and "Value" are terms used to describe different components of a neural network architecture. Here's what they mean:
    1. Query - The query component is used to retrieve information from memory banks during attention computation. It represents a set of learned parameters that enable the model to focus on specific parts of an input sequence when processing it. In other words, the query function defines how much importance we should give to each part of our input data while computing self-attention weights.
    2. Key - The key component is used to determine which parts of the input sequence are most relevant for a particular output location in the model's memory bank. It represents another set of learned parameters that help us identify important features in an input sequence during attention computation. In other words, the key function helps us decide what we should focus on when computing self-attention weights.
    3. Value - The value component is used to store the actual data corresponding to each memory bank location in a neural network architecture. It represents our stored knowledge or "memory" that can be retrieved later during attention computation. In other words, the value function holds all of the information we need to compute an output based on self-attention weights.
    In summary, query, key and value are components of a neural network architecture used in self-attention that help us focus on specific parts of our input sequence, identify important features within it, and retrieve relevant stored knowledge/memory to compute outputs.

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

    > Running step d5fb46ea-de7a-4e8b-ace7-7ed3ae6a9706. Step input: What is the meaning of Query, Key and Value in the context of self-attention?
    Thought: (Implicit) I can answer without any more tools!
    Answer:  In the context of natural language processing (NLP), "Query", "Key" and "Value" are used as components for a type of neural network architecture called Transformer model. The Transformer model employs self-attention mechanism to improve its ability to process sequential data such as text or audio. 
    Here's how these terms relate to the model:
    1. Query - A query is an input vector that represents the current state of a sequence being processed by the transformer network. It contains information about which words or tokens are currently being attended to, and helps guide the attention mechanism towards relevant parts of the input sequence.
    2. Key - The key component in a transformer model refers to a set of learned weights that help determine how much importance should be given to each word or token during self-attention computation. These keys are computed for all words or tokens in an input sequence and they form part of the attention mechanism used by the Transformer network.
    3. Value - The value component is responsible for storing information from a specific memory slot corresponding to a particular input token in the transformer model. It represents the output produced when we apply a transformation function on the query vector (which contains contextual information about the current word or token being processed) using learned weights, and then weighted-summed with the key vectors.
    In summary, Query, Key and Value are components of a neural network architecture used in Transformer models for NLP that help us process sequential data such as text by guiding attention towards relevant parts of an input sequence, identifying important features within it, and computing outputs based on self-attention weights.
     In the context of natural language processing (NLP), "Query", "Key" and "Value" are used as components for a type of neural network architecture called Transformer model. The Transformer model employs self-attention mechanism to improve its ability to process sequential data such as text or audio. 
    Here's how these terms relate to the model:
    1. Query - A query is an input vector that represents the current state of a sequence being processed by the transformer network. It contains information about which words or tokens are currently being attended to, and helps guide the attention mechanism towards relevant parts of the input sequence.
    2. Key - The key component in a transformer model refers to a set of learned weights that help determine how much importance should be given to each word or token during self-attention computation. These keys are computed for all words or tokens in an input sequence and they form part of the attention mechanism used by the Transformer network.
    3. Value - The value component is responsible for storing information from a specific memory slot corresponding to a particular input token in the transformer model. It represents the output produced when we apply a transformation function on the query vector (which contains contextual information about the current word or token being processed) using learned weights, and then weighted-summed with the key vectors.
    In summary, Query, Key and Value are components of a neural network architecture used in Transformer models for NLP that help us process sequential data such as text by guiding attention towards relevant parts of an input sequence, identifying important features within it, and computing outputs based on self-attention weights.

The model still tries to answer without the tool.

Let’s try to ask a more specific question:

``` python
response = agent.chat("How does the paper 'Attention is all you need' define the term self attention?")
print(str(response))
```

    > Running step 3c1b3050-f4a0-4b46-9006-366161df0219. Step input: How does the paper 'Attention is all you need' define the term self attention?
    Thought: (Implicit) I can answer without any more tools!
    Answer:  In the paper "Attention Is All You Need", the authors present a novel Transformer model that relies heavily on an attention mechanism to improve its ability to process sequential data such as text or audio. The paper introduces several key concepts related to this mechanism, including the notion of "self-attention". 
    Self-attention is defined in the paper as follows: given a sequence of input tokens (or words), self-attention enables us to compute contextualized representations for each token by computing attention weights over all other tokens in the sequence. These attention weights reflect how much importance we should give to each token when computing our output representation. In particular, during training, these weights are learned based on the input data itself and can be adjusted dynamically as new inputs come in. The resulting contextualized representations produced by self-attention provide a rich source of information for downstream tasks like language modeling or machine translation.
     In the paper "Attention Is All You Need", the authors present a novel Transformer model that relies heavily on an attention mechanism to improve its ability to process sequential data such as text or audio. The paper introduces several key concepts related to this mechanism, including the notion of "self-attention". 
    Self-attention is defined in the paper as follows: given a sequence of input tokens (or words), self-attention enables us to compute contextualized representations for each token by computing attention weights over all other tokens in the sequence. These attention weights reflect how much importance we should give to each token when computing our output representation. In particular, during training, these weights are learned based on the input data itself and can be adjusted dynamically as new inputs come in. The resulting contextualized representations produced by self-attention provide a rich source of information for downstream tasks like language modeling or machine translation.

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

    > Running step 9326aba5-48cf-40dd-8b85-b5da82554e5c. Step input: How does the paper 'Attention is all you need' define the term self attention?
    Thought: The current language of the user is English. I need to use a tool to help me answer the question.
    Action: find_references
    Action Input: {'properties': AttributedDict([('question', AttributedDict([('title', 'self-attention definition'), ('type', 'string')]))]), 'required': ['question'], 'type': 'object'}
    Observation: Error: find_references() got an unexpected keyword argument 'properties'
    > Running step b9bd6255-c348-473e-a031-2fd1e4e74cdf. Step input: None
    Thought: The current language of the user is English. I need to use a tool to help me answer the question, but it seems like find_references doesn't support the properties argument.
    Action: find_references
    Action Input: {'question': "How does the paper 'Attention is all you Need' define the term self attention?"}
    Observation: 0.69 - Self-attention, sometimes called intra-attention is an attention mechanism relating different positions
    of a single sequence in order to compute a representation of the sequence. Self-attention has been
    used successfully in a variety of tasks including reading comprehension, abstractive summarization,
    textual entailment and learning task-independent sentence representations [4, 27, 28, 22].
    0.52 - .                       .              .
                                                                                                                     <EOS>       <EOS>            <EOS>                 <EOS>
                                                                                                                      <pad>      <pad>             <pad>                <pad>




         Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5
         Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution.
    0.5 - <EOS>
                                                                                                                                               <pad>
                                                                                                                                                       <pad>
                                                                                                                                                       <pad>
                                                                                                                                                               <pad>
                                                                                                                                                                       <pad>
                                                                                                                                                                               <pad>
    Figure 3: An example of the attention mechanism following long-distance dependencies in the
    encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of
    the verb ‘making’, completing the phrase ‘making...more difficult’.
    0.5 - Each layer has two
    sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-
    wise fully connected feed-forward network. We employ a residual connection [11] around each of
    the two sub-layers, followed by layer normalization [1].
    0.49 - 4     Why Self-Attention
    In this section we compare various aspects of self-attention layers to the recurrent and convolu-
    tional layers commonly used for mapping one variable-length sequence of symbol representations
    (x1 , ..., xn ) to another sequence of equal length (z1 , ..., zn ), with xi , zi ∈ Rd , such as a hidden
    layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we
    consider three desiderata.
    0.47 - In the following sections, we will describe the Transformer, motivate
    self-attention and discuss its advantages over models such as [17, 18] and [9].


    3   Model Architecture

    Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35].
    Here, the encoder maps an input sequence of symbol representations (x1 , ..., xn ) to a sequence
    of continuous representations z = (z1 , ..., zn ).
    0.45 - .                       .             .
                                                                                                                  <EOS>       <EOS>            <EOS>                <EOS>
                                                                                                                   <pad>      <pad>             <pad>               <pad>




         sentence. We give two such examples above, from two different heads from the encoder self-attention
         Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the
    0.44 - Operations
          Self-Attention                      O(n2 · d)             O(1)                O(1)
          Recurrent
    0.43 - • The encoder contains self-attention layers. In a self-attention layer all of the keys, values
               and queries come from the same place, in this case, the output of the previous layer in the
               encoder. Each position in the encoder can attend to all positions in the previous layer of the
               encoder.
    0.42 - As side benefit, self-attention could yield more interpretable models. We inspect attention distributions
    from our models and present and discuss examples in the appendix. Not only do individual attention
    heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic
    and semantic structure of the sentences.


    5     Training

    This section describes the training regime for our models.
    > Running step ac9c9225-596e-4c84-8e86-1518a4fd7d55. Step input: None
    Thought: The current language of the user is English. I was able to retrieve relevant information about self-attention from the paper "Attention is all you Need". It seems that the authors define self-attention as an attention mechanism that relates different positions of a single sequence in order to compute a representation of the sequence.
    Answer: Self-attention, also known as intra-attention, is an attention mechanism that computes a representation of a sequence by attending to different positions within the same sequence. It has been used successfully in various tasks such as reading comprehension, abstractive summarization, textual entailment, and learning task-independent sentence representations.
    Self-attention, also known as intra-attention, is an attention mechanism that computes a representation of a sequence by attending to different positions within the same sequence. It has been used successfully in various tasks such as reading comprehension, abstractive summarization, textual entailment, and learning task-independent sentence representations.

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

     In the context of self-attention, "Query" refers to the keys that are used to retrieve relevant information from a sequence. "Key" represents the values associated with each element in the sequence, which determine their importance or relevance. "Value" corresponds to the actual data being processed by the attention mechanism.

Finally an answer we can work with!
