This introductory script on the topic of "Generative AI" was created for the elective module "Generative AI" as given to the master students of the master programme "Data Science" at the University of Applied Sciences Kiel and was built using `quarto`.

The intention of this script is not to be a complete guide to all things generative AI, but to give an overview of the topics and applications of these emerging techniques.


# Contents and learning objectives

From the [module database entry](https://moduldatenbank.fh-kiel.de/de-DE/Module/Details/762426b4-8da1-468e-b89a-98263c047d27?versionId=1):

Open Source Language Models
 - Overview of model lists
 - Ollama
 - Generation of synthetic text as training sets

 Agent Systems
 - Llamaindex, LangChain & Haystack
 - Function calling
 - Data analysis

 Embeddings and Vector Stores
 - Semantic Search
 - Retrieval-augmented generation
 - Recommendations

 AI Image Generators
 - Generative Adversarial Networks (GANs)
 - Variational Autoencoders / Diffusion Models
 - Generative approaches for image dataset augmentation

 Fine-Tuning of LLMs and Diffusion Models
 - Examples: LoRA, QLoRA, MoRA


 The goal is for the students to reach the following learning objectives:

Students
- know the fundamentals of generative AI systems.
- know various modern applications of generative AI systems.
- know the theoretical foundations and practical applications of generative AI systems.

Students
- are able to explain and apply various open-source language models.
- are able to implement and utilize agent systems and their functionalities.
- are able to understand and use embeddings and vector stores for semantic search and recommendations.
- are able to explain and practically apply different methods for image generation.
- are able to fine-tune large language models (LLMs) and diffusion models for specific tasks.

Students
- are able to successfully organize teamwork for generative AI projects.
- are able to report and present team solutions for practical project tasks.
- are able to interpret and communicate the approaches in technical and functional terms.

Students
- are able to work professionally in the field of generative AI systems.
- are able to give and accept professional feedback to different topics of generative AI systems.
- are able to select relevant scientific literature about generative AI systems.

## Schedule:

| Number: | CW: | Date: | Title: | Topics: |
|---:|:--:|---:|---|---|
|1|46| 12.11.  | Getting started with (L)LMs| Language Model Basics  |
||  |         || Choosing open source models |
||  |         || Basics of using open source models (Huggingface, Ollama, LLM-Studio, Llama.cpp, ...)  |
|2|46|  13.11. | Prompting| Prompting strategies  |
||  |         || Generation of synthetic texts  |
|3|47|  19.11. | Agent basics| Fundamentals of agents and train-of-thought prompting |
||  |         || Examples of agent-frameworks (Llamaindex, LangChain & Haystack)|
|4|47|  20.11. | Embedding-based agent-systems| Semantic embeddings and vector stores |
||  |         || Retrieval augmented and interleaved generation  |
|5|48| 26.11.  | Function Calling| Code generation and function calling  |
||  |         || Data analysis  |
|6|48|  27.11. | Agent interaction | Constitutional AI Tuning |
||  |         || Preventing prompt injections |
|7|49| 3.12.  |AI image generation I|  AI image generator basics |
||  |         || Basics of using Open Source AI image generation models |
||  |         ||  Generative Adversarial Networks (GANs) |
|8|49| 4.12.  |AI image generation II| Multimodal embeddings |
||  |         || Variational Autoencoders / Diffusion Models  |
|9|50|  10.12. |Augmentation of image datasets| (Generative) approaches for image dataset augmentation |
|10|50| 11.12.  |Finetuning Basics| Basics of Finetuning strategies  |
||  |         || Alignment and Finetuning of (L)LMs |
|11|51|  17.12. |Rank adaptation| Fundamentals of High and Low-Rank Adaptation of Language and Diffusion Models  |
||  |         || (Q)LoRA fine-tuning using Unsloth |
|12|51| 18.12.  |Project presentations||



## In-session Aufgaben
Termin:
1. Kleine Sprachmodelle zum Laufen bringen mit unterschiedlichen Backends
2. Prompting-Strategien testen [Datensatz](https://huggingface.co/datasets/mteb/mtop_intent/viewer/de)
3. Agent, der Zeit bis Semesterende, Kieler Woche, ... ausrechnet
4. Datensatz(?) in vector-store überführen und dann durchsuchen und Agent zugänglich machen
5. Agent-System zur deskriptiven Analyse, Beispiel mit PandasAI?
6. Update eines Datensatzes nach Constitution
