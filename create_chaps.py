chapters = [
  {
    "Number": 1,
    "CW": 46,
    "Date": "12.11.",
    "Title": "Getting started with (L)LMs",
    "Topics": [
      "Language Model Basics",
      "Choosing open source models",
      "Basics of using open source models (Huggingface, Ollama, LLM-Studio, Llama.cpp, ...)"
    ],
    "File": "getting_started_with_llms.qmd"
  },
  {
    "Number": 2,
    "CW": 46,
    "Date": "13.11.",
    "Title": "Prompting",
    "Topics": [
      "Prompting strategies",
      "Generation of synthetic texts"
    ],
    "File": "prompting.qmd"
  },
  {
    "Number": 3,
    "CW": 47,
    "Date": "19.11.",
    "Title": "Agent basics",
    "Topics": [
      "Fundamentals of agents and train-of-thought prompting",
      "Examples of agent-frameworks (Llamaindex, LangChain & Haystack)"
    ],
    "File": "agent_basics.qmd"
  },
  {
    "Number": 4,
    "CW": 47,
    "Date": "20.11.",
    "Title": "Embedding-based agent-systems",
    "Topics": [
      "Semantic embeddings and vector stores",
      "Retrieval augmented and interleaved generation"
    ],
    "File": "embeddings.qmd"
  },
  {
    "Number": 5,
    "CW": 48,
    "Date": "26.11.",
    "Title": "Function Calling",
    "Topics": [
      "Code generation and function calling",
      "Data analysis"
    ],
    "File": "function_calling.qmd"
  },
  {
    "Number": 6,
    "CW": 48,
    "Date": "27.11.",
    "Title": "Agent interaction",
    "Topics": [
      "Constitutional AI Tuning",
      "Preventing prompt injections"
    ],
    "File": "agent_interaction.qmd"
  },
  {
    "Number": 7,
    "CW": 49,
    "Date": "3.12.",
    "Title": None,
    "Topics": [],
    "File": ""
  },
  {
    "Number": 8,
    "CW": 49,
    "Date": "4.12.",
    "Title": "AI image generation",
    "Topics": [
      "AI image generator basics",
      "Basics of using Open Source AI image generation models",
      "Generative Adversarial Networks (GANs)"
    ],
    "File": "generator_basics.qmd"
  },
  {
    "Number": 9,
    "CW": 50,
    "Date": "10.12.",
    "Title": "AI image generation II",
    "Topics": [
      "Multimodal embeddings",
      "Variational Autoencoders / Diffusion Models"
    ],
    "File": ""
  },
    {
    "Number": 10,
    "CW": 50,
    "Date": "11.12.",
    "Title": "Augmentation of image datasets",
    "Topics": [
      "(Generative) approaches for image dataset augmentation"
    ],
    "File": "augmentation.qmd"
  },
  {
    "Number": 11,
    "CW": 51,
    "Date": "17.12.",
    "Title": "Finetuning Approaches",
    "Topics": [
      "Basics of Finetuning strategies",
      "Alignment and Finetuning of (L)LMs"
    ],
    "File": "finetuning_approaches.qmd"
  },
  {
    "Number": 12,
    "CW": 51,
    "Date": "18.12.",
    "Title": "Rank adaptation",
    "Topics": [
      "Fundamentals of High and Low-Rank Adaptation of Language and Diffusion Models",
      "(Q)LoRA fine-tuning using Unsloth"
    ],
    "File": "rank_adaptation.qmd"
  }
]

import os
for chapter in chapters:
    if chapter['Title'] is not None and len(chapter['Title']) > 0:
      try:
        with open(os.path.join("content/", chapter["File"]), 'w') as f:
            f.write(
                "---\ntoc-title: '![](../imgs/cover.jpg){width=240px}" +
                f"<br> <h3>{chapter['Title']}</h3>'\n---\n\n" +
                f"# {chapter['Title']}\n\n" +
                "\n\n".join([f"## {topic}" for topic in chapter["Topics"]]) +
                "\n\n## Further Readings\n\n" 
            )
      except IsADirectoryError:
        pass