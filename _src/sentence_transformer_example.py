from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import textwrap


model = SentenceTransformer("all-MiniLM-L6-v2")

question = 'Why is the sky blue?'

facts = [
    "The exact origin of the idiom 'feel blue' is uncertain. It is believed to have emerged during the 17th century. At that time, the word 'blue' was associated with sadness or gloom in various cultures. The connection might have stemmed from phrases like 'a blue devils' or 'blue Monday,' which were used to describe feelings of depression or despondency.",
    "Sunlight reaches Earth's atmosphere and is scattered in all directions by all the gases and particles in the air. Blue light is scattered more than the other colors because it travels as shorter, smaller waves. This is why we see a blue sky most of the time.",
    "Ultramarine was historically the most prestigious and expensive of blue pigments. It was produced from lapis lazuli, a mineral whose major source was the mines of Sar-e-Sang in what is now northeastern Afghanistan.",
    "'Blue (Da Ba Dee)' is a song by Italian music group Eiffel 65. It was first released in October 1998 in Italy by Skooby Records and became internationally successful the following year.[3] It is the lead single of the group's 1999 debut album, Europop.",
    "Blue's Clues is an American interactive educational children's television series created by Traci Paige Johnson, Todd Kessler, and Angela C. Santomero. It premiered on Nickelodeon's Nick Jr. block on September 8, 1996,[2] and concluded its run on August 6, 2006,[1] with a total of six seasons and 143 episodes"
]

embeddings = model.encode([question] + facts)

dists = cosine_similarity(embeddings)[0]

output_path = "imgs/retrieval_illustration.png"

G = nx.Graph()

nodes = ["Question"] + [f"Fact {i+1}" for i in range(len(facts))]
G.add_nodes_from(nodes)

for i, dist in enumerate(dists[1:]):
    G.add_edge("Question", f"Fact {i+1}", weight=dist)

pos = nx.spring_layout(G,k=1)

plt.figure(figsize=(12, 10))

nx.draw(
    G, pos, with_labels=True, node_color="lightblue", font_size=10, font_weight="bold"
)

edge_labels = nx.get_edge_attributes(G, "weight")

nx.draw_networkx_edge_labels(
    G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}
)

annotations = {
    "Question": question,
    **{f"Fact {i+1}": fact for i, fact in enumerate(facts)},
}

max_width = 100
for node, text in annotations.items():
    x, y = pos[node]
    cut_text = textwrap.wrap(text, width=max_width)
    wrapped_text = "\n".join(cut_text)
    plt.text(
        x, y + len(cut_text) * .005 + .04, wrapped_text, fontsize=8, wrap=True, 
        ha="center", bbox=dict(boxstyle="round", 
                               facecolor="white", 
                               alpha=0.5)
    )


plt.savefig(output_path, bbox_inches="tight")

plt.close()