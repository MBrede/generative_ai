from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

question = 'Why is the sky blue?'

facts = [
    "The exact origin of the idiom 'feel blue' is uncertain. It is believed to have emerged during the 17th century. At that time, the word 'blue' was associated with sadness or gloom in various cultures. The connection might have stemmed from phrases like 'a blue devils' or 'blue Monday,' which were used to describe feelings of depression or despondency.",
    "Sunlight reaches Earth's atmosphere and is scattered in all directions by all the gases and particles in the air. Blue light is scattered more than the other colors because it travels as shorter, smaller waves. This is why we see a blue sky most of the time.",
    "Ultramarine was historically the most prestigious and expensive of blue pigments. It was produced from lapis lazuli, a mineral whose major source was the mines of Sar-e-Sang in what is now northeastern Afghanistan.",
    "'Blue (Da Ba Dee)' is a song by Italian music group Eiffel 65. It was first released in October 1998 in Italy by Skooby Records and became internationally successful the following year.[3] It is the lead single of the group's 1999 debut album, Europop.",
    "Blue's Clues is an American interactive educational children's television series created by Traci Paige Johnson, Todd Kessler, and Angela C. Santomero. It premiered on Nickelodeon's Nick Jr. block on September 8, 1996,[2] and concluded its run on August 6, 2006,[1] with a total of six seasons and 143 episodes"
]