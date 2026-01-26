#!/usr/bin/env python3
"""Generate test corpus from REAL natural text - LONGER passages.

Uses actual coherent text passages and compares:
- Intact: original text (long-range dependencies preserved)
- Shuffled: same sentences in random order (dependencies broken)

Documents are ~600-800 tokens to allow proper analysis with default settings.
"""

import json
import random
import re
from pathlib import Path


# Longer real text passages - each should be 600+ tokens
# These are extended versions with more content

NATURAL_PASSAGES = [
    # Passage 1: Amazon rainforest (extended)
    """The Amazon rainforest is the world's largest tropical rainforest, covering approximately 5.5 million square kilometers. It spans across nine countries in South America, with the majority located in Brazil. The forest is often called the "lungs of the Earth" because it produces about 20 percent of the world's oxygen. Scientists estimate that the Amazon is home to approximately 390 billion individual trees, representing around 16,000 different species. The biodiversity found here is unmatched anywhere else on the planet.

The Amazon River, which flows through the heart of the forest, is the second longest river in the world. It carries more water than any other river system, accounting for roughly 20 percent of all freshwater that flows into the world's oceans. The river and its tributaries support an incredible variety of aquatic life, including pink river dolphins, giant otters, and piranhas. More than 3,000 species of fish have been identified in its waters, with scientists believing many more remain undiscovered.

Indigenous peoples have inhabited the Amazon for at least 11,000 years. Today, approximately 400 distinct indigenous groups live within the forest, speaking more than 300 different languages. These communities have developed sophisticated knowledge of the forest's medicinal plants and sustainable harvesting techniques. Their traditional practices have helped maintain the forest's ecological balance for generations.

Deforestation poses the greatest threat to the Amazon's survival. Each year, thousands of square kilometers of forest are cleared for cattle ranching, soybean farming, and logging. This destruction releases massive amounts of carbon dioxide into the atmosphere, contributing to global climate change. Scientists warn that continued deforestation could push the Amazon past a tipping point, transforming large portions of the rainforest into savanna.

Conservation efforts have intensified in recent decades. Brazil has established numerous protected areas and indigenous territories that limit development. International organizations have invested billions of dollars in preservation programs. Satellite monitoring now tracks deforestation in real time, enabling faster responses to illegal clearing. Despite these efforts, the battle to save the Amazon remains ongoing and uncertain.""",

    # Passage 2: Marie Curie (extended)
    """Marie Curie was born Maria Sklodowska in Warsaw, Poland, on November 7, 1867. She grew up in a family that valued education, despite living under Russian occupation that restricted Polish culture and language. Her father was a physics and mathematics teacher, and her mother ran a prestigious boarding school. From an early age, Marie showed exceptional intelligence and a passion for learning.

At the time, women in Poland were not permitted to attend university. Marie worked as a governess for several years to help fund her older sister's medical studies in Paris. In exchange, her sister later helped support Marie's own education. In 1891, at age 24, Marie finally moved to Paris to study physics and mathematics at the Sorbonne, one of the few European universities that admitted women.

In Paris, Marie lived in a tiny attic apartment and often survived on little more than bread and chocolate. Despite these hardships, she excelled in her studies, earning degrees in both physics and mathematics. It was during this time that she met Pierre Curie, a professor at the School of Physics. They shared a passion for scientific research and were married in 1895. Their partnership would prove to be one of the most productive collaborations in scientific history.

Marie became fascinated by Henri Becquerel's recent discovery of mysterious rays emitted by uranium. She decided to investigate these rays for her doctoral thesis. Working in a converted shed with minimal equipment, she and Pierre made breakthrough after breakthrough. Marie coined the term "radioactivity" to describe the phenomenon. In 1898, she announced the discovery of two new elements: polonium, named after her homeland, and radium.

In 1903, Marie Curie became the first woman to win a Nobel Prize, sharing the physics prize with Pierre and Henri Becquerel. Tragically, Pierre was killed in a street accident in 1906. Despite her grief, Marie continued their research and took over his teaching position at the Sorbonne, becoming the university's first female professor. In 1911, she won a second Nobel Prize, this time in chemistry, for her work on radium. She remains the only person ever to win Nobel Prizes in two different sciences. Marie died in 1934 from aplastic anemia, almost certainly caused by her prolonged exposure to radiation.""",

    # Passage 3: The Human Heart (extended)
    """The human heart is a remarkable muscular organ that serves as the body's primary pump, circulating blood throughout the entire cardiovascular system. Located in the chest cavity between the lungs, slightly to the left of center, it is roughly the size of a clenched fist and weighs between 8 to 12 ounces. Despite its relatively small size, the heart performs an astonishing amount of work, beating approximately 100,000 times per day and pumping about 2,000 gallons of blood.

The heart consists of four chambers: two upper chambers called atria and two lower chambers called ventricles. The right atrium receives deoxygenated blood from the body through two large veins, the superior and inferior vena cava. This blood then passes through the tricuspid valve into the right ventricle, which pumps it to the lungs for oxygenation. The newly oxygenated blood returns to the heart, entering the left atrium through the pulmonary veins.

From the left atrium, blood flows through the mitral valve into the left ventricle, the heart's most powerful chamber. The left ventricle's thick muscular walls generate enough force to pump blood throughout the entire body via the aorta, the body's largest artery. This continuous cycle of receiving, oxygenating, and distributing blood occurs thousands of times each day without conscious effort.

The heart's electrical system controls its rhythmic contractions. The sinoatrial node, often called the heart's natural pacemaker, generates electrical impulses that spread through the atria, causing them to contract. These impulses then reach the atrioventricular node, which relays them to the ventricles after a brief delay. This coordinated electrical activity ensures that the chambers contract in the proper sequence, maximizing the efficiency of blood flow.

Heart disease remains the leading cause of death worldwide. Risk factors include high blood pressure, high cholesterol, smoking, obesity, and physical inactivity. Regular exercise strengthens the heart muscle, improves circulation, and helps maintain healthy blood pressure. A balanced diet low in saturated fats and rich in fruits, vegetables, and whole grains supports cardiovascular health. Medical advances have dramatically improved treatment options, including medications, surgical procedures, and implantable devices like pacemakers and defibrillators.""",

    # Passage 4: Great Wall of China (extended)
    """The Great Wall of China stands as one of humanity's most impressive architectural achievements, stretching across thousands of miles of mountains, deserts, and grasslands. Construction of defensive walls in China began as early as the 7th century BC, when individual states built barriers to protect their territories from neighboring rivals and nomadic tribes from the north. These early walls were typically made of packed earth and were relatively modest in scale.

The first emperor of unified China, Qin Shi Huang, initiated the most ambitious wall-building project in 221 BC. After conquering the warring states and establishing the Qin Dynasty, he ordered the connection and extension of existing walls to create a continuous defensive barrier. Hundreds of thousands of soldiers, peasants, and prisoners were conscripted for this massive undertaking. Working conditions were brutal, and countless laborers died during construction. Their bodies were sometimes buried within the wall itself.

Subsequent dynasties continued to maintain, rebuild, and extend the wall over the following centuries. The Han Dynasty pushed the wall westward to protect the Silk Road trade routes. The wall fell into disrepair during some periods and was extensively renovated during others. The most iconic sections that tourists visit today were built during the Ming Dynasty, from the 14th to 17th centuries. These sections feature brick and stone construction, watchtowers, and garrison stations.

The Great Wall is not a single continuous structure but rather a series of walls and fortifications. Its total length, including all branches and secondary sections, extends approximately 13,000 miles. The main wall follows the historical northern borders of China, traversing diverse terrain from the mountains near Beijing to the deserts of the Gobi. Watchtowers placed at regular intervals allowed soldiers to send smoke signals during the day and fire signals at night to warn of approaching enemies.

Today, the Great Wall attracts millions of visitors from around the world each year. Several sections near Beijing have been restored and developed for tourism, complete with cable cars and tourist facilities. However, many remote sections continue to crumble from neglect and erosion. Conservation efforts face significant challenges due to the wall's immense length and the difficulty of accessing isolated areas. The Great Wall was designated a UNESCO World Heritage Site in 1987 and remains a powerful symbol of Chinese civilization.""",

    # Passage 5: Photosynthesis (extended)
    """Photosynthesis is the fundamental biological process by which plants, algae, and certain bacteria convert light energy into chemical energy, producing the oxygen that sustains most life on Earth. This remarkable process takes place primarily in the leaves of plants, where specialized cells contain organelles called chloroplasts. Each chloroplast houses the molecular machinery necessary for capturing sunlight and transforming it into usable energy.

Chlorophyll, the green pigment that gives plants their characteristic color, plays a central role in photosynthesis. This molecule absorbs light most efficiently in the red and blue portions of the visible spectrum while reflecting green light, which is why plants appear green to our eyes. When chlorophyll absorbs a photon of light, it becomes energized, triggering a cascade of chemical reactions known as the light-dependent reactions.

During these light-dependent reactions, water molecules are split into hydrogen and oxygen. The oxygen is released into the atmosphere as a byproduct, the very oxygen that humans and other animals breathe. The hydrogen ions and energized electrons are used to produce ATP and NADPH, two molecules that carry chemical energy. These energy carriers are essential for the next stage of photosynthesis.

The second stage, called the Calvin cycle or light-independent reactions, takes place in the stroma of the chloroplast. Here, the energy stored in ATP and NADPH is used to convert carbon dioxide from the air into glucose, a simple sugar. This process, known as carbon fixation, is catalyzed by the enzyme RuBisCO, one of the most abundant proteins on Earth. The glucose produced serves as the primary energy source for the plant's growth and metabolism.

Plants have evolved various adaptations to optimize photosynthesis under different environmental conditions. Some plants, called C4 plants, have developed mechanisms to concentrate carbon dioxide and reduce water loss in hot, dry environments. Cacti and other succulents use a variation called CAM photosynthesis, opening their stomata only at night to conserve water. These adaptations demonstrate the remarkable flexibility of photosynthetic organisms in adapting to diverse ecological niches.

The importance of photosynthesis extends far beyond individual plants. Through this process, plants form the foundation of virtually all food chains on Earth. They convert solar energy into forms that animals can consume, passing energy through ecosystems. Additionally, photosynthesis has shaped Earth's atmosphere over billions of years, producing the oxygen-rich environment that enabled complex life to evolve. Understanding photosynthesis is crucial for addressing challenges like food security and climate change.""",
]


def split_into_sentences(text: str) -> list:
    """Split text into sentences."""
    # Handle paragraph breaks and sentence endings
    text = text.replace('\n\n', ' ')
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def generate_intact_document(passage: str, doc_id: int) -> dict:
    """Generate document with original intact text."""
    # Clean up whitespace
    text = ' '.join(passage.split())
    return {
        "doc_id": f"intact_{doc_id:03d}",
        "author_id": f"author_{doc_id % 3}",
        "population": "intact",
        "text": text,
    }


def generate_shuffled_document(passage: str, doc_id: int, seed: int) -> dict:
    """Generate document with sentences shuffled randomly."""
    rng = random.Random(seed + doc_id)
    text = ' '.join(passage.split())  # Clean whitespace first
    sentences = split_into_sentences(text)
    rng.shuffle(sentences)

    return {
        "doc_id": f"shuffled_{doc_id:03d}",
        "author_id": f"author_{doc_id % 3}",
        "population": "shuffled",
        "text": " ".join(sentences),
    }


def main():
    output_dir = Path(__file__).parent.parent / "sanity_check_data"
    output_dir.mkdir(exist_ok=True)

    print("Generating natural text test corpus (LONGER passages)...")
    print()

    documents = []

    # Generate multiple versions of each passage with different shuffle seeds
    for repeat in range(4):  # 4 repeats = 20 intact + 20 shuffled
        for i, passage in enumerate(NATURAL_PASSAGES):
            doc_id = repeat * len(NATURAL_PASSAGES) + i

            intact_doc = generate_intact_document(passage, doc_id)
            shuffled_doc = generate_shuffled_document(passage, doc_id, seed=42 + repeat * 100)

            documents.append(intact_doc)
            documents.append(shuffled_doc)

    # Save corpus
    corpus_path = output_dir / "natural_test_corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    print(f"Saved {len(documents)} documents to: {corpus_path}")
    print(f"  - {len(documents)//2} intact (original order)")
    print(f"  - {len(documents)//2} shuffled (random order)")
    print()

    # Check token lengths
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        lengths = [len(tokenizer.encode(doc['text'])) for doc in documents[:10]]
        print(f"Token lengths (first 10 docs): min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
        print()
    except:
        pass

    # Show samples
    print("=" * 70)
    print("SAMPLE: Intact (first 300 chars)")
    print("=" * 70)
    print(documents[0]["text"][:300] + "...")
    print()

    print("=" * 70)
    print("SAMPLE: Shuffled (first 300 chars)")
    print("=" * 70)
    print(documents[1]["text"][:300] + "...")
    print()

    print("=" * 70)
    print("Run with these settings in GUI:")
    print("  - Max documents: 15+")
    print("  - Distance preset: Default [32, 64, 128, 256]")
    print("  - Max windows per doc: 2")
    print("  - Max targets per window: 3")
    print("=" * 70)


if __name__ == "__main__":
    main()
