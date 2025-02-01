import spacy
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def preprocess_text(text):
    doc = nlp(text)
    return doc

def extract_entities(doc):
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_relationships(doc):
    relationships = []
    for token in doc:
        if token.dep_ in ("nsubj", "dobj", "pobj", "poss", "prep"):
            head = token.head.text
            child = token.text
            relationships.append((head, child))
    return relationships


def build_network_graph(entities, relationships):
    G = nx.DiGraph()
    
    # Add entity nodes
    for entity, label in entities:
        G.add_node(entity, label=label)
    
    # Add edges based on relationships
    for head, child in relationships:
        G.add_edge(head, child)
    
    # Draw the graph
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    
    text = """
"Pristina Airport – Possible administrative irregularity regarding tender procedures involving Vendor 1 and Vendor 2

Allegation

Two companies with the same owner took part at least three times in the same Airport tenders.

Background Information

The Kosovo citizen, Vendor 1 and Vendor 2 Representative, is the owner and Director of the Pristina-based Vendor 1 and also a 51% shareholder of the Pristina-Ljubljana-based company Vendor 2. Both companies have their residences at the same address in Pristina.

Both Vendor 1 and Vendor 2 submitted three times in 2003 for the same tenders:

Supply and Mounting of Sonic System in the Fire Station Building. Winner was Vendor 2 with €1,530 followed by Vendor 1 with €1,620. The third company, Vendor 3, did not provide a price offer.

Cabling of Flat Display Information System (FIDS). Winner was Vendor 1 with €15,919 followed by Vendor 2 with €19,248.70. The other two competitors, Vendor 3 and Vendor 4, offered prices of Euro 19,702 and Euro 21,045.

Purchase and fixing of Cramer Antenna. Winner was again Vendor 1 with €3,627.99 followed by Vendor 2 with €3,921. The other two competitors, Vendor 3 and Vendor 4, offered prices of €4,278 and €4,670."
    """
    
    doc = preprocess_text(text)
    entities = extract_entities(doc)
    relationships = extract_relationships(doc)
    
    print("Extracted Entities:", entities)
    print("Extracted Relationships:", relationships)
    
    build_network_graph(entities, relationships)
    
    
# df = pd.read_csv("news_dataset.csv")

# df["Text"] = df["Text"].str.replace(r'\n+', ' ', regex=True)

# text_list = df["Text"].tolist()

# print(text_list[1:])