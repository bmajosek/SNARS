import networkx as nx
import matplotlib.pyplot as plt

G1 = nx.karate_club_graph()

G2 = nx.davis_southern_women_graph()

G3 = nx.les_miserables_graph()

G4 = nx.florentine_families_graph()

G5 = nx.random_geometric_graph(50, 0.25)

G6 = nx.krackhardt_kite_graph()

graphs = [
    ("Zachary's Karate Club", G1),
    ("Davis Southern Women", G2),
    ("Les Misérables", G3),
    ("Florentine Families", G4),
    ("Random Geometric Network", G5),
    ("Krackhardt Kite", G6),
]

for name, G in graphs:
    plt.figure(figsize=(7, 5))
    nx.draw(G, with_labels=False, node_size=100, node_color='skyblue')
    plt.title(f"{name}")
    plt.show()
    
    # dodac pdopsiy