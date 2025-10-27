import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def spring_energy(G, pos, k=0.1, repulsion=0.01):
    e = 0
    nodes = list(G.nodes())
    for i, v in enumerate(nodes):
        for j, u in enumerate(nodes):
            if i >= j:
                continue
            delta = pos[v] - pos[u]
            dist = np.linalg.norm(delta) + 1e-4
            e += repulsion / dist**2
            if G.has_edge(v, u):
                e += 0.5 * k * (dist - 1.0)**2
    return e


def spring_forces(G, pos, k=0.1, repulsion=0.01):
    F = {v: np.zeros(2) for v in G.nodes()}
    nodes = list(G.nodes())

    for i, v in enumerate(nodes):
        for j, u in enumerate(nodes):
            if i == j:
                continue
            delta = pos[v] - pos[u]
            dist = np.linalg.norm(delta) + 1e-4
            dir = delta / dist

            F[v] += repulsion * dir / dist**2
            if G.has_edge(v, u):
                F[v] += -k * (dist - 1.0) * dir

    return F


def simulated_annealing_with_animation(G, steps=200, temp=1.0, cooling=0.97):
    pos = {v: np.random.rand(2) * 5 for v in G.nodes()}
    history = []

    for _ in range(steps):
        forces = spring_forces(G, pos)
        new_pos = {v: pos[v] + temp * forces[v] for v in G.nodes()}

        e_old = spring_energy(G, pos)
        e_new = spring_energy(G, new_pos)

        if e_new < e_old or np.random.rand() < np.exp((e_old - e_new) / temp):
            pos = new_pos

        history.append({v: pos[v].copy() for v in G.nodes()})
        temp *= cooling

    return pos, history


def animate_graph(G, history, interval=100):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        ax.clear()
        ax.set_title(f"Krok {frame}")
        nx.draw(G, history[frame], node_size=300, node_color="lightblue", edge_color="gray", ax=ax)

    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, repeat=False)
    plt.show()
    return anim


def run():
    G = nx.barabasi_albert_graph(15, 2)
    pos, history = simulated_annealing_with_animation(G, steps=150, temp=0.5, cooling=0.98)

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_size=300, node_color="lightgreen", edge_color="gray")
    plt.title("Końcowy układ (metoda sprężynowa)")
    plt.show()

    animate_graph(G, history)


if __name__ == "__main__":
    run()
