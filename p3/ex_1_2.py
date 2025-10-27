import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import geom
from scipy.special import zeta as riemann_zeta

def plot_histogram(data, bins=50, title="", xlabel="", ylabel="", log=False, sigma_lines=False):
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='green')

    if sigma_lines:
        mu, sigma = np.mean(data), np.std(data)
        for i in range(-3, 4):
            x = mu + i * sigma
            plt.axvline(x, color='red', linestyle='--')
            plt.text(x, plt.ylim()[1] * 0.05, f'{i}σ', rotation=90, color='red')

    if log:
        plt.xscale('log')
        plt.yscale('log')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def exercise_1_normal():
    data = np.random.randn(10_000)
    plot_histogram(data, title="Rozkład normalny (z liniami sigma)", sigma_lines=True)


def exercise_2_geometric(p=0.3):
    k = np.arange(1, 30)
    pk = geom.pmf(k, p)
    plt.bar(k, pk, color='skyblue')
    plt.title('Rozkład geometryczny')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.tight_layout()
    plt.show()


def exercise_3_powerlaw(alpha=3, x_min=1):
    x = (np.random.pareto(alpha - 1, 10_000) + 1) * x_min
    plot_histogram(x, title="Rozkład potęgowy (log-log)", xlabel='x', ylabel='gęstość', log=True)


def exercise_4_zeta(s=2.5, k_max=30):
    k = np.arange(1, k_max + 1)
    normalization = riemann_zeta(s)
    pk = 1 / (k ** s * normalization)
    plt.bar(k, pk, color='orange')
    plt.title('Rozkład Zeta (Zipfa)')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.tight_layout()
    plt.show()


def exercise_5_networks(n=100, m=3, p=0.01):
    G_ba = nx.barabasi_albert_graph(n, m)
    G_er = nx.erdos_renyi_graph(n, p)

    deg_ba = [d for _, d in G_ba.degree()]
    deg_er = [d for _, d in G_er.degree()]

    plt.hist(deg_ba, bins=5, alpha=0.6, label='BA')
    plt.hist(deg_er, bins=5, alpha=0.6, label='ER')
    plt.title('Rozkład stopni w sieciach BA i ER')
    plt.xlabel('Stopień wierzchołka')
    plt.ylabel('Liczność')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"BA - średni stopień: {np.mean(deg_ba):.2f}, wariancja: {np.var(deg_ba):.2f}")
    print(f"ER - średni stopień: {np.mean(deg_er):.2f}, wariancja: {np.var(deg_er):.2f}")

    return G_ba


# P3.1

def p3_1_graph_layouts(G):
    """Porównanie różnych układów graficznych (layoutów) dla tego samego grafu."""
    layouts = {
        "Spring layout": nx.spring_layout(G),
        "Kamada-Kawai": nx.kamada_kawai_layout(G),
        "Kołowy": nx.circular_layout(G),
        "Warstwowy (shell)": nx.shell_layout(G)
    }

    for name, layout in layouts.items():
        plt.figure()
        nx.draw(G, pos=layout, node_size=20, edge_color='gray', alpha=0.7)
        plt.title(name)
        plt.tight_layout()
        plt.show()


def run_all():
    print("=== Ćwiczenie 1 ===")
    exercise_1_normal()

    print("=== Ćwiczenie 2 ===")
    exercise_2_geometric()

    print("=== Ćwiczenie 3 ===")
    exercise_3_powerlaw()

    print("=== Ćwiczenie 4 ===")
    exercise_4_zeta()

    print("=== Ćwiczenie 5 ===")
    G = exercise_5_networks()

    print("=== P3.1 ===")
    p3_1_graph_layouts(G)


run_all()
