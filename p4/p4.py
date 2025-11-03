import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import networkit as nk
import powerlaw
import random

# ------------------------------------------------------------
# --- Utility functions ---
# ------------------------------------------------------------

def create_ba_graph(num_nodes=100000, m=500):
    """Generate Barabási-Albert graph and degree list."""
    G = nx.barabasi_albert_graph(num_nodes, m)
    degrees = np.array([deg for _, deg in G.degree()])
    return G, degrees

# ------------------------------------------------------------
# --- Exercises 1-8: Power law analysis ---
# ------------------------------------------------------------

def ex1_histogram(degrees):
    plt.hist(degrees, bins=50)
    plt.title("Exercise 1 - Degree Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

def ex2_double_log_scale(degrees):
    plt.figure(figsize=(8,5))
    plt.hist(degrees, bins=50, color='steelblue', edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Exercise 2 - Histogram (double log scale)")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def ex3_log_binning(degrees):
    df = pd.DataFrame(degrees, columns=['Degree'])
    max_d = max(degrees)
    min_d = max(min(degrees), 1)
    bins = np.logspace(np.log10(min_d), np.log10(max_d), num=5)
    sns.histplot(df, x='Degree', bins=bins)
    plt.title("Exercise 3 - Logarithmic Binning")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

def ex4_survival_function(G):
    degrees = np.array([deg for _, deg in G.degree()])
    degree_counts = np.bincount(degrees)
    total_nodes = G.number_of_nodes()
    degree_range = np.arange(len(degree_counts))
    survival = np.array([sum(degree_counts[k:]) / total_nodes for k in degree_range])
    plt.figure(figsize=(10,6))
    plt.loglog(range(len(survival)), survival, marker='o', linestyle='-', markersize=5)
    plt.xlabel('Degree k')
    plt.ylabel('S(k)')
    plt.title('Exercise 4 - Survival Function')
    plt.grid(True)
    plt.show()
    return degree_range, survival, degree_counts

def ex6_linear_regression(degree_range, survival, degree_counts):
    x_min = 5
    valid = degree_counts[degree_range] > x_min
    k = degree_range[valid]
    S_k = survival[valid]
    log_k, log_Sk = np.log10(k), np.log10(S_k)
    slope, intercept, *_ = stats.linregress(log_k, log_Sk)
    alpha = 1 - slope
    print(f"Exercise 6 - Estimated alpha (α): {alpha:.4f}")
    plt.figure(figsize=(10,6))
    plt.loglog(k, S_k, 'o', label='Data')
    plt.loglog(k, 10**(intercept + slope*log_k), '--', color='red', label='Linear Fit')
    plt.legend()
    plt.title("Linear Regression Fit (Exercise 6)")
    plt.show()
    return alpha

def ex7_mle_alpha(degrees, x_min):
    filtered = degrees[degrees >= x_min]
    n = len(filtered)
    log_ratios = np.log(filtered / x_min)
    alpha_hat = 1 + n / np.sum(log_ratios)
    print(f"Exercise 7 - MLE α (x_min={x_min}): {alpha_hat:.4f}")
    return alpha_hat

def ex8_compare_xmin(degrees):
    print("Exercise 8 - α estimates for various x_min values:")
    for x_min in range(1, 20):
        ex7_mle_alpha(degrees, x_min)

def analyze_powerlaw(G, degrees):
    ex1_histogram(degrees)
    ex2_double_log_scale(degrees)
    ex3_log_binning(degrees)
    degree_range, survival, degree_counts = ex4_survival_function(G)
    ex6_linear_regression(degree_range, survival, degree_counts)
    ex7_mle_alpha(degrees, 5)
    ex8_compare_xmin(degrees)

# ------------------------------------------------------------
# --- Project 4.1-4.9 ---
# ------------------------------------------------------------

# P4.2-P4.4 Nearest neighbor degree and random edge switching

def random_edge_switching(graph, num_switches):
    nk_graph = nk.nxadapter.nx2nk(graph)
    switch = nk.randomization.EdgeSwitching(nk_graph, num_switches)
    switch.run()
    return nk.nxadapter.nk2nx(switch.getGraph())

def average_neighbor_degree_plot(G, ax, title):
    avg_nn = nx.average_neighbor_degree(G)
    degs = dict(G.degree())
    ax.scatter(list(degs.values()), list(avg_nn.values()), alpha=0.6, edgecolors='k')
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Average Neighbor Degree")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax

def p4_2_to_p4_4():
    G = nx.barabasi_albert_graph(1000, 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    average_neighbor_degree_plot(G, axes[0], "Original Network")
    G_switched = random_edge_switching(G, len(G.nodes()))
    average_neighbor_degree_plot(G_switched, axes[1], "After Random Edge Switching")
    plt.tight_layout()
    plt.show()
    print("P4.6 - Degree correlation before:", nx.degree_assortativity_coefficient(G))
    print("P4.6 - Degree correlation after :", nx.degree_assortativity_coefficient(G_switched))
    return G, G_switched

# P4.7 Compute metrics for various networks

def calculate_stats(graph):
    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    avg_deg = np.mean([d for _, d in graph.degree()])
    deg_list = [d for _, d in graph.degree()]
    fit = powerlaw.Fit(deg_list, discrete=True, verbose=False)
    alpha = fit.power_law.alpha
    path_len = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else np.nan
    corr = nx.degree_pearson_correlation_coefficient(graph)
    return {
        "Nodes": n,
        "Edges": e,
        "Avg_degree": avg_deg,
        "Alpha": alpha,
        "Path_length": path_len,
        "Assortativity": corr
    }

def p4_7_metrics(G, G_switched):
    graphs = {
        "BA Original": G,
        "BA Switched": G_switched,
        "Karate Club": nx.karate_club_graph(),
        "Watts-Strogatz": nx.watts_strogatz_graph(100, 6, 0.3),
        "Barabási-Albert": nx.barabasi_albert_graph(100, 3),
        "Random Geometric": nx.random_geometric_graph(100, 0.2),
        "Erdős-Rényi": nx.erdos_renyi_graph(100, 0.05)
    }
    results = {name: calculate_stats(g) for name, g in graphs.items()}
    df = pd.DataFrame(results).T
    print("\nP4.7 - Network Metrics Summary:")
    print(df.round(4))
    return df

# P4.8 Erdős number histograms (synthetic approximation)

def p4_8_erdos_histograms():
    np.random.seed(42)
    def generate(size, min_v, max_v, mean_v):
        d = np.random.normal(mean_v, (max_v - min_v)/4, size)
        return np.clip(np.round(d), min_v, max_v).astype(int)
    data = {
        "Fields Medal": generate(56, 2, 6, 3.3),
        "Nobel Economics": generate(76, 2, 8, 4.1),
        "Nobel Chemistry": generate(172, 3, 10, 5.5),
        "Nobel Medicine": generate(210, 3, 12, 5.5),
        "Nobel Physics": generate(200, 2, 12, 5.6)
    }
    for k, v in data.items():
        plt.figure(figsize=(8,5))
        plt.hist(v, bins=range(v.min(), v.max()+2), edgecolor='black')
        plt.title(f"P4.8 - Erdős Number Distribution ({k})")
        plt.xlabel("Erdős Number")
        plt.ylabel("Frequency")
        plt.show()

# P4.9 Bacon number histogram (static example)

def p4_9_bacon_histogram():
    data = {
        "Actor": [
            "Cillian Murphy", "Robert Downey Jr.", "Emma Stone",
            "Da'Vine Joy Randolph", "Hayao Miyazaki", "Toshio Suzuki",
            "Christopher Nolan", "Kris Bowers", "Ludwig Göransson",
            "Brad Booker", "Arthur Harari"
        ],
        "Bacon_Number": [2,2,1,2,4,4,4,3,2,3,3]
    }
    df = pd.DataFrame(data)
    plt.figure(figsize=(8,5))
    plt.hist(df["Bacon_Number"], bins=range(1, df["Bacon_Number"].max()+2), edgecolor='black')
    plt.title("P4.9 - Bacon Numbers among Selected Actors")
    plt.xlabel("Bacon Number")
    plt.ylabel("Frequency")
    plt.show()

###############################################################
# MAIN#

def main():
    G, degrees = create_ba_graph()
    analyze_powerlaw(G, degrees)
    G, G_switched = p4_2_to_p4_4()
    p4_7_metrics(G, G_switched)
    p4_8_erdos_histograms()
    p4_9_bacon_histogram()

if __name__ == "__main__":
    main()
