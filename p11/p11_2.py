import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class SISModelBA:
    def __init__(self, N, m, beta, gamma, initial_infected_fraction=0.1):
        """
        Initialize SIS model on BA graph
        
        Parameters:
        N: int - number of nodes
        m: int - number of edges to attach from new node to existing nodes
        beta: float - infection rate
        gamma: float - recovery rate
        initial_infected_fraction: float - initial fraction of infected nodes
        """
        self.N = N
        self.beta = beta
        self.gamma = gamma
        
        # Create BA graph
        self.G = nx.barabasi_albert_graph(N, m)
        self.avg_degree = np.mean([d for n, d in self.G.degree()])
        self.second_moment = np.mean([d*d for n, d in self.G.degree()])
        
        # Initialize node states (0: Susceptible, 1: Infected)
        initial_infected = int(N * initial_infected_fraction)
        self.states = np.zeros(N)
        infected_nodes = np.random.choice(N, initial_infected, replace=False)
        self.states[infected_nodes] = 1
        
    def step(self):
        """Simulate one time step of the SIS model"""
        new_states = self.states.copy()
        
        # Process each node
        for node in range(self.N):
            if self.states[node] == 1:  # Infected
                # Recovery process
                if np.random.random() < self.gamma:
                    new_states[node] = 0
            else:  # Susceptible
                # Infection process
                neighbors = list(self.G.neighbors(node))
                infected_neighbors = sum(self.states[neighbor] for neighbor in neighbors)
                if infected_neighbors > 0:
                    infection_prob = 1 - (1 - self.beta) ** infected_neighbors
                    if np.random.random() < infection_prob:
                        new_states[node] = 1
        
        self.states = new_states
        return np.mean(self.states)  # Return fraction of infected nodes

    def run_simulation(self, timesteps):
        """Run simulation for specified number of timesteps"""
        infected_fractions = np.zeros(timesteps)
        for t in tqdm(range(timesteps)):
            infected_fractions[t] = self.step()
        return infected_fractions

def analytical_predictions(avg_degree, second_moment, beta, gamma):
    """Calculate analytical predictions for the SIS model on BA network
    
    Parameters:
    avg_degree: float - average degree of the network
    second_moment: float - second moment of degree distribution
    beta: float - infection rate
    gamma: float - recovery rate
    
    Returns:
    threshold: float - epidemic threshold
    equilibrium: float - predicted equilibrium infection rate
    """
    # From lecture slides: λc = ⟨k⟩/⟨k²⟩
    threshold = avg_degree / second_moment
    
    lambda_param = beta / gamma
    
    # Calculate predicted equilibrium state
    if lambda_param > threshold:
        equilibrium = 1 - gamma / (beta * avg_degree)
    else:
        equilibrium = 0
        
    return threshold, equilibrium

def run_comparison(N=1000, m=5, beta_values=None, gamma=0.2, timesteps=200, num_runs=3, num_graphs=3):
    """Run simulations for different beta values on multiple BA graphs with averaging"""
    if beta_values is None:
        beta_values = np.linspace(0.01, 0.5, 5)
    
    # Handle network parameters
    if isinstance(N, int):
        N_values = [N] * num_graphs
    else:
        N_values = N
        num_graphs = len(N)
    
    if isinstance(m, int):
        m_values = [m] * num_graphs
    else:
        m_values = m
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create sample model to get threshold
    sample_model = SISModelBA(N=N_values[0], m=m_values[0], beta=beta_values[0], gamma=gamma)
    threshold, _ = analytical_predictions(sample_model.avg_degree, sample_model.second_moment, 
                                       beta_values[0], gamma)
    threshold_lambda = threshold * gamma
    
    print(f"\n{'='*70}")
    print(f"Running {num_runs} simulations × {num_graphs} different graphs per beta")
    print(f"Total: {num_runs * num_graphs} simulations per beta value")
    print(f"BA Networks:")
    for i, (n, m_val) in enumerate(zip(N_values, m_values)):
        print(f"  Graph {i+1}: N={n}, m={m_val}")
    print(f"Critical threshold: λc = {threshold_lambda:.4f}")
    print(f"{'='*70}\n")
    
    # Store results for each beta
    results_by_beta = {}
    
    # Run simulations for each beta value
    for beta in beta_values:
        lambda_param = beta / gamma
        print(f"Processing β = {beta:.4f}, λ = {lambda_param:.4f}")
        
        all_runs = []
        graph_info = []
        
        for graph_idx, (N_val, m_val) in enumerate(zip(N_values, m_values)):
            for run in range(num_runs):
                model = SISModelBA(N=N_val, m=m_val, beta=beta, gamma=gamma)
                infected_fractions = model.run_simulation(timesteps)
                all_runs.append(infected_fractions)
            
            model = SISModelBA(N=N_val, m=m_val, beta=beta, gamma=gamma)
            graph_info.append((N_val, m_val, model.avg_degree, model.second_moment))
        
        all_runs = np.array(all_runs)
        mean_fractions = np.mean(all_runs, axis=0)
        std_fractions = np.std(all_runs, axis=0)
        
        avg_avg_degree = np.mean([info[2] for info in graph_info])
        avg_second_moment = np.mean([info[3] for info in graph_info])
        _, equilibrium = analytical_predictions(avg_avg_degree, avg_second_moment, beta, gamma)
        
        results_by_beta[lambda_param] = {
            'mean': mean_fractions,
            'std': std_fractions,
            'equilibrium': equilibrium
        }
    
    # Plot mean trajectories with confidence bands
    colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))
    for idx, (lambda_param, results) in enumerate(sorted(results_by_beta.items())):
        plt.plot(results['mean'], color=colors[idx], linewidth=2.5, label=f'λ = {lambda_param:.4f}')
        plt.fill_between(range(len(results['mean'])), 
                        results['mean'] - results['std'],
                        results['mean'] + results['std'],
                        color=colors[idx], alpha=0.2)
        plt.axhline(y=results['equilibrium'], color=colors[idx], linestyle=':', alpha=0.5, linewidth=1.5)
    
    plt.xlabel('Time steps', fontsize=12)
    plt.ylabel('Fraction of infected nodes', fontsize=12)
    plt.title(f'SIS Model on BA Network: Mean Infection Dynamics\n(averaged across all graphs and runs, shaded = +-1 std dev, Critical λc = {threshold_lambda:.4f})', fontsize=13)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('sis_ba_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run the simulation
if __name__ == "__main__":
    # Parameters
    N = 1000  # Number of nodes
    m = 5     # Number of edges for preferential attachment
    gamma = 0.2  # Recovery rate
    beta_values = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25])  # Infection rates
    
    run_comparison(N=N, m=m, beta_values=beta_values, gamma=gamma)


# From chat help:
# The mean-field theory used for analytical predictions (i∞ = 1 - γ/(β⟨k⟩)) makes strong assumptions:

# Infinite network size
# Perfect mixing
# Homogeneous contact rates
# No dynamical correlations

# BA Network Reality:
# BA networks violate several of these assumptions:

# We have finite size (N=1000)
# Contact patterns are highly heterogeneous (some nodes are hubs)
# Strong local clustering effects
# The degree distribution follows a power law


# Scale-Free Nature Impact:
# The very low critical threshold (λc = 0.009) is correct and is a key feature of BA networks
# This explains why infection persists even with low transmission rates
# But the actual equilibrium values differ from theory because the mean-field approximation doesn't capture the network's hierarchical structure