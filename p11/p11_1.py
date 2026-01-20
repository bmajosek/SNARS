import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class SISModel:
    def __init__(self, N, p, beta, gamma, initial_infected_fraction=0.1):
        """
        Initialize SIS model on ER graph
        
        Parameters:
        N: int - number of nodes
        p: float - probability of edge creation (ER parameter)
        beta: float - infection rate
        gamma: float - recovery rate
        initial_infected_fraction: float - initial fraction of infected nodes
        """
        self.N = N
        self.beta = beta
        self.gamma = gamma
        
        # Create ER graph
        self.G = nx.erdos_renyi_graph(N, p)
        self.avg_degree = np.mean([d for n, d in self.G.degree()])
        
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

def analytical_predictions(avg_degree, beta, gamma):
    """Calculate analytical predictions for the SIS model
    
    Parameters:
    avg_degree: float - average degree of the network
    beta: float - infection rate
    gamma: float - recovery rate
    
    Returns:
    threshold: float - epidemic threshold
    equilibrium: float - predicted equilibrium infection rate
    """
    lambda_param = beta / gamma
    threshold = 1 / avg_degree
    
    # Calculate predicted equilibrium state
    if lambda_param > threshold:
        equilibrium = 1 - gamma / (beta * avg_degree)
    else:
        equilibrium = 0
        
    return threshold, equilibrium


def run_comparison(N=1000, p=0.01, beta_values=None, gamma=0.2, timesteps=200, num_runs=3, num_graphs=3):
    """Run simulations for different beta values on multiple ER graphs with averaging"""
    if beta_values is None:
        beta_values = np.linspace(0.01, 0.5, 5)
    
    # Handle network parameters
    if isinstance(N, int):
        N_values = [N] * num_graphs
    else:
        N_values = N
        num_graphs = len(N)
    
    if isinstance(p, float) or isinstance(p, int):
        p_values = [p] * num_graphs
    else:
        p_values = p
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create sample model to get threshold
    sample_model = SISModel(N=N_values[0], p=p_values[0], beta=beta_values[0], gamma=gamma)
    threshold, _ = analytical_predictions(sample_model.avg_degree, beta_values[0], gamma)
    threshold_lambda = threshold * gamma
    
    print(f"\n{'='*70}")
    print(f"Running {num_runs} simulations x {num_graphs} different graphs per beta")
    print(f"Total: {num_runs * num_graphs} simulations per beta value")
    print(f"ER Networks:")
    for i, (n, p_val) in enumerate(zip(N_values, p_values)):
        print(f"  Graph {i+1}: N={n}, p={p_val}")
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
        
        for graph_idx, (N_val, p_val) in enumerate(zip(N_values, p_values)):
            for run in range(num_runs):
                model = SISModel(N=N_val, p=p_val, beta=beta, gamma=gamma)
                infected_fractions = model.run_simulation(timesteps)
                all_runs.append(infected_fractions)
            
            model = SISModel(N=N_val, p=p_val, beta=beta, gamma=gamma)
            graph_info.append((N_val, p_val, model.avg_degree))
        
        all_runs = np.array(all_runs)
        mean_fractions = np.mean(all_runs, axis=0)
        std_fractions = np.std(all_runs, axis=0)
        
        avg_avg_degree = np.mean([info[2] for info in graph_info])
        _, equilibrium = analytical_predictions(avg_avg_degree, beta, gamma)
        
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
    plt.title(f'SIS Model on ER Graph: Mean Infection Dynamics\n(averaged across all graphs and runs, shaded = +-1 std dev, Critical λc = {threshold_lambda:.4f})', fontsize=13)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('sis_er_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run the simulation
if __name__ == "__main__":
    # Parameters
    N = 1000  # Number of nodes
    p = 0.01  # ER graph edge probability
    gamma = 0.2  # Recovery rate
    beta_values = np.array([0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25])  # Infection rates
    
    run_comparison(N=N, p=p, beta_values=beta_values, gamma=gamma)


# equilibrium state: i∞ = 1 - γ/(β⟨k⟩)
#  <k> - avg degree

# λc = 1/(⟨k⟩s(0)) where:


# λ = β/γ (ratio of infection rate to recovery rate)
# ⟨k⟩ is the average degree
# s(0) is the initial fraction of susceptible nodes


# The discrepancy likely comes from:


# Finite size effects (N=1000 nodes)
# Network structure (ER graph doesn't provide complete mixing)
# Local clustering effects that aren't captured by the mean-field theory