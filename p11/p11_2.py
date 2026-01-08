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

def run_comparison(N=1000, m=5, beta_values=None, gamma=0.2, timesteps=200):
    """Run simulations for different beta values and compare with analytical threshold"""
    if beta_values is None:
        beta_values = np.linspace(0.01, 0.5, 5)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create model to get network properties
    model = SISModelBA(N=N, m=m, beta=beta_values[0], gamma=gamma)
    
    # Calculate threshold once (it's the same for all beta values)
    threshold, _ = analytical_predictions(model.avg_degree, model.second_moment, 
                                       beta_values[0], gamma)
    
    # Run simulation for each beta value
    for beta in beta_values:
        # Run simulation
        model = SISModelBA(N=N, m=m, beta=beta, gamma=gamma)
        infected_fractions = model.run_simulation(timesteps)
        
        # Calculate predictions
        lambda_param = beta / gamma
        _, equilibrium = analytical_predictions(model.avg_degree, model.second_moment, 
                                             beta, gamma)
        
        # Plot simulation results
        plt.plot(infected_fractions, label=f'λ = {lambda_param:.2f}')
        
        # Plot predicted equilibrium
        plt.axhline(y=equilibrium, color='gray', linestyle=':', alpha=0.5)
    
    # Calculate and display critical lambda
    threshold_lambda = threshold * gamma
    plt.text(0.02, 0.05, f'Critical λc = {threshold_lambda:.3f}', 
             transform=plt.gca().transAxes, color='red')
    
    plt.xlabel('Time steps')
    plt.ylabel('Fraction of infected nodes')
    plt.title('SIS Model on BA Network: Infection Evolution')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the simulation
if __name__ == "__main__":
    # Parameters
    N = 1000  # Number of nodes
    m = 5     # Number of edges for preferential attachment
    gamma = 0.2  # Recovery rate
    beta_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25])  # Infection rates
    
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