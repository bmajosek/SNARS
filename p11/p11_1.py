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

def run_comparison(N=1000, p=0.01, beta_values=None, gamma=0.2, timesteps=200):
    """Run simulations for different beta values and compare with analytical threshold"""
    if beta_values is None:
        beta_values = np.linspace(0.01, 0.5, 5)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create model to get network properties
    model = SISModel(N=N, p=p, beta=beta_values[0], gamma=gamma)
    
    # Calculate threshold once (it's the same for all beta values)
    threshold, _ = analytical_predictions(model.avg_degree, beta_values[0], gamma)
    
    # Run simulation for each beta value
    for beta in beta_values:
        # Run simulation
        model = SISModel(N=N, p=p, beta=beta, gamma=gamma)
        infected_fractions = model.run_simulation(timesteps)
        
        # Calculate predictions
        lambda_param = beta / gamma
        _, equilibrium = analytical_predictions(model.avg_degree, beta, gamma)
        
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
    plt.title('SIS Model on ER Graph: Infection Evolution')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the simulation
if __name__ == "__main__":
    # Parameters
    N = 1000  # Number of nodes
    p = 0.01  # ER graph edge probability
    gamma = 0.2  # Recovery rate
    beta_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25])  # Infection rates
    
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