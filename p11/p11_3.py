import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class SIRSModel:
    def __init__(self, N, p, beta, gamma, alpha, initial_infected_fraction=0.1):
        """
        Initialize SIRS model on ER graph
        
        Parameters:
        N: int - number of nodes
        p: float - probability of edge creation (ER parameter)
        beta: float - infection rate
        gamma: float - recovery rate
        alpha: float - rate of losing immunity (R→S transition)
        initial_infected_fraction: float - initial fraction of infected nodes
        """
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        
        # Create network
        self.G = nx.erdos_renyi_graph(N, p)
        self.avg_degree = np.mean([d for n, d in self.G.degree()])
        
        # Initialize node states (0: Susceptible, 1: Infected, 2: Recovered)
        self.states = np.zeros(N)
        initial_infected = int(N * initial_infected_fraction)
        infected_nodes = np.random.choice(N, initial_infected, replace=False)
        self.states[infected_nodes] = 1
        
    def step(self):
        """Simulate one time step of the SIRS model"""
        new_states = self.states.copy()
        
        # Process each node
        for node in range(self.N):
            if self.states[node] == 0:  # Susceptible
                # Infection process
                neighbors = list(self.G.neighbors(node))
                infected_neighbors = sum(self.states[neighbor] == 1 for neighbor in neighbors)
                if infected_neighbors > 0:
                    infection_prob = 1 - (1 - self.beta) ** infected_neighbors
                    if np.random.random() < infection_prob:
                        new_states[node] = 1
                        
            elif self.states[node] == 1:  # Infected
                # Recovery process
                if np.random.random() < self.gamma:
                    new_states[node] = 2
                    
            else:  # Recovered
                # Loss of immunity process
                if np.random.random() < self.alpha:
                    new_states[node] = 0
        
        self.states = new_states
        return self.get_state_fractions()
    
    def get_state_fractions(self):
        """Return fractions of nodes in each state"""
        total = len(self.states)
        susceptible = np.sum(self.states == 0) / total
        infected = np.sum(self.states == 1) / total
        recovered = np.sum(self.states == 2) / total
        return susceptible, infected, recovered

    def run_simulation(self, timesteps):
        """Run simulation for specified number of timesteps"""
        susceptible = np.zeros(timesteps)
        infected = np.zeros(timesteps)
        recovered = np.zeros(timesteps)
        
        for t in tqdm(range(timesteps)):
            s, i, r = self.step()
            susceptible[t] = s
            infected[t] = i
            recovered[t] = r
            
        return susceptible, infected, recovered

def run_simulations(N=1000, p=0.01, beta=0.3, gamma=0.2, alpha_values=None, timesteps=500):
    """Run simulations for different alpha values"""
    if alpha_values is None:
        alpha_values = [0.05, 0.1, 0.2, 0.3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for one alpha value with all states
    model = SIRSModel(N=N, p=p, beta=beta, gamma=gamma, alpha=alpha_values[0])
    s, i, r = model.run_simulation(timesteps)
    
    ax1.plot(s, label='Susceptible', color='blue')
    ax1.plot(i, label='Infected', color='red')
    ax1.plot(r, label='Recovered', color='green')
    ax1.set_title(f'SIRS Model Evolution (α = {alpha_values[0]})')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Fraction of nodes')
    ax1.legend()
    ax1.grid(True)
    
    # Plot infected fraction for different alpha values
    for alpha in alpha_values:
        model = SIRSModel(N=N, p=p, beta=beta, gamma=gamma, alpha=alpha)
        s, i, r = model.run_simulation(timesteps)
        ax2.plot(i, label=f'α = {alpha}')
    
    ax2.set_title('Infected Population for Different α Values')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Fraction of infected nodes')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run the simulation
if __name__ == "__main__":
    # Parameters
    N = 1000  # Number of nodes
    p = 0.01  # Edge probability
    beta = 0.3  # Infection rate
    gamma = 0.2  # Recovery rate
    alpha_values = [0.05, 0.1, 0.2, 0.3]  # Rates of losing immunity
    
    run_simulations(N=N, p=p, beta=beta, gamma=gamma, alpha_values=alpha_values)


# Model Structure:


# Three states: Susceptible (S), Infected (I), Recovered (R)
# Full cycle possible: S → I → R → S
# New parameter α controls rate of immunity loss (R→S transition)


# Transitions:


# S→I: Same as SIS model (depends on infected neighbors)
# I→R: Recovery with probability γ
# R→S: Loss of immunity with probability α


# Visualization:


# Left plot: Shows all three populations for one α value
# Right plot: Compares infection levels for different α values


# Expected behaviors:


# Higher α leads to more frequent reinfections
# System may show oscillatory behavior
# Eventually reaches a dynamic equilibrium