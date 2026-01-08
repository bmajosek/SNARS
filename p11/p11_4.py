import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class EnhancedVoterModel:
    def __init__(self, N, k, propaganda_strength=0.2, ideology_strength=0.3):
        """
        Initialize enhanced voter model
        
        Parameters:
        N: int - number of nodes
        k: int - average degree for random graph
        propaganda_strength: float - strength of external influence (0 to 1)
        ideology_strength: float - resistance to change (0 to 1)
        """
        self.N = N
        self.propaganda_strength = propaganda_strength
        
        # Create random graph
        p = k / (N - 1)  # probability for desired average degree
        self.G = nx.erdos_renyi_graph(N, p)
        
        # Initialize opinions (-1 or 1)
        self.opinions = np.random.choice([-1, 1], size=N)
        
        # Initialize individual ideology strengths
        self.ideology = np.random.beta(2, 5, size=N) * ideology_strength
        
        # Track opinion distribution over time
        self.opinion_history = []
        
    def apply_propaganda(self, opinion):
        """Apply propaganda effect (biased towards +1)"""
        if np.random.random() < self.propaganda_strength:
            return 1
        return opinion
    
    def step(self):
        """Simulate one time step of the voter model"""
        new_opinions = self.opinions.copy()
        
        # Randomly select a node to update
        node = np.random.randint(0, self.N)
        
        # Get neighbors' opinions
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return
            
        # Calculate neighbor influence
        neighbor_opinion = np.random.choice([self.opinions[n] for n in neighbors])
        
        # Apply ideology resistance
        if np.random.random() > self.ideology[node]:
            # If passes ideology check, consider changing opinion
            
            # Apply propaganda effect
            neighbor_opinion = self.apply_propaganda(neighbor_opinion)
            
            # Update opinion
            new_opinions[node] = neighbor_opinion
            
        self.opinions = new_opinions
        
        # Track the fraction of +1 opinions
        positive_fraction = np.mean(self.opinions == 1)
        self.opinion_history.append(positive_fraction)
        
    def run_simulation(self, timesteps):
        """Run simulation for specified number of timesteps"""
        self.opinion_history = []
        for _ in tqdm(range(timesteps)):
            self.step()
        return np.array(self.opinion_history)

def run_comparison(N=1000, k=10, timesteps=10000):
    """Compare different propaganda and ideology settings"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compare propaganda strengths
    propaganda_values = [0.0, 0.1, 0.3, 0.5]
    for prop in propaganda_values:
        model = EnhancedVoterModel(N=N, k=k, propaganda_strength=prop, 
                                 ideology_strength=0.2)
        history = model.run_simulation(timesteps)
        ax1.plot(history, label=f'Propaganda = {prop}')
    
    ax1.set_title('Effect of Propaganda Strength')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Fraction of positive opinions')
    ax1.legend()
    ax1.grid(True)
    
    # Compare ideology strengths
    ideology_values = [0.0, 0.2, 0.4, 0.6]
    for ideo in ideology_values:
        model = EnhancedVoterModel(N=N, k=k, propaganda_strength=0.2, 
                                 ideology_strength=ideo)
        history = model.run_simulation(timesteps)
        ax2.plot(history, label=f'Ideology = {ideo}')
    
    ax2.set_title('Effect of Ideology Strength')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Fraction of positive opinions')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_network_state(model, ax=None):
    """Visualize current state of the network"""
    if ax is None:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
    
    # Use stronger repulsion and more iterations for better spacing
    pos = nx.spring_layout(model.G, k=1/np.sqrt(model.N), iterations=50)
    
    # Scale node sizes by degree
    degrees = dict(model.G.degree())
    node_sizes = [20 + 10*degrees[node] for node in model.G.nodes()]
    
    # Draw edges with transparency
    nx.draw_networkx_edges(model.G, pos, alpha=0.2, ax=ax)
    
    # Draw nodes
    colors = ['red' if opinion == -1 else 'blue' for opinion in model.opinions]
    nx.draw_networkx_nodes(model.G, pos, node_color=colors, 
                          node_size=node_sizes, ax=ax)
    
    ax.set_title('Network State: Node size indicates number of connections\nRed = -1, Blue = +1')

# Run the simulation
if __name__ == "__main__":
    # Parameters
    N = 500  # Number of nodes (reduced for faster visualization)
    k = 4    # Average degree
    
    # Run comparison simulations
    run_comparison(N=N, k=k)
    
    # Visualize a single network state
    model = EnhancedVoterModel(N=N, k=k)
    model.run_simulation(1000)
    plt.figure(figsize=(8, 8))
    plot_network_state(model)
    plt.show()
    
# Visualization Algorithm:

# We're using NetworkX's spring_layout, which uses the Fruchterman-Reingold force-directed algorithm
# It positions nodes by simulating physical forces:

# Connected nodes attract each other
# All nodes repel each other
# The system tries to find an equilibrium

# The Dense Center Pattern:

# Nodes with more connections (higher degree) tend to be pulled toward the center
# Nodes with fewer connections can end up at the periphery
# The density in the middle suggests many nodes are well-connected to each other