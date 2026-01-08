import networkx as nx
import numpy as np
from collections import defaultdict
import logging
import time

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class LouvainCommunityDetection:
    def __init__(self, G):
        logging.info("Initializing LouvainCommunityDetection")
        self.G = G.copy()
        self.N = G.number_of_nodes()
        self.m = G.number_of_edges()
        # Each node starts in its own community
        self.node_to_community = {node: i for i, node in enumerate(G.nodes())}
        # Keep track of nodes in each community
        self.community_to_nodes = {i: [node] for i, node in enumerate(G.nodes())}
        # Store degrees
        self.node_degrees = dict(G.degree(weight='weight'))
        logging.info(f"Network has {self.N} nodes and {self.m} edges")

    def get_node_community(self, node):
        return self.node_to_community[node]

    def get_community_nodes(self, community):
        return self.community_to_nodes[community]

    def compute_modularity_gain(self, node, to_community):
        """Compute modularity gain for moving node to community"""
        from_community = self.get_node_community(node)
        if from_community == to_community:
            return 0.0

        # Calculate weights to the target community
        weight_to_comm = sum(wt.get('weight', 1.0)
                        for nbr, wt in self.G[node].items()
                        if self.get_node_community(nbr) == to_community)

        # Calculate total weight of target community
        comm_degree = sum(self.node_degrees[n] 
                         for n in self.get_community_nodes(to_community))

        # Node's degree
        node_degree = self.node_degrees[node]

        # Calculate modularity gain
        gain = (weight_to_comm - (node_degree * comm_degree) / (2 * self.m))
        return gain / self.m

    def first_phase(self):
        """Optimize modularity by moving individual nodes"""
        start_time = time.time()
        logging.info("Starting first phase...")
        
        total_moves = 0
        nodes = list(self.G.nodes())
        
        for iteration in range(100):  # Max 100 iterations
            moves_this_iteration = 0
            np.random.shuffle(nodes)
            
            for node in nodes:
                # Get current community
                current_comm = self.get_node_community(node)
                best_comm = current_comm
                best_gain = 0.0
                
                # Get neighboring communities
                neighbor_communities = set()
                for neighbor in self.G.neighbors(node):
                    neighbor_communities.add(self.get_node_community(neighbor))
                
                # Find best community
                for comm in neighbor_communities:
                    if comm != current_comm:
                        gain = self.compute_modularity_gain(node, comm)
                        if gain > best_gain:
                            best_gain = gain
                            best_comm = comm
                
                # Move to best community if there's improvement
                if best_comm != current_comm:
                    # Update mappings
                    self.community_to_nodes[current_comm].remove(node)
                    self.community_to_nodes[best_comm].append(node)
                    self.node_to_community[node] = best_comm
                    moves_this_iteration += 1
                    total_moves += 1
            
            logging.info(f"Iteration {iteration + 1}: made {moves_this_iteration} moves")
            
            if moves_this_iteration == 0:
                break
        
        logging.info(f"First phase completed in {time.time() - start_time:.2f}s "
                    f"with {total_moves} total moves")
        return total_moves > 0

    def second_phase(self):
        """Aggregate communities into super-nodes"""
        start_time = time.time()
        logging.info("Starting second phase...")
        
        # Create new graph where nodes are communities
        new_G = nx.Graph()
        communities = set(self.node_to_community.values())
        
        # Add nodes
        for comm in communities:
            new_G.add_node(comm)
        
        # Calculate edges between communities
        edge_weights = defaultdict(float)
        for (node1, node2, weight) in self.G.edges(data='weight', default=1.0):
            comm1 = self.get_node_community(node1)
            comm2 = self.get_node_community(node2)
            edge_weights[(comm1, comm2)] += weight
        
        # Add all edges to the new graph
        for (comm1, comm2), weight in edge_weights.items():
            if comm1 != comm2:  # Skip self-loops
                new_G.add_edge(comm1, comm2, weight=weight)
        
        # Update instance variables
        self.G = new_G
        old_mapping = self.node_to_community.copy()
        
        # Reset community mappings for the new graph
        self.node_to_community = {node: node for node in new_G.nodes()}
        self.community_to_nodes = {node: [node] for node in new_G.nodes()}
        self.node_degrees = dict(new_G.degree(weight='weight'))
        
        logging.info(f"Second phase completed in {time.time() - start_time:.2f}s")
        logging.info(f"New network: {new_G.number_of_nodes()} nodes, "
                    f"{new_G.number_of_edges()} edges")
        return old_mapping

    def run(self):
        """Run the complete Louvain algorithm"""
        start_time = time.time()
        logging.info("Starting Louvain algorithm...")
        
        # Keep track of community mapping at each level
        mappings = []
        
        while True:
            improved = self.first_phase()
            if not improved:
                break
                
            mapping = self.second_phase()
            mappings.append(mapping)
            
            if self.G.number_of_nodes() < 3:
                break
        
        # Construct final community assignment
        final_communities = self.node_to_community.copy()
        for mapping in reversed(mappings):
            final_communities = {node: mapping.get(community, community)
                               for node, community in final_communities.items()}
        
        logging.info(f"Algorithm completed in {time.time() - start_time:.2f}s")
        return final_communities

def detect_communities(G):
    """Wrapper function to detect communities in graph G"""
    detector = LouvainCommunityDetection(G)
    return detector.run()

if __name__ == "__main__":
    # Create test graph
    G = nx.karate_club_graph()
    
    # Detect communities
    logging.info("Starting community detection.")
    communities = detect_communities(G)
    
    # Print results
    n_communities = len(set(communities.values()))
    logging.info(f"Found {n_communities} communities")
    
    # Print community sizes
    community_sizes = defaultdict(int)
    for comm in communities.values():
        community_sizes[comm] += 1
    
    print("\nCommunity sizes:")
    for comm, size in sorted(community_sizes.items()):
        print(f"Community {comm}: {size} nodes")
        
    print("\nNode assignments:")
    for node, comm in sorted(communities.items()):
        print(f"Node {node} -> Community {comm}")


# 1. Purpose
# - The algorithm aims to find communities in networks by optimizing modularity
# - Modularity measures the density of links inside communities compared to links between communities
# - Higher modularity means better community structure

# 2. Initial Setup
# - Start with each node in its own community
# - Track the degree (number of connections) of each node
# - Consider the weight of edges if the network is weighted

# 3. The Algorithm Works in Two Phases that Repeat:

#    Phase 1: Local Optimization
#    - Consider each node individually
#    - Look at its neighboring communities (communities of connected nodes)
#    - Calculate the modularity gain for moving the node to each neighboring community
#    - Move the node to the community that gives the highest positive gain
#    - Repeat until no moves improve modularity

#    Phase 2: Network Aggregation
#    - Take communities found in Phase 1
#    - Create a new network where:
#      * Each community becomes a single node
#      * Edges between communities become weighted edges
#      * The weights represent total connections between communities
#    - Self-loops represent connections within communities

# 4. Iteration
# - Repeat Phases 1 and 2 until no further improvement is possible
# - Each iteration creates a hierarchical level of communities
# - The algorithm stops when:
#   * No nodes can be moved to improve modularity
#   * Or the network has been reduced to just a few nodes