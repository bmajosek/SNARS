import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import logging
from ast import literal_eval

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Read the CSV files
logger.info("Reading CSV files...")
tweets_df = pd.read_csv('tweets.csv')
users_df = pd.read_csv('users.csv')

# Create a directed graph
logger.info("Creating directed graph...")
G = nx.DiGraph()

# Add user nodes
logger.info("Adding user nodes...")
for _, user in users_df.iterrows():
    G.add_node(user['screen_name'], 
               node_type='user',
               followers=user['followers_count'],
               friends=user['friends_count'],
               verified=user['verified'])

# Process tweets and add edges
logger.info("Processing tweets and adding edges...")
for _, tweet in tweets_df.iterrows():
    # Add tweet node
    G.add_node(tweet['tweet_id'], 
               node_type='tweet',
               text=tweet['text'])
    
    # Connect user to their tweet
    G.add_edge(tweet['user_key'], tweet['tweet_id'], edge_type='authored')
    
    # Process mentions (stored as string representation of list)
    try:
        mentions = literal_eval(tweet['mentions'])
        for mention in mentions:
            G.add_edge(tweet['tweet_id'], mention, edge_type='mentions')
    except:
        logger.warning(f"Failed to process mentions for tweet {tweet['tweet_id']}")
    
    # Process retweets
    if tweet['retweeted_status_id']:
        G.add_edge(tweet['user_key'], tweet['retweeted_status_id'], edge_type='retweeted')
    
    # Process replies
    if tweet['in_reply_to_status_id']:
        G.add_edge(tweet['tweet_id'], tweet['in_reply_to_status_id'], edge_type='reply_to')

# Create visualization
logger.info("Creating visualization...")
logger.info("Setting up figure with dimensions 15x10...")
plt.figure(figsize=(15, 10))

# Select a sample of active users and their network
logger.info("Sampling users and their network...")
# Get most active users (top 10)
user_activity = {}
for edge in G.edges(data=True):
    if edge[2]['edge_type'] == 'authored':
        user = edge[0]
        user_activity[user] = user_activity.get(user, 0) + 1

selected_users = [user for user, _ in sorted(user_activity.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:10]]

# Create a subgraph with selected users and their immediate connections
nodes_to_keep = set(selected_users)
logger.info(f"Selected {len(selected_users)} most active users")

# Add tweets by selected users
tweet_count = 0
for edge in G.edges(data=True):
    if edge[2]['edge_type'] == 'authored' and edge[0] in selected_users:
        nodes_to_keep.add(edge[1])  # Add the tweet
        tweet_count += 1
    # Add mentioned users and their tweets
    elif edge[2]['edge_type'] == 'mentions' and edge[0] in nodes_to_keep:
        nodes_to_keep.add(edge[1])

G_viz = G.subgraph(nodes_to_keep)
logger.info(f"Reduced network size: {len(G_viz)} nodes ({len(selected_users)} users, {tweet_count} tweets)")

# Calculate layout for the smaller network
logger.info("Calculating spring layout with k=2 and 25 iterations...")
logger.info("This may take a few minutes for large networks...")
pos = nx.spring_layout(G_viz, 
                      k=2,
                      iterations=25,  # Reduced iterations
                      seed=42)
logger.info("Layout calculation complete")

# Drawing the network
logger.info("Drawing network elements...")
logger.info("Identifying user and tweet nodes...")

# Get user and tweet nodes from the visualization graph
user_nodes = [n for n, attr in G_viz.nodes(data=True) if attr.get('node_type') == 'user']
tweet_nodes = [n for n, attr in G_viz.nodes(data=True) if attr.get('node_type') == 'tweet']

# Draw different types of edges with different colors
logger.info("Drawing edges with color coding...")
edge_colors = {'authored': 'green',
               'mentions': 'blue',
               'retweeted': 'red',
               'reply_to': 'orange'}

for edge_type, color in edge_colors.items():
    edge_list = [(u, v) for (u, v, d) in G_viz.edges(data=True) if d.get('edge_type') == edge_type]
    nx.draw_networkx_edges(G_viz, pos, edgelist=edge_list, edge_color=color, alpha=0.5)

# Draw nodes with different colors and sizes
logger.info("Drawing nodes with size based on follower count...")
nx.draw_networkx_nodes(G_viz, pos, nodelist=user_nodes, 
                      node_color='lightblue',
                      node_size=[G.nodes[node].get('followers', 100) + 100 for node in user_nodes],
                      alpha=0.7)
nx.draw_networkx_nodes(G_viz, pos, nodelist=tweet_nodes,
                      node_color='lightgreen',
                      node_size=50,
                      alpha=0.5)

# Add labels to user nodes only (to avoid overcrowding)
logger.info("Adding labels to user nodes...")
labels = {node: node for node in user_nodes}
nx.draw_networkx_labels(G_viz, pos, labels, font_size=8)

# Add legend
logger.info("Adding legend and finalizing plot...")
legend_elements = [plt.Line2D([0], [0], color=c, label=t) 
                  for t, c in edge_colors.items()]
legend_elements.extend([
    plt.scatter([0], [0], c='lightblue', label='Users', alpha=0.7),
    plt.scatter([0], [0], c='lightgreen', label='Tweets', alpha=0.5)
])
plt.legend(handles=legend_elements)

# Remove axes
plt.axis('off')

# Add title
plt.title('Twitter Network: Users, Tweets, and Their Relationships')

# Save the plot
logger.info("Saving visualization to twitter_network.png...")
plt.savefig('twitter_network.png', dpi=300, bbox_inches='tight')

# Print some network statistics
logger.info("Calculating network statistics...")
print("\nNetwork Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Number of users: {len(user_nodes)}")
print(f"Number of tweets: {len(tweet_nodes)}")

# Analyze user activity
logger.info("Analyzing user activity...")
print("\nMost active users (by number of tweets):")
user_activity = {}
for edge in G.edges(data=True):
    if edge[2]['edge_type'] == 'authored':
        user = edge[0]
        user_activity[user] = user_activity.get(user, 0) + 1
        
for user, count in sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{user}: {count} tweets")

# Analyze mentions and interactions
logger.info("Analyzing mentions and interactions...")
print("\nMost mentioned users:")
mention_counts = {}
for edge in G.edges(data=True):
    if edge[2]['edge_type'] == 'mentions':
        mentioned = edge[1]
        mention_counts[mentioned] = mention_counts.get(mentioned, 0) + 1
        
for user, count in sorted(mention_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{user}: mentioned {count} times")

logger.info("Analysis complete!")