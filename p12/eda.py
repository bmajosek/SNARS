import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
from ast import literal_eval
import numpy as np

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

class TwitterAnalyzer:
    def __init__(self, tweets_path, users_path):
        """Initialize with paths to data files"""
        logger.info("Loading data files...")
        self.tweets_df = pd.read_csv(tweets_path)
        self.users_df = pd.read_csv(users_path)
        self.user_stats = None
        
        # Convert relevant columns to appropriate types
        self.users_df['verified'] = self.users_df['verified'].map({'true': True, 'false': False})
        
        # Initialize counts to 0 where NaN
        numeric_columns = ['followers_count', 'friends_count', 'statuses_count', 'favourites_count']
        for col in numeric_columns:
            self.users_df[col] = self.users_df[col].fillna(0).astype(int)
            
    def safe_eval_list(self, list_str):
        """Safely evaluate string representations of lists"""
        try:
            if pd.isna(list_str) or list_str == '':
                return []
            return literal_eval(list_str)
        except:
            logger.warning(f"Failed to parse list string: {list_str}")
            return []

    def calculate_user_statistics(self):
        """Calculate comprehensive user statistics"""
        logger.info("Calculating user statistics...")
        
        # Initialize user statistics dictionary with default values
        user_stats = defaultdict(lambda: {
            'tweet_count': 0,          # tweets in our dataset
            'retweet_count': 0,        # retweets made
            'reply_count': 0,          # replies made
            'mentions_made': 0,        # mentions of other users
            'mentions_received': 0,     # times mentioned by others
            'followers': 0,            # follower count
            'friends': 0,              # following count
            'verified': False,         # verification status
            'total_statuses': 0,       # total tweets from profile
            'favorites': 0             # favorites count
        })
        
        # Add base user info from users_df
        for _, user in self.users_df.iterrows():
            screen_name = user['screen_name']
            user_stats[screen_name].update({
                'followers': int(user['followers_count']),
                'friends': int(user['friends_count']),
                'verified': bool(user['verified']),
                'total_statuses': int(user['statuses_count']),
                'favorites': int(user['favourites_count'])
            })
        
        # Process tweets
        for _, tweet in self.tweets_df.iterrows():
            user = tweet['user_key']
            
            # Count basic tweet types
            user_stats[user]['tweet_count'] += 1
            
            # Count retweets
            if pd.notna(tweet['retweeted_status_id']) and tweet['retweeted_status_id'] != '':
                user_stats[user]['retweet_count'] += 1
            
            # Count replies
            if pd.notna(tweet['in_reply_to_status_id']) and tweet['in_reply_to_status_id'] != '':
                user_stats[user]['reply_count'] += 1
            
            # Process mentions
            mentions = self.safe_eval_list(tweet['mentions'])
            user_stats[user]['mentions_made'] += len(mentions)
            for mentioned in mentions:
                if mentioned in user_stats:
                    user_stats[mentioned]['mentions_received'] += 1
        
        self.user_stats = pd.DataFrame.from_dict(user_stats, orient='index')
        return self.user_stats
    
    def plot_user_activity_distribution(self):
        """Plot distribution of user activity metrics"""
        if self.user_stats is None:
            self.calculate_user_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('User Activity Distributions', fontsize=16, y=1.02)
        
        # Total statuses distribution (from profile)
        non_zero_statuses = self.user_stats['total_statuses'][self.user_stats['total_statuses'] > 0]
        
        # Use log-spaced bins
        min_val = max(1, non_zero_statuses.min())  # Avoid log(0)
        max_val = non_zero_statuses.max()
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 30)
        
        # Create histogram with log-spaced bins
        axes[0,0].hist(non_zero_statuses, bins=bins, alpha=0.7)
        axes[0,0].set_xscale('log')  # Use log scale for x-axis too
        axes[0,0].set_title('Total Tweets Distribution (from profile)')
        axes[0,0].set_xlabel('Number of Total Tweets')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_yscale('log')  # Only y-axis in log scale
        
        # Add some stats to the plot
        stats_text = f'Mean: {non_zero_statuses.mean():.0f}\nMedian: {non_zero_statuses.median():.0f}'
        axes[0,0].text(0.95, 0.95, stats_text,
                      transform=axes[0,0].transAxes,
                      verticalalignment='top',
                      horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Mentions distribution
        non_zero_mentions = self.user_stats['mentions_received'][self.user_stats['mentions_received'] > 0]
        
        # Calculate better bins for mentions using percentile-based edges
        mention_percentiles = np.linspace(0, 100, 15)  # Create 15 bins based on percentiles
        mention_bins = np.percentile(non_zero_mentions, mention_percentiles)
        
        # Create histogram with custom bins
        axes[0,1].hist(non_zero_mentions, bins=mention_bins, alpha=0.7)
        axes[0,1].set_yscale('log')  # Only y-axis in log scale
        axes[0,1].set_title('Mentions Received Distribution')
        axes[0,1].set_xlabel('Number of Mentions')
        axes[0,1].set_ylabel('Count (log scale)')
        
        # Followers vs Friends scatter plot
        axes[1,0].scatter(self.user_stats['followers'], 
                         self.user_stats['friends'],
                         alpha=0.5)
        axes[1,0].set_title('Followers vs Friends')
        axes[1,0].set_xlabel('Followers')
        axes[1,0].set_ylabel('Friends')
        # Add log scales for better visualization
        axes[1,0].set_xscale('log')
        axes[1,0].set_yscale('log')
        
        # Activity breakdown for top users
        # Select top 10 users by tweet_count
        top_users = self.user_stats.nlargest(10, 'tweet_count')
        
        # Create the stacked bar chart
        activity_data = pd.DataFrame({
            'Original Tweets': top_users['tweet_count'] - top_users['retweet_count'] - top_users['reply_count'],
            'Retweets': top_users['retweet_count'],
            'Replies': top_users['reply_count']
        })
        
        activity_data.plot(kind='bar', 
                         stacked=True, 
                         ax=axes[1,1],
                         colormap='viridis')
        
        axes[1,1].set_title('Activity Breakdown for Top 10 Users')
        axes[1,1].set_xlabel('Users')
        axes[1,1].set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('user_activity_analysis.png', 
                   dpi=300, 
                   bbox_inches='tight')
        plt.close()
        
        logger.info("Saved activity distribution plots")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the dataset"""
        if self.user_stats is None:
            self.calculate_user_statistics()
            
        summary = {
            'total_users': len(self.user_stats),
            'total_tweets_in_sample': self.user_stats['tweet_count'].sum(),
            'total_retweets': self.user_stats['retweet_count'].sum(),
            'total_replies': self.user_stats['reply_count'].sum(),
            'total_mentions': self.user_stats['mentions_made'].sum(),
            'verified_users': self.user_stats['verified'].sum(),
            'avg_followers': self.user_stats['followers'].mean(),
            'median_followers': self.user_stats['followers'].median(),
            'avg_tweets_per_user': self.user_stats['tweet_count'].mean(),
            'median_tweets_per_user': self.user_stats['tweet_count'].median()
        }
        
        return pd.Series(summary)

# Usage example
if __name__ == "__main__":
    analyzer = TwitterAnalyzer('tweets.csv', 'users.csv')
    
    # Calculate user statistics
    stats = analyzer.calculate_user_statistics()
    print("\nTop 10 users by tweet count:")
    print(stats.nlargest(10, 'tweet_count')[['tweet_count', 'mentions_received', 'followers']])
    
    # Generate summary statistics
    summary = analyzer.generate_summary_statistics()
    print("\nDataset Summary:")
    print(summary)
    
    # Generate visualizations
    analyzer.plot_user_activity_distribution()
    
    # Save detailed statistics to CSV
    stats.to_csv('user_statistics.csv')