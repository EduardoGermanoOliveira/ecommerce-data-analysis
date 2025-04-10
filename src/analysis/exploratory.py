import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcommerceAnalyzer:
    """Class for performing exploratory data analysis on e-commerce data."""
    
    def __init__(self, df_chunks):
        """
        Initialize the analyzer with DataFrame chunks.
        
        Args:
            df_chunks: TextFileReader object containing DataFrame chunks
        """
        self.df_chunks = df_chunks
        self._cache = {}
        
    def _process_chunks(self, func):
        """Process all chunks and combine results."""
        results = []
        for chunk in self.df_chunks:
            try:
                # Reset the index to ensure proper concatenation
                chunk = chunk.reset_index(drop=True)
                result = func(chunk)
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error processing chunk: {str(e)}")
                continue
                
        if not results:
            logger.warning("No valid results from any chunk")
            return pd.DataFrame()  # Return empty DataFrame if no valid results
            
        try:
            if isinstance(results[0], pd.DataFrame):
                # Ensure all DataFrames have the same columns
                all_columns = set()
                for df in results:
                    all_columns.update(df.columns)
                
                for df in results:
                    for col in all_columns - set(df.columns):
                        df[col] = 0
                
                return pd.concat(results, axis=0)
            else:
                return sum(results)
        except Exception as e:
            logger.error(f"Error combining results: {str(e)}")
            return pd.DataFrame()
    
    def analyze_event_distribution(self) -> pd.DataFrame:
        """
        Analyze the distribution of different event types.
        
        Returns:
            DataFrame with event type counts and percentages
        """
        def process_chunk(chunk):
            return chunk['event_type'].value_counts()
        
        event_counts = self._process_chunks(process_chunk)
        if event_counts.empty:
            return pd.DataFrame({'count': [], 'percentage': []})
            
        total_events = event_counts.sum()
        event_percentages = (event_counts / total_events * 100).round(2)
        
        return pd.DataFrame({
            'count': event_counts,
            'percentage': event_percentages
        })
    
    def analyze_category_performance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze performance metrics by category.
        
        Args:
            top_n: Number of top categories to return
            
        Returns:
            DataFrame with category performance metrics
        """
        def process_chunk(chunk):
            # Filter purchases first
            purchases = chunk[chunk['event_type'] == 'purchase']
            if purchases.empty:
                logger.debug("No purchase events found in chunk")
                return None
                
            # Use category_code if available, otherwise use category_id
            purchases['category'] = purchases['category_code'].fillna(purchases['category_id'].astype(str))
            
            # Group by category and calculate metrics
            metrics = purchases.groupby('category').agg({
                'event_type': 'count',
                'price': ['mean', 'sum'],
                'user_id': 'nunique',
                'product_id': 'nunique'
            })
            
            # Flatten column names
            metrics.columns = ['total_purchases', 'avg_price', 'total_revenue', 
                             'unique_users', 'unique_products']
            
            return metrics
        
        category_metrics = self._process_chunks(process_chunk)
        if category_metrics.empty:
            logger.warning("No category metrics available - check if there are any purchase events")
            return pd.DataFrame()
            
        # Calculate additional metrics
        category_metrics['conversion_rate'] = (
            category_metrics['total_purchases'] / category_metrics['unique_users']
        ).round(3)
        
        category_metrics['products_per_user'] = (
            category_metrics['unique_products'] / category_metrics['unique_users']
        ).round(2)
        
        return category_metrics.nlargest(top_n, 'total_revenue')
    
    def analyze_brand_performance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze performance metrics by brand.
        
        Args:
            top_n: Number of top brands to return
            
        Returns:
            DataFrame with brand performance metrics
        """
        def process_chunk(chunk):
            purchases = chunk[chunk['event_type'] == 'purchase']
            if purchases.empty:
                logger.debug("No purchase events found in chunk")
                return None
                
            # Fill NaN brands with 'unknown'
            purchases['brand'] = purchases['brand'].fillna('unknown')
            
            # Group by brand and calculate metrics
            metrics = purchases.groupby('brand').agg({
                'event_type': 'count',
                'price': ['mean', 'sum'],
                'user_id': 'nunique',
                'product_id': 'nunique'
            })
            
            # Flatten column names
            metrics.columns = ['total_purchases', 'avg_price', 'total_revenue',
                             'unique_users', 'unique_products']
            
            return metrics
        
        brand_metrics = self._process_chunks(process_chunk)
        if brand_metrics.empty:
            logger.warning("No brand metrics available - check if there are any purchase events")
            return pd.DataFrame()
            
        brand_metrics['conversion_rate'] = (
            brand_metrics['total_purchases'] / brand_metrics['unique_users']
        ).round(3)
        
        return brand_metrics.nlargest(top_n, 'total_revenue')
    
    def analyze_user_behavior(self) -> Dict[str, float]:
        """
        Analyze user behavior metrics.
        
        Returns:
            Dictionary with user behavior metrics
        """
        def process_chunk(chunk):
            # Add debug logging
            event_types = chunk['event_type'].value_counts()
            logger.debug(f"Event types in chunk: {event_types.to_dict()}")
            
            user_metrics = chunk.groupby('user_id').agg({
                'event_type': lambda x: x.value_counts().to_dict(),
                'price': 'sum',
                'product_id': 'nunique',
                'user_session': 'nunique'
            })
            return user_metrics
        
        user_metrics = self._process_chunks(process_chunk)
        if user_metrics.empty:
            logger.warning("No user metrics available - check if there are any events")
            return {}
            
        # Calculate average metrics per user
        avg_metrics = {
            'avg_views_per_user': user_metrics['event_type'].apply(lambda x: x.get('view', 0)).mean(),
            'avg_carts_per_user': user_metrics['event_type'].apply(lambda x: x.get('cart', 0)).mean(),
            'avg_purchases_per_user': user_metrics['event_type'].apply(lambda x: x.get('purchase', 0)).mean(),
            'avg_spent_per_user': user_metrics['price'].mean(),
            'avg_products_viewed': user_metrics['product_id'].mean(),
            'avg_sessions_per_user': user_metrics['user_session'].mean()
        }
        
        # Log the metrics for debugging
        logger.debug(f"Average metrics per user: {avg_metrics}")
        
        return {k: round(v, 2) for k, v in avg_metrics.items()}
    
    def analyze_time_patterns(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze temporal patterns in user behavior.
        
        Returns:
            Tuple of DataFrames with hourly and daily patterns
        """
        def process_chunk(chunk):
            chunk['hour'] = pd.to_datetime(chunk['event_time']).dt.hour
            chunk['day_of_week'] = pd.to_datetime(chunk['event_time']).dt.dayofweek
            hourly = chunk.groupby('hour')['event_type'].value_counts().unstack(fill_value=0)
            daily = chunk.groupby('day_of_week')['event_type'].value_counts().unstack(fill_value=0)
            return pd.DataFrame({'hourly': hourly, 'daily': daily})
        
        patterns = self._process_chunks(process_chunk)
        if patterns.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        return patterns['hourly'], patterns['daily']
    
    def plot_event_distribution(self, save_path: Optional[str] = None):
        """Plot event type distribution."""
        event_dist = self.analyze_event_distribution()
        if event_dist.empty:
            logger.warning("No event distribution data to plot")
            return
            
        plt.figure(figsize=(10, 6))
        sns.barplot(data=event_dist.reset_index(), x='event_type', y='percentage')
        plt.title('Event Type Distribution')
        plt.xlabel('Event Type')
        plt.ylabel('Percentage')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_category_performance(self, metric: str = 'total_revenue', 
                                save_path: Optional[str] = None):
        """Plot category performance for a specific metric."""
        category_metrics = self.analyze_category_performance()
        if category_metrics.empty:
            logger.warning("No category performance data to plot")
            return
            
        plt.figure(figsize=(12, 6))
        sns.barplot(data=category_metrics.reset_index(), 
                   x='category', y=metric)
        plt.title(f'Top Categories by {metric}')
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_brand_performance(self, metric: str = 'total_revenue',
                             save_path: Optional[str] = None):
        """Plot brand performance for a specific metric."""
        brand_metrics = self.analyze_brand_performance()
        if brand_metrics.empty:
            logger.warning("No brand performance data to plot")
            return
            
        plt.figure(figsize=(12, 6))
        sns.barplot(data=brand_metrics.reset_index(), 
                   x='brand', y=metric)
        plt.title(f'Top Brands by {metric}')
        plt.xticks(rotation=45, ha='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 