import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet
from typing import Tuple, Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandForecaster:
    """Class for forecasting demand using different models."""
    
    def __init__(self, df_chunks):
        """
        Initialize the forecaster with preprocessed data chunks.
        
        Args:
            df_chunks: TextFileReader object containing DataFrame chunks
        """
        self.df_chunks = df_chunks
        self.models = {}
        self._cache = {}
        self._chunks = []  # Store chunks in memory
        
    def _load_chunks(self):
        """Load all chunks into memory if not already loaded."""
        if not self._chunks:
            logger.info("Loading chunks into memory...")
            for chunk in self.df_chunks:
                try:
                    if chunk is not None and not chunk.empty:
                        self._chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Error loading chunk: {str(e)}")
                    continue
            logger.info(f"Loaded {len(self._chunks)} chunks into memory")
        
    def _process_chunks(self, func):
        """Process all chunks and combine results."""
        self._load_chunks()  # Ensure chunks are loaded
        results = []
        
        for chunk in self._chunks:
            try:
                # Process the chunk with the provided function
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
    
    def _get_full_dataframe(self):
        """
        Process all chunks and return a complete DataFrame.
        This is useful for analyses that need the complete dataset.
        
        Returns:
            Complete DataFrame with all data
        """
        if 'full_df' in self._cache:
            return self._cache['full_df']
            
        self._load_chunks()  # Ensure chunks are loaded
        
        if not self._chunks:
            logger.warning("No valid chunks found")
            return pd.DataFrame()
            
        try:
            full_df = pd.concat(self._chunks, ignore_index=True)
            self._cache['full_df'] = full_df
            logger.info(f"Complete DataFrame created with shape {full_df.shape}")
            return full_df
        except Exception as e:
            logger.error(f"Error creating complete DataFrame: {str(e)}")
            return pd.DataFrame()
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for forecasting by aggregating daily metrics.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Try using the full DataFrame for this analysis
        full_df = self._get_full_dataframe()
        if not full_df.empty:
            try:
                # Convert event_time to datetime
                full_df['event_time'] = pd.to_datetime(full_df['event_time'])
                full_df['date'] = full_df['event_time'].dt.date
                
                # Calculate daily metrics
                daily_metrics = full_df.groupby('date').agg({
                    'event_type': lambda x: x.value_counts().to_dict(),
                    'price': ['sum', 'mean', 'count'],
                    'user_id': 'nunique',
                    'product_id': 'nunique'
                })
                
                # Flatten column names
                daily_metrics.columns = ['event_counts', 'total_revenue', 'avg_price', 
                                       'total_events', 'unique_users', 'unique_products']
                
                # Extract event counts into separate columns
                event_counts = daily_metrics['event_counts'].apply(pd.Series)
                event_counts = event_counts.fillna(0)
                
                # Create feature matrix
                X = pd.concat([
                    event_counts,
                    daily_metrics[['total_revenue', 'avg_price', 'total_events', 
                               'unique_users', 'unique_products']]
                ], axis=1)
                
                # Target variable: next day's total revenue
                y = daily_metrics['total_revenue'].shift(-1)
                
                # Remove last row (no target value)
                X = X[:-1]
                y = y[:-1]
                
                # Add time-based features
                X['day_of_week'] = pd.to_datetime(daily_metrics.index).dayofweek[:-1]
                X['day_of_month'] = pd.to_datetime(daily_metrics.index).day[:-1]
                
                return X, y
            except Exception as e:
                logger.warning(f"Error preparing data with full DataFrame: {str(e)}")
        
        # Fall back to chunk processing if full DataFrame approach fails
        def process_chunk(chunk):
            try:
                # Convert event_time to datetime
                chunk['event_time'] = pd.to_datetime(chunk['event_time'])
                chunk['date'] = chunk['event_time'].dt.date
                
                # Calculate daily metrics
                daily_metrics = chunk.groupby('date').agg({
                    'event_type': lambda x: x.value_counts().to_dict(),
                    'price': ['sum', 'mean', 'count'],
                    'user_id': 'nunique',
                    'product_id': 'nunique'
                })
                
                # Flatten column names
                daily_metrics.columns = ['event_counts', 'total_revenue', 'avg_price', 
                                       'total_events', 'unique_users', 'unique_products']
                
                return daily_metrics
            except Exception as e:
                logger.warning(f"Error processing chunk in prepare_data: {str(e)}")
                return None
        
        daily_data = self._process_chunks(process_chunk)
        if daily_data.empty:
            logger.error("No valid data available for forecasting")
            return pd.DataFrame(), pd.Series()
            
        try:
            # Extract event counts into separate columns
            event_counts = daily_data['event_counts'].apply(pd.Series)
            event_counts = event_counts.fillna(0)
            
            # Create feature matrix
            X = pd.concat([
                event_counts,
                daily_data[['total_revenue', 'avg_price', 'total_events', 
                           'unique_users', 'unique_products']]
            ], axis=1)
            
            # Target variable: next day's total revenue
            y = daily_data['total_revenue'].shift(-1)
            
            # Remove last row (no target value)
            X = X[:-1]
            y = y[:-1]
            
            # Add time-based features
            X['day_of_week'] = pd.to_datetime(daily_data.index).dayofweek[:-1]
            X['day_of_month'] = pd.to_datetime(daily_data.index).day[:-1]
            
            return X, y
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train a Random Forest model for demand forecasting.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with model metrics
        """
        if X.empty or y.empty:
            logger.error("No data available for training Random Forest model")
            return {'metrics': {}}
            
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.rf_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.rf_model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Store model and feature importance
            self.models['random_forest'] = {
                'model': self.rf_model,
                'feature_importance': dict(zip(X.columns, self.rf_model.feature_importances_))
            }
            
            return {'metrics': metrics}
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return {'metrics': {}}
    
    def train_prophet(self, target_col: str = 'daily_purchases',
                     forecast_periods: int = 30) -> Dict[str, Any]:
        """
        Train a Prophet model for demand forecasting.
        
        Args:
            target_col: Column to forecast
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with model and forecast
        """
        # Try using the full DataFrame for this analysis
        full_df = self._get_full_dataframe()
        if not full_df.empty:
            try:
                full_df['event_time'] = pd.to_datetime(full_df['event_time'])
                daily_data = full_df[full_df['event_type'] == 'purchase'].groupby(
                    full_df['event_time'].dt.date
                )[target_col].sum()
                
                # Prepare data for Prophet
                prophet_data = daily_data.reset_index()
                prophet_data.columns = ['ds', 'y']
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True
                )
                
                model.fit(prophet_data)
                
                future = model.make_future_dataframe(periods=forecast_periods)
                forecast = model.predict(future)
                
                self.models['prophet'] = {
                    'model': model,
                    'forecast': forecast
                }
                
                return self.models['prophet']
            except Exception as e:
                logger.warning(f"Error training Prophet model with full DataFrame: {str(e)}")
        
        # Fall back to chunk processing if full DataFrame approach fails
        def process_chunk(chunk):
            chunk['event_time'] = pd.to_datetime(chunk['event_time'])
            return chunk[chunk['event_type'] == 'purchase'].groupby(
                chunk['event_time'].dt.date
            )[target_col].sum()
        
        # Process all chunks and combine
        daily_data = self._process_chunks(process_chunk)
        
        # Prepare data for Prophet
        prophet_data = daily_data.reset_index()
        prophet_data.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        
        model.fit(prophet_data)
        
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        
        self.models['prophet'] = {
            'model': model,
            'forecast': forecast
        }
        
        return self.models['prophet']
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the Random Forest model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if 'random_forest' not in self.models:
            raise ValueError("Random Forest model not trained yet")
            
        importance_df = pd.DataFrame({
            'feature': self.models['random_forest']['feature_importance'].keys(),
            'importance': self.models['random_forest']['feature_importance'].values()
        })
        
        return importance_df.sort_values('importance', ascending=False) 