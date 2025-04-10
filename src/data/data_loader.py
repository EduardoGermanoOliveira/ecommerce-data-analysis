import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcommerceDataLoader:
    """Class to handle loading and preprocessing of e-commerce data."""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data directory or file
        """
        self.data_path = Path(data_path)
        
    def load_data(self, chunksize: Optional[int] = None) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Load the e-commerce data, optionally in chunks for large files.
        
        Args:
            chunksize: Number of rows to read at a time. If None, reads entire file.
            
        Returns:
            DataFrame or TextFileReader object
        """
        try:
            if chunksize:
                logger.info(f"Loading data in chunks of {chunksize} rows...")
                return pd.read_csv(
                    self.data_path,
                    chunksize=chunksize,
                    parse_dates=['event_time']
                )
            else:
                logger.info("Loading entire dataset...")
                return pd.read_csv(
                    self.data_path,
                    parse_dates=['event_time']
                )
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Handle missing values
        df['category_code'] = df['category_code'].fillna('unknown')
        df['brand'] = df['brand'].fillna('unknown')
        
        # Add time-based features
        df['hour'] = df['event_time'].dt.hour
        df['day_of_week'] = df['event_time'].dt.dayofweek
        df['month'] = df['event_time'].dt.month
        
        # Convert categorical columns
        categorical_cols = ['event_type', 'category_code', 'brand']
        for col in categorical_cols:
            df[col] = df[col].astype('category')
            
        return df 