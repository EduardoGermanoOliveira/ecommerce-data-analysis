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
        
    def _process_chunks(self, chunk_iter) -> pd.DataFrame:
        """
        Processa iterador de chunks e retorna DataFrame consolidado.

        Args:
            chunk_iter: Iterator de DataFrames

        Returns:
            DataFrame combinado e pré-processado
        """
        logger.info("Processando chunks...")
        results = []
        
        for i, chunk in enumerate(chunk_iter):
            if chunk is not None and not chunk.empty:
                logger.info(f"Processando chunk {i+1} com shape {chunk.shape}")
                chunk = self.preprocess_data(chunk)
                results.append(chunk)
            else:
                logger.warning(f"Chunk {i+1} vazio ou inválido. Ignorado.")

        if not results:
            logger.warning("Nenhum chunk processado com sucesso!")
            return pd.DataFrame()  # Retorna DataFrame vazio para evitar erro

        return pd.concat(results, ignore_index=True)
        
    def load_data(self, chunksize: Optional[int] = None) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Load the e-commerce data, optionally in chunks for large files.
        
        Args:
            chunksize: Number of rows to read at a time. If None, reads entire file.
            
        Returns:
            DataFrame with preprocessed data or TextFileReader for chunked reading
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
                df = pd.read_csv(
                    self.data_path,
                    parse_dates=['event_time']
                )
                return self.preprocess_data(df)
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