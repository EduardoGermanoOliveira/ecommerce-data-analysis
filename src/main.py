import pandas as pd
import logging
import sys
from pathlib import Path
from data.data_loader import EcommerceDataLoader
from analysis.exploratory import EcommerceAnalyzer
from models.demand_forecast import DemandForecaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the e-commerce analysis pipeline."""
    
    # Create output directories
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    
    try:
        # Get CSV file path from command line argument or use default
        csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/2019-Oct.csv'
        logger.info(f"Using data file: {csv_file}")
        
        # Load and preprocess data
        logger.info("Loading data...")
        data_loader = EcommerceDataLoader(csv_file)
        df_chunks = data_loader.load_data(chunksize=100000)  # Load in chunks due to large file size
        
        # Initialize analyzer
        logger.info("Starting exploratory analysis...")
        analyzer = EcommerceAnalyzer(df_chunks)
        
        # Perform exploratory analysis
        event_dist = analyzer.analyze_event_distribution()
        logger.info("\nEvent Distribution:\n%s", event_dist)
        
        category_perf = analyzer.analyze_category_performance(top_n=10)
        logger.info("\nTop 10 Categories by Performance:\n%s", category_perf)
        
        # Generate plots
        analyzer.plot_event_distribution(
            save_path=output_dir / 'plots' / 'event_distribution.png'
        )
        analyzer.plot_category_performance(
            save_path=output_dir / 'plots' / 'category_performance.png'
        )
        
        # Prepare data for forecasting
        logger.info("Preparing data for demand forecasting...")
        forecaster = DemandForecaster(df_chunks)
        X, y = forecaster.prepare_data()
        
        # Train Random Forest model if we have enough data
        if not X.empty and not y.empty:
            logger.info("Training Random Forest model...")
            rf_results = forecaster.train_random_forest(X, y)
            logger.info("Random Forest Metrics: %s", rf_results['metrics'])
            
            # Get feature importance
            try:
                feature_importance = forecaster.get_feature_importance()
                logger.info("\nFeature Importance:\n%s", feature_importance)
                
                # Save results
                feature_importance.to_csv(
                    output_dir / 'models' / 'feature_importance.csv',
                    index=False
                )
            except ValueError as e:
                logger.warning(f"Could not get feature importance: {str(e)}")
        else:
            logger.warning("Not enough data to train Random Forest model")
        
        # Train Prophet model if we have enough data
        if not X.empty and not y.empty:
            logger.info("Training Prophet model...")
            try:
                prophet_results = forecaster.train_prophet(forecast_periods=30)
                logger.info("Prophet forecast completed")
            except Exception as e:
                logger.warning(f"Error training Prophet model: {str(e)}")
        else:
            logger.warning("Not enough data to train Prophet model")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error("Error during analysis: %s", str(e))
        raise

if __name__ == "__main__":
    main() 