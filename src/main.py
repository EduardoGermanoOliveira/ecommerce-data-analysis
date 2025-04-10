import pandas as pd
import logging
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
        # Load and preprocess data
        logger.info("Loading data...")
        data_loader = EcommerceDataLoader('data/raw/ecommerce_data.csv')
        df = data_loader.load_data(chunksize=100000)  # Load in chunks due to large file size
        
        # Initialize analyzer
        logger.info("Starting exploratory analysis...")
        analyzer = EcommerceAnalyzer(df)
        
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
        forecaster = DemandForecaster(df)
        X, y = forecaster.prepare_data()
        
        # Train Random Forest model
        logger.info("Training Random Forest model...")
        rf_results = forecaster.train_random_forest(X, y)
        logger.info("Random Forest Metrics: %s", rf_results['metrics'])
        
        # Get feature importance
        feature_importance = forecaster.get_feature_importance()
        logger.info("\nFeature Importance:\n%s", feature_importance)
        
        # Train Prophet model
        logger.info("Training Prophet model...")
        prophet_results = forecaster.train_prophet(forecast_periods=30)
        logger.info("Prophet forecast completed")
        
        # Save results
        feature_importance.to_csv(
            output_dir / 'models' / 'feature_importance.csv',
            index=False
        )
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error("Error during analysis: %s", str(e))
        raise

if __name__ == "__main__":
    main() 