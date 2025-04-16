import pandas as pd
import logging
import sys
from pathlib import Path
from data.data_loader import EcommerceDataLoader
from analysis.exploratory import EcommerceAnalyzer
import matplotlib.pyplot as plt

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
        logger.info("Generating visualizations...")
        
        # Event distribution
        analyzer.plot_event_distribution(
            save_path=output_dir / 'plots' / 'event_distribution.png'
        )
        
        # Category performance
        analyzer.plot_category_performance(
            save_path=output_dir / 'plots' / 'category_performance.png'
        )
        
        # Time patterns
        hourly_patterns, daily_patterns = analyzer.analyze_time_patterns()
        if not hourly_patterns.empty:
            plt.figure(figsize=(12, 6))
            hourly_patterns.plot(kind='bar')
            plt.title('Event Distribution by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Number of Events')
            plt.tight_layout()
            plt.savefig(output_dir / 'plots' / 'hourly_patterns.png')
            plt.close()
            
        if not daily_patterns.empty:
            plt.figure(figsize=(12, 6))
            daily_patterns.plot(kind='bar')
            plt.title('Event Distribution by Day of Week')
            plt.xlabel('Day of Week')
            plt.ylabel('Number of Events')
            plt.tight_layout()
            plt.savefig(output_dir / 'plots' / 'daily_patterns.png')
            plt.close()
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error("Error during analysis: %s", str(e))
        raise

if __name__ == "__main__":
    main() 