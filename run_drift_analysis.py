#!/usr/bin/env python3
"""
Standalone script for running data drift analysis on the bike sharing dataset.

This script provides a convenient way to run data drift analysis independently
of the main DVC pipeline, with options for custom configuration and output.

Usage:
    python run_drift_analysis.py [options]

Example:
    python run_drift_analysis.py --test-data data/processed/bike_sharing_test_cleaned.csv --model models/random_forest.pkl
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_drift import DataDriftOrchestrator, DriftDetectionConfig


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('drift_analysis.log')
        ]
    )


def validate_inputs(args) -> None:
    """Validate input arguments."""
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    if not args.model.endswith('.pkl'):
        raise ValueError("Model file must be a pickle file (.pkl)")


def create_drift_config(args) -> DriftDetectionConfig:
    """Create drift detection configuration from arguments."""
    return DriftDetectionConfig(
        psi_threshold=args.psi_threshold,
        ks_threshold=args.ks_threshold,
        js_threshold=args.js_threshold,
        rmse_degradation_threshold=args.rmse_degradation_threshold,
        r2_degradation_threshold=args.r2_degradation_threshold,
        feature_drift_magnitude=args.feature_drift_magnitude,
        missing_data_ratio=args.missing_data_ratio,
        noise_factor=args.noise_factor
    )


def print_analysis_summary(results: dict) -> None:
    """Print a summary of the drift analysis results."""
    print("\n" + "="*60)
    print("DATA DRIFT ANALYSIS SUMMARY")
    print("="*60)
    
    summary = results.get('summary', {})
    baseline = results.get('baseline_metrics', {})
    
    # Baseline performance
    print(f"\nBaseline Model Performance:")
    print(f"  RMSE: {baseline.get('rmse', 0):.2f}")
    print(f"  MAE:  {baseline.get('mae', 0):.2f}")
    print(f"  R²:   {baseline.get('r2', 0):.4f}")
    print(f"  MAPE: {baseline.get('mape', 0):.2f}%")
    
    # Summary statistics
    print(f"\nDrift Analysis Results:")
    print(f"  Total scenarios analyzed: {summary.get('total_scenarios_analyzed', 0)}")
    print(f"  Scenarios with drift: {summary.get('scenarios_with_drift', 0)}")
    print(f"  Scenarios with performance degradation: {summary.get('scenarios_with_performance_degradation', 0)}")
    
    # Most affected features
    affected_features = summary.get('most_affected_features', {})
    if affected_features:
        print(f"\nMost affected features:")
        for feature, count in list(affected_features.items())[:3]:
            print(f"  {feature}: {count} scenarios")
    
    # Worst performing scenario
    worst_scenario = summary.get('worst_performing_scenario')
    if worst_scenario:
        print(f"\nWorst performing scenario: {worst_scenario}")
        if worst_scenario in results.get('scenarios', {}):
            worst_results = results['scenarios'][worst_scenario]['performance_degradation']
            rmse_change = worst_results.get('rmse_change_pct', 0)
            r2_change = worst_results.get('r2_change_pct', 0)
            print(f"  RMSE change: +{rmse_change:.1f}%")
            print(f"  R² change: -{r2_change:.1f}%")
    
    # Recommendations
    recommendations = summary.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "="*60)


def main():
    """Main function to run drift analysis."""
    parser = argparse.ArgumentParser(
        description="Run data drift analysis on bike sharing dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--test-data',
        default='data/processed/bike_sharing_test_cleaned.csv',
        help='Path to test dataset CSV file'
    )
    parser.add_argument(
        '--model',
        default='models/random_forest.pkl',
        help='Path to trained model pickle file'
    )
    parser.add_argument(
        '--target',
        default='cnt',
        help='Target column name'
    )
    parser.add_argument(
        '--output-dir',
        default='reports/drift_analysis',
        help='Output directory for drift analysis results'
    )
    
    # Feature configuration
    parser.add_argument(
        '--num-cols',
        nargs='+',
        default=['temp', 'hum', 'windspeed'],
        help='Numerical feature columns'
    )
    parser.add_argument(
        '--cat-cols',
        nargs='+',
        default=['season', 'yr', 'mnth', 'hr', 'weathersit', 'weekday', 'holiday', 'workingday'],
        help='Categorical feature columns'
    )
    
    # Drift detection thresholds
    parser.add_argument(
        '--psi-threshold',
        type=float,
        default=0.2,
        help='Population Stability Index threshold for drift detection'
    )
    parser.add_argument(
        '--ks-threshold',
        type=float,
        default=0.1,
        help='Kolmogorov-Smirnov test threshold for drift detection'
    )
    parser.add_argument(
        '--js-threshold',
        type=float,
        default=0.1,
        help='Jensen-Shannon distance threshold for drift detection'
    )
    
    # Performance degradation thresholds
    parser.add_argument(
        '--rmse-degradation-threshold',
        type=float,
        default=0.15,
        help='RMSE degradation threshold (15% = 0.15)'
    )
    parser.add_argument(
        '--r2-degradation-threshold',
        type=float,
        default=0.1,
        help='R² degradation threshold (10% = 0.1)'
    )
    
    # Drift simulation parameters
    parser.add_argument(
        '--feature-drift-magnitude',
        type=float,
        default=0.3,
        help='Magnitude of feature drift in standard deviations'
    )
    parser.add_argument(
        '--missing-data-ratio',
        type=float,
        default=0.1,
        help='Proportion of missing data to introduce in drift scenarios'
    )
    parser.add_argument(
        '--noise-factor',
        type=float,
        default=0.2,
        help='Noise multiplication factor for drift simulation'
    )
    
    # Other options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Validate inputs
        logger.info("Validating input arguments...")
        validate_inputs(args)
        
        # Create drift configuration
        logger.info("Creating drift detection configuration...")
        drift_config = create_drift_config(args)
        
        if args.verbose:
            logger.info(f"Drift configuration: PSI={drift_config.psi_threshold}, "
                       f"KS={drift_config.ks_threshold}, JS={drift_config.js_threshold}")
        
        # Initialize orchestrator
        logger.info("Initializing data drift orchestrator...")
        orchestrator = DataDriftOrchestrator(drift_config)
        
        # Run drift analysis
        logger.info("Starting data drift analysis...")
        logger.info(f"  Test data: {args.test_data}")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Target: {args.target}")
        logger.info(f"  Output directory: {args.output_dir}")
        
        results = orchestrator.run_drift_analysis(
            reference_data_path=args.test_data,
            model_path=args.model,
            target_col=args.target,
            num_cols=args.num_cols,
            cat_cols=args.cat_cols,
            output_dir=args.output_dir
        )
        
        # Print summary
        if args.verbose or args.log_level == 'INFO':
            print_analysis_summary(results)
        
        # Success message
        logger.info(f"Data drift analysis completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Check for alerts
        summary = results.get('summary', {})
        if summary.get('scenarios_with_drift', 0) > 0:
            logger.warning(f"⚠️  ALERT: Data drift detected in {summary['scenarios_with_drift']} scenarios!")
        
        if summary.get('scenarios_with_performance_degradation', 0) > 0:
            logger.warning(f"⚠️  ALERT: Performance degradation detected in {summary['scenarios_with_performance_degradation']} scenarios!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during drift analysis: {str(e)}")
        if args.log_level == 'DEBUG':
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)