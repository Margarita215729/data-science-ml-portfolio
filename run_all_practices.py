#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Science & Machine Learning Portfolio
Automated execution of all practical assignments

This script runs all 5 machine learning projects in sequence:
1. World Population Analysis
2. Heart Attack Prediction
3. Weather Prediction with XGBoost
4. Real Estate Price Prediction  
5. Customer Segmentation with PCA

Author: Margarita Makeeva
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_practice(practice_file, description):
    """Execute a single practice file with error handling"""
    logger.info(f" Starting: {description}")
    logger.info(f" Executing: {practice_file}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, practice_file], 
                              capture_output=True, text=True, timeout=600)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f" Completed: {description}")
            logger.info(f"  Execution time: {execution_time:.2f} seconds")
            return True
        else:
            logger.error(f" Failed: {description}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f" Timeout: {description} exceeded 10 minutes")
        return False
    except Exception as e:
        logger.error(f" Exception in {description}: {str(e)}")
        return False

def check_requirements():
    """Check if all required packages are installed"""
    logger.info(" Checking system requirements...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'xgboost', 'kaggle'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f" Missing packages: {', '.join(missing_packages)}")
        logger.error("Please run: pip install -r requirements.txt")
        return False
    
    logger.info(" All required packages are installed")
    return True

def create_output_directories():
    """Create necessary output directories"""
    directories = ['plots', 'datasets', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f" Created/verified directory: {directory}")

def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info(" DATA SCIENCE & MACHINE LEARNING PORTFOLIO")
    logger.info(" Comprehensive ML Projects Execution")
    logger.info(" Author: Margarita Makeeva")
    logger.info("=" * 80)
    
    # System checks
    if not check_requirements():
        sys.exit(1)
    
    create_output_directories()
    
    # Define all practices
    practices = [
        {
            'file': 'practice_1.py',
            'description': ' World Population Analysis & Visualization'
        },
        {
            'file': 'practice_2.py', 
            'description': ' Heart Attack Prediction Classification'
        },
        {
            'file': 'practice_3.py',
            'description': ' Weather Prediction with XGBoost & Clustering'
        },
        {
            'file': 'practice_4.py',
            'description': ' Real Estate Price Prediction'
        },
        {
            'file': 'practice_5.py',
            'description': ' Customer Segmentation with PCA'
        }
    ]
    
    # Execute all practices
    successful_runs = 0
    total_start_time = time.time()
    
    for i, practice in enumerate(practices, 1):
        logger.info("-" * 60)
        logger.info(f" Project {i}/5: {practice['description']}")
        logger.info("-" * 60)
        
        if Path(practice['file']).exists():
            success = run_practice(practice['file'], practice['description'])
            if success:
                successful_runs += 1
        else:
            logger.error(f"❌ File not found: {practice['file']}")
    
    # Summary
    total_execution_time = time.time() - total_start_time
    logger.info("=" * 80)
    logger.info(" EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Successful projects: {successful_runs}/{len(practices)}")
    logger.info(f"  Total execution time: {total_execution_time/60:.2f} minutes")
    
    if successful_runs == len(practices):
        logger.info(" ALL PROJECTS COMPLETED SUCCESSFULLY!")
        logger.info(" Check the 'plots' directory for visualizations")
        logger.info(" Check 'portfolio_execution.log' for detailed logs")
    else:
        logger.warning(f"  {len(practices) - successful_runs} project(s) failed")
    
    logger.info("=" * 80)
    logger.info(" Portfolio demonstration complete!")
    logger.info(" Ready for professional presentation")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
