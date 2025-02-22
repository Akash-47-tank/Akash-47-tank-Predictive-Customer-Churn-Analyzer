"""
Logger utility for the Customer Churn Prediction project.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from src.config.config import LOG_FORMAT, LOG_FILE

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger, typically __name__ from the calling module
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
