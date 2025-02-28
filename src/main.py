import os
import sys
import logging
import yaml

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AIVOL")

# Load Configuration
CONFIG_PATH = "config/config.yaml"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        logger.error("Configuration file not found: %s", CONFIG_PATH)
        sys.exit(1)
    
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully.")
        return config

# Initialize Components
def initialize_system():
    logger.info("Initializing AI-Powered Voice-Assisted Object Locator (AIVOL)...")
    
    # Check Python version
    if sys.version_info[:2] != (3, 11):
        logger.warning("Python 3.11 is required. Current version: %s", sys.version)
    
    # Check required directories
    required_dirs = ["models/yolo", "logs"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info("Created missing directory: %s", directory)
    
    logger.info("System initialization complete.")

if __name__ == "__main__":
    logger.info("Starting AIVOL Main Controller...")
    
    # Load configuration
    config = load_config()
    
    # Initialize system
    initialize_system()
    
    logger.info("AIVOL is ready for further development!")