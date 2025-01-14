# Import general modules


# Import project modules
from utils.logger import Logger


# Main function
if __name__ == "__main__":

    # Create a logger instance
    logger = Logger(__name__)

    # Load the data
    logger.info("Program successfully started!")

    # Run the comparison_results.py script
    logger.info("Running comparison_results.py script...")
    from models import comparison_results


