# Import general modules
import os
import glob
import asyncio      # A library for asynchronous programming


# Import project modules
from utils.logger import Logger
from utils.data_utils import process_files
from api.pubmed_client import PubMedClient
from api.openalex_client import OpenAlexClient
from api.crossref_client import CrossrefClient


# Create a logger instance
logger = Logger(__name__)


async def _data_processing():
    """
    The main function responsible for processing data.

    It sets up the rate limits for API requests, initializes clients for PubMed and OpenAlex, 
    and processes all CSV files in the input directory. The processing is done asynchronously 
    to handle concurrent operations efficiently.

    It also manages the creation of directories for output and logs any important actions or errors.
    """

    # Define rate limiting parameters: 5 concurrent requests, 1-second interval between requests
    rate_limit = {
        "limit": 3,
        "interval": 1.0
    }

    # Initialize PubMedClient with the specified rate limit
    pubmed_client = PubMedClient(rate_limit=rate_limit)
    pubmed_client.logger.info("Base API is working")

    # Initialize OpenAlexClient with the specified rate limit
    openalex_client = OpenAlexClient(rate_limit=rate_limit)
    openalex_client.logger.info("Base API is working")

    # Initialize CrossRefClient with the specified rate limit
    crossref_client = CrossrefClient(rate_limit=rate_limit)
    crossref_client.logger.info("CrossRef API client initialized.")

    # Define the input and output directories
    input_dir = "./Data/original/"
    output_dir = "./Data/preprocessed/"
    os.makedirs(output_dir, exist_ok=True)

    # Process all files in the input directory
    file_paths = glob.glob(os.path.join(input_dir, "*.csv"))
    logger.info(f"Found {len(file_paths)} files in the input directory")

    # Process all files concurrently using asyncio for better performance
    for file_path in file_paths:
        # For each file found, call the `process_files` function asynchronously
        # `process_files` will handle the interaction with PubMed and OpenAlex clients
        await process_files(file_path, pubmed_client, openalex_client, crossref_client, output_dir)


# Main function
if __name__ == "__main__":
    """
    This is the entry point of the script. It initializes the data processing and handles the execution flow.
    The main function also manages logging and the overall start-up process.
    """
    
    # Load the data
    logger.info("Program successfully started!")

    # Run the comparison_results.py script
    logger.info("Running comparison_results.py script...")

    # Run the main function
    asyncio.run(_data_processing())


