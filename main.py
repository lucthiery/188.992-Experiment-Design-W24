# Import general modules
import os
import glob
import asyncio      # A library for asynchronous programming
import subprocess


# Import project modules - helpers
from utils.logger import Logger
from utils.data_utils import process_files

# Import project modules - API clients
from api.pubmed_client import PubMedClient
from api.openalex_client import OpenAlexClient
from api.crossref_client import CrossrefClient

# Import project modules - models
from models.naive_bayes_function import naive_bayes_main
from models.alternative_nb_with_cv import alternate_nb_with_cv_main


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

    # Ask the user if they want to load the data initially
    user_input = input("Do you want to load the data initially? (yes/no): ").strip().lower()
    if user_input == "yes":
        logger.info("Data loading started by the user.")
        # Run the main function
        asyncio.run(_data_processing())

    # Install the required R packages if user confirms to do so
    user_input = input("Do you want to install the required R packages? (yes/no): ").strip().lower()
    if user_input == "yes":
        logger.info("Installing required R packages...")
        
        # Install the packages using the requirements.R script
        r_script_path_requirements = os.path.join("requirements.R")
        logger.info(f"Running R script: {r_script_path_requirements}")
        subprocess.run(["Rscript", r_script_path_requirements], check=True)

    # Run the naive bayes model
    print("\n")
    logger.info("Running the naive bayes model...")
    naive_bayes_main()
    logger.info("Naive bayes model completed successfully!")

    # Run the alternative naive bayes model with cross-validation
    print("\n")
    logger.info("Running the alternative naive bayes model with cross-validation...")
    alternate_nb_with_cv_main()
    logger.info("Alternative naive bayes model with cross-validation completed successfully!")
    
    # Rund the d2v+svm.R script
    r_script_path_d2v_svm = os.path.join("models", "d2v+svm.R")
    logger.debug(f"Running R script: {r_script_path_d2v_svm}")
    result = subprocess.run(
        ["Rscript", r_script_path_d2v_svm],
        capture_output=True,  # Fängt Standardausgabe und Fehlerausgabe ein
        text=True             # Gibt die Ausgabe als String zurück
    )
    
    # Output the results from the R script
    logger.info("R Script Output:")
    logger.info(result.stdout)
    logger.info("R Script Errors (if existing):")
    logger.info(result.stderr)



    # End of the program
    logger.info("Program successfully completed!")
    logger.info("Goodbye!")


