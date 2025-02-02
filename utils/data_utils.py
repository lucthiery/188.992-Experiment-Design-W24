# Import required libraries
import os
import asyncio
import aiohttp
import datetime
import pandas as pd


from typing import List, Tuple, Dict, Union

# Import project-specific modules
from utils.logger import Logger
from api.pubmed_client import PubMedClient
from api.openalex_client import OpenAlexClient

# Create a logger instance
logger = Logger(__name__)

async def process_files(
        file_path:str,
        pubmed_client: PubMedClient,
        openalex_client: OpenAlexClient,
        output_dir: str
        ) -> None:
    """
    Process a CSV file, fetch metadata (title and abstract) for each record, and save the preprocessed data.
    
    Args:
        file_path (str): Path to the CSV file to be processed.
        pubmed_client (PubMedClient): The client for fetching metadata from PubMed.
        openalex_client (OpenAlexClient): The client for fetching metadata from OpenAlex.
        output_dir (str): Directory where the preprocessed CSV file will be saved.
    """

    try:
        logger.info(f"Loading dataset from: {file_path}")       # Log the file being loaded
        df = pd.read_csv(file_path)                             # Load the CSV file into a pandas DataFrame
        logger.info(f"Dataset shape: {df.shape}")               # Log the shape of the DataFrame (number of rows, columns)

        # Add metadata to the dataset using fallback method including OpenAlex and PubMed
        df_meta = await append_metadata_with_fallback_async(
                df, 
                pubmed_client, 
                openalex_client, 
                pmid_column="pmid", 
                openalex_column="openalex_id"
            )
        
        # Store the preprocessed dataset in the processed data directory
        # To do so, create a timestamped filename and save the dataset as a CSV file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(file_path)
        output_file_name = f"{timestamp}_{base_name}"                       # Construct the output file name
        output_file_path = os.path.join(output_dir, output_file_name)       # Full path for output file

        # Save the preprocessed dataset
        df_meta.to_csv(output_file_path, index=False)
        logger.info(f"Preprocessed dataset saved to: {output_file_path}")

    except Exception as e:
        # Handle exceptions during the processing of the file
        logger.error(f"Error processing file: {file_path}")
        logger.error(e)




async def _fetch_with_fallback(
            session: aiohttp.ClientSession,
            row: pd.Series,
            pubmed_client: PubMedClient,
            openalex_client: OpenAlexClient,
            pmid_column: str = "pmid",
            openalex_column: str = "openalex_id"
        ) -> Tuple[str, str]:
    """
    Fetch metadata (title and abstract) for a given row using PubMed and OpenAlex with a fallback mechanism.
    
    First tries to fetch metadata using PubMed, and if PubMed doesn't return valid data,
    it falls back to OpenAlex if the OpenAlex ID is available.
    
    Args:
        session (aiohttp.ClientSession): The session used for making HTTP requests.
        row (pd.Series): A row of data containing the identifiers (PMID, OpenAlex ID).
        pubmed_client (PubMedClient): The PubMed client used for fetching metadata from PubMed.
        openalex_client (OpenAlexClient): The OpenAlex client used for fetching metadata from OpenAlex.
        pmid_column (str): The column name in the DataFrame that contains the PubMed ID.
        openalex_column (str): The column name in the DataFrame that contains the OpenAlex ID.
        
    Returns:
        Tuple[str, str]: A tuple containing the title and abstract of the work.
    """

    pmid = row[pmid_column]             # Extract the PubMed ID
    openalex_id = row[openalex_column]  # Extract the OpenAlex ID

    # First, try fetching data from PubMed
    title, abstract = await pubmed_client.fetch_pubmed_data(session, str(pmid))
    
    # If PubMed didn't return valid data, and OpenAlex ID is available, fallback to OpenAlex
    # if title == "No title" and abstract == "No abstract":
    #     # Check if the OpenAlex ID is available and not empty
    #     if pd.notna(openalex_id) and str(openalex_id).strip() != "":
    #         logger.info(f"Falling back to OpenAlex for paper with PMID {pmid} and OpenAlex ID {openalex_id}")
    #         title, abstract = await openalex_client.fetch_openalex_data(session, str(openalex_id))
    
    # Return the title and abstract
    return title, abstract




async def append_metadata_with_fallback_async(
          df: pd.DataFrame,
        pubmed_client: PubMedClient,
        openalex_client: OpenAlexClient,
        pmid_column: str = "pmid",
        openalex_column: str = "openalex_id"
    ) -> pd.DataFrame:
    """
    Append metadata (titles and abstracts) to a DataFrame using a fallback mechanism.
    
    This method iterates over each row in the DataFrame and fetches metadata using PubMed, 
    and if necessary, falls back to OpenAlex. It fetches the data asynchronously.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the identifiers (PMID, OpenAlex ID).
        pubmed_client (PubMedClient): The PubMed client used for fetching metadata from PubMed.
        openalex_client (OpenAlexClient): The OpenAlex client used for fetching metadata from OpenAlex.
        pmid_column (str): The column name in the DataFrame that contains the PubMed ID.
        openalex_column (str): The column name in the DataFrame that contains the OpenAlex ID.
        
    Returns:
        pd.DataFrame: The DataFrame with added columns for titles and abstracts.
    """

    # Create a copy of the original DataFrame to work on
    result_df = df.copy()

    # Fetch metadata for each row asynchronously
    async with aiohttp.ClientSession() as session:  # Create an asynchronous HTTP session
        tasks = []  # List to hold all the asynchronous tasks
        for _, row in result_df.iterrows(): # Iterate over each row in the DataFrame
            task = asyncio.create_task(
                _fetch_with_fallback(session, row, pubmed_client, openalex_client, pmid_column, openalex_column)
            )   # Create a task for each row to fetch metadata
            tasks.append(task)  # Append the task to the list of tasks
        results = await asyncio.gather(*tasks)  # Wait for all tasks to complete

    # Add the fetched titles and abstracts to the DataFrame
    result_df["titles"] = [res[0] for res in results]
    result_df["abstracts"] = [res[1] for res in results]
    return result_df



