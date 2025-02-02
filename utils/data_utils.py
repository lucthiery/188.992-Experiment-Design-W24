# ./utils/data_utils.py


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
from api.crossref_client import CrossrefClient

# Create a logger instance for this module
logger = Logger(__name__)

async def process_files(
        file_path:str,
        pubmed_client: PubMedClient,
        openalex_client: OpenAlexClient,
        crossref_client: CrossrefClient,
        output_dir: str
        ) -> None:
    """
    Processes a CSV file by loading it, fetching metadata for each article using a cascading
    fallback mechanism (PubMed -> OpenAlex -> CrossRef), and saving the augmented DataFrame as
    a new CSV file in the specified output directory.
    
    Args:
        file_path (str): The path to the input CSV file.
        pubmed_client (PubMedClient): The client instance for PubMed.
        openalex_client (OpenAlexClient): The client instance for OpenAlex.
        crossref_client (CrossrefClient): The client instance for CrossRef.
        output_dir (str): The directory where the processed CSV file will be saved.
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
                crossref_client,
                pmid_column="pmid", 
                openalex_column="openalex_id",
                doi_column="doi"
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
            crossref_client: CrossrefClient,
            pmid_column: str = "pmid",
            openalex_column: str = "openalex_id",
            doi_column: str = "doi"
        ) -> Tuple[str, str]:
    """
    Retrieve metadata (title and abstract) using a cascading fallback mechanism.
    
    The function attempts to fetch metadata in the following order:
        1. From PubMed using the PMID.
        2. From OpenAlex using the OpenAlex ID if PubMed returns default values.
        3. From CrossRef using the DOI if both PubMed and OpenAlex fail.
    
    Args:
        session (aiohttp.ClientSession): The aiohttp session used for HTTP requests.
        row (pd.Series): A row from the DataFrame containing the article identifiers.
        pubmed_client (PubMedClient): The client instance for PubMed.
        openalex_client (OpenAlexClient): The client instance for OpenAlex.
        crossref_client (CrossrefClient): The client instance for CrossRef.
        pmid_column (str): Name of the column containing the PubMed ID.
        openalex_column (str): Name of the column containing the OpenAlex ID.
        doi_column (str): Name of the column containing the DOI.
    
    Returns:
        Tuple[str, str]: A tuple containing the title and abstract.
    """

    # Extract identifiers from the row
    pmid = row[pmid_column]             # Extract the PubMed ID
    openalex_id = row[openalex_column]  # Extract the OpenAlex ID
    doi = row[doi_column]               # Extract the DOI

    # Attempt 1: Try to fetch data from PubMed using the PMID.
    title, abstract = await pubmed_client.fetch_pubmed_data(session, str(pmid))
    
    # Attempt 2: If PubMed returns default values, try OpenAlex using the OpenAlex ID.
    if title == "No title" and abstract == "No abstract":
        if pd.notna(openalex_id) and str(openalex_id).strip() != "":
            logger.info(f"Falling back to OpenAlex for PMID {pmid} with OpenAlex ID {openalex_id}")
            title, abstract = await openalex_client.fetch_openalex_data(session, str(openalex_id))

    # Attempt 3: If both PubMed and OpenAlex return default values, use CrossRef with the DOI.
    if title == "No title" and abstract == "No abstract":
        if pd.notna(doi) and str(doi).strip() != "":
            logger.info(f"Falling back to CrossRef for DOI {doi}")
            title, abstract = await crossref_client.fetch_crossref_data(session, str(doi))

    return title, abstract




async def append_metadata_with_fallback_async(
        df: pd.DataFrame,
        pubmed_client: PubMedClient,
        openalex_client: OpenAlexClient,
        crossref_client: CrossrefClient,
        pmid_column: str = "pmid",
        openalex_column: str = "openalex_id",
        doi_column: str = "doi"
    ) -> pd.DataFrame:
    """
    Appends metadata (titles and abstracts) to the DataFrame using a cascading fallback mechanism.
    
    This function processes each row asynchronously. It first attempts to retrieve data from PubMed,
    then falls back to OpenAlex, and finally to CrossRef if needed. The results are added as new
    columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing article identifiers.
        pubmed_client (PubMedClient): The client instance for PubMed.
        openalex_client (OpenAlexClient): The client instance for OpenAlex.
        crossref_client (CrossrefClient): The client instance for CrossRef.
        pmid_column (str): Name of the column with the PubMed ID.
        openalex_column (str): Name of the column with the OpenAlex ID.
        doi_column (str): Name of the column with the DOI.
    
    Returns:
        pd.DataFrame: The DataFrame with added "titles" and "abstracts" columns.
    """

    # Create a copy of the original DataFrame to work on
    result_df = df.copy()

    # Fetch metadata for each row asynchronously
    async with aiohttp.ClientSession() as session:  # Create an asynchronous HTTP session
        tasks = []  # List to hold all the asynchronous tasks
        # Create an asynchronous task for each row to fetch metadata.
        for _, row in result_df.iterrows():
            task = asyncio.create_task(
                _fetch_with_fallback(
                    session,
                    row,
                    pubmed_client,
                    openalex_client,
                    crossref_client,
                    pmid_column,
                    openalex_column,
                    doi_column
                )
            )
            tasks.append(task)  # Append the task to the list of tasks
        # Wait for all tasks to complete concurrently.
        results = await asyncio.gather(*tasks)

    # Add the fetched titles and abstracts to the DataFrame
    result_df["titles"] = [res[0] for res in results]
    result_df["abstracts"] = [res[1] for res in results]
    return result_df



