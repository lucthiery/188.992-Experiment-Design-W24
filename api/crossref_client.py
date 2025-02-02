# ./api/crossref_client.py

import aiohttp
import json
from typing import Dict, Any, Tuple

from utils.logger import Logger
from .base_client import BaseAPIClient

class CrossrefClient(BaseAPIClient):
    """
    Client for interacting with the CrossRef API to fetch metadata (title and abstract)
    for scholarly articles based on their DOI.

    This client inherits from BaseAPIClient, leveraging its rate-limiting and fetch
    functionalities to perform asynchronous HTTP requests.
    """

    def __init__(self, rate_limit: Dict[str, Any], base_url: str = "https://api.crossref.org/works"):
        """
        Initializes the CrossrefClient with the specified rate limiting parameters and base URL.

        Args:
            rate_limit (Dict[str, Any]): A dictionary containing the rate limiting parameters.
                For example: {"limit": 3, "interval": 1.0}.
            base_url (str): The base URL for CrossRef API requests. Defaults to "https://api.crossref.org/works".
        """
        # Initialize the base API client with rate limit and base URL
        super().__init__(rate_limit, base_url)
        # Set up a logger instance for this client
        self.logger = Logger(self.__class__.__name__)
        self.logger.info("CrossrefClient initialized with base URL: {}".format(self.base_url))

    async def fetch_crossref_data(self, session: aiohttp.ClientSession, doi: str) -> Tuple[str, str]:
        """
        Fetches metadata (title and abstract) from the CrossRef API using the provided DOI.

        The method first cleans the DOI (removing any 'https://doi.org/' prefix), then constructs
        the endpoint by appending the cleaned DOI to the base URL. It makes an asynchronous GET
        request using the inherited `fetch` method. If successful, it parses the JSON response
        and extracts the title and abstract. Otherwise, default values are returned.

        Args:
            session (aiohttp.ClientSession): The aiohttp session used for asynchronous HTTP requests.
            doi (str): The DOI of the article. This string may include the "https://doi.org/" prefix.

        Returns:
            Tuple[str, str]: A tuple containing the title and abstract.
                             Returns ("No title", "No abstract") if data retrieval fails.
        """
        # Remove any "https://doi.org/" prefix and extra whitespace from the DOI
        doi_clean = doi.replace("https://doi.org/", "").strip()
        # The cleaned DOI is used as the endpoint for the request
        endpoint = doi_clean

        try:
            # Make an asynchronous GET request using the inherited fetch method
            response_data = await self.fetch(session, endpoint, params={})
            if response_data:
                # Parse the response data as JSON
                data = json.loads(response_data)
                # Extract the metadata contained in the "message" field
                message = data.get("message", {})
                # Extract the title, which is typically provided as a list; use the first element
                title_list = message.get("title", [])
                title = title_list[0] if title_list else "No title"
                # Extract the abstract; if not present, default to "No abstract"
                abstract = message.get("abstract", "No abstract")
                self.logger.info(f"Successfully fetched CrossRef data for DOI {doi}")
                return title, abstract
            else:
                # Log a warning if no data is returned and return default values
                self.logger.warning(f"No data returned from CrossRef for DOI {doi}")
                return "No title", "No abstract"
        except Exception as e:
            # Log any errors encountered during the fetch process and return default values
            self.logger.error(f"Error fetching CrossRef data for DOI {doi}: {e}")
            return "No title", "No abstract"
