# Import required libraries
import aiohttp
import json

from typing import Dict, Any, Tuple, Optional

# Import project specific modules
from utils.logger import Logger
from .base_client import BaseAPIClient


# Create a logger instance
logger = Logger(__name__)


class OpenAlexClient(BaseAPIClient):
    """
    A client class to interact with the OpenAlex API.

    This class is used to fetch metadata such as titles and abstracts of scholarly works
    from the OpenAlex database. It also parses and returns the relevant information in a usable format.
    """

    def __init__(self, rate_limit: Dict[str, Any], base_url: str = "https://api.openalex.org/"):
        """
        Initialize the OpenAlexClient.

        Args:
            rate_limit (dict): A dictionary that specifies rate limits for API requests.
            base_url (str): The base URL for the OpenAlex API.
        """

        # Initialize the base client class with the provided rate_limit and base_url
        super().__init__(rate_limit, base_url)

        # Initialize a logger for the class, which helps in logging important messages
        self.logger = Logger(self.__class__.__name__)



    def _parse_openalex_json(self, json_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Parse the JSON data returned from OpenAlex and extract the article title and abstract.

        Args:
            json_data (dict): The JSON data returned by the OpenAlex API.

        Returns:
            Tuple[str, str]: A tuple containing the article title and abstract.
                             If either is missing, default values are returned: "No title" and "No abstract".
        """

        try:
            # Load the JSON data into a dictionary
            data = json.loads(json_data)

            # Extract the article title, use default value "No title" if not found
            title = json_data.get("title", "No title")

            # Set the abstract text to "No abstract" if not found
            abstract = "No abstract"

            # Extract the abstract text if available
            # Handle abstract inversion format
            if "abstract_inverted_index" in data:
                abstract_parts = []

                for word, positions in data["abstract_inverted_index"].items():
                    for pos in positions:
                        # Reconstruct the abstract based on word positions
                        abstract_parts.insert(pos, word)

                # Join all parts into a single string
                abstract = " ".join(abstract_parts)

            elif data.get("abstract"):
                # If the abstract is directly available, use it
                abstract = data["abstract"]
            
            # Return the extracted title and abstract
            return title, abstract
        
        # Handle JSON parsing errors
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return "No title", "No abstract"




    async def fetch_openalex_data(self, session: aiohttp.ClientSession, openalex_id: str) -> Tuple[str, str]:
        """
        Fetch metadata for an OpenAlex work ID.
        
        Args:
            openalex_id (str): OpenAlex work ID (format: W123456789)
            
        Returns:
            Tuple[str, str]: (title, abstract)
        """

        # Log the start of the data fetching process for the given OpenAlex work ID
        logger.info(f"Fetching OpenAlex data for work ID {openalex_id}")

        # Build the endpoint URL dynamically using the work ID
        endpoint = f"works/{openalex_id}"

        # Define the parameters for the GET request
        # params = {"mailto": "your@email.com"}  # Recommended by OpenAlex guidelines
        
        try:
            # Fetch the data from the OpenAlex API using the `fetch` method from the base client
            response_data = await self.fetch(session, endpoint, params={})

            if response_data:
                # If data is successfully fetched, parse it using the helper function
                self.logger.info(f"Successfully fetched OpenAlex data for work ID {openalex_id}")

                # Return parsed title and abstract
                return self._parse_openalex_json(response_data)
            
            # If no data is returned, log a warning and return default values
            self.logger.warning(f"No data returned for OpenAlex work ID {openalex_id}")
            return "No title", "No abstract"
            
        # Catch and handle any errors during the API request
        except Exception as e:
            self.logger.error(f"Error fetching OpenAlex data: {e}")
            return "No title", "No abstract"