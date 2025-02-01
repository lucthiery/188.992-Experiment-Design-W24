# Import required libaries
import aiohttp
import asyncio

from typing import Dict, Tuple, Any, Optional
from xml.etree import ElementTree as ET


# Import project specific modules
from utils.logger import Logger
from .base_client import BaseAPIClient


class PubMedClient(BaseAPIClient):
    """
    A client to fetch data from the PubMed API and parse the results.

    This class provides methods to fetch PubMed article data by PMID (PubMed Identifier)
    and parse the returned XML data to extract the article title and abstract.
    """


    def __init__(self, rate_limit: Dict[str, Any], base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"):
        """
        Initialize the PubMedClient.

        Args:
            rate_limit (dict): A dictionary that specifies rate limits for API requests.
            base_url (str): The base URL for the PubMed API.
        """

        # Initialize the base client class with the provided rate_limit and base_url
        super().__init__(rate_limit, base_url)

        # Initialize a logger for the class, which helps in logging important messages
        self.logger = Logger(self.__class__.__name__)




    def _parse_pubmed_xml(self, xml_data: str) -> Tuple[str, str]:
        """
        Parse the XML data returned from PubMed and extract the article title and abstract.

        Args:
            xml_data (str): The raw XML string returned by the PubMed API.

        Returns:
            Tuple[str, str]: A tuple containing the article title and abstract.
                             If either is missing, default values are returned: "No title" and "No abstract".
        """

        try:
            # Parse the XML data using ElementTree
            root = ET.fromstring(xml_data)

            # Extract the article title, use default value "No title" if not found
            title = root.findtext(".//ArticleTitle", default="No title")

            # Extract the abstract text, join the texts of all AbstractText elements, and use default if empty
            abstract_texts = [elem.text.strip() for elem in root.findall(".//AbstractText") if elem.text]
            abstract = " ".join(abstract_texts) if abstract_texts else "No abstract"
            
            # Return the extracted title and abstract
            return title, abstract

        except ET.ParseError as e:
            # Log XML parsing error and return default values
            self.logger.error(f"Error while parsing XML data: {e}")
            return "No title", "No abstract"
        



    async def fetch_pubmed_data(self, session:aiohttp.ClientSession, pmid: str) -> Tuple[str, str]:
        """
        Fetch a PubMed article by its PMID and parse the title and abstract.

        Args:
            pmid (str): The PubMed Identifier for the article.

        Returns:
            Tuple[str, str]: A tuple containing the article title and abstract.
                             Returns "No title" and "No abstract" if fetching or parsing fails.
        """
        # Define the specific endpoint and parameters for fetching PubMed data
        endpoint = "efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }

        try:
            # Fetch the XML data from PubMed asynchronously using the fetch method from the base class
            xml_data = await self.fetch(session, endpoint, params)

            if xml_data:
                # If data is successfully fetched, parse the XML data to extract title and abstract
                title, abstract = self._parse_pubmed_xml(xml_data)

                # Log the success of fetching the article
                self.logger.info(f"Successfully fetched PubMed article: {pmid} with title: {title}")
                return title, abstract
            
            else:
                # If no data is returned, log a warning
                self.logger.warning(f"No data returned for PubMed ID {pmid}")


        # Handle exceptions that may occur during the request
        except Exception as e:

            # If there is an exception while fetching the data (network errors, timeouts, etc.), log the error
            self.logger.error(f"Failed to fetch PubMed data for {pmid}: {e}")


        # Return default values in case of failure
        return "No title", "No abstract"
    


