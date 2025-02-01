
# Import required libraries
import aiohttp
import asyncio

import pandas as pd

from typing import Optional, Dict, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from xml.etree import ElementTree as ET

# Import project specific modules
from utils.logger import Logger


class BaseAPIClient:
    """
    A base class for API clients using aiohttp for asynchronous requests.

    This class provides functionality for rate-limiting, handling API requests,
    and managing an aiohttp session.
    """


    def __init__(self, rate_limit: Dict[str, float], base_url: str):
        """
        Initialize the BaseAPIClient with rate limiting and session management.

        Args:
            rate_limit (dict): Dictionary with keys 'limit' (int) for concurrent requests
                               and 'interval' (float) for delay between requests.
            base_url (str): The base URL for API requests.
        """
        
        self.semaphore = asyncio.Semaphore(rate_limit["limit"])     # Limits concurrent requests
        self.interval = rate_limit["interval"]                      # Interval between requests
        self.base_url = base_url.rstrip('/')                        # Ensures a consistent URL format
        self.logger = Logger(__name__)                              # Logger instance

    async def close(self):
        """
        Close the aiohttp session.

        Should be called when the client is no longer needed to free resources.
        """
        if self.session is not None:
            await self.session.close()
            self.logger.info("aiohttp session closed.")


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def fetch(
            self, 
            session: aiohttp.ClientSession, 
            endpoint: str, 
            params: Dict[str, Any]
        ) -> Optional[str]:
        """
        Perform an asynchronous GET request to the specified API endpoint with retries.

        Args:
            endpoint (str): The API endpoint to send the request to (appended to base_url).
            params (dict): Query parameters for the GET request.

        Returns:
            Optional[str]: The response text if the request is successful, else None.
        """


        url = f"{self.base_url}/{endpoint.lstrip('/')}"


        # if self.session is None:
        #     raise RuntimeError("Session not opened. Call 'await open()' before fetching.")
        

        async with self.semaphore:  # Enforce rate limiting via semaphore
            try:
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.text()
                        self.logger.info(f"Successfully fetched data from {url} with params {params}")
                        return data
                    else:
                        # Log the error with status and response text
                        error_text = await response.text()
                        self.logger.error(f"Error {response.status} while fetching {url} with params {params}: {error_text}")
                        return None
                    
            except aiohttp.ClientError as e:
                self.logger.error(f"HTTP ClientError while fetching {url} with params {params}: {e}")
                raise  # Raise exception to trigger retry

            except asyncio.TimeoutError:
                self.logger.error(f"TimeoutError while fetching {url} with params {params}")
                raise  # Raise exception to trigger retry

            finally:
                await asyncio.sleep(self.interval)