# Import general libraries
import logging

from datetime import datetime


# Create a logger class to log the information
class Logger:

    # Initialize the logger
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    # Retrieve the logger instance
    def __getattr__(self, attr):
        """Delegate attribute access to the internal logger instance."""
        return getattr(self.logger, attr)