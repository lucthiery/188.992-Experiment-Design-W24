# Import general libraries
import os
import pandas as pd
from pathlib import Path

# Import the asreview specific libraries
from asreview.data import ASReviewData
from asreview.models import NaiveBayesClassifier
from asreview.models.query import MaxQuery
from asreview.models.balance import DoubleBalance
from asreview.models.feature_extraction import Tfidf
from asreview.review import ReviewSimulate
from asreview import ASReviewProject


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sentence_transformers import SentenceTransformer


# Import project specific libraries
from utils.logger import Logger


################## Load data ##################
# Create logger instance
logger = Logger(__name__)

# Get the script directory
script_dir = os.path.dirname(__file__)
logger.debug(f"Script directory: {script_dir}")

# Get the data path
data_path = os.path.join(script_dir, '../Data/preprocessed/')
logger.debug(f"Data path: {data_path}")

# Load the preprocessed data
calcium_preprocessed = pd.read_csv(f"{data_path}/calcium_preprocessed.csv")
logger.debug(f"Calcium preprocessed shape: {calcium_preprocessed.shape}")

# Create an ASReviewData object
as_data = ASReviewData(df = calcium_preprocessed)
logger.debug(f"ASReviewData object created. Number of records: {as_data}")

# Create a project directory
project_path = Path("tmp_data")
project_path.mkdir(exist_ok=True)

# Create an ASReview project
project = ASReviewProject.create(
    project_path=project_path / "api_simulation",
    project_id="api_example",
    project_mode="simulate",
    project_name="api_example",
)
logger.debug("ASReviewProject created successfully.")

# Add dataset to the project folder
dataset_path = project_path / "api_simulation" / "data"
dataset_path.mkdir(parents=True, exist_ok=True)
calcium_preprocessed.to_csv(dataset_path / "calcium_preprocessed.csv", index=False)
project.add_dataset("calcium_preprocessed.csv")
logger.debug("Dataset added to the project.")



################## Settings ##################
# Train the model
train_model = NaiveBayesClassifier()
logger.debug(f"Model: {train_model}")

# Define the query model
query_model = MaxQuery()
logger.debug(f"Query model: {query_model}")

# Define the balance model
balance_model = DoubleBalance()
logger.debug(f"Balance model: {balance_model}")

# Define the feature model
feature_model = Tfidf()
logger.debug(f"Feature model: {feature_model}")



################## Start the Review ##################
reviewer_full_schoot = ReviewSimulate(
      as_data=as_data, 
      model=train_model,
      query_model=query_model,
      balance_model=balance_model,
      feature_model=feature_model,
      n_instances=10,
      project=project,
      init_seed=10,
      n_prior_included=1,
      n_prior_excluded=1
      
  )
logger.debug("ReviewSimulate initialized successfully.")

# Start the review process
try:
    project.update_review(status="review")
    reviewer_full_schoot.review()
    project.mark_review_finished()
    logger.debug("Review process completed successfully.")
except Exception as err:
    project.update_review(status="error")
    logger.error(f"Error during review: {err}")
    raise

# Export the project
export_path = project_path / "api_example.asreview"
project.export(export_path)
logger.debug(f"Project exported to {export_path}")
