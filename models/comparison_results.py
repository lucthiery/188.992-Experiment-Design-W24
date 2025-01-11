import asreview
from asreview.models import *
from asreview.query_strategies import *
from asreview.balance_strategies import *
from asreview.feature_extraction import *

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


from sentence_transformers import SentenceTransformer

    
calcium_preprocessed = pd.read_csv('calcium_preprocessed.csv')

as_data = asreview.data.ASReviewData(df = calcium_preprocessed)
# Settings
train_model = NaiveBayesClassifier()
query_model = MaxQuery()
balance_model = DoubleBalance()
feature_model = Tfidf()



# Start the review process
reviewer_full_schoot = asreview.ReviewSimulate(
      as_data, 
      model=train_model,
      query_model=query_model,
      balance_model=balance_model,
      feature_model=feature_model,
      n_instances=10,
      init_seed=10,
      n_prior_included=1,
      n_prior_excluded=1,
      state_file="schoot_tfidf_SVM.h5"
  )
reviewer_full_schoot.review()
