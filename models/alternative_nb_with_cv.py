import pandas as pd 
import requests 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import  classification_report,  confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

import nltk
nltk.download('stopwords')

from utils.logger import Logger
from utils.evaluation_metrics import wss


logger = Logger(__name__)

def model_nb(data,dataset_name, label):
    nltk.download('stopwords')
    X = data.drop(label, axis = 1)
    y = data[label]
    # First, split the data into training (60%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Then, split the training set into training (60% of total) and validation (20% of total) sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


    X_valid['combined'] = X_valid['title'] + " " + X_valid['abstracts']
    X_train['combined'] = X_train['title'] + " " + X_train['abstracts']
    X_test['combined'] = X_test['title'] + " " + X_test['abstracts']
    X_val = X_valid['combined'].fillna('').astype(str)
    X_train = X_train['combined'].fillna('').astype(str)
    X_test = X_test['combined'].fillna('').astype(str)
    """Perform hyperparameter tuning using GridSearchCV."""
    #here we need X_train and y_train 
    # Define the pipeline for preprocessing and model training
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),  # Vectorizer with stopword removal
        ('nb', MultinomialNB())       # Naive Bayes classifier
    ])

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'tfidf__max_features': [2000, 3000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        #'tfidf__min_df': [0,1, 2, 3],
        'nb__alpha': [1.0, 0.5, 0.1, 0.01]
    }

    # Perform grid search with stratified cross-validation
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words=stopwords.words('english'))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

    # Evaluate the best model on the validation set
    y_val_pred = best_model.predict(X_val)
    print("Validation Classification Report:")
    print(classification_report(y_valid, y_val_pred))

    
    # Retrain the best model on the full training data (including validation)
    best_model.fit(X_train, y_train)

    # Evaluate the retrained model on the test set
    y_test_pred = best_model.predict(X_test)
    # Confusion-Matrix-Komponenten extrahieren
    #tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    # Gesamtanzahl der Beispiele
    n = len(y_test)

    wss_85 = wss(y_test, y_test_pred, 0.85)
    wss_95 = wss(y_test, y_test_pred, 0.95)
    # WSS nach Formel berechnen
    #wss_85 = (tn + fn) / n - (1 -85)
    #wss_95 = (tn + fn) / n - (1 - 95)

    #result = pd.DataFrame({'label': y_test, 'prediction_nb': y_test_pred})
    print(wss_85)
    print(wss_95)

    result_data = pd.DataFrame([{
        "Dataset": dataset_name,
        "Model": "Plain NB",
        "WSS@85": wss_85,
        "WSS@95": wss_95
    }])

    results_file="../results.csv"
    # Append to the CSV file if it exists, otherwise create it
    if os.path.isfile(results_file):
        result_data.to_csv(results_file, mode='a', header=False, index=False)
    else:
        result_data.to_csv(results_file, mode='w', header=True, index=False)
    #print(f"Appended results for {dataset_name} - {"Plain NB"} to {results_file}")



def alternate_nb_with_cv_main():
        # Get the absolute path of the project root (not the current script's folder)
    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the absolute path to 'Data/preprocessed/'
    preprocessed_folder = os.path.join(project_folder, 'Data', 'preprocessed')

    # Check if the folder exists
    if os.path.exists(preprocessed_folder):
        logger.info(f"The folder exists: {preprocessed_folder}")
    else:
        logger.error(f"The folder does not exist: {preprocessed_folder}")

    # Construct the full paths to the CSV files
    calcium_file_path = os.path.join(preprocessed_folder, 'calcium_preprocessed.csv')
    virus_file_path = os.path.join(preprocessed_folder, 'virus_preprocessed.csv')
    depression_file_path = os.path.join(preprocessed_folder, 'depression_preprocessed.csv')

    # Verify each file exists before loading
    for file_path in [calcium_file_path, virus_file_path, depression_file_path]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            
    # Load the CSV files (only if they exist)
    try:
        calcium_preprocessed = pd.read_csv(calcium_file_path)
        virus_preprocessed = pd.read_csv(virus_file_path)
        depression_preprocessed = pd.read_csv(depression_file_path)
    except FileNotFoundError as e:
        logger.error(f"Error loading CSV file: {e}")

    result_calcium = model_nb(calcium_preprocessed, "Calcium", 'label_included')
    result_depression = model_nb(depression_preprocessed, "Depression", 'label_included')
    result_virus = model_nb(virus_preprocessed, "Virus", 'label_included')

    print(result_calcium)
    print(result_depression)
    print(result_virus)
    


