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

import nltk
nltk.download('stopwords')


calcium_preprocessed = pd.read_csv('../data/preprocessed/calcium_preprocessed.csv')
virus_preprocessed = pd.read_csv('../data/preprocessed/virus_preprocessed.csv')
depression_preprocessed = pd.read_csv('../data/preprocessed/depression_preprocessed.csv')

def model_nb(data, label, recall_level):
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
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    # Gesamtanzahl der Beispiele
    n = len(y_test)

    # WSS nach Formel berechnen
    wss = (tn + fn) / n - (1 - recall_level)

    #result = pd.DataFrame({'label': y_test, 'prediction_nb': y_test_pred})
    print(wss)
    return(tn,fn,tp, fp,n,wss)


data_result = model_nb(calcium_preprocessed, 'label_included', 0.85)


