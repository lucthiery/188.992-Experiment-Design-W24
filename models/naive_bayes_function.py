import pandas as pd
import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE


#You can choose between: 
#1. You use the plain Naive Bayes model without the consideration of unbalanced data, for this the function is already predefined unbalanced = False 
# If you want to include SMOTE for balancing data, set unbalanced = True  
# If you want to use a different model, that already includes balancing (ComplementNB, see https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf for further information=) set the switch_model to True

def nb_function(df,testsize, unbalanced = False, switch_model = False): 
    
    #Create a combined column of title and abstract for embeddings later 
    df['combined']= df['title'] + df['abstracts']   
    #split the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(df['combined'], df['label_included'], test_size = testsize, stratify = df['label_included'], random_state = 42)
    
    #do the TFIDF Transformation 
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features = 1000, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    if unbalanced == True: 
        smote = SMOTE(random_state=42)
        X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)   
    
    #train naive bayes model 
    if switch_model == True: 
        model = ComplementNB()
    else: 
        model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    #make the predictions 
    y_pred = model.predict(X_test_tfidf)
    #Also use the probabilities for evaluation metrics
    y_pred_proba = model.predict_proba(X_test_tfidf)
    y_pred_proba = np.max(y_pred_proba, axis = 1)
    
    return(y_test, y_pred, y_pred_proba)



def wss(y_test, y_pred, recall_level): 
    
    #we need the length for the formula (can either be from pred or true, should be equal)
    n = len(y_test)
    #compute the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    wss = (tn + fn) / n - (1 - recall_level)
    return(wss) 


calcium_preprocessed = pd.read_csv('calcium_preprocessed.csv')


df = pd.DataFrame(calcium_preprocessed)


y_test, y_pred, y_pred_proba = nb_function(df,0.2, False, True)

print(wss(y_test,y_pred, 0.85))















