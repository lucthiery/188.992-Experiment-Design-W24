import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from utils.logger import Logger
from utils.evaluation_metrics import wss

logger = Logger(__name__)

#You can choose between: 
#1. You use the plain Naive Bayes model without the consideration of unbalanced data, for this the function is already predefined unbalanced = False 
# If you want to include SMOTE for balancing data, set unbalanced = True  
# If you want to use a different model, that already includes balancing (ComplementNB, see https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf for further information=) set the switch_model to True

def nb_function(df,testsize, unbalanced = False, switch_model = False): 
    
    #Create a combined column of title and abstract for embeddings later 
    df['combined']= df['title'] + df['abstracts']   
    #split the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(df['combined'], df['label_included'], test_size = testsize, stratify = df['label_included'], random_state = 12345)
    
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



def naive_bayes_main():

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


    df = pd.DataFrame(calcium_preprocessed)

    df2 = pd.DataFrame(virus_preprocessed)

    df3 = pd.DataFrame(depression_preprocessed)

    df['title'] = df['title']

    df2['title'] = df2['title']

    df3['title'] = df3['title']


    #First we start with the plain model without any balancing 
    y_test, y_pred, y_pred_proba = nb_function(df,0.3, unbalanced = False, switch_model = False)
    print('WSS@85 for plain model',wss(y_test,y_pred, 0.85))
    print('WSS@95 for plain model',wss(y_test,y_pred, 0.95))


    #Next we try the balanced model 
    y_test, y_pred, y_pred_proba = nb_function(df,0.3, unbalanced = True, switch_model = False)
    print('WSS@85 for balanced model',wss(y_test,y_pred, 0.85))
    print('WSS@95 for balanced model',wss(y_test,y_pred, 0.95))

    #Now we try a different approach by using the complementNB , here we dont need balancing
    y_test, y_pred, y_pred_proba = nb_function(df,0.3, unbalanced = False, switch_model = True)
    print('WSS@85 for ComplementNB model',wss(y_test,y_pred, 0.85))
    print('WSS@95 for ComplementNB model',wss(y_test,y_pred, 0.95))



    #Doing the same for the virus dataset
    print('Results for Virus dataset')
    y_test2, y_pred2, y_pred_proba2 = nb_function(df2,0.3, unbalanced = False, switch_model = False)
    print('WSS@85 for plain model',wss(y_test2,y_pred2, 0.85))
    print('WSS@95 for plain model',wss(y_test2,y_pred2, 0.95))

    #Next we try the balanced model 
    y_test2, y_pred2, y_pred_proba2 = nb_function(df2,0.3, unbalanced = True, switch_model = False)
    print('WSS@85 for balanced model',wss(y_test2,y_pred2, 0.85))
    print('WSS@95 for balanced model',wss(y_test2,y_pred2, 0.95))

    #Now we try a different approach by using the complementNB , here we dont need balancing
    y_test2, y_pred2, y_pred_proba2 = nb_function(df2,0.3, unbalanced = False, switch_model = True)
    print('WSS@85 for ComplementNB model',wss(y_test2,y_pred2, 0.85))
    print('WSS@95 for ComplementNB model',wss(y_test2,y_pred2, 0.95))


    #Doing the same for the depression dataset
    print('Results for Depression dataset')
    y_test3, y_pred3, y_pred_proba3 = nb_function(df3,0.3, unbalanced = False, switch_model = False)
    print('WSS@85 for plain model',wss(y_test3,y_pred3, 0.85))
    print('WSS@95 for plain model',wss(y_test3,y_pred3, 0.95))

    #Next we try the balanced model 
    y_test3, y_pred3, y_pred_proba3 = nb_function(df3,0.3, unbalanced = True, switch_model = False)
    print('WSS@85 for balanced model',wss(y_test3,y_pred3, 0.85))
    print('WSS@95 for balanced model',wss(y_test3,y_pred3, 0.95))

    #Now we try a different approach by using the complementNB , here we dont need balancing
    y_test3, y_pred3, y_pred_proba3 = nb_function(df3,0.3, unbalanced = False, switch_model = True)
    print('WSS@85 for ComplementNB model',wss(y_test3,y_pred3, 0.85))
    print('WSS@95 for ComplementNB model',wss(y_test3,y_pred3, 0.95))


    results = {
        "Dataset": [],
        "Model": [],
        "WSS@85": [],
        "WSS@95": []
    }

    def store_results(dataset_name, model_name, wss85, wss95):
        """
        Store WSS results for a specific dataset and model configuration.
        """
        results["Dataset"].append(dataset_name)
        results["Model"].append(model_name)
        results["WSS@85"].append(wss85)
        results["WSS@95"].append(wss95)

    # Dataset 1: Calcium
    print('Results for Calcium dataset')
    y_test, y_pred, y_pred_proba = nb_function(df, 0.3, unbalanced=False, switch_model=False)
    store_results("Calcium", "Plain Model", wss(y_test, y_pred, 0.85), wss(y_test, y_pred, 0.95))

    y_test, y_pred, y_pred_proba = nb_function(df, 0.3, unbalanced=True, switch_model=False)
    store_results("Calcium", "Balanced Model", wss(y_test, y_pred, 0.85), wss(y_test, y_pred, 0.95))

    y_test, y_pred, y_pred_proba = nb_function(df, 0.3, unbalanced=False, switch_model=True)
    store_results("Calcium", "ComplementNB", wss(y_test, y_pred, 0.85), wss(y_test, y_pred, 0.95))

    # Dataset 2: Virus
    print('Results for Virus dataset')
    y_test2, y_pred2, y_pred_proba2 = nb_function(df2, 0.3, unbalanced=False, switch_model=False)
    store_results("Virus", "Plain Model", wss(y_test2, y_pred2, 0.85), wss(y_test2, y_pred2, 0.95))

    y_test2, y_pred2, y_pred_proba2 = nb_function(df2, 0.3, unbalanced=True, switch_model=False)
    store_results("Virus", "Balanced Model", wss(y_test2, y_pred2, 0.85), wss(y_test2, y_pred2, 0.95))

    y_test2, y_pred2, y_pred_proba2 = nb_function(df2, 0.3, unbalanced=False, switch_model=True)
    store_results("Virus", "ComplementNB", wss(y_test2, y_pred2, 0.85), wss(y_test2, y_pred2, 0.95))

    # Dataset 3: Depression
    print('Results for Depression dataset')
    y_test3, y_pred3, y_pred_proba3 = nb_function(df3, 0.3, unbalanced=False, switch_model=False)
    store_results("Depression", "Plain Model", wss(y_test3, y_pred3, 0.85), wss(y_test3, y_pred3, 0.95))

    y_test3, y_pred3, y_pred_proba3 = nb_function(df3, 0.3, unbalanced=True, switch_model=False)
    store_results("Depression", "Balanced Model", wss(y_test3, y_pred3, 0.85), wss(y_test3, y_pred3, 0.95))

    y_test3, y_pred3, y_pred_proba3 = nb_function(df3, 0.3, unbalanced=False, switch_model=True)
    store_results("Depression", "ComplementNB", wss(y_test3, y_pred3, 0.85), wss(y_test3, y_pred3, 0.95))

    # Convert results to DataFrame and save as CSV
    file_outpu_path = os.path.join(project_folder, "results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_outpu_path, index=False)
    logger.info(f"Results saved to {file_outpu_path}")










