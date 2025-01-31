#Calculate the preformance metrices 

#1. WSS 

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def wss(y_true, y_pred, recall_level): 
    
    #we need the length for the formula (can either be from pred or true, should be equal)
    n = len(y_true)
    #compute the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    wss = (tn + fn) / n - (1 - recall_level)
    return(wss) 


    
def calculate_rrf(y_true, y_scores, percentage=10):
    # Calculate number of documents to screen
    N = len(y_true)
    n_to_screen = int(np.ceil(N * percentage / 100))
    
    # Sort by prediction scores
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Count relevant documents in the screened portion
    relevant_in_screen = np.sum(y_true_sorted[:n_to_screen] == 1)
    
    total_relevant = np.sum(y_true == 1)
    
    # Calculate RRF (avoid division by zero)
    if total_relevant > 0:
        RRF = relevant_in_screen / total_relevant
    else:
        RRF = 0.0
    
    return RRF

    
    
    
