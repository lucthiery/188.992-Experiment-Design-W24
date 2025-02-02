#!/usr/bin/env python3
"""
scores.py

This module provides sample functions to calculate performance scores used in systematic reviews:
  
  - RRF (Relevant References Found) using various sampling approaches:
      1. Ranking-based method: Screens the top percentage of documents based on prediction scores.
      2. Random Split method: Screens a random percentage of the documents.
      3. Stratified Sampling method: Screens a random percentage from each class (relevant and non-relevant)
         to preserve the original class distribution.
         
  - WSS (Work Saved over Sampling): Measures the reduction in screening workload based on a specified
    recall level. The formula used is:
    
      WSS = (TN + FN) / N - (1 - recall_level)
      
    where:
      - TN: True Negatives
      - FN: False Negatives
      - N: Total number of documents
      - recall_level: The target recall level (e.g., 0.85 for 85% recall)

"""

import numpy as np                              # Import numpy for numerical operations
from sklearn.metrics import confusion_matrix    # Import confusion_matrix for WSS computation


def rrf_ranking(y_true, y_scores, percentage=10):
    """
    Calculate the RRF (Relevant References Found) using a ranking-based approach.
    
    This function sorts documents by their prediction scores in descending order,
    screens the top 'percentage'% of the documents, and counts how many of them are relevant.
    
    Parameters:
        y_true (array-like): Array of true labels (1 for relevant, 0 for non-relevant).
        y_scores (array-like): Array of prediction scores or probabilities.
        percentage (float): The percentage of documents to screen (default is 10).
    
    Returns:
        int: Number of relevant documents found in the top 'percentage'% of the ranked list.
    """
    y_true = np.asarray(y_true)                      # Convert y_true to a numpy array
    y_scores = np.asarray(y_scores)                  # Convert y_scores to a numpy array
    N = len(y_true)                                  # Determine the total number of documents
    n_to_screen = int(np.ceil(N * percentage / 100)) # Calculate the number of documents to screen
    
    if n_to_screen <= 0:                             # Check if the calculated number is non-positive
        return 0                                   # Return 0 if no documents are to be screened
    
    # Use np.argpartition to efficiently select the indices of the top n_to_screen documents.
    # We negate y_scores because np.argpartition returns indices for ascending order.
    top_indices = np.argpartition(-y_scores, n_to_screen - 1)[:n_to_screen]
    
    # Count the number of relevant documents among the selected indices.
    rrf = int(np.sum(y_true[top_indices] == 1))
    return rrf                                     # Return the computed RRF value


def rrf_random(y_true, percentage=10, seed=None):
    """
    Calculate the RRF (Relevant References Found) using a random split approach.
    
    This function randomly selects a subset of documents corresponding to the given percentage,
    then counts how many of these randomly selected documents are relevant.
    
    Parameters:
        y_true (array-like): Array of true labels (1 for relevant, 0 for non-relevant).
        percentage (float): The percentage of documents to randomly sample (default is 10).
        seed (int, optional): Seed for reproducibility (default is None).
    
    Returns:
        int: Number of relevant documents found in the random sample.
    """
    y_true = np.asarray(y_true)                      # Convert y_true to a numpy array
    N = len(y_true)                                  # Get the total number of documents
    n_to_screen = int(np.ceil(N * percentage / 100)) # Calculate the number of documents to sample
    
    if seed is not None:                             # If a seed is provided,
        np.random.seed(seed)                         # set the random seed for reproducibility
    
    # Randomly select n_to_screen indices from the range [0, N) without replacement.
    random_indices = np.random.choice(np.arange(N), size=n_to_screen, replace=False)
    
    # Count the number of relevant documents in the randomly selected indices.
    rrf_random = int(np.sum(y_true[random_indices] == 1))
    return rrf_random                              # Return the RRF value for the random split approach


def rrf_stratified(y_true, percentage=10, seed=None):
    """
    Calculate the RRF (Relevant References Found) using a stratified sampling approach.
    
    This function performs stratified sampling by randomly selecting a percentage of documents
    from each class (relevant and non-relevant) to preserve the original class distribution,
    and then counts the number of relevant documents in the combined sample.
    
    Parameters:
        y_true (array-like): Array of true labels (1 for relevant, 0 for non-relevant).
        percentage (float): The percentage of documents to sample from each class (default is 10).
        seed (int, optional): Seed for reproducibility (default is None).
    
    Returns:
        int: Number of relevant documents found in the stratified sample.
    """
    y_true = np.asarray(y_true)                      # Convert y_true to a numpy array
    
    if seed is not None:                             # If a seed is provided,
        np.random.seed(seed)                         # set the random seed for reproducibility
    
    # Identify indices for relevant documents (where y_true equals 1)
    indices_relevant = np.where(y_true == 1)[0]
    # Identify indices for non-relevant documents (where y_true equals 0)
    indices_irrelevant = np.where(y_true == 0)[0]
    
    n_relevant = len(indices_relevant)               # Count the number of relevant documents
    n_irrelevant = len(indices_irrelevant)           # Count the number of non-relevant documents
    
    # Calculate the number of samples to draw from each class based on the specified percentage.
    n_relevant_to_sample = int(np.ceil(n_relevant * percentage / 100))
    n_irrelevant_to_sample = int(np.ceil(n_irrelevant * percentage / 100))
    
    # Ensure that the sample size does not exceed the available documents in each class.
    n_relevant_to_sample = min(n_relevant_to_sample, n_relevant)
    n_irrelevant_to_sample = min(n_irrelevant_to_sample, n_irrelevant)
    
    # Randomly sample indices from the relevant documents if any are available.
    sampled_relevant = (np.random.choice(indices_relevant, size=n_relevant_to_sample, replace=False)
                        if n_relevant > 0 else np.array([], dtype=int))
    
    # Randomly sample indices from the non-relevant documents if any are available.
    sampled_irrelevant = (np.random.choice(indices_irrelevant, size=n_irrelevant_to_sample, replace=False)
                          if n_irrelevant > 0 else np.array([], dtype=int))
    
    # Combine the sampled indices from both classes.
    sampled_indices = np.concatenate((sampled_relevant, sampled_irrelevant))
    
    # Count the number of relevant documents in the combined sample.
    rrf_stratified = int(np.sum(y_true[sampled_indices] == 1))
    return rrf_stratified                          # Return the computed RRF value for stratified sampling


def wss(y_true, y_pred, recall_level):
    """
    Calculate the WSS (Work Saved over Sampling) score using a confusion matrix.
    
    The WSS score is computed as:
    
        WSS = (TN + FN) / N - (1 - recall_level)
    
    where:
        - TN: Number of true negatives.
        - FN: Number of false negatives.
        - N: Total number of documents.
        - recall_level: The target recall level (e.g., 0.85 for 85% recall).
    
    This metric quantifies the proportion of documents that are not screened (i.e., work saved)
    relative to the desired recall level.
    
    Parameters:
        y_true (array-like): Array of true labels (1 for relevant, 0 for non-relevant).
        y_pred (array-like): Array of predicted binary labels (1 for predicted relevant, 0 for predicted non-relevant).
        recall_level (float): The target recall level (expressed as a decimal, e.g., 0.85 for 85% recall).
    
    Returns:
        float: The calculated WSS score.
    """
    y_true = np.asarray(y_true)                      # Convert y_true to a numpy array
    y_pred = np.asarray(y_pred)                      # Convert y_pred to a numpy array
    n = len(y_true)                                  # Determine the total number of documents
    
    # Compute the confusion matrix and extract the TN, FP, FN, and TP values.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compute the WSS score using the formula: (TN + FN) / n - (1 - recall_level)
    wss_score = (tn + fn) / n - (1 - recall_level)
    return wss_score                               # Return the computed WSS score


if __name__ == "__main__":
    # Example usage of the scoring functions for demonstration purposes.
    
    # Example true labels: 1 indicates a relevant document, 0 indicates a non-relevant document.
    y_true_example = [1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
    
    # Example prediction scores (e.g., probabilities from a classifier).
    y_scores_example = [0.95, 0.10, 0.80, 0.40, 0.30, 0.85, 0.70, 0.20, 0.15, 0.90]
    
    # Generate binary predictions based on a threshold (e.g., 0.5) for demonstration.
    y_pred_example = [1 if score >= 0.5 else 0 for score in y_scores_example]
    
    percentage = 10           # Percentage of documents to screen for RRF calculations (10%).
    recall_level_85 = 0.85    # Target recall level (85% recall).
    recall_level_95 = 0.95    # Target recall level (95% recall).
    """
    # Calculate RRF using the ranking-based method.
    rrf_ranking = calculate_rrf_by_ranking(y_true_example, y_scores_example, percentage=percentage)
    # Calculate RRF using the random split method (with a fixed seed for reproducibility).
    rrf_random = calculate_rrf_random(y_true_example, percentage=percentage, seed=42)
    # Calculate RRF using the stratified sampling method (with a fixed seed for reproducibility).
    rrf_stratified = calculate_rrf_stratified(y_true_example, percentage=percentage, seed=42)
    
    # Calculate WSS using the true labels, binary predictions, and target recall levels.
    wss_value_85 = calculate_wss(y_true_example, y_pred_example, recall_level=recall_level_85)
    wss_value_95 = calculate_wss(y_true_example, y_pred_example, recall_level=recall_level_95)
    
    # Print the calculated scores for demonstration.
    print("Score Calculations:")
    print(f"RRF (Ranking-based, {percentage}% screened): {rrf_ranking} relevant documents found")
    print(f"RRF (Random Split, {percentage}% screened): {rrf_random} relevant documents found")
    print(f"RRF (Stratified Sampling, {percentage}% screened): {rrf_stratified} relevant documents found")
    print(f"WSS (Work Saved over Sampling, {int(recall_level_85*100)}% recall): {wss_value_85:.4f}")
    print(f"WSS (Work Saved over Sampling, {int(recall_level_95*100)}% recall): {wss_value_95:.4f}")
"""