# 188.992-Experiment-Design-W24
188.992 Experiment Design for Data Science (VU 2,0) 2024W

Overleaf Document:
https://www.overleaf.com/project/67754981243b583663860790


Project(SciMine)/

├── main.py              # main pipeline for step by step use of functions

├── data/                # Datasets and preprocessed data

├── models/              # Machine learning models

├── utils/               # Helper functions and utilities

├── experiments/         # Scripts to run experiments and evaluate results

├── results/             # Output results and logs

├── notebooks/           # Jupyter notebooks for testing/debugging

├── requirements.txt     # Python dependencies

├── README.md            # Project overview and setup instructions


main.py - pipeline:
1. Load Data: utils/data_preparation.py to preprocess and split datasets.
2. Learn Representations: models/representation_learning.py to generate document and phrase embeddings.
3. Train Models:
   
    3.1 models/document_classifier.py for the VAE document-level classifier.

    3.2 models/phrase_classifier.py for the phrase-level Random Forest classifier.
   
5. Rank and Ensemble: experiments/ranking_and_evaluation.py for ranking and combining predictions.
6. Evaluate: Compute evaluation metrics and log results.
7. Log and Save Outputs: Save intermediate results (e.g., embeddings, rankings) to results/ and store final evaluation reports and logs.


