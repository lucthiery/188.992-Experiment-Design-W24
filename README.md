# 188.992-Experiment-Design-W24

## Resources

188.992 Experiment Design for Data Science (VU 2,0) 2024W

- **Project Documentation on Overleaf:** [Overleaf Document](https://www.overleaf.com/project/67754981243b583663860790)
- **Setup Guide:** [Setting Up the Python Environment and Installing Dependencies](docs/python_env_setup.md)
- **SciMine Paper:** [SciMine: An Efficient Systematic Prioritization Model Based on Richer Semantic Information](https://dl.acm.org/doi/10.1145/3539618.3591764)
- **WSS Metric definition:** [An analysis of work saved over sampling in the evaluation of automated citation screening in systematic literature reviews](https://www.sciencedirect.com/science/article/pii/S2667305323000182?ref=pdf_download&fr=RR-2&rr=90b1528aca755ae8)

## Team Members

This repository is for the course **188.992 Experiment Design for Data Science (VU 2,0) 2024W**. It includes code, data, and documentation for implementing and analyzing experimental design concepts in data science.
The following individuals contributed to the project:

- **Theresa Brucker**
- **Stefanie Gröger**
- **Luc Thiery**
- **Felix Kapfer**

## Overview

This repository aims to reproduce the phrase-level feature classification from the SciMine framework. The project focuses on extracting and clustering semantically relevant phrases from scientific literature to enhance document ranking. The pipeline includes data preprocessing, phrase embedding generation, clustering with Louvain, and classification using Random Forest. Current challenges include embedding size optimization and integration with document-level classification. Future steps involve ranking refinement and model evaluation.

## Goal

The primary goal of this project is to reproduce the SciMine framework. However, we encountered significant challenges due to missing resources. The linked GitHub repository, which was expected to contain the datasets and implementation code, was empty, making the replication process difficult from the start.

Additionally, the datasets used in the study were not provided, leaving us without the necessary data to train and evaluate the models. The implementation details were also not specified, including the programming environment, package dependencies, and their corresponding versions.

Furthermore, key aspects related to the baseline experiments were missing, such as:

The seed value used for reproducibility
Details on the train/test split
The resampling method applied
Due to these limitations and lack of documentation, fully reproducing SciMine as described in the paper has proven to be extremely challenging.

## Datasets

The SciMine paper references five datasets used for systematic review screening across different research domains:

- Calcium – Focuses on calcium channel blockers in medical research.
- Nudging – Examines the impact of behavioral nudging on healthcare professionals.
- Depression – Contains preclinical studies on depression in non-human animals.
- Virus – Investigates viral metagenomic sequencing in livestock.
- AgriDiv – A newly created dataset analyzing agricultural diversification in rice production.

These datasets can be found within the Synergy dataset using the following link: [Synergy Dataset GitHub](https://github.com/asreview/synergy-dataset)

## Project Structure
// TODO - Not done yet
```text
├── Data
│   ├── original
│   │   ├── Bannach-Brown_2019_ids.csv
│   │   ├── Cohen_2006_CalciumChannelBlockers_ids.csv
│   │   ├── Nagtegaal_2019_ids.csv
│   │   └── Kwok_2020_ids.csv
│   ├── preprocessed
│   │   ├── 20250201_102214_Kwok_2020_ids.csv
│   │   └── calcium_preprocessed.csv
│   └── representation_embeddings.csv
├── README.md
├── api
│   ├── base_client.py
│   ├── openalex_client.py
│   └── pubmed_client.py
├── docs
│   └── python_env_setup.md
├── main.py
├── models
│   ├── comparison_results.py
│   ├── glove_svm_classifier.py
│   ├── phrase.py
│   └── representation_learning.py
├── requirements.txt
├── tmp
│   └── tmp.py
└── utils
    ├── READme
    ├── data_preprocessing.py
    ├── data_utils.py
    └── logger.py

```















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
   
4. Rank and Ensemble: experiments/ranking_and_evaluation.py for ranking and combining predictions.
5. Evaluate: Compute evaluation metrics and log results.
6. Log and Save Outputs: Save intermediate results (e.g., embeddings, rankings) to results/ and store final evaluation reports and logs.


1. Loading and Preprocessing -- Theresa
2. Specter und Scibert -- Luc
3.  Variational Aoutoencoder 4.2 -- Steffi
4.  4.3 Classifer -- Felix
5.  Results von anderen Methoden nachbauen 

