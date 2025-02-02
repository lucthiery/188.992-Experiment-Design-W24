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

## Metrics

In our project, we evaluate the performance of our systematic review pipeline using two main metrics: **Work Saved over Sampling (WSS)** and **Relevant References Found (RRF)**.

### Work Saved over Sampling (WSS)

WSS quantifies the reduction in screening workload achieved by the system relative to random sampling. It is defined as:

$$
\text{WSS} = \frac{TN + FN}{N} - (1 - \text{recall\_level})
$$

where:

- **TN**: Number of true negatives (non-relevant documents correctly identified as such)
- **FN**: Number of false negatives (relevant documents that were missed)
- **N**: Total number of documents
- **recall\_level**: The target recall level (e.g., 0.85 for 85% recall)

This metric measures how much work (i.e., document screening) is saved at a given recall target compared to screening the entire dataset.

### Relevant References Found (RRF)

RRF is intended to measure the number of relevant documents identified when a fixed percentage of the unlabeled dataset is screened. The reference paper [4] provides the following description:

> "RRF@10 evaluates how many relevant documents can be identified when 10% of the unlabeled documents have been screened."

However, the paper did not provide a sufficiently detailed definition for RRF, and our attempts to compute RRF highlighted a few challenges:

- The formula (1) for WSS in [4] does not match the reported values (differences are never exactly 0.1).
- For RRF, the description lacks details on how to split or sort the data. For example, using a ranking based on prediction scores versus a random or stratified split can lead to significantly different results.

#### Our Approach to RRF

Due to these ambiguities, we implemented three different approaches to approximate RRF in our project (see the `scores.py` module):

1. **Ranking-based RRF:**  
   - The documents are sorted in descending order by their prediction scores.
   - The top *X*% of documents are screened.
   - The RRF value is the count of relevant documents within this screened set.

2. **Random Split RRF:**  
   - A random sample corresponding to *X*% of the dataset is selected.
   - The number of relevant documents in this random subset is counted.

3. **Stratified Sampling RRF:**  
   - A stratified random sample (preserving the original class distribution) is drawn.
   - The number of relevant documents in the stratified sample is computed.

Because the reference paper does not specify the exact methodology, reproducing RRF accurately is challenging. Consequently, the values from our implementations might differ from those reported in SciMine.

### Implementation Details

The code for computing these metrics is provided in the `scores.py` module. The module includes the following functions:

- `calculate_rrf_by_ranking(y_true, y_scores, percentage=10)`
- `calculate_rrf_random(y_true, percentage=10, seed=None)`
- `calculate_rrf_stratified(y_true, percentage=10, seed=None)`
- `calculate_wss(y_true, y_pred, recall_level)`

## Pipeline Overview

The main pipeline (`main.py`) orchestrates the following steps:

### Metadata Retrieval

A key component of our project is the metadata retrieval pipeline. It is implemented as follows:

1. **PubMed API Client:** Fetches article metadata (title and abstract) using the PubMed identifier (PMID).  
2. **OpenAlex API Client:** Acts as the first fallback when PubMed returns default values.  
3. **CrossRef API Client:** Uses the DOI as a final fallback to retrieve metadata when both PubMed and OpenAlex fail.  

These API clients are implemented in the `api/` directory:

- `pubmed_client.py`
- `openalex_client.py`
- `crossref_client.py` (newly implemented with extensive documentation)

The fallback logic is defined in the `utils/data_utils.py` file, where asynchronous tasks are created for each dataset row.  The cascading fallback mechanism (PubMed → OpenAlex → CrossRef) in `utils/data_utils.py` to enrich the data with article metadata. It fetches the metadata such as title and abstract from the given links in the datasets, that can be found in `Data/original`

## Project Structure

```text
|-- Data
|   |-- original
|   |   |-- Archiv
|   |   |   `-- Nagtegaal_2019_ids.csv
|   |   |-- Bannach-Brown_2019_ids.csv
|   |   |-- Cohen_2006_CalciumChannelBlockers_ids.csv
|   |   `-- Kwok_2020_ids.csv
|   `-- preprocessed
|       |-- calcium_preprocessed.csv
|       |-- depression_preprocessed.csv
|       `-- virus_preprocessed.csv
|-- README.md
|-- api
|   |-- base_client.py
|   |-- crossref_client.py
|   |-- openalex_client.py
|   `-- pubmed_client.py
|-- discarded\ approaches
|   |-- 4.2.R
|   |-- comparison_results.py
|   |-- phrase.py
|   `-- representation_learning.py
|-- docs
|   `-- python_env_setup.md
|-- main.py
|-- models
|   |-- alternative_nb_with_cv.py
|   |-- d2v+svm.R
|   `-- naive_bayes_function.py
|-- requirements.R
|-- requirements.txt
|-- results.csv
`-- utils
    |-- READme
    |-- data_utils.py
    |-- evaluation_metrics.py
    `-- logger.py

```
