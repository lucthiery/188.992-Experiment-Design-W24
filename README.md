# 188.992-Experiment-Design-W24

## Resources

188.992 Experiment Design for Data Science (VU 2,0) 2024W

- **Link to published Paper ond Zenodo:** [SciMine: An Efficient Systematic Prioritization Model Based on Richer Semantic Information - Experiment Design Report Group 22](https://zenodo.org/me/requests/e9dad706-0f2e-4603-9018-0e25dcfa4967)
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

The fallback logic is defined in the `utils/data_utils.py` file, where asynchronous tasks are created for each dataset row.  The cascading fallback mechanism (PubMed → OpenAlex → CrossRef) in `utils/data_utils.py` to enrich the data with article metadata. It fetches the metadata such as title and abstract from the given links in the datasets, that can be found in `Data/original`.

However, the metadata retrieval will create the preprocessed csv files in the folder `DATA\preprocessed`. However. the filename will start with the timestamp. So you need to rename the filenames manually after executing the preprocessing step in order to use the latest preprocessed data.

## Baseline Recreation

We decided to recreate two of the given baslines, the Naive Bayes model with TF-IDF transformation and the SVM with doc2vec embeddings. You can find further information in the folder models in the files "naive_bayes_function.py" and "alternative_nb_with_cv.py" for Naive Bayes and in "d2v+svm.R" for SVM.

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

## Setting Up the Python Environment and Installing Dependencies

This guide explains how to set up a Python environment using `venv` and install the required dependencies from the `requirements.txt` file.

### Prerequisites

- Python 3.11 must be installed on your system.
- Ensure `pip` is installed and accessible.

## Steps to Set Up the Environment

1. **Create a Virtual Environment:**

   Open a terminal in the root directory of the project and run the following command to create a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Replace `name_of_virtual_environment` with your desired name for the environment.

2. **Activate the Virtual Environment:**
   - On **Windows**:

     ```bash
     name_of_virtual_environment\Scripts\activate
     ```

   - On **macOS/Linux**:

     ```bash
     source name_of_virtual_environment/bin/activate
     ```

3. **Upgrade `pip`:**
   It is recommended to upgrade `pip` to the latest version before installing dependencies:

   ```bash
   pip install --upgrade pip
   ```

4. **Install PythonDependencies:**
   Use the following command to install the required packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

5. However, as some code runs in R. You also have to install R and make sure the following dependencies are installed:

   5.1 **R Version**: 4.3.1
      - Ensure that you have R version 4.3.1 installed. You can download R from the official [R website](https://cran.r-project.org/).

   5.2 **Required Libraries**:
      - **`text2vec`** (version 0.6.4) for text embeddings, including doc2vec features.
        Install it using:

        ```R
        install.packages("text2vec", repos = "https://cloud.r-project.org/")
        ```

      - **`e1071`** (version 1.7-14) for Support Vector Machines (SVM) implementation with a linear kernel.
        Install it using:

        ```R
        install.packages("e1071")
        ```

      - **`caTools`** for splitting the dataset into training and test sets.
        Install it using:

        ```R
        install.packages("caTools")
        ```

      - **`tidyr`** for data manipulation, including functions to handle missing values and reshape the data.
        Install it using:

        ```R
        install.packages("tidyr")
        ```

   5.4. **GloVe Model for Word Embeddings**:
      - A GloVe model must be constructed to generate word embeddings. The process involves averaging word vectors to create document embeddings. You can download GloVe from the [GloVe website](https://nlp.stanford.edu/projects/glove/) and use it in your project.

   `IMPORTANT NOTE:` However we also provided an installation scirpt for the R modules that can be found in requirements.R and which is executed during the execution of main.py. However, it might not working under your dev environment and you might have to install the R modules manually by using the provided documentation above.

6. **Verify Installation:**
   Check if the necessary packages are installed correctly by running:

   ```bash
   pip list
   ```

   This will display all installed packages and their versions.

7. **Run the code:**

   The code is designed to run automatically, be executing the `main.py` file in the `root directory`.

   To do so, you can run the following command in your cli, assuming you are working with a Linux distribution. For Windows please check the python documentation on how to run python code on Windows.

   ```text
   python3 main.py
   ````

   When executing this code, make sure you are within the project directory. Otherwise specify the path.

   **`IMPORTANT NOTE:`** In order to run the code, you also need to have installed R with Version 4.4 or higher, als the main.py is also running a subprocess for executing R code. In R make sure, you have the following libraries installed, as referenced in our report paper in section 3.3.

8. **Deactivate the Virtual Environment:**

   Once you have completed your work, deactivate the virtual environment by running:

   ```bash
   deactivate
   ```

## Notes

- Always activate the virtual environment before working on the project to ensure you use the correct Python environment.
- If new dependencies are added to the project, update the `requirements.txt` file using:

  ```bash
  pip freeze > requirements.txt
  ```

This will regenerate the `requirements.txt` file with all currently installed packages.

By following these steps, you can ensure a consistent and isolated environment for your project.
