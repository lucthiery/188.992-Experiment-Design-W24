#Experiment Design Project 2 
#Fetching metadata


import pandas as pd 
import requests 
from xml.etree import ElementTree as ET

calcium = pd.read_csv('Cohen_2006_CalciumChannelBlockers_ids.csv')
depression = pd.read_csv('Bannach-Brown_2019_ids.csv')
virus = pd.read_csv('Kwok_2020_ids.csv')

print('Calcium missing values')
print(calcium.isna().sum())
print('Depression missing values')
print(depression.isna().sum())
print('Virus missing values')
print(virus.isna().sum())

#starting with extracting data from Pubmed using pmid
#use site https://www.kaggle.com/code/binitagiri/extract-data-from-pubmed-using-python
#needs first a pip install metapub 

#fetch = PubMedFetcher()
#articles = {}
#for pmid in calcium['pmid']:
#    articles[pmid] = fetch.article_by_pmid(pmid)

#print('done')

def fetch_pubmed_data(pmid):
    #get base url for the Pubmed site 
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    xml_data = response.text

    # Parse the XML to extract title and abstract
    root = ET.fromstring(xml_data)

    title = root.findtext(".//ArticleTitle", default="No title")
    abstract = " ".join(
        elem.text for elem in root.findall(".//AbstractText") if elem.text
        ) or "No abstract"

    return title, abstract


def append_metadata(df, column): 
    titles = []
    abstracts = []
    for id in df[column]: 
        title, abstract = fetch_pubmed_data(str(id))
        titles.append(title)
        abstracts.append(abstract)
    df['title'] = titles
    df['abstracts'] = abstracts
    
    return df 

#test_data = calcium[2:10]

calcium_preprocessed = append_metadata(calcium, 'pmid') 

print(calcium_preprocessed[calcium_preprocessed['abstracts'] == 'No abstract'].count())


#vorläufiges Speichern 
calcium_preprocessed.to_csv('calcium_preprocessed.csv')

#TO DO: 
# ADD Procedure for ID for openalex and OID with if condition (if pmid not exists use oid or openalexid)
# Create the other datasets virus and depression 

#extracting metadata from openalex  by using pip install pyalex package
#see https://github.com/J535D165/pyalex for mor information 






