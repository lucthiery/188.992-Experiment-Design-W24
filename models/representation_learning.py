import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_document_embeddings(data, model_name="allenai/specter"):
    """
    Generate document-level embeddings for each row in the dataset.
    
    Args:
        data (pd.DataFrame): Preprocessed dataset containing 'title' and 'abstracts' columns.
        model_name (str): Name of the pre-trained model to use (default: SPECTER).
        
    Returns:
        pd.DataFrame: DataFrame with added 'doc_embedding' column containing embeddings.
    """
    model = SentenceTransformer(model_name)
    
    # Combine title and abstracts for embedding
    documents = data['title'] + " " + data['abstracts']
    embeddings = model.encode(documents.tolist(), show_progress_bar=True)
    
    # Add embeddings to the DataFrame
    data['doc_embedding'] = embeddings.tolist()
    return data

def generate_phrase_embeddings(data, model_name="allenai/scibert_scivocab_uncased"):
    """
    Generate phrase-level embeddings by mining and encoding key phrases in abstracts.
    
    Args:
        data (pd.DataFrame): Dataset containing 'abstracts' column.
        model_name (str): Name of the pre-trained model to use (default: SciBERT).
        
    Returns:
        pd.DataFrame: DataFrame with added 'phrase_embeddings' column containing embeddings.
    """
    model = SentenceTransformer(model_name)
    
    # Placeholder for phrase-level embeddings
    phrase_embeddings = []
    
    for abstract in data['abstracts']:
        # Here you can use a phrase-mining tool like AutoPhrase to extract phrases
        phrases = extract_phrases(abstract)  # Placeholder for phrase mining
        if phrases:
            embeddings = model.encode(phrases, show_progress_bar=False)
            phrase_embeddings.append(embeddings)
        else:
            phrase_embeddings.append([])
    
    data['phrase_embeddings'] = phrase_embeddings
    return data

def extract_phrases(text):
    """
    Extract key phrases from the given text (placeholder function).
    Replace with an actual implementation (e.g., using AutoPhrase).
    
    Args:
        text (str): Input text (abstract).
        
    Returns:
        list: Extracted key phrases.
    """
    # Placeholder: Split text into "phrases" (this should use an actual phrase-mining algorithm)
    return text.split(". ")

if __name__ == "__main__":
    # Example usage
    filepath = "data/calcium_preprocessed.csv"
    data = pd.read_csv(filepath)
    
    # Generate document embeddings
    data_with_doc_embeddings = generate_document_embeddings(data)
    print("Document embeddings generated.")
    
    # Generate phrase embeddings
    data_with_phrase_embeddings = generate_phrase_embeddings(data_with_doc_embeddings)
    print("Phrase embeddings generated.")
    
    # Save the output
    data_with_phrase_embeddings.to_csv("data/representation_embeddings.csv", index=False)
    print("Embeddings saved to 'data/representation_embeddings.csv'.")
