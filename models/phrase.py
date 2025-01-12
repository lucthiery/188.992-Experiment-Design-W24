from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import os


def extract_phrases(texts, max_phrases=10):
    """
    Extract key phrases from a list of texts using a simple split-based method.
    Replace this placeholder with a real keyphrase extraction method for better results.

    Args:
        texts (list): A list of strings (abstracts).
        max_phrases (int): Maximum number of phrases to extract per text.

    Returns:
        list: A list of lists, where each sublist contains the extracted phrases for a text.
    """
    phrases_list = []
    for text in texts:
        # Placeholder: Split by periods (.) and treat each as a phrase
        phrases = text.split(". ")
        phrases_list.append(phrases[:max_phrases])  # Limit to max_phrases
    return phrases_list


def generate_phrase_embeddings(data, model_name="allenai/scibert_scivocab_uncased", max_phrases=10):
    """
    Generate phrase-level embeddings using SciBERT.

    Args:
        data (pd.DataFrame): Dataset with 'abstracts' column.
        model_name (str): Hugging Face model name for SciBERT.
        max_phrases (int): Maximum number of phrases to extract and embed.

    Returns:
        pd.DataFrame: Updated DataFrame with 'phrases' and 'phrase_embeddings' columns.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Extract phrases from abstracts
    data['phrases'] = extract_phrases(data['abstracts'].tolist(), max_phrases=max_phrases)

    # Generate embeddings for phrases
    phrase_embeddings = []
    for phrases in data['phrases']:
        embeddings = []
        for phrase in phrases:
            # Tokenize and embed each phrase
            inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
                embeddings.append(cls_embedding.squeeze().numpy())
        phrase_embeddings.append(embeddings)

    # Add phrase embeddings to the DataFrame
    data['phrase_embeddings'] = phrase_embeddings
    return data


if __name__ == "__main__":
    # Example usage
    filepath = "../data/preprocessed/calcium_preprocessed.csv"
    #filepath = "/Users/lucthiery/Desktop/1_UNI/TU/188.992 Experiment Design for Data Science/EX-2/188.992-Experiment-Design-W24/Data/preprocessed/calcium_preprocessed.csv"
    data = pd.read_csv(filepath)

    # Generate phrase embeddings
    data_with_phrase_embeddings = generate_phrase_embeddings(data, max_phrases=10)
    print("Phrase embeddings generated.")

    # Save the resulting DataFrame
    #data_with_phrase_embeddings.to_csv("/Users/lucthiery/Desktop/1_UNI/TU/188.992 Experiment Design for Data Science/EX-2/188.992-Experiment-Design-W24/Data/embeddings/phrase_embeddings.csv", index=False)
    data_with_phrase_embeddings.to_csv(
        "../Data/embeddings/phrase_embeddings.csv",
        index=False)

    print("Phrase embeddings saved to '../Data/preprocessed/phrase_embeddings.csv'.")

