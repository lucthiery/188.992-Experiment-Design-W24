from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd


def generate_document_embeddings(data, model_name="allenai/specter2_base"):
    """
    Generate document-level embeddings using the SPECTER2 model.

    Args:
        data (pd.DataFrame): Dataset containing 'title' and 'abstracts'.
        model_name (str): Hugging Face model name for SPECTER2.

    Returns:
        pd.DataFrame: Updated DataFrame with 'doc_embedding' column.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Combine title and abstract
    text_batch = (data['title'] + tokenizer.sep_token + data['abstracts']).tolist()

    embeddings = []
    for text in text_batch:
        # Tokenize the input text
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

        # Generate embeddings with the model
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
            embeddings.append(cls_embedding.squeeze().numpy())

    # Add embeddings to the DataFrame
    data['doc_embedding'] = embeddings
    return data






# Example usage
if __name__ == "__main__":
    # Example usage
    filepath = "../data/preprocessed/calcium_preprocessed.csv"
    data = pd.read_csv(filepath)

    # Generate document embeddings
    data_with_doc_embeddings = generate_document_embeddings(data)
    print("Document embeddings generated.")

    # Save the output
    data_with_doc_embeddings.to_csv("../data/preprocessed/representation_embeddings.csv", index=False)
    print("Embeddings saved to '../Data/embeddings/representation_embeddings.csv'.")


