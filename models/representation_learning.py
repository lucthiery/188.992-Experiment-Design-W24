from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
import torch
import pandas as pd


def generate_document_embeddings(data, model_name="allenai/specter2_base"):
    """
    Generate document-level embeddings using the SPECTER model.

    Args:
        data (pd.DataFrame): Preprocessed dataset with 'title' and 'abstracts'.
        model_name (str): Hugging Face model name for SPECTER.

    Returns:
        pd.DataFrame: Updated DataFrame with 'doc_embedding'.
    """
    # Load SPECTER model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Load and activate the SPECTER2 adapter
    print("start load model")
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    print("loaded model")
    # Combine title and abstract into a single text field
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
    data_with_doc_embeddings.to_csv("../Data/embeddings/representation_embeddings.csv", index=False)
    print("Embeddings saved to '../Data/embeddings/representation_embeddings.csv'.")
