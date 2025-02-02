import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from glove import Corpus, Glove

# Stelle sicher, dass die benötigten NLTK-Ressourcen vorhanden sind (z. B. Punkt)
nltk.download('punkt')

# Funktion zur Tokenisierung: Text in Kleinbuchstaben umwandeln und tokenisieren
def tokenize_text(text):
    return word_tokenize(text.lower())

# 1. Daten einlesen und Spalten auswählen (analog zu R: read.csv2 und subset)
calcium = pd.read_csv("calcium_preprocessed.csv", sep=",")
calcium = calcium[['title', 'label_included']]  # Entsprechend R: select("title", "label_included")
data = calcium

# 2. Train/Test-Split
# Wir können stratify=... verwenden, um die Labelverteilung zu erhalten
train_data, test_data = train_test_split(data, test_size=0.3, random_state=12345, stratify=data['label_included'])

# 3. Tokenisierung der Texte (hier: Spalte 'title')
train_tokens = train_data['title'].apply(tokenize_text).tolist()
test_tokens = test_data['title'].apply(tokenize_text).tolist()

# 4. Erzeuge Korpus und Vokabular sowie Term-Co-occurrence-Matrix mit glove-python
corpus = Corpus()  # Initialisiere ein Corpus-Objekt
# Das Corpus erwartet eine Liste von Dokumenten (Liste von Tokens)
# Wir setzen hier window=5 analog zu skip_grams_window=5 in R.
corpus.fit(train_tokens, window=5)

# 5. GloVe-Modell trainieren
# Wir wählen hier no_components=80 (entspricht rank=80) und x_max=20, n_iter=20 (entspricht n_iter=20)
glove = Glove(no_components=80, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)  # Fügt das Vokabular dem Modell hinzu

# 6. Funktion zur Erstellung von Dokument-Embeddings
def create_document_embeddings(docs_tokens, glove_model):
    embeddings = []
    for doc in docs_tokens:
        # Für jedes Wort prüfen, ob es im GloVe-Vokabular vorhanden ist
        valid_vectors = []
        for word in doc:
            if word in glove_model.dictionary:
                idx = glove_model.dictionary[word]
                valid_vectors.append(glove_model.word_vectors[idx])
        if valid_vectors:
            # Mittelwert der Wortvektoren bilden
            doc_embedding = np.mean(valid_vectors, axis=0)
        else:
            # Falls keine gültigen Wörter vorhanden sind, ein Null-Vektor
            doc_embedding = np.zeros(glove_model.no_components)
        embeddings.append(doc_embedding)
    return np.vstack(embeddings)

# Erstelle Dokument-Embeddings für Trainings- und Testdaten
train_embeddings = create_document_embeddings(train_tokens, glove)
test_embeddings = create_document_embeddings(test_tokens, glove)

print(f"Train Embedding Dimensions: {train_embeddings.shape[0]} x {train_embeddings.shape[1]}")
print(f"Test Embedding Dimensions: {test_embeddings.shape[0]} x {test_embeddings.shape[1]}")

# 7. SVM-Klassifikator trainieren
# Wir verwenden einen linearen Kernel, C=1 und skalieren die Features vorher.
# Skaliere die Daten (optional, da SVC oft von standardisierten Daten profitiert)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)

# Erstelle den SVM-Klassifikator
svm_model = SVC(kernel='linear', C=1, random_state=12345)
# Beachte: In R wurden die Labels als Faktor verwendet. In Python gehen wir davon aus,
# dass label_included bereits in einem passenden Format (z.B. 0/1 oder string) vorliegt.
svm_model.fit(train_embeddings_scaled, train_data['label_included'])

# 8. Vorhersagen auf Testdaten
test_predictions = svm_model.predict(test_embeddings_scaled)

# 9. Evaluation: Konfusionsmatrix und Accuracy
conf_matrix = confusion_matrix(test_data['label_included'], test_predictions)
accuracy = accuracy_score(test_data['label_included'], test_predictions)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 10. Weitere Rechnungen analog zu den R-Berechnungen

# Für "calcium" (entspricht R: WSS <- 335/(335+30)-0.05, WSS85 <- 335/(335+30)-0.15)
WSS = 335 / (335 + 30) - 0.05
WSS85 = 335 / (335 + 30) - 0.15
print(f"WSS (calcium): {WSS}")
print(f"WSS85 (calcium): {WSS85}")

# Virus-Daten
virus = pd.read_csv("virus_preprocessed.csv", sep=",")
virus = virus[['titles', 'label_included']]
# Die R-Formeln:
WSS95_v = 744 / (744 + 36) - 0.05
WSS85_v = 744 / (744 + 36) - 0.15
print(f"WSS95 (virus): {WSS95_v}")
print(f"WSS85 (virus): {WSS85_v}")

# Depression-Daten
depression = pd.read_csv("depression_preprocessed.csv", sep=",")
depression = depression[['titles', 'label_included']]
WSS95_d = (488 + 26) / (488 + 26 + 61 + 23) - 0.05
WSS85_d = (488 + 26) / (488 + 26 + 61 + 23) - 0.15
print(f"WSS95 (depression): {WSS95_d}")
print(f"WSS85 (depression): {WSS85_d}")
