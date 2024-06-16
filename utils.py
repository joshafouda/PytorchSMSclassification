from torchtext.data.utils import get_tokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import re

# Assurez-vous que les ressources nécessaires sont téléchargées
import nltk
nltk.download('stopwords')

def preprocess_texts(texts, threshold=1):
    tokenizer = get_tokenizer("basic_english")
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def tokenize_and_clean(text):
        # Tokenisation
        tokens = tokenizer(text)
        # Suppression des tokens contenant des ponctuations et des éléments non alphanumériques
        tokens = [token for token in tokens if re.match(r'^[a-zA-Z0-9]+$', token)]
        # Suppression des mots vides
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def stem_tokens(tokens):
        return [stemmer.stem(token) for token in tokens]

    # Tokenisation, nettoyage et stemming
    processed_texts = [stem_tokens(tokenize_and_clean(text)) for text in texts]

    # Calcul de la fréquence des tokens pour suppression des mots rares
    all_tokens = [token for sublist in processed_texts for token in sublist]
    freq_dist = FreqDist(all_tokens)
    threshold = threshold

    def remove_rare_tokens(tokens):
        return [token for token in tokens if freq_dist[token] > threshold]

    final_tokens = [remove_rare_tokens(tokens) for tokens in processed_texts]

    # Convertir les listes de tokens en une liste de chaînes pour CountVectorizer
    final_texts = [' '.join(tokens) for tokens in final_tokens]

    return final_texts