import streamlit as st
import torch
import joblib
from utils import preprocess_texts

# Charger input_dim (si nécessaire)
input_dim = torch.load('input_dim.pth')

# Charger le modèle PyTorch
model_loaded = torch.nn.Sequential(
    torch.nn.Linear(input_dim, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
)
state_dict = torch.load('trained_model.pth', map_location=torch.device('cpu'))
model_loaded.load_state_dict(state_dict)
model_loaded.eval()

# Charger le vectorizer
vectorizer_loaded = joblib.load('vectorizer.pkl')

# Fonction de prédiction
def predict_spam(text):
    # Prétraiter le texte
    preprocessed_text = preprocess_texts([text])
    # Vectoriser le texte prétraité
    text_vectorized = vectorizer_loaded.transform(preprocessed_text)
    # Convertir en tenseur PyTorch
    text_tensor = torch.from_numpy(text_vectorized.toarray()).float()
    # Passer le texte à travers le modèle pour la prédiction
    with torch.no_grad():
        output = model_loaded(text_tensor)
        predicted_class = torch.argmax(output).item()
    return predicted_class

# Définir l'application Streamlit
st.title("Détection de Spam dans les SMS")
st.markdown("[Josué AFOUDA](https://www.linkedin.com/in/josu%C3%A9-afouda)")

# Interface utilisateur pour la prédiction
user_input = st.text_area("Entrez votre SMS ici :", height=50)

if st.button("Prédire"):
    if user_input.strip() == "":
        st.error("Veuillez entrer un SMS.")
    else:
        prediction = predict_spam(user_input)
        if prediction == 1:
            st.error("Ce SMS est un spam.")
        else:
            st.success("Ce SMS n'est pas un spam.")