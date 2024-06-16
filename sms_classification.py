# %%
# Téléchargement du dossier Zip contenant les données 
# Commande à exécuter dans le terminal
#!wget https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip


# %%
print("Extraction des fichiers contenus dans le dossier Zip")
import zipfile
import os

# Nom du fichier ZIP local
nom_fichier_zip = 'sms+spam+collection.zip'

# Dossier où vous souhaitez extraire les fichiers
dossier_extraction = './sms_spam_dataset/'

# Vérifier si le dossier d'extraction existe, sinon le créer
if not os.path.exists(dossier_extraction):
    os.makedirs(dossier_extraction)

# Extraction du contenu du fichier ZIP
with zipfile.ZipFile(nom_fichier_zip, 'r') as zip_ref:
    zip_ref.extractall(dossier_extraction)

# Liste tous les fichiers dans le dossier
fichiers = os.listdir(dossier_extraction)
print(fichiers)

print("Extraction terminée.")

# %%
print("Importation des données dans Pandas")
import pandas as pd

#Le fichier de données est 'SMSSpamCollection' dans ce cas
fichier_sms = 'SMSSpamCollection'

# Chemin complet vers le fichier SMS
chemin_sms = os.path.join(dossier_extraction, fichier_sms)

# Colonnes dans le fichier SMSSpamCollection
colonnes = ['label', 'message']

# Charger les données dans un DataFrame Pandas
df = pd.read_csv(chemin_sms, sep='\t', header=None, names=colonnes)

# Afficher les 5 premières lignes pour vérification
df.head()

# %%
df.info()

# %%
df.label.value_counts(normalize=True)

# %%
print("Prétraiter les Données")
from utils import preprocess_texts

# %%
# Appliquer le prétraitement
df["cleaned_message"] = preprocess_texts(df["message"])
df.head()

# %%
print("Extraction des Features et Création des TensorDatasets et des DataLoaders")
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader
import joblib

# Diviser les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_message'], df['label'],
    test_size=0.2, random_state=seed
)

# Initialiser CountVectorizer
vectorizer = CountVectorizer()

# Adapter et transformer les données prétraitées
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Sauvegarde de vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

# Convertir les labels en format numérique
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Créer les TensorDatasets
train_dataset = TensorDataset(
    torch.tensor(X_train_vectorized.toarray(), dtype=torch.float32),
    torch.tensor(y_train_encoded, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test_vectorized.toarray(), dtype=torch.float32),
    torch.tensor(y_test_encoded, dtype=torch.long)
)

# Créer les DataLoaders
batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Vérification
for data, labels in train_loader:
    print(data)
    print(labels)
    break

# %%
# Conversion de la matrice sparse en DataFrame
X_train_df = pd.DataFrame(
    X_train_vectorized.toarray(),
    columns=vectorizer.get_feature_names_out()
)
X_train_df.head()

# %%
print("Construction du modèle")
import torch.nn as nn

# Définir les dimensions d'entrée
input_dim = X_train_vectorized.shape[1]

# Sauvegarder input_dim dans un fichier
torch.save(input_dim, 'input_dim.pth')

# Construire le modèle avec nn.Sequential
model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 2)  # Deux sorties : spam ou non spam
)

# Affichage de la structure du modèle
print(model)

# %%
print("Entraînement du Modèle")
import torch.optim as optim
import joblib

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Initialiser des listes pour enregistrer la perte
train_losses = []
test_losses = []

# Entraînement du modèle
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculer la perte moyenne pour l'ensemble d'entraînement
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Évaluation sur l'ensemble de test
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Calculer la perte moyenne pour l'ensemble de test
    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Sauvegarde du modèle
    if epoch == num_epochs - 1:
      torch.save(model.state_dict(), 'trained_model.pth')

# Tracer les pertes d'entraînement et de validation
import matplotlib.pyplot as plt
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# %%
print("Évaluer le Modèle")
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

# %%
print("Inférence sur de nouveaux SMS")
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Charger input_dim
input_dim = torch.load('input_dim.pth')


# Définir la même structure du modèle
model_loaded = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Charger le state_dict du modèle
state_dict = torch.load('trained_model.pth', map_location=torch.device('cpu'))  # Assurez-vous de spécifier le périphérique correctement si nécessaire
model_loaded.load_state_dict(state_dict)

# Mettre le modèle en mode évaluation
model_loaded.eval()

# Charger le vectorizer
vectorizer_loaded = joblib.load('vectorizer.pkl')

# %%
# Simuler de nouveaux textes SMS
new_texts = [
    "Have you had a good day? Mine was really busy are you up to much tomorrow night?",
    "ree entry in 2 a weekly comp for a chance to win an ipod. Txt POD to 80182 to get entry (std txt rate) T&C's apply 08452810073 for details 18+",
    "Hope you are not scared!",
    "Call 09094100151 to use ur mins! Calls cast 10p/min (mob vary). Service provided by AOM, just GBP5/month. AOM Box61,M60 1ER until u stop. Ages 18+ only!"
]

# Prétraiter les nouveaux textes SMS
new_texts_preprocessed = preprocess_texts(new_texts)  # Assumer que preprocess_text fait le nettoyage et la tokenisation
print(new_texts_preprocessed)


# Vectoriser les nouveaux textes
X_new_vectorized = vectorizer_loaded.transform(new_texts_preprocessed)

# Convertir en tenseur PyTorch
X_new_tensor = torch.from_numpy(X_new_vectorized.toarray()).float()
X_new_tensor

# %%
# Effectuer l'inférence
with torch.no_grad():
    outputs = model_loaded(X_new_tensor)
    _, predicted = torch.max(outputs, 1)

# Afficher les prédictions
print(predicted)

for text, prediction in zip(new_texts, predicted):
    label = "spam" if prediction.item() == 1 else "non spam"
    print(f"Text: {text} -> Prediction: {label}")

# %%
