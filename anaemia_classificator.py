import os
import cv2
import numpy as np
import pandas as pd
from skimage import io, color
import skimage.filters as filters
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# per la segmentazione delle immagini
from segmentation import segmentation_congiuntiva

# eper l'estrazione delle feature
from extractor import extract_features

# Percorso alla cartella contenente le immagini
# inserire percorso dataset
script_directory = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(script_directory, 'dataset')

print("Percorso cartella: ", image_folder_path);

# Lista per memorizzare le feature e le etichette
features_list = []
labels_list = []

# Definizione della soglia per i livelli di emoglobina
soglia = 0.5  # Modifica questo valore in base ai tuoi requisiti

# Ciclo attraverso le immagini nella cartella
for filename in os.listdir(image_folder_path):
    print("entro nel ciclo for...");
    if filename.endswith('.jpg') or filename.endswith('.png'):
        print("entro nell'if...");
        # Costruzione del percorso completo dell'immagine
        image_path = os.path.join(image_folder_path, filename)

        # print("percorso immagine: ", image_path);

        # Caricamento dell'immagine utilizzando skimage
        image = io.imread(image_path)

        # Esegui la segmentazione per isolare la parte relativa alla congiuntiva
        congiuntiva_region = segmentation_congiuntiva(image)

        # Esegui l'estrazione delle feature sulla parte della congiuntiva
        features = extract_features(congiuntiva_region)

        # Determina l'etichetta in base ai livelli di emoglobina
        label = 1 if features['hemoglobin_level'] > soglia else 0  # Personalizza la soglia secondo necessità

        # Aggiunta dell'etichetta se necessario
        labels_list.append(label)

        # Aggiunta delle feature alla lista
        features_list.append(features)

percentage_second_class = 0.2

# Creazione di un DataFrame pandas con le feature
df = pd.DataFrame(features_list)

# Stampa il DataFrame come report di esempio
print(df.head())

# Aggiunta di colonne con le etichette se necessario
df['Label'] = labels_list

df['Label'] = np.random.choice([0, 1], size=len(df), p=[1 - percentage_second_class, percentage_second_class])

print(df['Label'].value_counts())

# Esegui l'analisi delle feature e l'addestramento del modello come nel tuo caso
# Assumendo che tu abbia già creato e popolato il DataFrame 'df' con le feature estratte

# Divisione del DataFrame in feature (X) ed etichette (y)
X = df.drop('Label', axis=1)  # Rimuovi la colonna 'Label' se presente
y = df['Label']

# Divisione del dataset in set di addestramento e set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello di regressione logistica
model = LogisticRegression()
model.fit(X_train, y_train)

# Predizioni sul set di test
predictions = model.predict(X_test)

# Valutazione delle prestazioni del modello
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Stampa risultati
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')

# servono per capire quanto bene lavora il modello di regressione logistica
# matrice di confusione valuta i falsi positivi e negativi e i veri positivi e negativi
# Visualizzazione della matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.xlabel('Predizione')
plt.ylabel('Verità')
plt.title('Matrice di Confusione')
plt.show()

# Curve ROC
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# fornisce una visione complessiva delle prestazioni del modello al variare della soglia di decisione
# Visualizzazione della curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
