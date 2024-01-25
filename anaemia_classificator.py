import os
import cv2
import numpy as np
import pandas as pd
from skimage import io, color, feature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Funzione per l'estrazione delle feature da un'immagine
def extract_features(image_path):
    print("chiamata a extract features...")
    print("Percorso immagine: ", image_path)
    # Caricamento dell'immagine utilizzando la libreria skimage
    image = cv2.imread(image_path)

    # gestisce il caso in cui non legge l'immagine
    if image is None:
        print(f"Errore: Impossibile leggere l'immagine {image_path}")
        return None

    # Conversione in scala di grigi, se l'immagine non è già in scala di grigi
    gray_image = color.rgb2gray(image)

    # Esempio di estrazione di feature: utilizziamo il numero medio di edge rilevati con Canny
    canny_edges = feature.canny(gray_image)
    canny_edges_mean = canny_edges.mean()

    # Puoi aggiungere ulteriori feature qui in base alle tue esigenze
    # ...

    # Restituzione delle feature estratte come un dizionario
    return {
        'canny_edges_mean': canny_edges_mean,
        # Aggiungi altre feature qui se necessario
    }

# Funzione per estrarre la parte della congiuntiva da un'immagine
def segment_congiuntiva(original_image):
    print("chiamata a segmentcongiutiva...");
    # Converti l'immagine in scala di grigi
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Applica un filtro di smooth per ridurre il rumore
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Esegui la segmentazione utilizzando il metodo di Otsu
    _, binary_mask = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Trova i contorni nella maschera binaria
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trova il contorno più grande (presumibilmente la congiuntiva)
    largest_contour = max(contours, key=cv2.contourArea)

    # Crea una maschera vuota delle stesse dimensioni dell'immagine originale
    mask = np.zeros_like(original_image)

    # Disegna il contorno sulla maschera
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Estrai la parte dell'immagine originale corrispondente alla maschera
    congiuntiva_region = cv2.bitwise_and(original_image, mask)

    return congiuntiva_region

# Percorso alla cartella contenente le immagini
# inserire percorso dataset
image_folder_path = 'C:\\Users\\ruggi\\Desktop\\Dimauro\\database_sclere\\Italiano congiuntive\\Dataset congiuntive gruppo anemia  organizzato 28 mar 2020\\Trasfusionale congiuntive\\'

print("Percorso cartella: ", image_folder_path);

# Lista per memorizzare le feature e le etichette
features_list = []
labels_list = []

# Ciclo attraverso le immagini nella cartella
for filename in os.listdir(image_folder_path):
    print("entro nel ciclo for...");
    if filename.endswith('.jpg') or filename.endswith('.png'):
        print("entro nell'if...");
        # Costruzione del percorso completo dell'immagine
        image_path = os.path.join(image_folder_path, filename)

        print("percorso immagine: ", image_path);

        # Caricamento dell'immagine utilizzando OpenCV
        original_image = cv2.imread(image_path)

        # Esegui la segmentazione per isolare la parte relativa alla congiuntiva
        congiuntiva_region = segment_congiuntiva(original_image)

        # Esegui l'estrazione delle feature sulla parte della congiuntiva
        # features = extract_features(congiuntiva_region)
        features = extract_features(image_path)

        # Determina l'etichetta in base alle tue condizioni
        # Ad esempio, se il nome del file contiene 'congiuntiva', l'etichetta potrebbe essere 1, altrimenti 0
        label = 1 if 'congiuntiva' in filename.lower() else 0

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
