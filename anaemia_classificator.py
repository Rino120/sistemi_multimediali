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
def extract_features(congiuntiva_region):
    print("chiamata a extract features...")
    #print("Percorso immagine: ", image_path)
    
    # Caricamento dell'immagine utilizzando la libreria skimage
    # passo l'immagine segmentata
    image = congiuntiva_region

    # gestisce il caso in cui non legge l'immagine
    if image is None:
        print(f"Errore: Impossibile leggere l'immagine {image_path}")
        return None

    # Estrai la componente rossa dell'immagine
    red_channel = image[:, :, 2]

    # Calcola la media dei livelli di emoglobina (o la tua metrica desiderata)
    hemoglobin_level = np.mean(red_channel)

    print("livello emoglobina: ", hemoglobin_level);

    # Visualizza l'immagine originale e la componente rossa
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Immagine originale')

    # plt.subplot(1, 2, 2)
    # plt.imshow(red_channel, cmap='gray')
    # plt.title('Componente rossa')

    # plt.show()

    # Restituzione delle feature estratte come un dizionario
    return {
        'hemoglobin_level': hemoglobin_level,
        # Aggiungi altre feature qui in base alle tue esigenze
    }

# Funzione per estrarre la parte della congiuntiva da un'immagine
def segment_congiuntiva(original_image):
    print("chiamata a segmentcongiutiva...")

    # Estrai il canale V (value) dalla rappresentazione HSV dell'immagine
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:,:,2]

    # Fase 1: Rimuovi le parti scure
    _, binary_mask_dark = cv2.threshold(v_channel, 100, 255, cv2.THRESH_BINARY_INV)
    binary_mask_dark = cv2.morphologyEx(binary_mask_dark, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    binary_mask_dark = cv2.morphologyEx(binary_mask_dark, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

    # Fase 2: Rimuovi le parti chiare
    _, binary_mask_light = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)
    binary_mask_light = cv2.morphologyEx(binary_mask_light, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    binary_mask_light = cv2.morphologyEx(binary_mask_light, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

    # Combinazione delle maschere utilizzando l'operatore AND
    binary_mask_combined = cv2.bitwise_and(binary_mask_dark, binary_mask_light)

    # Visualizza le maschere binarie
    plt.subplot(2, 3, 1)
    plt.imshow(binary_mask_dark, cmap='gray')
    plt.title('Maschera Scure')

    plt.subplot(2, 3, 2)
    plt.imshow(binary_mask_light, cmap='gray')
    plt.title('Maschera Chiare')

    plt.subplot(2, 3, 3)
    plt.imshow(binary_mask_combined, cmap='gray')
    plt.title('Maschera Combinata')

    # Trova i contorni nella maschera binaria combinata
    contours, _ = cv2.findContours(binary_mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra i contorni in base a proprietà morfologiche (ad esempio, lunghezza)
    min_contour_length = 100  # Regola questo valore in base alle tue esigenze
    filtered_contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_contour_length]

    # Verifica se ci sono contorni prima di procedere
    if filtered_contours:
        # Trova il contorno più grande tra quelli filtrati
        largest_contour = max(filtered_contours, key=cv2.contourArea)

        # Crea una maschera vuota delle stesse dimensioni dell'immagine originale
        mask = np.zeros_like(original_image)

        # Disegna il contorno sulla maschera
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Estrai la parte dell'immagine originale corrispondente alla maschera
        congiuntiva_region = cv2.bitwise_and(original_image, mask)
    else:
        print("Nessun contorno trovato.")
        congiuntiva_region = np.zeros_like(original_image)  # Immagine vuota

    # Visualizza l'immagine originale e la parte della congiuntiva
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Immagine originale')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(congiuntiva_region, cv2.COLOR_BGR2RGB))
    plt.title('Parte della congiuntiva')

    plt.show()

    return congiuntiva_region

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

        # Caricamento dell'immagine utilizzando OpenCV
        original_image = cv2.imread(image_path)

        # Esegui la segmentazione per isolare la parte relativa alla congiuntiva
        congiuntiva_region = segment_congiuntiva(original_image)

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
