
import os

# regressione logistica
from sklearn.linear_model import LogisticRegression

# valutazione del modello
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# per l'estrazione delle feature
from skimage import io
from extractor import np, extract_features, assign_label
from segmentation import segmentation_congiuntiva

# utils
from tqdm import tqdm

# 1. Passa il dataset e suddividilo in train, vbalidation e test set. Definizione delle classi
dataset_directory = os.path.dirname(os.path.abspath(__file__))

class_names = ["a rischio", "non a rischio"]

# ottengo le directory contenenti i dataset
train_directory = os.path.join(dataset_directory, 'dataset\\train')
test_directory = os.path.join(dataset_directory, 'dataset\\test')

print("path: ", train_directory)
print("path: ", test_directory)

# 2. Estrazione delle feaeture dalle immagini gia segmentate del train set
# crea funzione per rendere il tutto piu leggibile

# contiene le feature di tutte le immagini presenti nel train set
# caratteristiche del set di addestramento
X_train = []

# labels per ogni immagine presente nel train set
# etichette corrispondenti del set di addestramento
# per il set di addestramento assegnare le label in maniera manuale
y_train = []

# Cicla attraverso tutte le immagini nella directory di addestramento
print("Estrazione delle feature dal train set...")

# lista dei file nel train set
train_files = [file for file in os.listdir(train_directory) if file.endswith('.jpg') or file.endswith('.png')]

# con tqdm è possibile avere una barra di caricamento
for file_name in tqdm(train_files, desc="Processing images", unit="image"):
    class_dir = os.path.join(train_directory, file_name)
    train_image_path = os.path.join(train_directory, file_name)

    # Estrai le caratteristiche e assegna l'etichetta a ciascuna immagine
    try:
        image = io.imread(train_image_path)
        features = extract_features(image)
        X_train.append(features)

        label = assign_label(features)
        y_train.append(label)
    except Exception as e:
        print(f"Errore durante l'estrazione delle caratteristiche dall'immagine {train_image_path}: {str(e)}")


# 3. Segmentazione delle immagini presenti nel test set
congiuntivas = [] # insieme delle immagini di test segmentate

print("segmentazione delle immagini del test set...")

# Ottieni la lista dei file nella directory del test set
test_files = [file for file in os.listdir(test_directory) if file.endswith('.jpg') or file.endswith('.png')]

# con tqdm è possibile avere una barra di caricamento
for file_name in tqdm(test_files, desc="Processing test images", unit="image"):
    class_dir = os.path.join(test_directory, file_name)
    test_image_path = os.path.join(test_directory, file_name)

    try:
        # segmenta le immagini del test set
        image = io.imread(test_image_path)

        # Se l'immagine ha quattro canali, rimuovi il quarto canale (alfa)
        # la segmentazione agisce solo sui i canali rgb, canale alfa ignorato
        if image.shape[2] == 4:
            image = image[:, :, :3] 

        congiuntiva = segmentation_congiuntiva(image)
        congiuntivas.append(congiuntiva)

    except Exception as e:
        print(f"Errore durante la segmentazione dell'immagine {test_image_path}: {str(e)}")
                
# 3.1 Estrazione delle feature dal test set segmentato
# caratteristiche del set di test
X_test = []

# etichette reali delle immagini di test
y_true = []

print("Estrazione delle feature dal test set...")
for congiuntiva_image in tqdm(congiuntivas, desc="Processing test images", unit="image"):
     # Estrai le caratteristiche dall'immagine e aggiungile a X_train
    try:
        # Converti l'immagine in un array NumPy 
        image_array = np.array(congiuntiva_image)
        features = extract_features(image_array)
        X_test.append(features)

        # assegna l'etichetta
        label = assign_label(features)
        y_true.append(label)
    except Exception as e:
    # Se si verifica un errore, stampa il messaggio di errore
        print(f"Errore durante l'estrazione delle caratteristiche dall'immagine {image}: {str(e)}")

# 4. Addestramento del modello (Regressione logistica) sul train set
logistic_regression_model = LogisticRegression()

print("Addestramento del modello...")
try:
    logistic_regression_model.fit(X_train, y_train)
except Exception as e:
    print(f"Errore nell'addestramento del modello: {str(e)}")

# 5. Applicazione del modello 
print("Applicazione del modello...")
try:
    y_pred = logistic_regression_model.predict(X_test)
    print("Predizioni del modello: ", y_pred)
    print("Valori reali: ", y_true)
except Exception as e:
    print(f"Errore nell'applicazione del modello: {str(e)}")

# 5.1 verifica l'accuratezza del modello
# calcola l'accuratezza
try:
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
except Exception as e:
    print(f"Errore nel calcolo dell'accuratezza del modello: {str(e)}")

# 5.2 calcolo della precisione del modello
# Calcola la precisione
try:
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)
except Exception as e:
    print(f"Errore nel calcolo della precisione del modello: {str(e)}")

# 5.3 Calcola il richiamo
# misura la capacità del modello di identificare correttamente tutti gli esempi positivi
# tra quelli effettivamente positivi presenti nel dataset
try:
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)
except Exception as e:
    print(f"Errore nel calcolo del richiamo del modello: {str(e)}")

# Calcola l'F1-Score
# la media armonica tra precisione e richiamo, utile quando le classi hanno un numero diverso
# di campioni o quando gli errori di falsi positivi e falsi negativi hanno conseguenze diverse
try:
    f1 = f1_score(y_true, y_pred)
    print("F1 score:", f1)
except Exception as e:
    print(f"Errore nel calcolo dell'f1-score del modello: {str(e)}")