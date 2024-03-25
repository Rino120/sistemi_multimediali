
import os

# regressione logistica
from sklearn.linear_model import LogisticRegression

# valutazione del modello
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

from PIL import Image

from tqdm import tqdm

# per l'estrazione delle feature
from skimage import io
from extractor import extract_features, np
from segmentation import segmentation_congiuntiva

from label_assignator import assign_label

# 1. Passa il dataset e suddividilo in train, vbalidation e test set. Definizione delle classi
dataset_directory = os.path.dirname(os.path.abspath(__file__))

class_names = ["a rischio", "non a rischio"]

# ottengo le directory contenenti i dataset
train_directory = os.path.join(dataset_directory, 'dataset\\train')
validation_directory = os.path.join(dataset_directory, 'dataset\\validation')
test_directory = os.path.join(dataset_directory, 'dataset\\test')

print("path: ", train_directory)
print("path: ", validation_directory)
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

# Creare una lista dei file nella directory di addestramento
train_files = [file for file in os.listdir(train_directory) if file.endswith('.jpg') or file.endswith('.png')]

# Utilizzare tqdm per iterare sui file con una barra di avanzamento
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

# # Utilizza tqdm per iterare sui file con una barra di avanzamento
for file_name in tqdm(test_files, desc="Processing test images", unit="image"):
    class_dir = os.path.join(test_directory, file_name)
    test_image_path = os.path.join(test_directory, file_name)

    # Esegui le operazioni su ciascuna immagine
    try:
        # segmenta le immagini del test set
        image = io.imread(test_image_path)

        # if(file_name == "test_occhio.png"): 
        #     if image.shape[2] == 4:
        #         # Se l'immagine ha quattro canali, rimuovi il quarto canale (alfa)
        #         image = image[:, :, :3]  # Mantieni solo i primi tre canali (RGB)

        #     plt.imshow(image)
        #     plt.axis('off')  # Nasconde gli assi
        #     plt.show()
        #     congiuntiva.show()

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
        # Prova ad estrarre le caratteristiche dall'immagine
        # Converti l'immagine in un array NumPy 
        image_array = np.array(congiuntiva_image)
        features = extract_features(image_array)
        print(features)
        X_test.append(features)

        # assegna l'etichetta
        label = assign_label(features)
        y_true.append(label)
    except Exception as e:
    # Se si verifica un errore, stampa il messaggio di errore
        print(f"Errore durante l'estrazione delle caratteristiche dall'immagine {image}: {str(e)}")

# Utilizza tqdm per iterare sui file con una barra di avanzamento
# for file_name in tqdm(test_files, desc="Processing images", unit="image"):
#     class_dir = os.path.join(test_directory, file_name)
#     test_image_path = os.path.join(test_directory, file_name)

#     # Esegui le operazioni su ciascuna immagine
#     try:
#         # segmenta le immagini del test set
#         image = io.imread(test_image_path)
#         features = extract_features(image)
#         X_test.append(features)

#         label = assign_label(features)
#         y_true.append(label)
#     except Exception as e:
#         print(f"Errore durante la segmentazione dell'immagine {test_image_path}: {str(e)}")

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
except Exception as e:
    print(f"Errore nell'applicazione del modello: {str(e)}")

# 5.1 verifica l'accuratezza del modello
# calcola l'accuratezza
accuracy = accuracy_score(y_true, y_pred)

print("Accuracy:", accuracy)

# predizioni del modello
y_pred_proba = logistic_regression_model.predict_proba(X_test)

# 5.2 verifica la perdita logaritmica
# misura la discrepanza tra le previsioni del modello e le etichette di classe reali
# calcola la perdita logaritmica
loss = log_loss(y_true, y_pred_proba)

print("Log Loss:", loss)

# 6. Visualizzazione delle predizioni
    
# 7. itera e valuta il modello