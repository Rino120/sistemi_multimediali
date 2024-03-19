
import os

# regressione logistica
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tqdm import tqdm

# per l'estrazione delle feature
from skimage import io
from extractor import extract_features
from segmentation import segmentation_congiuntiva

from label_assignator import assign_label

# from keras.preprocessing.image import ImageDataGenerator

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
train_features = []

# labels per ogni immagine presente nel train set
train_labels = []

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
        train_features.append(features)

        label = assign_label(features)
        train_labels.append(label)
    except Exception as e:
        print(f"Errore durante l'estrazione delle caratteristiche dall'immagine {train_image_path}: {str(e)}")


# 3. Segmentazione delle immagini presenti nel test set
congiuntivas = [] # insieme delle immagini di test segmentate

print("segmentazione delle immagini del test set...")

# Ottieni la lista dei file nella directory del test set
test_files = [file for file in os.listdir(test_directory) if file.endswith('.jpg') or file.endswith('.png')]

# Utilizza tqdm per iterare sui file con una barra di avanzamento
for file_name in tqdm(test_files, desc="Processing images", unit="image"):
    class_dir = os.path.join(test_directory, file_name)
    test_image_path = os.path.join(test_directory, file_name)

    # Esegui le operazioni su ciascuna immagine
    try:
        # segmenta le immagini del test set
        image = io.imread(test_image_path)
        congiuntiva = segmentation_congiuntiva(image)
        congiuntivas.append(congiuntiva)
    except Exception as e:
        print(f"Errore durante la segmentazione dell'immagine {test_image_path}: {str(e)}")
                
# 3.1 Estrazione delle feature dal test set segmentato
test_features = []

test_labels = []

print("Estrazione delle feature dal test set...")
for congiuntiva_image in tqdm(congiuntivas, desc="Processing test images", unit="image"):
     # Estrai le caratteristiche dall'immagine e aggiungile a train_features
    try:
        # Prova ad estrarre le caratteristiche dall'immagine
        features = extract_features(congiuntiva_image)
        test_features.append(features)

        # aggiungere assegnazione dell'etichetta, classe di appertenenza dell'immagine
        # provvisorio aggiungere algoritmo per assegnazione delle etichette
        label = assign_label(features)
        test_labels.append(label)

    except Exception as e:
    # Se si verifica un errore, stampa il messaggio di errore
        print(f"Errore durante l'estrazione delle caratteristiche dall'immagine {image}: {str(e)}")

# 4. Applicazione del modello (Regressione logistica) sul train set
logistic_regression_model = LogisticRegression()

print("Applicazione del modello...")
try:
    logistic_regression_model.fit(train_features, train_labels)
except Exception as e:
    print(f"Errore nell'applicazione della regressione logistica: {str(e)}")

# Effettua la cross-validation per ottenere le predizioni del modello
# train_predictions = cross_val_predict(logistic_regression_model, train_features, train_labels, cv=5)

# # Calcola la matrice di confusione
# conf_matrix = confusion_matrix(train_labels, train_predictions)

# # Visualizza la matrice di confusione utilizzando seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()
    
# ottimizzazione degli iperparametri con il validation set
# validation_accuracy = accuracy_score(y_val, logistic_regression_model.predict(X_val))
# print("Validation set accuracy:", validation_accuracy)


# 5. Valutazione dei risultati e prestazioni del modello tramite precisione, accuratezza e richiamo
# Applicazione del modello sul test set
# test_accuracy = logistic_regression_model.score(test_features, test_labels)

# 5.1 applicazione della curva ROC
# Calcola le probabilità delle classi positive
# y_prob = logistic_regression_model.predict_proba(test_features)[:, 1]

# # Calcola la curva ROC e l'AUC
# fpr, tpr, thresholds = roc_curve(test_labels, y_prob)
# roc_auc = auc(fpr, tpr)

# # Disegna la curva ROC
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# 6. Itera e ottimizza il modello

# Definisci i generatori di immagini per set di addestramento, validazione e test
# train_datagen = ImageDataGenerator(rescale=1./255)
# validation_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# # Definisci i generatori di dati per caricare le immagini dai rispettivi percorsi
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_height, img_width),  # Specifica le dimensioni delle immagini
#     batch_size=batch_size,
#     class_mode='binary')  # Imposta la modalità di classificazione

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')

# Ora hai generatori di dati pronti per l'addestramento, la validazione e il test
# Puoi passare questi generatori direttamente ai metodi di addestramento e valutazione del modello