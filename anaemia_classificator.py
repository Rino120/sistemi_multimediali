
import os

# per l'estrazione delle feature
from extractor import extract_features
from segmentation import segmentation_congiuntiva

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
# for file_name in os.listdir(train_directory):
#         class_dir = os.path.join(train_directory, file_name)
#         if file_name.endswith('.jpg') or file_name.endswith('.png'):
#             train_image_path = os.path.join(train_directory, file_name)

#             print(train_image_path)

#             # Estrai le caratteristiche dall'immagine e aggiungile a train_features
#             try:
#                 # Prova ad estrarre le caratteristiche dall'immagine
#                 features = extract_features(train_image_path)
#                 train_features.append(features)

#                 # aggiungere assegnazione dell'etichetta, classe di appertenenza dell'immagine
#             except Exception as e:
#                 # Se si verifica un errore, stampa il messaggio di errore
#                 print(f"Errore durante l'estrazione delle caratteristiche dall'immagine {train_image_path}: {str(e)}")


# 3. Segmentazione delle immagini presenti nel test set
for file_name in os.listdir(test_directory):
    class_dir = os.path.join(test_directory, file_name)
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        test_image_path = os.path.join(test_directory, file_name)

        print(test_image_path)

        # segmenta le immagini del test set per una maggiore facilita di analisi delle caratteristiche
        try:
            # provvisorio
            # creare array dove salvare il test set processato
            # da migliorare algoritmo di segmentazione
            congiuntiva = segmentation_congiuntiva(test_image_path)
        except Exception as e:
            # Se si verifica un errore, stampa il messaggio di errore
            print(f"Errore durante la segmentazione dell'immagine {test_image_path}: {str(e)}")

# 4. Applicazione del modello (Regressione logistica) sul test set

# 5. Valutazione dei risultati e prestazioni del modello tramite precisione, accuratezza e richiamo

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