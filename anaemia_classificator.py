import os
import numpy as np
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# per la segmentazione delle immagini
from segmentation import segmentation_congiuntiva

# per l'estrazione delle feature
from extractor import extract_features

from statistic_graphic import draw_graphic

# Percorso alla cartella contenente le immagini
# inserire percorso dataset
script_directory = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(script_directory, 'dataset')

print("Percorso cartella: ", image_folder_path);

# Leggi file excel
# Sostituisci 'nome_file_excel.xlsx' con il nome effettivo del tuo file Excel
script_directory = os.path.dirname(os.path.abspath(__file__))
excel_file_path = os.path.join(script_directory, 'dataset_pazienti.xlsx')

print(excel_file_path);

# leggi le colonne codice, sesso, eta della riga 2 dal file excel
# aggiungi al data frame relativo al file excel
excel_df = pd.read_excel(excel_file_path, usecols=['Codice', 'Sesso', 'Età'], header=1)

# data frame contenente solo le informazioni relative alle immagini selezionate
eyes_df = pd.DataFrame(columns=['Codice', 'Sesso', 'Età'])

# print(excel_df.head())

# Lista per memorizzare le feature e le etichette
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

        # Estrai il codice dell'immagine dal nome del file (T_i da T_i_84375893)
        image_code = '_'.join(filename.split('_')[:2])
        print("codice immagine estratto: ", image_code);

        # estrai riga corrispondente dal file excel
        selected_row = excel_df.loc[excel_df['Codice'] == image_code]

        # Aggiungi la riga selezionata al nuovo DataFrame
        # eyes_df = eyes_df.append(selected_row, ignore_index=True)

        # Aggiungi la riga selezionata al nuovo DataFrame
        eyes_df = pd.concat([eyes_df, selected_row], ignore_index=True)

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

        # Aggiungi una nuova colonna per ciascuna feature nel DataFrame excel_df
        for feature_name, feature_value in features.items():
            eyes_df.loc[eyes_df['Codice'] == image_code, feature_name] = feature_value

        # Visualizza solo la riga corrispondente all'immagine in fase di elaborazione
        print(excel_df[excel_df['Codice'] == image_code])

# Seleziona solo le colonne necessarie per il plot
plot_data = eyes_df.loc[excel_df['Sesso'] == 'M', ['Età', 'hemoglobin_level', 'Sesso']]
draw_graphic(plot_data, 'M')

plot_data = eyes_df.loc[excel_df['Sesso'] == 'F', ['Età', 'hemoglobin_level', 'Sesso']]
draw_graphic(plot_data, 'F')
