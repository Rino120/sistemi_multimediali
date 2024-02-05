import os
import pandas as pd

# Sostituisci 'nome_file_excel.xlsx' con il nome effettivo del tuo file Excel
script_directory = os.path.dirname(os.path.abspath(__file__))
excel_file_path = os.path.join(script_directory, 'dataset_pazienti.xlsx')

print(excel_file_path);

# leggi le colonne codice, sesso, eta della riga 2 dal file excel
excel_data = pd.read_excel(excel_file_path, usecols=['Codice', 'Sesso', 'Età'], header=1)

print(excel_data.head())