import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# disegna il grafico a dispersione lineare e la regressione lineare
def draw_graphic(plot_data, patients_sex):
    plot_data['Età'] = pd.to_numeric(plot_data['Età'], errors='coerce')
    plot_data['hemoglobin_level'] = pd.to_numeric(plot_data['hemoglobin_level'], errors='coerce')

    # Calcola i coefficienti per la regressione lineare
    coefficients = np.polyfit(plot_data['Età'], plot_data['hemoglobin_level'], 1)

    # Calcola i valori previsti per i dati di età utilizzando i coefficienti della regressione lineare
    age_range = np.linspace(plot_data['Età'].min(), plot_data['Età'].max(), 100)
    predicted_hemoglobin = np.polyval(coefficients, age_range)

    # Plot dell'andamento della media dei livelli di emoglobina in relazione all'età
    plt.figure(figsize=(10, 6))
    plt.scatter(plot_data['Età'], plot_data['hemoglobin_level'], marker='o', label='Dati reali')
    plt.plot(age_range, predicted_hemoglobin, color='red', linestyle='-', label='Andamento stimato')

    if(patients_sex == 'M'):
        plt.title('Andamento dei livelli di emoglobina in relazione all\'età per pazienti maschi')
    
    elif(patients_sex == 'F'):
        plt.title('Andamento dei livelli di emoglobina in relazione all\'età per pazienti femmine')
    
    plt.xlabel('Età del paziente')
    plt.ylabel('Livelli di emoglobina')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()