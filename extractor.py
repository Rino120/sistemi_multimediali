import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_features(congiuntiva_region):
    print("chiamata a extract features...")
    #print("Percorso immagine: ", image_path)
    
    # Caricamento dell'immagine utilizzando la libreria skimage
    # passo l'immagine segmentata
    image = congiuntiva_region

    # gestisce il caso in cui non legge l'immagine
    if image is None:
        print(f"Errore: Impossibile leggere l'immagine")
        return None

    # Estrai la componente rossa dell'immagine
    red_channel = image[:, :, 2]

    # Calcola la media dei livelli di emoglobina (o la tua metrica desiderata)
    hemoglobin_level = np.mean(red_channel)

    print("livello emoglobina: ", hemoglobin_level);

    # Visualizza l'immagine originale e la componente rossa
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(congiuntiva_region, cv2.COLOR_BGR2RGB))
    # plt.title('Immagine originale')

    # plt.subplot(1, 2, 2)
    # plt.imshow(red_channel, cmap='gray')
    # plt.title('Componente rossa')

    # plt.show()

    # Restituzione delle feature estratte come un dizionario
    return {
        'hemoglobin_level': hemoglobin_level,
    } 