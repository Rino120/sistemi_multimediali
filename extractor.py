import numpy as np
from skimage import color, filters, morphology

# valori di riferimento 
MIN_RED_LEVEL = 7000     # soglia che indica il valore minimo di rosso per indicare che l'occhio è sano
MIN_WHITE_LEVEL = 0.0003   # soglia che indica il valore minimo di bianco per indicare che l'occhio non è sano

def extract_features(image):
    # Estrae la percentuale di rosso presente nell'immagine
    red_level = extract_red_level(image)

    # estrai le zone chiare dalle congiutive, di conseguenza il numero di pixel bianchi presente nell'immagine
    # ritorna la percentuale di bianco presente nell'immagine
    white_level = extract_white_level(image);

    # Altre caratteristiche...
    # print("\nlivello rosso: ", red_level)
    # print("livello bianco: ", white_level)
    
    # return red_level piu le altre feature
    return red_level, white_level

def extract_red_level(image):
    # Estrai il canale rosso dall'immagine
    red_channel = image[:, :, 0]

    # Calcola la quantità totale di rosso nell'immagine
    total_red = np.sum(red_channel)

    # Calcola la percentuale di rosso rispetto al totale dei pixel nell'immagine
    total_pixels = image.shape[0] * image.shape[1]
    red_percentage = (total_red / total_pixels) * 100

    return red_percentage

# da sistemare ma funziona
def extract_white_level(image):
    image_rgb = image[:, :, :3]

    gray_image = color.rgb2gray(image_rgb)

    # Applica un filtro di edge detection per evidenziare i contorni
    edges = filters.sobel(gray_image)

    # Applica un'operazione di binarizzazione per ottenere una maschera delle regioni bianche
    white_mask = edges < 0.1  # Modifica il valore della soglia in base alla tua immagine

    # Applica un'operazione di erosione per rimuovere eventuali piccoli dettagli indesiderati
    white_mask = morphology.binary_erosion(white_mask, morphology.disk(2))

    # Conta il numero di regioni bianche
    num_white_regions = morphology.label(white_mask).max()

    # Calcola la percentuale di bianco rispetto al totale dei pixel nell'immagine
    total_pixels = white_mask.size
    white_percentage = (num_white_regions / total_pixels) * 100

    return white_percentage

def assign_label(features):
    # 0 red_level
    # 1 white_level

    # se il minimo rosso non viene raggiunto
    # e il minimo bianco viene raggiunto 
    # l'occhio non è sano
    if features[0] < MIN_RED_LEVEL and features[1] > MIN_WHITE_LEVEL:
        return 1  # Assegna la label 1 se il livello di rosso è inferiore a 10 e il livello di bianco è superiore a 10
    else:
        return 0  # Altrimenti assegna la label 0