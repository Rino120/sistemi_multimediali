import numpy as np
from skimage import io, color, filters, morphology
import matplotlib.pyplot as plt

def extract_features(image_path):
    # print("chiamata ad extract_features")
    # print("image path: ", image_path)

    # Carica l'immagine utilizzando OpenCV
    # image = cv2.imread(image_path)
    image = io.imread(image_path)

    # plt.imshow(image)
    # plt.axis('off')  # Nasconde gli assi
    # plt.show()
    
    # Altre caratteristiche...
    
    # Estrae la percentuale di rosso presente nell'immagine
    red_level = extract_red_level(image)

    print("percentuale di rosso: ", red_level)

    # estrai le zone chiare dalle congiutive, di conseguenza il numero di pixel bianchi presente nell'immagine
    # ritorna la percentuale di bianco presente nell'immagine
    white_level = extract_white_level(image);

    print("percentuale di bianco: ", white_level)
    
    # Altre caratteristiche...
    
    # return red_level piu le altre feature
    return red_level, white_level

def extract_red_level(image):
    # Estrai il canale rosso dall'immagine
    red_channel = image[:, :, 0]

    # plt.imshow(red_channel, cmap='gray')  # Usa 'gray' colormap per visualizzare in scala di grigi
    # plt.axis('off')  # Nasconde gli assi
    # plt.title("Canale Rosso")
    # plt.show()

    # Calcola la quantità totale di rosso nell'immagine
    total_red = np.sum(red_channel)

    # Calcola la percentuale di rosso rispetto al totale dei pixel nell'immagine
    total_pixels = image.shape[0] * image.shape[1]
    red_percentage = (total_red / total_pixels) * 100

    # print("Quantità totale di rosso nell'immagine:", total_red)
    # print("Percentuale di rosso nell'immagine:", red_percentage)
    
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

    # print("Numero di zone bianche:", num_white_regions)
    # print("Percentuale di bianco nell'immagine:", white_percentage)

    # # Visualizza l'immagine originale e la maschera delle regioni bianche
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].imshow(image)
    # axes[0].set_title("Immagine Originale")
    # axes[0].axis('off')
    # axes[1].imshow(white_mask, cmap='gray')
    # axes[1].set_title("Maschera delle Regioni Bianche")
    # axes[1].axis('off')
    # plt.show()

    return white_percentage