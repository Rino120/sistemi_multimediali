import os
import skimage.io as io
import skimage.filters as filters
import skimage.segmentation as seg
import skimage.color as color
import numpy as np

# Carica l'immagine
script_directory = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(script_directory, 'dataset/T_2_20190606_093707.jpg')
image = io.imread(image_folder_path)

# Converti l'immagine in spazio colore Lab
lab_image = color.rgb2lab(image)

# Estrai il canale a* (che rappresenta la differenza tra rosso e verde)
a_channel = lab_image[:, :, 1]

# Applica la sogliatura adattiva sul canale a*
thresh = filters.threshold_li(a_channel)

# Crea una maschera binaria dove i pixel con valore maggiore della soglia sono bianchi
binary_mask = a_channel > thresh

# Trova i contorni nella maschera binaria
contours = seg.find_boundaries(binary_mask)

# Crea una maschera vuota delle stesse dimensioni dell'immagine originale
mask = np.zeros_like(image)

# Disegna i contorni sulla maschera
mask[contours] = 255

# Estrai la parte dell'immagine originale corrispondente alla maschera
# congiuntiva_region = np.bitwise_and(image, mask)
congiuntiva_region = np.where(mask > 0, image, 0)

# Visualizza l'immagine originale e la parte della congiuntiva
io.imshow_collection([image, congiuntiva_region])
io.show()
