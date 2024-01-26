import os
import skimage.io as io
import skimage.filters as filters
import skimage.segmentation as seg
import skimage.color as color
import cv2
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

# Trova i contorni nella maschera binaria
contours, _ = cv2.findContours((a_channel > thresh).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Trova il contorno più grande
largest_contour = max(contours, key=cv2.contourArea)

# Crea una maschera vuota delle stesse dimensioni dell'immagine originale
mask = np.zeros_like(image, dtype=np.uint8)

# Disegna il contorno sulla maschera
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Usa np.where per ottenere l'immagine con la congiuntiva e il resto dell'immagine
congiuntiva_region = np.where(mask > 0, image, 0)

# Visualizza l'immagine originale e la parte della congiuntiva
io.imshow_collection([image, congiuntiva_region])
io.show()
