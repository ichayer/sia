import numpy as np
import random
from colr import color

def salt_and_pepper_noise(image, salt_ratio, pepper_ratio):
    # Copiar la imagen original
    noisy_image = np.copy(image)
    
    # Obtener las coordenadas de los píxeles a cambiar
    salt_coords = np.random.choice(
        a=[False, True],
        size=image.shape,
        p=[1 - salt_ratio, salt_ratio]
    )
    
    pepper_coords = np.random.choice(
        a=[False, True],
        size=image.shape,
        p=[1 - pepper_ratio, pepper_ratio]
    )
    
    # Cambiar los píxeles correspondientes en la copia de la imagen
    noisy_image[salt_coords] = 1
    noisy_image[pepper_coords] = 0
    
    return noisy_image

def flip_bits(image, probability):
    noisy_image = np.copy(image)
    
    for row in range(len(image)):
        for col in range(len(image[row])):
            if random.random() < probability:
                noisy_image[row][col] = 1 - noisy_image[row][col]
    
    return noisy_image

def add_line_noise(image, prob_horizontal=0.2, prob_vertical=0.2):
    """Agrega ruido de línea a una imagen binaria.
    
    Args:
        image (np.ndarray): Imagen binaria representada como un array de Numpy.
        prob_horizontal (float): Probabilidad de agregar una línea de ruido horizontal.
        prob_vertical (float): Probabilidad de agregar una línea de ruido vertical.
        
    Returns:
        np.ndarray: Imagen con ruido de línea agregado.
    """
    height, width = image.shape
    line_noise = np.zeros_like(image)
    
    # Generar línea de ruido horizontal
    if np.random.random() < prob_horizontal:
        row = np.random.randint(0, height)
        line_noise[row, :] = np.random.randint(0, 2, size=width)
        
    # Generar línea de ruido vertical
    if np.random.random() < prob_vertical:
        col = np.random.randint(0, width)
        line_noise[:, col] = np.random.randint(0, 2, size=height)
        
    # Agregar ruido de línea a la imagen
    noisy_image = np.where(line_noise == 1, 1 - image, image)
    return noisy_image


def print_images(images):
    for i in range(len(images[0])):
        for k in range(len(images)):
            for j in range(len(images[0][0])):
                if(images[k][i][j] == 0):
                    print(color('  ', fore=(0, 0, 0), back=(0,0,0)) ,end='')
                else:
                    print(color('  ', fore=(0, 0, 0), back=(255,255,255)) ,end='')
            print("\t", end='')
        print()
