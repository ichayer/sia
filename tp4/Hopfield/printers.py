import numpy as np
from matplotlib import pyplot as plt

def print_letter(s: np.ndarray, charwidth: int, pluschar = '⬛', minuschar = '⬜'):
    for i in range(0, len(s), charwidth):
        print("".join([(pluschar if s[j] > 0 else minuschar) for j in range(i, i + charwidth)]))

def plot_image(s: np.ndarray, size: tuple[int, int]):
    img = [[] for _ in range(size[1])]
    for y in range(size[1]):
        for x in range(size[0]):
            img[y].append(1 if s[x + y*size[0]] < 0.5 else 0)
    plt.imshow(img, cmap='gray')
    return plt
