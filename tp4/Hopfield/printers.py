import numpy as np
from matplotlib import pyplot as plt

def print_letter(s: np.ndarray, charwidth: int, pluschar = '⬛', minuschar = '⬜'):
    for i in range(0, len(s), charwidth):
        print("".join([(pluschar if s[j] > 0 else minuschar) for j in range(i, i + charwidth)]))

def plot_image(s: np.ndarray, size: tuple[int, int]):
    img = [[] for _ in range(size[1])]
    for y in range(size[1]):
        for x in range(size[0]):
            img[y].append(-s[x + y*size[0]])
    plt.imshow(img, cmap='gray')
    return plt

def plot_image_diff(s: np.ndarray, expected: np.ndarray, size: tuple[int, int]):
    img = [[] for _ in range(size[1])]
    for y in range(size[1]):
        for x in range(size[0]):
            i = x + y*size[0]
            if s[i] == expected[i]:
                img[y].append((255, 255, 255) if s[i] < 0 else (0, 0, 0))
            else:
                img[y].append((255, 176, 176) if s[i] < 0 else (84, 0, 0))
    plt.imshow(img, cmap='gray')
    return plt
