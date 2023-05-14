import numpy as np

def print_letter(s: np.ndarray, charwidth: int, pluschar = '⬛', minuschar = '⬜'):
    for i in range(0, len(s), charwidth):
        print("".join([(pluschar if s[j] > 0 else minuschar) for j in range(i, i + charwidth)]))
