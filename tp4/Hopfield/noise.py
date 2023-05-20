import numpy as np

def salt_and_pepper(s: np.ndarray, salt_percentage: float, pepper_percentage: float):
    s = np.copy(s)
    salts_count = int(salt_percentage * len(s))
    peppers_count = int(pepper_percentage * len(s))
    
    for i in range(salts_count):
        s[np.random.randint(0, len(s))] = -1
    
    for i in range(peppers_count):
        s[np.random.randint(0, len(s))] = 1
    return s

def salt_and_pepper_fast(s: np.ndarray, salt_percentage: float, pepper_percentage: float):
    return np.sign(s + (np.random.random(len(s)) < salt_percentage) - (np.random.random(len(s)) < pepper_percentage) + 0.1)
