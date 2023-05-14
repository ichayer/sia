import numpy as np

def map_reverse_search(m: dict[str, np.ndarray], v: np.ndarray) -> (str | None):
    for key, value in m.items():
        if np.array_equal(value, v):
            return key
    return None

def load_pattern_map(filename: str, plusone_representation = 'X', minusone_representation = '.') -> dict[str, np.ndarray]:
    patterns = {}
    with open(filename, 'r') as file:
        data = file.read()
    for entry in data.replace('\n', '').replace('\r', '').split(':'):
        entry = entry.strip()
        if len(entry) == 0:
            continue
        e = entry.split('=')
        name = e[0].strip()
        if name in patterns:
            raise ValueError('Error: pattern defined twice in pattern file: ' + name)
        pattern_text = e[1]
        pattern = []
        while len(pattern_text) != 0:
            if pattern_text.startswith(plusone_representation):
                pattern.append(1)
                pattern_text = pattern_text[len(plusone_representation):]
            elif pattern_text.startswith(minusone_representation):
                pattern.append(-1)
                pattern_text = pattern_text[len(minusone_representation):]
            else:
                raise ValueError('Value in pattern expression is not a valid representation: ' + pattern_text)
        patterns[name] = np.array(pattern)
    return patterns
