import numpy as np
from hopfield import Hopfield
from pattern_loader import load_pattern_map, map_reverse_search

letter_patterns = load_pattern_map('tp4/letters.txt')

letters = 'AFJU'
# letter_query = 'J' # The system correctly identifies this letter as J.
letter_query = 'T' # The system identifies this letter as a J (T is not a learnt pattern)
# letter_query = 'X' # The system does not identify this letter ==> Spurious state!

patterns = []
for letter in letters:
    patterns.append(letter_patterns[letter])
patterns = np.array(patterns)

net = Hopfield(patterns=patterns)

query = letter_patterns[letter_query]

print(f"Hopfeld network learned {len(patterns)} patterns: letters {letters}")
print(f"Querying for letter: {letter_query}")

def printer(s, converged, epochs):
    print(f"Epoch {epochs}.")

s, converged, epochs = net.evaluate(query=query, max_epochs=20, printer=printer)

if converged:
    print(f"Done! Converged after {epochs} epochs: (letter {map_reverse_search(letter_patterns, s[-1])}) {s[-1]}")
else:
    print(f"Failed to converge after {epochs} epochs: (letter {map_reverse_search(letter_patterns, s[-1])}) {s[-1]}")
