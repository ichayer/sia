import numpy as np
from hopfield import Hopfield

patterns = np.array([
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

net = Hopfield(patterns=patterns)

query = np.array([1, -1, -1, -1])

print("Patterns:")
print(patterns)
print(f"Weights: ({net.N}):")
print(net.weights)
print(f"Query: {query}")

def printer(s, converged, epochs):
    print(f"Epoch {epochs}: {s[-1]}")

s, converged, epochs = net.evaluate(query=query, max_epochs=20, printer=printer)

if converged:
    print(f"Done! Converged after {epochs} epochs: {s[-1]}")
else:
    print(f"Failed to converge after {epochs} epochs: {s[-1]}")
