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

def printer(s_history, h_history, converged, epochs):
    print(f"Epoch {epochs}: (energy {h_history[-1]}) {s_history[-1]}")

s_history, h_history, converged, epochs = net.evaluate(query=query, max_epochs=20, printer=printer)

print(f"{'Done! Converged' if converged else 'Failed to converge'} after {epochs} epochs: (energy {h_history[-1]}) {h_history[-1]}")
