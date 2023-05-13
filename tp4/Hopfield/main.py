import numpy as np

patterns = np.array([
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

N = len(patterns[0])
weights = np.transpose(patterns) @ patterns / N * (1 - np.identity(N))

query = np.array([1, -1, -1, -1])

print("Patterns:")
print(patterns)
print(f"Weights: ({N}):")
print(weights)
print(f"Query: {query}")

s = [query]
converged = False
epochs = 0
max_epochs = 20
while not converged and epochs < max_epochs:
    if epochs % 1 == 0:
        print(f"Epoch {epochs}: {s[-1]}")
    s.append(np.sign(weights @ s[-1]))
    epochs += 1
    converged = np.array_equal(s[-2], s[-1])

if epochs < max_epochs:
    print(f"Done! Converged after {epochs} epochs: {s[-1]}")
else:
    print(f"Failed to converge after {epochs} epochs: {s[-1]}")
