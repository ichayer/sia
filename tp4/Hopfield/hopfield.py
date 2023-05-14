import numpy as np

class Hopfield:
    def __init__(self, patterns: np.ndarray) -> None:
        self.p = len(patterns)
        self.N = len(patterns[0])
        self.weights = np.transpose(patterns) @ patterns / self.N * (1 - np.identity(self.N))
        if self.p * 20 / 3 > self.N:
            print('⚠️⚠️⚠️ WARNING: Hopfields algorithm received more patterns than recommended!')
        for i in range(len(patterns)):
            for j in range(i):
                if np.abs(np.dot(patterns[i], patterns[j])) < 0.9:
                    print(f'⚠️⚠️⚠️ WARNING: Hopfields algorithm non-somewhat-orthogonal patterns at indexes {i} and {j}: {patterns[i]}, {patterns[j]}')
    
    def evaluate(self, query: np.ndarray, max_epochs: int, printer = None):
        if len(query) != self.N:
            raise ValueError(f'Length of query vector {len(query)} must be equal to self.N {self.N}')
        
        s_history = [query]
        h_history = [self.energy_at(query)]
        converged = False
        epochs = 0
        
        if printer is not None:
            printer(s_history, h_history, converged, epochs)
                
        while not converged and epochs < max_epochs:
            s_history.append(np.sign(self.weights @ s_history[-1]))
            h_history.append(self.energy_at(s_history[-1]))
            epochs += 1
            converged = np.array_equal(s_history[-2], s_history[-1])
            if printer is not None:
                printer(s_history, h_history, converged, epochs)
        return s_history, h_history, converged, epochs
    
    def energy_at(self, s):
        return (self.weights @ s @ s) / -2.0
