import numpy as np

# NUMPY:
# Multiply one-by-one elements of two arrays, then sum everything up (same as dot product): a @ b


def theta(h: float) -> float:
    return 1 if h == 0 else np.sign(h)

learning_rate = 0.1

class simple_perceptron:
    def __init__(self, input_size: int, initial_weight_range: tuple[float, float] = [-1, 1]) -> None:
        if not input_size > 0:
            raise f'Error: Cannot create perceptron with {input_size} inputs'
        self.w = np.random.random(input_size + 1) * (initial_weight_range[1] - initial_weight_range[0]) + initial_weight_range[0]
        self.delta_w = np.zeros(input_size + 1)
    
    def evaluate(self, input) -> float:
        if len(input) + 1 != len(self.w):
            raise f'Error: specified {len(input)} inputs to a simple_perceptron with {len(self.w)} weights (there should be {len(self.w) - 1} inputs)'
        
        h = self.w[1:] @ input
        return theta(h - self.w[0])
    
    def evaluate_and_adjust(self, input, expected_output) -> float:
        output = self.evaluate(input)
        if output != expected_output:
            self.delta_w += 2 * learning_rate * expected_output * np.concatenate(([-1], input))
        
        return output
    
    def update_weights(self):
        self.w = self.w + self.delta_w
        self.delta_w.fill(0)


dataset = [
    np.array([1.1946, 3.8427]),
    np.array([0.8788, 1.6595]),
    np.array([1.1907, 1.6117]),
    np.array([1.4180, 3.8272]),
    np.array([0.2032, 1.9208]),
    np.array([2.7571, 1.0931]),
    np.array([4.7125, 2.8166]),
    np.array([3.9392, 1.1032]),
    np.array([1.2072, 0.8132]),
    np.array([3.4799, 0.9982]),
    np.array([0.4763, 0.1020]),
]

dataset_outputs = [
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
]

def evaluate_perceptron(perceptron: simple_perceptron, print_output: bool) -> int:
    amount_ok = 0
    for i in range(len(dataset)):
        output = perceptron.evaluate(dataset[i])
        expected = dataset_outputs[i]
        amount_ok += 1 if expected == output else 0
        if print_output:
            print(f"[{i}] {'✅' if output == expected else '❌'} expected: {expected} got: {output} data: {dataset[i]}")
    return amount_ok

per = simple_perceptron(2)
print(f"Initialized perceptron with weights: ")
print(per.w)

evaluate_perceptron(per, True)

print("--------------------------------------------------------------------------------")
print("--------------------------------- TRAINING TIME --------------------------------")
print("--------------------------------------------------------------------------------")

print_every = 10
max_epochs = 50
use_batch_increments = False
for epoch_idx in range(1, max_epochs+1):
    for i in range(len(dataset)):
        per.evaluate_and_adjust(dataset[i], dataset_outputs[i])
        if not use_batch_increments:
            per.update_weights()

    if epoch_idx % print_every == 0:
        print("--------------------------------------------------")
        print(f"RESULTS AFER EPOCH {epoch_idx} (weights {per.w})")
    amount_ok = evaluate_perceptron(per, epoch_idx % print_every == 0)
    if amount_ok == len(dataset):
        break
    
    if use_batch_increments:
        per.update_weights()

if epoch_idx < print_every:
    print("...")

print("--------------------------------------------------------------------------------")
print("-------------------------------- DONE TRAINING ---------------------------------")
print("--------------------------------------------------------------------------------")

evaluate_perceptron(per, True)

print(f"Got {round(amount_ok * 100 / len(dataset), 2)}% accuracy after {epoch_idx} epoch{'' if epoch_idx == 1 else 's'} {'✅' if amount_ok==len(dataset) else '❌'}")
print(f"Final weights: {per.w}")