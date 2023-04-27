import numpy as np
from src.perceptron import Perceptron
import src.theta_funcs as theta_funcs
from src.trainer import train_perceptron, evaluate_perceptron
import src.error_funcs as error_funcs

'''
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
'''

dataset = [
    np.array([1, 1]),
    np.array([1, -1]),
    np.array([-1, 1]),
    np.array([-1, -1]),
]

'''
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
'''

dataset_outputs = [
    1,
    -1,
    -1,
    -1,
]


initial_w = np.random.random(len(dataset[0]) + 1) * 2 - 1
per = Perceptron(initial_weights=initial_w, theta_func=theta_funcs.simple)

print(f"Initialized perceptron with weights: ")
print(per.w)

evaluate_perceptron(perceptron=per, dataset=dataset, dataset_outputs=dataset_outputs, error_func=error_funcs.count_nonmatching, print_output=True)

print("--------------------------------------------------------------------------------")
print("--------------------------------- TRAINING TIME --------------------------------")
print("--------------------------------------------------------------------------------")
train_perceptron(perceptron=per, dataset=dataset, dataset_outputs=dataset_outputs, error_func=error_funcs.count_nonmatching, acceptable_error=0, learning_rate=0.1, max_epochs=100, use_batch_increments=False, print_every=10)
print("--------------------------------------------------------------------------------")
print("-------------------------------- DONE TRAINING ---------------------------------")
print("--------------------------------------------------------------------------------")

error = evaluate_perceptron(perceptron=per, dataset=dataset, dataset_outputs=dataset_outputs, error_func=error_funcs.count_nonmatching, print_output=True)
amount_ok = len(dataset) - error

epoch_idx = -420
print(f"Got {round(amount_ok * 100 / len(dataset), 2)}% accuracy after {epoch_idx} epoch{'' if epoch_idx == 1 else 's'} {'✅' if amount_ok==len(dataset) else '❌'}")
print(f"Final weights: {per.w}")