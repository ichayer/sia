from tp5.optimizers import *
from tp5.perceptron import *
from tp5.trainer import *
import matplotlib.pyplot as plt
from tp4.Hopfield.pattern_loader import *

config = TrainerConfig.from_file("config.json")

dataset_input = load_pattern_map('characters.txt')

perceptrons_by_layer = [35, 10, 2, 10, 35]
perceptrons = []

for i in range(len(perceptrons_by_layer)):
    perceptrons.append([])
    for p in range(perceptrons_by_layer[i]):
        perceptrons[-1].append([])

for i in range(len(perceptrons_by_layer)):
    for j in range(perceptrons_by_layer[i]):
        if i == 0:
            perceptrons[i][j] = Perceptron(
                initial_weights=np.random.random(len(dataset_input['a']) + 1) * 0.8 - 0.4,
                theta_func=config.theta
            )
        else:
            perceptrons[i][j] = Perceptron(
                initial_weights=np.random.random(perceptrons_by_layer[i - 1] + 1) * 0.8 - 0.4,
                theta_func=config.theta
            )


mp = MultilayerPerceptron(perceptrons, lambda: Adam())

result = train_multilayer_perceptron(
    multilayer_perceptron=mp,
    dataset=list(dataset_input.values()),
    dataset_outputs=list(dataset_input.values()),
    config=config
)


# print(f"\nEpoch: {result.epoch_num}, End Reason: {result.end_reason}, Error: {result.error_history[-1]:.4f}\n")

# print(f"-------Evaluating after training-------\n")
# avg_err = evaluate_multilayer_perceptron(
#     multilayer_perceptron=mp,
#     dataset=dataset_input[8:],
#     dataset_outputs=dataset_outputs[8:],
#     print_output=True,
# )

# print(f"\nMultilayer perceptron after training for {result_parity.epoch_num} epoch{''if result_parity.epoch_num == 1 else 's'} has "
#       f"an average error of {avg_err} {'✅' if avg_err <=config.acceptable_error else '❌'}\n")