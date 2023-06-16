from tp5.optimizers import *
from tp5.trainer import *
from tp5.perceptron import *
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
    config=config
)

for i in range(len(dataset_input)):
    to_predict = list(dataset_input.values())[i]
    encoder_output, decoder_output = mp.feed_forward(to_predict)
    print("Input: ", to_predict)
    print("Encoded: ", encoder_output)
    print("Decoded: ", decoder_output)
    print("Error: ", np.mean(np.abs(to_predict - decoder_output)))
    print("=====================================")
