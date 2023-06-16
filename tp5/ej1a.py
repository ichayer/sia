from matplotlib import pyplot as plt
from tp5.optimizers import *
from tp5.trainer import *
from tp5.perceptron import *
from tp4.Hopfield.pattern_loader import *

fonts_headers = np.array(
    ["`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
     "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL"]
)

def graph_fonts(original, decoded):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Decoded')

    ax1.imshow(np.array(original).reshape((7, 5)), cmap='gray')
    ax1.set_xticks(np.arange(0, 5, 1))
    ax1.set_yticks(np.arange(0, 7, 1))
    ax1.set_xticklabels(np.arange(1, 6, 1))
    ax1.set_yticklabels(np.arange(1, 8, 1))

    ax2.imshow(np.array(decoded).reshape((7, 5)), cmap='gray')
    ax2.set_xticks(np.arange(0, 5, 1))
    ax2.set_yticks(np.arange(0, 7, 1))
    ax2.set_xticklabels(np.arange(1, 6, 1))
    ax2.set_yticklabels(np.arange(1, 8, 1))

    fig.show()

def graph_latent_space(dots):
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    for j in range(dots.__len__()):
        plt.scatter(dots[j][0], dots[j][1])
        plt.annotate(fonts_headers[j], xy=dots[j], xytext=(dots[j][0] + 0.05, dots[j][1] + 0.05), fontsize=12)
    plt.show()

def create_multilayer_perceptron(perceptrons_by_layer : list[int], config : TrainerConfig, dataset_input : dict[str, ndarray]):
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
    return MultilayerPerceptron(perceptrons, lambda: Adam())

def exercise_a(perceptrons_by_layer: list[int], limit = fonts_headers.size - 1):

    if not perceptrons_by_layer:
        raise Exception("Perceptrons by layer is null")

    config = TrainerConfig.from_file("config.json")
    dataset_input = load_pattern_map('characters.txt')

    mp = create_multilayer_perceptron(perceptrons_by_layer, config, dataset_input)

    result = train_multilayer_perceptron(
        multilayer_perceptron=mp,
        dataset=list(dataset_input.values())[0:limit],
        config=config
    )

    dots = []

    for i in range(limit):
        to_predict = list(dataset_input.values())[i]
        encoder_output, decoder_output = mp.feed_forward(to_predict)

        print("Input: ", to_predict)
        print("Encoded: ", encoder_output)
        print("Decoded: ", decoder_output)
        print("Error: ", np.mean(np.abs(to_predict - decoder_output)))
        print("=====================================")

        # 1.2)
        graph_fonts(to_predict, decoder_output)

        dot = (encoder_output[0], encoder_output[1])
        dots.append(dot)

    # 1.3)
    graph_latent_space(dots)

if __name__ == "__main__":
    exercise_a([35, 10, 2, 10, 35], 2)