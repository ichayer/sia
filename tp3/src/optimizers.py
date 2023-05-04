class Optimizer:
    def __init__(self):
        pass

    def adjust(self, perceptron, gt, learning_rate):
        raise NotImplementedError

class GradientDescent(Optimizer):
    def __init__(self):
        super().__init__()

    def adjust(self, perceptron, gt, learning_rate):
      return learning_rate * gt
      
class Momentum(Optimizer):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def adjust(self, perceptron, gt, learning_rate):
      return learning_rate * gt - self.alpha * perceptron.previous_delta_w