from .base_layer import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        self.cache = None

    def forward(self, input_x: np.ndarray):
        # TODO: Implement RELU activation function forward pass
        # output =
        # Store the input in cache, required for backward pass
        self.cache = input_x.copy()
        output = np.maximum(0, self.cache)
        return output # (num_inputs, hidden_size)

    def backward(self, dout):
        # Load the input from the cache
        x_temp = self.cache
        # Calculate gradient for RELU
        dx = (x_temp > 0) * dout
        return dx
