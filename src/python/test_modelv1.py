
import numpy as np
from keras.datasets import mnist

from verilog.ml.layer import LayerSpec
from knet_v1.layer1 import Layer1
from knet_v1.layer2 import Layer2
from verilog.ml.model_testbench import ModelTB


if __name__ == '__main__':
    int_width = 44  # 5
    dec_width = 44  # 27
    width = int_width + dec_width

    input_size_l1 = 4
    input_size_l2 = 3
    output_size = 5

    input_np = np.array([*range(input_size_l1)])
    weights1_np = 0.01 * np.array([[*range(input_size_l2)]
                                   for _ in range(input_size_l1)])

    weights2_np = 0.01 * np.array([[*range(output_size)]
                                   for _ in range(input_size_l2)])
    _, output2_size = weights2_np.shape

    # saved_weights = np.load("weights.npz")
    # w1, w2 = (saved_weights[f] for f in saved_weights.files)
    # # print(w1.shape, w2.shape)

    # (X, Y), (X_test, Y_test) = mnist.load_data()
    # x_test = X_test[0].reshape(28 * 28,)
    # y_test = Y_test[0]

    # input_np = x_test
    # weights1_np = w1
    # weights2_np = w2

    model_tb = ModelTB(int_width, dec_width, input_np,
                       LayerSpec(Layer1, weights1_np, [], -1, -1),
                       LayerSpec(Layer2, weights2_np, [], -1, -1))
    model_tb.simulate(headless=True)
