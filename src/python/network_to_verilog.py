import numpy as np
from keras.datasets import mnist

from config import FIX_CONV_PARAMS, INPUTS, EXPECTED_OUTPUT
from verilog.utils import dec2bin
from verilog.core.vmodule import VDataSpec, write_config
from mnist_model_v1._layer1 import Layer1

"""
TODO:
- fixed mult for any size
- fixed division for any size
- exponential for any size
- sigmoid
- softmax

- probably clock it all
"""


def main():
    # load the weights
    saved_weights = np.load("weights.npz")
    w1, w2 = (saved_weights[f] for f in saved_weights.files)
    # print(w1.shape, w2.shape)

    (X, Y), (X_test, Y_test) = mnist.load_data()
    # X_test = X_test.reshape(-1, 28*28)
    # print(X_test.shape, Y_test.shape)
    # ko = X_test[0]
    # oo = ko.dot(w1)
    # print(ko.shape, w1.shape, oo.shape)
    x_test = X_test[0].reshape(28 * 28,)
    y_test = Y_test[0]

    bit_width = sum(FIX_CONV_PARAMS)
    input_spec = VDataSpec(
        bit_width,
        [dec2bin(x, *FIX_CONV_PARAMS) for x in x_test],
        INPUTS)

    layer1 = Layer1(bit_width, w1)
    layer1.write()

    output_spec = VDataSpec(
        bit_width,
        [dec2bin(y_test, *FIX_CONV_PARAMS)],
        EXPECTED_OUTPUT)

    write_config(
        input_spec,
        [layer1.get_weight_spec()],
        output_spec)


if __name__ == '__main__':
    main()
