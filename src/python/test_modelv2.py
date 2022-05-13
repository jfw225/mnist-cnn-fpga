import numpy as np
from knet_v2.maxpool import Maxpool
from knet_v2.conv2d import Conv2D
from knet_v2.maxpool_nosoft import Maxpool_NoSoft
from knet_v2.network import convolve, convolve_flat, forward, get_weights, load_data, maxpool, maxpool_flat, sigmoid
from keras.datasets import mnist
from verilog.core.vsyntax import V_FixedPoint
from verilog.ml.layer import LayerSpec

from verilog.ml.model_testbench import ModelTB
from verilog.utils import fixedbin2dec, fixedfloat2dec


def real_inputs():
    w1, b1, w3, b3 = get_weights()
    # print(w1.shape, b1.shape)

    # (X, Y), (X_test, Y_test) = mnist.load_data()
    # x_test = X_test[0]
    # y_test = Y_test[0]

    # x_test = x_test.reshape(1, *x_test.shape)
    x, y = load_data()
    x_test = x[2000] / 256
    y_test = y[2000]
    print(x_test.shape, y_test)

    return w1, b1, w3, b3, x_test, y_test


def fake_inputs():
    input_size = 3
    conv_filter_size = 2
    conv_filter_len = 4

    conv_input = np.arange(
        input_size * input_size).reshape(input_size, input_size)

    w1 = np.arange(conv_filter_size ** 2 * conv_filter_len).reshape(
        conv_filter_size, conv_filter_size, conv_filter_len)
    b1 = np.arange(conv_filter_len)

    # mp_input = sigmoid(convolve_flat(conv_input, w1))
    mp_input = np.arange(5 * 5 * 4).reshape(5, 5, 4)
    mp_input = mp_input - mp_input.mean()

    w3_d1 = np.prod(maxpool_flat(mp_input).shape)
    w3_d2 = 10
    w3 = np.arange(w3_d1 * w3_d2).reshape(w3_d1, w3_d2) * 0.001
    b3 = np.arange(w3_d2) * 0.001

    return [w1, b1, w3, b3, conv_input, mp_input]
    # return [0.01 * v for v in [w1, b1, w3, b3, x_test]]


class Model2TB(ModelTB):

    def presim(self):
        # w1, b1, w3, b3, conv_input, _ = real_inputs()
        # w1, b1, w3, b3, conv_input, mp_input = fake_inputs()

        # layer1: Conv2D = self.model.layers[0]
        # # layer2: Maxpool = self.model.layers[1]
        # layer2: Maxpool = self.model.layers[0]

        # out = convolve_flat(conv_input, w1)
        # mp_input = sigmoid(out)
        # mp_out = maxpool_flat(mp_input)

        # # # # self.log(layer2.window)
        # self.expect(layer2.mp_out, [V_FixedPoint(
        #     v, self.int_width, self.dec_width) for v in mp_out.flatten()])

        # self.log(layer2.curr_y)
        # self.log(layer2.out_y)
        # self.log(layer2.curr_x)
        # self.log(layer2.out_x)

        # self.log(layer2.out_addr)
        # self.log(layer2.out_data)
        # self.log(layer2.out_we)
        # self.log(layer2.w_addr)
        # self.log(layer2.w_data)
        # self.log(layer2.b_addr)
        # self.log(layer2.b_data)

        # self.log(layer2.prod)
        # self.log(layer2.dp)
        # self.log(layer2.exp_reset)
        # self.log(layer2.exp_done)
        # self.log(layer2.exp_out)
        # self.log(layer2.exp_sum)
        # self.log(layer2.exp_arr)

        # self.log(layer2.div_reset)
        # self.log(layer2.div_done)

        return super().presim()

    def postsim(self, data):
        *_, exp_out = self.specs[-1]
        print(exp_out)

        output = data[self.signal_of_obj(self.output_mem.memory)].iloc[-1]

        m_i, m_e, m_o = 0, 0, 0
        for i, (e, o) in enumerate(zip(exp_out, output)):
            d = fixedbin2dec(o, self.int_width, self.dec_width)

            print(f"Index: {i} | Expected: {e} | Actual: {d}")

            if d > m_o:
                m_i = i
                m_e = e
                m_o = d

        print(f" Predicted: {m_i} | ({m_e}, {m_o})")


if __name__ == '__main__':
    int_width = 44  # 44  # 5
    dec_width = 44  # 27
    width = int_width + dec_width

    w1, b1, w3, b3, conv_input, y_test = real_inputs()
    # w1, b1, w3, b3, conv_input, mp_input = fake_inputs()

    # out = convolve_flat(conv_input, w1)
    # mp_input = sigmoid(out + b1)

    model_tb = Model2TB(int_width, dec_width, conv_input,
                        LayerSpec(Conv2D, w1, b1, -1, -1),
                        LayerSpec(Maxpool_NoSoft, w3, b3, -1, -1),
                        timeout=1e8)

    # model_tb = Model2TB(int_width, dec_width, mp_input,
    #                     LayerSpec(Maxpool_NoSoft, w3, b3, -1, -1),
    #                     timeout=1e8)
    model_tb.simulate(headless=True)
    # model_tb._simulator.compile()

    # mp_out = maxpool_flat(conv_out)

    # mp_out_alt = mp_out.reshape(-1)
    # Z3 = mp_out_alt.dot(w3)

    # mp_inp_flat = mp_input.flatten().reshape(-1, mp_input.shape[-1])
    # mp_inp_flat = mp_input.flatten()
    # print(mp_input)
    # print(mp_inp_flat)
    # print(mp_input.shape)
    # # exit()
    # xMax, yMax, zMax = mp_input.shape
    # for x in range(xMax):
    #     for y in range(yMax):
    #         for z in range(zMax):
    #             a = mp_input[x, y, z]
    #             b = mp_inp_flat[z + y * zMax + x * yMax * zMax]

    #             assert a == b
    #             # print(a, b)

    # y, x, n = mp_input.shape
    # for j in range(y):
    #     for i in range(x):
    #         for k in range(n):
    #             a = mp_input[j, i, k]
    #             b = mp_inp_flat[k + i * n + j * x * n]

    #             assert a == b

    # exit()
    # print(mp_input, mp_input.shape)
    # print("----")
    # mp_out = maxpool_flat(mp_input)
    # print(mp_out.shape, np.prod(mp_out.shape))

    # mp_reshp = mp_out.reshape(-1)
    # dot_out = mp_reshp.dot(w3)
    # # print(mp_reshp.shape, w3.shape, dot_out.shape, w3)
    # print(w3.flatten())
    # exit()

    # for i, mp in enumerate([V_FixedPoint(v, 44, 44) for v in dot_out.flatten()]):
    # for a, b in zip(mp_reshp, w3[i]):
    #     c = a * b
    #     aa = str(V_FixedPoint(a, 44, 44)).replace(
    #         "88'b", "").replace("_", "")
    #     bb = str(V_FixedPoint(b, 44, 44)).replace(
    #         "88'b", "").replace("_", "")
    #     cc = str(V_FixedPoint(c, 44, 44)).replace(
    #         "88'b", "").replace("_", "")

    #     print(f"a: {int(aa, 2)}")
    #     print(f"b: {int(bb, 2)}")
    #     print(f"c: {int(cc, 2)}")
    #     print("----")
    # print("=======")
    # k = str(mp).replace("88'b", "").replace("_", "")
    # print(f"DP: {int(k, 2)}")

    # print(dot_out)
    # print(b3)
    # print(dot_out + b3)
    # for i, mp in enumerate([V_FixedPoint(v, 44, 44) for v in np.exp(dot_out + b3).flatten()]):
    # for a, b in zip(mp_reshp, w3[i]):
    #     c = a * b
    #     aa = str(V_FixedPoint(a, 44, 44)).replace(
    #         "88'b", "").replace("_", "")
    #     bb = str(V_FixedPoint(b, 44, 44)).replace(
    #         "88'b", "").replace("_", "")
    #     cc = str(V_FixedPoint(c, 44, 44)).replace(
    #         "88'b", "").replace("_", "")

    #     print(f"a: {int(aa, 2)}")
    #     print(f"b: {int(bb, 2)}")
    #     print(f"c: {int(cc, 2)}")
    #     print("----")
    # print("=======")
    # k = str(mp).replace("88'b", "").replace("_", "")
    # print(f"Exp: {int(k, 2)}")

    # expect this
    # for mp in [V_FixedPoint(v, 44, 44) for v in mp_out.flatten()]:
    #     k = str(mp).replace("88'b", "").replace("_", "")
    #     print(int(k, 2))
