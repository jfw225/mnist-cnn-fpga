
import numpy as np
from typing import Optional
from knet_v3.conv2dsum1d import Conv2DSum1D
from knet_v3.maxpoolfc import Maxpool_FC
from knet_v3.conv2d import Conv2D
from knet_v3.maxpool import Maxpool
from knet_v3.network import convolve_flat, get_weights, load_data, maxpool_flat
from verilog.core.vspecial import V_Done, V_Valid
from verilog.core.vsyntax import V_Array, V_FixedPoint
from verilog.core.vtypes import BitWidth, V_Block, V_Output, V_ParameterArray, V_Wire
from verilog.ml.layer import LayerSpec
from verilog.testing.vtestbench import V_Testbench
from verilog.ml.model_testbench import ModelTB
from verilog.utils import fixedbin2dec


def real_inputs():
    w1, b1, w3, b3, w5, b5 = get_weights()
    # print(w1.shape, b1.shape)

    # (X, Y), (X_test, Y_test) = mnist.load_data()
    # x_test = X_test[0]
    # y_test = Y_test[0]

    # x_test = x_test.reshape(1, *x_test.shape)
    x, y = load_data()
    x_test = x[4000] / 256
    y_test = y[4000]
    # print(x_test.shape, y_test)

    return w1, b1, w3, b3, w5, b5, x_test, y_test


class Model3TB(ModelTB):
    """
    """

    def presim(self):
        """"""
        # l1: Conv2D = self.model.layers[0]
        # self.log(l1.jj)
        # self.log(l1.r)
        # self.log(l1.w_addr)
        # self.log(l1.w_data)
        # self.log(l1.b_addr)
        # self.log(l1.b_data)
        # self.log(l1.out_addr)
        # self.log(l1.out_data)

        l3: Conv2DSum1D = self.model.layers[0]

        w1, b1, w3, b3, w5, b5, conv_input, y_test = real_inputs()

        mp_inpt = convolve_flat(conv_input, w1) + b1
        x = maxpool_flat(mp_inpt)
        *_, c = x.shape

        Z1 = np.array([convolve_flat(x[:, :, i], w3[:, :, :, i])
                       for i in range(c)])

        # self.expect(l3.conv_out_arr, [V_FixedPoint(
        #     v, self.int_width, self.dec_width) for v in Z1.flatten()])

        # self.log(l3.wc)
        # self.log(l3.oc)
        # self.log(l3.tj)
        # self.log(l3.ti)
        # self.log(l3.sum_arr)
        # self.log(l3.out_addr)
        # self.log(l3.out_we)
        # self.log(l3.cr)

        return super().presim()

    def postsim(self, data):
        # *_, y_test = real_inputs()
        *_, exp_out = self.specs[-1]
        # print(exp_out)
        y_test = np.argmax(exp_out)

        output = data[self.signal_of_obj(self.output_mem.memory)].iloc[-1]

        m_i, m_e, m_o = 0, 0, 0
        for i, (e, o) in enumerate(zip(exp_out, output)):
            d = fixedbin2dec(o, self.int_width, self.dec_width)

            print(f"Index: {i} | Expected: {e} | Actual: {d}")

            if d > m_o:
                m_i = i
                m_e = e
                m_o = d

        print(f"Actual: {y_test} | Predicted: {m_i} | ({m_e}, {m_o})\n\n\n\n")


def export_sim(img):
    int_width = 10  # 44
    dec_width = 10  # 44
    w1, b1, w3, b3, w5, b5, conv_input, y_test = real_inputs()

    model_tb = Model3TB(int_width, dec_width, img,
                        LayerSpec(Conv2D, w1, b1, -1, -1),
                        LayerSpec(Maxpool, None, None, -1, -1),
                        LayerSpec(Conv2DSum1D, w3, b3, -1, -1),
                        LayerSpec(Maxpool_FC, w5, b5, -1, -1),
                        timeout=1e8)
    model_tb.model.name = "ModelV3"
    model_tb.simulate(headless=False)


if __name__ == '__main__':
    int_width = 6  # 44
    dec_width = 10  # 44

    w1, b1, w3, b3, w5, b5, conv_input, y_test = real_inputs()

    # mp_input = convolve_flat(conv_input, w1) + b1
    # conv_input = maxpool_flat(mp_input)

    # model_tb = Model3TB(int_width, dec_width, conv_input,
    #                     # LayerSpec(Conv2D, w1, b1, -1, -1),
    #                     # LayerSpec(Maxpool, None, None, -1, -1),
    #                     LayerSpec(Conv2DSum1D, w3, b3, -1, -1),
    #                     # LayerSpec(Maxpool_FC, w5, b5, -1, -1),
    #                     timeout=1e8)
    model_tb = Model3TB(int_width, dec_width, conv_input,
                        LayerSpec(Conv2D, w1, b1, -1, -1),
                        LayerSpec(Maxpool, None, None, -1, -1),
                        LayerSpec(Conv2DSum1D, w3, b3, -1, -1),
                        LayerSpec(Maxpool_FC, w5, b5, -1, -1),
                        timeout=1e8)
    model_tb.model.name = "ModelV3"
    model_tb.simulate(headless=True)
    # model_tb._simulator.compile()

    # import json
    # print(json.dumps(model_tb.nbits, indent=2))
