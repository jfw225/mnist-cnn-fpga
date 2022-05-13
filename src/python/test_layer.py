import numpy as np
from mnist_model_v1.layer2 import Layer2
from verilog.core.vfile import V_FileWriter
from verilog.core.vmodule import V_Module
from verilog.core.vspecial import V_Clock, V_High, V_Reset
from verilog.core.vsyntax import V_Array, V_FixedPoint
from verilog.core.vtypes import V_Block, V_DType, V_ParameterArray, V_Wire
from verilog.iterables.m10k import M10K
from mnist_model_v1.layer1 import Layer1


class LayerTB(V_Module):

    def __init__(
        self,
        input_mem: M10K,
        output_mem: M10K,
        layer: Layer1
    ):
        super().__init__()
        self.name = "LayerTB"

        self.input_mem = input_mem
        self.output_mem = output_mem
        self.layer = layer

        self.out = self.add_port(self.output_mem.read_data,
                                 dtype=V_DType,
                                 name="out")

    def generate(self) -> V_Block:
        input_mem = self.input_mem
        output_mem = self.output_mem
        layer = self.layer

        clk = self.add_var(V_Clock())
        reset = self.add_var(V_Reset())

        # wire connected to layer's valid and done flags
        layer_valid = self.add_var(layer.valid, name="layer_valid")
        layer_done = self.add_var(layer.done, name="layer_done")

        input_we = self.add_var(input_mem.write_en, name="input_we")
        input_ra = self.add_var(input_mem.read_addr, name="input_ra")
        input_wa = self.add_var(input_mem.write_addr, name="input_wa")
        input_rd = self.add_var(input_mem.read_data,
                                dtype=V_Wire, name="input_rd")
        input_wd = self.add_var(input_mem.write_data, name="input_wd")

        output_we = self.add_var(output_mem.write_en, name="output_we")
        output_ra = self.add_var(output_mem.read_addr, name="output_ra")
        output_wa = self.add_var(output_mem.write_addr, name="output_wa")
        output_rd = self.add_var(output_mem.read_data,
                                 dtype=V_Wire, name="output_rd")
        output_wd = self.add_var(output_mem.write_data, name="output_wd")

        # instantiate the input m10k
        input_mem_ins = self.input_mem(
            self,
            clk,
            reset,
            (input_we, input_mem.write_en),
            (input_ra, input_mem.read_addr),
            (input_wa, input_mem.write_addr),
            (input_rd, input_mem.read_data),
            (input_wd, input_mem.write_data)
        )

        # instantiate the output m10k
        output_mem_ins = self.output_mem(
            self,
            clk,
            reset,
            (output_we, output_mem.write_en),
            (output_ra, output_mem.read_addr),
            (output_wa, output_mem.write_addr),
            (output_rd, output_mem.read_data),
            (output_wd, output_mem.write_data)
        )

        # instantiate the first layer
        layer_ins = layer.instantiate(
            self,
            clk,
            reset,
            layer_valid,
            layer_done
        )

        return V_Block(
            "// initialize clock and reset, and drive reset",
            "initial begin",
            f"\t{clk.name} = 1'd0;",
            f"\t{reset.name} = 1'd1;",
            f"\t#20",
            "\t" + layer_valid.set(V_High),
            f"\t{reset.name} = 1'd0;",
            "end\n",

            "// toggle the clocks",
            "always begin",
            f"\t#10",
            f"\t{clk.name} = !{clk.name};",
            "end\n\n",

            "\n\n// instantiate the input memory",
            *input_mem_ins,

            "\n// instantiate the output mempry",
            *output_mem_ins,

            "\n// instantiate the first layer",
            *layer_ins,

            "\n\n// assign the output",
            self.out.set(output_rd)
        )


def sigmoid(x):
    return 1/(np.exp(-x)+1)


def softmax(x):
    exp_element = np.exp(x)
    return exp_element / np.sum(exp_element, axis=0)


if __name__ == '__main__':
    int_width = 44  # 5
    dec_width = 44  # 27
    width = int_width + dec_width

    input_size = 4
    input_np = np.array([*range(input_size)])
    weights1_np = 0.01 * np.array([[*range(input_size - 1)]
                                   for _ in range(input_size)])
    _, output1_size = weights1_np.shape

    weights2_np = 0.01 * np.array([[*range(output1_size + 1)]
                                   for _ in range(output1_size)])
    _, output2_size = weights2_np.shape

    input_mem = M10K(width=width, size=input_size)
    input_mem.set_init_data(V_Array(
        V_ParameterArray,
        width,
        input_size,
        True,
        [V_FixedPoint(i, int_width, dec_width) for i in input_np]
    ))
    input_file = input_mem.tofile("input_mem")

    # output 1
    cor_out1_np = sigmoid(input_np.dot(weights1_np))
    assert output1_size == len(cor_out1_np)

    output1_mem = M10K(width=width, size=output1_size)
    output1_mem.set_init_data(V_Array(
        V_ParameterArray,
        width,
        output1_size,
        True,
        [V_FixedPoint(i, int_width, dec_width) for i in cor_out1_np],
        name="cor_out1"
    ))
    output1_file = output1_mem.tofile("output1_mem")

    # output 2
    cor_out2_np = softmax(cor_out1_np.dot(weights2_np))
    assert output2_size == len(cor_out2_np)

    output2_mem = M10K(width=width, size=output2_size)
    output2_mem.set_init_data(V_Array(
        V_ParameterArray,
        width,
        output2_size,
        True,
        [V_FixedPoint(i, int_width, dec_width) for i in cor_out2_np],
        name="cor_out2"
    ))
    output2_file = output2_mem.tofile("output2_mem")

    # layer1
    layer1 = Layer1(int_width, dec_width, weights1_np, input_mem, output1_mem)
    layer1_file = layer1.tofile("layer1")

    # layer2
    layer2 = Layer2(int_width, dec_width, weights2_np,
                    output1_mem, output2_mem)
    layer2_file = layer2.tofile("layer2")

    # layer_tb = LayerTB(input_mem, output1_mem, layer1)
    layer_tb = LayerTB(output1_mem, output2_mem, layer2)
    tb_file = V_FileWriter("layer_tb", objects=[layer_tb], includes=[
                           input_file, output1_file, output2_file, layer1_file, layer2_file])
    tb_file.write()
