import numpy as np
from mnist_model_v1._layer1 import Layer1
from verilog.core.vfile import V_FileWriter
from verilog.core.vmodule import V_Module
from verilog.core.vspecial import V_Clock, V_Done, V_Reset
from verilog.core.vsyntax import V_Array
from verilog.core.vtypes import V_DType, V_ParameterArray, V_Wire
from verilog.iterables.m10k import M10K
from verilog.utils import format_int


class Layer1TB(V_Module):

    def __init__(
        self,
        input_mem: M10K,
        output_mem: M10K,
        layer1: Layer1
    ):
        super().__init__()
        self.name = "Layer1TB"

        self.input_mem = input_mem
        self.output_mem = output_mem
        self.layer1 = layer1

        self.out = self.add_port(self.output_mem.read_data,
                                 dtype=V_DType,
                                 name="out")

    def generate(self):
        input_mem = self.input_mem
        output_mem = self.output_mem
        layer1 = self.layer1

        clk = self.add_var(V_Clock())
        reset = self.add_var(V_Reset())

        # wire connected to layer1's done flag
        layer1_done = self.add_var(layer1.done, name="layer1_done")

        input_we = self.add_var(input_mem.write_en, name="input_we")
        input_ra = self.add_var(input_mem.read_addr, name="input_ra")
        input_wa = self.add_var(input_mem.write_addr, name="input_wa")
        input_rd = self.add_var(input_mem.read_data,
                                dtype=V_Wire, name="input_rd")
        input_wd = self.add_var(input_mem.write_data, name="input_wd")

        weights_we = self.add_var(layer1.weights.write_en, name="weights_we")
        weights_ra = self.add_var(layer1.weights.read_addr, name="weights_ra")
        weights_wa = self.add_var(layer1.weights.write_addr, name="weights_wa")
        weights_rd = self.add_var(layer1.weights.read_data,
                                  dtype=V_Wire, name="weights_rd")
        weights_wd = self.add_var(layer1.weights.write_data, name="weights_wd")

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

        # instantiate the weights (move inside layer 1 later)
        weights_ins = self.layer1.weights(
            self,
            clk,
            reset,
            (weights_we, layer1.weights.write_en),
            (weights_ra, layer1.weights.read_addr),
            (weights_wa, layer1.weights.write_addr),
            (weights_rd, layer1.weights.read_data),
            (weights_wd, layer1.weights.write_data)
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
        layer1_ins = self.layer1.instantiate(
            self,
            clk,
            reset,
            layer1_done
        )

        return [
            "// initialize clock and reset, and drive reset",
            "initial begin",
            f"\t{clk.name} = 1'd0;",
            f"\t{reset.name} = 1'd1;",
            f"\t#20",
            f"\t{reset.name} = 1'd0;",
            "end\n",

            "// toggle the clocks",
            "always begin",
            f"\t#10",
            f"\t{clk.name} = !{clk.name};",
            "end\n\n",

            "\n\n// instantiate the input memory",
            *input_mem_ins,

            "\n\n// instantiate the weights",
            *weights_ins,

            "\n// instantiate the output mempry",
            *output_mem_ins,

            "\n// instantiate the first layer",
            *layer1_ins,

            "\n// tie output ra to output wa",
            # output_ra.set(output_wa),

            "\n\n// assign the output",
            self.out.set(output_rd)
        ]


if __name__ == '__main__':

    bit_width = 10
    input_size = 4

    weights = np.array([[*range(input_size - 1)] for _ in range(input_size)])
    _, output_size = weights.shape

    input_mem = M10K(width=bit_width, size=input_size)
    input_mem.set_init_data(V_Array(
        V_ParameterArray,
        bit_width,
        input_size,
        False,
        [format_int(i, bit_width) for i in range(input_size)]
    ))
    input_file = input_mem.tofile("input_mem")

    output_mem = M10K(width=bit_width, size=output_size)
    output_file = output_mem.tofile("output_mem")

    layer1 = Layer1(bit_width, weights, input_mem, output_mem)
    layer1_file = layer1.tofile("layer1")

    layer1_tb = Layer1TB(input_mem, output_mem, layer1)
    tb_file = V_FileWriter("layer1_tb", objects=[layer1_tb], includes=[
                           input_file, output_file, layer1_file])
    tb_file.write()
