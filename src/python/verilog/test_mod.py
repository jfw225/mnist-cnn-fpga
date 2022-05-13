
from mnist_model_v1._layer1 import Layer1
from verilog.core.vspecial import V_Clock, V_Done, V_Reset
from verilog.iterables.m10k import M10K
from verilog.core.vmodule import V_Module
from verilog.core.vmap import V_Map
from verilog.core.vsyntax import V_Always, V_Connection, V_Else, V_If, V_Variable
from verilog.core.vtypes import V_DType, V_Output, V_PosEdge, V_Reg, V_Wire
from verilog.targets.mult import Mult

"""
TODO: 
- `next` flag for giving next round of input data
- `done` flag for output is valid
- on `next` increment input address
- on `done` increment output address
- target might need a signal to know that new data is ready
- target might also need a signal that increments vs sets address to base

- maybe have `next` flag for each input

- have layer be a target function to a vmap
- inside layer, have another vmap
"""


class TestMod(V_Module):

    def __init__(
        self,
        input_mem: M10K,
        layer1: Layer1
    ):
        self.input_mem = input_mem
        self.layer1 = layer1

        self.output_mem = M10K(self.input_mem.width, self.input_mem.size)
        self.target = Mult(self.input_mem.width)
        self.vmap = V_Map([self.input_mem, self.input_mem],
                          [self.output_mem], self.target)
        super().__init__(objects=[self.output_mem, self.target, self.vmap])

        self.out = self.add_port(self.input_mem.read_data, name="out")
        self.out.dtype = V_DType

    def generate(self):
        input_mem = self.input_mem
        output_mem = self.output_mem
        layer1 = self.layer1

        clk = self.add_var(V_Clock())
        reset = self.add_var(V_Reset())
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

        input_we_reg = self.add_var(input_we, dtype=V_Reg, name="input_re_reg")
        input_ra_reg = self.add_var(input_ra, dtype=V_Reg, name="input_ra_reg")

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
        layer1_ins = self.layer1.instantiate(
            self,
            clk,
            reset,
            input_we,
            input_ra,
            input_wa,
            input_rd,
            input_wd
        )

        # wire connected to vmap's done flag
        vmap_done = self.add_var(V_Done("vmap_done"))

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

            "\n// instantiate the output mempry",
            *output_mem_ins,

            "\n// instantiate the vmap",
            *self.vmap.instantiate(self, clk, reset, vmap_done),

            "\n// instantiate the first layer",
            # *layer1_ins,

            "\n// tie output ra to output wa",
            output_ra.set(output_wa),

            "\n\n// assign the output",
            self.out.set(output_rd)
        ]
