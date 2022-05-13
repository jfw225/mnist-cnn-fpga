import numpy as np
from typing import Iterable, Tuple

from verilog.ml.layer import LayerSpec

from verilog.ml.model import Model
from verilog.core.vmodule import V_ConnSpec
from verilog.core.vspecial import V_Done, V_High, V_Low, V_Ready, V_Valid
from verilog.core.vsyntax import V_Array, V_FixedPoint
from verilog.testing.vtestbench import V_TB_Initial, V_Testbench
from verilog.core.vtypes import BitWidth, V_Block, V_Output, V_ParameterArray
from verilog.iterables.m10k import M10K
from verilog.testing.vwavedata import V_WaveData


class ModelTB(V_Testbench):

    def __init__(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        input_np: np.ndarray,
        *specs: Iterable[LayerSpec],
        **kwargs

    ):

        self.int_width = int_width
        self.dec_width = dec_width
        self.width = int_width + dec_width

        self.input_np = np.array(input_np)

        # holds (layer spec, expected input, expected output)
        self.specs: Iterable[Tuple[LayerSpec, np.ndarray, np.ndarray]] = list()
        self._create_specs(*specs)

        # get the input/output sizes for the model
        self.input_size = np.prod(self.input_np.shape)
        (*_, self.output_shape), *_, output_np = self.specs[-1]
        self.output_size = np.prod(self.output_shape)

        # create the input/output memories
        self.input_mem = M10K(self.width, self.input_size, name="input_mem")
        self.output_mem = M10K(self.width, self.output_size, name="output_mem")

        # set initial data for the input memory
        self.input_mem.set_init_data(V_Array(
            module=self.input_mem,
            dtype=V_ParameterArray,
            width=self.width,
            size=self.input_size,
            signed=True,
            data=[V_FixedPoint(v, self.int_width, self.dec_width)
                  for v in self.input_np.flatten()]
        ), save_mif=True)

        # set expected output as initial data for output memory
        self.output_mem.set_init_data(V_Array(
            module=self.output_mem,
            dtype=V_ParameterArray,
            width=self.width,
            size=self.output_size,
            signed=True,
            data=[V_FixedPoint(v, self.int_width, self.dec_width)
                  for v in output_np.flatten()]
        ))

        super().__init__(objects=[self.input_mem,
                                  self.output_mem], name="ModelTB", **kwargs)

        # create valid, done and ready lines for the model inputs
        self.model_valid = self.add_var(
            V_Valid(module=self, name="model_valid"))
        self.model_done = self.add_var(V_Done(module=self, name="model_done"))
        self.model_ready = self.add_var(
            V_Ready(module=self, name="model_ready"))

        # create the model
        self.model = Model(
            self.int_width,
            self.dec_width,
            self.input_mem,
            self.output_mem,
            *[spec[0] for spec in self.specs]
        )

        # write model to file
        self.model_file = self.model.tofile("model")

        # include model file
        self.include(self.model_file)

        self.out = self.port(V_Output, name="out")

    def generate(self) -> V_Block:
        # create a conn spec for the input memory
        input_mem_cs = V_ConnSpec[M10K](self,
                                        self.input_mem,
                                        prefix="input_mem",
                                        clk=self.clk,
                                        reset=self.reset
                                        )

        # create a conn spec for the output memory
        output_mem_cs = V_ConnSpec[M10K](self,
                                         self.output_mem,
                                         prefix="output_mem",
                                         clk=self.clk,
                                         reset=self.reset
                                         )

        # instantiate the input memory
        self.input_mem_ins = self.input_mem(self, *input_mem_cs)

        # instantiate the output memory
        self.output_mem_ins = self.output_mem(self, *output_mem_cs)

        # instantiate the model
        model_ins = self.model.instantiate(
            self,
            self.clk,
            self.reset,
            self.model_valid,
            self.model_done,
            self.model_ready
        )

        return super().generate(V_Block(
            "// drive model valid",
            *V_TB_Initial(
                "#20",
                self.model_valid.set(V_High),
                "#30",
                self.model_valid.set(V_Low)
            ),

            "\n// instantiate the input memory",
            *self.input_mem_ins,

            "\n// instantiate the output memory",
            *self.output_mem_ins,

            "\n// instantiate the model",
            *model_ins,

            "\n// assign the test bench done flag",
            self.done.set(self.model_done)
        ))

    def _create_specs(self, *specs: Iterable[LayerSpec]):
        # create temporary I/O memories
        input_mem = M10K(self.width, 1)
        output_mem = M10K(self.width, 1)

        # initialize expected layer inputs/outputs
        input_np, output_np = self.input_np, None

        for (LayerT, weights_np, biases_np, _, _) in specs:

            # put the input through the layer
            output_np = LayerT.forward(input_np, weights_np, biases_np)

            # create and store the spec along with expected input/output
            self.specs.append((
                LayerSpec(LayerT, weights_np, biases_np,
                          input_np.shape, output_np.shape),
                input_np,
                output_np))

            # update the input
            input_np = output_np

    def presim(self):
        # expect the layer outputs to be correct
        for i, (_, exp_input, exp_output) in enumerate(self.specs):
            exp_input = exp_input.flatten()
            exp_output = exp_output.flatten()

            layer = self.model.layers[i]

            # expect the done vlag to be `HIGH`
            self.expect(layer.done, V_High)

            # check the input
            self.expect(layer.input_mem.memory, [V_FixedPoint(
                v, layer.int_width, layer.dec_width) for v in exp_input])

            # check the output
            self.expect(layer.output_mem.memory, [V_FixedPoint(
                v, layer.int_width, layer.dec_width) for v in exp_output])

    def postsim(self, data: V_WaveData):
        """
        Function ran after the simulation. 
        """

        # print(data)
