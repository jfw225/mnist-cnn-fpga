from math import ceil, log2
from typing import Iterable, Optional, Tuple, Type
import numpy as np
from verilog.core.vmodule import V_Module
from verilog.core.vspecial import V_Clock, V_Done, V_Empty, V_Reset, V_Valid
from verilog.core.vstate import V_StateMachine
from verilog.core.vsyntax import V_Array, V_FixedPoint, V_ObjectBase
from verilog.core.vtypes import ArraySize, BitWidth, V_Block, V_DType, V_Input, V_Output, V_ParameterArray, V_Reg, V_Wire
from verilog.core.vtarget import V_Target
from verilog.iterables.m10k import M10K
from verilog.utils import id_generator


Weights = np.ndarray
Biases = np.ndarray
InputShape = Tuple[int]
OutputShape = Tuple[int]


class Layer(V_Target):
    def __init__(
        self,
        *,
        int_width: BitWidth,
        dec_width: BitWidth,
        weights_np: Optional[np.ndarray] = None,
        biases_np: Optional[np.ndarray] = None,
        input_mem: M10K,
        output_mem: M10K,
        input_shape: Optional["InputShape"],
        output_shape: Optional["OutputShape"],
        objects: Optional[Iterable[V_ObjectBase or V_Module]] = list()
    ):
        self.weights_np = np.array(
            weights_np) if weights_np is not None else None
        self.biases_np = np.array(biases_np) if biases_np is not None else None

        # convert the weights and biases to parameter arrays
        weights, biases = self.convert_model_params(
            int_width, dec_width, weights_np, biases_np)

        if self.weights_np is not None:
            assert isinstance(
                weights, V_Array) and weights.dtype is V_ParameterArray

            # adjust the name of the weights
            weights.name = f"{weights}_{id_generator()}"

        if self.biases_np is not None:
            assert self.biases_np.ndim == 1, self.biases_np
            assert isinstance(
                biases, V_Array) and biases.dtype is V_ParameterArray

            # adjust the name of the weights
            biases.name = f"{biases}_{id_generator()}"

        assert isinstance(input_mem, M10K)
        assert isinstance(output_mem, M10K)

        # save the bit width and the weights
        self.int_width = int_width
        self.dec_width = dec_width
        self.width = int_width + dec_width

        # store the weights and biases params
        self.weights = weights
        self.biases = biases

        # save the input and output memories
        self.input_mem = input_mem
        self.output_mem = output_mem

        # store the input and output shape
        self.input_shape = input_shape
        self.output_shape = output_shape

        # local objects
        local_objects = [obj for obj_np, obj in [
            (self.weights_np, self.weights), (self.biases_np, self.biases)] if obj_np is not None]
        super().__init__(objects=objects + local_objects)

        self._configure_nets()

    def convert_model_params(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        weights_np: np.ndarray,
        biases_np: np.ndarray
    ) -> Tuple[V_Array, V_Array]:
        """
        Transforms model weights `weights_np` and model biases `biases_np` to
        the format required by the module and outputs a `V_Array` object with
        data type `V_ParameterArray`. This functionshould be overloaded for
        each layer.
        """

        raise Exception(
            f"{self.convert_model_params} should be overloaded by {self}")

        return None, None

    @staticmethod
    def forward(
        x: np.ndarray,
        weights_np: Weights,
        biases_np: Biases
    ):
        """
        Performs the Python equivalent mapping for which this layer is
        supposed to do. This should be overloaded.
        """

        return x.dot(weights_np) + biases_np

    def generate(self, vsm: V_StateMachine, layer_code: V_Block) -> V_Block:
        """
        To be called by the subclass rather than `self.generate()`.
        """

        # if weights exist, set up the variables
        weights_vblock = V_Block()
        if self.weights_np is not None:
            weights_vblock = V_Block(
                "\n// tie the weight data to the weight address",
                self.w_data.set(self.weights[self.w_addr]),
            )

        # if biases exist, set up the variables
        biases_vblock = V_Block()
        if self.biases_np is not None:
            biases_vblock = V_Block(
                "\n// tie the bias data to the bias address",
                self.b_data.set(self.biases[self.b_addr])
            )

        return V_Block(
            *weights_vblock,

            *biases_vblock,

            "\n// instantiate the state machine",
            *vsm(self, self.clk, self.reset, self.done),

            # instantiate the layer code
            *layer_code
        )

    def _configure_nets(self):
        """
        Creates all of the ports and variables to constitute the base of a
        layer. Other ports and variables can be added as needed.
        """

        # configure the input ports
        inp_addr, inp_data = self.input_mem.read

        # create local copies input read
        self.inp_addr = self.add_port(inp_addr,
                                      port_type=V_Output,
                                      dtype=V_Reg,
                                      name=f"inp_addr")

        self.inp_data = self.add_port(inp_data,
                                      port_type=V_Input,
                                      dtype=V_DType,
                                      name=f"inp_data")

        # create local copies of input write
        # inp_write_addr, inp_write_data, inp_write_we = self.input_mem.write
        # self.inp_write_addr = self.add_port(inp_write_addr,
        #                                     port_type=V_Output,
        #                                     dtype=V_Reg,
        #                                     name=f"inp_write_addr")

        # self.out_data = self.add_port(inp_write_data,
        #                               port_type=V_Output,
        #                               dtype=V_Reg,
        #                               name=f"inp_write_data")

        # self.out_we = self.add_port(out_we,
        #                             port_type=V_Output,
        #                             dtype=V_Reg,
        #                             name=f"inp_write_we")

        # configure the output ports
        out_addr, out_data, out_we = self.output_mem.write

        # create local copies
        self.out_addr = self.add_port(out_addr,
                                      port_type=V_Output,
                                      dtype=V_Reg,
                                      name=f"out_addr")

        self.out_data = self.add_port(out_data,
                                      port_type=V_Output,
                                      dtype=V_Reg,
                                      name=f"out_data")

        self.out_we = self.add_port(out_we,
                                    port_type=V_Output,
                                    dtype=V_Reg,
                                    name=f"out_we")

        # create the weight variables
        self.w_addr, self.w_data = None, None
        if self.weights_np is not None:
            self.w_addr = self.var(dtype=V_Reg, width=ceil(
                log2(self.weights.size)), name="weights_ra")
            self.w_data = self.var(
                dtype=V_Wire, width=self.width, name="weights_rd")

        # create the bias variables
        self.b_addr, self.b_data = None, None
        if self.biases_np is not None:
            self.b_addr = self.var(dtype=V_Reg, width=ceil(
                log2(self.biases.size)), name="biases_ra")
            self.b_data = self.var(
                dtype=V_Wire, width=self.width, name="biases_rd")


class LayerSpec(Tuple[Type[Layer], Weights, Biases, InputShape, OutputShape]):
    def __new__(
        cls,
        layer: Type[Layer],
        weights: Weights,
        biases: Biases,
        input_shape: InputShape,
        output_shape: OutputShape
    ):
        return super().__new__(cls, [layer, weights, biases, input_shape, output_shape])
