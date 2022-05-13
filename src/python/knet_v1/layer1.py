from typing import Optional
import numpy as np
from verilog.ml.layer import Biases, InputShape, Layer, OutputShape, Weights
from verilog.core.vspecial import V_Done, V_Empty, V_High, V_Low, V_Reset, V_Valid
from verilog.core.vstate import V_StDone, V_State, V_StateMachine

from verilog.core.vsyntax import V_Array, V_Else, V_FixedPoint, V_If
from verilog.core.vtypes import BitWidth, V_Block, V_ParameterArray, V_Reg, V_Wire
from verilog.iterables.m10k import M10K
from verilog.targets.signed_dot_product import SignedDotProduct
from verilog.targets.signed_mult import SignedMult
from verilog.targets.signed_sigmoid import SignedSigmoid


class Layer1(Layer):

    def __init__(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        weights_np: np.ndarray,
        biases_np: np.ndarray,
        input_mem: M10K,
        output_mem: M10K,
        input_shape: Optional[InputShape],
        output_shape: Optional[OutputShape]
    ):
        """
        Let the weights be an array of shape `(n, m)`. Weights are expected to
        be in the following format.

        Suppose the input to the model is
            `I=[v_1, ..., v_n]`.

        Then the weights must be of the form
        `W=[
            [w^1_1, ..., w^1_m],
            ...,
            [w^n_1, ..., w^n_m]
        ]`.
        """

        # create the mult and sigmoid modules
        self.signed_mult = SignedMult(int_width, dec_width)
        self.sigmoid = SignedSigmoid(int_width, dec_width)

        super().__init__(
            int_width=int_width,
            dec_width=dec_width,
            weights_np=weights_np,
            input_mem=input_mem,
            output_mem=output_mem,
            input_shape=input_shape,
            output_shape=output_shape,
            objects=[self.signed_mult, self.sigmoid]
        )

        # create variable to hold the value of the dot product
        self.dp = self.add_var(self.signed_mult.out,
                               dtype=V_Reg, name="dot_product")

        # create variable to hold the output of the multiplier
        self.prod = self.add_var(self.signed_mult.out,
                                 dtype=V_Wire, name="prod")

        # create reset, valid, and done flags for `self.sigmoid`
        self.sig_reset = self.add_var(
            V_Reset(module=self, name="sig_reset"), dtype=V_Reg)
        self.sig_valid = self.add_var(
            V_Valid(module=self, name="sig_valid"), dtype=V_Reg)
        self.sig_done = self.add_var(
            V_Done(module=self, name="sig_done"), dtype=V_Wire)

    def convert_model_params(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        weights_np: np.ndarray,
        biases_np: np.ndarray
    ):

        assert weights_np.ndim == 2

        w_flat = [V_FixedPoint(v, int_width, dec_width)
                  for v in weights_np.T.flatten()]

        (n, m) = weights_np.shape

        return V_Array(
            module=self,
            dtype=V_ParameterArray,
            width=int_width + dec_width,
            size=n * m,
            signed=True,
            data=w_flat,
            name="layer1_weights"
        ), None

    @staticmethod
    def forward(
        x: np.ndarray,
        weights_np: Weights,
        biases_np: Biases
    ):

        def sigmoid(x):
            return 1/(np.exp(-x)+1)

        return sigmoid(x.dot(weights_np))

    def generate(self) -> V_Block:
        mult, sig = self.signed_mult, self.sigmoid

        vsm = V_StateMachine(_StReset, _StWaitValid,
                             _StWaitDotProduct, _StWaitSigmoid, _StWriteData)

        return super().generate(vsm, V_Block(
            "\n// instantiate the multiplier",
            *mult.instantiate(
                self,
                (V_Empty(), mult.clk),
                (V_Empty(), mult.reset),
                (V_Empty(), mult.valid),
                (V_Empty(), mult.done),
                (self.inp_data, mult.input_ports[0]),
                (self.w_data, mult.input_ports[1]),
                (self.prod, mult.output_ports[0])

            ),

            "\n// instantiate the sigmoid module",
            *sig.instantiate(
                self,
                self.clk,
                (self.sig_reset, sig.reset),
                (self.sig_valid, sig.valid),
                (self.sig_done, sig.done),
                (self.dp, sig.input),
                (self.out_data, sig.output)
            )
        ))


"""
State Machine:
Let `n` be the number of inputs and let `n,m` be the
shape of the arrays.
"""


class _StReset(V_State):
    """
    - set sig reset
    - init input addr to base
    - init weights addr to base
    - init output addr to base
    - clear output write enable

    - init dot product to zero

    - go to StWaitValid
    """

    def generate(self, m: Layer1) -> V_Block:

        return V_Block(
            "// set sig reset",
            m.sig_reset.set(V_High),

            "\n",
            "// initialize memories",
            m.inp_addr.set(m.input_mem.base_addr),
            m.w_addr.set(V_Low),
            m.out_addr.set(m.output_mem.base_addr),
            m.out_we.set(V_Low),

            "\n",
            "// initialize the dot product",
            m.dp.set(V_Low),

            _StWaitValid
        )


class _StWaitValid(V_State):
    """
    - if (valid)

        - go to StWaitDotProduct
    """

    def generate(self, m: Layer1) -> V_Block:

        return V_Block(
            *V_If(m.valid)(

                _StWaitDotProduct
            )
        )


class _StWaitDotProduct(V_State):
    """
    - add prod to dot product
    - inc input addr
    - inc weights addr
    - if (input_addr == max - 1)
        - clear sig reset
        - set sig valid

        - go to StWaitSigmoid
    """

    def generate(self, m: Layer1) -> V_Block:
        max_inp_addr = m.input_mem.size - 1

        return V_Block(
            m.dp.set(m.dp + m.prod),
            m.inp_addr.set(m.inp_addr + 1),
            m.w_addr.set(m.w_addr + 1),
            "\n",
            *V_If(m.inp_addr == max_inp_addr)(
                m.sig_reset.set(V_Low),
                m.sig_valid.set(V_High),

                _StWaitSigmoid
            )
        )


class _StWaitSigmoid(V_State):
    """
    - clear sig valid
    - if (sig done)
        - raise the output write enable

        - go to StWriteData
    """

    def generate(self, m: Layer1) -> V_Block:

        return V_Block(
            m.sig_valid.set(V_Low),
            *V_If(m.sig_done)(
                m.out_we.set(V_High),

                _StWriteData
            )
        )


class _StWriteData(V_State):
    """
    - set input addr to base
    - inc output addr
    - clear the output write enable

    - clear the dot product

    - if (output addr == max - 1)
        - go to StDone
    - else
        - set sig reset

        - go to StWaitDotProduct
    """

    def generate(self, m: Layer1) -> V_Block:
        max_out_addr = m.output_mem.size - 1

        return V_Block(
            m.inp_addr.set(m.input_mem.base_addr),
            m.out_addr.set(m.out_addr + 1),
            m.out_we.set(V_Low),

            m.dp.set(V_Low),

            *V_If(m.out_addr == max_out_addr)(
                m.done.set(V_High),

                V_StDone
            ), *V_Else(
                m.sig_reset.set(V_High),

                _StWaitDotProduct
            )
        )
