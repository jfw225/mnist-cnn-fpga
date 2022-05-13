from turtle import forward
from typing import Optional
import numpy as np
from verilog.ml.layer import Biases, InputShape, Layer, OutputShape, Weights
from verilog.core.vspecial import V_Done, V_Empty, V_High, V_Low, V_Reset, V_Valid
from verilog.core.vstate import V_StDone, V_State, V_StateMachine
from verilog.core.vsyntax import V_Array, V_Else, V_FixedPoint, V_If
from verilog.core.vtypes import BitWidth, V_Block, V_ParameterArray, V_Reg, V_RegArray, V_Wire
from verilog.iterables.m10k import M10K
from verilog.targets.signed_div import SignedDiv
from verilog.targets.signed_mult import SignedMult
from verilog.targets.signed_exponential import SignedExponential


class Layer2(Layer):
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

        # create the mult, exponential, and div modules
        self.signed_mult = SignedMult(int_width, dec_width)
        self.exponential = SignedExponential(int_width, dec_width)
        self.signed_div = SignedDiv(int_width, dec_width)

        super().__init__(
            int_width=int_width,
            dec_width=dec_width,
            weights_np=weights_np,
            input_mem=input_mem,
            output_mem=output_mem,
            input_shape=input_shape,
            output_shape=output_shape,
            objects=[self.signed_mult, self.exponential, self.signed_div]
        )

        # create variable to hold the value of the dot product
        self.dp = self.add_var(self.signed_mult.out,
                               dtype=V_Reg, name="dot_product")

        # create variable to hold the output of the multiplier
        self.prod = self.add_var(self.signed_mult.out,
                                 dtype=V_Wire, name="prod")

        # create reset, valid, and done flags for `self.exponential`
        self.exp_reset = self.add_var(
            V_Reset(module=self, name="exp_reset"), dtype=V_Reg)
        self.exp_valid = self.add_var(
            V_Valid(module=self, name="exp_valid"), dtype=V_Reg)
        self.exp_done = self.add_var(
            V_Done(module=self, name="exp_done"), dtype=V_Wire)

        # create a variable to hold the output of the exponential
        self.exp_out = self.add_var(
            self.exponential.output, dtype=V_Wire, name="exp_out")

        # create a reg array to hold the values out of the exponential
        self.exp_arr = self.var(
            V_RegArray, self.width, self.output_mem.size, signed=True, name="exp_arr")

        # create variable to hold the value of the exp sum
        self.exp_sum = self.add_var(self.signed_mult.out,
                                    dtype=V_Reg, name="exp_sum")

        # create reset, valid, and done flags for `self.signed_div`
        self.div_reset = self.add_var(
            V_Reset(module=self, name="div_reset"), dtype=V_Reg)
        self.div_valid = self.add_var(
            V_Valid(module=self, name="div_valid"), dtype=V_Reg)
        self.div_done = self.add_var(
            V_Done(module=self, name="div_done"), dtype=V_Wire)

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
            name="layer2_weights"
        ), None

    @staticmethod
    def forward(
        x: np.ndarray,
        weights_np: Weights,
        biases_np: Biases
    ):

        def softmax(x):
            exp_element = np.exp(x)
            return exp_element / np.sum(exp_element, axis=0)

        return softmax(x.dot(weights_np))

    def generate(self) -> V_Block:
        mult, exp, div = self.signed_mult, self.exponential, self.signed_div

        vsm = V_StateMachine(_StReset, _StWaitValid, _StWaitDotProduct, _StWaitExponential,
                             _StCheckSoftMaxReady, _StWaitDiv, _StWriteData, _StClearDivReset)

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
            *exp.instantiate(
                self,
                self.clk,
                (self.exp_reset, exp.reset),
                (self.exp_valid, exp.valid),
                (self.exp_done, exp.done),
                (self.dp, exp.input),
                (self.exp_out, exp.output)
            ),

            "\n// instantiate the divider",
            *div.instantiate(
                self,
                self.clk,
                (self.div_reset, div.reset),
                (self.div_valid, div.valid),
                (self.div_done, div.done),
                (self.exp_arr[self.out_addr], div.dividend),
                (self.exp_sum, div.divisor),
                (self.out_data, div.quotient)
            )
        ))


"""
State Machine:
Let `n` be the number of inputs and let `n,m` be the
shape of the arrays.
"""


class _StReset(V_State):
    """
    - set exp reset
    - set div reset 

    - init input addr to base
    - init weights addr to base
    - init output addr to base
    - clear output write enable

    - init dot product and exp sum

    - go to StWaitValid
    """

    def generate(self, m: Layer2) -> V_Block:

        return V_Block(
            "// set exp resets",
            m.exp_reset.set(V_High),
            m.div_reset.set(V_High),

            "\n",
            "// initialize memories",
            m.inp_addr.set(m.input_mem.base_addr),
            m.w_addr.set(V_Low),
            m.out_addr.set(m.output_mem.base_addr),
            m.out_we.set(V_Low),

            "\n",
            "// initialize the sum",
            m.dp.set(V_Low),
            m.exp_sum.set(V_Low),

            _StWaitValid
        )


class _StWaitValid(V_State):
    """
    - if (valid)

        - go to StWaitDotProduct
    """

    def generate(self, m: Layer2) -> V_Block:

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
        - clear exp reset
        - set exp valid

        - go to StWaitExponential
    """

    def generate(self, m: Layer2) -> V_Block:
        max_inp_addr = m.input_mem.size - 1

        return V_Block(
            m.dp.set(m.dp + m.prod),
            m.inp_addr.set(m.inp_addr + 1),
            m.w_addr.set(m.w_addr + 1),
            "\n",
            *V_If(m.inp_addr == max_inp_addr)(
                m.exp_reset.set(V_Low),
                m.exp_valid.set(V_High),

                _StWaitExponential
            )
        )


class _StWaitExponential(V_State):
    """
    - clear exp valid
    - if (exp done)
        - add exp out to exp sum
        - write exp out to exp_arr

        - go to StCheckSoftMaxReady
    """

    def generate(self, m: Layer2) -> V_Block:

        return V_Block(
            m.exp_valid.set(V_Low),
            *V_If(m.exp_done)(
                m.exp_arr[m.out_addr].set(m.exp_out),
                m.exp_sum.set(m.exp_sum + m.exp_out),

                _StCheckSoftMaxReady
            )
        )


class _StCheckSoftMaxReady(V_State):
    """
    - set input addr to base

    - clear the dot product

    - if (output addr == max - 1)
        - set output addr to base 

        - clear div reset
        - set div valid
        - go to StWaitDiv
    - else
        - set exp reset
        - inc output addr

        - go to StWaitDotProduct
    """

    def generate(self, m: Layer2) -> V_Block:
        max_out_addr = m.output_mem.size - 1

        return V_Block(
            m.inp_addr.set(m.input_mem.base_addr),
            m.dp.set(V_Low),

            *V_If(m.out_addr == max_out_addr)(
                m.out_addr.set(m.output_mem.base_addr),

                m.div_reset.set(V_Low),
                m.div_valid.set(V_High),

                _StWaitDiv
            ), *V_Else(
                m.exp_reset.set(V_High),
                m.out_addr.set(m.out_addr + 1),

                _StWaitDotProduct
            )
        )


class _StWaitDiv(V_State):
    """
    - clear div valid
    - if (div done)
        - raise the output write enable

        - go to StWriteData
    """

    def generate(self, m: Layer2) -> V_Block:

        return V_Block(
            m.div_valid.set(V_Low),
            *V_If(m.div_done)(
                m.out_we.set(V_High),

                _StWriteData
            )
        )


class _StWriteData(V_State):
    """
    - clear the output write enable

    - if (output addr == max - 1)
        - go to StDone
    - else
        - set div reset
        - inc output addr

        - go to StClearDivReset
    """

    def generate(self, m: Layer2) -> V_Block:
        max_out_addr = m.output_mem.size - 1

        return V_Block(
            m.out_we.set(V_Low),

            *V_If(m.out_addr == max_out_addr)(
                m.done.set(V_High),

                V_StDone
            ), *V_Else(
                m.div_reset.set(V_High),
                m.out_addr.set(m.out_addr + 1),

                _StClearDivReset
            )
        )


class _StClearDivReset(V_State):
    """
    - clear div reset
    - set div valid

    - go to _StWaitDiv
    """

    def generate(self, m: Layer2) -> V_Block:

        return V_Block(
            m.div_reset.set(V_Low),
            m.div_valid.set(V_High),

            _StWaitDiv
        )
