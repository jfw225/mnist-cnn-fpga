from typing import Optional, Tuple
from knet_v2.network import convolve_flat, sigmoid
from verilog.core.vmodule import V_ConnSpec
from verilog.core.vspecial import V_Empty, V_High, V_Low
from verilog.core.vstate import V_StDone, V_State, V_StateMachine
from verilog.core.vsyntax import V_Array, V_Else, V_FixedPoint, V_If, V_Sum
from verilog.core.vtypes import BitWidth, V_Block, V_ParameterArray, V_Reg, V_RegArray, V_Wire
import numpy as np
from verilog.iterables.m10k import M10K
from verilog.ml.layer import Biases, InputShape, Layer, OutputShape, Weights
from verilog.targets.signed_mult import SignedMult
from verilog.targets.signed_sigmoid import SignedSigmoid
from verilog.utils import id_generator, nameof


"""
TODO:

create a register array for the values of sub
have a register array that keeps the values of the filter
"""


class Conv2D(Layer):
    """
    img of size (n, n)
    kernel of size (x, y, z)

    We have z=8 filters, each of which are x=2 by y=2
    Let FIL_1 denote the first filter(i, j) using the notation above, then:
    kernel = [ [f1      [h1
                f2       h2
                g1 ,     j1 ...
                g2]      j2]       ]

           = [FIL_1.T, FIL_2.T, ... , FIL_8.T]  --> this is hard to think about
                                            since these filters are transposed

    at each iteration i=1,..., n - 1 and j=1,..., n - 1:
    sub(i, j) of size (x, y) is subset of the image
    sub(i, j) = [[f1, ..., fx],
                 [g1, ..., gy]]

    kernel = [[ F1 = [o1, ..., oz],
                ...
                Fx = [p1, ..., pz]],
              [ G1 = [q1, ..., qz],
                ...
                Gy = [r1, ..., rz]]

    prod(i, j) = sub(i, j) * kernel
               = [[ f1 * F1,
                    ...
                    fx * Fx],
                  [ g1 * G1,
                    gy * Gy]]
    sum(i, j) = prod.sum(axis=(0, 1)) =
        [
            f1 * o1 + ... + fx * p1 + g1 * q1 + ... + gy * r1,
            ...
            f1 * oz + ... + fx * pz + g1 * qz + ... + gy * rz
        ]

    output is array of size (n - 1, n - 1, z) where
    output[i, j] = sum(i, j)
    ------------
    This very abstract representation makes hard to develop the intuition,
    let's ignore this for now
    ------------
    mat is a 2x2 slice of the image --> img[i:i+2, j:j+2]
    Let's say:
    mat = np.array([[[0.1],
                     [0.5]],
                     [[1.],
                     [1.]]])
    kernel = np.ones( (2,2,8) )

    prod(i, j) = mat * kernel
               = np.array([[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
                            [[1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ],
                            [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ]]])
    output(i, j) = np.array([2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6])
                                             --> we sum the columns
    """

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

        # create the signed mult and sigmoid modules
        self.signed_mult = SignedMult(int_width, dec_width)

        super().__init__(
            int_width=int_width,
            dec_width=dec_width,
            weights_np=weights_np,
            biases_np=biases_np,
            input_mem=input_mem,
            output_mem=output_mem,
            input_shape=input_shape,
            output_shape=output_shape,
            objects=[self.signed_mult]
        )

        # make sure input data is signed
        self.inp_data.signed = True

        # make output data a reg
        self.out_data.dtype = V_Wire

        # config w/b addrs
        self.w_addr.dtype = V_Wire
        self.b_addr.dtype = V_Wire

        x, y, z = weights_np.shape

        assert x == y, weights_np

        # the constant used to scale the inputs (>> 8 -> / 256)
        self.input_scaler = 8

        # store the target size (** MIGHT BE SOURCE OF ERROR)
        self.target_size = int(np.sqrt(self.input_mem.size)) - 1
        print(f"{nameof(self)} Target Size: {self.target_size}")

        # the number of rows in the flattened weights
        self.w_rows = x * x

        # create registers used as iteration indices
        # ti=0, ..., target_size - 1
        self.ti = self.var(dtype=V_Reg,
                           width=self.input_mem.addr_width,
                           name="ti")

        # tj=0, ..., target_size - 1
        self.tj = self.var(dtype=V_Reg,
                           width=self.input_mem.addr_width,
                           name="tj")

        # wc=0, ..., z - 1
        self.wc = self.var(dtype=V_Reg,
                           width=self.w_addr.width,
                           name="wc")

        # wr=0, ..., x * x - 1
        self.wr = self.var(dtype=V_Reg,
                           width=self.w_addr.width,
                           name="wr")

        # create a register array to hold a copy of the inputs
        self.inp_reg = self.var(
            dtype=V_RegArray,
            width=self.input_mem.width,
            size=self.input_mem.size,
            signed=True,
            name="input_reg"
        )

        # create a register array for the sub image
        self.sub = self.var(
            dtype=V_RegArray,
            width=self.width,
            size=x * x,
            signed=True,
            name="sub_img"
        )

        # create a segment to hold weights
        self.w_seg = self.var(
            dtype=V_RegArray,
            width=self.width,
            size=self.w_rows,
            signed=True,
            name="w_seg"
        )

        # create `self.w_rows` multiplier outputs
        self.mult_outs = [self.var(dtype=V_Wire,
                                   width=self.signed_mult.width,
                                   signed=True,
                                   name=f"mult_out_{wr}")
                          for wr in range(self.w_rows)]

    @staticmethod
    def forward(
        x: np.ndarray,
        weights_np: Weights,
        biases_np: Biases
    ):

        Z1 = convolve_flat(x, weights_np)

        return Z1 + biases_np

    def convert_model_params(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        weights_np: np.ndarray,
        biases_np: np.ndarray
    ) -> Tuple[V_Array, V_Array]:

        assert biases_np.ndim == 1, biases_np

        w_flat = weights_np.flatten()

        weights = V_Array(
            module=self,
            dtype=V_ParameterArray,
            width=int_width + dec_width,
            size=len(w_flat),
            signed=True,
            data=[V_FixedPoint(v, int_width, dec_width) for v in w_flat],
            name=f"conv2d_weights"
        )

        biases = V_Array(
            module=self,
            dtype=V_ParameterArray,
            width=int_width + dec_width,
            size=len(biases_np),
            signed=True,
            data=[V_FixedPoint(v, int_width, dec_width)
                  for v in biases_np],
            name=f"conv2d_biases"
        )

        return weights, biases

    def generate(self) -> V_Block:
        x, y, z = self.weights_np.shape

        # vsm = V_StateMachine(_StReset, _StWaitValid, _StInitInputReg,
        #                      _StSetSub, _StComputeConv,
        #                      _StIncWeightIndices, _StIncTargetIndices)
        states = [_StReset, _StWaitValid, _StInitInputReg,
                  _StSetSub, _StPreConv, _StComputeConv,
                  _StPostConv, _StIncWeightIndices, _StIncTargetIndices]
        # instantiate x * x signed mult modules
        mult = self.signed_mult
        self.signed_mult_inss = [self.signed_mult.instantiate(
            self,
            (V_Empty(), mult.clk),
            (V_Empty(), mult.reset),
            (V_Empty(), mult.valid),
            (V_Empty(), mult.done),
            (V_Empty(), mult.ready),
            (self.sub[wr], mult.input_ports[0]),
            (self.w_seg[wr], mult.input_ports[1]),
            (self.mult_outs[wr], mult.output_ports[0])
        ) for wr in range(self.w_rows)]

        return super().generate(states, V_Block(
            "// instantiate the multipliers",
            *[line for mult_ins in self.signed_mult_inss for line in mult_ins],

            "\n// assign the output of the convolution",
            self.out_data.set(
                V_Sum(*self.mult_outs, self.b_data)),

            self.w_addr.set(self.wc + self.wr * z),
            self.b_addr.set(self.wc)
        ))


"""
Conv2D State Machine:
img of size (n, n)
kernel of size (x, y, z)


first load each input into a local register array
reasoning:
    at each iteration, we need to get 4 new inputs from input memory. this
    would take at least 5 cycles to set input addr and get new data

    getting all the inputs into a reg array would allow us to get new values
    in same cycle

call this reg array `inp_reg`
for iteration i=1,..., n - 1 and j=1,..., n - 1:
filter(i, j) is of size (x, y)


set the sub by 
for ti in range(target_size):
    for tj in range(target_size):
        sub[0:x * x - 1] = inp_reg[ti:ti + x, ti:ti + x]


go straight to computing this the prod.sum:
sum(i, j) = prod.sum(axis=(0, 1)) =
        [
            f1 * o1 + ... + fx * p1 + g1 * q1 + ... + gy * r1,
            ...
            f1 * oz + ... + fx * pz + g1 * qz + ... + gy * rz
        ]
        =
        [
            sub[0] * W[0 + 0 * z] + ... + sub[wr] * W[0 + wr * z],
            sub[0] * W[1 + 0 * z] + ... + sub[wr] * W[1 + wr * z],
            ...
            sub[0] * W[wc + 0 * z] + ... + sub[wr] * W[wc + wr * z]
        ]
where wr=0, ..., x * x - 1 and is the row index and 
      wc=0, ...., z - 1
"""


class _StReset(V_State):
    """
    init input read addr
    init output write addr
    init output data

    lower output write enable


    init iteration variables

    raise sig reset
    lower sig valid

    go to StWaitValid
    """

    def generate(self, m: Conv2D) -> V_Block:

        return V_Block(
            m.inp_addr.set(m.input_mem.base_addr),
            m.out_addr.set(m.output_mem.base_addr),
            m.out_we.set(V_Low),

            m.ti.set(V_Low),
            m.tj.set(V_Low),
            m.wc.set(V_Low),
            m.wr.set(V_Low),

            _StInitInputReg
        )


class _StWaitValid(V_State):
    """
    if (valid)
        go to StInitInputReg
    """

    def generate(self, m: Conv2D) -> V_Block:

        return V_Block(
            *V_If(m.valid)(

                _StInitInputReg
            )
        )


class _StInitInputReg(V_State):
    """
    copy and scale each input in input memory into a local register 

    if (all values have been copied)
        go to StSetSub
    """

    def generate(self, m: Conv2D) -> V_Block:
        max_inp_addr = m.input_mem.size

        return V_Block(
            m.inp_reg[m.inp_addr - 1].set(m.inp_data),
            # m.inp_reg[m.inp_addr - 1].set(m.inp_data >> m.input_scaler),
            m.inp_addr.set(m.inp_addr + 1),

            *V_If(m.inp_addr == max_inp_addr)(
                _StSetSub
            )
        )


class _StSetSub(V_State):
    """
    set the sub by 
    for ti in range(target_size):
        for tj in range(target_size):
            sub[0:x * x - 1] = inp_reg[ti:ti + x, ti:ti + x]

            set wc to zero

            go to StComputeConv
    """

    def generate(self, m: Conv2D) -> V_Block:
        # should be the first dim of the input shape
        n = m.target_size + 1
        k, *_ = m.weights_np.shape

        indices = [(ii * n, jj) for ii in range(k) for jj in range(k)]

        return V_Block(
            *[m.sub[r].set(m.inp_reg[m.tj + jj + ii + m.ti * n])
              for r, (ii, jj) in enumerate(indices)],

            m.wc.set(V_Low),

            _StPreConv
        )


class _StPreConv(V_State):
    """
    set wr to 0
    set out_data to biases[m.wc]

    go to _StComputeConv
    """

    def generate(self, m: Conv2D) -> V_Block:

        return V_Block(
            m.wr.set(V_Low),
            # m.out_data.set(m.b_data),

            _StComputeConv
        )


class _StComputeConv(V_State):
    """
    set out data += mult out
    if (wr == x * x - 1)
        raise output write enable
        clear wr

        go to _StIncWeightIndices
    else
        inc wr
    """

    def generate(self, m: Conv2D) -> V_Block:
        max_wr = m.w_rows

        return V_Block(
            m.w_seg[m.wr].set(m.w_data),
            # m.out_data.set(m.out_data + m.mult_out),
            *V_If(m.wr == max_wr)(
                # m.out_we.set(V_High),
                # m.wr.set(V_Low),

                _StPostConv
            ), *V_Else(
                m.wr.set(m.wr + 1)
            )
        )


class _StPostConv(V_State):
    def generate(self, m: Conv2D) -> V_Block:
        return V_Block(
            m.out_we.set(V_High),
            m.wr.set(V_Low),

            _StIncWeightIndices
        )


class _StIncWeightIndices(V_State):
    """
    clear the output write enable
    clear out data

    if (wc == w_rows - 1)
        set wc to 0

        if (output addr = max output addr - 1)
            go to StDone
        else 
            inc the output addr 

            go to StIncTargetIndices
    else
        inc wc
        inc the output addr

        go to StComputeConv
    """

    def generate(self, m: Conv2D) -> V_Block:
        max_out_addr = m.output_mem.size - 1
        # max_rows = m.w_rows - 1
        x, y, z = m.weights_np.shape
        max_cols = z - 1

        return V_Block(
            m.out_we.set(V_Low),

            *V_If(m.wc == max_cols)(
                m.wc.set(V_Low),

                *V_If(m.out_addr == max_out_addr)(
                    V_StDone
                ), *V_Else(
                    m.out_addr.set(m.out_addr + 1),

                    _StIncTargetIndices
                )
            ), *V_Else(
                m.wc.set(m.wc + 1),
                m.out_addr.set(m.out_addr + 1),

                _StPreConv
            )
        )


class _StIncTargetIndices(V_State):
    """

    if (m.tj == m.target_size - 1)
        set m.tj to 0

        if (m.ti == m.target_size - 1)
            go to StDone
        else
            inc m.ti
            go to StSetSub
    else
        inc m.tj

        go to StSetSub
    """

    def generate(self, m: Conv2D) -> V_Block:
        max_iter = m.target_size - 1

        return V_Block(
            *V_If(m.tj == max_iter)(
                m.tj.set(V_Low),

                *V_If(m.ti == max_iter)(
                    V_StDone
                ), *V_Else(
                    m.ti.set(m.ti + 1),
                    _StSetSub
                )
            ), *V_Else(
                m.tj.set(m.tj + 1),

                _StSetSub
            )
        )
