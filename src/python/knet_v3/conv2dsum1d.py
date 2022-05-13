
from math import ceil, log2
from typing import Optional, Tuple
from knet_v2.network import convolve_flat, sigmoid
from verilog.core.vmodule import V_ConnSpec
from verilog.core.vspecial import V_Empty, V_High, V_Low
from verilog.core.vstate import V_StDone, V_State, V_StateMachine
from verilog.core.vsyntax import V_Array, V_Else, V_FixedPoint, V_If, V_Par, V_Sum
from verilog.core.vtypes import BitWidth, V_Block, V_ParameterArray, V_Reg, V_RegArray, V_Wire
import numpy as np
from verilog.iterables.m10k import M10K
from verilog.ml.layer import Biases, InputShape, Layer, OutputShape, Weights
from verilog.targets.signed_mult import SignedMult
from verilog.targets.signed_sigmoid import SignedSigmoid
from verilog.utils import id_generator, nameof


class Conv2DSum1D(Layer):
    """
    img_array of size (n, n, c)
    kernel_array of size (x, y, z, c)

    img_i = img_array[:, :, cr]
           = img_array[i + n * n * cr]
    kernel_i = kernel_array[:, :, :, cr]
             = kernel_array[i + x * y * z * cr]


    where row counter cr is swept over
    cr=0, ..., c - 1.

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

    final sum
    output of shape (o1, o2, c)
    conv_out_arr of shape (c, o1, o2, c)

    for flattened output out of len (o1 * o2 * c),
    out[oc] = sum(conv_out_arr[i * o1 * o2 * c + oc])
    for i=0, ..., c - 1,
    for oc=0, ..., o1 * o2 * c - 1.
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

        # make output data a wire
        self.out_data.dtype = V_Reg

        print(self.input_shape, weights_np.shape, self.output_shape)
        n1, n2, c = self.input_shape
        assert n1 == n2
        n = n1

        x, y, z, c = weights_np.shape

        assert x == y, weights_np

        *_, o1, o2, c = self.output_shape
        assert o1 == o2

        # the constant used to scale the inputs (>> 8 -> / 256)
        self.input_scaler = 8

        # store the target size (** MIGHT BE SOURCE OF ERROR)
        self.target_size = n - 1
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

        # cr=0, ..., c - 1
        self.cr = self.var(dtype=V_Reg,
                           width=self.input_mem.addr_width,
                           name="cr")

        # oc=0, ..., o1 * o2 * c - 1 (output col)
        self.oc = self.var(dtype=V_Reg,
                           width=ceil(log2(c * o1 * o2 * c)),
                           name="oc")

        # ii=0, ..., x - 1
        self.ii = self.var(dtype=V_Reg,
                           width=self.input_mem.addr_width,
                           name="ii")

        # jj=0, ..., x - 1
        self.jj = self.var(dtype=V_Reg,
                           width=self.input_mem.addr_width,
                           name="jj")

        # r=0, ..., x * x - 1
        self.r = self.var(dtype=V_Reg,
                          width=self.input_mem.addr_width,
                          name="r")

        # create a register array for the sub image
        self.sub = self.var(
            dtype=V_RegArray,
            width=self.width,
            size=x * x,
            signed=True,
            name="sub_img"
        )

        # holds the c elements of sum(i, j), used to sum
        # over the c rows of outputs
        self.sum_arr = self.var(
            dtype=V_RegArray,
            width=self.width,
            size=c,
            signed=True,
            name="sum_arr")

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

        *_, c = x.shape

        # Z1 = np.array([convolve_flat(x[:, :, i], weights_np[:, :, :, i])
        #                for i in range(c)])
        Z1 = sum([convolve_flat(x[:, :, i], weights_np[:, :, :, i])
                 for i in range(c)])

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
            name=f"conv2dsum1d_weights"
        )

        biases = V_Array(
            module=self,
            dtype=V_ParameterArray,
            width=int_width + dec_width,
            size=len(biases_np),
            signed=True,
            data=[V_FixedPoint(v, int_width, dec_width)
                  for v in biases_np],
            name=f"conv2dsum1d_biases"
        )

        return weights, biases

    def generate(self) -> V_Block:
        x, y, z, c = self.weights_np.shape

        *_, o1, o2, c = self.output_shape

        vsm = V_StateMachine(_StReset, _StWaitValid, _StResetSumArr, _StResetSub, _StSetInpAddr,
                             _StSubBuffer, _StSetSub, _StComputeConv,
                             _StIncWeightIndices, _StIncTargetIndices, _StWriteData, _StWriteBuffer)

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
            (self.weights[(self.wc * c) + self.cr +
             (wr * z * c)], mult.input_ports[1]),
            (self.mult_outs[wr], mult.output_ports[0])
        ) for wr in range(self.w_rows)]

        return super().generate(vsm, V_Block(
            "// instantiate the multipliers",
            *[line for mult_ins in self.signed_mult_inss for line in mult_ins],

        ))


"""
Conv2DSum1D State Machine:
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
    init the bias addr
    init output write addr
    lower output write enable

    init iteration variables

    go to StWaitValid
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:

        return V_Block(
            m.inp_addr.set(m.input_mem.base_addr),
            m.b_addr.set(V_Low),
            m.out_addr.set(m.output_mem.base_addr),
            m.out_we.set(V_Low),

            m.ti.set(V_Low),
            m.tj.set(V_Low),
            m.wc.set(V_Low),

            m.cr.set(V_Low),
            m.oc.set(V_Low),

            _StWaitValid
        )


class _StWaitValid(V_State):
    """
    if (valid)
        go to StInitInputReg
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:

        return V_Block(
            *V_If(m.valid)(

                _StResetSumArr
            )
        )


class _StResetSumArr(V_State):
    """
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:
        x, y, z, c = m.weights_np.shape

        return V_Block(
            *[m.sum_arr[i].set(V_Low) for i in range(c)],

            _StResetSub
        )


class _StResetSub(V_State):
    """
    reset the counters used for sub
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:
        return V_Block(
            m.ii.set(V_Low),
            m.jj.set(V_Low),
            m.r.set(V_Low),

            _StSetInpAddr
        )


class _StSetInpAddr(V_State):
    """
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:
        n = m.target_size + 1
        x, y, z, c = m.weights_np.shape

        return V_Block(
            m.inp_addr.set(
                (V_Par(m.ti + m.ii) * n * c) +
                (V_Par(m.tj + m.jj) * c) +
                m.cr
            ),

            _StSubBuffer
        )


class _StSubBuffer(V_State):
    """
    delay 1 cycle
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:
        return V_Block(

            _StSetSub
        )


class _StSetSub(V_State):
    """
    kernel of size (x, y, z, c)

    set the sub by
    for ti in range(target_size):
        for tj in range(target_size):
            sub[0:x * x - 1] = inp_reg[ti:ti + x, ti:ti + x]

            set wc to zero

            go to StComputeConv

    inp_reg of size (n, n, c)
    inp_reg[i, j, cr] = inp_reg[(i * n * c) + (j * c) + cr]
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:
        # should be the first dim of the input shape
        n = m.target_size + 1

        x, y, z, c = m.weights_np.shape

        max_ii = x - 1
        max_jj = x - 1
        max_r = x * x - 1
        indices = [(ii * n * c, jj * c) for ii in range(x) for jj in range(x)]

        return V_Block(
            # *[m.sub[r].set(m.inp_reg[
            #     (m.ti * n * c + ii) + (m.tj * c + jj) + m.cr]
            # ) for r, (ii, jj) in enumerate(indices)],
            # *[m.sub[r].set(m.inp_reg[m.tj + (jj + ii) + m.ti * n + (m.cr * n * n)])
            #   for r, (ii, jj) in enumerate(indices)],

            m.sub[m.r].set(m.inp_data),

            *V_If(m.r == max_r)(
                m.wc.set(V_Low),

                _StComputeConv
            ), *V_Else(
                m.r.set(m.r + 1),

                # dont need to handle the case of max_ii because m.r == max_r
                *V_If(m.jj == max_jj)(
                    m.ii.set(m.ii + 1),
                    m.jj.set(V_Low)
                ), *V_Else(
                    m.jj.set(m.jj + 1)
                ),

                _StSetInpAddr
            )

            # m.wc.set(V_Low),

            # _StComputeConv
        )


class _StComputeConv(V_State):
    """
    store the multiplier output in the convolutional output array
    go to StComputeSig
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:

        return V_Block(
            "// assign the output of the convolution",
            # m.conv_out_arr[m.oc].set(
            #     V_Sum(*m.mult_outs, m.biases[m.wc])),
            # m.conv_out_arr[m.oc].set(V_Sum(*m.mult_outs)),
            m.sum_arr[m.wc].set(V_Sum(m.sum_arr[m.wc], *m.mult_outs)),
            # m.out_we.set(V_High),

            _StIncWeightIndices
        )


class _StIncWeightIndices(V_State):
    """
    out_we.set(V_Low),

    if (wc == w_rows - 1)
        set wc to 0

        if (m.oc = c * o1 * o2 * c - 1)
            set m.oc to 0

            go to StIncTargetIndices
        else
            inc m.oc

            go to StIncTargetIndices
    else
        inc wc
        inc m.oc

        go to StComputeConv
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:
        *_, o1, o2, c = m.output_shape
        max_oc = c * o1 * o2 * c - 1

        # max_rows = m.w_rows - 1
        x, y, z, c = m.weights_np.shape
        max_cols = z - 1

        return V_Block(

            # m.out_we.set(V_Low),
            *V_If(m.wc == max_cols)(
                m.wc.set(V_Low),

                *V_If(m.oc == max_oc)(
                    m.oc.set(V_Low),

                    _StIncTargetIndices
                ), *V_Else(
                    # m.out_addr.set(m.out_addr + 1),
                    m.oc.set(m.oc + 1),

                    _StIncTargetIndices
                )
            ), *V_Else(
                m.wc.set(m.wc + 1),
                m.oc.set(m.oc + 1),
                # m.out_addr.set(m.out_addr + 1),

                _StComputeConv
            )
        )


class _StIncTargetIndices(V_State):
    """
    if (m.cr == c - 1)
        set m.cr to 0

        if (m.tj == m.target_size - 1)
            set m.tj to 0

            if (m.ti == m.target_size - 1)
                set m.ti to 0
                raise the output write enable

                go to _StSum1D
            else
                inc m.ti

                go to _StResetSumArr
        else
            inc m.tj

            go to _StResetSumArr

    else
        inc m.cr

        go to _StResetSub

    """

    def generate(self, m: Conv2DSum1D) -> V_Block:
        max_iter = m.target_size - 1

        x, y, z, c = m.weights_np.shape
        max_cr = c - 1

        return V_Block(
            *V_If(m.cr == max_cr)(
                m.cr.set(V_Low),


                *V_If(m.tj == max_iter)(
                    m.tj.set(V_Low),

                    *V_If(m.ti == max_iter)(
                        m.ti.set(V_Low),

                    ), *V_Else(
                        m.ti.set(m.ti + 1),

                    )
                ), *V_Else(
                    m.tj.set(m.tj + 1),

                ),

                _StWriteData
            ), *V_Else(
                m.cr.set(m.cr + 1),

                _StResetSub
            ),
        )


class _StWriteData(V_State):

    def generate(self, m: Conv2DSum1D) -> V_Block:
        max_out_addr = m.output_mem.size
        x, y, z, c = m.weights_np.shape
        max_wc = z

        return V_Block(
            # m.out_we.set(V_Low),

            m.out_data.set(m.sum_arr[m.wc] + m.biases[m.wc]),

            *V_If(m.out_addr == max_out_addr)(

                V_StDone
            ), *V_Else(
                *V_If(m.wc == max_wc)(
                    m.wc.set(V_Low),

                    _StResetSumArr
                ), *V_Else(

                    m.out_we.set(V_High),
                    _StWriteBuffer
                )
            )

        )


class _StWriteBuffer(V_State):
    """
    """

    def generate(self, m: Conv2DSum1D) -> V_Block:

        return V_Block(
            m.out_we.set(V_Low),
            # m.out_we.set(V_High),
            m.out_addr.set(m.out_addr + 1),
            m.wc.set(m.wc + 1),

            _StWriteData
        )
