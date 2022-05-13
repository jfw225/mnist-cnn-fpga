from math import ceil, log2
from typing import Optional, Tuple
import numpy as np
from knet_v2.network import maxpool, maxpool_flat, softmax
from verilog.core.vspecial import V_Done, V_Empty, V_High, V_Low, V_Reset, V_Valid
from verilog.core.vstate import V_StDone, V_State, V_StateMachine
from verilog.core.vsyntax import V_Array, V_Else, V_FixedPoint, V_If, V_Par

from verilog.core.vtypes import BitWidth, V_Block, V_Expression, V_ParameterArray, V_Reg, V_RegArray, V_Wire, V_WireArray
from verilog.iterables.m10k import M10K
from verilog.ml.layer import Biases, InputShape, Layer, OutputShape, Weights
from verilog.targets.signed_div import SignedDiv
from verilog.targets.signed_exponential import SignedExponential
from verilog.targets.signed_max import SignedMax
from verilog.targets.signed_mult import SignedMult
from verilog.utils import id_generator


class Maxpool(Layer):
    """
    Let the input to this layer be

    mp_input of size (y, x, n)
    window is of size (f * f * n)

    window = [[
        A1 = [o1, ..., on],
        ...,
        Af = [p1, ..., pn]
    ], [
        B1 = [q1, ..., qn],
        ...,
        Bf = [r1, ..., rn]
    ]]

    the output at iteration (i, j) is the columnwise max of
    [A1, ..., Af, B1, ..., Bf], or rather,
    max(i, j) of size (n) = [
        max(o1, ..., p1, q1, ..., r1),
        ...,
        max(on, ..., pn, qn, ..., rn)
    ] = [
        max(
            window[0 + 0 * z * z]
            + ... + window[0 + wr * z * z] + ... +
            window[0 + (z * z - 1) * z * z]
        ), ...,
        max(
            window[wc + 0 * z * z]
            + ... + window[wc + wr * z * z] + ... +
            window[wc + (z * z - 1) * z * z]
        ), ...,
        max(
            window[(n - 1) + 0 * z * z]
            + ... + window[(n - 1) + wr * z * z] + ... +
            window[(n - 1) + (z * z - 1) * z * z]
        )
        max(window[wc + wr * z * z])
    ]

    where wc=0, ..., n - 1 is the column sweep and
    wr=0, ..., z * z - 1 is the row sweep.
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

        # create the max. mult, exponential, and div modules
        self.signed_max = SignedMax(int_width, dec_width)

        super().__init__(
            int_width=int_width,
            dec_width=dec_width,
            weights_np=weights_np,
            biases_np=biases_np,
            input_mem=input_mem,
            output_mem=output_mem,
            input_shape=input_shape,
            output_shape=output_shape,
            objects=[self.signed_max]
        )

        # configure the output data as a wire
        self.out_data.dtype = V_Reg

        # store f and stride values
        self._f = 2
        self._s = 2

        # we assume the f is even for now
        assert self._f % 2 == 0, self._f

        # create a register array for the sliding window
        self.window = self.var(dtype=V_RegArray,
                               width=self.width,
                               size=self.f * self.f * self.n,
                               signed=True,
                               name="window")

        # create iteration variables as in `maxpool_flat`
        self.curr_y = self.var(dtype=V_Reg,
                               width=self.input_mem.addr_width,
                               name="curr_y")
        self.out_y = self.var(dtype=V_Reg,
                              width=self.output_mem.addr_width,
                              name="out_y")

        self.curr_x = self.var(dtype=V_Reg,
                               width=self.input_mem.addr_width,
                               name="curr_x")
        self.out_x = self.var(dtype=V_Reg,
                              width=self.output_mem.addr_width,
                              name="out_x")

        # jj=0, ..., f - 1
        self.jj = self.var(dtype=V_Reg,
                           width=self.input_mem.addr_width,
                           name="jj")

        # ii=0, ..., f - 1
        self.ii = self.var(dtype=V_Reg,
                           width=self.input_mem.addr_width,
                           name="ii")

        # kk=0, ..., n - 1
        self.kk = self.var(dtype=V_Reg,
                           width=self.input_mem.addr_width,
                           name="kk")

        # ww=0, ..., f * f * n - 1
        self.ww = self.var(dtype=V_Reg,
                           width=ceil(log2(self.window.size)),
                           name="ww")

        # wc=0, ..., n - 1
        self.wc = self.var(dtype=V_Reg,
                           width=self.output_mem.addr_width,
                           name="wc")

        # wr=0, ..., f * f - 1
        self.wr = self.var(dtype=V_Reg,
                           width=ceil(log2(self.window.size)),
                           name="wr")

        # holds the output of the signed max module
        self.max_out = self.var(dtype=V_Wire,
                                width=self.width,
                                signed=True,
                                name="max_out")

    @property
    def f(self):
        """
        The `f` parameter in `maxpool_flat`.
        TODO: KEN COME BACK AND EXPLAIN THIS
        """

        return self._f

    @property
    def s(self):
        """
        The stride parameter in `maxpool_flat`.
        """

        return self._s

    @property
    def y(self):
        """
        Input array is of size `(y, x, n)`.
        """

        y, x, n = self.input_shape

        return y

    @property
    def x(self):
        """
        Input array is of size `(y, x, n)`.
        """

        y, x, n = self.input_shape

        return x

    @property
    def n(self):
        """
        Input array is of size `(y, x, n)`.
        """

        y, x, n = self.input_shape

        return n

    @staticmethod
    def forward(
        x: np.ndarray,
        weights_np: Weights,
        biases_np: Biases
    ):

        return maxpool_flat(x)

    def convert_model_params(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        weights_np: np.ndarray,
        biases_np: np.ndarray
    ) -> Tuple[V_Array, V_Array]:

        return None, None

    def generate(self) -> V_Block:

        vsm = V_StateMachine(_StReset, _StWaitValid,
                             _StResetWindow, _StSetInpAddr,
                             _StWindowBuffer, _StSetWindow,
                             _StMaxReset, _StGetMaxOverWindow,
                             _StFindMax, _StMaxBuffer,
                             _StIncMaxpool, _StWriteData)

        sm = self.signed_max

        return super().generate(vsm, V_Block(
            "// instantiate the signed maximum module",
            *self.signed_max.instantiate(
                self,
                (V_Empty(), sm.clk),
                (V_Empty(), sm.reset),
                (V_Empty(), sm.valid),
                (V_Empty(), sm.done),
                (V_Empty(), sm.ready),
                (self.window[self.wc], sm.input_ports[0]),
                (self.window[self.wc + self.wr * self.n], sm.input_ports[1]),
                (self.max_out, sm.output_ports[0])
            )
        ))


class _StReset(V_State):
    """
    init input read addr
    init output write addr
    lower output write enable

    init iteration variables

    go to _StWaitValid
    """

    def generate(self, m: Maxpool) -> V_Block:

        return V_Block(
            m.inp_addr.set(m.input_mem.base_addr),
            m.out_addr.set(m.output_mem.base_addr),
            m.out_we.set(V_Low),

            m.curr_y.set(V_Low),
            m.out_y.set(V_Low),
            m.curr_x.set(V_Low),
            m.out_x.set(V_Low),


            _StWaitValid
        )


class _StWaitValid(V_State):
    """
    if (valid)
        go to StInitInputReg
    """

    def generate(self, m: Maxpool) -> V_Block:

        return V_Block(
            *V_If(m.valid)(

                _StResetWindow
            )
        )


class _StResetWindow(V_State):
    """
    """

    def generate(self, m: Maxpool) -> V_Block:
        return V_Block(
            m.jj.set(V_Low),
            m.ii.set(V_Low),
            m.kk.set(V_Low),
            m.ww.set(V_Low),

            _StSetInpAddr
        )


class _StSetInpAddr(V_State):
    """
    """

    def generate(self, m: Maxpool) -> V_Block:

        return V_Block(
            m.inp_addr.set(
                (V_Par(m.curr_y + m.jj) * m.x * m.n) +
                (V_Par(m.curr_x + m.ii) * m.n) +
                m.kk),

            _StWindowBuffer
        )


class _StWindowBuffer(V_State):
    """
    """

    def generate(self, m: Maxpool) -> V_Block:

        return V_Block(

            _StSetWindow
        )


class _StSetWindow(V_State):
    """
    set window = inp_reg[curr_y:curr_y + f, curr_x:curr_x + f]

    inp_reg of size(y, x, n)
    inp_reg[j, i, k] = inp_reg[(j * x * n) + (i * n) + k]

    need to sweep
    j=curr_y, ..., curr_y + f
    i=curr_x, ..., curr_x + f
    k=0, ..., n

    go to _StFindMax
    """

    def generate(self, m: Maxpool) -> V_Block:
        max_jj = m.f - 1
        max_ii = m.f - 1
        max_kk = m.n - 1
        max_ww = m.f * m.f * m.n - 1

        return V_Block(
            m.window[m.ww].set(m.inp_data),

            *V_If(m.ww == max_ww)(

                _StMaxReset
            ), *V_Else(
                m.ww.set(m.ww + 1),

                *V_If(m.kk == max_kk)(
                    m.kk.set(V_Low),

                    # dont need to set jj low because that is case
                    # where m.ww == max_ww
                    *V_If(m.ii == max_ii)(
                        m.ii.set(V_Low),
                        m.jj.set(m.jj + 1)
                    ), *V_Else(
                        m.ii.set(m.ii + 1)
                    )
                ), *V_Else(
                    m.kk.set(m.kk + 1)
                ),

                _StSetInpAddr
            )
        )


class _StMaxReset(V_State):
    """
    """

    def generate(self, m: Maxpool) -> V_Block:

        return V_Block(
            m.wc.set(V_Low),
            m.wr.set(V_High),

            _StGetMaxOverWindow
        )


class _StGetMaxOverWindow(V_State):
    """
    """

    def generate(self, m: Maxpool) -> V_Block:
        max_wr = m.f * m.f - 1

        return V_Block(
            m.window[m.wc].set(m.max_out),

            *V_If(m.wr == max_wr)(

                _StFindMax
            ), *V_Else(
                m.wr.set(m.wr + 1),

                _StGetMaxOverWindow
            )
        )


class _StFindMax(V_State):
    """
    assign each of the column-wise maximums to the mp_out register

    mp_out of size (y // 2, x // 2, n)

    mp_out[(out_y * x // 2 * n) + (out_x * n) + wc] = max_outputs[wc]
    for window column index wc=0,..., n - 1.
    """

    def generate(self, m: Maxpool) -> V_Block:

        return V_Block(
            m.out_we.set(V_Low),
            m.out_addr.set(
                (m.out_y * (m.x // 2) * m.n) +
                (m.out_x * m.n) +
                m.wc
            ),
            # m.out_data.set(m.max_out_arr[m.wc]),
            m.out_data.set(m.max_out),

            _StMaxBuffer
        )


class _StMaxBuffer(V_State):
    """
    """

    def generate(self, m: Maxpool) -> V_Block:
        max_wc = m.n - 1

        return V_Block(
            m.out_we.set(V_High),

            *V_If(m.wc == max_wc)(
                _StIncMaxpool
            ), *V_Else(
                m.wc.set(m.wc + 1),
                m.wr.set(1),

                _StGetMaxOverWindow
            )

        )


class _StIncMaxpool(V_State):
    """
    set each of the previously set mp_outs to to themselves

    if (curr_x <= x - s - f)
        set curr_x to 0
        set out_x to 0

        if (curr_y <= y - s - f)
            set the output write enable

            go to _StDotProduct
        else
            inc curr_y by s
            inc out_y by 1

            go to _StResetWindow

    else
        inc curr_x by s
        inc out_x

        go to _StResetWindow
    """

    def generate(self, m: Maxpool) -> V_Block:
        max_y = m.y - m.s - m.f
        max_x = m.x - m.s - m.f

        print(max_y, max_x)

        return V_Block(
            m.out_we.set(V_Low),

            *V_If(m.curr_x > max_x)(
                m.curr_x.set(V_Low),
                m.out_x.set(V_Low),

                *V_If(m.curr_y > max_y)(
                    # m.out_we.set(V_High),
                    V_StDone

                    # _StWriteData,
                ), *V_Else(
                    m.curr_y.set(m.curr_y + m.s),
                    m.out_y.set(m.out_y + 1),

                    _StResetWindow
                )
            ), *V_Else(
                m.curr_x.set(m.curr_x + m.s),
                m.out_x.set(m.out_x + 1),

                _StResetWindow
            )
        )


class _StWriteData(V_State):
    """
    if (output addr == max)
        clear output write enable

        go to StDone
    else
        inc output addr
    """

    def generate(self, m: Maxpool) -> V_Block:
        max_out_addr = m.output_mem.size - 1

        return V_Block(

            *V_If(m.out_addr == max_out_addr)(
                m.out_we.set(V_Low),

                V_StDone
            ), *V_Else(
                m.out_addr.set(m.out_addr + 1)
            )
        )
