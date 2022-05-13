from math import ceil, log2
from typing import Optional, Tuple
import numpy as np
from knet_v2.network import maxpool, maxpool_flat, softmax
from verilog.core.vspecial import V_Done, V_Empty, V_High, V_Low, V_Reset, V_Valid
from verilog.core.vstate import V_StDone, V_State, V_StateMachine
from verilog.core.vsyntax import V_Array, V_Else, V_FixedPoint, V_If

from verilog.core.vtypes import BitWidth, V_Block, V_ParameterArray, V_Reg, V_RegArray, V_Wire
from verilog.iterables.m10k import M10K
from verilog.ml.layer import Biases, InputShape, Layer, OutputShape, Weights
from verilog.targets.signed_div import SignedDiv
from verilog.targets.signed_exponential import SignedExponential
from verilog.targets.signed_max import SignedMax
from verilog.targets.signed_mult import SignedMult
from verilog.utils import id_generator


class Maxpool_NoSoft(Layer):
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
        # the number of terms used in the exponent taylor series approximation
        self.exp_num_terms = 10

        # create the max. mult, exponential, and div modules
        self.signed_max = SignedMax(int_width, dec_width)
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
            objects=[self.signed_max, self.signed_mult]
        )

        # store f and stride values
        self._f = 2
        self._s = 2

        # we assume the f is even for now
        assert self._f % 2 == 0, self._f

        # create a register array to hold a copy of the inputs
        self.inp_reg = self.var(
            dtype=V_RegArray,
            width=self.width,
            size=self.input_mem.size,
            signed=True,
            name="input_reg"
        )

        # create a register array for the sliding window
        self.window = self.var(dtype=V_RegArray,
                               width=self.width,
                               size=self.f * self.f * self.n,
                               signed=True,
                               name="window")

        # create a register for the output of the maxpool
        self.mp_out = self.var(dtype=V_RegArray,
                               width=self.width,
                               size=(self.y // 2) * (self.x // 2) * self.n,
                               signed=True,
                               name="mp_out")

        # create iteration variables as in `maxpool_flat`
        self.curr_y = self.var(dtype=V_Reg,
                               width=self.input_mem.addr_width,
                               name="curr_y")
        self.out_y = self.var(dtype=V_Reg,
                              width=ceil(log2(self.mp_out.size)),
                              name="out_y")

        self.curr_x = self.var(dtype=V_Reg,
                               width=self.input_mem.addr_width,
                               name="curr_x")
        self.out_x = self.var(dtype=V_Reg,
                              width=ceil(log2(self.mp_out.size)),
                              name="out_x")

        # define a list to hold the output of the top level max moudle instances
        # self.max_outputs[wc] holds the max of window column wc
        self.max_outputs = list()

        # define a list to hold the max module instances
        self.max_instances = list()

        # configure the logic for the max functionality
        self._configure_max_logic()

        # create variable to hold the value of the dot product
        self.dp = self.add_var(self.signed_mult.out,
                               dtype=V_Reg, name="dot_product")

        # create variable to hold the output of the multiplier
        self.prod = self.add_var(self.signed_mult.out,
                                 dtype=V_Wire, name="prod")

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

        Z2 = maxpool_flat(x)
        Z3 = Z2.reshape(-1).dot(weights_np)

        return Z3 + biases_np
        # return softmax(Z3 + biases_np)

    def convert_model_params(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        weights_np: np.ndarray,
        biases_np: np.ndarray
    ) -> Tuple[V_Array, V_Array]:

        assert biases_np.ndim == 1, biases_np

        w_flat = weights_np.T.flatten()

        weights = V_Array(
            module=self,
            dtype=V_ParameterArray,
            width=int_width + dec_width,
            size=len(w_flat),
            signed=True,
            data=[V_FixedPoint(v, int_width, dec_width) for v in w_flat],
            name=f"maxpool_weights"
        )

        biases = V_Array(
            module=self,
            dtype=V_ParameterArray,
            width=int_width + dec_width,
            size=len(biases_np),
            signed=True,
            data=[V_FixedPoint(v, int_width, dec_width)
                  for v in biases_np],
            name=f"maxpool_biases"
        )

        return weights, biases

    def generate(self) -> V_Block:
        mult = self.signed_mult

        vsm = V_StateMachine(_StReset, _StWaitValid,
                             _StInitInputReg, _StSetWindow, _StFindMax,
                             _StIncMaxpool, _StWaitDotProduct, _StWriteData)

        return super().generate(vsm, V_Block(
            "// instantiate the signed maximum modules",
            *[line for sm_ins in self.max_instances for line in sm_ins],

            "\n// instantiate the multiplier",
            *mult.instantiate(
                self,
                (V_Empty(), mult.clk),
                (V_Empty(), mult.reset),
                (V_Empty(), mult.valid),
                (V_Empty(), mult.done),
                (self.mp_out[self.out_x], mult.input_ports[0]),
                (self.w_data, mult.input_ports[1]),
                (self.prod, mult.output_ports[0])
            ),


        ))

    def _configure_max_logic(self) -> None:
        """
        Configures the layer to be capable of handling the maximum 
        functionality for the window as described in _StFindMax. 
        """

        # localize `self.signed_max` for less typing
        sm = self.signed_max

        # create pairings for the maximum logic
        inds = np.arange(self.f * self.f *
                         self.n).reshape(self.f * self.f, self.n)

        def create_pairs(indices):
            # indices should always contain at least one
            n = len(indices)
            assert n > 0, indices
            if n == 1:
                return (*indices, None)
            elif n == 2:
                return (*indices,)

            return (create_pairs(indices[:2]), create_pairs(indices[2:]))

        def create_max_instances(pairs):
            """
            Recursive function that creates the required instances of 
            `SignedMax`. Each iteration returns a reference to a register in 
            `self.window` or creates a `SignedMax` instance and returns a wire 
            connected to its output. 
            """

            A, B = pairs

            if isinstance(A, int):
                # if A is an int, B can either be an int or None
                assert isinstance(B, int) or B is None, B

                # if B is None, we simply return the window register value
                if B is None:
                    return self.window[A]

                # otherwise, we set A and B to their register references
                A, B = self.window[A], self.window[B]

            # otherwise, A and B must be pairs
            else:
                assert not isinstance(B, int) and B is not None, B

                # get the corresponding output wires of pairs A and B
                A, B = create_max_instances(A), create_max_instances(B)

            # create an output wire and an instance that computes max(A, B)
            out = self.var(dtype=V_Wire,
                           width=self.width,
                           signed=True,
                           name=f"max_logic_{id_generator()}")

            self.max_instances.append(sm.instantiate(
                self,
                (V_Empty(), sm.clk),
                (V_Empty(), sm.reset),
                (V_Empty(), sm.valid),
                (V_Empty(), sm.done),
                (A, sm.input_ports[0]),
                (B, sm.input_ports[1]),
                (out, sm.output_ports[0])
            ))

            # return the output wire
            return out

        # transpose because we are taking the column-wise maximum in the window
        for col in inds.T:
            pairs = create_pairs([*map(int, col)])
            out = create_max_instances(pairs)
            self.max_outputs.append(out)


class _StReset(V_State):
    """
    init input read addr
    init weights addr to base
    init biases addr to base
    init output write addr
    lower output write enable

    init iteration variables

    init dot product and exp sum
    """

    def generate(self, m: Maxpool_NoSoft) -> V_Block:

        return V_Block(
            m.inp_addr.set(m.input_mem.base_addr),
            m.w_addr.set(V_Low),
            m.b_addr.set(V_Low),
            m.out_addr.set(m.output_mem.base_addr),
            m.out_we.set(V_Low),

            m.curr_y.set(V_Low),
            m.out_y.set(V_Low),
            m.curr_x.set(V_Low),
            m.out_x.set(V_Low),

            m.dp.set(V_Low),

            _StWaitValid
        )


class _StWaitValid(V_State):
    """
    if (valid)
        go to StInitInputReg
    """

    def generate(self, m: Maxpool_NoSoft) -> V_Block:

        return V_Block(
            *V_If(m.valid)(

                _StInitInputReg
            )
        )


class _StInitInputReg(V_State):
    """
    copy each input in input memory into a local register 

    if (all values have been copied)
        go to _StSetWindow
    """

    def generate(self, m: Maxpool_NoSoft) -> V_Block:
        max_inp_addr = m.input_mem.size

        return V_Block(
            m.inp_reg[m.inp_addr - 1].set(m.inp_data),
            m.inp_addr.set(m.inp_addr + 1),

            *V_If(m.inp_addr == max_inp_addr)(
                _StSetWindow
            )
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

    def generate(self, m: Maxpool_NoSoft) -> V_Block:
        # (j, i, k)
        indices = [(j * m.x * m.n, i * m.n, k)
                   for j in range(m.f)
                   for i in range(m.f)
                   for k in range(m.n)]

        return V_Block(
            *[m.window[w].set(m.inp_reg[
                (m.curr_y * m.x * m.n + j) +
                (m.curr_x * m.n + i) + k
            ]) for w, (j, i, k) in enumerate(indices)],

            _StFindMax
        )


class _StFindMax(V_State):
    """
    assign each of the column-wise maximums to the mp_out register

    mp_out of size (y // 2, x // 2, n)

    mp_out[(out_y * x // 2 * n) + (out_x * n) + wc] = max_outputs[wc]
    for window column index wc=0,..., n - 1.
    """

    def generate(self, m: Maxpool_NoSoft) -> V_Block:

        return V_Block(
            *[m.mp_out[
                (m.out_y * (m.x // 2) * m.n) + (m.out_x * m.n) + wc
            ].set(m.max_outputs[wc])
                for wc in range(m.n)],

            _StIncMaxpool
        )


class _StIncMaxpool(V_State):
    """
    set each of the previously set mp_outs to to themselves

    if (curr_x <= x - s - f)
        set curr_x to 0
        set out_x to 0

        if (curr_y <= y - s - f)
            go to _StDotProduct
        else
            inc curr_y by s
            inc out_y by 1

            go to _StSetWindow

    else
        inc curr_x by s
        inc out_x

        go to _StSetWindow
    """

    def generate(self, m: Maxpool_NoSoft) -> V_Block:
        max_y = m.y - m.s - m.f
        max_x = m.x - m.s - m.f

        print(max_y, max_x)

        inds = [(m.out_y * (m.x // 2) * m.n) +
                (m.out_x * m.n) + wc for wc in range(m.n)]

        return V_Block(
            *[m.mp_out[ind].set(m.mp_out[ind]) for ind in inds],

            *V_If(m.curr_x > max_x)(
                m.curr_x.set(V_Low),
                m.out_x.set(V_Low),

                *V_If(m.curr_y > max_y)(

                    _StWaitDotProduct
                ), *V_Else(
                    m.curr_y.set(m.curr_y + m.s),
                    m.out_y.set(m.out_y + 1),

                    _StSetWindow
                )
            ), *V_Else(
                m.curr_x.set(m.curr_x + m.s),
                m.out_x.set(m.out_x + 1),

                _StSetWindow
            )
        )


class _StWaitDotProduct(V_State):
    """
    ** use out_x to iterate over the mp_out array

    - add prod to dot product

    - if (out_x == max - 1)
        - inc weights addr

        - raise the output write enable
        - set the output data to dp + prod + bias

        - go to StWaitExponential
    - else
        - inc out x
        - inc weights addr
    """

    def generate(self, m: Maxpool_NoSoft) -> V_Block:
        max_mp_out_addr = m.mp_out.size - 1

        return V_Block(
            m.dp.set(m.dp + m.prod),
            "\n",
            *V_If(m.out_x == max_mp_out_addr)(
                m.w_addr.set(m.w_addr + 1),
                m.out_we.set(V_High),
                m.out_data.set(m.dp + m.prod + m.biases[m.out_addr]),

                _StWriteData
            ), *V_Else(
                m.out_x.set(m.out_x + 1),
                m.w_addr.set(m.w_addr + 1),
            )
        )


class _StWriteData(V_State):
    """
    lower output write enable
    clear out_x
    clear dot product

    if (output addr == max)
        go to StDone
    else
        inc out addr

        go to _StWaitDotProduct
    """

    def generate(self, m: Maxpool_NoSoft) -> V_Block:
        max_out_addr = m.output_mem.size - 1

        return V_Block(
            m.out_we.set(V_Low),
            m.out_x.set(V_Low),
            m.dp.set(V_Low),

            *V_If(m.out_addr == max_out_addr)(

                V_StDone
            ), *V_Else(
                m.out_addr.set(m.out_addr + 1),

                _StWaitDotProduct
            )
        )
