import numpy as np

from typing import Optional
from verilog.core.vspecial import V_Done, V_Empty
from verilog.core.vsyntax import V_Chain_Op, V_FixedPoint, V_Int, V_Sum
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_Block, V_DType, V_Expression, V_Input, V_Output, V_Wire
from verilog.targets.signed_div import SignedDiv
from verilog.targets.signed_mult import SignedMult
from verilog.utils import dec2bin


class SignedExponential(V_Target):
    """
    Approximates the `e^x` using the taylor expansion:
    e^x = 1 + x + x^2 / 2! + x^3 / 3! + ... + x^n / n!
    """

    def __init__(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        num_terms: int
    ):

        assert num_terms > 0

        # create mult and div modules
        self.mult = SignedMult(int_width, dec_width)
        self.div = SignedDiv(int_width, dec_width)

        super().__init__(objects=[self.mult, self.div])

        # force `self.done` to be a wire to make it assignable
        self.done.dtype = V_DType

        self.int_width = int_width
        self.dec_width = dec_width
        self.width = int_width + dec_width

        # the number of terms in the taylor series for approximation
        self.num_terms = num_terms

        self.input = self.port(V_Input, width=self.width,
                               signed=True, name="x_1")
        self.output = self.port(V_Output, width=self.width, signed=True)

    def generate(self) -> V_Block:
        n = self.num_terms

        # create `n - 2` x variables
        xs = [dec2bin(1, self.int_width, self.dec_width), self.input
              ] + [self.var(V_Wire, width=self.width, signed=True,
                            name=f"x_{i}") for i in range(2, n)]

        # create `n - 2` div out variables
        x_div = [V_FixedPoint(1, self.int_width, self.dec_width), self.input
                 ] + [self.var(V_Wire, width=self.width, signed=True,
                               name=f"x_{i}_div_{i}fac") for i in range(2, n)]

        # need `n - 2` mults
        mults = [self.mult.instantiate(
            self,
            self.clk,
            self.reset,
            self.valid,
            (V_Empty(), self.mult.done),
            (self.input, self.mult.input_ports[0]),
            (xs[i - 1], self.mult.input_ports[1]),
            (xs[i], self.mult.output_ports[0])
        ) for i in range(2, n)]

        # need `n - 2` done flags for the divs
        div_dones = [None, None] + [self.add_var(
            self.done, dtype=V_Wire, name=f"div_done_{i}") for i in range(2, n)]

        # need `n - 2` divs
        divs = [self.div.instantiate(
            self,
            self.clk,
            self.reset,
            self.valid,
            (div_dones[i], self.div.done),
            (xs[i], self.div.input_ports[0]),
            (V_FixedPoint(np.math.factorial(i), self.int_width, self.dec_width),
             self.div.input_ports[1]),
            (x_div[i], self.div.output_ports[0])
        ) for i in range(2, n)]

        return V_Block(
            "// instantiate the multipliers",
            *[line for ins in mults for line in ins],

            "\n// instantiate the dividers",
            *[line for ins in divs for line in ins],

            "\n// module is done when all divs are done",
            self.done.set(V_Chain_Op("&", *div_dones[2:])),

            "\n",
            self.output.set(V_Sum(*x_div))
        )
