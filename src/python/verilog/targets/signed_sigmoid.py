from typing import Optional
from verilog.core.vspecial import V_Done
from verilog.core.vsyntax import V_FixedPoint
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_Block, V_Input, V_Output, V_Wire
from verilog.targets.signed_div import SignedDiv
from verilog.targets.signed_exponential import SignedExponential


class SignedSigmoid(V_Target):
    def __init__(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        num_terms: int,
    ):
        # create the div and exponential modules
        self.div = SignedDiv(int_width, dec_width)
        self.exp = SignedExponential(int_width, dec_width, num_terms)

        super().__init__(objects=[self.div, self.exp])

        self.int_width = int_width
        self.dec_width = dec_width
        self.width = int_width + dec_width

        self.input = self.port(V_Input, width=self.width, signed=True)
        self.output = self.port(V_Output, width=self.width, signed=True)

    def generate(self) -> V_Block:
        exp_done = self.add_var(self.done, dtype=V_Wire, name="exp_done")

        exp_output = self.add_var(self.input, name="exp_output")

        return V_Block(
            "// instantiate the exponential module",
            *self.exp(
                self,
                self.clk,
                self.reset,
                self.valid,
                (exp_done, self.exp.done),
                (-self.input, self.exp.input_ports[0]),
                (exp_output, self.exp.output_ports[0])
            ),

            f"\n// instantiate the div module",
            *self.div(
                self,
                self.clk,
                self.reset,

                # input is valid when `self.exp` is done
                (exp_done, self.div.valid),

                # this module is done when `self.div` is done
                self.done,
                (V_FixedPoint(1, self.int_width, self.dec_width),
                 self.div.dividend),
                (exp_output + V_FixedPoint(1, self.int_width,
                                           self.dec_width), self.div.divisor),
                (self.output, self.div.quotient)
            )
        )
