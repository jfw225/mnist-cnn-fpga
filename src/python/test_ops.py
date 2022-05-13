import numpy as np
from verilog.core.vsyntax import V_Chain_Op, V_FixedPoint
from verilog.core.vfile import V_FileWriter
from verilog.core.vmodule import V_Module
from verilog.core.vspecial import V_Clock, V_Done, V_Empty, V_High, V_Low, V_Reset, V_Valid
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import V_Block, V_Output, V_Parameter, V_Reg, V_Wire
from verilog.targets.signed_div import SignedDiv
from verilog.targets.signed_div_noclk import SignedDivNoClk
from verilog.targets.signed_dot_product import SignedDotProduct
from verilog.targets.signed_exponential import SignedExponential
from verilog.targets.signed_mult import SignedMult
from verilog.targets.signed_sigmoid import SignedSigmoid
from verilog.testing.vtestbench import V_TB_Initial, V_Testbench
from verilog.utils import dec2bin


class OpsTB(V_Testbench):

    def __init__(
        self,
        num0: float or int,
        num1: float or int,
        div: SignedDiv,
        div_noclk: SignedDivNoClk,
        mult: SignedMult,
        exp: SignedExponential,
        sig: SignedSigmoid,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = "OpsTB"

        self.ops_valid = self.add_var(V_Valid(self, name="ops_valid"))

        self.num0 = num0
        self.num1 = num1
        self.div = div
        self.div_noclk = div_noclk
        self.mult = mult
        self.exp = exp
        self.sig = sig

        self.div_out = self.port(
            V_Output, width=self.div.width, signed=True, name="div_out")
        self.div_noclk_out = self.port(
            V_Output, width=self.div_noclk.width, signed=True, name="div_noclk_out")
        self.mult_out = self.port(
            V_Output, width=self.mult.width, signed=True, name="mult_out")
        self.exp_out = self.port(
            V_Output, width=self.exp.width, signed=True, name="exp_out")
        self.sig_out = self.port(
            V_Output, width=self.sig.width, signed=True, name="sig_out")

    def generate(self):
        div_done = self.add_var(self.done, name="div_done")
        div_noclk_done = self.add_var(V_Done(self, name="div_noclk_done"))
        mult_done = self.add_var(V_Done(self, name="mult_done"))
        exp_done = self.add_var(V_Done(self, name="exp_done"))
        sig_done = self.add_var(V_Done(self, name="sig_done"))

        int_width = self.div.int_width
        dec_width = self.div.dec_width
        width = self.div.width

        num0 = self.var(V_Parameter, width, signed=True, data=dec2bin(
            self.num0, int_width, dec_width), name="num0")
        num1 = self.var(V_Parameter, width, signed=True, data=dec2bin(
            self.num1, int_width, dec_width), name="num1")

        def sigmoid(x): return 1/(np.exp(-x)+1)
        _cor_sig = sigmoid(self.num0)
        # _cor_sig = np.exp(-self.num0) + 1

        cor_div = self.var(V_Parameter, width, data=dec2bin(
            self.num0 / self.num1, int_width, dec_width), signed=True, name="cor_div")
        cor_mult = self.var(V_Parameter, width, data=dec2bin(
            self.num0 * self.num1, int_width, dec_width), signed=True, name="cor_mult")
        self.cor_exp = self.var(V_Parameter, width, data=dec2bin(
            np.exp(self.num0), int_width, dec_width), signed=True, name="cor_exp")
        cor_sig = self.var(V_Parameter, width, data=dec2bin(
            _cor_sig, int_width, dec_width), signed=True, name="cor_sig")

        self.temp = self.add_var(self.exp_out, name="temp")

        return super().generate(V_Block(
            "// drive op valid signal",
            *V_TB_Initial(
                "#20",
                self.ops_valid.set(V_High),
                "#30",
                self.ops_valid.set(V_Low)
            ),

            "\n// instantiate the divider",
            *self.div(
                self,
                self.clk,
                self.reset,
                (self.ops_valid, self.div.valid),
                (div_done, self.div.done),
                (num0, self.div.input_ports[0]),
                (num1, self.div.input_ports[1]),
                (self.div_out, self.div.output_ports[0])
            ),

            "\n// instantiate the unclocked divider",
            *self.div_noclk(
                self,
                self.clk,
                self.reset,
                (self.ops_valid, self.div_noclk.valid),
                (div_noclk_done, self.div_noclk.done),
                (num0, self.div_noclk.input_ports[0]),
                (num1, self.div_noclk.input_ports[1]),
                (self.div_noclk_out, self.div_noclk.output_ports[0])
            ),

            "\n// instantiate the multiplier",
            *self.mult(
                self,
                self.clk,
                self.reset,
                (self.ops_valid, self.mult.valid),
                (mult_done, self.mult.done),
                (num0, self.mult.input_ports[0]),
                (num1, self.mult.input_ports[1]),
                (self.mult_out, self.mult.output_ports[0])
            ),

            "\n// instantiate the exponential module",
            *self.exp(
                self,
                self.clk,
                self.reset,
                (self.ops_valid, self.exp.valid),
                (exp_done, self.exp.done),
                (num0, self.exp.input_ports[0]),
                (self.exp_out, self.exp.output_ports[0])
            ),

            "\n// instantiate the sigmoid module",
            *self.sig(
                self,
                self.clk,
                self.reset,
                (self.ops_valid, self.sig.valid),
                (sig_done, self.sig.done),
                (num0, self.sig.input_ports[0]),
                (self.sig_out, self.sig.output_ports[0])
            ),

            "\n// tie the done signal to all of the operations",
            self.done.set(V_Chain_Op("&",
                          div_done,
                                     #   div_noclk_done,
                          mult_done,
                          exp_done,
                          sig_done))
        ))

    def presim(self):
        self.log(self.div.quotient)
        self.log(self.ops_valid)
        self.log(self.done)
        self.log(self.cor_exp)

        cor_div = V_FixedPoint(self.num0 / self.num1, int_width, dec_width)
        cor_exp = V_FixedPoint(np.exp(self.num0), int_width, dec_width)

        self.expect(self.div_out, cor_div)
        self.expect(self.exp_out, cor_exp)

    def postsim(self, data):
        """"""


if __name__ == '__main__':
    int_width = 44  # 32 - 15
    dec_width = 44  # 15

    num0 = 3.05
    num1 = 3

    signed_div = SignedDiv(int_width, dec_width)
    div_file = signed_div.tofile("signed_div")

    signed_div_noclk = SignedDivNoClk(int_width, dec_width)
    div_noclk_file = signed_div_noclk.tofile("signed_div_noclk")

    signed_mult = SignedMult(int_width, dec_width)
    mult_file = signed_mult.tofile("signed_mult")

    signed_exp = SignedExponential(int_width, dec_width, num_terms=8)
    exp_file = signed_exp.tofile("signed_exp")

    signed_sig = SignedSigmoid(int_width, dec_width, 8)
    sig_file = signed_sig.tofile("signed_sig")

    signed_dot = SignedDotProduct(int_width, dec_width)
    dot_file = signed_dot.tofile("signed_dot_product")

    ops_tb = OpsTB(num0, num1, signed_div, signed_div_noclk,
                   signed_mult, signed_exp, signed_sig,
                   objects=[signed_div,
                            signed_div_noclk,
                            signed_mult,
                            signed_exp,
                            signed_sig,
                            signed_dot]
                   )

    # ops_tb.include(div_file)
    # ops_tb.include(div_noclk_file)
    # ops_tb.include(mult_file)
    # ops_tb.include(exp_file)
    # ops_tb.include(sig_file)
    # ops_tb.include(dot_file)

    # file = ops_tb.tofile("OpsTB")
    # file.write()
    ops_tb.simulate(headless=False)
    # ops_tb._simulator.compile()
