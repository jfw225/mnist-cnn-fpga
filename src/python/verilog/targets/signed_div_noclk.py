from verilog.core.vsyntax import V_Int, V_ObjectBase, V_Ternary
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_Block, V_Expression, V_Input, V_Output, V_Reg, V_Wire


class SignedDivNoClk(V_Target):

    def __init__(
        self,
        int_width: BitWidth,
        dec_width: BitWidth
    ):
        super().__init__()

        self.int_width = int_width
        self.dec_width = dec_width
        self.width = int_width + dec_width

        # dividend / divisor = quotient
        self.dividend = self.port(
            V_Input, self.width, signed=True, name="dividend")
        self.divisor = self.port(
            V_Input, self.width, signed=True, name="divisor")
        self.quotient = self.port(
            V_Output, self.width, signed=True, name="quotient")

        # implementation of kenneth's signed div module
        # `+ 1` to account for `- 1` in variable constructor while
        # maintaining kenneth's code
        N, Q = self.width, self.dec_width

        # our working copy of the quotient
        self.working_quotient = self.var(
            V_Wire, width=(2 * N + Q - 3) + 1, signed=True, name="working_quotient")

        # final quotient
        # self.reg_quotient = self.var(
        #     V_Reg, width=(N - 1) + 1, signed=True, name="reg_quotient")

        # working copy of the dvidend
        self.working_dividend = self.var(
            V_Wire, width=(N - 2 + Q) + 1, signed=True, name="working_dividend")

        # working copy of the divisor
        self.working_divisor = self.var(
            V_Wire, width=(2 * N + Q - 3) + 1, signed=True, name="working_divisor")

        # register used for dividend correction
        self.dividend_correction = self.var(
            V_Wire, width=(N - 1) + 1, signed=True, name="dividend_correction"
        )

        # register used for divisor correction
        self.divisor_correction = self.var(
            V_Wire, width=(N - 1) + 1, signed=True, name="divisor_correction"
        )

        """
        This is obviously a lot bigger than it needs to be, as we only need
		count to N-1+Q but, computing that number of bits requires a
		logarithm (base 2), and I don't know how to do that in a
		way that will work for everyone
        """
        # self.reg_count = self.var(V_Reg, width=(
        #     N - 1) + 1, signed=True, name="reg_count")

        # the quotient's sign bit
        self.sign = self.var(V_Wire, width=1, name="sign")

    def generate(self) -> V_Block:
        N, Q = self.width, self.dec_width

        reg_count = N + Q - 1

        dividends = [self.working_dividend] + [self.add_var(
            self.working_dividend, name=f"{self.working_dividend}_{i}"
        ) for i in range(1, reg_count + 2)]
        divisors = [self.working_divisor] + [self.add_var(
            self.working_divisor, name=f"{self.working_divisor}_{i}"
        ) for i in range(1, reg_count + 2)]

        blocks = []

        for i in range(1, reg_count + 2):
            dividend = dividends[i]
            divisor = divisors[i]

            blocks.extend(V_Block(
                "\n",
                divisor.set(divisors[i - 1] >> 1),
                dividend.set(
                    dividends[i - 1] - divisors[i - 1],
                    dividends[i - 1],
                    dividends[i - 1] >= divisors[i - 1]
                ),
                self.working_quotient[reg_count - i + 1].set(
                    V_Int(1, width=1),
                    V_Int(0, width=1),
                    dividends[i - 1] >= divisors[i - 1]
                )

            ))

        return V_Block(
            self.sign.set(self.dividend[N - 1] ^ self.divisor[N - 1]),
            "\n",
            self.dividend_correction.set(-self.dividend),
            self.divisor_correction.set(-self.divisor),
            "\n",
            self.working_dividend.set(
                (self.dividend_correction[N - 2:0], 0),
                (self.dividend[N - 2:0], 0),
                self.dividend[N - 1]
            ),
            self.working_divisor.set(
                (self.divisor_correction[N - 2:0], 0),
                (self.divisor[N - 2:0], 0),
                self.divisor[N - 1]
            ),
            "\n",
            *blocks,

            "\n// assign the unassigned bits of the working quotient",
            self.working_quotient[2 * N + Q - 3:reg_count + 1].set(0),

            "\n// assign output",
            self.quotient.set(
                -V_ObjectBase.to_brace(0,
                                       self.working_quotient[N - 2:0],
                                       self.quotient.width,
                                       signed=True),
                (self.sign, self.working_quotient[N - 2:0]),
                self.sign
            )
        )
