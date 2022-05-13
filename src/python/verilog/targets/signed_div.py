from verilog.core.vspecial import V_High, V_Low
from verilog.core.vstate import V_StDone, V_State, V_StateMachine
from verilog.core.vsyntax import V_Else, V_If, V_ObjectBase, V_Ternary
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_Block, V_Expression, V_Input, V_Output, V_Reg

"""
TODO:
put initial statements in reset block
"""


class SignedDiv(V_Target):

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
        self.reg_working_quotient = self.var(
            V_Reg, width=(2 * N + Q - 3) + 1, signed=True, name="reg_working_quotient")

        # final quotient
        self.reg_quotient = self.var(
            V_Reg, width=(N - 1) + 1, signed=True, name="reg_quotient")

        # working copy of the dvidend
        self.reg_working_dividend = self.var(
            V_Reg, width=(N - 2 + Q) + 1, signed=True, name="reg_working_dividend")

        # working copy of the divisor
        self.reg_working_divisor = self.var(
            V_Reg, width=(2 * N + Q - 3) + 1, signed=True, name="reg_working_divisor")

        # register used for dividend correction
        self.dividend_correction = self.var(
            V_Reg, width=(N - 1) + 1, signed=True, name="dividend_correction"
        )

        # register used for divisor correction
        self.divisor_correction = self.var(
            V_Reg, width=(N - 1) + 1, signed=True, name="divisor_correction"
        )

        """
        This is obviously a lot bigger than it needs to be, as we only need
		count to N-1+Q but, computing that number of bits requires a
		logarithm (base 2), and I don't know how to do that in a
		way that will work for everyone
        """
        self.reg_count = self.var(V_Reg, width=(
            N - 1) + 1, signed=True, name="reg_count")

        # the quotient's sign bit
        self.reg_sign = self.var(V_Reg, width=1, name="reg_sign")

        # create the state machine
        self.vsm = V_StateMachine(
            _StReset, _StWaitValid, _StDivCorrection, _StDivision)

    def generate(self) -> V_Block:
        N, Q = self.width, self.dec_width

        return V_Block(
            "// the division result",
            # self.quotient[N - 2:0].set(self.reg_quotient[N - 2:0]),
            # self.quotient[N - 1].set(self.reg_sign),
            self.quotient.set(V_Ternary(self.reg_sign)(
                -V_ObjectBase.from_obj(
                    self.reg_quotient,
                    name=f"{{1'b0, {self.reg_quotient[N - 2:0]}}}"
                ),
                V_ObjectBase.from_obj(
                    self.reg_quotient,
                    name=f"{{{self.reg_sign}, {self.reg_quotient[N - 2:0]}}}"
                )
            )),

            "\n//instantiate the state machine",
            *self.vsm(
                self,
                self.clk,
                self.reset,
                self.done
            )
        )


class _StReset(V_State):
    """
    - init `reg_sign`
    - init `reg_quotient`
    - init `reg_working_dividend`
    - init `reg_working_divisor`

    - go to StWaitValid
    """

    def generate(self, m: SignedDiv) -> V_Block:

        return V_Block(
            m.reg_sign.set(V_Low),
            m.reg_quotient.set(V_Low),
            m.reg_working_dividend.set(V_Low),
            m.reg_working_divisor.set(V_Low),
            "\n",

            _StWaitValid
        )


class _StWaitValid(V_State):
    """
    This state waits for the input data to be valid.

    - if (valid)
        - lower the done flag
        - set `reg_count` to `N + Q - 1`
        - set `reg_sign` based on input

        - init `reg_working_quotient`
        - set `reg_working_dividend` based on input
        - set `reg_working_divisor` based on input

        - set `dividend_correction`
        - set `divisor_correction`

        - go to StDivCorrection
    """

    def generate(self, m: SignedDiv) -> V_Block:
        N, Q = m.width, m.dec_width

        return V_Block(
            *V_If(m.valid)(
                m.done.set(V_Low),
                m.reg_count.set(N + Q - 1),
                m.reg_sign.set(m.dividend[N - 1] ^ m.divisor[N - 1]),
                "\n",
                m.reg_working_quotient.set(V_Low),
                m.reg_working_dividend[N + Q - 2:Q].set(m.dividend[N - 2:0]),
                m.reg_working_divisor[2 * N + Q -
                                      3:N + Q - 1].set(m.divisor[N - 2:0]),
                "\n",
                m.dividend_correction.set(-m.dividend),
                m.divisor_correction.set(-m.divisor),
                "\n",
                _StDivCorrection
            )
        )


class _StDivCorrection(V_State):
    """
    This state is a fix for negative inputs.

    - update `reg_working_dividend` based on the input
    - update `reg_working_divisor` based on the input

    - go to StDivision
    """

    def generate(self, m: SignedDiv) -> V_Block:
        N, Q = m.width, m.dec_width

        return V_Block(
            m.reg_working_dividend[N + Q - 2:Q].set(
                V_Ternary(m.dividend[N - 1])(
                    V_Expression(f"\t\t\t\t{m.dividend_correction[N - 2:0]}"),
                    V_Expression(f"\t\t\t\t{m.dividend[N - 2:0]}")
                )
            ),
            m.reg_working_divisor[2 * N + Q - 3:N + Q - 1].set(
                V_Ternary(m.divisor[N - 1])(
                    V_Expression(f"\t\t\t\t{m.divisor_correction[N - 2:0]}"),
                    V_Expression(f"\t\t\t\t{m.divisor[N - 2:0]}")
                )
            ),
            _StDivision
        )


class _StDivision(V_State):
    """
    - right shift `reg_working_divisor` (reduce the divisor)

    - if (`reg_working_dividend` >= `reg_working_divisor`)
        - set `reg_working_quotient[reg_count] to 1
        - subtract `reg_working_divisor` from `reg_working_dividend`

    - if (reg_count == 0)
        - raise the done flag
        - set `reg_quotient` to `reg_working_quotient`
        - _StWaitValid
    - else
        - decrement `reg_count`
    """

    def generate(self, m: SignedDiv) -> V_Block:

        return V_Block(
            m.reg_working_divisor.set(m.reg_working_divisor >> 1),
            "\n",
            "// if the dividend is greater than the divisor",
            *V_If(m.reg_working_dividend >= m.reg_working_divisor)(
                m.reg_working_quotient[m.reg_count].set(1),
                m.reg_working_dividend.set(
                    m.reg_working_dividend - m.reg_working_divisor)
            ),
            "\n",
            "// stop condition",
            *V_If(m.reg_count == 0)(
                m.done.set(V_High),
                m.reg_quotient.set(m.reg_working_quotient),
                V_StDone
            ), *V_Else(
                m.reg_count.set(m.reg_count - 1)
            )
        )
