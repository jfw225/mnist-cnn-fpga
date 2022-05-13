from verilog.core.vspecial import V_High
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_Block, V_DType, V_Expression, V_Input, V_Output, V_Wire


class SignedMult(V_Target):
    """
    The signed multiplication verilog module.
    """

    def __init__(
        self,
        int_width: BitWidth,
        dec_width: BitWidth
    ):
        super().__init__()

        self.int_width = int_width
        self.dec_width = dec_width
        self.width = int_width + dec_width

        self.a = self.port(V_Input, self.width, signed=True)
        self.b = self.port(V_Input, self.width, signed=True)
        self.out = self.port(V_Output, self.width, signed=True)

        # change the data type of `self.done` so it can be assigned
        self.done.dtype = V_DType

    def generate(self):

        N, M = self.int_width, self.dec_width

        mult_out = self.var(V_Wire, width=(
            (N + M) * 2 - 1) + 1, signed=True, name="mult_out")

        return V_Block(
            "// tie `done` to `HIGH`",
            self.done.set(V_High),

            "\n// intermediate full bit length mult",
            mult_out.set(self.a * self.b),

            "\n// select bits for `N.M` fixed point",
            self.out.set(V_Expression(
                f"{{{mult_out[(N + M) * 2 - 1]}, {mult_out[M + (M + N - 2):M]}}}"
            ))
        )
