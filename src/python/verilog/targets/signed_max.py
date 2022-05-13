from verilog.core.vspecial import V_High
from verilog.core.vsyntax import V_Ternary
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_DType, V_Input, V_Output


class SignedMax(V_Target):
    """
    The signed maximum verilog module. Returns max(a, b).
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

        self.input0 = self.port(V_Input, self.width, signed=True)
        self.input1 = self.port(V_Input, self.width, signed=True)
        self.out = self.port(V_Output, self.width, signed=True)

        # change the data type of `self.done` so it can be assigned
        self.done.dtype = V_DType

    def generate(self):

        return [
            "// tie `done` to `HIGH`",
            self.done.set(V_High),

            "\n// use ternary to determine the max",
            self.out.set(V_Ternary(self.input0 > self.input1)(
                self.input0,
                self.input1
            )),
        ]
