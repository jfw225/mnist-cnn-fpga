

from verilog.core.vspecial import V_High, V_Low
from verilog.core.vsyntax import V_Always, V_Else, V_If
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_Input, V_Output, V_PosEdge


class Mult(V_Target):

    def __init__(
        self,
        width: BitWidth
    ) -> None:

        super().__init__()

        self.width = width

        self.input0 = self.port(V_Input, self.width)
        self.input1 = self.port(V_Input, self.width)
        self.output = self.port(V_Output, self.width)

    def generate(self):
        input0, input1 = self.input0, self.input1
        output = self.output
        return [
            *V_Always(V_PosEdge, self.clk)(
                *V_If(self.reset)(
                    self.done.set(V_Low)
                ), *V_Else(
                    self.done.set(V_High)
                )
            ),
            output.set(
                f"{input0.name} * {input1.name}")
        ]
