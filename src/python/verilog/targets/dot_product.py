from typing import Optional
from verilog.core.vspecial import V_High, V_Low
from verilog.core.vstate import V_State, V_StateMachine
from verilog.core.vsyntax import V_If
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_Block, V_Input, V_Output, V_Reg, V_Wire
from verilog.targets.mult import Mult
from verilog.targets.signed_mult import SignedMult


class DotProduct(V_Target):
    """
    This module takes ingests two `V_Iterables`--let those two objects 
    be `A` and `B`, and let their elements be `a_i` and `b_i` 
    respectively.

    Then this module returns the sum of all `a_i * b_i` for 
    `0 <= i <= |A| - 1 = |B| - 1`. Note, it must be the case that
     `|A|==|B|`.

    This module accomplishes its task by taking two inputs `a` and `b` and 
    one output `out` in addition to the inherited inputs 
    `clk, reset, done, valid`. It also uses a local register called `sum` 
    to keep track of the sum.

    On `reset`, the `sum` is set to zero and `done` is cleared. 

    When `valid` is HIGH, `done` is cleared. 

    When `done` is LOW, the module is processing.

    When `done` is HIGH, `out` is valid and module idles until reset.
    """

    def __init__(
        self,
        width: BitWidth,
        mult: Optional[Mult or SignedMult] = None
    ) -> None:

        if mult is not None:
            assert isinstance(mult, (Mult, SignedMult))
            assert mult.width == width

        # create the multiplier
        self.mult = mult or Mult(width)

        super().__init__(objects=[self.mult])

        self.width = width

        self.a = self.port(V_Input, self.width)
        self.b = self.port(V_Input, self.width)
        self.out = self.port(V_Output, self.width)

        self.sum = self.var(V_Reg, self.width, name="sum")

        # product of mult
        self.prod = self.var(V_Wire, self.width, name="prod")

        # reset flag for mult
        self.mult_reset = self.var(V_Reg, 1, name="mult_reset")

        # done flag for mult
        self.mult_done = self.var(V_Wire, 1, name="mult_done")

        # create the state machine
        self.vsm = V_StateMachine(
            _StReset, _StWaitValid, _StWaitMultDone
        )

    def generate(self):
        a, b, out = self.a, self.b, self.out

        return V_Block(
            "// instantiate the state machine",
            *self.vsm.generate(
                self,
                self.clk,
                self.reset,
                self.done
            ),

            "\n\n// instantiate the multiplier",
            *self.mult(
                self,
                self.clk,
                (self.mult_reset, self.mult.reset),
                (1, self.mult.valid),
                (self.mult_done, self.mult.done),
                (self.a, self.mult.input_ports[0]),
                (self.b, self.mult.input_ports[1]),
                (self.prod, self.mult.output_ports[0])
            ),

            "\n// assign the output",
            out.set(self.sum)
        )


"""
states:
reset
valid
"""


class _StReset(V_State):
    """
    - clear `done`
    - set `mult_reset`
    - set `sum` to zero
    - go to StWaitValid
    """

    def generate(self, module: DotProduct) -> V_Block:

        return V_Block(
            module.mult_reset.set(V_High),
            module.sum.set(0),
            _StWaitValid
        )


class _StWaitValid(V_State):
    """
    - if `valid`:
        - clear `done`
        - clear `mult_reset`
        - go to StWaitMultDone
    """

    def generate(self, module: DotProduct) -> V_Block:

        return V_Block(
            *V_If(module.valid)(
                module.done.set(V_Low),
                module.mult_reset.set(V_Low),
                _StWaitMultDone
            )
        )


class _StWaitMultDone(V_State):
    """
    - if `mult_done`:
        - set `mult_reset`
        - add `prod` to `sum` 
        - set `done`
        - go to StWaitValid
    """

    def generate(self, module: DotProduct) -> V_Block:

        return V_Block(
            *V_If(module.mult_done)(
                module.mult_reset.set(V_High),
                module.sum.set(module.sum + module.prod),
                module.done.set(V_High),
                _StWaitValid
            )
        )
