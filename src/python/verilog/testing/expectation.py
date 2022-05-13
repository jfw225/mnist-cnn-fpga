
from typing import Iterable
from verilog.core.vinstance import V_Signal
from verilog.core.vsyntax import V_FixedPoint, V_Int

ExpectationData = (
    int or
    V_Int or
    V_FixedPoint or
    Iterable[V_Int or V_FixedPoint]
)


class Expectation:
    def __init__(
        self,
        signal: V_Signal,
        data: ExpectationData
    ) -> None:
        pass
