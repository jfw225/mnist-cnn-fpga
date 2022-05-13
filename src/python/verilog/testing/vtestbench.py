from math import ceil, log2
from os import PathLike
from typing import Iterable, Optional
from verilog.config import V_MSIM_COMPILE, V_MSIM_VSIM, V_WORK_FOLDER
from verilog.core.vinstance import V_Instance, V_Signal

from verilog.core.vmodule import V_Module
from verilog.testing.vsimulator import V_Simulator
from verilog.core.vspecial import V_Clock, V_Done, V_High, V_Low, V_Reset, V_Stop
from verilog.core.vsyntax import V_Always, V_FixedPoint, V_Int, V_ObjectBase
from verilog.core.vtypes import V_Block, V_Expression, V_File, V_Line, V_PosEdge, V_Reg, V_Wire
from verilog.testing.vwavedata import V_WaveData
from verilog.utils import run_cmd


class V_Testbench(V_Module):

    def __init__(
        self,
        timeout: Optional[int or float] = 1e5,
        **kwargs
    ):
        super().__init__(**kwargs)

        if "name" in kwargs:
            self.name = kwargs["name"]

        # create clock and reset variables
        self.clk = self.add_var(V_Clock(self))
        self.reset = self.add_var(V_Reset(self))

        # create a done variable and store the timeout
        self.done = self.add_var(V_Done(self), dtype=V_Wire)
        self.timeout = int(timeout)

        # create a clock cycle counter
        self.counter = self.var(V_Reg, width=ceil(
            log2(timeout)) + 1, name="counter")

        # create the simulator
        self._simulator = V_Simulator(self)

    @property
    def nbits(self):
        """
        Returns the total number of bits utilized by the modules
        in this testbench. 
        """

        return super().nbits

    def generate(self, tb_code: Optional[V_Block] = V_Block()) -> V_Block:
        """
        Subclasses should call `super().generate`. 
        """

        return V_Block(
            "// initialize clock, reset, and counter, and drive reset",
            *V_TB_Initial(
                self.clk.set(V_Low),
                self.reset.set(V_High),
                self.counter.set(V_Low),
                "#20",
                self.reset.set(V_Low)
            ),

            "\n// toggle the clocks",
            *V_TB_Always(
                "#10",
                self.clk.set(~self.clk)
            ),

            "\n// the stop condition for the simulation",
            *V_TB_Always(
                V_TB_Wait(self.done._or(self.counter >= self.timeout)),
                V_TB_Stop
            ),

            "\n// increment the cycle counter",
            *V_Always(V_PosEdge, self.clk)(
                self.counter.set(self.counter + 1)
            ),
            "\n\n",

            *tb_code
        )

    def ins_path(self, signal: V_Signal):
        """
        Returns the full instance path for `signal`. 
        """

        # if the signal is a variable in the test bench, just return the name
        if any([signal is obj for obj in self.ports + self.variables]):
            return f"{self}/{signal}"

        assert isinstance(
            signal, V_Signal), f"{signal} is not a `V_Signal` object."

        def _ins_path_dfs(module: V_Module):
            """
            Base Cases:
                - `module.instances` is empty -> return `False`
                - `module.instances` contains `signal.instance` -> return `path`
            """

            for instance in module.instances:
                if instance is signal.instance:
                    return f"{instance}/{signal}"

                result = _ins_path_dfs(instance.module)
                if result is not False:
                    return f"{instance}/{result}"

            return False

        # dfs to find the instance
        path = _ins_path_dfs(self)
        # print(path)

        if path is False:
            raise Exception(
                f"Could not find full instance path for signal: {signal}")

        return f"{self}/{path}"

    def log(self, *signals: Iterable[V_Signal]):
        """
        Adds `V_Signal` object `signal` to the list of signals that will be 
        logged during the simulation. 
        """

        for signal in signals:
            self._simulator.log(signal)

    def expect(
        self,
        signal: V_Signal,
        data: V_Int or V_FixedPoint or Iterable[V_Int or V_FixedPoint]
    ) -> None:
        """
        Creates an expectation and adds it to the wave data expectations.
        """

        return self._simulator.expect(signal, data)

    def presim(self):
        """
        Children should overload this function and put every `self.log` and 
        `self.expect` call in this function. 
        """

    def postsim(self, data: V_WaveData):
        """
        Children should overload this function to analyze the data from the 
        simulation.
        """

    def signal_of_obj(self, obj: V_ObjectBase):
        """
        Attempts to find and return the `V_Signal` copy of `obj`. If it is not 
        found, an error is raised.
        """

        assert isinstance(obj, V_ObjectBase), obj

        if isinstance(obj, V_Signal):
            return obj

        assert isinstance(obj.module, V_Module), obj

        instances = obj.module.own_instances
        if len(instances) != 1:
            raise Exception(
                f"For the signal of {obj} to be inferred, it must be the case that {obj.module} is instantiated exactly once. Otherwise, you must expect/log the signal within the instance you are trying to analyze.")

        [instance] = instances

        try:
            net, *_ = [key for key, val in obj.module.__dict__.items()
                       if val is obj]

            return getattr(instance, net)
        except Exception as e:
            print(obj, instance)

            raise e

    def simulate(self, headless: Optional[bool] = True):
        """
        Simulates this test bench in modelsim. Can be overloaded to use as 
        space for log calls, but must call `super().simulate()`.
        """

        self._simulator.run(headless, presim=self.presim, postsim=self.postsim)


def V_TB_Always(
    *lines: Iterable[V_Line]
) -> V_Block:

    return V_Block(
        "always begin",
        *[f"\t{line}" for line in lines],
        "end"
    )


def V_TB_Initial(
    *lines: Iterable[V_Line]
) -> V_Block:

    return V_Block(
        "initial begin",
        *[f"\t{line}" for line in lines],
        "end"
    )


def V_TB_Forever(
    *lines: Iterable[V_Line]
) -> V_Block:

    return V_Block(
        "forever begin",
        *[f"\t{line}" for line in lines],
        "end"
    )


V_TB_Stop: V_Line = V_Line("$stop;")


def V_TB_Wait(expr: V_Expression) -> V_Line:

    return V_Line(
        f"wait ({expr});"
    )
