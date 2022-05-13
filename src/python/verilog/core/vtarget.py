
from verilog.core.vspecial import V_Clock, V_Done, V_Ready, V_Reset, V_Valid
from verilog.core.vmodule import V_Module
from verilog.core.vsyntax import V_Port
from verilog.core.vtypes import V_DType, V_Input, V_Output


class V_Target(V_Module):
    """
    The implementation of a module that can be used as the target 
    function in a `V_Map` call. 

    Attributes:
        `clk` -- 1-bit input signal
            The port connecting this module to the clock line.

        `reset` -- 1-bit input signal
            The port that enables the caller to reset this module. 

        `valid` -- 1-bit input signal
            The port that indicates whether or not the input data is valid. The
            caller can manipulate this signal to indicate whether or not the 
            input data is valid.

        `done` -- 1-bit output signal
            The port indicating whether or not the target function has finished 
            computation for a given input. When this flag is `HIGH`, the output 
            data is valid. Thus, the target module should raise this signal 
            when it finishes its task.

        `ready` -- 1-bit output signal
            The port indicating whether or not the target module is ready 
            to begin a task.
    """

    def __init__(self, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.clk = self.add_port(
            V_Clock(self), port_type=V_Input, dtype=V_DType)
        self.reset = self.add_port(
            V_Reset(self), port_type=V_Input, dtype=V_DType)
        self.valid = self.add_port(
            V_Valid(self), port_type=V_Input, dtype=V_DType)
        self.done = self.add_port(V_Done(self), port_type=V_Output)
        # self.ready = self.add_port(V_Ready(self), port_type=V_Output)

    @property
    def input_ports(self):
        """
        Returns this module's input ports without `clk` or `reset`.
        """
        return [port for port in super().input_ports if
                port is not self.clk and
                port is not self.reset and
                port is not self.valid]

    @property
    def output_ports(self):
        """
        Returns this module's output ports without `done`. 
        """

        return [port for port in super().output_ports if
                port is not self.done]
        # port is not self.ready]
