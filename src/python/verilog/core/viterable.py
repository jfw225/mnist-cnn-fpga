
from math import ceil, log2
from typing import Dict, Iterable, Optional, Tuple
from verilog.core.vspecial import V_Clock, V_Reset
from verilog.utils import format_int, id_generator
from verilog.core.vmodule import V_Module
from verilog.core.vsyntax import V_Array, V_FixedPoint, V_Int, V_ObjectBase, V_Port, V_Variable
from verilog.core.vtypes import ArraySize, BitWidth, NetName, V_File, V_Input, V_Output, V_ParameterArray, V_Reg, V_Wire


class V_Iterable(V_Module):
    """
    An object that implements an iterable verilog data structure. 
    """

    def __init__(
        self,
        width: BitWidth,
        size: ArraySize,
        **kwargs
    ) -> None:

        super().__init__()

        self.size = size
        self.width = width
        # self.addr_width = ceil(log2(self.width))
        self.addr_width = ceil(log2(self.size))
        self.init_data = None

        super().__init__(**kwargs)

        self._base_addr = 0

        self.clk = self.port(V_Input, name="clk")
        self.reset = self.port(V_Input, name="reset")
        self.write_en = self.port(V_Input, name="write_enable")
        self.read_addr = self.port(V_Input, self.addr_width, name="read_addr")
        self.write_addr = self.port(
            V_Input, self.addr_width, name="write_addr")
        self.read_data = self.port(
            V_Output, self.width, V_Reg, name="read_data")
        self.write_data = self.port(V_Input, self.width, name="write_data")

    @property
    def data(self) -> Iterable[V_Int or V_FixedPoint]:
        """
        Returns `self.init_data.data` if it exists. Otherwise, returns an empty 
        list. 
        """

        if self.init_data is not None:

            return self.init_data.data

        return []

    @property
    def read(self) -> Tuple[V_Port, V_Port]:
        """
        Returns two ports:
            read_addr: The port that controls from where data is read.
            read_data: The port into which data is read.
        """

        return (self.read_addr, self.read_data)

    @property
    def write(self) -> Tuple[V_Port, V_Port, V_Port]:
        """
        Returns three ports:
            write_addr: The port that controls to where data is written. 
            write_data: The port from which data is written. 
            write_enable: The port that enables/disables writing.
        """

        return (self.write_addr, self.write_data, self.write_en)

    @property
    def write_enable(self) -> V_Port:
        """
        Returns the port that controls whether or not writing is 
        enabled. 
        """

        return self.write_en

    @property
    def base_addr(self) -> str:
        """
        Returns the base address of this iterable. 
        """

        return format_int(self._base_addr, self.addr_width)

    def set_init_data(
        self,
        init_data: Optional[V_Array] = None,
        file: Optional[V_File] = None
    ):
        if init_data is not None:
            assert (init_data.dtype is V_ParameterArray and
                    init_data.size == self.size)

        # reset objects
        self._objects = [init_data]

        self.init_data = init_data
        if file is not None:
            self.include(file)
