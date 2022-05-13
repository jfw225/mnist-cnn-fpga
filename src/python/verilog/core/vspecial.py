"""
This file holds special variables and ports. 
"""
from typing import Optional
from verilog.core.vsyntax import V_Port, V_Variable
from verilog.core.vtypes import BitWidth, NetName, V_Reg, V_Wire

""" The value of a signal in a `LOW` state. """
V_Low = 0

""" The value of a signal in a `HIGH` state. """
V_High = 1


class V_Special(V_Variable):

    def __init__(
        self,
        module,
        width: BitWidth,
        base_name: NetName,
        name: Optional[NetName]
    ) -> None:

        if name is not None and base_name not in name:
            name = name + "_" + base_name

        super().__init__(module=module, dtype=V_Reg, width=width, name=name or base_name)

    @classmethod
    def isinstance(cls, instance) -> bool:

        if not isinstance(instance, (V_Port, V_Variable)):
            return False

        return cls.WIDTH == instance.width and cls.BASE_NAME in instance.name


class V_Clock(V_Special):
    """
    The variable used as a clock line.
    """

    WIDTH = 1
    BASE_NAME = "clk"

    def __init__(self, module, name: Optional[NetName] = None):

        super().__init__(module, V_Clock.WIDTH, V_Clock.BASE_NAME, name)


class V_Reset(V_Special):
    """
    The variable used to indicate whether or not a module should reset. 
    """

    WIDTH = 1
    BASE_NAME = "reset"

    def __init__(self, module, name: Optional[NetName] = None):

        super().__init__(module, V_Reset.WIDTH, V_Reset.BASE_NAME, name)


class V_Stop(V_Special):
    """
    The variable used in a testbench that is used to halt the simulation. 
    """

    WIDTH = 1
    BASE_NAME = "stop"

    def __init__(self, module, name: Optional[NetName] = None) -> None:
        super().__init__(module, V_Stop.WIDTH, V_Stop.BASE_NAME, name)


class V_Valid(V_Special):
    """
    The variable used to indicate to a module whether or not its inputs are 
    valid, and thus, it should begin its task.
    """

    WIDTH = 1
    BASE_NAME = "valid"

    def __init__(self, module, name: Optional[NetName] = None):

        super().__init__(module, V_Valid.WIDTH, V_Valid.BASE_NAME, name)


class V_Done(V_Special):
    """
    The variable used to indicate whether or not a module has finished 
    processing. 
    """

    WIDTH = 1
    BASE_NAME = "done"

    def __init__(self, module, name: Optional[NetName] = None):

        super().__init__(module, V_Done.WIDTH, V_Done.BASE_NAME, name)


class V_Ready(V_Special):
    """
    The variable used to indicate whether or not a module is ready 
    to start processing. 
    """

    WIDTH = 1
    BASE_NAME = "ready"

    def __init__(self, module, name: Optional[NetName] = None):

        super().__init__(module, V_Ready.WIDTH, V_Ready.BASE_NAME, name)


class V_Empty(V_Variable):
    """
    The variable used to indicate an empty connection.
    """

    def __init__(self):
        super().__init__(None, V_Wire)
        self.name = ""
