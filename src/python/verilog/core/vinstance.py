
from typing import Dict, Iterable, List, Optional, Tuple, Type
from verilog.core.vmodule import V_Module
from verilog.core.vtypes import ArraySize, BitWidth, NetName, V_Block, V_DType, V_ParameterArray, V_RegArray, V_WireArray
from verilog.utils import id_generator
from verilog.core.vsyntax import V_Array, V_Connection, V_ObjectBase, V_Port, V_Variable
from verilog.core.vspecial import V_Empty


class V_Signal(V_ObjectBase):
    """
    Like a `V_ObjectBase` object, but exclusively used for testing.
    """

    def __init__(
        self,
        *,
        instance: "V_Instance",
        dtype: Type[V_DType],
        width: BitWidth,
        size: Optional[ArraySize] = None,
        signed: bool,
        name: NetName
    ) -> None:

        super().__init__(module=instance.module,
                         dtype=dtype,
                         width=width,
                         signed=signed,
                         name=name)

        # record the size
        self.size = size

        # store the instance
        self.instance = instance

    @classmethod
    def from_obj(cls, ins: "V_Instance", obj: V_ObjectBase) -> "V_Signal":
        assert isinstance(obj, V_ObjectBase)

        size = None
        if obj.dtype in [V_ParameterArray, V_RegArray, V_WireArray]:
            size = obj.size

        return cls(instance=ins,
                   dtype=obj.dtype,
                   width=obj.width,
                   size=size,
                   signed=obj.signed,
                   name=obj.name)


class V_Instance:
    def __init__(
        self,
        instantiator: V_Module,
        module: V_Module,
        *connections: Iterable[V_Variable or
                               Tuple[V_Variable, V_Port] or
                               V_Connection]
    ) -> None:
        """
        Parameters:
            `instantiator` -- `V_Module` object
                The module that comprises this instance.

            `module` -- `V_Module` object
                The module of which this is an instance.

            `name` -- `NetName` object
                The name of this instance.
        """

        assert isinstance(instantiator, V_Module), f"{instantiator}"

        self.instantiator = instantiator
        self.module = module
        self.name = f"{module}_{id_generator()}"
        self._connections: List[V_Connection] = []
        self._nets: List[NetName] = []
        self._signals: List[V_Signal] = []

        self._configure_connections(*connections)
        self._configure_signals()

    @property
    def ins_path(self):
        """
        Returns the instantiation path up until this instance. 
        """
        # name = f"{instance.instantiator}/{instance}/{name}"

    def infer_connection(self, obj):
        """
        Attempts to infer the connections of `obj` and `self.module`.
        """

        if not isinstance(obj, (V_Port, V_Variable)):
            return V_Connection(*obj)

        # infer the connection by name
        try:
            [match] = [
                match for match in self.module.ports if match.name == obj.name]
            assert match.width == obj.width, "Must have equal widths."
        except Exception as e:
            raise Exception("Was not able to infer a connection for " +
                            f'variable "{obj}" due to the following error: {e}')

        return V_Connection(obj, match)

    def __iter__(self):
        """
        Used to fill a `V_Block` with an instantiation of `self.module`.
        """

        return iter(V_Block(
            # instance header
            f"{self.module} {self}(",

            # net connections
            *[f"\t{c}, " for c in self._connections[:-1]],
            f"\t{self._connections[-1]}",

            # instance footer
            ");\n"
        ))

    def __str__(self):
        """
        Returns `self.name`.
        """

        return self.name

    def _configure_connections(
        self,
        *connections: Iterable[V_Variable or
                               Tuple[V_Variable, V_Port] or
                               V_Connection]
    ):
        # attempt to infer connections
        connections = [*map(self.infer_connection, connections)]

        port_map = {}
        for conn in connections:
            # verify that `conn.port` is a valid port
            port_match = [
                port for port in self.module.ports if port.name == conn.port.name]

            if ((len(port_match) != 1 or conn.var.width != conn.port.width)
                    and not isinstance(conn.var, V_Empty)):
                raise Exception(
                    f'"{conn}" is not a valid connection:\nPort Match: {[str(port) for port in port_match]}\nConn Var Width: {conn.var.width}\nConn Port Width: {conn.port.width}')

            port_map[port_match[0].name] = 1

        # verify that every port is in a connection
        for port in self.module.ports:
            if not port_map.get(port.name):
                raise Exception(f'"{port.name}" was left unconnected.')

        # store connections
        self._connections = connections
        self.module._connections[self.instantiator.name] = connections
        self.module._own_instances.append(self)
        self.instantiator._instances.append(self)

    def _configure_signals(self):
        """
        Creates a `V_Signal` object for every port and variable in 
        `self.module`.
        """

        # get the net name of every port, variable, and array
        self._nets = [
            key for key, val in self.module.__dict__.items()
            if isinstance(val, (V_Port, V_Variable, V_Array))
        ]

        for net in self._nets:
            # create a new signal from the existing object
            signal = V_Signal.from_obj(self, getattr(self.module, net))

            # store the signal
            self._signals.append(signal)
            setattr(self, net, signal)
