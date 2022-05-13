
from optparse import Option
import os
from sqlite3 import connect
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, Type, TypeVar
from config import V_BASE_PATH
from verilog.core.vspecial import V_Empty
from verilog.utils import nameof
from verilog.utils import id_generator
from verilog.core.vsyntax import V_Array, V_Connection, V_ObjectBase, V_Port, V_Variable
from verilog.core.vtypes import ArraySize, BitWidth, NetName, V_Block, V_DType, V_File, V_Input, V_Line, V_Output, V_Parameter, V_ParameterArray, V_PortType, V_Reg, V_RegArray, V_Wire, V_WireArray


class V_Module:
    """
    The base class for a verilog module.
    """

    def __init__(
        self,
        name: Optional[NetName] = None,
        objects: Optional[Iterable[V_Variable or V_Array or "V_Module"]] = list()
    ):

        # holds the connections made during instantiation
        self._connections: Dict[NetName, Iterable[V_Connection]] = dict()

        # holds the instances created by this module at generation-time
        self._instances = list()

        # hold the instances of this module created at generation-time
        self._own_instances = list()

        # the name of the module and verilog file
        self.name: NetName = name or f"{nameof(self)}_{id_generator()}"

        # the list of objects that will be written along with this module
        self._objects: V_ObjectBase or V_Module = objects

        # the path to the file to which this verilog module is written
        self.path = V_BASE_PATH

        # verilog file imports
        self.includes: Iterable[V_File] = list()

        # inputs and output ports
        self.ports: Iterable[V_Port] = list()

        # local variables
        self.variables: Iterable[V_Variable or V_Array] = list()

    @property
    def connections(self) -> Dict[NetName, Iterable[V_Connection]]:
        """
        Returns the connections made during instantiation. If this is called 
        before `self.instantiate`, an error is thrown. 
        """

        if len(self._connections) == 0:
            raise Exception("Cannot get exceptions before instantiation!")

        return self._connections

    @property
    def instances(self) -> List[Any]:
        """
        Returns a list of `V_Instance` objects created by this module. 
        """

        return self._instances

    @property
    def own_instances(self) -> List[Any]:
        """
        Returns a list of `V_Instance` objects of this module.
        """

        return self._own_instances

    @property
    def input_ports(self):
        """ Returns all the input ports. """

        return [port for port in self.ports if port.port_type is V_Input]

    @property
    def output_ports(self):
        """ Returns all the output ports. """

        return [port for port in self.ports if port.port_type is V_Output]

    @property
    def objects(self):
        """ Returns a flatten list of subobjects used by this module. """

        return [self
                ] + [obj for object in self._objects for obj in object.objects]

    @property
    def nbits(self):
        """ Returns the total number of bits utilized by this module. """

        total = 0
        desc = dict()
        for obj in self.ports + self.variables:
            n = obj.width * obj.size if isinstance(obj, V_Array) else obj.width

            total += n
            desc |= {str(obj): str(n)}

        for obj in self._objects:
            if (not isinstance(obj, V_ObjectBase) or
                    obj.dtype not in [V_Parameter, V_ParameterArray]):
                continue

            n = obj.width * obj.size if isinstance(obj, V_Array) else obj.width
            total += n
            desc |= {str(obj): str(n)}

        for ins in self.instances:
            ins_desc = ins.module.nbits

            if ins.module.name in desc:
                desc[str(ins.module)]["instances"] += 1
            else:
                desc |= ins_desc

            total += int(ins_desc[str(ins.module)]["nbits"])

        return {str(self): {
            "instances": 1,
            "desc": desc,
            "nbits": str(total)
        }}

    # ======================================================================= #
    # Setters

    def add_port(
        self,
        port: V_Port or V_Variable,
        *,
        port_type: Optional[V_PortType] = None,
        dtype: Optional[Type[V_DType]] = None,
        name: Optional[NetName] = None
    ) -> V_Port:
        """
        Adds an existing module port `port` to this module by creating a copy
        of the port with net name `name`. If `name` is `None`, then `port.name` 
        is used.

        If `port` is an instance of `V_Variable`, `port_type` must be provided.

        If `dtype` is provided, the copy is cast to data type `dtype`. 
        Otherwise, `port.dtype` is used.
        """

        assert isinstance(port, (V_Port, V_Variable))

        port_type = port_type or port.port_type
        dtype = dtype or port.dtype
        name = name or port.name

        port_copy = V_Port(module=self,
                           port_type=port_type,
                           width=port.width,
                           dtype=dtype,
                           signed=port.signed,
                           name=name)
        self.ports.append(port_copy)

        return port_copy

    def add_var(
        self,
        var: V_Port or V_Variable or V_Array,
        *,
        dtype: Optional[Type[V_DType]] = None,
        name: Optional[NetName] = None
    ) -> V_Variable or V_Array:
        """
        Adds an existing local variable `var` to this module by creating a copy 
        of the variable with the net name `name`. If `name` is `None`, then 
        `var.name` is used. 

        If `dtype` is provided, the copy is cast to data type `dtype`. 
        Otherwise, `var.dtype` is used.
        """

        if isinstance(var, V_Empty):
            return var

        assert isinstance(var, (V_Port, V_Variable, V_Array))

        if dtype is None:
            dtype = V_Wire if var.dtype is V_DType else var.dtype

        name = name or var.name

        if isinstance(var, (V_Port, V_Variable)):
            var_copy = V_Variable(module=self,
                                  dtype=dtype,
                                  width=var.width,
                                  signed=var.signed,
                                  data=var.data,
                                  name=name)

        elif isinstance(var, V_Array):
            var_copy = V_Array(module=self,
                               dtype=dtype,
                               width=var.width,
                               size=var.size,
                               signed=var.signed,
                               data=var.data,
                               name=name)

        else:
            raise Exception(f"Error adding variable: {var.name}")

        self.variables.append(var_copy)

        return var_copy

    def include(self, v_file: V_File):
        """ Imports modules from verilog file `v_file`. """

        self.includes.append(v_file)

    def port(
        self,
        port_type: Type[V_PortType],
        width: Optional[BitWidth] = BitWidth(1),
        dtype: Optional[Type[V_DType]] = V_DType,
        signed: Optional[bool] = False,
        name: Optional[NetName] = None
    ) -> V_Port:
        """ Adds a new port to the module. """

        name = name or f"{port_type()}_{id_generator()}"

        port = V_Port(module=self,
                      port_type=port_type,
                      width=width,
                      dtype=dtype,
                      signed=signed,
                      name=name)
        self.ports.append(port)

        return port

    def var(
        self,
        dtype: Optional[Type[V_DType]] = V_DType,
        width: Optional[BitWidth] = BitWidth(1),
        size: Optional[ArraySize] = ArraySize(0),
        signed: Optional[bool] = False,
        data: Optional[any] = None,
        name: Optional[NetName] = None
    ) -> V_Variable or V_Array:
        """ Adds a new variable to the module. """

        if dtype is V_Reg or dtype is V_Wire or dtype is V_Parameter:
            var = V_Variable(module=self,
                             dtype=dtype,
                             width=width,
                             signed=signed,
                             data=data,
                             name=name)
        elif dtype is V_RegArray or dtype is V_WireArray:
            var = V_Array(module=self,
                          dtype=dtype,
                          width=width,
                          size=size,
                          signed=signed,
                          data=data,
                          name=name)
        else:
            raise Exception(f'"{dtype}" is not a valid data type.')
        self.variables.append(var)

        return var

    def add_objects(
        self,
        objects: Iterable[V_Variable or V_Array or "V_Module"]
    ) -> None:

        return self._objects.extend(objects)

    # ======================================================================= #
    # Verilog Generation and Output functions

    def generate(self) -> V_Block:
        """ Overload this function to write module generation. """

        return V_Block(V_Line(""))

    def _generate(self) -> V_Block:
        """
        Don't overload this function. Used by `V_File` to write the module.
        """

        assert len(self.ports) > 0, "Must have at least one port."

        # generate the inner lines
        inner_lines = []
        for line in self.generate():
            stripped_line = line.lstrip("\n")
            num_new_lines = len(line) - len(stripped_line)

            inner_lines.append(
                "\n" * num_new_lines + "\t" + stripped_line + "\n")

        return V_Block(
            # the module header
            f"module {self.name}(",

            # the input and output ports
            *[f"\n\t{inp.define()}," for inp in self.ports[:-1]],
            f"\n\t{self.ports[-1].define()}\n);\n",

            # parameter instantiations
            *[f"\t`{obj}\n" for obj in self._objects
              if isinstance(obj, V_ObjectBase) and (
                  obj.dtype is V_Parameter or
                  obj.dtype is V_ParameterArray
              )],

            # the variables and arrays
            * [f"\n\t{var.define()}" for var in self.variables], "\n\n",

            # the inner function
            *inner_lines,

            # the module footer
            "\nendmodule\n\n"
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.instantiate(*args, **kwds)

    def instantiate(
        self,
        instantiator: "V_Module",
        *connections: Iterable[
            V_Variable or
            Tuple[V_Variable, V_Port] or
            V_Connection]
    ):
        """
        We require `instantiator` to keep track of the net names used to 
        instantiate this module based on the name of the instantiator. 

        It must be the case that there is exactly one connection in
        `connections` for each port in `self.ports`. 

        If a `V_Variable` object exists in, it's name must be 
        exactly that of a port in this module.
        """

        from verilog.core.vinstance import V_Instance

        return V_Instance(instantiator, self, *connections)

    def tofile(self, name: V_File):
        from verilog.core.vfile import V_FileWriter

        writer = V_FileWriter(
            name,
            objects=[self]
        )
        # writer.write()

        return writer

    def __str__(self) -> str:

        # print(type(self))
        return self.name


# must be a `V_Module`
M = TypeVar("M")


class V_ConnSpec(Generic[M]):
    """
    Forms all the connections required for some `V_Module` object `other`
    by creating local variables in `V_Module` object `module` with prefix 
    `prefix`. Existing local variables can be provided as key word arguments, 
    but it must be the case that `other.net` exists for some `vars[net]`.
    """

    def __init__(
        self,
        module: V_Module,
        other: M,
        prefix: Optional[NetName] = None,
        **vars: Optional[Dict[NetName, V_ObjectBase]]
    ):
        prefix = prefix or id_generator()

        self._nets = [
            key for key, val in other.__dict__.items()
            if isinstance(val, V_Port)
        ]

        self._connections: Iterable[V_Connection] = list()

        for net in self._nets:
            port = getattr(other, net)
            if net in vars:
                var = vars[net]
            else:
                var = module.add_var(port, name=f"{prefix}_{port}")

            setattr(self, net, var)
            self._connections.append(V_Connection(var, port))

    def __iter__(self):

        return iter(self._connections)
