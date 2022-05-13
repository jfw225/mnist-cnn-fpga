
from typing import Any, Dict, Generic, Iterable, Optional, Tuple, Type, TypeVar

from verilog.core.vtypes import ArraySize, V_Block, V_Comment, V_DType, V_Expression, V_Input, V_Line, V_NegEdge, V_Output, V_Parameter, V_ParameterArray, V_PosEdge, V_Reg, V_RegArray, V_Wire, V_WireArray
from verilog.utils import dec2bin, format_int, id_generator

from verilog.core.vtypes import BitWidth, NetName, V_File, V_PortType


class V_Int(V_Expression):
    def __init__(self, val: int, width: BitWidth):

        super().__init__()

        self.value = val
        self.width = width
        self.name = None

    def __eq__(self, other: int or "V_Int") -> bool:
        assert isinstance(other, (int, V_Int)), other

        if isinstance(other, int):
            return self.value == other

        return self.value == other.value and self.width == other.width

    def __new__(cls, val: int, width: BitWidth):
        v_int = super().__new__(cls, format_int(val, width))
        v_int.name = v_int

        return v_int


class V_FixedPoint(V_Expression):
    def __init__(
        self,
        val: int or float,
        int_width: BitWidth,
        dec_width: BitWidth
    ):
        super().__init__()

        self.value = val
        self.int_width = int_width
        self.dec_width = dec_width
        self.width = int_width + dec_width

    def __eq__(self, other: int or float or "V_FixedPoint") -> bool:
        assert isinstance(other, (int, float, V_FixedPoint)), other

        if isinstance(other, (int, float)):
            return self.value == other

        return (self.value == other.value and
                self.int_width == other.int_width and
                self.dec_width == other.dec_width)

    def __new__(
        cls,
        val: int or float,
        int_width: BitWidth,
        dec_width: BitWidth
    ):
        v_fix = super().__new__(cls, dec2bin(val, int_width, dec_width))

        return v_fix


class V_ObjectBase:
    def __init__(
        self,
        dtype: V_DType,
        width: BitWidth,
        signed: bool,
        module: Any,
        name: Optional[NetName] = None
    ) -> None:

        self.dtype = dtype
        self.width = width
        self.size = 0
        self.signed = signed
        self.data = None
        self.module = module
        self.name = name or id_generator()

    @classmethod
    def from_obj(
        cls,
        obj: "V_ObjectBase",
        name: Optional[NetName] = None
    ) -> "V_ObjectBase":

        return cls(
            dtype=obj.dtype,
            width=obj.width,
            signed=obj.signed,
            module=obj.module,
            name=name
        )

    @classmethod
    def to_brace(
        cls,
        obj0: int or V_Int or V_FixedPoint or "V_ObjectBase",
        obj1: int or V_Int or V_FixedPoint or "V_ObjectBase",
        width: BitWidth,
        *,
        signed: Optional[bool] = False
    ) -> "V_ObjectBase":

        assert isinstance(obj0, (int, V_Int, V_FixedPoint, V_ObjectBase))
        assert isinstance(obj1, (int, V_Int, V_FixedPoint, V_ObjectBase))
        assert not isinstance(obj0, int) or not isinstance(obj1, int)

        module = obj0.module if isinstance(obj1, int) else obj1.module

        if isinstance(obj0, int):
            obj0 = V_Int(obj0, width - obj1.width)
        elif isinstance(obj1, int):
            obj1 = V_Int(obj1, width - obj0.width)

        assert width == obj0.width + \
            obj1.width, f"{obj0} + {obj1} must equal {width}"

        return V_ObjectBase(module=module,
                            dtype=V_DType,
                            width=width,
                            signed=signed,
                            name=f"{{{obj0}, {obj1}}}")

    @property
    def objects(self):
        """
        Needed for `V_FileWriter`. Returns `[self]` since object cannot have
        any dependencies.
        """

        return [self]

    def set(
        self,
        var0: (int or V_Expression or "V_ObjectBase" or
               Tuple[int or V_Expression or "V_ObjectBase"]),
        var1: (Optional[int or V_Expression or "V_ObjectBase" or
                        Tuple[int or V_Expression or "V_ObjectBase"]]) = None,
        ternary: Optional[V_Expression or "V_ObjectBase"] = None
    ) -> V_Line:
        """ Stores the value of `var` in `self`. """

        assert self.dtype in [V_DType, V_Reg, V_Wire]

        prefix = "" if self.dtype is V_Reg else "assign "
        op = "<=" if self.dtype is V_Reg else "="

        var0, var1 = [V_ObjectBase.to_brace(
            *var, self.width) if isinstance(var, tuple) else var
            for var in [var0, var1]]

        var0 = V_Int(var0, self.width) if isinstance(var0, int) else var0
        var1 = V_Int(var1, self.width) if isinstance(var1, int) else var1

        if var1 is not None:
            assert ternary is not None
            var = V_Ternary(ternary)(var0, var1)

        else:
            var = var0

        return f"{prefix}{self} {op} {var};"

    def __all_exp__(self, other: int or "V_ObjectBase", op: str) -> V_Expression:
        if isinstance(other, int):
            other = V_Int(other, self.width)

        return V_ObjectBase.from_obj(self, name=f"{self} {op} {other}")

    def __add__(
        self,
        other: int or V_Expression or "V_ObjectBase"
    ) -> V_Line:
        """ Formats add operation as a verilog line. """

        return self.__all_exp__(other, "+")

    def __eq__(self, other: int or "V_ObjectBase"):

        return self.__all_exp__(other, "==")

    def __getitem__(self, index: int or slice) -> "V_Variable":
        start = index.start if isinstance(index, slice) else index
        stop = index.stop if isinstance(index, slice) else 0

        if isinstance(index, V_ObjectBase):
            start = 0  # indexing by var is a width of one

        dtype = V_Wire if self.dtype is V_DType else self.dtype

        index = f"{index.start}:{index.stop}" if isinstance(
            index, slice) else index

        return V_Variable(module=self.module,
                          dtype=dtype,
                          width=start - stop + 1,
                          signed=self.signed,
                          name=f"{self.name}[{index}]")

    def __ge__(self, other: int or "V_ObjectBase") -> V_Expression:
        if isinstance(other, int):
            other = V_Int(other, self.width)

        return V_Expression(f"{self} >= {other}")

    def __gt__(self, other: int or "V_ObjectBase"):

        return self.__all_exp__(other, ">")

    def __invert__(self):

        return V_ObjectBase.from_obj(self, name=f"~{self}")

    def __le__(self, other: int or "V_ObjectBase"):

        return self.__all_exp__(other, "<=")

    def __lshift__(self, other: int or "V_ObjectBase"):
        # verilog has signed shift operators
        op = "<<<" if self.signed else "<<"

        return self.__all_exp__(other, op)

    def __lt__(self, other: int or "V_ObjectBase"):
        if isinstance(other, int):
            other = V_Int(other, self.width)

        return V_Expression(f"{self} < {other}")

    def __mul__(self, other: int or "V_ObjectBase"):

        return self.__all_exp__(other, "*")

    def __neg__(self) -> V_Expression:
        assert self.signed, f"{self} must be signed to use __neg__"

        return V_ObjectBase.from_obj(self, name=~self + 1)

    def _or(self, other: int or "V_ObjectBase"):

        return self.__all_exp__(other, "||")

    def __rshift__(self, other: int or "V_ObjectBase") -> V_Expression:
        # verilog has sign shift operators
        op = ">>>" if self.signed else ">>"

        return self.__all_exp__(other, op)

    def __str__(self) -> V_Expression:

        return V_Expression(self.name)

    def __sub__(self, other: int or "V_ObjectBase") -> V_Expression:

        return self.__all_exp__(other, "-")

    def __xor__(self, other: int or "V_ObjectBase") -> V_Expression:
        if isinstance(other, int):
            other = V_Int(other, self.width)

        return V_Expression(f"{self} ^ {other}")


class V_Port(V_ObjectBase):
    def __init__(
        self,
        module,
        port_type: Type[V_PortType],
        width: Optional[BitWidth] = BitWidth(1),
        dtype: Optional[Type[V_DType]] = V_DType,
        signed: Optional[bool] = False,
        name: Optional[NetName] = None
    ) -> None:

        assert port_type is V_Input or port_type is V_Output

        super().__init__(
            dtype=dtype, width=width, signed=signed, module=module, name=name)

        self.port_type = port_type

    def define(self):
        port_type = self.port_type()
        dtype = self.dtype()
        signed = "signed" if self.signed else ""
        width = f"[{self.width - 1}:0]" if self.width > 1 else ""

        return f"{port_type} {dtype} {signed} {width} {self.name}"


class V_Variable(V_ObjectBase):
    def __init__(
        self,
        module,
        dtype: Optional[Type[V_DType]] = V_DType,
        width: Optional[BitWidth] = BitWidth(1),
        signed: Optional[bool] = False,
        data: Optional[Any] = None,
        name: Optional[NetName] = None
    ) -> None:

        super().__init__(
            dtype=dtype, width=width, signed=signed, module=module, name=name)

        self.data = data

        # self.name = f"{name}_{id_generator()}" if name else id_generator()

        assert (
            self.dtype is V_Reg or
            self.dtype is V_Wire or
            self.dtype is V_Parameter
        ), f'"{self.dtype}" is not a valid data type for a variable.'

        if (data is not None):
            assert self.dtype is V_Parameter

    def define(self):
        dtype = self.dtype()
        signed = "signed" if self.signed else ""
        width = f"[{self.width - 1}:0]" if self.width > 1 else ""

        data = ""
        if self.dtype is V_Parameter:
            data = f" = {self.data}"

        return f"{dtype} {signed} {width} {self.name}{data};"


class V_ArrayMeta(type):

    def __len__(self):
        return self.size


class V_Array(V_ObjectBase, metaclass=V_ArrayMeta):
    """
    If `data` is specified, it is stored as the original `data. When formatted,
    however, the indices are reversed to conform to verilog array indexing.
    """

    def __init__(
        self,
        module,
        dtype: Optional[Type[V_DType]] = V_DType,
        width: Optional[BitWidth] = BitWidth(1),
        size: Optional[ArraySize] = ArraySize(1),
        signed: Optional[bool] = False,
        data: Optional[Iterable[Any]] = None,
        name: Optional[NetName] = None
    ) -> None:

        super().__init__(
            dtype=dtype, width=width, signed=signed, module=module, name=name)

        self.size = size
        self.data = data

        assert (
            self.dtype is V_RegArray or
            self.dtype is V_WireArray or
            self.dtype is V_ParameterArray
        )
        assert self.size > 0, "Array size must be greater than 0."

        if data is not None:
            assert self.dtype is V_ParameterArray and len(
                self.data) == self.size

    def define(self):
        dtype = self.dtype()
        signed = "signed" if self.signed else ""
        width = f"[{self.width - 1}:0]" if self.width > 1 else ""
        size = f"[{self.size - 1}:0]"
        data = ""

        if self.dtype is not V_ParameterArray:
            return f"{dtype} {signed} {width} {self.name} {size}{data};"

        data = " = '{ \\\n\t" + \
            ", \\\n\t".join(map(str, self.data[::-1])) + "\\\n}"

        s = f"`define {self.name} parameter bit {width} {self.name} {size}{data};"

        return s

    def get(
        self,
        index: int or V_Line or V_Port or V_Variable,
    ) -> V_Variable:
        """ Gets `self[index]` and returns a V_Variable. """

        if isinstance(index, (V_Port, V_Variable)):
            index = index.name

        if self.dtype is V_WireArray:
            return V_Variable(module=self.module,
                              dtype=V_Wire,
                              width=self.width,
                              signed=self.signed,
                              data=None,
                              name=f"{self.name}[{index}]")
        elif self.dtype is V_RegArray or self.dtype is V_ParameterArray:
            return V_Variable(module=self.module,
                              dtype=V_Reg,
                              width=self.width,
                              signed=self.signed,
                              data=None,
                              name=f"{self.name}[{index}]")

        raise Exception("Error getting in array named: " + self.name)

    def set(
        self,
        index: int or V_Expression or V_Port or V_Variable,
        var: V_Expression or V_Port or V_Variable
    ) -> V_Line:
        """ Stores the value of `var` in `self[index]`. """

        if isinstance(index, (V_Port, V_Variable)):
            index = index.name

        if isinstance(var, (V_Port, V_Variable)):
            var = var.name

        if self.dtype is V_WireArray:
            return f"assign {self.name}[{index}] = {var};"

        elif self.dtype is V_RegArray:
            return f"{self.name}[{index}] <= {var};"

        raise Exception("Error setting in array named: " + self.name)

    def __getitem__(
        self,
        index: int or V_Line or V_Port or V_Variable
    ) -> "V_Variable":

        return self.get(index)

    def __len__(self):

        return self.size


class V_Connection:
    """
    The object representing a connection between a variable and some port of a
    module.
    """

    def __init__(
        self,
        var: int or V_Expression or V_ObjectBase,
        port: V_Port
    ):

        assert isinstance(
            var, (int, V_Expression, V_ObjectBase)), f"{var}"
        assert isinstance(port, V_Port), f"{port}"

        if isinstance(var, int):
            var = V_Int(var, port.width)

        self.var = var
        self.port = port

    def __str__(self) -> V_Line:

        return f".{self.port.name}({self.var})"

    def __iter__(self) -> V_Variable or V_Port:
        for obj in [self.var, self.port]:
            yield obj


def V_Always(
    edge: V_PosEdge or V_NegEdge,
    signal: V_Expression or V_Port or V_Variable
):

    def build(*lines: V_Block) -> V_Block:
        return V_Block(
            f"always @ ({edge()} {signal.name}) begin",
            *[f"\t{line}" for line in lines],
            "end"
        )

    return build


def V_Cases(
    expression: V_Expression or V_Port or V_Variable
):

    assert isinstance(expression, (V_Expression, V_Port, V_Variable))

    def build(
        *cases: Tuple[V_Line, V_Block, Optional[V_Comment]],
        default: Optional[V_Block] = []
    ) -> V_Block:
        return V_Block(
            f"case ({expression})",
            *[line for item, lines, *comment in cases for line in [
                f"\t// {comment[0]}" if comment else "",
                f"\t{item}: begin",
                *[
                    f"\t\t\t{line}" for line in lines
                ],
                "\t\tend"
            ]],

            "endcase"
        )

    return build


def V_If(
    predicate: V_Expression or V_ObjectBase
):

    assert isinstance(predicate, (V_Expression, V_ObjectBase)
                      ), f'"{predicate}" is not a valid predicate.'

    def build(*lines: Iterable[V_Line]) -> V_Block:
        return V_Block(
            f"if ({predicate}) begin",
            *[f"\t{line}" for line in lines],
            "end"
        )

    return build


def V_Else(
    *lines: Iterable[V_Line]
) -> V_Block:

    return V_Block(
        f"else begin",
        *[f"\t{line}" for line in lines],
        "end"
    )


def V_Par(
    obj: V_ObjectBase
) -> V_ObjectBase:

    assert isinstance(obj, V_ObjectBase)

    return V_ObjectBase.from_obj(obj, name=f"({obj})")


def V_Chain_Op(
    operator: str,
    *items: Iterable[V_Expression or V_ObjectBase]
) -> V_Expression:
    assert isinstance(operator, str)

    return V_Expression(
        f" {operator} \n\t\t".join(map(str, items))
    )


def V_Sum(*items: Iterable[V_Expression or V_ObjectBase]) -> V_Expression:

    return V_Chain_Op("+", *items)


def V_Ternary(predicate: V_Expression or V_ObjectBase):
    assert isinstance(predicate, (V_Expression, V_ObjectBase)), f"{predicate}"

    def build(
        expr_true: V_Expression or V_ObjectBase,
        expr_false: V_Expression or V_ObjectBase
    ) -> V_Expression:
        assert isinstance(
            expr_true, (V_Expression, V_ObjectBase)), f"{expr_true}"
        assert isinstance(
            expr_false, (V_Expression, V_ObjectBase)), f"{expr_false}"

        return V_Expression(
            f"({predicate}) ? \n\t\t{expr_true} : \n\t\t{expr_false}"
        )

    return build
