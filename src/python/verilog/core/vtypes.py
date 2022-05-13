from typing import List, Type


class NetName(str):
    """
    The type representing the name of a verilog variable.
    """


class ArraySize(int):
    """
    The type representing the size of an array. 
    """


class BitWidth(int):
    """
    The type representing the bit width of a value.
    """


class V_File(str):
    """
    The type representing the path to a verilog file. 
    """


class V_Comment(str):
    """
    The type representing a verilog comment. 
    """


class V_Expression(str):
    """
    The type representing a verilog expression.
    """


class V_Line(str):
    """
    The type representing a line of verilog code tabbed to the relative level.
    """

    def __new__(cls, val: str):
        # val_stripped = str(val).lstrip("\t")
        # num_t = len(str(val)) - len(val_stripped)
        # val = "\t" * num_t + val_stripped.replace("\n", "\n" + "\t" * num_t)

        return super().__new__(cls, val)


class V_Block(List[str]):
    """
    The type representing a block of verilog code tabbed to the relative level.
    """

    def __init__(self, *lines: List[str]):
        super().__init__([V_Line(line) for line in lines])


class V_PortType(str):
    """
    The type representing a verilog module port.
    """


class V_Input(V_PortType):
    """
    The type representing a verilog module input port. 
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "input")


class V_Output(V_PortType):
    """
    The type representing a verilog module output port.
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "output")


class V_DType(str):
    """
    The type representing a verilog data type.
    """

    def __new__(cls, dtype="") -> None:
        return super().__new__(cls, dtype)


class V_Reg(V_DType):
    """
    The type representing a verilog register.
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "reg")


class V_Wire(V_DType):
    """
    The type representing a verilog wire. 
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "wire")


class V_Parameter(V_DType):
    """
    The type representing a verilog parameter.
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "parameter")


class V_RegArray(V_DType):
    """
    The type representing a verilog register array.
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "reg")


class V_WireArray(V_DType):
    """
    The type representing a verilog wire array. 
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "wire")


class V_ParameterArray(V_DType):
    """
    The type representing a verilog parameter array.
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "parameter")


class V_PosEdge(str):
    """
    The type representing the positive edge of a signal. 
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "posedge")


class V_NegEdge(str):
    """
    The type representing the negative edge of a signal. 
    """

    def __new__(cls) -> None:
        return super().__new__(cls, "negedge")
