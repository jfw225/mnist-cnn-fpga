import os
from typing import Iterable
from verilog.config import V_BASE_PATH, V_MIF_EXT
from verilog.core.vsyntax import V_Array, V_FixedPoint, V_Int
from verilog.core.vtypes import V_File


class MIF_Writer:

    def __init__(
        self,
        file_name: V_File,
        objs: Iterable[V_Int or V_FixedPoint]
    ) -> None:

        assert len(objs) > 0, objs
        assert isinstance(objs[0], (V_Int, V_FixedPoint)), objs

        if isinstance(objs[0], V_Int):
            raise NotImplementedError()

        self.file_name = file_name
        self.path = os.path.join(V_BASE_PATH, file_name + V_MIF_EXT)
        self.objs = objs

    def write(self):
        depth = len(self.objs)
        width = self.objs[0].width

        lines = [
            f"DEPTH = {depth};\n",
            f"WIDTH = {width};\n\n",
            "ADDRESS_RADIX = DEC;\n",
            "DATA_RADIX = BIN;\n\n",
            "CONTENT\n",
            "BEGIN\n",
            f"[0..{depth}\t:\t]" + "0" * width + ";\n",
            *[
                f"{i}\t:\t" +
                "".join([o for o in f"{obj}"
                         if o is "0" or o is "1"]) + ";\n"
                for i, obj in enumerate(self.objs)],
            "END ;"
        ]

        with open(self.path, "w") as f:
            f.writelines(lines)
