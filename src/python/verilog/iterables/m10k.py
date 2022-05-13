from math import ceil, log2
from typing import Optional
from verilog.iterables.mifwriter import MIF_Writer
from verilog.utils import id_generator, nameof
from verilog.core.viterable import V_Iterable
from verilog.core.vmodule import V_Module
from verilog.core.vsyntax import V_Always, V_Array, V_Else, V_If, V_Int, V_Variable
from verilog.core.vtypes import ArraySize, BitWidth, V_Block, V_File, V_Input, V_Line, V_Output, V_ParameterArray, V_PosEdge, V_Reg, V_RegArray


class M10K(V_Iterable):

    def __init__(
        self,
        width: BitWidth,
        size: ArraySize,
        **kwargs,
    ):
        super().__init__(width, size, **kwargs)

        self.memory = V_Array(self, V_RegArray, self.width,
                              self.size, name="memory")

        self.syn_style = '/* synthesis ramstyle = "no_rw_check, M10K" */'

    def set_init_data(
        self,
        init_data: V_Array,
        file: Optional[V_File] = None,
        *,
        save_mif: Optional[bool] = False
    ):

        # if save_mif:
        #     mif_writer = MIF_Writer(init_data.name, init_data.data)
        #     mif_writer.write()

        return super().set_init_data(init_data, file)

    def generate(self):

        mem_fmt_base, *_ = self.memory.define().split(";")

        initialize_data = [""] if self.init_data is None else [
            self.memory.set(i, self.init_data.get(i)) for i in range(self.size)
        ]

        if self.init_data is not None:
            return V_Block(
                "// set synthesis style",
                f'{mem_fmt_base} {self.syn_style} ;'
                "\n",
                *V_Always(V_PosEdge, self.clk)(
                    *V_If(~self.reset)(
                        *V_If(self.write_en)(
                            self.memory.set(self.write_addr, self.write_data)
                        ),
                        # *V_If(self.read_addr <= V_Int(self.size - 1, self.addr_width))(
                        self.read_data.set(self.memory.get(self.read_addr))
                        # )
                    ),
                    *V_Else(*initialize_data)
                ),
            )
        else:
            return V_Block(
                "// set synthesis style",
                f'{mem_fmt_base} {self.syn_style} ;'
                "\n",
                *V_Always(V_PosEdge, self.clk)(
                    *V_If(self.write_en)(
                        self.memory.set(self.write_addr, self.write_data)
                    ),
                    self.read_data.set(self.memory.get(self.read_addr))
                )
            )
