
from os import PathLike
import os
from typing import Iterable, Optional
from config import V_BASE_PATH
from verilog.config import V_SV_EXT
from verilog.core.vsyntax import V_Array, V_Variable

from verilog.core.vtypes import V_File
from verilog.core.vmodule import V_Module


class V_FileWriter:
    def __init__(
        self,
        name: V_File,
        objects: Optional[Iterable[V_Variable or V_Array or V_Module]] = list(),
        includes: Optional[Iterable[V_File] or "V_FileWriter"] = list(),
        dir: Optional[PathLike] = None
    ):

        self.name = name
        self.dir = dir or V_BASE_PATH
        self.objects = [obj for object in objects for obj in object.objects]
        for v_file in includes:
            assert isinstance(
                v_file, V_FileWriter), f"{v_file} must be an object of type {V_FileWriter}"
        self.includes = includes

    def add_objects(
        self,
        *objects: Iterable[V_Variable or V_Array or V_Module]
    ):

        return self.objects.extend([
            obj for object in objects for obj in object.objects])

    def get_path(self) -> PathLike:

        return os.path.join(self.dir, self.name + V_SV_EXT)

    def include(self, v_file: "V_FileWriter") -> None:
        """ Imports modules from verilog file `v_file`. """

        assert isinstance(
            v_file, V_FileWriter), f"{v_file} must be an object of type {V_FileWriter}"

        self.includes.append(v_file)

    def write(self):
        # add this line to make sure quartus can synthesize SV
        quartus_comment = "// synthesis VERILOG_INPUT_VERSION SYSTEMVERILOG_2005\n"

        # format the header
        header = f"__{self.name.upper()}_SV__"

        # filter the objects to remove duplicates
        object_map = {}

        def remove_duplicates(object):
            if object_map.get(object.name):
                return False

            object_map[object.name] = 1

            return True
        objects = [obj for obj in self.objects if remove_duplicates(obj)]

        # separate the variables and objects
        vars = [
            obj for obj in objects if isinstance(obj, (V_Variable, V_Array))
        ]
        modules = [obj for obj in objects if isinstance(obj, V_Module)]

        # format all of the objects
        objects = [
            var.define() + "\n\n" for var in vars
        ] + [
            line
            for module in modules
            for line in [*module._generate(), "\n\n"]
        ]

        # get a unique set of includes
        includes = set([
            lib for module in modules for lib in module.includes
            if isinstance(module, V_Module)
        ] + self.includes)

        # write each library
        for lib in includes:
            assert isinstance(
                lib, V_FileWriter), f"{lib} must be an object of type {V_FileWriter}"
            lib.write()

        # generate each line
        lines = [
            quartus_comment,

            # the file header
            f"`ifndef {header}\n`define {header}\n\n",

            # the includes
            "\n".join(
                [f'`include "./{lib.name}{V_SV_EXT}"' for lib in includes]) + "\n\n",

            # the objects
            *objects,

            # the file footer
            "`endif"
        ]

        # create directory if it doesn't exist
        os.makedirs(V_BASE_PATH, exist_ok=True)

        # write the file
        with open(self.get_path(), "w") as f:
            f.writelines(lines)
