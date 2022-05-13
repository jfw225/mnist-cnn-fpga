import os

""" Enumeration of the paths of each verilog file. """
V_BASE_PATH = os.path.join("src", "verilog")

""" Enumeration of the work folder. """
V_WORK_FOLDER = "work"

""" Enumeration of the data folder. """
V_DATA_FOLDER = "data"

""" Enumeration of the system verilog extension. """
V_SV_EXT = ".sv"

""" Enumeration of the do file extension. """
V_DO_EXT = ".do"

""" Enumeration of the wlf file extension. """
V_WLF_EXT = ".wlf"

""" Enumeration of the mif file extension. """
V_MIF_EXT = ".mif"

""" Enumeration of the verilog compiler error prefix. """
V_MSIM_ERROR_PREFIX = "**"

""" Enumeration of the verilog compiler warning infix. """
V_MSIM_WARNING_INFIX = "Warning"

""" Enumeration of the default simulation run duration. """
V_VSIM_DEFAULT_RUN_DURATION = 1000

""" Enumeration of the default timeout for a simulation . """
V_VSIM_DEFAULT_TIMEOUT = 5


class V_MSIM_COMPILE:
    """ Enumeration of the modelsim compile command and command options. """

    CMD = "vlog"

    WORK = "-work"


class V_MSIM_VSIM:
    """ Enumeration of the modelsim simulator command and command options. """

    CMD = "vsim"

    def DO(do_path): return ["-do", do_path]

    HEADLESS = "-c"

    GUI = "-gui"

    def WLF(wlf_path): return ["-wlf", f"{wlf_path}"]

    def WORK(module): return f"work.{module}"


class V_MSIM_DUMP:
    """ Enumeration of the modelsim log dumper command and command options. """

    CMD = "dumplog64"


STR_CHECK_MARK = u"\u2713"

STR_CROSS_MARK = u"\u2717"
