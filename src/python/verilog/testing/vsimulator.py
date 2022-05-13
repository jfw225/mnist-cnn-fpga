"""
TODO:
compiler
simulator
parser
"""

import os
from typing import Iterable, Optional
from verilog.config import V_BASE_PATH, V_MSIM_DUMP, V_MSIM_ERROR_PREFIX, V_DATA_FOLDER, V_DO_EXT, V_MSIM_COMPILE, V_MSIM_VSIM, V_MSIM_WARNING_INFIX, V_VSIM_DEFAULT_RUN_DURATION, V_VSIM_DEFAULT_TIMEOUT, V_WLF_EXT, V_WORK_FOLDER
from verilog.core.vinstance import V_Instance, V_Signal
from verilog.core.vsyntax import V_FixedPoint, V_Int, V_ObjectBase
from verilog.testing.expectation import ExpectationData
from verilog.testing.vwavedata import V_WaveData

from verilog.utils import id_generator, run_cmd


class V_Simulator:
    def __init__(self, testbench):
        from verilog.testing.vtestbench import V_Testbench
        assert isinstance(testbench, V_Testbench)

        # store the test bench
        self._tb = testbench

        # the signals to be logged during simulation
        self._signals = [self._tb.clk, self._tb.reset]

        # store the path to the .do file generated for the simulation
        self._do_path = os.path.join(
            V_BASE_PATH, self._tb.name + V_DO_EXT).replace('\\', '/')

        # TODO: GENERATE RANDOM ID FOR WLF PATH
        # store the path to do .wlf file generated during the simulation
        os.makedirs(os.path.join(V_BASE_PATH, V_DATA_FOLDER), exist_ok=True)
        self._wlf_path = os.path.join(
            V_BASE_PATH, V_DATA_FOLDER, id_generator() + V_WLF_EXT)

        # create the wave data object
        self._wd = V_WaveData(self._tb)

    def compile(self):
        """
        Writes and compiles the code needed in test bench `self._tb`.
        """

        # get the writer and write the files
        file = self._tb.tofile(self._tb.name)
        file.write()

        # get the path to the test bench
        path = file.get_path()

        # compile the test bench
        output = run_cmd(V_MSIM_COMPILE.CMD,
                         V_MSIM_COMPILE.WORK,
                         V_WORK_FOLDER,
                         path)

        # check if there were any errors during compilation
        compile_error = False
        for line in output:
            if V_MSIM_ERROR_PREFIX in line:
                if V_MSIM_WARNING_INFIX not in line:
                    compile_error = True
                print(line)

        if compile_error:
            raise Exception(f"Error compiling test bench located at: {path}")

        return output

    def expect(
        self,
        signal: V_ObjectBase,
        data: ExpectationData
    ) -> None:
        """
        Creates an expectation and adds it to the wave data expectations.
        """

        assert isinstance(signal, V_ObjectBase), signal

        # if `signal` is not a `V_Signal` object
        if (not isinstance(signal, V_Signal) and
                not any([signal is obj for obj in self._tb.ports + self._tb.variables])):
            # attempt to find it's signal copy
            signal = self._tb.signal_of_obj(signal)

        if not any([signal for other in self._signals if signal is other]):
            self.log(signal)

        if isinstance(data, int):
            data = V_Int(data, signal.width)

        return self._wd.expect(signal, data)

    def generate_do(self, headless: Optional[bool] = True):
        """
        Generates the do command used during the simulation of `self._tb`.
        """

        # get the full instantiation path of each signal
        signal_paths = [self._tb.ins_path(signal) for signal in self._signals]

        commands = [
            "radix -unsigned;\n",
            *[f"add wave {path};\n" for path in signal_paths],
            # *[f"add log {path};\n" for path in signal_paths],
            # f"run {V_VSIM_DEFAULT_RUN_DURATION};\n"
            f"run -all;\n"
        ]

        if headless:
            commands.append(f"quit -f;\n")

        return "".join(commands)

    def log(self, signal: V_Signal):
        """
        Adds `V_Signal` object `signal` to the list of signals that will be
        logged during the simulation.
        """

        assert isinstance(signal, V_ObjectBase), signal

        # if `signal` is not a `V_Signal` object
        if (not isinstance(signal, V_Signal) and
                not any([signal is obj for obj in self._tb.ports + self._tb.variables])):
            # attempt to find it's signal copy
            signal = self._tb.signal_of_obj(signal)

        self._signals.append(signal)

    def parse(self):
        """
        Parses the waveform data from the simulation.
        """

        output = run_cmd(V_MSIM_DUMP.CMD, self._wlf_path)

        print(f"Starting to parse: {self._wlf_path}")
        # parse the output
        self._wd.parse(self._signals, output)

        # match signals with data
        self._wd.match_signals(*self._signals)

    def run(
        self,
        headless: Optional[bool] = True,
        presim: Optional["function"] = None,
        postsim: Optional["function"] = None
    ):
        """
        Runs the simulation of `self._tb`.
        """

        assert len(
            self._signals) > 0, f"Testbench \"{self._tb}\" must log at least one signal."

        # write and compile the test bench
        self.compile()

        # run the presim function
        if presim:
            presim()

        print("Starting simulation...")

        # run the simulation
        self.simulate(headless)

        if headless:
            print("Finished simulation!")

        # parse the output
        self.parse()

        # evaluate the expectations
        self._wd.eval_expectations()

        # run the postsim function
        if postsim:
            postsim(self._wd)

    def simulate(self, headless: Optional[bool] = True):
        """
        Starts the simulation and checks the output.
        """

        # set the mode
        mode = V_MSIM_VSIM.HEADLESS if headless else V_MSIM_VSIM.GUI

        # generate the do commands
        do_commands = self.generate_do(headless)

        # run the command
        output = run_cmd(V_MSIM_VSIM.CMD,
                         *V_MSIM_VSIM.WLF(self._wlf_path),
                         mode,
                         V_MSIM_VSIM.WORK(self._tb.name),
                         *V_MSIM_VSIM.DO(do_commands))

        # check if there were any errors during simulation
        simulation_error = False
        for line in output:
            if V_MSIM_ERROR_PREFIX in line and "$stop" not in line:
                if "Warning" not in line:
                    simulation_error = True
                print(line)

        if simulation_error:
            raise Exception(
                f"Error simulating test bench: {self._tb}\nDo Commands:\n{do_commands}")
