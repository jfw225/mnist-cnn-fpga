from typing import Dict, Iterable, Optional, Tuple
from verilog.core.viterable import V_Iterable
from verilog.core.vmodule import V_Module
from verilog.core.vspecial import V_Clock, V_Done, V_High, V_Low, V_Reset
from verilog.core.vstate import V_StDone, V_State, V_StateMachine
from verilog.core.vsyntax import V_Else, V_If, V_Port
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import V_DType, V_Input, V_Output, V_Reg, V_Wire


class V_Map(V_Target):
    """
    Maps the data stored in the `V_Iterable` objects `inputs` to the
    `V_Iterable` objects `outputs` using a target module `target`.

    Each input port of the target is matched in order with the
    read lines of each input in `inputs`.

    Similarly, each output port of the target is matched in order with
    the write lines of each output in `outputs`.

    We start by retreiving the data at the inputs' base address and storing it
    locally--let's call it `x`. We then give `x` to `target` and wait for
    `target` to indicate that it has finished. Afterward, `x` is saved to
    the output and the next element `x+1` is retreived. After the entire
    input has been exhausted, this module indicates that it has finished.
    """

    def __init__(
        self,
        inputs: Iterable[V_Iterable],
        outputs: Iterable[V_Iterable],
        target: V_Target,
        state_machine: Optional[V_StateMachine] = None,
        **kwargs
    ):

        for _input in inputs:
            assert isinstance(_input, V_Iterable)

        for output in outputs:
            assert isinstance(output, V_Iterable)

        assert isinstance(target, V_Target)

        assert len(inputs) == len(
            target.input_ports), "there must be the same number of input iterables as target input ports."
        assert len(outputs) == len(
            target.output_ports), "there must be the same number of output iterables as target output ports."

        if state_machine is not None:
            assert isinstance(state_machine, V_StateMachine)

        super().__init__(**kwargs)

        # save arguments
        self.inputs = inputs
        self.outputs = outputs
        self.target = target

        # create the verilog state machine
        self.vsm = state_machine or V_StateMachine(_StReset, _StWait, _StInc)

        # create an obj to get local copies of inputs/outputs
        self.local_io = {}

        # create a local address for iteration
        input_addr, *_ = inputs[0].read
        self.local_addr = self.add_var(
            input_addr, dtype=V_Reg, name="local_addr")

        # create variables to connect to the target's reset, valid, done flags
        self.target_reset = self.add_var(V_Reset("target_reset"), dtype=V_Reg)
        self.target_valid = self.var(V_Reg, 1, name="target_valid")
        self.target_done = self.add_var(V_Done("target_done"), dtype=V_Wire)

        self.target_connections = [
            (self.clk, self.target.clk),
            (self.target_reset, self.target.reset),
            (self.target_done, self.target.done)
        ]

        # the local ports for each `input.read` and `output.write`
        self.reads: Dict[int, Tuple[V_Port, V_Port]] = {}
        self.writes: Dict[int, Tuple[V_Port, V_Port, V_Port]] = {}

        # configure the ports
        self.configure_ports()

    def generate(self):
        # get rid of self
        target = self.target

        output_addr, output_data, output_write_en = self.outputs[0].write

        # target module instantiation
        target_ins = target(
            self,
            *self.target_connections,
        )

        return [
            "\n\n// connect each of the addresses",
            *[addr.set(self.local_addr) for addr, *_ in self.reads.values()],
            *[addr.set(self.local_addr) for addr, *_ in self.writes.values()],
            "\n\n",

            "\n// the state machine",
            *self.vsm.generate(self, self.clk, self.reset, self.done),

            "\n\n// instantiate the target module",
            *target_ins
        ]

    def instantiate(
        self,
        instantiator: V_Module,
        clk: V_Clock,
        reset: V_Reset,
        done: V_Done
    ):
        """
        Handles the instantiation of this module. Requires the caller to create
        a `done` flag which is raised when the mapping is finished.
        """

        assert isinstance(instantiator, V_Module)
        assert V_Clock.isinstance(clk)
        assert V_Reset.isinstance(reset)
        assert V_Done.isinstance(done)

        port_connections = [
            (clk, self.clk),
            (reset, self.reset),
            (done, self.done),
            (1, self.valid)
        ]

        for i, _input in enumerate(self.inputs):
            read_addr, read_data = _input.read

            assert instantiator.name in _input.connections, f"{instantiator.name} has not instantiated {_input.name} yet."
            connections = _input.connections[instantiator.name]
            [addr_conn] = filter(lambda conn: conn.port is read_addr,
                                 connections)
            [data_conn] = filter(lambda conn: conn.port is read_data,
                                 connections)

            read_addr_i, read_data_i = self.reads[i]

            # save port connections
            port_connections += [(addr_conn.var, read_addr_i),
                                 (data_conn.var, read_data_i)]

        for i, output in enumerate(self.outputs):
            write_addr, write_data, write_en = output.write

            assert instantiator.name in output.connections, f"{instantiator.name} has not instantiated {output.name} yet."
            connections = output.connections[instantiator.name]
            [addr_conn] = filter(lambda conn: conn.port is write_addr,
                                 connections)
            [data_conn] = filter(lambda conn: conn.port is write_data,
                                 connections)
            [we_conn] = filter(lambda conn: conn.port is write_en,
                               connections)

            write_addr_i, write_data_i, write_en_i = self.writes[i]

            # save port connections
            port_connections += [(addr_conn.var, write_addr_i),
                                 (data_conn.var, write_data_i),
                                 (we_conn.var, write_en_i)]

        return super().instantiate(instantiator, *port_connections)

    def configure_ports(self):
        # configure the input ports
        for i, _input in enumerate(self.inputs):
            # get the ports from the input
            read_addr, read_data = _input.read

            # create local copies
            read_addr_i = self.add_port(read_addr,
                                        port_type=V_Output,
                                        dtype=V_Reg,
                                        name=f"read_addr_{i}")

            read_data_i = self.add_port(read_data,
                                        port_type=V_Input,
                                        dtype=V_DType,
                                        name=f"read_data_{i}")

            # save local read ports
            self.reads[i] = (read_addr_i, read_data_i)
            self.local_io[_input] = (read_addr_i, read_data_i)

            # save target connections
            self.target_connections.append(
                (read_data_i, self.target.input_ports[i])
            )

        # configure the output ports
        for i, output in enumerate(self.outputs):
            # get the ports from the output
            write_addr, write_data, write_en = output.write

            # create local copies
            write_addr_i = self.add_port(write_addr,
                                         port_type=V_Output,
                                         dtype=V_Reg,
                                         name=f"write_addr_{i}")

            write_data_i = self.add_port(write_data,
                                         port_type=V_Output,
                                         dtype=V_Reg,
                                         name=f"write_data_{i}")

            write_en_i = self.add_port(write_en,
                                       port_type=V_Output,
                                       dtype=V_Reg,
                                       name=f"write_en_{i}")

            # save local write ports
            self.writes[i] = (write_addr_i, write_data_i, write_en_i)
            self.local_io[output] = (write_addr_i, write_data_i, write_en_i)

            # save target connections
            self.target_connections.append(
                (write_data_i, self.target.output_ports[i])
            )


"""
V_Map State Machine:
"""


class _StReset(V_State):
    """
    1. StReset: 
        - reset the target
        - initialize address
        - lower each of the output write enables
        - go to StWait
    """

    def generate(self, module: V_Map):
        base_addr = module.inputs[0].base_addr

        return [
            module.target_reset.set(V_High),
            module.local_addr.set(base_addr),
            "\n",
            "// clear each write enable",
            *[write_en.set(V_Low)
              for *_, write_en in module.writes.values()],

            _StWait
        ]


class _StWait(V_State):
    """
    2. StWait:
        - lower target reset
        - if (target done)
            - raise each of the output write enables
            - go to StInc
    """

    def generate(self, module: V_Map):

        return [
            "// clear the target reset",
            module.target_reset.set(V_Low),
            "\n",
            "// wait for target done flag",
            *V_If(module.target_done)(
                "// set each write enable",
                *[write_en.set(V_High)
                  for *_, write_en in module.writes.values()],
                _StInc
            )
        ]


class _StInc(V_State):
    """
    3. StInc:
        - lower each of the output write enables
        - reset the target
        if (address == input.size - 1)
            - go to StDone
        else
            - increment the address 
            - go to StWait
    """

    def generate(self, module: V_Map):
        max_addr = module.inputs[0].size - 1

        return [
            "// reset the target",
            module.target_reset.set(V_High),
            "\n",

            "// clear each write enable",
            *[write_en.set(V_Low)
              for *_, write_en in module.writes.values()],
            "\n",

            "// if all inputs have been exhausted",
            *V_If(module.local_addr == max_addr)(
                V_StDone
            ), *V_Else(
                module.local_addr.set(module.local_addr + 1),
                _StWait
            )
        ]
