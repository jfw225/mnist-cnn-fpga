from typing import Dict, Iterable, List, Tuple, Type

import numpy as np
from verilog.ml.layer import Layer, LayerSpec
from verilog.core.vmodule import V_ConnSpec, V_Module
from verilog.core.vspecial import V_Clock, V_Done, V_Empty, V_High, V_Low, V_Ready, V_Reset, V_Valid
from verilog.core.vstate import V_StDone, V_State, V_StateMachine
from verilog.core.vsyntax import V_If
from verilog.core.vtarget import V_Target
from verilog.core.vtypes import BitWidth, V_Block, V_DType, V_Input, V_Output, V_Reg
from verilog.iterables.m10k import M10K
V_ConnSpec


class Model(V_Target):
    def __init__(
        self,
        int_width: BitWidth,
        dec_width: BitWidth,
        input_mem: M10K,
        output_mem: M10K,
        *specs: Iterable[LayerSpec]
    ):

        self.int_width = int_width
        self.dec_width = dec_width
        self.width = int_width + dec_width

        assert self.width == input_mem.width and self.width == output_mem.width
        assert len(specs) > 0

        self.input_mem = input_mem
        self.output_mem = output_mem

        super().__init__()

        self._configure_ports()

        # ordered list of layers
        self.layers: List[Layer] = []

        # {Layer: (layer_cs, input_mem, output_mem)}
        self.layer_map: Dict[Layer, Tuple[V_ConnSpec[Layer], M10K, M10K]] = {}

        self.memories: Dict[M10K, V_ConnSpec[M10K]] = {
            self.input_mem: V_ConnSpec[M10K](self,
                                             self.input_mem,
                                             prefix="input_mem",
                                             clk=self.clk,
                                             reset=self.reset,
                                             write_en=V_Empty(),
                                             read_addr=self.inp_addr,
                                             write_addr=V_Empty(),
                                             read_data=self.inp_data,
                                             write_data=V_Empty()
                                             ),
            self.output_mem: V_ConnSpec[M10K](self,
                                              self.output_mem,
                                              prefix="output_mem",
                                              clk=self.clk,
                                              reset=self.reset,
                                              write_en=self.out_we,
                                              read_addr=V_Empty(),
                                              write_addr=self.out_addr,
                                              read_data=V_Empty(),
                                              write_data=self.out_data
                                              )
        }

        self._set_up_layers(*specs)

        # set each layer reset flag to be a reg
        for layer_cs, *_ in self.layer_map.values():
            layer_cs.reset.dtype = V_Reg

        # create each of the layer states
        self.layer_states: List[Type[V_State]] = self._create_layer_states(0)

    def generate(self) -> V_Block:
        transition_memories = {mem: cs for mem, cs in self.memories.items()
                               if mem not in [self.input_mem, self.output_mem]}

        vsm = V_StateMachine(_StReset, _StWaitValid, *self.layer_states)

        self.layer_ins = [layer.instantiate(
            self, *cs) for layer, (cs, *_) in self.layer_map.items()]

        return V_Block(
            "\n// the model state machine",
            *vsm.generate(self, self.clk, self.reset, self.done),

            "\n// instantiate the transition memories",
            *[line for mem, cs in transition_memories.items()
              for line in mem(self, *cs)],

            "\n// instantiate the layers",
            *[line for layer_ins in self.layer_ins for line in layer_ins]
        )

    def instantiate(
        self,
        instantiator: V_Module,
        clk: V_Clock,
        reset: V_Reset,
        valid: V_Valid,
        done: V_Done,
        ready: V_Ready
    ):
        assert isinstance(instantiator, V_Module)
        assert V_Clock.isinstance(clk)
        assert V_Reset.isinstance(reset)
        assert V_Valid.isinstance(valid)
        assert V_Done.isinstance(done)
        assert V_Ready.isinstance(ready)

        port_connections = [
            (clk, self.clk),
            (reset, self.reset),
            (valid, self.valid),
            (done, self.done),
            (ready, self.ready)
        ]

        inp_addr, inp_data = self.input_mem.read

        assert instantiator.name in self.input_mem.connections, (
            f"{instantiator.name} has not instantiated {self.input_mem} yet.")

        connections = self.input_mem.connections[instantiator.name]
        [addr_conn] = filter(lambda conn: conn.port is inp_addr,
                             connections)
        [data_conn] = filter(lambda conn: conn.port is inp_data,
                             connections)

        # save port connections
        port_connections += [(addr_conn.var, self.inp_addr),
                             (data_conn.var, self.inp_data)]

        write_addr, write_data, write_en = self.output_mem.write

        assert instantiator.name in self.output_mem.connections, (
            f"{instantiator.name} has not instantiated {self.output_mem} yet.")

        connections = self.output_mem.connections[instantiator.name]
        [addr_conn] = filter(lambda conn: conn.port is write_addr,
                             connections)
        [data_conn] = filter(lambda conn: conn.port is write_data,
                             connections)
        [we_conn] = filter(lambda conn: conn.port is write_en,
                           connections)

        # save port connections
        port_connections += [(addr_conn.var, self.out_addr),
                             (data_conn.var, self.out_data),
                             (we_conn.var, self.out_we)]

        return super().instantiate(instantiator, *port_connections)

    def _configure_ports(self):
        """
        Creates all of the necessary ports. 
        """

        # configure the input ports
        inp_addr, inp_data = self.input_mem.read

        # create local copies
        self.inp_addr = self.add_port(inp_addr,
                                      port_type=V_Output,
                                      dtype=V_Reg,
                                      name=f"inp_addr")

        self.inp_data = self.add_port(inp_data,
                                      port_type=V_Input,
                                      dtype=V_DType,
                                      name=f"inp_data")

        # configure the output ports
        out_addr, out_data, out_we = self.output_mem.write

        # create local copies
        self.out_addr = self.add_port(out_addr,
                                      port_type=V_Output,
                                      dtype=V_Reg,
                                      name=f"out_addr")

        self.out_data = self.add_port(out_data,
                                      port_type=V_Output,
                                      dtype=V_Reg,
                                      name=f"out_data")

        self.out_we = self.add_port(out_we,
                                    port_type=V_Output,
                                    dtype=V_Reg,
                                    name=f"out_we")

    def _set_up_layers(self, *specs: Iterable[LayerSpec]) -> None:
        """
        Sets up each layer with the appropriate input/output objects. 
        """

        # set initial I/O objects
        input_mem, output_mem = self.input_mem, None

        # create a list to hold the objects that need to get created
        objects = list()

        for i, (LayerT, weights_np, biases_np, input_shape, output_shape) in enumerate(specs):
            print(f"Creating Layer: {LayerT}")

            input_size = np.prod(input_shape)
            output_size = np.prod(output_shape)

            assert input_mem.size == input_size, f"{input_mem} must have size of {input_size}"

            # get the input mem conn spec
            im_cs = self.memories[input_mem]

            # skip last layer
            if i + 1 == len(specs):
                break

            # create new output memory
            output_mem = M10K(self.width, output_size, name=f"trans_mem{i}")

            # add output memory to `objects` so it's generated
            objects.append(output_mem)

            # create layer object
            layer: Layer = LayerT(int_width=self.int_width,
                                  dec_width=self.dec_width,
                                  weights_np=weights_np,
                                  biases_np=biases_np,
                                  input_mem=input_mem,
                                  output_mem=output_mem,
                                  input_shape=input_shape,
                                  output_shape=output_shape)

            # create the output mem conn spec
            om_cs = V_ConnSpec[M10K](
                self,
                output_mem,
                prefix=output_mem.name,
                clk=self.clk,
                reset=self.reset
            )

            layer_cs = V_ConnSpec[Layer](self,
                                         layer,
                                         prefix=layer.name,
                                         clk=self.clk,
                                         inp_addr=im_cs.read_addr,
                                         inp_data=im_cs.read_data,
                                         out_addr=om_cs.write_addr,
                                         out_data=om_cs.write_data,
                                         out_we=om_cs.write_en)

            # adjust layer cs local variable dtypes
            layer_cs.valid.dtype = V_Reg

            # get layer file writer
            layer_file = layer.tofile(f"layer{i + 1}")

            # include layer
            self.include(layer_file)

            # store the output mem conn spec
            self.memories[output_mem] = om_cs

            # add layer instance to list
            self.layers.append(layer)

            # store an association between the layer and it's I/O memories
            self.layer_map[layer] = (layer_cs, input_mem, output_mem)

            # update input memory
            input_mem = output_mem

        ###############

        # set the objects field of `self`
        self._objects = objects

        # get the conn spec of the output memory
        om_cs = self.memories[self.output_mem]

        # create the last layer
        layer: Layer = LayerT(int_width=self.int_width,
                              dec_width=self.dec_width,
                              weights_np=weights_np,
                              biases_np=biases_np,
                              input_mem=input_mem,
                              output_mem=self.output_mem,
                              input_shape=input_shape,
                              output_shape=output_shape)

        # create the last conn spec
        layer_cs = V_ConnSpec[Layer](self,
                                     layer,
                                     prefix=layer.name,
                                     clk=self.clk,
                                     inp_addr=im_cs.read_addr,
                                     inp_data=im_cs.read_data,
                                     out_addr=om_cs.write_addr,
                                     out_data=om_cs.write_data,
                                     out_we=om_cs.write_en)

        # adjust layer cs local variable dtypes
        layer_cs.valid.dtype = V_Reg

        # add last layer instance to list
        self.layers.append(layer)

        # store last association between the layer and it's I/O memories
        self.layer_map[layer] = (layer_cs, input_mem, output_mem)

        # get layer file writer of last layer
        layer_file = layer.tofile(f"layer{i + 1}")

        # include last layer
        self.include(layer_file)

    def _create_layer_states(self, i) -> List[Type[V_State]]:

        def generate(self, m: Model) -> V_Block:
            """
            clear layer valid
            if (layer done)
                clear next layer reset
                set next layer valid

                go to next layer state or st done
            """

            layer = m.layers[i]
            layer_cs, *_ = m.layer_map[layer]

            if i == len(m.layers) - 1:
                block = V_Block(V_StDone)
            else:
                next_layer = m.layers[i + 1]
                next_layer_cs, *_ = m.layer_map[next_layer]

                block = V_Block(
                    next_layer_cs.reset.set(V_Low),
                    next_layer_cs.valid.set(V_High),
                    m.layer_states[i + 1]
                )

            return V_Block(
                layer_cs.valid.set(V_Low),
                *V_If(layer_cs.done)(
                    *block
                )
            )

        st = type(f"_StWaitLayer{i + 1}Done",
                  (V_State, ), {"generate": generate})

        if i == len(self.layers) - 1:
            return [st]

        return [st] + self._create_layer_states(i + 1)


"""
Model State Machine:
"""


class _StReset(V_State):
    """
    set each layer reset
    clear each layer valid

    go to StWaitValid
    """

    def generate(self, m: Model) -> V_Block:

        return V_Block(
            *[cs.reset.set(V_High) for cs, *_ in m.layer_map.values()],
            *[cs.valid.set(V_Low) for cs, *_ in m.layer_map.values()],

            _StWaitValid
        )


class _StWaitValid(V_State):
    """
    if (valid)
        clear layer1 reset
        set layer1 valid

        go StWaitLayer1Done
    """

    def generate(self, m: Model) -> V_Block:
        assert len(m.layer_states) > 0

        layer1 = m.layers[0]
        layer_cs, *_ = m.layer_map[layer1]

        return V_Block(
            *V_If(m.valid)(
                layer_cs.reset.set(V_Low),
                layer_cs.valid.set(V_High),

                m.layer_states[0]
            )
        )
