
from math import ceil, log2
from typing import Dict, Iterable, Optional, Type
from verilog.core.vspecial import V_Clock, V_Done, V_High, V_Low, V_Reset
from verilog.utils import format_int, nameof
from verilog.core.vmodule import V_Module
from verilog.core.vsyntax import V_Always, V_Cases, V_Else, V_If, V_Variable

from verilog.core.vtypes import BitWidth, NetName, V_Block, V_Expression, V_Line, V_NegEdge, V_PosEdge, V_Reg


class V_StateID(int):
    """
    The type representing the ID of a `V_State` object.
    """


class _V_State_Meta(type):
    def __str__(cls) -> str:
        name, *_ = nameof(object.__str__(cls)).split("'")

        return name


class V_State(metaclass=_V_State_Meta):
    """
    The object representing a state in a verilog state machine.
    """

    def __init__(self, state_id: V_StateID, width: BitWidth) -> None:
        self.state_id = state_id
        self.width = width

    def __call__(self, module: V_Module):
        return self.generate(module)

    def __iter__(self):
        """
        Simply returns `self.state_id` formatted as value of size `self.width`.
        Overloaded for convenience.
        """

        yield format_int(self.state_id, self.width)

    def generate(self, m: V_Module) -> V_Block:
        """
        This function comprises the state logic: both what is done in the state
        and how the state transitions. The return value should be some verilog
        code and at least one state. The state machine will iterate through
        the returned list of code and replace each `Type[V_State]` with an
        assignment to change the state value. This should be overloaded.

        If no state is found, the next state will be `V_StDone` and the
        module's `done` flag will be raised.

        The parameter `module` is the `V_Module` object in which this state
        will be used.
        """

        return [
            *V_If(V_Expression(format_int(V_High, 1)))(
                V_State
            )
        ]


class V_StDone(V_State):
    """
    The state indicating that the state machine is finished. 
    """

    def __init__(self, done: V_Done, *args, **kwargs) -> None:
        assert V_Done.isinstance(done)

        super().__init__(*args, **kwargs)

        self.done = done

    def generate(self, m: V_Module) -> V_Block:

        return V_Block(
            "// idle until reset",
            self.done.set(V_High),
            V_StDone
        )


class V_StateMachine:
    """ 
    Verilog implementation of a finite state machine.
     """

    def __init__(
        self,
        reset_state: Type[V_State],
        *states: Iterable[Type[V_State]]
    ) -> None:

        # determine bit width needed for a state variable
        # (+ 1 is for `V_StDone`)
        self.width = BitWidth(ceil(log2(len(states) + 1)))

        # set the reset state
        self._reset_state = reset_state(
            state_id=V_StateID(0), width=self.width)

        # maps class types to initialized objects
        self._state_map: Dict[NetName, V_State] = {
            str(state): state(state_id=V_StateID(i), width=self.width)
            for i, state in enumerate(states)
        }

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(
        self,
        module: V_Module,
        clk: V_Clock,
        reset: V_Reset,
        done: V_Done,
        edge: Optional[V_PosEdge or V_NegEdge] = V_PosEdge
    ) -> V_Block:
        """
        This function generates a verilog state machine from `self._state_map` 
        that operates on the edge `edge of clock `clk`.

        Each state has a `generate` function that should return an iterable 
        containing lines of verilog and at least `V_State` object. If no 
        `V_State` object is found, an error will be thrown.  

        This function will create a state variable `state` in the verilog 
        module `module`, and will assign each of the states a value. We will 
        initialize `state` to `self._start_state` when `reset` is HIGH. 

        Let `V_i` denote the value assigned to the `i`-th state `S_i`. Then this 
        function will search through the iterables returned from each of the 
        `V_State.generate` calls and replace each `S_i` with `state <= V_i`.

        The state machine will continue to run until `V_StDone` is 
        reached--at which point, the state machine will raise `done` and idle 
        until `reset` is set.
        """

        assert isinstance(module, V_Module)
        assert V_Done.isinstance(done)

        # create local copies
        reset_state, state_map = self._reset_state, self._state_map

        # generate a state variable
        state_var = module.var(V_Reg, self.width, name="state")

        # create the state formatter
        format_st = self.create_state_formatter(module, state_var)

        # check if `V_StDone` is needed
        if any(
            [line
             for st in [reset_state, *state_map.values()]
             for line in st.generate(module)
             if str(V_StDone) in line]
        ):
            # add `V_StDone` to the state map
            state_map[nameof(V_StDone)] = V_StDone(done,
                                                   state_id=len(state_map),
                                                   width=self.width)

        # make sure that `done` is not assigned anywhere in the reset state
        if any([1] for line in format_st(reset_state) if V_Done.BASE_NAME in line):
            raise Exception(
                f'"{V_Done.BASE_NAME}" can not be modified in the reset state--this is already done for you.')

        return V_Block(
            *V_Always(edge, clk)(

                *V_If(reset)(
                    "// clear done flag and go to starting state",
                    done.set(V_Low),
                    *format_st(reset_state)
                ), *V_Else(
                    *V_Cases(state_var)(
                        *[(*st, format_st(st), nameof(st)) for st in state_map.values()]
                    )
                )
            )
        )

    def create_state_formatter(
        self,
        module: V_Module,
        state_var: V_Variable
    ) -> Iterable[V_Line]:
        """
        Returns a function that ingests some `V_State` object `state` and 
        replaces each `Type[V_State]` that is found in `state.generate(module)` 
        with a statement that sets the value of `state_var` to that of `state`.
        """

        def format_st(state: V_State):
            new_lines = []
            state_found = False

            for line in state.generate(module):
                num_tabs = 0

                # if `line` is a string
                if isinstance(line, str):
                    # strip away tabs (should only have tabs)
                    stripped = line.lstrip("\t")
                    num_tabs = len(line) - len(stripped)
                    state_instance = self._state_map.get(stripped)

                # if `line` is a `V_State` object
                elif isinstance(line, type(V_State)):
                    state_instance = self._state_map.get(nameof(line))

                else:
                    raise Exception(f"{line} is not a valid line.")

                if state_instance is not None:
                    state_found = True
                    line = "\t" * num_tabs + state_var.set(*state_instance)

                new_lines.append(line)

            if not state_found:
                raise Exception(
                    f"No transition state found for state: {state}")
                # new_lines.append(state_var.set(*self._state_map[V_StDone]))

            return new_lines

        return format_st


if __name__ == '__main__':
    pass
