
import re
import numpy as np
import pandas as pd
from typing import Dict, Iterable
from verilog.config import STR_CHECK_MARK, STR_CROSS_MARK

from verilog.core.vinstance import V_Signal
from verilog.core.vsyntax import V_Array, V_FixedPoint, V_Int
from verilog.core.vtypes import BitWidth, V_ParameterArray, V_RegArray, V_WireArray
from verilog.testing.expectation import ExpectationData
from verilog.utils import fixedbin2dec, fixedfloat2dec, mean_squared_error


class V_WaveData:
    def __init__(self, testbench):
        # store the testbench
        self._tb = testbench

        # hold all of the expectations to be evaluated after the simulation
        self._expectations = list()

        # mapping of signals to wave form names
        self._signal_map: Dict[V_Signal, str] = dict()

        # the dataframe the will hold all of the wave data
        self._wave_data = pd.DataFrame()

    def dec_to_vobj(
        self,
        dec_val: str,
        cls_type: V_Int or V_FixedPoint,
        *widths: Iterable[BitWidth]
    ) -> V_Int or V_FixedPoint:

        if cls_type is V_Int:
            assert len(widths) == 1, widths

            return V_Int(dec_val, *widths)

        # otherwise, fixed point
        assert len(widths) == 2

        return V_FixedPoint(dec_val, *widths)

    def bin_to_vobj(
        self,
        bin_val: str,
        cls_type: V_Int or V_FixedPoint,
        *widths: Iterable[BitWidth]
    ) -> V_Int or V_FixedPoint:

        if "x" in bin_val:
            raise Exception(f"Value was not initialized: {bin_val}")

        if cls_type is V_Int:
            assert len(widths) == 1, widths

            return V_Int(int(bin_val, 2), *widths)

        # otherwise, fixed point
        assert len(widths) == 2

        return V_FixedPoint(fixedbin2dec(bin_val, *widths), *widths)

    def expect(
        self,
        signal: V_Signal,
        data: ExpectationData
    ) -> None:
        """
        Creates an expectation and adds it to `self._expectations`.
        """

        assert isinstance(signal, V_Signal) or any(
            [signal is obj for obj in self._tb.ports + self._tb.variables])

        if signal.dtype in [V_ParameterArray, V_RegArray, V_WireArray]:
            fn = self.eq_arr
        else:
            fn = self.eq_var

        self._expectations.append((fn, signal, data))

    def eq_arr(
        self,
        signal: V_Signal,
        data: Iterable[V_Int or V_FixedPoint]
    ) -> str:
        assert len(data) > 0, f"data must contain more than one element"
        d0 = data[0]
        assert isinstance(d0, (V_Int, V_FixedPoint)), data

        for di in data:
            assert isinstance(di, d0.__class__), di
            assert di.width == signal.width, di

        # retreive the last values
        last_vals = self[signal].iloc[-1]

        # convert the values to decimal
        widths = [d0.width] if d0.__class__ is V_Int else [
            d0.int_width, d0.dec_width]
        try:
            # assume the values are  binary
            temp = [int(bin_val, 2) for bin_val in last_vals]

            # if values are not binary, and error would be thrown
            last_vals = [self.bin_to_vobj(bin_val, d0.__class__, *widths)
                         for bin_val in last_vals]

        except ValueError:
            widths_ff2d = [d0.width, 0] if d0.__class__ is V_Int else [
                d0.int_width, d0.dec_width]

            last_vals = [self.dec_to_vobj(fixedfloat2dec(
                int(d_val), *widths_ff2d), d0.__class__, *widths) for d_val in last_vals]

        except Exception as e:
            print(f"Error: {e}\nLast Vals: {last_vals}")

        assert len(data) == len(
            last_vals), f"Length of data must be equal to the length of the `V_Array` for signal {signal}: {len(data)} != {len(last_vals)}"

        for di, vi in zip(data, last_vals):

            assert vi.width == signal.width, di

            if di != vi:
                break
        else:
            return f"{self.spath(signal)}: {STR_CHECK_MARK}"

        # for i, (a, b) in enumerate([
        #     (di.value, convert(vi).value) for di, vi in zip(data, last_vals)
        # ]):
        #     if b < 0:
        #         print(a, b, i, last_vals[i])
        mse = mean_squared_error(*[
            (di.value, vi.value) for di, vi in zip(data, last_vals)
        ])

        return f"{self.spath(signal)}: {STR_CROSS_MARK}\n\tMSE: {mse}"

    def eq_var(
        self,
        signal: V_Signal,
        data: V_Int or V_FixedPoint
    ) -> str:
        assert isinstance(data, (V_Int, V_FixedPoint)), data

        last_val = self[signal].iloc[-1]

        try:
            last_val = int(last_val, 2)
        except ValueError:
            if isinstance(data, V_Int):
                last_val = fixedfloat2dec(int(last_val), data.width, 0)
            else:
                last_val = fixedfloat2dec(
                    int(last_val), data.int_width, data.dec_width)

        def convert(v):
            widths = [data.width] if data.__class__ is V_Int else [
                data.int_width, data.dec_width]

            return self.dec_to_vobj(v, data.__class__, *widths)

        last_val = convert(last_val)

        if data != last_val:
            mse = mean_squared_error((data.value, last_val.value))

            return f"{self.spath(signal)}: {STR_CROSS_MARK}\n\tMSE: {mse}"

        return f"{self.spath(signal)}: {STR_CHECK_MARK}"

    def eval_expectations(self) -> bool:
        """
        Evaluates all the expectations in `self._expectations`.
        """

        for fn, signal, data in self._expectations:
            result_msg = fn(signal, data)
            print(result_msg, "\n")

    def match_signals(
        self,
        *signals: Iterable[V_Signal]
    ) -> None:
        """
        Matches the signals with each of the wave form names.
        """

        # get the signal paths
        signal_paths = [self._tb.ins_path(signal) for signal in signals]

        for path in signal_paths:
            expr = re.compile(f".*{path}$")
            matches = [
                col for col in self._wave_data.columns if expr.match(col)]
            if len(matches) != 1:
                raise Exception(
                    f"Error trying to find a match for {path} in {self._wave_data.columns}\n Matches found: {matches}")

            [wave_name] = matches
            self._signal_map[path] = wave_name

    def parse(
            self,
            signals: Iterable[V_Signal],
            lines: Iterable[str]
    ) -> None:
        """
        TODO:
        use clk as a baseline for time stamps
        create a table with structure:
        | time | *objects |

        match signals with objects
        """

        # find the wave name prefix
        prefix = None
        prefix_re = re.compile(".*\s+(\w+:).*")
        for line in lines:
            if prefix_re.match(line):
                _, prefix, _ = prefix_re.split(line)
                break

        assert prefix is not None, prefix

        info = dict()
        # initialize data and compile signal data regex
        for signal in signals:
            path = f"{prefix}/{self._tb.ins_path(signal)}"

            init_data = np.nan
            is_array = False
            data_re = (re.compile(f".*{path}\s+(-*\d+)"), None)
            size = None
            if signal.dtype in [V_ParameterArray, V_RegArray, V_WireArray]:
                init_data = [np.nan for _ in range(signal.size)]
                is_array = True
                data_re = (re.compile(f".*{path}\s+(.*)"),
                           re.compile(f".*{path}\[(\d+)\]\s+(-*\d+)"))
                size = signal.size
            info[path] = (
                {"time": [-1], path: [init_data]},
                is_array,
                data_re,
                size
            )

        data_line_re = re.compile("^(\d+)\s+((\w|:|/)+)\[?\d*\]?\s+.*")

        # match signals to wave names
        for line in lines:
            # if "Obj" not in line and "memory" in line:
            #     print(line, data_line_re.match(line))

            # get object data
            if data_line_re.match(line):
                # print(line, data_line_re.split(line))
                # get the time and object name
                _, t, obj, *_ = data_line_re.split(line)
                t = int(t)

                # add the data to the objects array
                data, is_array, (data_re0, data_re1), size = info[obj]
                # print(data, data_re, is_array, data_re.split(line))

                # in this case, obj is not an array and one value is given
                if not is_array:
                    # make sure that it maches obj expression
                    if not data_re0.match(line):
                        continue

                    _, new_data, _ = data_re0.split(line)

                # in this case, all of the array values are given
                elif data_re0.match(line):
                    _, new_data, _ = data_re0.split(line)
                    # reverse the values to conform with python index order
                    new_data = new_data.split()[::-1]
                    # print("here", obj, new_data)
                    if len(new_data) != size:
                        # print(obj, new_data)
                        # assert False
                        continue
                    # print(new_data, len(new_data))
                    # print(df[obj])
                    # print(is_array, data_re0.split(line))
                    # print(line, obj, new_data)
                    # exit()

                # in this case, only one array value is given for an index
                elif data_re1.match(line):
                    last_t = data["time"][-1]

                    # if t < last_t, data is useless
                    if t < last_t:
                        continue

                    _, index, new_data, *_ = data_re1.split(line)
                    # if t > last_t, we need to append t and copy last data
                    if t > last_t:

                        data["time"].append(t)
                        data[obj].append([v for v in data[obj][-1]])

                    # otherwise, t == last_t and we need to update the data
                    # print(obj, index, data[obj])
                    curr_data = data[obj][-1]
                    curr_data[int(index)] = new_data

                    continue
                else:
                    assert False, line

                data["time"].append(t)
                data[obj].append(new_data)

        key, *_ = [key for key in info.keys() if "clk" in key]
        data, *_ = info.pop(key)
        df = pd.DataFrame(data).set_index("time")
        for data, *_ in info.values():
            # if data is empty, skip
            if not data["time"]:
                continue
            df = df.join(pd.DataFrame(data).set_index("time"))
            # df = df.join(other_df)

        # only keep the last 20 values for now
        self._wave_data = df.fillna(method="ffill").drop(
            df.index[:-20], axis=0)

    def spath(self, signal: V_Signal) -> str:
        return self._tb.ins_path(signal)

    def __getitem__(self, signal: V_Signal):
        """
        Returns the column data for `signal`.
        """

        path = self._tb.ins_path(signal)

        return self._wave_data[self._signal_map[path]]

    def __str__(self) -> str:

        return "\n".join(str(self._wave_data[col]) for col in self._wave_data.columns)
