@REM must be run from `labs-ece5760/final`

@REM run python code to generate the verilog
@REM python ./src/python/test_ops.py
@REM python ./src/python/test_layer.py
python ./src/python/test_model.py

@REM compile the verilog
SETLOCAL ENABLEDELAYEDEXPANSION
@echo off

set error=""
@REM FOR /F "tokens=* USEBACKQ" %%F IN (`vlog -reportprogress 30 -work work ./src/verilog/ops_tb.sv`) DO (
@REM FOR /F "tokens=* USEBACKQ" %%F IN (`vlog -reportprogress 30 -work work ./src/verilog/layer_tb.sv`) DO (
FOR /F "tokens=* USEBACKQ" %%F IN (`vlog -reportprogress 30 -work work ./src/verilog/model_tb.sv`) DO (
@REM   ECHO %%F
    set copy=%%F
    set match_str=!copy:** Error:=!
    set at_str=!copy:** at=!
    if not !match_str!==%%F set error="error" && echo %%F
    if not !at_str!==%%F echo %%F
)

@REM run the simulation
@REM if !error!=="" vsim -gui work.OpsTB -do "do ./scripts/simulate.do"
@REM if !error!=="" vsim -gui work.LayerTB -do "do ./scripts/simulate.do"
if !error!=="" vsim -gui work.ModelTB -do "do ./scripts/simulate.do"


endlocal
