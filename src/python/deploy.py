"""
TODO:
add // synthesis VERILOG_INPUT_VERSION SYSTEMVERILOG_2005
add "bit" in front of each param
figure out bit width
convert m10k blocks to registers and connect long arrays
write script that converts code for deployment
"""

import numpy as np

from knet_v2.network import get_weights


if __name__ == '__main__':
    print(sum([np.prod(d.shape) for d in get_weights()]))
