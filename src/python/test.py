import numpy as np
from mnist_model_v1._layer1 import Layer1
from verilog.utils import dec2bin, format_int, nameof
from verilog.test_mod import TestMod
from verilog.core.vfile import V_FileWriter
from verilog.core.vmodule import write_config
# from verilog.signed_mult import SignedMult
# from verilog.dot_product import DotProduct
# from verilog.test_mod import TestMod
from verilog.iterables.m10k import M10K
from verilog.core.vsyntax import V_Array, V_Variable
from verilog.core.vtypes import V_Parameter, V_ParameterArray

"""
@TODO:
rewrite the dot product module (simple one to test integer mult) 
make the m10k block iterable to make a for loop equivalent
make a way to apply a module to an iterable and save in another iterable
make some iterable type (must rigidly define each port needed to get and set)

start with simple apply multiply 
"""


def main():
    x = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y = np.array([[1, 0, 1, 0]]).T
    weights = np.random.random((3, 1))

    def sigmoid(x): return 1/(1+np.exp(-x))
    for epoch in range(10000):
        z = x @ weights  # Matrix multiplication is equivalent to taking dot products of each training example with weights
        a = sigmoid(z)
        error = (y - a)
        da_dz = a * (1 - a)
        # weights -= np.dot(x.T, -error*sigmoidDerivative)
        weights += np.dot(x.T, error*da_dz)

    int_width, dec_width = 4, 23
    bin_weights = [dec2bin(w, int_width, dec_width) for w in weights]
    inp_arr = np.array([0, 1, 1])
    new_z = np.dot(inp_arr, weights)

    write_config(
        int_width + dec_width,
        bin_weights,
        [dec2bin(i, int_width, dec_width) for i in inp_arr],
        dec2bin(new_z, int_width, dec_width))

    # signed_mult = SignedMult(int_width + dec_width)
    # signed_mult.write()

    # dot_prod = DotProduct([inp_arr.shape[0], int_width + dec_width])
    # dot_prod.write()

    # test_mod = TestMod(int_width + dec_width, len(bin_weights))
    # test_mod.write()


if __name__ == '__main__':
    # main()

    width = 5
    size = 4

    input_mem = M10K(width, size)
    input_mem.set_init_data(V_Array(
        V_ParameterArray,
        width,
        size,
        False,
        [format_int(i, input_mem.width) for i in range(size)]
    ))
    input_file = input_mem.tofile("input_mem")
    # input_file = V_FileWriter("input_mem", [input_mem])
    # input_file.write()

    layer1 = Layer1(
        width,
        weights=[[i for i in range(size)]],
        input_mem=input_mem)

    layer1_file = V_FileWriter("layer1", [layer1])
    layer1_file.write()

    test_mod = TestMod(input_mem, layer1)
    # force test mod name for compilation
    test_mod.name = nameof(test_mod)

    test_mod_file = V_FileWriter("test_mod", [test_mod])
    test_mod_file.include(input_file)
    test_mod_file.include(layer1_file)
    test_mod_file.write()
