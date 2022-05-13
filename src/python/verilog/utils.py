import subprocess
import string
import random
import numpy as np
from typing import Any, Iterable, Optional, Tuple

from verilog.core.vtypes import BitWidth


def format_binary(
    binary: str,
    width: int
):
    prefix, suffix = binary.split(".")

    return f"{width}'b{prefix}_{suffix}"


def twos_complement(binary):
    """
    binary is a STRING of 0s and 1s (no underscore)
    """
    def invert(x): return '1' if x == '0' else '0' if x == '1' else '_'
    first_one = False
    result = ''
    for i in range(len(binary)-1, -1, -1):
        if first_one:
            result += invert(binary[i])
        else:
            result += binary[i]
        if binary[i] == '1' and first_one == False:
            first_one = True
    return result[::-1]


def fixedfloat2dec(num, N_integer=1, M_dec=17):
    """
    num is an INTEGER value from modelsim that represents our N.M fixed point
    N_integer is an INTEGER number of desired integer bits 
    M_dec is an INTEGER number of desired decimal bits 
    """
    negative = -1 if num < 0 else 1
    num = abs(num)
    binary = '0' * (N_integer + M_dec - len(bin(num)[2:])) + bin(num)[2:]
    binary = binary if negative == 1 else twos_complement(binary)
    result = 0
    binary_weights = [2**i for i in range(N_integer-1, -M_dec - 1, -1)]
    binary_weights[0] = -binary_weights[0] if negative == - \
        1 else binary_weights[0]
    for i, w in zip(binary, binary_weights):
        result += eval(i) * w
    return result


def fixedbin2dec(binary, N_integer=1, M_dec=17):
    """
    binary is a STRING of 0s and 1s
    N_integer is an INTEGER number of desired integer bits 
    M_dec is an INTEGER number of desired decimal bits 
    """
    negative = -1 if binary[0] == '1' else 1
    result = 0
    binary_weights = [2**i for i in range(N_integer-1, -M_dec - 1, -1)]
    binary_weights[0] = -binary_weights[0] if negative == - \
        1 else binary_weights[0]
    for i, w in zip(binary, binary_weights):
        result += eval(i) * w
    return result


def dec2bin(num, integer=1, k_prec=17):
    int_width, dec_width = integer, k_prec

    negative = True if num < 0 else False
    num = num * -1 if negative else num

    binary = ""

    # Fetch the integral part of
    # decimal number
    Integral = int(num)

    # Fetch the fractional part
    # decimal number
    fractional = num - Integral

    # Conversion of integral part to
    # binary equivalent
    while (Integral):

        rem = Integral % 2

        # Append 0 in binary
        binary += str(rem)

        Integral //= 2

    # Reverse string to get original
    # binary equivalent
    binary = binary[:: -1]
    # Append point before conversionof fractional part
    binary += '.'

    # Conversion of fractional part
    # to binary equivalent
    while (k_prec):

        # Find next bit in fraction
        fractional *= 2
        fract_bit = int(fractional)
        if (fract_bit == 1):

            fractional -= fract_bit
            binary += '1'
        else:
            binary += '0'
        k_prec -= 1

    binary = '0' * (integer - len(binary.split('.')[0])) + binary

    def twos_complement(binary):
        def invert(x): return '1' if x == '0' else '0' if x == '1' else '.'
        first_one = False
        result = ''
        for i in range(len(binary)-1, -1, -1):
            if first_one:
                result += invert(binary[i])
            else:
                result += binary[i]
            if binary[i] == '1' and first_one == False:
                first_one = True
        return result[::-1]

    binary = twos_complement(binary) if negative else binary

    return format_binary(binary, int_width + dec_width)


def id_generator(size=5, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))


def nameof(obj):
    # print(obj, str(obj))

    name, *_ = object.__str__(obj).split(".")[-1].split()[0].split("'")
    return name
    # return object.__str__(obj).split(".")[-1].split()[0]


def format_int(integer: int, width: BitWidth):
    if 2 ** width < integer:
        raise Exception(
            f'"{integer} can not be represented using "{width}" bits.')

    return f"{width}'d{integer}"


def run_cmd(
    *commands: Iterable[str],
    timeout: Optional[float] = None
) -> Iterable[str]:
    """
    Runs a system command determined by `commands`. 
    """

    output = subprocess.run(commands,

                            capture_output=True,
                            text=True,
                            timeout=timeout)

    return output.stdout.split("\n")


def mean_squared_error(*pairs: Tuple[Any, Any]) -> float:
    assert len(pairs) > 0, pairs

    A, B = [np.array(arr) for arr in zip(*pairs)]

    i_m, a_m, b_m, d_m = 0, 0, 0, 0
    for i, (a, b) in enumerate(pairs):
        # print(i, a, b, a-b)

        if np.square(a - b) > d_m:
            i_m = i
            a_m = a
            b_m = b
            d_m = np.square(a - b)

    # print("--")
    # print("max", i_m, a_m, b_m, d_m)
    return np.square(A - B).mean()
