from verilog.core.vtypes import BitWidth
from verilog.targets.dot_product import DotProduct
from verilog.targets.signed_mult import SignedMult


class SignedDotProduct(DotProduct):

    def __init__(
        self,
        int_width: BitWidth,
        dec_width: BitWidth
    ):
        super().__init__(width=int_width + dec_width,
                         mult=SignedMult(int_width, dec_width))

        self.int_width = int_width
        self.dec_width = dec_width

        # change ports and variables to be signed
        self.a.signed = True
        self.b.signed = True
        self.out.signed = True
        self.sum.signed = True
        self.prod.signed = True
