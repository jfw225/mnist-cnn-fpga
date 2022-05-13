`timescale 1ns / 1ps


module signed_mult #(
    parameter N = 4,
    parameter M = 23
    )
    (
	input 	signed	[N+M-1:0] 	a,
	input 	signed	[N+M-1:0] 	b,
    output 	signed  [N+M-1:0]	out
    );
	// intermediate full bit length
	wire 	signed	[(N+M)*2-1:0]	mult_out;
	assign mult_out = a * b;
	// select bits for N.M fixed point
	assign out = {mult_out[(N+M)*2-1], mult_out[ M  + (M + N - 2): M]};
endmodule




module mult_tb;

	wire [26:0] a;
	wire [26:0] b;
	wire [26:0] out;
	//signed_mult #(5, 22) uut(
	//		.a(a),
	//		.b(b),
	//		.out(out)
	//);

	
	assign a = 27'b11110_1100110011001100110100; //0.875
	assign b = 27'b11100_1111001100110011001101; //1.5

	

endmodule	



	