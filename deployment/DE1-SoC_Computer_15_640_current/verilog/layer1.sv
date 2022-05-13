// synthesis VERILOG_INPUT_VERSION SYSTEMVERILOG_2005
`ifndef __LAYER1_SV__
`define __LAYER1_SV__



`define conv2d_weights_SWAOQ parameter bit [15:0] conv2d_weights_SWAOQ [31:0] = '{ \
	16'b000000_0000100000, \
	16'b000000_1001111000, \
	16'b000000_0011010001, \
	16'b000000_0001001001, \
	16'b000000_1010000110, \
	16'b000000_0011101000, \
	16'b000000_0101001100, \
	16'b000000_0011100001, \
	16'b000000_0001011001, \
	16'b000000_1101001101, \
	16'b111111_1100110011, \
	16'b000000_0100000110, \
	16'b000000_0100110110, \
	16'b000000_0001011110, \
	16'b000000_0100111101, \
	16'b000000_0000011001, \
	16'b111111_1110101000, \
	16'b000000_0001110100, \
	16'b000000_0010010011, \
	16'b000000_0000011001, \
	16'b000000_0110100011, \
	16'b000000_0001100111, \
	16'b000000_0001001000, \
	16'b000000_1010001100, \
	16'b000000_0000000101, \
	16'b000000_0101011100, \
	16'b111111_1110010110, \
	16'b000000_0010111101, \
	16'b000000_0001111111, \
	16'b111111_1110001100, \
	16'b000000_0000100011, \
	16'b000000_0111100111\
};

`define conv2d_biases_SDRHO parameter bit [15:0] conv2d_biases_SDRHO [7:0] = '{ \
	16'b111111_1111111111, \
	16'b111111_1111111000, \
	16'b111111_1111111101, \
	16'b111111_1111111101, \
	16'b111111_1111111010, \
	16'b000000_1100011001, \
	16'b000000_1011110111, \
	16'b111111_1111111011\
};

module Conv2D_XSIDO(
	input    clk,
	input    reset,
	input    valid,
	output reg   done,
	output reg   ready,
	output reg  [9:0] inp_addr,
	input  signed [15:0] inp_data,
	output reg  [12:0] out_addr,
	output wire  [15:0] out_data,
	output reg   out_we
);
	`conv2d_weights_SWAOQ
	`conv2d_biases_SDRHO

	reg  [4:0] weights_ra;
	wire  [15:0] weights_rd;
	reg  [2:0] biases_ra;
	wire  [15:0] biases_rd;
	reg  [9:0] ti;
	reg  [9:0] tj;
	reg  [9:0] ii;
	reg  [9:0] jj;
	reg  [9:0] r;
	reg  [4:0] wc;
	reg signed [15:0] sub_img [3:0];
	wire signed [15:0] conv_out;
	wire signed [15:0] mult_out_0;
	wire signed [15:0] mult_out_1;
	wire signed [15:0] mult_out_2;
	wire signed [15:0] mult_out_3;
	reg  [3:0] state;


	// tie the weight data to the weight address
	assign weights_rd = conv2d_weights_SWAOQ[weights_ra];

	// tie the bias data to the bias address
	assign biases_rd = conv2d_biases_SDRHO[biases_ra];

	// instantiate the state machine
	always @ (posedge clk) begin
		if (reset) begin
			// clear done flag and go to starting state
			done <= 1'd0;
			inp_addr <= 10'd0;
			out_addr <= 13'd0;
			out_we <= 1'd0;
			ti <= 10'd0;
			tj <= 10'd0;
			wc <= 5'd0;
			state <= 4'd0;
		end
		else begin
			case (state)
				// _StWaitValid
				4'd0: begin
						if (valid) begin
							state <= 4'd1;
						end
					end
				// _StResetSub
				4'd1: begin
						ii <= 10'd0;
						jj <= 10'd0;
						r <= 10'd0;
						state <= 4'd2;
					end
				// _StSetInpAddr
				4'd2: begin
						inp_addr <= (ti + ii) * 10'd28 + tj + jj;
						state <= 4'd3;
					end
				// _StSubBuffer
				4'd3: begin
						state <= 4'd4;
					end
				// _StSetSub
				4'd4: begin
						sub_img[r] <= inp_data;
						if (r == 10'd3) begin
							wc <= 5'd0;
							state <= 4'd5;
						end
						else begin
							r <= r + 10'd1;
							if (jj == 10'd1) begin
								ii <= ii + 10'd1;
								jj <= 10'd0;
							end
							else begin
								jj <= jj + 10'd1;
							end
							state <= 4'd2;
						end
					end
				// _StComputeConv
				4'd5: begin
						out_we <= 1'd1;
						state <= 4'd6;
					end
				// _StIncWeightIndices
				4'd6: begin
						out_we <= 1'd0;
						if (wc == 5'd7) begin
							wc <= 5'd0;
							if (out_addr == 5831) begin
								state <= 4'd8;
							end
							else begin
								out_addr <= out_addr + 13'd1;
								state <= 4'd7;
							end
						end
						else begin
							wc <= wc + 5'd1;
							out_addr <= out_addr + 13'd1;
							state <= 4'd5;
						end
					end
				// _StIncTargetIndices
				4'd7: begin
						if (tj == 10'd26) begin
							tj <= 10'd0;
							if (ti == 10'd26) begin
								state <= 4'd8;
							end
							else begin
								ti <= ti + 10'd1;
								state <= 4'd1;
							end
						end
						else begin
							tj <= tj + 10'd1;
							state <= 4'd1;
						end
					end
				// V_StDone
				4'd8: begin
						// idle until reset
						done <= 1'd1;
						state <= 4'd8;
					end
			endcase
		end
	end
	// instantiate the multipliers
	SignedMult_BQGMO SignedMult_BQGMO_GIJSZ(
		.clk(), 
		.reset(), 
		.valid(), 
		.done(), 
		.ready(), 
		.input_IAMLC(sub_img[0]), 
		.input_BPUAH(conv2d_weights_SWAOQ[wc + 5'd0]), 
		.output_PFMCH(mult_out_0)
	);

	SignedMult_BQGMO SignedMult_BQGMO_LCBVV(
		.clk(), 
		.reset(), 
		.valid(), 
		.done(), 
		.ready(), 
		.input_IAMLC(sub_img[1]), 
		.input_BPUAH(conv2d_weights_SWAOQ[wc + 5'd8]), 
		.output_PFMCH(mult_out_1)
	);

	SignedMult_BQGMO SignedMult_BQGMO_FUEYC(
		.clk(), 
		.reset(), 
		.valid(), 
		.done(), 
		.ready(), 
		.input_IAMLC(sub_img[2]), 
		.input_BPUAH(conv2d_weights_SWAOQ[wc + 5'd16]), 
		.output_PFMCH(mult_out_2)
	);

	SignedMult_BQGMO SignedMult_BQGMO_PJCOF(
		.clk(), 
		.reset(), 
		.valid(), 
		.done(), 
		.ready(), 
		.input_IAMLC(sub_img[3]), 
		.input_BPUAH(conv2d_weights_SWAOQ[wc + 5'd24]), 
		.output_PFMCH(mult_out_3)
	);


	// assign the output of the convolution
	assign out_data = mult_out_0 + 
		mult_out_1 + 
		mult_out_2 + 
		mult_out_3 + 
		conv2d_biases_SDRHO[wc];

endmodule



module SignedMult_BQGMO(
	input    clk,
	input    reset,
	input    valid,
	output    done,
	output reg   ready,
	input  signed [15:0] input_IAMLC,
	input  signed [15:0] input_BPUAH,
	output  signed [15:0] output_PFMCH
);

	wire signed [31:0] mult_out;

	// tie `done` to `HIGH`
	assign done = 1'd1;

	// intermediate full bit length mult
	assign mult_out = input_IAMLC * input_BPUAH;

	// select bits for `N.M` fixed point
	assign output_PFMCH = {mult_out[31], mult_out[24:10]};

endmodule



`endif