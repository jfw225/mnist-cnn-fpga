// synthesis VERILOG_INPUT_VERSION SYSTEMVERILOG_2005
`ifndef __LAYER2_SV__
`define __LAYER2_SV__



module Maxpool_ZSYYF(
	input    clk,
	input    reset,
	input    valid,
	output reg   done,
	output reg   ready,
	output reg  [12:0] inp_addr,
	input   [15:0] inp_data,
	output reg  [10:0] out_addr,
	output reg  [15:0] out_data,
	output reg   out_we
);

	reg signed [15:0] window [31:0];
	reg  [12:0] curr_y;
	reg  [10:0] out_y;
	reg  [12:0] curr_x;
	reg  [10:0] out_x;
	reg  [12:0] jj;
	reg  [12:0] ii;
	reg  [12:0] kk;
	reg  [4:0] ww;
	reg  [10:0] wc;
	reg  [4:0] wr;
	wire signed [15:0] max_out;
	reg  [3:0] state;


	// instantiate the state machine
	always @ (posedge clk) begin
		if (reset) begin
			// clear done flag and go to starting state
			done <= 1'd0;
			inp_addr <= 13'd0;
			out_addr <= 11'd0;
			out_we <= 1'd0;
			curr_y <= 13'd0;
			out_y <= 11'd0;
			curr_x <= 13'd0;
			out_x <= 11'd0;
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
				// _StResetWindow
				4'd1: begin
						jj <= 13'd0;
						ii <= 13'd0;
						kk <= 13'd0;
						ww <= 5'd0;
						state <= 4'd2;
					end
				// _StSetInpAddr
				4'd2: begin
						inp_addr <= (curr_y + jj) * 13'd27 * 13'd8 + (curr_x + ii) * 13'd8 + kk;
						state <= 4'd3;
					end
				// _StWindowBuffer
				4'd3: begin
						state <= 4'd4;
					end
				// _StSetWindow
				4'd4: begin
						window[ww] <= inp_data;
						if (ww == 5'd31) begin
							state <= 4'd5;
						end
						else begin
							ww <= ww + 5'd1;
							if (kk == 13'd7) begin
								kk <= 13'd0;
								if (ii == 13'd1) begin
									ii <= 13'd0;
									jj <= jj + 13'd1;
								end
								else begin
									ii <= ii + 13'd1;
								end
							end
							else begin
								kk <= kk + 13'd1;
							end
							state <= 4'd2;
						end
					end
				// _StMaxReset
				4'd5: begin
						wc <= 11'd0;
						wr <= 5'd1;
						state <= 4'd6;
					end
				// _StGetMaxOverWindow
				4'd6: begin
						window[wc] <= max_out;
						if (wr == 5'd3) begin
							state <= 4'd7;
						end
						else begin
							wr <= wr + 5'd1;
							state <= 4'd6;
						end
					end
				// _StFindMax
				4'd7: begin
						out_we <= 1'd0;
						out_addr <= out_y * 11'd13 * 11'd8 + out_x * 11'd8 + wc;
						out_data <= max_out;
						state <= 4'd8;
					end
				// _StMaxBuffer
				4'd8: begin
						out_we <= 1'd1;
						if (wc == 11'd7) begin
							state <= 4'd9;
						end
						else begin
							wc <= wc + 11'd1;
							wr <= 5'd1;
							state <= 4'd6;
						end
					end
				// _StIncMaxpool
				4'd9: begin
						out_we <= 1'd0;
						if (curr_x > 13'd23) begin
							curr_x <= 13'd0;
							out_x <= 11'd0;
							if (curr_y > 13'd23) begin
								state <= 4'd11;
							end
							else begin
								curr_y <= curr_y + 13'd2;
								out_y <= out_y + 11'd1;
								state <= 4'd1;
							end
						end
						else begin
							curr_x <= curr_x + 13'd2;
							out_x <= out_x + 11'd1;
							state <= 4'd1;
						end
					end
				// _StWriteData
				4'd10: begin
						if (out_addr == 1351) begin
							out_we <= 1'd0;
							state <= 4'd11;
						end
						else begin
							out_addr <= out_addr + 11'd1;
						end
					end
				// V_StDone
				4'd11: begin
						// idle until reset
						done <= 1'd1;
						state <= 4'd11;
					end
			endcase
		end
	end
	// instantiate the signed maximum module
	SignedMax_CPTRB SignedMax_CPTRB_LEARM(
		.clk(), 
		.reset(), 
		.valid(), 
		.done(), 
		.ready(), 
		.input_NADFR(window[wc]), 
		.input_DNWTO(window[wc + wr * 5'd8]), 
		.output_NEPMP(max_out)
	);


endmodule



module SignedMax_CPTRB(
	input    clk,
	input    reset,
	input    valid,
	output    done,
	output reg   ready,
	input  signed [15:0] input_NADFR,
	input  signed [15:0] input_DNWTO,
	output  signed [15:0] output_NEPMP
);


	// tie `done` to `HIGH`
	assign done = 1'd1;

	// use ternary to determine the max
	assign output_NEPMP = (input_NADFR > input_DNWTO) ? 
		input_NADFR : 
		input_DNWTO;

endmodule



`endif