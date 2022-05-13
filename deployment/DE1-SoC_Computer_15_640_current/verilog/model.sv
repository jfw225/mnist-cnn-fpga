// synthesis VERILOG_INPUT_VERSION SYSTEMVERILOG_2005
`ifndef __MODEL_SV__
`define __MODEL_SV__

`include "./layer3.sv"
`include "./layer1.sv"
`include "./layer4.sv"
`include "./layer2.sv"

module ModelV3(
	input    clk,
	input    reset,
	input    valid,
	output reg   done,
	output reg   ready,
	output reg  [9:0] inp_addr,
	input   [15:0] inp_data,
	output reg  [3:0] out_addr,
	output reg  [15:0] out_data,
	output reg   out_we
);

	wire   trans_mem0_write_enable;
	wire  [12:0] trans_mem0_read_addr;
	wire  [12:0] trans_mem0_write_addr;
	reg  [15:0] trans_mem0_read_data;
	wire  [15:0] trans_mem0_write_data;
	reg   Conv2D_XSIDO_reset;
	reg   Conv2D_XSIDO_valid;
	reg   Conv2D_XSIDO_done;
	reg   Conv2D_XSIDO_ready;
	wire   trans_mem1_write_enable;
	wire  [10:0] trans_mem1_read_addr;
	wire  [10:0] trans_mem1_write_addr;
	reg  [15:0] trans_mem1_read_data;
	wire  [15:0] trans_mem1_write_data;
	reg   Maxpool_ZSYYF_reset;
	reg   Maxpool_ZSYYF_valid;
	reg   Maxpool_ZSYYF_done;
	reg   Maxpool_ZSYYF_ready;
	wire   trans_mem2_write_enable;
	wire  [10:0] trans_mem2_read_addr;
	wire  [10:0] trans_mem2_write_addr;
	reg  [15:0] trans_mem2_read_data;
	wire  [15:0] trans_mem2_write_data;
	reg   Conv2DSum1D_LXJTZ_reset;
	reg   Conv2DSum1D_LXJTZ_valid;
	reg   Conv2DSum1D_LXJTZ_done;
	reg   Conv2DSum1D_LXJTZ_ready;
	reg   Maxpool_FC_YJWNR_reset;
	reg   Maxpool_FC_YJWNR_valid;
	reg   Maxpool_FC_YJWNR_done;
	reg   Maxpool_FC_YJWNR_ready;
	reg  [2:0] state;


	// the model state machine
	always @ (posedge clk) begin
		if (reset) begin
			// clear done flag and go to starting state
			done <= 1'd0;
			Conv2D_XSIDO_reset <= 1'd1;
			Maxpool_ZSYYF_reset <= 1'd1;
			Conv2DSum1D_LXJTZ_reset <= 1'd1;
			Maxpool_FC_YJWNR_reset <= 1'd1;
			Conv2D_XSIDO_valid <= 1'd0;
			Maxpool_ZSYYF_valid <= 1'd0;
			Conv2DSum1D_LXJTZ_valid <= 1'd0;
			Maxpool_FC_YJWNR_valid <= 1'd0;
			state <= 3'd0;
		end
		else begin
			case (state)
				// _StWaitValid
				3'd0: begin
						if (valid) begin
							Conv2D_XSIDO_reset <= 1'd0;
							Conv2D_XSIDO_valid <= 1'd1;
							state <= 3'd1;
						end
					end
				// _StWaitLayer1Done
				3'd1: begin
						Conv2D_XSIDO_valid <= 1'd0;
						if (Conv2D_XSIDO_done) begin
							Maxpool_ZSYYF_reset <= 1'd0;
							Maxpool_ZSYYF_valid <= 1'd1;
							state <= 3'd2;
						end
					end
				// _StWaitLayer2Done
				3'd2: begin
						Maxpool_ZSYYF_valid <= 1'd0;
						if (Maxpool_ZSYYF_done) begin
							Conv2DSum1D_LXJTZ_reset <= 1'd0;
							Conv2DSum1D_LXJTZ_valid <= 1'd1;
							state <= 3'd3;
						end
					end
				// _StWaitLayer3Done
				3'd3: begin
						Conv2DSum1D_LXJTZ_valid <= 1'd0;
						if (Conv2DSum1D_LXJTZ_done) begin
							Maxpool_FC_YJWNR_reset <= 1'd0;
							Maxpool_FC_YJWNR_valid <= 1'd1;
							state <= 3'd4;
						end
					end
				// _StWaitLayer4Done
				3'd4: begin
						Maxpool_FC_YJWNR_valid <= 1'd0;
						if (Maxpool_FC_YJWNR_done) begin
							state <= 3'd5;
						end
					end
				// V_StDone
				3'd5: begin
						// idle until reset
						done <= 1'd1;
						state <= 3'd5;
					end
			endcase
		end
	end

	// instantiate the transition memories
	trans_mem0 trans_mem0_BYNIY(
		.clk(clk), 
		.reset(reset), 
		.write_enable(trans_mem0_write_enable), 
		.read_addr(trans_mem0_read_addr), 
		.write_addr(trans_mem0_write_addr), 
		.read_data(trans_mem0_read_data), 
		.write_data(trans_mem0_write_data)
	);

	trans_mem1 trans_mem1_DIDYK(
		.clk(clk), 
		.reset(reset), 
		.write_enable(trans_mem1_write_enable), 
		.read_addr(trans_mem1_read_addr), 
		.write_addr(trans_mem1_write_addr), 
		.read_data(trans_mem1_read_data), 
		.write_data(trans_mem1_write_data)
	);

	trans_mem2 trans_mem2_HRQXH(
		.clk(clk), 
		.reset(reset), 
		.write_enable(trans_mem2_write_enable), 
		.read_addr(trans_mem2_read_addr), 
		.write_addr(trans_mem2_write_addr), 
		.read_data(trans_mem2_read_data), 
		.write_data(trans_mem2_write_data)
	);


	// instantiate the layers
	Conv2D_XSIDO Conv2D_XSIDO_CWNFV(
		.clk(clk), 
		.reset(Conv2D_XSIDO_reset), 
		.valid(Conv2D_XSIDO_valid), 
		.done(Conv2D_XSIDO_done), 
		.ready(Conv2D_XSIDO_ready), 
		.inp_addr(inp_addr), 
		.inp_data(inp_data), 
		.out_addr(trans_mem0_write_addr), 
		.out_data(trans_mem0_write_data), 
		.out_we(trans_mem0_write_enable)
	);

	Maxpool_ZSYYF Maxpool_ZSYYF_TPVBU(
		.clk(clk), 
		.reset(Maxpool_ZSYYF_reset), 
		.valid(Maxpool_ZSYYF_valid), 
		.done(Maxpool_ZSYYF_done), 
		.ready(Maxpool_ZSYYF_ready), 
		.inp_addr(trans_mem0_read_addr), 
		.inp_data(trans_mem0_read_data), 
		.out_addr(trans_mem1_write_addr), 
		.out_data(trans_mem1_write_data), 
		.out_we(trans_mem1_write_enable)
	);

	Conv2DSum1D_LXJTZ Conv2DSum1D_LXJTZ_XATMI(
		.clk(clk), 
		.reset(Conv2DSum1D_LXJTZ_reset), 
		.valid(Conv2DSum1D_LXJTZ_valid), 
		.done(Conv2DSum1D_LXJTZ_done), 
		.ready(Conv2DSum1D_LXJTZ_ready), 
		.inp_addr(trans_mem1_read_addr), 
		.inp_data(trans_mem1_read_data), 
		.out_addr(trans_mem2_write_addr), 
		.out_data(trans_mem2_write_data), 
		.out_we(trans_mem2_write_enable)
	);

	Maxpool_FC_YJWNR Maxpool_FC_YJWNR_HVROP(
		.clk(clk), 
		.reset(Maxpool_FC_YJWNR_reset), 
		.valid(Maxpool_FC_YJWNR_valid), 
		.done(Maxpool_FC_YJWNR_done), 
		.ready(Maxpool_FC_YJWNR_ready), 
		.inp_addr(trans_mem2_read_addr), 
		.inp_data(trans_mem2_read_data), 
		.out_addr(out_addr), 
		.out_data(out_data), 
		.out_we(out_we)
	);


endmodule



module trans_mem0(
	input    clk,
	input    reset,
	input    write_enable,
	input   [12:0] read_addr,
	input   [12:0] write_addr,
	output reg  [15:0] read_data,
	input   [15:0] write_data
);


	// force M10K ram style
	reg  [15:0] memory [5831:0]  /* synthesis ramstyle = "no_rw_check, M10K" */;

	always @ (posedge clk) begin
		if (write_enable) begin
			memory[write_addr] <= write_data;
		end
		read_data <= memory[read_addr];
	end

endmodule



module trans_mem1(
	input    clk,
	input    reset,
	input    write_enable,
	input   [10:0] read_addr,
	input   [10:0] write_addr,
	output reg  [15:0] read_data,
	input   [15:0] write_data
);


	// force M10K ram style
	reg  [15:0] memory [1351:0]  /* synthesis ramstyle = "no_rw_check, M10K" */;

	always @ (posedge clk) begin
		if (write_enable) begin
			memory[write_addr] <= write_data;
		end
		read_data <= memory[read_addr];
	end

endmodule



module trans_mem2(
	input    clk,
	input    reset,
	input    write_enable,
	input   [10:0] read_addr,
	input   [10:0] write_addr,
	output reg  [15:0] read_data,
	input   [15:0] write_data
);


	// force M10K ram style
	reg  [15:0] memory [1151:0]  /* synthesis ramstyle = "no_rw_check, M10K" */;

	always @ (posedge clk) begin
		if (write_enable) begin
			memory[write_addr] <= write_data;
		end
		read_data <= memory[read_addr];
	end

endmodule



`endif