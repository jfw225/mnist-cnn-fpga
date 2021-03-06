module qdiv #(
	parameter Q = 15,
	parameter N = 32
	)
	(
	input 	[N-1:0] i_dividend,
	input 	[N-1:0] i_divisor,
	input 	i_start, // valid
	input 	i_clk,
	output 	[N-1:0] o_quotient_out,
	output 	o_complete, // done
	output	o_overflow // done need
	);
 
	reg [2*N+Q-3:0]	reg_working_quotient;	//	Our working copy of the quotient
	reg [N-1:0] 		reg_quotient;				//	Final quotient
	reg [N-2+Q:0] 		reg_working_dividend;	//	Working copy of the dividend
	reg [2*N+Q-3:0]	reg_working_divisor;		// Working copy of the divisor
 
	reg [N-1:0] 			reg_count; 		//	This is obviously a lot bigger than it needs to be, as we only need 
													//		count to N-1+Q but, computing that number of bits requires a 
													//		logarithm (base 2), and I don't know how to do that in a 
													//		way that will work for everyone
										 
	reg					reg_done;			//	Computation completed flag
	reg					reg_sign;			//	The quotient's sign bit
	reg					reg_overflow;		//	Overflow flag
 
	initial reg_done = 1'b1;				//	Initial state is to not be doing anything
	initial reg_overflow = 1'b0;			//		And there should be no woverflow present
	initial reg_sign = 1'b0;				//		And the sign should be positive

	initial reg_working_quotient = 0;	
	initial reg_quotient = 0;				
	initial reg_working_dividend = 0;	
	initial reg_working_divisor = 0;		
 	initial reg_count = 0; 		

 
//	assign o_quotient_out[N-2:0] = reg_quotient[N-2:0];	//	The division results
//	assign o_quotient_out[N-1] = reg_sign;						//	The sign of the quotient
	assign o_quotient_out = reg_sign? ~{1'b0, reg_quotient[N-2:0]} + 1: {reg_sign, reg_quotient[N-2:0]};
	assign o_complete = reg_done;
	assign o_overflow = reg_overflow;
 
	reg [N-1:0] dividend_correction ;
	reg [N-1:0] divisor_correction ;
	reg div_correction;
	initial div_correction = 0;
	always @( posedge i_clk ) begin
		if( reg_done && i_start ) begin										//	This is our startup condition
			//  Need to check for a divide by zero right here, I think....
			reg_done <= 1'b0;												//	We're not done			
			reg_count <= N+Q-1;											//	Set the count
			reg_working_quotient <= 0;									//	Clear out the quotient register
			reg_working_dividend <= 0;									//	Clear out the dividend register 
			reg_working_divisor <= 0;									//	Clear out the divisor register 
			reg_overflow <= 1'b0;										//	Clear the overflow register

			reg_working_dividend[N+Q-2:Q] <= i_dividend[N-2:0];				//	Left-align the dividend in its working register
			reg_working_divisor[2*N+Q-3:N+Q-1] <= i_divisor[N-2:0];		//	Left-align the divisor into its working register

			reg_sign <= i_dividend[N-1] ^ i_divisor[N-1];		//	Set the sign bit
			
			dividend_correction <= ~i_dividend + 1;
			divisor_correction <= ~i_divisor + 1;
			div_correction <= 1'b1;
		end 
		else if (div_correction && !reg_done) begin	
			reg_working_dividend[N+Q-2:Q] <= i_dividend[N-1]? dividend_correction[N-2:0] : i_dividend[N-2:0];				
			reg_working_divisor[2*N+Q-3:N+Q-1] <= i_divisor[N-1]? divisor_correction[N-2:0] : i_divisor[N-2:0];		
			div_correction <= 1'b0;

		end
		else if(!div_correction && !reg_done) begin
			reg_working_divisor <= reg_working_divisor >> 1;	//	Right shift the divisor (that is, divide it by two - aka reduce the divisor)
			reg_count <= reg_count - 1;								//	Decrement the count

			//	If the dividend is greater than the divisor
			if(reg_working_dividend >= reg_working_divisor) begin
				reg_working_quotient[reg_count] <= 1'b1;										//	Set the quotient bit
				reg_working_dividend <= reg_working_dividend - reg_working_divisor;	//		and subtract the divisor from the dividend
				end
 
			//stop condition
			if(reg_count == 0) begin
				reg_done <= 1'b1;										//	If we're done, it's time to tell the calling process
				reg_quotient <= reg_working_quotient;			//	Move in our working copy to the outside world
				if (reg_working_quotient[2*N+Q-3:N]>0)
					reg_overflow <= 1'b1;
					end
			else
				reg_count <= reg_count - 1;	
			end
		end
endmodule





module TestDiv;

	// Inputs
	reg [31:0] i_dividend;
	reg [31:0] i_divisor;
	reg i_start;
	reg i_clk;

	// Outputs
	wire [31:0] o_quotient_out;
	wire o_complete;
	wire o_overflow;

	// Instantiate the Unit Under Test (UUT)
	qdiv #(15,32) uut (
		.i_dividend(i_dividend), 
		.i_divisor(i_divisor), 
		.i_start(i_start), 
		.i_clk(i_clk), 
		.o_quotient_out(o_quotient_out), 
		.o_complete(o_complete), 
		.o_overflow(o_overflow)
	);

	reg [10:0]	count;
	parameter signed [31:0] cor_div = 32'b11111111111111111_010101010101011;

	initial begin
		// Initialize Inputs
				
		i_dividend= 32'b00000000000000010_000000000000000; //-1
		i_divisor = 32'b11111111111111101_000000000000000; //5
		i_start = 0;      
		i_clk = 0;	
				
				
		count <= 0;

		// Wait 100 ns for global reset to finish
		#100;

		// Add stimulus here
		forever #2 i_clk = ~i_clk;
	end
        
		always @(posedge i_clk) begin
			if (count == 47) begin
				count <= 0;
				i_start <= 1'b1;
				end
			else begin				
				count <= count + 1;
				i_start <= 1'b0;
				end
			end

		always @(count) begin
			if (count == 47) begin
					i_divisor = i_divisor;
				end
			end
	

	always @(posedge o_complete)
		$display ("%b,%b,%b, %b", i_dividend, i_divisor, o_quotient_out, o_overflow);		//	Monitor the stuff we care about

endmodule
