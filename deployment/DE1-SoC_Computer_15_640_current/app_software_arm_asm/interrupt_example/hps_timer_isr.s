				.include	"address_map_arm.s"
				.extern TICK

/******************************************************************************
 * HPS timer0 interrupt service routine
 *                                                                          
 * This code increments the TICK global variable, and clears the interrupt
 *****************************************************************************/
					
				.global	HPS_TIMER_ISR
HPS_TIMER_ISR:

				LDR		R0, =HPS_TIMER0_BASE		// base address of timer
				LDR		R1, =TICK					// used by main program

				LDR		R2, [R1]
				ADD		R2, R2, #1
				STR		R2, [R1]						// ++tick

				LDR		R0, [R0, #0xC]				// read timer end of interrupt register to
															// clear the interrupt
				BX			LR
				.end
