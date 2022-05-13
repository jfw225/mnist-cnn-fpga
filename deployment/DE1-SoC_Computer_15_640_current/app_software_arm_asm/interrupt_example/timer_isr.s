				.include	"address_map_arm.s"
				.include	"defines.s"
				/* externally defined variables */
				.extern	KEY_PRESSED				
				.extern	SHIFT_DIR					
				.extern	PATTERN
/*****************************************************************************
 * Interval timer interrupt service routine
 *                                                                          
 * Shifts a PATTERN being displayed on the HEX displays. The shift direction 
 * is determined by the external variable KEY_PRESSED.
 * 
 *****************************************************************************/
					
				.global	TIMER_ISR
TIMER_ISR:	
				PUSH		{R4-R7}
				LDR		R1, =TIMER_BASE		// interval timer base address
				MOVS		R0, #0
				STR		R0, [R1]					// clear the interrupt

				LDR		R1, =HEX3_HEX0_BASE	// HEX3_HEX0 base address
				LDR		R2, =PATTERN			// set up a pointer to the pattern for HEX displays
				LDR		R3, =KEY_PRESSED		// set up a pointer to the key pressed
				LDR		R7, =SHIFT_DIR			// set up a pointer to the shift direction variable

				LDR		R6, [R2]					// load pattern for HEX displays
				STR		R6, [R1]					// store to HEX3 ... HEX0

				LDR		R4, [R3]					// check which key has been pressed
CHK_KEY0:	CMP		R4, #KEY0
				BNE		CHK_KEY1					
				LDR		R1, =SW_BASE			// SW switches base address
				LDR		R6, [R1]					// load a new pattern from the SW switches
				B	 		SHIFT

CHK_KEY1:	CMP		R4, #KEY1
				BNE		CHK_KEY2					
				MOVS		R5, #RIGHT				// KEY1 sets rotation direction to the right
				STR		R5, [R7]
				B	 		SHIFT
CHK_KEY2:
				CMP		R4, #KEY2
				BNE		CHK_KEY3					
				MOVS		R5, #LEFT 				// KEY2 sets rotation direction to the left
				STR		R5, [R7]
				B	 		SHIFT
CHK_KEY3:
				CMP		R4, #KEY3
				BNE		SHIFT					
				MOVS		R5, #NONE 				// KEY3 sets rotation direction to no rotation
				STR		R5, [R7]
SHIFT:
				MOVS		R5, #NONE
				STR		R5, [R3]					// keypressed handled, so clear
				LDR		R5, [R7]					// get shift direction
				CMP		R5, #RIGHT
				BNE		SHIFT_L
				MOVS		R5, #1 					// used to rotate right by 1 position
				RORS		R6, R5					// rotate right for KEY1
				B			END_TIMER_ISR
SHIFT_L:		
				CMP		R5, #LEFT
				BNE		END_TIMER_ISR
				MOVS		R5, #31 					// used to rotate left by 1 position
				RORS		R6, R5					
END_TIMER_ISR:
				STR		R6, [R2]					// store HEX display pattern
				POP		{R4-R7}
				BX			LR

				.end
