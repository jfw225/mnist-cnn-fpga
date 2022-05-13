				.include	"address_map_arm.s"

/****************************************************************************************
 * Pushbutton - Interrupt Service Routine                                
 *                                                                          
 * This routine checks which KEY has been pressed. It writes to HEX0
 ***************************************************************************************/
				
				.global	KEY_ISR
KEY_ISR:		
				LDR		R0, =KEY_BASE			// base address of pushbutton KEY port
				LDR		R1, [R0, #0xC]			// read edge capture register
				MOV		R2, #0xF
				STR		R2, [R0, #0xC]			// clear the interrupt

				LDR		R0, =HEX3_HEX0_BASE	// based address of HEX display
CHECK_KEY0:
				MOV		R3, #0x1
				ANDS		R3, R3, R1				// check for KEY0
				BEQ		CHECK_KEY1
				MOV		R2, #0b00111111
				STR		R2, [R0]					// display "0"
				B			END_KEY_ISR
CHECK_KEY1:
				MOV		R3, #0x2
				ANDS		R3, R3, R1				// check for KEY1
				BEQ		CHECK_KEY2
				MOV		R2, #0b00000110
				STR		R2, [R0]					// display "1"
				B			END_KEY_ISR
CHECK_KEY2:
				MOV		R3, #0x4
				ANDS		R3, R3, R1				// check for KEY2
				BEQ		IS_KEY3
				MOV		R2, #0b01011011
				STR		R2, [R0]					// display "2"
				B			END_KEY_ISR
IS_KEY3:
				MOV		R2, #0b01001111
				STR		R2, [R0]					// display "3"
END_KEY_ISR:
				BX			LR

				.end
