				.include	"address_map_arm.s"
				.include	"defines.s"
				.extern	KEY_PRESSED					// externally defined variable
/****************************************************************************************
 * Pushbutton - Interrupt Service Routine                                
 *                                                                          
 * This routine checks which KEY has been pressed. It writes this value to the global 
 * variable KEY_PRESSED. 
 ***************************************************************************************/
				
				.global	KEY_ISR
KEY_ISR:		
				LDR		R0, =KEY_BASE				// base address of pushbutton KEY parallel port
				LDR		R1, [R0, #0xC]				// read edge capture register
				STR		R1, [R0, #0xC]				// clear the interrupt

				LDR		R0, =KEY_PRESSED			// global variable to return the result
CHECK_KEY0:
				MOVS		R3, #0x1
				ANDS		R3, R1						// check for KEY0
				BEQ		CHECK_KEY1
				MOVS		R2, #KEY0
				STR		R2, [R0]						// return KEY0 value
				B			END_KEY_ISR
CHECK_KEY1:
				MOVS		R3, #0x2
				ANDS		R3, R1						// check for KEY1
				BEQ		CHECK_KEY2
				MOVS		R2, #KEY1
				STR		R2, [R0]						// return KEY1 value
				B			END_KEY_ISR
CHECK_KEY2:
				MOVS		R3, #0x4
				ANDS		R3, R1						// check for KEY2
				BEQ		IS_KEY3
				MOVS		R2, #KEY2
				STR		R2, [R0]						// return KEY2 value
				B			END_KEY_ISR
IS_KEY3:
				MOVS		R2, #KEY3
				STR		R2, [R0]						// return KEY3 value
END_KEY_ISR:
				BX			LR

				.end
