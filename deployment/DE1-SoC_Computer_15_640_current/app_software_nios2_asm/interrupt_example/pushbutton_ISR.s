.include "address_map_nios2.s"
.include "key_codes.s"				/* defines values for KEY1, KEY2, KEY3 */
.extern	KEY_PRESSED					/* externally defined variable */
.extern	PATTERN						/* externally defined variable */
/***************************************************************************************
 * Pushbutton - Interrupt Service Routine                                
 *                                                                          
 * This routine checks which KEY has been pressed. It writes this value to the global 
 * variable KEY_PRESSED. 
****************************************************************************************/
	.global	PUSHBUTTON_ISR
PUSHBUTTON_ISR:
	subi		sp, sp, 20					/* reserve space on the stack */
   stw		ra, 0(sp)
   stw		r10, 4(sp)
   stw		r11, 8(sp)
   stw		r12, 12(sp)
   stw		r13, 16(sp)

	movia		r10, KEY_BASE				/* base address of pushbutton KEY parallel port */
	ldwio		r11, 0xC(r10)				/* read edge capture register */
    
	stwio		r11, 0xC(r10)				/* clear the interrupt */     

	movia		r10, KEY_PRESSED			/* global variable to return the result */
CHECK_KEY0:
	andi		r13, r11, 0b0001			/* check KEY0 */
	beq		r13, zero, CHECK_KEY1
	movi		r12, KEY0
	stw		r12, 0(r10)					/* return KEY0 value */
	br			END_PUSHBUTTON_ISR

CHECK_KEY1:
	andi		r13, r11, 0b0010			/* check KEY1 */
	beq		r13, zero, CHECK_KEY2
	movi		r12, KEY1
	stw		r12, 0(r10)					/* return KEY1 value */
	br			END_PUSHBUTTON_ISR

CHECK_KEY2:
	andi		r13, r11, 0b0100			/* check KEY2 */
	beq		r13, zero, DO_KEY3
	movi		r12, KEY2
	stw		r12, 0(r10)					/* return KEY2 value */
	br			END_PUSHBUTTON_ISR

DO_KEY3:
	movi		r12, KEY3
	stw		r12, 0(r10)					/* return KEY3 value */

END_PUSHBUTTON_ISR:
   ldw		ra,  0(sp)					/* Restore all used register to previous */
   ldw		r10, 4(sp)
   ldw		r11, 8(sp)
   ldw		r12, 12(sp)
   ldw		r13, 16(sp)
   addi		sp,  sp, 20

	ret
	.end
	
