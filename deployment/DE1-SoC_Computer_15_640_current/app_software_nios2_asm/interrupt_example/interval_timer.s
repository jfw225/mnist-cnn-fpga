.include "address_map_nios2.s"
.include "key_codes.s"				/* defines values for KEY1, KEY2, KEY3 */
.extern	PATTERN						/* externally defined variables */
.extern	KEY_PRESSED					
.extern	SHIFT_DIR					

/*****************************************************************************
 * Interval timer interrupt service routine
 *                                                                          
 * Shifts a PATTERN being displayed on the HEX displays. The shift direction 
 * is determined by the external variable KEY_PRESSED.
 * 
******************************************************************************/
	.global INTERVAL_TIMER_ISR
INTERVAL_TIMER_ISR:					
	subi		sp,  sp, 40				/* reserve space on the stack */
   stw		ra, 0(sp)
   stw		r4, 4(sp)
   stw		r5, 8(sp)
   stw		r6, 12(sp)
   stw		r8, 16(sp)
   stw		r10, 20(sp)
   stw		r20, 24(sp)
   stw		r21, 28(sp)
   stw		r22, 32(sp)
   stw		r23, 36(sp)

	movia		r10, TIMER_BASE		/* interval timer base address */
	sthio		r0,  0(r10)				/* clear the interrupt */

	movia		r20, HEX3_HEX0_BASE	/* HEX3_HEX0 base address */
	movia		r21, PATTERN			/* set up a pointer to the pattern for HEX displays */
	movia		r22, KEY_PRESSED		/* set up a pointer to the key pressed */
	movia		r23, SHIFT_DIR			/* set up a pointer to the shift direction variable */

	ldw		r6, 0(r21)				/* load pattern for HEX displays */
	stwio		r6, 0(r20)				/* store to HEX3 ... HEX0 */

	ldw		r4, 0(r22)				/* check which key has been pressed */
CHK_KEY0:	
	movi		r8, KEY0					/* code to check for KEY1 */
	bne		r4, r8, CHK_KEY1
	movia		r20, SW_BASE
	ldw		r6, 0(r20)				/* load a new pattern from the SW switches */
	br	 		SHIFT

CHK_KEY1:	
	movi		r8, KEY1					/* code to check for KEY1 */
	bne		r4, r8, CHK_KEY2
	movi		r5, RIGHT				/* KEY1 sets rotation direction to the right */
	stw		r5, 0(r23)
	br	 		SHIFT
CHK_KEY2:
	movi		r8, KEY2					/* code to check for KEY2 */
	bne		r4, r8, CHK_KEY3
	movi		r5, LEFT					/* KEY2 sets rotation direction to the left */
	stw		r5, 0(r23)
	br	 		SHIFT
CHK_KEY3:
	movi		r8, KEY3					/* code to check for KEY3 */
	bne		r4, r8, SHIFT
	movi		r5, NONE					/* KEY3 sets rotation to no rotation */
	stw		r5, 0(r23)

SHIFT:
	movi		r5, NONE
	stw		r5, 0(r22)				/* key press handled, so clear */
	ldw		r5, 0(r23)				/* get shift direction */
	movi		r8, RIGHT
	bne		r5, r8, SHIFT_L	
	movi		r5, 1		 				/* set r5 to the constant value 1 */
	ror		r6, r6, r5				/* rotate the displayed pattern right */
	br	 		END_INTERVAL_TIMER_ISR

SHIFT_L:
	movi		r8, LEFT
	bne		r5, r8, END_INTERVAL_TIMER_ISR	
	movi		r5, 1		 				/* set r5 to the constant value 1 */
	rol		r6, r6, r5				/* shift left */

END_INTERVAL_TIMER_ISR:
	stw		r6, 0(r21)				/* store HEX display pattern */

   ldw		ra, 0(sp)				/* restore registers */
   ldw		r4, 4(sp)
  	ldw		r5, 8(sp)
   ldw		r6, 12(sp)
   ldw		r8, 16(sp)
   ldw		r10, 20(sp)
   ldw		r20, 24(sp)
   ldw		r21, 28(sp)
   ldw		r22, 32(sp)
   ldw		r23, 36(sp)	
   addi		sp,  sp, 40				/* release the reserved space on the stack */

	ret

	.end
	
