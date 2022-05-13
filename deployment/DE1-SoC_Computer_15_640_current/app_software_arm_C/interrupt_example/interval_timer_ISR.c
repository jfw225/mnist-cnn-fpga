#include "address_map_arm.h" 	
#include "defines.h" 	

extern volatile int key_pressed;
extern volatile int pattern;
extern volatile int shift_dir;
/*****************************************************************************
 * Interval timer interrupt service routine
 *                                                                          
 * Shifts a PATTERN being displayed on the HEX displays. The shift direction 
 * is determined by the external variable key_pressed.
 * 
******************************************************************************/
void interval_timer_ISR( )
{
	volatile int * interval_timer_ptr = (int *) TIMER_BASE;	// Altera timer address
  	volatile int * slider_switch_ptr = (int *) SW_BASE;		// SW base address
	volatile int * HEX3_HEX0_ptr	= (int *) HEX3_HEX0_BASE;	// HEX3_HEX0 address

	*(interval_timer_ptr) = 0; 				// Clear the interrupt

	*(HEX3_HEX0_ptr) = pattern;				// display pattern on HEX3 ... HEX0

	/* rotate the pattern shown on the HEX displays */
	if (key_pressed == KEY0)					// for KEY0 load a new pattern
		pattern = *(slider_switch_ptr); 		// read a new pattern from the SW slider switches
	else if (key_pressed == KEY1)				// for KEY1 rotate right
		shift_dir = RIGHT;
	else if (key_pressed == KEY2)				// for KEY2 rotate left
		shift_dir = LEFT;
	else if (key_pressed == KEY3)				// for KEY3 don't rotate
		shift_dir = NONE;

	key_pressed = NONE;							// key press handled, so clear

	if (shift_dir == LEFT)
	{
		if (pattern & 0x80000000)
			pattern = (pattern << 1) | 1;	
		else
			pattern = pattern << 1;			
	}
	else if (shift_dir == RIGHT)
	{
		if (pattern & 0x00000001)			
			pattern = (pattern >> 1) | 0x80000000;
		else
			pattern = (pattern >> 1) & 0x7FFFFFFF;
	}
	// else don't shift

	return;
}

