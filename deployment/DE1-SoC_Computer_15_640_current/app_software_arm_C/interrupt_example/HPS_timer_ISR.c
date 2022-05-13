#include "address_map_arm.h" 	

extern volatile int tick;

/******************************************************************************
 * HPS timer0 interrupt service routine
 *                                                                          
 * This code increments the tick variable, and clears the interrupt
 *****************************************************************************/
void HPS_timer_ISR( )
{
	volatile int * HPS_timer_ptr = (int *) HPS_TIMER0_BASE;	// HPS timer address
	
	++tick;										// used by main program

	*(HPS_timer_ptr + 3); 					// Read timer end of interrupt register to
													// clear the interrupt
	return;
}

