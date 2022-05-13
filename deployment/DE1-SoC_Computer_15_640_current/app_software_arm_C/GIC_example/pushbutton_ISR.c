#include "address_map_arm.h"

/****************************************************************************************
 * Pushbutton - Interrupt Service Routine                                
 *                                                                          
 * This routine checks which KEY has been pressed. It writes to HEX0
 ***************************************************************************************/
void pushbutton_ISR( void )
{
	volatile int * KEY_ptr = (int *) KEY_BASE;
	volatile int * HEX3_HEX0_ptr = (int *) HEX3_HEX0_BASE;
	int press, HEX_bits;

	press = *(KEY_ptr + 3);					// read the pushbutton interrupt register
	*(KEY_ptr + 3) = press;					// Clear the interrupt

	if (press & 0x1)							// KEY0
		HEX_bits = 0b00111111;
	else if (press & 0x2)					// KEY1
		HEX_bits = 0b00000110;
	else if (press & 0x4)					// KEY2
		HEX_bits = 0b01011011;
	else 											// press & 0x8, which is KEY3
		HEX_bits = 0b01001111;

	*HEX3_HEX0_ptr = HEX_bits;
	return;
}
