#include "address_map_arm.h" 	// defines values for KEY1, KEY2, KEY3
#include "defines.h" 		// defines values for KEY1, KEY2, KEY3

extern volatile int key_pressed;

/****************************************************************************************
 * Pushbutton - Interrupt Service Routine                                
 *                                                                          
 * This routine checks which KEY has been pressed. It writes this value to the global 
 * variable KEY_PRESSED. 
 ***************************************************************************************/
void pushbutton_ISR( void )
{
	volatile int * KEY_ptr = (int *) KEY_BASE;
	int press;

	press = *(KEY_ptr + 3);					// read the pushbutton interrupt register
	*(KEY_ptr + 3) = press;					// Clear the interrupt

	if (press & 0x1)							// KEY0
		key_pressed = KEY0;
	else if (press & 0x2)					// KEY1
		key_pressed = KEY1;
	else if (press & 0x4)					// KEY2
		key_pressed = KEY2;
	else 											// press & 0x8, which is KEY3
		key_pressed = KEY3;

	return;
}
