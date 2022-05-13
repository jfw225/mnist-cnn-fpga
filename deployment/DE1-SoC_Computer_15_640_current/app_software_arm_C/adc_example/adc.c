#include <stdio.h>
#include "address_map_arm.h"

/********************************************************************************
 * This program demonstrates use of the ADC port in the DE1-SoC and DE0-Nano-SoC
 * Computers.
 *
 * It performs the following:
 * 1. Performs a conversion operation on all eight channels.
 * 2. Displays the converted values on the terminal window.
********************************************************************************/

int main(void)
{
	/* Declare volatile pointers to I/O registers (volatile means that the locations
	 * will not be cached, even in registers) */
	volatile int * ADC_ptr = (int *) ADC_BASE;		// ADC port address

	volatile int delay_count;	// volatile so that the C compiler doesn't remove the loop
	int port, value;
	
	*(ADC_ptr + 1) = 1;	// Sets the ADC up to automatically perform conversions.
	while (1)
	{
		printf("\033[H"); // sets the cursor to the top-left of the terminal window
		for (port = 0; port < 8; ++port)
		{
			value = *(ADC_ptr + port);
			printf ("ADC port %d: 0x%x\n", port, value);
		}
		for (delay_count = 1500000; delay_count != 0; --delay_count); // delay loop
	}
}
