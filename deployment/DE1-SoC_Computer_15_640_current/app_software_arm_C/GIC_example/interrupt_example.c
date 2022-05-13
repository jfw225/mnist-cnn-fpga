#include "address_map_arm.h"

void disable_A9_interrupts (void);
void set_A9_IRQ_stack (void);
void config_GIC (void);
void config_KEYs (void);
void enable_A9_interrupts (void);

/* ********************************************************************************
 * This program demonstrates use of interrupts with C code.  The program responds 
 * to interrupts from the pushbutton KEY port in the FPGA.
 *
 * The interrupt service routine for the KEYs indicates which KEY has been pressed
 * on the display HEX0.
 ********************************************************************************/
int main(void)
{
	disable_A9_interrupts ();	// disable interrupts in the A9 processor
	set_A9_IRQ_stack ();			// initialize the stack pointer for IRQ mode
	config_GIC ();					// configure the general interrupt controller
	config_KEYs ();				// configure pushbutton KEYs to generate interrupts

	enable_A9_interrupts ();	// enable interrupts in the A9 processor

	while (1) 						// wait for an interrupt
		;
}

/* setup the KEY interrupts in the FPGA */
void config_KEYs()
{
	volatile int * KEY_ptr = (int *) KEY_BASE;	// pushbutton KEY base address

	*(KEY_ptr + 2) = 0xF; 	// enable interrupts for all four KEYs
}
