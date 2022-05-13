#include "address_map_arm.h"
#include "defines.h"
#include <stdio.h>
#include "JTAG_UART.h"

void set_A9_IRQ_stack (void);
void config_GIC (void);
void config_HPS_timer (void);
void config_HPS_GPIO1 (void);
void config_interval_timer (void);
void config_KEYs (void);
void enable_A9_interrupts (void);

/* These global variables are written by interrupt service routines; we have to declare 
 * these as volatile to avoid the compiler caching their values in registers */
volatile int tick = 0;					// set to 1 every time the HPS timer expires
volatile int key_pressed = NONE;		// stores a KEY value when pressed
volatile	int pattern = 0x0000000F;	// pattern for HEX displays
volatile	int shift_dir = LEFT;		// direction to shift the pattern

/* ********************************************************************************
 * This program demonstrates use of interrupts with C code. It first starts
 * two timers: an HPS timer, and the FPGA interval timer. The program responds 
 * to interrupts from these timers in addition to the pushbutton KEYs in the FPGA.
 *
 * The interrupt service routine for the HPS timer causes the main program to flash
 * the green light connected to the HPS GPIO1 port.
 * 
 * The interrupt service routine for the Altera interval timer displays a pattern 
 * on the HEX3-0 displays, and rotates this pattern either left or right:
 *		KEY[0]: loads a new pattern from the SW switches
 *		KEY[1]: rotates the displayed pattern to the right
 *		KEY[2]: rotates the displayed pattern to the left
 *		KEY[3]: stops the rotation
 ********************************************************************************/
int main(void)
{
	volatile int * HPS_GPIO1_ptr = (int *) HPS_GPIO1_BASE;	// GPIO1 base address
  	volatile int * LEDR_ptr = (int *) LEDR_BASE;					// LEDR base address
  	volatile int * slider_switch_ptr = (int *) SW_BASE;		// SW base address
	volatile int HPS_timer_LEDG = 0x01000000;		// value to turn on the HPS green light LEDG
	
	char text_string[] = "\nInterrupt example code\n\0";
	print_jtag (text_string);

	set_A9_IRQ_stack ();			// initialize the stack pointer for IRQ mode
	config_GIC ();					// configure the general interrupt controller
	config_HPS_timer ();			// configure the HPS timer
	config_HPS_GPIO1 ();			// configure the HPS GPIO1 interface
	config_interval_timer ();	// configure Altera interval timer to generate interrupts
	config_KEYs ();				// configure pushbutton KEYs to generate interrupts

	enable_A9_interrupts ();	// enable interrupts

	while (1) 
	{
		*(LEDR_ptr) = *(slider_switch_ptr); 	// light up the red lights
		if (tick)
		{
			tick = 0;
			*HPS_GPIO1_ptr = HPS_timer_LEDG;		// turn on/off the green light LEDG
			HPS_timer_LEDG ^= 0x01000000;			// toggle the bit that controls LEDG
		}
	}

	return 0;
}

/* setup HPS timer */
void config_HPS_timer()
{
	volatile int * HPS_timer_ptr = (int *) HPS_TIMER0_BASE;	// timer base address

	*(HPS_timer_ptr + 0x2) = 0;		// write to control register to stop timer
	/* set the timer period */
	int counter = 100000000;			// period = 1/(100 MHz) x (100 x 10^6) = 1 sec
	*(HPS_timer_ptr) = counter;		// write to timer load register

	/* write to control register to start timer, with interrupts */
	*(HPS_timer_ptr + 2) = 0b011;		// int mask = 0, mode = 1, enable = 1
}

/* setup HPS GPIO1. The GPIO1 port has one green light (LEDG) and one pushbutton KEY connected
 * for the DE1-SoC Computer. The KEY is connected to GPIO1[25], and is not used here. The green
 * LED is connected to GPIO1[24]. */
void config_HPS_GPIO1()
{
	volatile int * HPS_GPIO1_ptr = (int *) HPS_GPIO1_BASE;	// GPIO1 base address

	*(HPS_GPIO1_ptr + 0x1) = 0x01000000;	// write to the data direction register to set
														// bit 24 (LEDG) to be an output
	// Other possible actions include setting up GPIO1 to use the KEY, including setting
	// the debounce option and causing the KEY to generate an interrupt. We do not use the
	// KEY in this example.
}

/* setup the interval timer interrupts in the FPGA */
void config_interval_timer()
{
	volatile int * interval_timer_ptr = (int *) TIMER_BASE;	// interal timer base address

	/* set the interval timer period for scrolling the HEX displays */
	int counter = 5000000;	// 1/(100 MHz) x 5x10^6 = 50 msec
	*(interval_timer_ptr + 0x2) = (counter & 0xFFFF);
	*(interval_timer_ptr + 0x3) = (counter >> 16) & 0xFFFF;

	/* start interval timer, enable its interrupts */
	*(interval_timer_ptr + 1) = 0x7;	// STOP = 0, START = 1, CONT = 1, ITO = 1 
}

/* setup the KEY interrupts in the FPGA */
void config_KEYs()
{
	volatile int * KEY_ptr = (int *) KEY_BASE;	// pushbutton KEY address

	*(KEY_ptr + 2) = 0xF; 	// enable interrupts for all four KEYs
}
