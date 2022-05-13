#include "address_map_arm.h"
#include "interrupt_ID.h"
#include "defines.h"
/* This file:
 * 1. defines exception vectors for the A9 processor
 * 2. provides code that sets the IRQ mode stack, and that dis/enables interrupts
 * 3. provides code that initializes the generic interrupt controller
*/
void HPS_timer_ISR (void);
void interval_timer_ISR (void);
void pushbutton_ISR (void);

// Define the IRQ exception handler
void __attribute__ ((interrupt)) __cs3_isr_irq (void)
{
	// Read the ICCIAR from the processor interface 
	int address = MPCORE_GIC_CPUIF + ICCIAR; 
	int int_ID = *((int *) address); 
   
	if (int_ID == HPS_TIMER0_IRQ)				// check if interrupt is from the HPS timer
		HPS_timer_ISR ();
	else if (int_ID == INTERVAL_TIMER_IRQ)	// check if interrupt is from the Altera timer
		interval_timer_ISR ();
	else if (int_ID == KEYS_IRQ)				// check if interrupt is from the KEYs
		pushbutton_ISR ();
	else
		while (1);									// if unexpected, then stay here

	// Write to the End of Interrupt Register (ICCEOIR)
	address = MPCORE_GIC_CPUIF + ICCEOIR;
	*((int *) address) = int_ID;

	return;
} 

// Define the remaining exception handlers
void __attribute__ ((interrupt)) __cs3_reset (void)
{
    while(1);
}

void __attribute__ ((interrupt)) __cs3_isr_undef (void)
{
    while(1);
}

void __attribute__ ((interrupt)) __cs3_isr_swi (void)
{
    while(1);
}

void __attribute__ ((interrupt)) __cs3_isr_pabort (void)
{
    while(1);
}

void __attribute__ ((interrupt)) __cs3_isr_dabort (void)
{
    while(1);
}

void __attribute__ ((interrupt)) __cs3_isr_fiq (void)
{
    while(1);
}

/* 
 * Initialize the banked stack pointer register for IRQ mode
*/
void set_A9_IRQ_stack(void)
{
	int stack, mode;
	stack = A9_ONCHIP_END - 7;		// top of A9 onchip memory, aligned to 8 bytes
	/* change processor to IRQ mode with interrupts disabled */
	mode = INT_DISABLE | IRQ_MODE;
	asm("msr cpsr, %[ps]" : : [ps] "r" (mode));
	/* set banked stack pointer */
	asm("mov sp, %[ps]" : : [ps] "r" (stack));

	/* go back to SVC mode before executing subroutine return! */
	mode = INT_DISABLE | SVC_MODE;
	asm("msr cpsr, %[ps]" : : [ps] "r" (mode));
}

/* 
 * Turn on interrupts in the ARM processor
*/
void enable_A9_interrupts(void)
{
	int status = SVC_MODE | INT_ENABLE;
	asm("msr cpsr, %[ps]" : : [ps]"r"(status));
}

/* 
 * Configure the Generic Interrupt Controller (GIC)
*/
void config_GIC(void)
{
	int address;	// used to calculate register addresses

 	/* configure the HPS timer interrupt */
	*((int *) 0xFFFED8C4) = 0x01000000;
	*((int *) 0xFFFED118) = 0x00000080;

  	/* configure the FPGA interval timer and KEYs interrupts */
	*((int *) 0xFFFED848) = 0x00000101;
	*((int *) 0xFFFED108) = 0x00000300;
    
  	// Set Interrupt Priority Mask Register (ICCPMR). Enable interrupts of all priorities 
	address = MPCORE_GIC_CPUIF + ICCPMR;
  	*((int *) address) = 0xFFFF;       

  	// Set CPU Interface Control Register (ICCICR). Enable signaling of interrupts
	address = MPCORE_GIC_CPUIF + ICCICR;
	*((int *) address) = ENABLE;

	// Configure the Distributor Control Register (ICDDCR) to send pending interrupts to CPUs 
	address = MPCORE_GIC_DIST + ICDDCR;
	*((int *) address) = ENABLE;   
}
