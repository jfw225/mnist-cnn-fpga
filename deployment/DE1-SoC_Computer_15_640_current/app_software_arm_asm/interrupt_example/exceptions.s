				.include	"address_map_arm.s"
				.include "defines.s"
				.include	"interrupt_ID.s"
/* This file:
 * 1. defines exception vectors for the A9 processor
 * 2. provides code that initializes the generic interrupt controller
*/

/*--- IRQ ---------------------------------------------------------------------*/
				.global	SERVICE_IRQ
SERVICE_IRQ:
    			PUSH		{R0-R7, LR}
    
    			/* Read the ICCIAR from the CPU interface */
    			LDR		R4, =MPCORE_GIC_CPUIF
    			LDR		R5, [R4, #ICCIAR] 			// read the interrupt ID

HPS_TIMER_CHECK:
    			CMP		R5, #HPS_TIMER0_IRQ			// check for HPS timer interrupt
    			BNE		INTERVAL_TIMER_CHECK    
    
    			BL			HPS_TIMER_ISR
	 			B			EXIT_IRQ
    
INTERVAL_TIMER_CHECK:
    			CMP		R5, #INTERVAL_TIMER_IRQ		// check for FPGA timer interrupt
    			BNE		KEYS_CHECK    
    
    			BL			TIMER_ISR
	 			B			EXIT_IRQ
    
KEYS_CHECK:
    			CMP		R5, #KEYS_IRQ
UNEXPECTED:	BNE		UNEXPECTED    					// if not recognized, stop here
    
    			BL			KEY_ISR
EXIT_IRQ:
    			/* Write to the End of Interrupt Register (ICCEOIR) */
    			STR		R5, [R4, #ICCEOIR]
    
    			POP		{R0-R7, LR}
    			SUBS		PC, LR, #4


/*--- Undefined instructions --------------------------------------------------*/
				.global	SERVICE_UND
SERVICE_UND:
    			B			SERVICE_UND 
 
/*--- Software interrupts -----------------------------------------------------*/
				.global	SERVICE_SVC
SERVICE_SVC:				
    			B			SERVICE_SVC 

/*--- Aborted data reads ------------------------------------------------------*/
				.global	SERVICE_ABT_DATA
SERVICE_ABT_DATA:
    			B			SERVICE_ABT_DATA 

/*--- Aborted instruction fetch -----------------------------------------------*/
				.global	SERVICE_ABT_INST
SERVICE_ABT_INST:
    			B			SERVICE_ABT_INST 
 
/*--- FIQ ---------------------------------------------------------------------*/
				.global	SERVICE_FIQ
SERVICE_FIQ:
    			B			SERVICE_FIQ 

/* 
 * Configure the Generic Interrupt Controller (GIC)
*/
				.global	CONFIG_GIC
CONFIG_GIC:
    			/* configure the HPS timer interrupt */
	 			LDR		R0, =0xFFFED8C4		// ICDIPTRn: processor targets register
				LDR		R1, =0x01000000		// set target to cpu0
				STR		R1, [R0]

				LDR		R0, =0xFFFED118		// ICDISERn: set enable register
				LDR		R1, =0x00000080		// set interrupt enable
				STR		R1, [R0]
    
    			/* configure the FPGA IRQ0 (interval timer) and IRQ1 (KEYs) interrupts */
	 			LDR		R0, =0xFFFED848		// ICDIPTRn: processor targets register
				LDR		R1, =0x00000101		// set targets to cpu0
				STR		R1, [R0]

				LDR		R0, =0xFFFED108		// ICDISERn: set enable register
				LDR		R1, =0x00000300		// set interrupt enable
				STR		R1, [R0]
    
				/* configure the GIC CPU interface */
    			LDR		R0, =MPCORE_GIC_CPUIF	// base address of CPU interface
    			/* Set Interrupt Priority Mask Register (ICCPMR) */
    			LDR		R1, =0xFFFF 	// 0xFFFF enables interrupts of all priorities levels */
    			STR		R1, [R0, #ICCPMR]
    			/* Set the enable bit in the CPU Interface Control Register (ICCICR). This bit
				 * allows interrupts to be forwarded to the CPU(s) */
    			MOV		R1, #ENABLE
    			STR		R1, [R0, #ICCICR]
    
    			/* Set the enable bit in the Distributor Control Register (ICDDCR). This bit
				 * allows the distributor to forward interrupts to the CPU interface(s) */
    			LDR		R0, =MPCORE_GIC_DIST
    			STR		R1, [R0, #ICDDCR]    
    
    			BX			LR

				.end
