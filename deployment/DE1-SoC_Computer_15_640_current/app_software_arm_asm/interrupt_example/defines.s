			.equ		EDGE_TRIGGERED,         0x1
			.equ		LEVEL_SENSITIVE,        0x0
			.equ		CPU0,         				0x01	// bit-mask; bit 0 represents cpu0
			.equ		ENABLE, 						0x1

			.equ		KEY0, 						0
			.equ		KEY1, 						1
			.equ		KEY2,							2
			.equ		KEY3,							3
			.equ		NONE,							4

			.equ		RIGHT,						1
			.equ		LEFT,							2

			.equ		USER_MODE,					0b10000
			.equ		FIQ_MODE,					0b10001
			.equ		IRQ_MODE,					0b10010
			.equ		SVC_MODE,					0b10011
			.equ		ABORT_MODE,					0b10111
			.equ		UNDEF_MODE,					0b11011
			.equ		SYS_MODE,					0b11111

			.equ		INT_ENABLE,					0b01000000
			.equ		INT_DISABLE,				0b11000000
