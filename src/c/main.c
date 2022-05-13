#include <stdio.h>
#include <stdlib.h>
#include <stdfix.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <math.h>

/* Image file */
#define IMAGE_FILE "./test.txt"

/* Cyclone V FPGA devices */
#define FPGA_LW_BASE 0xff200000
//#define FPGA_LW_SPAN        0x00200000
#define FPGA_LW_SPAN 0x00005000
#define FPGA_LW_OFFSET 0x00000010

#define WAIT for (_i = 0; _i < 1000; _i++)
int _i;

// /dev/mem file id
int fd;

// the light weight bus base
void *h2p_lw_virtual_base;

volatile unsigned int *hps_input_addr = NULL;
volatile unsigned int *hps_input_data = NULL;
volatile unsigned int *hps_fclk = NULL;
volatile unsigned int *hps_valid = NULL;
volatile unsigned int *fpga_done = NULL;
volatile unsigned int *fpga_output_data = NULL;
volatile unsigned int *hps_reset = NULL;
volatile unsigned int *fpga_ack = NULL;

int main(void)
{

    // === need to mmap: =======================
    // FPGA_CHAR_BASE
    // FPGA_ONCHIP_BASE
    // FPGA_LW_BASE

    // === get FPGA addresses ==================
    // Open /dev/mem
    if ((fd = open("/dev/mem", (O_RDWR | O_SYNC))) == -1)
    {
        printf("ERROR: could not open \"/dev/mem\"...\n");
        return (1);
    }

    // get virtual addr that maps to physical
    h2p_lw_virtual_base = mmap(NULL, FPGA_LW_SPAN, (PROT_READ | PROT_WRITE), MAP_SHARED, fd, FPGA_LW_BASE);
    if (h2p_lw_virtual_base == MAP_FAILED)
    {
        printf("ERROR: mmap1() failed...\n");
        close(fd);
        return (1);
    }

    // assign PIO connections
    hps_input_addr = (unsigned int *)(h2p_lw_virtual_base + 0x100);
    hps_input_data = (unsigned int *)(h2p_lw_virtual_base + 0x110);
    hps_fclk = (unsigned int *)(h2p_lw_virtual_base + 0x120);
    hps_valid = (unsigned int *)(h2p_lw_virtual_base + 0x130);
    fpga_done = (unsigned int *)(h2p_lw_virtual_base + 0x140);
    fpga_output_data = (unsigned int *)(h2p_lw_virtual_base + 0x150);
    hps_reset = (unsigned int *)(h2p_lw_virtual_base + 0x160);
    fpga_ack = (unsigned int *)(h2p_lw_virtual_base + 0x170);

    unsigned int id = 0;
    FILE *fp = fopen(IMAGE_FILE, "r");
    fscanf(fp, "%u", &id);
    fclose(fp);

    // hold reset high until first input
    *hps_reset = 1;

    while (1)
    {
        unsigned int address = 0;
        unsigned int number = 0;
        FILE *fp = fopen(IMAGE_FILE, "r");

        fscanf(fp, "%u", &number);
        // printf("%u\n", number);

        if (number == id)
        {
            fclose(fp);
            continue;
        };
        id = number;

        // set the input to invalid
        *hps_valid = 0;

        // reset the system
        *hps_reset = 1;
        WAIT;
        *hps_reset = 0;
        WAIT;

        printf("Reset Ack: %u\n", *fpga_ack);
        if (*fpga_ack != 1)
            return 0;

        // start sending data after active HIGH acknowledgement
        while (fscanf(fp, "%u", &number) > 0)
        {
            *hps_fclk = 0;
            WAIT;
            *hps_input_addr = address;
            *hps_input_data = number;
            WAIT;
            *hps_fclk = 1;
            WAIT;

            // printf("%u\n", number);
            address += 1;
        }
        *hps_fclk = 0;

        // send the input valid signal
        *hps_valid = 1;
        WAIT;

        printf("Valid Ack: %u\n", *fpga_ack);
        if (*fpga_ack != 0)
            return 0;

        // clear valid after active LOW acknowledgement
        *hps_valid = 0;

        // wait for the fpga to be d one
        while (*fpga_done != 1)
            ;

        // print inference after model is finished
        printf("Inference: %u\n", *fpga_output_data);

        fclose(fp);
    }

    return 0;
}