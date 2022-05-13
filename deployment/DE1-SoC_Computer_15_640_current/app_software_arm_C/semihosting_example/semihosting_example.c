#include <stdio.h>
#include <time.h>

/* This program demonstrates use of functions in C Standard Library through semihosting of Arm A9.
 * It performs the following: 
 * 	1. prints characters to the terminal of the host computer 
 * 	2. reads characters from host computer to target system
*/
int main(void){
    
    char str[64];
    int age = 0;
    
    // Get the initial timestamp
    clock_t start_time = clock();
    
    // Printf, scanf, and some time functions
    while(1){
        printf("What is your name?\n");
        scanf("%s",str);
        printf("What is your age?\n");
        scanf("%d",&age);
        
        printf("%s, you are %d years old.\n",str,age);
        printf("It's been %d seconds since we started\n",(int)(clock()-start_time)/CLOCKS_PER_SEC);
        printf("It's been %d seconds since 1970 Jan 1\n",(int)time(NULL));
    }
    
    return 0;
}