#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <math.h>
#include <stdint.h>


#define REAL_RES 16
#define DEC_RES 7

//open the port
int serial_fd = -1;

void sendRaw(const char* cmd){
    if (serial_fd != -1) {
        write(serial_fd, cmd, strlen(cmd));
        write(serial_fd, "\r\n", 2);
        usleep(500000); // wait 500ms for the respond
    }
}

int configLora() {

    struct termios options;
    tcgetattr(serial_fd, &options);
    cfsetispeed(&options, B9600);
    cfsetospeed(&options, B9600);
    options.c_cflag |= (CLOCAL | CREAD | CS8);
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tcsetattr(serial_fd, TCSANOW, &options);

    sendRaw("AT+MODE=LWABP");
    sendRaw("AT+ID=DevAddr,\"745008B0\"");
    sendRaw("AT+KEY=NWKSKEY,\"3B66B743121308C7B44671D775AC4987\"");
    sendRaw("AT+KEY=APPSKEY,\"C53B979FDDEE6781F4A570221EDC713A\"");
    sendRaw("AT+DR=DR5");
    sendRaw("AT+CH=NUM,0");

    return 0;
}

// Encode float to integer packet
uint32_t encode(float f) {
    uint16_t real_part = (uint16_t)f;
    uint8_t dec_part = (uint8_t)round((f - real_part) * 100);

    // pack bits: [real_part << DEC_RES] | dec_part
    uint32_t packet = ((uint32_t)real_part << DEC_RES) | dec_part;

    return packet;
}

int sendLoraData(float val1, float val2, float val3, float val4, float val5){
    if(serial_fd!=-1){
        float tabValue[5] = {val1,val2,val3,val4,val5};
        uint32_t tabEncode[5]={0,0,0,0,0};
        for (int i = 0; i<5; i++){
            char hexBuffer[10];
            char finalCommand[32];
            tabEncode[i] = encode(tabValue[i]);
            sprintf(hexBuffer, "%06lX", (unsigned long)tabEncode[i]);
            snprintf(finalCommand, sizeof(finalCommand), "AT+MSGHEX=\"%s\"", hexBuffer);
            printf("Envoi du message %d : %s\n", i + 1, finalCommand);
            sendRaw(finalCommand);
            usleep(5000000); // 5s
        }
        return 0;
    }else{
        return -1;
    }
}
