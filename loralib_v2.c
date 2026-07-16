#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <math.h>
#include <stdint.h>

int serial_fd = -1;

// Lets the caller (Python via ctypes) hand in an already-opened file descriptor for the
// serial port, since this library does not open the port itself.
void setSerialFd(int fd) {
    serial_fd = fd;
}

void sendRaw(const char* cmd){
    if (serial_fd != -1) {
        write(serial_fd, cmd, strlen(cmd));
        write(serial_fd, "\r\n", 2);
        usleep(500000); // wait 500ms for the module to ack the AT command (NOT for the LoRa TX itself)
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

/*
 * Packet format (3 bytes total):
 *   Byte 0: bits [7:6] = device_id (0=Vac_Ext, 1=Prntr_3D, 2=Lazer_60W)
 *           bit  [5]   = on_off state (1=ON, 0=OFF)
 *           bits [4:0] = reserved (always 0 for now)
 *   Byte 1-2: pulse_count, uint16 big-endian (0-65535 pulses in the 1-min window)
 *
 * device_id mapping is fixed here, not derived from the GPIO pin number directly,
 * so the over-the-air format stays stable even if wiring/pins change later.
 */
#define DEV_VAC_EXT    0
#define DEV_PRNTR_3D   1
#define DEV_LAZER_60W  2

void encodePacket(uint8_t device_id, uint8_t on_off, uint16_t pulse_count, uint8_t out[3]) {
    out[0] = (uint8_t)(((device_id & 0x03) << 6) | ((on_off & 0x01) << 5));
    out[1] = (uint8_t)(pulse_count >> 8);
    out[2] = (uint8_t)(pulse_count & 0xFF);
}

// Sends one 3-byte packet as an AT+MSGHEX uplink. Returns 0 on success, -1 if port not open.
// This call blocks for ~500ms (the AT-command ack wait in sendRaw), NOT 5 seconds like the
// old per-value loop did -- there is only ever one AT+MSGHEX call per invocation now.
int sendLoraPacket(uint8_t device_id, uint8_t on_off, uint16_t pulse_count) {
    if (serial_fd == -1) {
        return -1;
    }

    uint8_t pkt[3];
    encodePacket(device_id, on_off, pulse_count, pkt);

    char hexBuffer[8];      // 3 bytes -> 6 hex chars + null terminator
    char finalCommand[32];

    snprintf(hexBuffer, sizeof(hexBuffer), "%02X%02X%02X", pkt[0], pkt[1], pkt[2]);
    snprintf(finalCommand, sizeof(finalCommand), "AT+MSGHEX=\"%s\"", hexBuffer);

    printf("Uplink: dev=%u on_off=%u pulses=%u -> %s\n",
           device_id, on_off, pulse_count, finalCommand);

    sendRaw(finalCommand);
    return 0;
}
