#ifndef EBpf_COLLECTOR_H
#define EBpf_COLLECTOR_H

#define EVENT_PATH_MAX 256
#define EVENT_COMM_MAX 16
#define EVENT_ADDR_MAX 16

enum event_kind {
    EVENT_PROCESS_START = 1,
    EVENT_NET_CONNECT = 2,
    EVENT_NET_ACCEPT = 3,
    EVENT_FILE_READ = 4,
    EVENT_FILE_WRITE = 5,
    EVENT_PROCESS_EXIT = 6,
};

struct event {
    unsigned long long ts_ns;
    unsigned int pid;
    unsigned int ppid;
    unsigned int uid;
    int result;
    unsigned int kind;
    unsigned short family;
    unsigned short port;
    unsigned char addr[EVENT_ADDR_MAX];
    char comm[EVENT_COMM_MAX];
    char path[EVENT_PATH_MAX];
};

#endif