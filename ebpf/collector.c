#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <bpf/libbpf.h>

#include "collector.h"
#include "collector.skel.h"

static volatile sig_atomic_t g_stop;

static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

static const char *event_name(unsigned int kind)
{
    switch (kind) {
    case EVENT_PROCESS_START:
        return "process_start";
    case EVENT_NET_CONNECT:
        return "net_connect";
    case EVENT_NET_ACCEPT:
        return "net_accept";
    case EVENT_FILE_READ:
        return "file_read";
    case EVENT_FILE_WRITE:
        return "file_write";
    case EVENT_PROCESS_EXIT:
        return "process_exit";
    default:
        return "unknown";
    }
}

static void print_escaped(const char *s)
{
    const unsigned char *p = (const unsigned char *)s;
    putchar('"');
    for (; *p; p++) {
        switch (*p) {
        case '"':
            fputs("\\\"", stdout);
            break;
        case '\\':
            fputs("\\\\", stdout);
            break;
        case '\n':
            fputs("\\n", stdout);
            break;
        case '\r':
            fputs("\\r", stdout);
            break;
        case '\t':
            fputs("\\t", stdout);
            break;
        default:
            if (*p < 0x20)
                printf("\\u%04x", *p);
            else
                putchar(*p);
        }
    }
    putchar('"');
}

static void format_target(char *out, size_t out_sz, const struct event *ev)
{
    if (ev->family == AF_INET) {
        snprintf(out, out_sz, "%u.%u.%u.%u:%u", ev->addr[0], ev->addr[1],
                 ev->addr[2], ev->addr[3], ev->port);
    } else {
        int n = snprintf(out, out_sz, "%02x%02x:%02x%02x:%02x%02x:%02x%02x:"
                         "%02x%02x:%02x%02x:%02x%02x:%02x%02x:%u",
                         ev->addr[0], ev->addr[1], ev->addr[2], ev->addr[3],
                         ev->addr[4], ev->addr[5], ev->addr[6], ev->addr[7],
                         ev->addr[8], ev->addr[9], ev->addr[10], ev->addr[11],
                         ev->addr[12], ev->addr[13], ev->addr[14], ev->addr[15],
                         ev->port);
        if (n < 0)
            snprintf(out, out_sz, "unknown:%u", ev->port);
    }
}

static void emit_json(const struct event *ev)
{
    char target[80];
    unsigned long long secs = ev->ts_ns / 1000000000ull;
    unsigned int millis = (ev->ts_ns / 1000000ull) % 1000ull;

    printf("{\"ts\":%llu.%03u,\"event\":\"%s\",\"pid\":%u,\"ppid\":",
           secs, millis, event_name(ev->kind), ev->pid);
    if (ev->ppid == UINT_MAX)
        printf("-1");
    else
        printf("%u", ev->ppid);
    printf(",\"comm\":");
    print_escaped(ev->comm);
    printf(",\"exe_path\":");
    if (ev->kind == EVENT_PROCESS_START)
        print_escaped(ev->path);
    else
        print_escaped("");
    printf(",\"uid\":%u,\"target_type\":", ev->uid);
    if (ev->kind == EVENT_NET_CONNECT || ev->kind == EVENT_NET_ACCEPT) {
        printf("\"net\",\"target\":");
        format_target(target, sizeof(target), ev);
        print_escaped(target);
    } else if (ev->kind == EVENT_FILE_READ || ev->kind == EVENT_FILE_WRITE) {
        printf("\"file\",\"target\":");
        print_escaped(ev->path);
    } else {
        printf("null,\"target\":null");
    }
    printf(",\"result\":%d}\n", ev->result);
}

static int on_event(void *ctx, void *data, size_t sz)
{
    (void)ctx;
    if (sz < sizeof(struct event))
        return 0;
    emit_json((const struct event *)data);
    return 0;
}

static int set_print(enum libbpf_print_level level, const char *fmt, va_list args)
{
    if (level <= LIBBPF_WARN)
        vfprintf(stderr, fmt, args);
    return 0;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    struct rlimit rl = {RLIM_INFINITY, RLIM_INFINITY};
    struct collector_bpf *skel = NULL;
    struct ring_buffer *rb = NULL;
    int err = 0;

    if (setrlimit(RLIMIT_MEMLOCK, &rl))
        fprintf(stderr, "warning: setrlimit(RLIMIT_MEMLOCK) failed: %s\n",
                strerror(errno));

    libbpf_set_print(set_print);

    skel = collector_bpf__open();
    if (!skel) {
        fprintf(stderr, "failed to open BPF skeleton\n");
        return 1;
    }

    err = collector_bpf__load(skel);
    if (err) {
        fprintf(stderr, "failed to load BPF skeleton: %s\n", strerror(-err));
        goto out;
    }

    err = collector_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "failed to attach BPF programs: %s\n", strerror(-err));
        goto out;
    }

    rb = ring_buffer__new(bpf_map__fd(skel->maps.rb), on_event, NULL, NULL);
    if (!rb) {
        fprintf(stderr, "failed to create ring buffer\n");
        err = 1;
        goto out;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    setvbuf(stdout, NULL, _IOLBF, 0);

    fprintf(stderr, "collector: tracing syscalls, press Ctrl-C to stop\n");

    while (!g_stop) {
        int n = ring_buffer__poll(rb, 200);
        if (n < 0 && errno != EINTR) {
            fprintf(stderr, "ring buffer poll error: %s\n", strerror(errno));
            err = 1;
            break;
        }
    }

out:
    ring_buffer__free(rb);
    collector_bpf__destroy(skel);
    return err ? 1 : 0;
}