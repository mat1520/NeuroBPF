#include "vmlinux.h"
#include <bpf_endian.h>
#include <bpf_helpers.h>
#include <bpf_tracing.h>
#include <bpf_core_read.h>
#include "collector.h"

char LICENSE[] SEC("license") = "GPL";

#define AF_INET 2
#define AF_INET6 10
#define O_RDWR 2
#define O_CREAT 64

struct sockaddr_in_view {
    __u16 family;
    __u16 port;
    __u32 addr;
};

struct sockaddr_in6_view {
    __u16 family;
    __u16 port;
    __u32 flowinfo;
    __u8 addr[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} rb SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);
    __type(value, unsigned long);
} connect_socks SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);
    __type(value, unsigned long);
} accept_socks SEC(".maps");

static __u32 fill_ppid(void)
{
    struct task_struct *task = bpf_get_current_task_btf();
    __u32 ppid = 0xffffffff;
    __u32 parent_tgid = 0;
    if (task && !bpf_core_read(&parent_tgid, sizeof(parent_tgid), &task->real_parent->tgid))
        ppid = parent_tgid;
    return ppid;
}

static void fill_common(struct event *ev)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u64 uid_gid = bpf_get_current_uid_gid();
    ev->ts_ns = bpf_ktime_get_ns();
    ev->pid = pid_tgid >> 32;
    ev->uid = uid_gid >> 32;
    ev->ppid = fill_ppid();
    bpf_get_current_comm(&ev->comm, sizeof(ev->comm));
}

static void fill_net(struct event *ev, unsigned long ptr)
{
    __u16 family = 0;
    if (!ptr || bpf_probe_read_user(&family, sizeof(family), (const void *)ptr))
        return;
    if (family == AF_INET) {
        struct sockaddr_in_view v = {};
        if (bpf_probe_read_user(&v, sizeof(v), (const void *)ptr))
            return;
        ev->family = AF_INET;
        ev->port = bpf_ntohs(v.port);
        ev->addr[0] = v.addr & 0xff;
        ev->addr[1] = (v.addr >> 8) & 0xff;
        ev->addr[2] = (v.addr >> 16) & 0xff;
        ev->addr[3] = (v.addr >> 24) & 0xff;
    } else if (family == AF_INET6) {
        struct sockaddr_in6_view v = {};
        if (bpf_probe_read_user(&v, sizeof(v), (const void *)ptr))
            return;
        ev->family = AF_INET6;
        ev->port = bpf_ntohs(v.port);
        __builtin_memcpy(ev->addr, v.addr, 16);
    }
}

static __u32 classify_open(unsigned long flags)
{
    if ((flags & 1) || (flags & O_RDWR) || (flags & O_CREAT))
        return EVENT_FILE_WRITE;
    return EVENT_FILE_READ;
}

static void emit_event(struct event *ev)
{
    struct event *dst = bpf_ringbuf_reserve(&rb, sizeof(*ev), 0);
    if (!dst)
        return;
    __builtin_memcpy(dst, ev, sizeof(*ev));
    bpf_ringbuf_submit(dst, 0);
}

SEC("tp/syscalls/sys_enter_execve")
int handle_execve(struct trace_event_raw_sys_enter *ctx)
{
    struct event ev = {};
    long r = bpf_probe_read_user_str(&ev.path, sizeof(ev.path), (const void *)ctx->args[0]);
    if (r <= 1)
        return 0;
    fill_common(&ev);
    ev.kind = EVENT_PROCESS_START;
    emit_event(&ev);
    return 0;
}

SEC("tp/syscalls/sys_enter_execveat")
int handle_execveat(struct trace_event_raw_sys_enter *ctx)
{
    struct event ev = {};
    long r = bpf_probe_read_user_str(&ev.path, sizeof(ev.path), (const void *)ctx->args[1]);
    if (r <= 1)
        return 0;
    fill_common(&ev);
    ev.kind = EVENT_PROCESS_START;
    emit_event(&ev);
    return 0;
}

SEC("tp/syscalls/sys_enter_open")
int handle_open(struct trace_event_raw_sys_enter *ctx)
{
    struct event ev = {};
    long r = bpf_probe_read_user_str(&ev.path, sizeof(ev.path), (const void *)ctx->args[0]);
    if (r <= 1)
        return 0;
    fill_common(&ev);
    ev.kind = classify_open(ctx->args[1]);
    emit_event(&ev);
    return 0;
}

SEC("tp/syscalls/sys_enter_openat")
int handle_openat(struct trace_event_raw_sys_enter *ctx)
{
    struct event ev = {};
    long r = bpf_probe_read_user_str(&ev.path, sizeof(ev.path), (const void *)ctx->args[1]);
    if (r <= 1)
        return 0;
    fill_common(&ev);
    ev.kind = classify_open(ctx->args[2]);
    emit_event(&ev);
    return 0;
}

SEC("tp/syscalls/sys_enter_connect")
int handle_connect(struct trace_event_raw_sys_enter *ctx)
{
    __u32 tid = bpf_get_current_pid_tgid() & 0xffffffff;
    unsigned long ptr = ctx->args[1];
    bpf_map_update_elem(&connect_socks, &tid, &ptr, BPF_ANY);
    return 0;
}

SEC("tp/syscalls/sys_exit_connect")
int handle_exit_connect(struct trace_event_raw_sys_exit *ctx)
{
    __u32 tid = bpf_get_current_pid_tgid() & 0xffffffff;
    unsigned long *ptr = bpf_map_lookup_elem(&connect_socks, &tid);
    unsigned long sockaddr;
    if (!ptr)
        return 0;
    sockaddr = *ptr;
    bpf_map_delete_elem(&connect_socks, &tid);
    if (ctx->ret != 0)
        return 0;
    struct event ev = {};
    fill_common(&ev);
    fill_net(&ev, sockaddr);
    if (!ev.family)
        return 0;
    ev.kind = EVENT_NET_CONNECT;
    emit_event(&ev);
    return 0;
}

static void do_accept_exit(struct trace_event_raw_sys_exit *ctx)
{
    __u32 tid = bpf_get_current_pid_tgid() & 0xffffffff;
    unsigned long *ptr = bpf_map_lookup_elem(&accept_socks, &tid);
    unsigned long sockaddr;
    if (!ptr)
        return;
    sockaddr = *ptr;
    bpf_map_delete_elem(&accept_socks, &tid);
    if (ctx->ret < 0)
        return;
    struct event ev = {};
    fill_common(&ev);
    fill_net(&ev, sockaddr);
    if (ev.family) {
        ev.kind = EVENT_NET_ACCEPT;
        emit_event(&ev);
    }
}

static void do_accept_enter(struct trace_event_raw_sys_enter *ctx)
{
    __u32 tid = bpf_get_current_pid_tgid() & 0xffffffff;
    unsigned long ptr = ctx->args[1];
    bpf_map_update_elem(&accept_socks, &tid, &ptr, BPF_ANY);
}

SEC("tp/syscalls/sys_enter_accept")
int handle_accept(struct trace_event_raw_sys_enter *ctx)
{
    do_accept_enter(ctx);
    return 0;
}

SEC("tp/syscalls/sys_enter_accept4")
int handle_accept4(struct trace_event_raw_sys_enter *ctx)
{
    do_accept_enter(ctx);
    return 0;
}

SEC("tp/syscalls/sys_exit_accept")
int handle_exit_accept(struct trace_event_raw_sys_exit *ctx)
{
    do_accept_exit(ctx);
    return 0;
}

SEC("tp/syscalls/sys_exit_accept4")
int handle_exit_accept4(struct trace_event_raw_sys_exit *ctx)
{
    do_accept_exit(ctx);
    return 0;
}

static void do_process_exit(void)
{
    struct event ev = {};
    fill_common(&ev);
    ev.kind = EVENT_PROCESS_EXIT;
    emit_event(&ev);
}

SEC("tp/syscalls/sys_exit_exit")
int handle_exit_syscall(struct trace_event_raw_sys_exit *ctx)
{
    (void)ctx;
    do_process_exit();
    return 0;
}

SEC("tp/syscalls/sys_exit_exit_group")
int handle_exit_group(struct trace_event_raw_sys_exit *ctx)
{
    (void)ctx;
    do_process_exit();
    return 0;
}