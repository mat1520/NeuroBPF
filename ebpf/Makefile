CLANG ?= clang
BPFTOOL ?= bpftool
CC ?= cc

BPF_CFLAGS := -g -O2 -Wall -Werror -Wno-missing-declarations -D__TARGET_ARCH_x86 -I/usr/include/bpf
CFLAGS := -g -O2 -Wall -Werror

all: collector

collect: collector

vmlinux.h:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

collector.bpf.o: collector.bpf.c collector.h vmlinux.h
	$(CLANG) $(BPF_CFLAGS) -target bpf -c collector.bpf.c -o collector.bpf.o

collector.skel.h: collector.bpf.o
	$(BPFTOOL) gen skeleton collector.bpf.o > collector.skel.h

collector: collector.c collector.h collector.skel.h
	$(CC) $(CFLAGS) -I. -o collector collector.c -lbpf -lelf -lz

clean:
	rm -f collector collector.bpf.o collector.skel.h vmlinux.h

.PHONY: all collect clean