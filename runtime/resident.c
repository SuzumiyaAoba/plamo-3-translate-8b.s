#include "resident.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <mach/mach.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

uint64_t plamo_reclaimable_memory(void) {
    vm_statistics64_data_t info;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    kern_return_t result = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&info, &count);
    mach_port_deallocate(mach_task_self(), host);
    long page = sysconf(_SC_PAGESIZE);
    if (result != KERN_SUCCESS || page <= 0) return 0;
    return ((uint64_t)info.free_count + info.external_page_count) * (uint64_t)page;
}

uint64_t plamo_physical_memory(void) {
    host_basic_info_data_t info;
    mach_msg_type_number_t count = HOST_BASIC_INFO_COUNT;
    mach_port_t host = mach_host_self();
    kern_return_t result = host_info(host, HOST_BASIC_INFO, (host_info_t)&info, &count);
    mach_port_deallocate(mach_task_self(), host);
    return result == KERN_SUCCESS ? info.max_mem : 0;
}

static void *rebase(void *pointer, const struct plamo_mapping *from, const struct plamo_mapping *to, size_t count) {
    uintptr_t address = (uintptr_t)pointer;
    for (size_t i = 0; i < count; ++i) {
        uintptr_t base = (uintptr_t)from[i].base;
        if (to[i].base && to[i].bytes >= from[i].bytes && address >= base && address - base < from[i].bytes)
            return (char *)to[i].base + (address - base);
    }
    return NULL;
}

int plamo_rebase_weights(struct plamo_weight_spec *specs, const struct plamo_mapping *from,
                         const struct plamo_mapping *to, size_t count) {
    for (struct plamo_weight_spec *s = specs; s->name; ++s)
        if (!s->slot || !rebase(*s->slot, from, to, count)) return 0;
    for (struct plamo_weight_spec *s = specs; s->name; ++s)
        *s->slot = rebase(*s->slot, from, to, count);
    return 1;
}

void plamo_resident_release(struct plamo_mapping *maps, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (maps[i].base) munmap(maps[i].base, (size_t)maps[i].bytes);
        maps[i] = (struct plamo_mapping){0};
    }
}

int plamo_resident_copy(const struct plamo_mapping *source, const int64_t *fds, size_t count,
                        uint64_t reserve, struct plamo_mapping *out, char *error, size_t cap) {
    uint64_t remaining = 0;
    memset(out, 0, count * sizeof(*out));
    for (size_t i = 0; i < count; ++i) {
        if (!source[i].base || !source[i].bytes || source[i].bytes > SIZE_MAX ||
            fds[i] < 0 || fds[i] > INT_MAX || UINT64_MAX - remaining < source[i].bytes) {
            snprintf(error, cap, "invalid source mapping or file descriptor");
            return 0;
        }
        remaining += source[i].bytes;
    }
    if (reserve) {
        uint64_t available = plamo_reclaimable_memory();
        if (remaining > UINT64_MAX - reserve) {
            snprintf(error, cap, "weight size plus memory reserve overflows");
            return 0;
        }
        if (available < remaining + reserve) {
            snprintf(error, cap, "BF16 preload needs %" PRIu64 " bytes including reserve; reclaimable estimate is %" PRIu64 " bytes", remaining + reserve, available);
            return 0;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        size_t size = (size_t)source[i].bytes;
        void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (base == MAP_FAILED) {
            snprintf(error, cap, "weight allocation failed: %s", strerror(errno));
            goto fail;
        }
        out[i] = (struct plamo_mapping){base, size};
        int fd = (int)fds[i];
        /* Avoid keeping a second, file-cache copy of a complete BF16 model. */
        (void)fcntl(fd, F_NOCACHE, 1);
        size_t offset = 0;
        while (offset < size) {
            if (reserve && !(offset % (256 * 1024 * 1024)) && plamo_reclaimable_memory() < remaining + reserve) {
                snprintf(error, cap, "memory pressure increased while preloading weights");
                (void)fcntl(fd, F_NOCACHE, 0);
                goto fail;
            }
            size_t bytes = size - offset;
            if (bytes > 8 * 1024 * 1024) bytes = 8 * 1024 * 1024;
            ssize_t n = pread(fd, (char *)base + offset, bytes, (off_t)offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                snprintf(error, cap, "weight read failed: %s", n ? strerror(errno) : "unexpected end of file");
                (void)fcntl(fd, F_NOCACHE, 0);
                goto fail;
            }
            offset += (size_t)n;
            remaining -= (uint64_t)n;
        }
        (void)fcntl(fd, F_NOCACHE, 0);
        if (mprotect(base, size, PROT_READ)) {
            snprintf(error, cap, "cannot protect preloaded weights: %s", strerror(errno));
            goto fail;
        }
    }
    return 1;
fail:
    plamo_resident_release(out, count);
    return 0;
}
