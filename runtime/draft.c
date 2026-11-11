#include "draft.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;

struct plamo_draft { pid_t pid; int in, out; };
struct reply { uint32_t ok, count, ids[PLAMO_DRAFT_MAX]; };

static int transfer(int fd, void *data, size_t bytes, int writing, int timeout_ms) {
    char *p = data;
    while (bytes) {
        struct pollfd ready = {fd, writing ? POLLOUT : POLLIN, 0};
        int n;
        do n = poll(&ready, 1, timeout_ms); while (n < 0 && errno == EINTR);
        if (n <= 0) return 0;
        ssize_t done = writing ? write(fd, p, bytes) : read(fd, p, bytes);
        if (done < 0 && errno == EINTR) continue;
        if (done <= 0) return 0;
        p += done; bytes -= (size_t)done;
    }
    return 1;
}

static int exchange(struct plamo_draft *d, uint32_t op, uint32_t value, uint32_t count,
                    const char *text, struct reply *reply) {
    uint32_t header[] = {op, value, count, text ? (uint32_t)strlen(text) : 0};
    return d && transfer(d->in, header, sizeof header, 1, 30000) &&
        (!header[3] || transfer(d->in, (void *)text, header[3], 1, 30000)) &&
        // Prompt processing can exceed a decode round by orders of magnitude.
        transfer(d->out, reply, sizeof *reply, 0, op == 1 ? 300000 : 30000) &&
        reply->ok == 1 && reply->count <= PLAMO_DRAFT_MAX;
}

void plamo_draft_close(struct plamo_draft *d) {
    if (!d) return;
    close(d->in); close(d->out);
    // Closing stdin normally terminates an idle peer. Do not leave a GPU model
    // behind if a failed peer is stuck in a driver or a malformed request.
    if (!waitpid(d->pid, NULL, WNOHANG)) { kill(d->pid, SIGTERM); waitpid(d->pid, NULL, 0); }
    free(d);
}

struct plamo_draft *plamo_draft_open(void) {
    const char *enabled = getenv("PLAMO_SPECULATIVE");
    if (enabled && !strcmp(enabled, "0")) return NULL;
    char executable[PATH_MAX], resolved[PATH_MAX], root[PATH_MAX];
    uint32_t size = sizeof executable;
    if (_NSGetExecutablePath(executable, &size) || !realpath(executable, resolved)) return NULL;
    char *slash = strrchr(resolved, '/');
    if (!slash) return NULL;
    *slash = 0;
    const char *worker = getenv("PLAMO_DRAFT_WORKER");
    if (worker) {
        if (!realpath(worker, executable)) return NULL;
    } else {
        if (snprintf(executable, sizeof executable, "%s/plamo-worker-0.8b", resolved) >= (int)sizeof executable) return NULL;
        if (access(executable, X_OK)) {
            if (snprintf(executable, sizeof executable, "%s/0.8b/plamo-worker", resolved) >= (int)sizeof executable || access(executable, X_OK)) return NULL;
        }
    }
    const char *data = getenv("PLAMO_DRAFT_DATA_DIR");
    if (data) {
        if (!realpath(data, root)) return NULL;
    } else if (!access("models/0.8b/tokenizer.jsonl", R_OK)) {
        if (!getcwd(root, sizeof root)) return NULL;
    } else {
        const char *xdg = getenv("XDG_DATA_HOME");
        const char *home = getenv("HOME");
        if (xdg && *xdg) {
            if (snprintf(root, sizeof root, "%s/plamo-3-translate-0.8b", xdg) >= (int)sizeof root) return NULL;
        } else if (!home || snprintf(root, sizeof root, "%s/.local/share/plamo-3-translate-0.8b", home) >= (int)sizeof root) return NULL;
    }
    int input[2], output[2];
    if (pipe(input)) return NULL;
    if (pipe(output)) { close(input[0]); close(input[1]); return NULL; }
    for (unsigned i = 0; i < 2; ++i) { fcntl(input[i], F_SETFD, FD_CLOEXEC); fcntl(output[i], F_SETFD, FD_CLOEXEC); }
    fcntl(input[1], F_SETNOSIGPIPE, 1);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, input[1]);
    posix_spawn_file_actions_addclose(&actions, output[0]);
    // SDK 15 provides the extension; SDK 26 renamed it without _np.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    posix_spawn_file_actions_addchdir_np(&actions, root);
#pragma clang diagnostic pop
    /* A partially retained proposer cannot amortize its I/O. The small Q4
       peer must keep its projections available for every draft step. */
    char *args[] = {executable, "--draft-peer", "--backend", "metal", "--memory", "resident", "--threads", "1", NULL};
    pid_t pid;
    int status = posix_spawn(&pid, executable, &actions, NULL, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(input[0]); close(output[1]);
    if (status) { close(input[1]); close(output[0]); return NULL; }
    struct plamo_draft *d = malloc(sizeof *d);
    if (!d) { close(input[1]); close(output[0]); kill(pid, SIGTERM); waitpid(pid, NULL, 0); return NULL; }
    *d = (struct plamo_draft){pid, input[1], output[0]};
    uint32_t ready[4];
    if (!transfer(d->out, ready, sizeof ready, 0, 30000) || ready[0] != PLAMO_DRAFT_MAGIC ||
        ready[1] != 1 || ready[2] != 107520 || ready[3] != PLAMO_DRAFT_MAX) {
        plamo_draft_close(d); return NULL;
    }
    return d;
}

int plamo_draft_reset(struct plamo_draft *d, const char *text, unsigned direction) {
    struct reply reply;
    return exchange(d, 1, direction, 0, text, &reply) && !reply.count;
}
unsigned plamo_draft_propose(struct plamo_draft *d, uint32_t first, unsigned limit, uint32_t *ids) {
    struct reply reply;
    if (!limit || limit > PLAMO_DRAFT_MAX || !exchange(d, 2, first, limit, NULL, &reply) ||
        !reply.count || reply.count > limit || reply.ids[0] != first) return 0;
    for (unsigned i = 0; i < reply.count; ++i) if (reply.ids[i] >= 107520) return 0;
    memcpy(ids, reply.ids, reply.count * sizeof *ids);
    return reply.count;
}
int plamo_draft_accept(struct plamo_draft *d, unsigned consumed) {
    struct reply reply;
    return exchange(d, 3, consumed, 0, NULL, &reply) && !reply.count;
}

struct saved_layer { struct mixer *mixer; uint64_t count; float *rows; };
struct plamo_draft_checkpoint { unsigned limit; uint64_t count; struct saved_layer layers[]; };

void plamo_draft_checkpoint_free(struct plamo_draft_checkpoint *s) {
    if (!s) return;
    for (uint64_t i = 0; i < s->count; ++i) free(s->layers[i].rows);
    free(s);
}
struct plamo_draft_checkpoint *plamo_draft_checkpoint(struct model *model, unsigned limit) {
    if (!limit || limit > PLAMO_DRAFT_MAX) return NULL;
    struct plamo_draft_checkpoint *s = calloc(1, sizeof *s + model->count * sizeof *s->layers);
    if (!s) return NULL;
    s->limit = limit; s->count = model->count;
    for (uint64_t i = 0; i < s->count; ++i) {
        struct mixer *m = model->layers[i]->mixer;
        struct cache *c = m->cache;
        if (c->capacity < limit || c->count > c->capacity) goto fail;
        uint64_t rows = c->count < limit ? c->count : limit;
        float *copy = malloc((size_t)(2 * rows * c->width * sizeof(float)) + 1);
        if (!copy) goto fail;
        s->layers[i] = (struct saved_layer){m, c->count, copy};
        memcpy(copy, c->k, (size_t)(rows * c->width * 4));
        memcpy(copy + rows * c->width, c->v, (size_t)(rows * c->width * 4));
    }
    return s;
fail:
    plamo_draft_checkpoint_free(s); return NULL;
}
int plamo_draft_restore(struct plamo_draft_checkpoint *s, unsigned processed, unsigned accepted) {
    if (!s || accepted > processed || processed > s->limit) return 0;
    for (uint64_t i = 0; i < s->count; ++i) {
        const struct saved_layer *l = &s->layers[i];
        struct cache *c = l->mixer->cache;
        uint64_t full = l->count + processed, keep = l->count + accepted;
        if ((!c->sliding && full > c->capacity) || c->count != (full < c->capacity ? full : c->capacity)) return 0;
        uint64_t lost = full > c->capacity ? full - c->capacity : 0;
        uint64_t skip = keep > c->capacity ? keep - c->capacity : 0;
        uint64_t delta = lost - skip, desired = keep - skip;
        uint64_t saved_rows = l->count < s->limit ? l->count : s->limit;
        if (lost > saved_rows || delta > desired) return 0;
    }
    for (uint64_t i = 0; i < s->count; ++i) {
        const struct saved_layer *l = &s->layers[i];
        struct cache *c = l->mixer->cache;
        uint64_t full = l->count + processed, keep = l->count + accepted;
        uint64_t lost = full > c->capacity ? full - c->capacity : 0;
        uint64_t skip = keep > c->capacity ? keep - c->capacity : 0;
        uint64_t delta = lost - skip, desired = keep - skip;
        uint64_t saved_rows = l->count < s->limit ? l->count : s->limit;
        if (delta) {
            size_t moved = (size_t)((desired - delta) * c->width * 4), restored = (size_t)(delta * c->width * 4);
            memmove(c->k + delta * c->width, c->k, moved);
            memmove(c->v + delta * c->width, c->v, moved);
            memcpy(c->k, l->rows + skip * c->width, restored);
            memcpy(c->v, l->rows + (saved_rows + skip) * c->width, restored);
        }
        c->count = desired;
        l->mixer->rotation->processed_count = 0;
        l->mixer->rotation->scale_bits = 0;
        l->mixer->gqa->tokens = desired;
    }
    return 1;
}
