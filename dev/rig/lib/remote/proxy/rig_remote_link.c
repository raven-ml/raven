/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Jobs and links: this process's ends of its job's connections.

   A link owns its socket and two threads that never hold the OCaml
   runtime. The sending thread sends the link's queue, the transfers of its
   rails and a beat after a second without a send; it waits on nothing but
   its queue and its socket. The receiving thread reads frames, places
   rail transfers, answers requests, and queues an agent's commands; it
   measures silence on its own.

   A job fails once. Whoever finds the failure, a thread or a caller, sets
   the root cause under the job's lock, then for each link: marks it
   failed, raises its rails' counts, sends an abort if no frame is being
   sent, shuts the socket down and wakes every waiter. A link's socket
   closes once both its threads ended and no abort is being sent.

   Reasons are formatted here, because the threads that find a failure hold
   no runtime: a fixed phrase, the system's error, or a peer's abort, each
   after the link's name, in memory never freed.

   A child of fork sees its parent's job failed, in its own copy: every
   entry compares the process id with the job's before it takes a lock. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rig_remote_proxy_stubs.h"

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#include <ws2tcpip.h>
#define getpid _getpid
#define poll WSAPoll
typedef WSAPOLLFD rig_remote_pollfd;
#define close_sock closesocket
#define SHUT_BOTH SD_BOTH
#else
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <unistd.h>
typedef struct pollfd rig_remote_pollfd;
#define close_sock close
#define SHUT_BOTH SHUT_RDWR
#endif

/* A beat after a second without a send; a failure after ten without a
   byte. */
#define BEAT_NS 1000000000LL
#define SILENCE_MS 10000

/* The bytes a link's queue holds before its writers wait. One frame larger
   than this still goes, alone. */
#define QUEUE_BYTES ((size_t)8 << 20)

/* The most bytes of an abort's reason. */
#define MAX_WHY 4096

/* A rail's counts: [ready], [sent] and [arrived], each in a 128-byte line. */
#define READY 0
#define SENT 1
#define ARRIVED 2
#define COUNT_STRIDE 128


/* What a receive answers besides 0 and a socket error. */
enum { ENDED = -1, SILENT = -2, MALFORMED = -3, TOO_LARGE = -4 };

/* Structures */

struct entry {
  struct entry *next;
  unsigned char *buf; /* header and payload */
  size_t n;
};

struct pending {
  struct pending *next;
  int done, refused;
  unsigned char *p;
  size_t n;
};

struct cmd {
  struct cmd *next;
  int kind;
  unsigned char *p;
  size_t n;
};

struct rail {
  struct rig_remote_link *link;
  struct rail *next;
  uint64_t id;
  uint64_t *send, *receive; /* src, dst and length of each transfer */
  uint64_t nsend, nreceive;
  unsigned char *out, *in, *counts;
  size_t out_stride, in_stride;
  uint64_t posted; /* the sending thread's last count sent */
  uint64_t placed; /* the receiving thread's last count placed */
  int users;       /* threads using it outside the link's lock */
};

static pthread_mutex_t jobs_mu = PTHREAD_MUTEX_INITIALIZER;
static struct rig_remote_job *last_job;

#define Job_val(v) ((struct rig_remote_job *)Nativeint_val(v))
#define Link_val(v) ((struct rig_remote_link *)Nativeint_val(v))

/* Time */

static int64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

/* Waits on [cv] for at most [ns] nanoseconds. */
static void wait_ns(pthread_cond_t *cv, pthread_mutex_t *mu, int64_t ns) {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  int64_t at = (int64_t)t.tv_nsec + ns;
  t.tv_sec += (time_t)(at / 1000000000LL);
  t.tv_nsec = (long)(at % 1000000000LL);
  pthread_cond_timedwait(cv, mu, &t);
}

static void put_header(unsigned char *p, uint64_t n, int kind) {
  rig_remote_put_u64(p, n);
  p[8] = (unsigned char)kind;
}

static _Atomic uint64_t *count(struct rail *r, int which) {
  return (_Atomic uint64_t *)(r->counts + COUNT_STRIDE * which);
}

/* Reasons */

static struct rig_remote_why failed_why = {14, "the job failed"};
static struct rig_remote_why forked_why = {
    53, "a child of fork does not use its parent's connections"};

/* A reason of the [n] bytes at [p], cut to [MAX_WHY]; NULL if memory ran
   out. */
static struct rig_remote_why *why_of(const char *p, size_t n) {
  if (n > MAX_WHY) n = MAX_WHY;
  struct rig_remote_why *w = malloc(sizeof *w);
  char *s = w != NULL ? malloc(n + 1) : NULL;
  if (s == NULL) {
    free(w);
    return NULL;
  }
  memcpy(s, p, n);
  s[n] = 0;
  w->n = n;
  w->s = s;
  return w;
}

static void why_free(struct rig_remote_why *w) {
  if (w == NULL) return;
  free((char *)w->s);
  free(w);
}

static struct rig_remote_why *reason(const char *fmt, const char *name,
                                     const char *what) {
  char buf[MAX_WHY + 1];
  int n = snprintf(buf, sizeof buf, fmt, name, what);
  return why_of(buf, n < 0 ? 0 : (size_t)n);
}

static void error_text(int e, char *buf, size_t n) {
#if defined(_WIN32)
  if (FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                     NULL, (DWORD)e, 0, buf, (DWORD)n, NULL) == 0)
    snprintf(buf, n, "socket error %d", e);
#elif defined(__GLIBC__)
  char *s = strerror_r(e, buf, n);
  if (s != buf) snprintf(buf, n, "%s", s);
#else
  if (strerror_r(e, buf, n) != 0) snprintf(buf, n, "error %d", e);
#endif
}

/* The root cause of a link's failure [code]: a socket error, or one of the
   receive answers. */
static struct rig_remote_why *link_reason(struct rig_remote_link *l,
                                          int code) {
  char buf[256];
  switch (code) {
  case ENDED:
    return reason("%s: closed its connection%s", l->name, "");
  case SILENT:
    return reason("%s: silent for 10 s%s", l->name, "");
  case MALFORMED:
    return reason("%s: a malformed frame%s", l->name, "");
  case TOO_LARGE:
    return reason("%s: a frame larger than this process can hold%s", l->name,
                  "");
  default:
#ifndef _WIN32
    if (code == EPIPE || code == ECONNRESET)
      return reason("%s: closed its connection%s", l->name, "");
#endif
    error_text(code, buf, sizeof buf);
    return reason("%s: %s", l->name, buf);
  }
}

/* Failure */

static void free_entries(struct rig_remote_link *l) {
  while (l->head != NULL) {
    struct entry *e = l->head;
    l->head = e->next;
    free(e->buf);
    free(e);
  }
  l->tail = NULL;
  l->queued = 0;
}

/* Closes [l]'s socket once nothing uses it. Holds [l]'s lock. */
static void close_if_idle(struct rig_remote_link *l) {
  if (l->threads == 0 && !l->sending && !l->fd_closed) {
    l->fd_closed = 1;
    close_sock(l->fd);
  }
}

/* Sends an abort with [why], a string as wire.mli lays it out, if the
   stream takes it at once. */
static void try_abort(struct rig_remote_link *l,
                      const struct rig_remote_why *why) {
  unsigned char buf[HEADER + 4 + MAX_WHY];
  put_header(buf, 4 + why->n, K_ABORT);
  for (int i = 0; i < 4; i++)
    buf[HEADER + i] = (unsigned char)(why->n >> (8 * i));
  memcpy(buf + HEADER + 4, why->s, why->n);
#ifdef _WIN32
  /* Winsock has no per-call MSG_DONTWAIT: the socket turns non-blocking for
     good, as nothing sends on it after the abort. */
  u_long on = 1;
  (void)ioctlsocket(l->fd, FIONBIO, &on);
  (void)send(l->fd, (const char *)buf, (int)(HEADER + 4 + why->n), 0);
#else
  (void)send(l->fd, buf, HEADER + 4 + why->n,
             MSG_DONTWAIT | RIG_REMOTE_NOSIGNAL);
#endif
}

/* Fails [j] with [why], which it takes, unless it failed or closed. */
static void fail_job(struct rig_remote_job *j, struct rig_remote_why *why) {
  pthread_mutex_lock(&j->mu);
  if (atomic_load(&j->state) != OPEN) {
    pthread_mutex_unlock(&j->mu);
    why_free(why);
    return;
  }
  atomic_store(&j->why, why != NULL ? why : &failed_why);
  atomic_store(&j->state, FAILED);
  for (struct rig_remote_link *l = j->links; l != NULL; l = l->next) {
    pthread_mutex_lock(&l->mu);
    atomic_store(&l->failed, 1);
    free_entries(l);
    for (struct rail *r = l->rails; r != NULL; r = r->next)
      for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
    int claim = !l->sending && !l->sent_close && !l->fd_closed;
    if (claim) l->sending = 1;
    pthread_cond_broadcast(&l->cv);
    pthread_mutex_unlock(&l->mu);
    if (claim) try_abort(l, atomic_load(&j->why));
    pthread_mutex_lock(&l->mu);
    if (claim) l->sending = 0;
    if (!l->fd_closed) shutdown(l->fd, SHUT_BOTH);
    close_if_idle(l);
    pthread_mutex_unlock(&l->mu);
  }
  pthread_cond_broadcast(&j->cv);
  pthread_mutex_unlock(&j->mu);
}

/* Fails [l]'s job with the reason of [code], unless the job failed. */
static void link_lost(struct rig_remote_link *l, int code) {
  if (atomic_load(&l->failed)) return;
  fail_job(l->job, link_reason(l, code));
}

int rig_remote_forked(struct rig_remote_job *j) {
  if (j->pid == (long)getpid()) return 0;
  if (atomic_load(&j->state) == OPEN) {
    atomic_store(&j->why, &forked_why);
    atomic_store(&j->state, FAILED);
  }
  return 1;
}

/* Thread ends */

static void thread_ends(struct rig_remote_link *l) {
  struct rig_remote_job *j = l->job;
  pthread_mutex_lock(&j->mu);
  pthread_mutex_lock(&l->mu);
  l->threads--;
  close_if_idle(l);
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  pthread_cond_broadcast(&j->cv);
  pthread_mutex_unlock(&j->mu);
}

/* The sending thread */

/* Sends the transfers of [l]'s rails whose [ready] advanced. Holds [l]'s
   lock, and releases it while it sends. [1] if it sent, [0] if not, and
   [-1] if the link failed. */
static int send_rails(struct rig_remote_link *l) {
  int sent = 0;
  for (struct rail *r = l->rails; r != NULL; r = r->next) {
    if (r->nsend == 0) continue;
    uint64_t ready = atomic_load_explicit(count(r, READY), memory_order_acquire);
    if (ready <= r->posted) continue;
    r->users++;
    l->sending = 1;
    pthread_mutex_unlock(&l->mu);
    int err = 0;
    while (r->posted < ready && err == 0 && !atomic_load(&l->failed)) {
      uint64_t c = r->posted + 1;
      uint64_t j = (c - 1) % r->nsend, k = ((c - 1) / r->nsend) % 2;
      uint64_t *t = r->send + 3 * j;
      unsigned char h[HEADER + 16];
      put_header(h, 16 + t[2], K_RAIL);
      rig_remote_put_u64(h + HEADER, r->id);
      rig_remote_put_u64(h + HEADER + 8, c);
      err = rig_remote_send_all(l->fd, h, sizeof h);
      if (err == 0)
        err = rig_remote_send_all(l->fd, r->out + k * r->out_stride + t[0],
                                  (size_t)t[2]);
      if (err == 0) {
        atomic_store_explicit(count(r, SENT), c, memory_order_release);
        r->posted = c;
      }
    }
    pthread_mutex_lock(&l->mu);
    r->users--;
    l->sending = 0;
    l->sent_ns = now_ns();
    pthread_cond_broadcast(&l->cv);
    if (err != 0) {
      pthread_mutex_unlock(&l->mu);
      link_lost(l, err);
      pthread_mutex_lock(&l->mu);
      return -1;
    }
    sent = 1;
  }
  return sent;
}

static void *sender(void *arg) {
  struct rig_remote_link *l = arg;
  pthread_mutex_lock(&l->mu);
  for (;;) {
    if (atomic_load(&l->failed)) break;
    if (l->head != NULL) {
      struct entry *e = l->head;
      l->head = e->next;
      if (l->head == NULL) l->tail = NULL;
      l->queued -= e->n;
      l->sending = 1;
      pthread_cond_broadcast(&l->cv);
      pthread_mutex_unlock(&l->mu);
      int err = rig_remote_send_all(l->fd, e->buf, e->n);
      int closed = e->buf[8] == K_CLOSE;
      free(e->buf);
      free(e);
      pthread_mutex_lock(&l->mu);
      l->sending = 0;
      l->sent_ns = now_ns();
      if (err != 0) {
        pthread_mutex_unlock(&l->mu);
        link_lost(l, err);
        pthread_mutex_lock(&l->mu);
        break;
      }
      if (closed) {
        l->sent_close = 1;
        pthread_cond_broadcast(&l->cv);
        break;
      }
      continue;
    }
    if (l->rails != NULL) {
      int r = send_rails(l);
      if (r < 0) break;
      if (r > 0) continue;
    }
    int64_t now = now_ns();
    if (now - l->sent_ns >= BEAT_NS) {
      unsigned char h[HEADER];
      put_header(h, 0, K_BEAT);
      l->sending = 1;
      pthread_mutex_unlock(&l->mu);
      int err = rig_remote_send_all(l->fd, h, sizeof h);
      pthread_mutex_lock(&l->mu);
      l->sending = 0;
      l->sent_ns = now_ns();
      if (err != 0) {
        pthread_mutex_unlock(&l->mu);
        link_lost(l, err);
        pthread_mutex_lock(&l->mu);
        break;
      }
      continue;
    }
    /* A rail's ready function, a frame queued and a failure wake it. */
    wait_ns(&l->cv, &l->mu, l->sent_ns + BEAT_NS - now);
  }
  if (atomic_load(&l->failed)) free_entries(l);
  pthread_mutex_unlock(&l->mu);
  thread_ends(l);
  return NULL;
}

/* The receiving thread */

/* Receives [n] bytes into [p]: 0, a socket error, [ENDED] or [SILENT].
   [last] is when the link's last byte came. */
static int recv_all(struct rig_remote_link *l, unsigned char *p, uint64_t n,
                    int64_t *last) {
  while (n > 0) {
    int64_t left = SILENCE_MS - (now_ns() - *last) / 1000000;
    if (left <= 0) return SILENT;
    rig_remote_pollfd pf = {0};
    pf.fd = l->fd;
    pf.events = POLLIN;
    int k = poll(&pf, 1, (int)left);
    if (k < 0) {
      int e = rig_remote_sock_error();
#ifndef _WIN32
      if (e == EINTR) continue;
#endif
      return e;
    }
    if (k == 0) return SILENT;
    int want = n > (1u << 30) ? (1 << 30) : (int)n;
    long m = (long)recv(l->fd, (char *)p, want, 0);
    if (m == 0) return ENDED;
    if (m < 0) {
      int e = rig_remote_sock_error();
#ifndef _WIN32
      if (e == EINTR || e == EAGAIN) continue;
#endif
      return e;
    }
    *last = now_ns();
    p += m;
    n -= (uint64_t)m;
  }
  return 0;
}

/* Receives a payload of [n] bytes into new memory. */
static int recv_payload(struct rig_remote_link *l, uint64_t n, unsigned char **p,
                        int64_t *last) {
  if (n > SIZE_MAX - 1) return TOO_LARGE;
  *p = malloc((size_t)n + 1);
  if (*p == NULL) return TOO_LARGE;
  int r = recv_all(l, *p, n, last);
  if (r != 0) {
    free(*p);
    *p = NULL;
  }
  return r;
}

/* Places a rail transfer of [n] bytes, its rail and count first. */
static int recv_rail(struct rig_remote_link *l, uint64_t n, int64_t *last) {
  unsigned char h[16];
  if (n < 16) return MALFORMED;
  int r = recv_all(l, h, 16, last);
  if (r != 0) return r;
  uint64_t id = rig_remote_get_u64(h), c = rig_remote_get_u64(h + 8);
  pthread_mutex_lock(&l->mu);
  struct rail *rl = l->rails;
  while (rl != NULL && rl->id != id) rl = rl->next;
  if (rl != NULL) rl->users++;
  pthread_mutex_unlock(&l->mu);
  if (rl == NULL) return MALFORMED;
  r = MALFORMED;
  if (rl->nreceive > 0 && c == rl->placed + 1) {
    uint64_t j = (c - 1) % rl->nreceive, k = ((c - 1) / rl->nreceive) % 2;
    uint64_t *t = rl->receive + 3 * j;
    if (t[2] == n - 16) {
      r = recv_all(l, rl->in + k * rl->in_stride + t[1], t[2], last);
      if (r == 0) {
        rl->placed = c;
        atomic_store_explicit(count(rl, ARRIVED), c, memory_order_release);
      }
    }
  }
  pthread_mutex_lock(&l->mu);
  rl->users--;
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  return r;
}

/* Takes the answer to the oldest pending request. */
static int recv_answer(struct rig_remote_link *l, uint64_t n, int64_t *last) {
  unsigned char *p;
  if (n < 1) return MALFORMED;
  int r = recv_payload(l, n, &p, last);
  if (r != 0) return r;
  pthread_mutex_lock(&l->mu);
  struct pending *q = l->pending;
  if (q == NULL || p[0] > 1) {
    pthread_mutex_unlock(&l->mu);
    free(p);
    return MALFORMED;
  }
  l->pending = q->next;
  if (l->pending == NULL) l->pending_last = NULL;
  q->refused = p[0];
  q->p = p;
  q->n = (size_t)n;
  q->done = 1;
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  return 0;
}

/* Queues a command of the controller for [next]. */
static int recv_cmd(struct rig_remote_link *l, int kind, uint64_t n, int64_t *last) {
  unsigned char *p;
  int r = recv_payload(l, n, &p, last);
  if (r != 0) return r;
  struct cmd *c = malloc(sizeof *c);
  if (c == NULL) {
    free(p);
    return TOO_LARGE;
  }
  c->next = NULL;
  c->kind = kind;
  c->p = p;
  c->n = (size_t)n;
  pthread_mutex_lock(&l->mu);
  if (l->cmds_last != NULL)
    l->cmds_last->next = c;
  else
    l->cmds = c;
  l->cmds_last = c;
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  return 0;
}

/* Proxies */

int rig_remote_add_dev(struct rig_remote_dev *d) {
  struct rig_remote_link *l = d->link;
  pthread_mutex_lock(&l->mu);
  if (d->id >= l->ndevs) {
    size_t n = d->id + 1 > 2 * l->ndevs ? d->id + 1 : 2 * l->ndevs;
    struct rig_remote_dev **a = realloc(l->devs, n * sizeof *a);
    if (a == NULL) {
      pthread_mutex_unlock(&l->mu);
      return -2;
    }
    memset(a + l->ndevs, 0, (n - l->ndevs) * sizeof *a);
    l->devs = a;
    l->ndevs = n;
  }
  int taken = l->devs[d->id] != NULL;
  if (!taken) l->devs[d->id] = d;
  pthread_mutex_unlock(&l->mu);
  return taken ? -1 : 0;
}

static struct rig_remote_dev *dev_of(struct rig_remote_link *l, uint64_t id) {
  return id < l->ndevs ? l->devs[id] : NULL;
}

/* Advances a proxy's word: once every copy into this process's memory of
   the values it covers has its bytes, and never past the last value
   handed over. */
static int recv_word(struct rig_remote_link *l, uint64_t n, int64_t *last) {
  unsigned char h[16];
  if (n != 16) return MALFORMED;
  int r = recv_all(l, h, 16, last);
  if (r != 0) return r;
  uint64_t id = rig_remote_get_u64(h), v = rig_remote_get_u64(h + 8);
  pthread_mutex_lock(&l->mu);
  struct rig_remote_dev *d = dev_of(l, id);
  /* A stopped proxy's word took its last value already, maybe before this
     report. */
  if (d != NULL && d->stopped && v <= atomic_load(&d->word)) {
    pthread_mutex_unlock(&l->mu);
    return 0;
  }
  if (d == NULL || v <= atomic_load(&d->word) || v > d->handed ||
      (d->locals != NULL && d->locals->value <= v)) {
    pthread_mutex_unlock(&l->mu);
    return MALFORMED;
  }
  while (d->flights != NULL && d->flights->value <= v) {
    struct rig_remote_flight *f = d->flights;
    d->flights = f->next;
    d->flying -= f->bytes;
    free(f);
  }
  if (d->flights == NULL) d->flights_last = NULL;
  atomic_store_explicit(&d->word, v, memory_order_release);
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  return 0;
}

/* Writes a copy's bytes where the oldest copy into this process's memory of
   its proxy named. */
static int recv_bytes(struct rig_remote_link *l, uint64_t n, int64_t *last) {
  unsigned char h[16];
  if (n < 16) return MALFORMED;
  int r = recv_all(l, h, 16, last);
  if (r != 0) return r;
  uint64_t id = rig_remote_get_u64(h), v = rig_remote_get_u64(h + 8);
  pthread_mutex_lock(&l->mu);
  struct rig_remote_dev *d = dev_of(l, id);
  struct rig_remote_local *c = d != NULL ? d->locals : NULL;
  if (c == NULL || c->value != v || c->bytes != n - 16) {
    pthread_mutex_unlock(&l->mu);
    return MALFORMED;
  }
  pthread_mutex_unlock(&l->mu);
  /* The copy stays pending while its bytes land, so a stop meanwhile leaves
     the word below it and rig keeps the memory. */
  r = recv_all(l, c->at, c->bytes, last);
  pthread_mutex_lock(&l->mu);
  d->locals = c->next;
  if (d->locals == NULL) d->locals_last = NULL;
  rig_remote_settle(d);
  pthread_mutex_unlock(&l->mu);
  free(c);
  return r;
}

void rig_remote_settle(struct rig_remote_dev *d) {
  struct rig_remote_link *l = d->link;
  if (!d->stopped || (d->locals != NULL && l->receiving)) return;
  if (atomic_load(&d->word) < d->handed)
    atomic_store_explicit(&d->word, d->handed, memory_order_release);
  pthread_cond_broadcast(&l->cv);
}

/* Handles one frame: 0 to read on, 1 once the peer closed, or a code. */
static int recv_frame(struct rig_remote_link *l, int kind, uint64_t n, int64_t *last) {
  int from_controller = l->peer == 0;
  switch (kind) {
  case K_RAIL:
    return recv_rail(l, n, last);
  case K_BEAT:
    return n == 0 ? 0 : MALFORMED;
  case K_CLOSE:
    if (n != 0) return MALFORMED;
    pthread_mutex_lock(&l->mu);
    l->got_close = 1;
    pthread_cond_broadcast(&l->cv);
    pthread_mutex_unlock(&l->mu);
    return 1;
  case K_ABORT: {
    /* A string: its length (u32) and its bytes. */
    unsigned char *p;
    if (n < 4 || n > 4 + MAX_WHY) return MALFORMED;
    int r = recv_payload(l, n, &p, last);
    if (r != 0) return r;
    uint64_t len = (uint64_t)p[0] | (uint64_t)p[1] << 8 |
                   (uint64_t)p[2] << 16 | (uint64_t)p[3] << 24;
    if (len != n - 4) {
      free(p);
      return MALFORMED;
    }
    fail_job(l->job, why_of((const char *)p + 4, (size_t)len));
    free(p);
    return 1;
  }
  case K_ANSWER:
    return from_controller ? MALFORMED : recv_answer(l, n, last);
  case K_WORD:
    return from_controller ? MALFORMED : recv_word(l, n, last);
  case K_BYTES:
    return from_controller ? MALFORMED : recv_bytes(l, n, last);
  case K_REQUEST:
  case K_HANDOVER:
  case K_DROP:
    return from_controller ? recv_cmd(l, kind, n, last) : MALFORMED;
  default:
    return MALFORMED;
  }
}

static void *receiver(void *arg) {
  struct rig_remote_link *l = arg;
  int64_t last = now_ns();
  for (;;) {
    unsigned char h[HEADER];
    int r = recv_all(l, h, HEADER, &last);
    if (r == 0) r = recv_frame(l, h[8], rig_remote_get_u64(h), &last);
    if (r == 1) break;
    if (r != 0) {
      link_lost(l, r);
      break;
    }
  }
  /* No copy's bytes land any more: stopped proxies take their last value. */
  pthread_mutex_lock(&l->mu);
  l->receiving = 0;
  for (size_t i = 0; i < l->ndevs; i++)
    if (l->devs[i] != NULL) rig_remote_settle(l->devs[i]);
  pthread_mutex_unlock(&l->mu);
  thread_ends(l);
  return NULL;
}

/* Queueing */

/* Appends [q], if not NULL, to the pending requests with the frame, and a
   close makes the frame the link's last. */
int rig_remote_queue(struct rig_remote_link *l, int kind,
                     const unsigned char *p, size_t np, struct pending *q) {
  struct entry *e = malloc(sizeof *e);
  size_t n = HEADER + np;
  unsigned char *buf = e != NULL ? malloc(n) : NULL;
  if (buf == NULL) {
    free(e);
    return -2;
  }
  put_header(buf, np, kind);
  if (np > 0) memcpy(buf + HEADER, p, np);
  e->next = NULL;
  e->buf = buf;
  e->n = n;
  pthread_mutex_lock(&l->mu);
  while (!atomic_load(&l->failed) && !l->closing && l->head != NULL &&
         l->queued + n > QUEUE_BYTES)
    pthread_cond_wait(&l->cv, &l->mu);
  if (atomic_load(&l->failed) || l->closing) {
    int r = atomic_load(&l->failed) ? -1 : -3;
    pthread_mutex_unlock(&l->mu);
    free(buf);
    free(e);
    return r;
  }
  if (l->tail != NULL)
    l->tail->next = e;
  else
    l->head = e;
  l->tail = e;
  l->queued += n;
  if (kind == K_CLOSE) l->closing = 1;
  if (q != NULL) {
    if (l->pending_last != NULL)
      l->pending_last->next = q;
    else
      l->pending = q;
    l->pending_last = q;
  }
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  return 0;
}

/* Stubs: jobs */

static value job_reason(struct rig_remote_job *j) {
  const struct rig_remote_why *w = atomic_load(&j->why);
  return w != NULL ? caml_alloc_initialized_string(w->n, w->s)
                   : caml_copy_string("");
}

/* A new open job, or 0 if one is open. Does not release the runtime. */
value caml_rig_remote_link_job(value unit) {
  (void)unit;
  pthread_mutex_lock(&jobs_mu);
  if (last_job != NULL && !rig_remote_forked(last_job) &&
      atomic_load(&last_job->state) == OPEN) {
    pthread_mutex_unlock(&jobs_mu);
    return caml_copy_nativeint(0);
  }
  struct rig_remote_job *j = calloc(1, sizeof *j);
  if (j == NULL) {
    pthread_mutex_unlock(&jobs_mu);
    caml_raise_out_of_memory();
  }
  pthread_mutex_init(&j->mu, NULL);
  pthread_cond_init(&j->cv, NULL);
  atomic_store(&j->state, OPEN);
  j->pid = (long)getpid();
  last_job = j;
  pthread_mutex_unlock(&jobs_mu);
  return caml_copy_nativeint((intnat)j);
}

/* The job's state, and its root cause if it failed, after waiting at most
   [ms] for it to leave OPEN. Releases the runtime. */
value caml_rig_remote_link_job_wait(value vj, value vms) {
  CAMLparam2(vj, vms);
  CAMLlocal1(r);
  struct rig_remote_job *j = Job_val(vj);
  int64_t ms = Long_val(vms);
  if (!rig_remote_forked(j) && atomic_load(&j->state) == OPEN && ms > 0) {
    caml_release_runtime_system();
    pthread_mutex_lock(&j->mu);
    int64_t until = now_ns() + ms * 1000000LL;
    for (int64_t now = now_ns();
         atomic_load(&j->state) == OPEN && now < until; now = now_ns())
      wait_ns(&j->cv, &j->mu, until - now);
    pthread_mutex_unlock(&j->mu);
    caml_acquire_runtime_system();
  }
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(atomic_load(&j->state)));
  Store_field(r, 1, job_reason(j));
  CAMLreturn(r);
}

/* Fails the job. Does not release the runtime: it waits on no peer. */
value caml_rig_remote_link_job_fail(value vj, value why) {
  struct rig_remote_job *j = Job_val(vj);
  if (rig_remote_forked(j)) return Val_unit;
  fail_job(j, why_of(String_val(why), caml_string_length(why)));
  return Val_unit;
}

/* Ends the job in order. Releases the runtime. */
value caml_rig_remote_link_job_close(value vj) {
  CAMLparam1(vj);
  struct rig_remote_job *j = Job_val(vj);
  if (rig_remote_forked(j)) CAMLreturn(Val_unit);
  caml_release_runtime_system();
  pthread_mutex_lock(&j->mu);
  if (atomic_load(&j->state) == OPEN) {
    for (struct rig_remote_link *l = j->links; l != NULL; l = l->next) {
      pthread_mutex_unlock(&j->mu);
      rig_remote_queue(l, K_CLOSE, NULL, 0, NULL);
      pthread_mutex_lock(&j->mu);
    }
    for (;;) {
      if (atomic_load(&j->state) != OPEN) break;
      int ended = 1;
      for (struct rig_remote_link *l = j->links; l != NULL; l = l->next) {
        pthread_mutex_lock(&l->mu);
        if (l->threads > 0) ended = 0;
        pthread_mutex_unlock(&l->mu);
      }
      if (ended) {
        atomic_store(&j->state, CLOSED);
        pthread_cond_broadcast(&j->cv);
        break;
      }
      pthread_cond_wait(&j->cv, &j->mu);
    }
  }
  pthread_mutex_unlock(&j->mu);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

/* Stubs: links */

/* A link of the job over [fd] to process [peer], or 0 if the job is
   closed. Does not release the runtime. */
value caml_rig_remote_link_make(value vj, value fd, value name, value peer) {
  struct rig_remote_job *j = Job_val(vj);
  struct rig_remote_link *l = calloc(1, sizeof *l);
  char *nm = strdup(String_val(name));
  if (l == NULL || nm == NULL) {
    free(l);
    free(nm);
    caml_raise_out_of_memory();
  }
  l->job = j;
  l->fd = rig_remote_sock_val(fd);
  l->name = nm;
  l->peer = Int_val(peer);
  pthread_mutex_init(&l->mu, NULL);
  pthread_cond_init(&l->cv, NULL);
  l->sent_ns = now_ns();
  rig_remote_quiet(l->fd);
  int one = 1;
  (void)setsockopt(l->fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one,
                   sizeof one);
  if (rig_remote_forked(j)) {
    atomic_store(&l->failed, 1);
    l->fd_closed = 1;
    close_sock(l->fd);
    return caml_copy_nativeint((intnat)l);
  }
  pthread_mutex_lock(&j->mu);
  if (atomic_load(&j->state) == CLOSED) {
    pthread_mutex_unlock(&j->mu);
    pthread_mutex_destroy(&l->mu);
    pthread_cond_destroy(&l->cv);
    free(nm);
    free(l);
    return caml_copy_nativeint(0);
  }
  l->next = j->links;
  j->links = l;
  if (atomic_load(&j->state) == FAILED) {
    atomic_store(&l->failed, 1);
    l->fd_closed = 1;
    close_sock(l->fd);
    pthread_mutex_unlock(&j->mu);
    return caml_copy_nativeint((intnat)l);
  }
  pthread_t ts, tr;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  l->threads = 2;
  l->receiving = 1;
  int e = pthread_create(&ts, &attr, sender, l);
  if (e != 0) l->threads -= 2;
  if (e == 0 && (e = pthread_create(&tr, &attr, receiver, l)) != 0)
    l->threads -= 1;
  if (e != 0) l->receiving = 0;
  pthread_attr_destroy(&attr);
  pthread_mutex_unlock(&j->mu);
  if (e != 0) link_lost(l, e);
  return caml_copy_nativeint((intnat)l);
}

/* Queues the frame of [kind] whose payload is [head]. Releases the runtime
   while the queue is full. */
value caml_rig_remote_link_post(value vl, value vkind, value head) {
  CAMLparam3(vl, vkind, head);
  struct rig_remote_link *l = Link_val(vl);
  if (rig_remote_forked(l->job)) CAMLreturn(Val_unit);
  size_t n = caml_string_length(head);
  unsigned char *b = malloc(n + 1);
  if (b == NULL) caml_raise_out_of_memory();
  memcpy(b, String_val(head), n);
  caml_release_runtime_system();
  int r = rig_remote_queue(l, Int_val(vkind), b, n, NULL);
  caml_acquire_runtime_system();
  free(b);
  if (r == -2) caml_raise_out_of_memory();
  CAMLreturn(Val_unit);
}

/* Queues the frame of [kind] whose payload is [head] then [area]'s bytes.
   Releases the runtime while the queue is full. */
value caml_rig_remote_link_post_area(value vl, value vkind, value head,
                                     value area) {
  CAMLparam4(vl, vkind, head, area);
  struct rig_remote_link *l = Link_val(vl);
  if (rig_remote_forked(l->job)) CAMLreturn(Val_unit);
  size_t nh = caml_string_length(head);
  size_t nb = caml_ba_byte_size(Caml_ba_array_val(area));
  unsigned char *b = malloc(nh + nb + 1);
  if (b == NULL) caml_raise_out_of_memory();
  memcpy(b, String_val(head), nh);
  memcpy(b + nh, Caml_ba_data_val(area), nb);
  caml_release_runtime_system();
  int r = rig_remote_queue(l, Int_val(vkind), b, nh + nb, NULL);
  caml_acquire_runtime_system();
  free(b);
  if (r == -2) caml_raise_out_of_memory();
  CAMLreturn(Val_unit);
}

static value area_of(unsigned char *p, size_t n) {
  return caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED,
                            1, p, (intnat)n);
}

static value area_of_bytes(const char *s, size_t n) {
  unsigned char *p = malloc(n + 1);
  if (p == NULL) caml_raise_out_of_memory();
  memcpy(p, s, n);
  return area_of(p, n);
}

/* Sends a request with the payload [head] and waits for its answer: (0,
   answer), (1, why) if the agent refused, or (2, root cause) if the job
   failed. Releases the runtime. */
value caml_rig_remote_link_request(value vl, value head) {
  CAMLparam2(vl, head);
  CAMLlocal2(r, a);
  struct rig_remote_link *l = Link_val(vl);
  int code = 2;
  struct pending *q = NULL;
  if (!rig_remote_forked(l->job)) {
    size_t n = caml_string_length(head);
    unsigned char *b = malloc(n + 1);
    q = calloc(1, sizeof *q);
    if (b == NULL || q == NULL) {
      free(b);
      free(q);
      caml_raise_out_of_memory();
    }
    memcpy(b, String_val(head), n);
    caml_release_runtime_system();
    int k = rig_remote_queue(l, K_REQUEST, b, n, q);
    free(b);
    if (k == 0) {
      pthread_mutex_lock(&l->mu);
      while (!q->done && !atomic_load(&l->failed))
        pthread_cond_wait(&l->cv, &l->mu);
      if (q->done && !atomic_load(&l->failed)) code = q->refused;
      pthread_mutex_unlock(&l->mu);
    }
    caml_acquire_runtime_system();
    /* A request the queue took stays pending until answered: once the job
       failed unanswered, the receiving thread may still hold it. */
    if (k != 0) free(q);
    if (k == -2) caml_raise_out_of_memory();
  }
  if (code == 2) {
    const struct rig_remote_why *w = atomic_load(&l->job->why);
    a = w != NULL ? area_of_bytes(w->s, w->n) : area_of_bytes("", 0);
  } else {
    size_t n = q->n - 1;
    unsigned char *p = malloc(n + 1);
    if (p != NULL) memcpy(p, q->p + 1, n);
    free(q->p);
    free(q);
    if (p == NULL) caml_raise_out_of_memory();
    a = area_of(p, n);
  }
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(code));
  Store_field(r, 1, a);
  CAMLreturn(r);
}

/* The next command of the controller: (kind, payload); (K_CLOSE, "") once
   it closed and every earlier command was read; or (0, root cause) once the
   job failed. Releases the runtime. */
value caml_rig_remote_link_next(value vl) {
  CAMLparam1(vl);
  CAMLlocal2(r, a);
  struct rig_remote_link *l = Link_val(vl);
  struct cmd *c = NULL;
  int closed = 0;
  if (!rig_remote_forked(l->job)) {
    caml_release_runtime_system();
    pthread_mutex_lock(&l->mu);
    while (l->cmds == NULL && !l->got_close && !atomic_load(&l->failed))
      pthread_cond_wait(&l->cv, &l->mu);
    if (!atomic_load(&l->failed) && l->cmds != NULL) {
      c = l->cmds;
      l->cmds = c->next;
      if (l->cmds == NULL) l->cmds_last = NULL;
    } else if (!atomic_load(&l->failed))
      closed = 1;
    pthread_mutex_unlock(&l->mu);
    caml_acquire_runtime_system();
  }
  int kind = c != NULL ? c->kind : closed ? K_CLOSE : 0;
  if (c != NULL)
    a = area_of(c->p, c->n);
  else if (closed)
    a = area_of_bytes("", 0);
  else {
    const struct rig_remote_why *w = atomic_load(&l->job->why);
    a = w != NULL ? area_of_bytes(w->s, w->n) : area_of_bytes("", 0);
  }
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(kind));
  Store_field(r, 1, a);
  free(c);
  CAMLreturn(r);
}

/* Stubs: rails */

/* A bigarray of [n] + a page's bytes and the offset of its first byte on a
   page. Does not release the runtime. */
value caml_rig_remote_link_area(value vn) {
  CAMLparam1(vn);
  CAMLlocal2(r, b);
#ifdef _WIN32
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  uintptr_t page = si.dwPageSize;
#else
  uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
#endif
  b = caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT, 1, NULL,
                         Long_val(vn) + (intnat)page);
  uintptr_t at = (uintptr_t)Caml_ba_data_val(b);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, b);
  Store_field(r, 1, Val_long((page - at % page) % page));
  CAMLreturn(r);
}

static uint64_t *transfers(value a) {
  mlsize_t n = Wosize_val(a);
  uint64_t *t = malloc((n + 1) * sizeof *t);
  if (t == NULL) caml_raise_out_of_memory();
  for (mlsize_t i = 0; i < n; i++) t[i] = (uint64_t)Long_val(Field(a, i));
  return t;
}

/* Advances a rail's [ready] to [c] with release order and wakes its link's
   sending thread. Calls nothing of the runtime: compiled host code calls it
   through its address. */
static void rail_ready(void *arg, uint64_t c) {
  struct rail *r = arg;
  struct rig_remote_link *l = r->link;
  atomic_store_explicit(count(r, READY), c, memory_order_release);
  if (rig_remote_forked(l->job)) return;
  pthread_mutex_lock(&l->mu);
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
}

value caml_rig_remote_link_ready(value vr, value vc) {
  rail_ready((void *)Nativeint_val(vr), (uint64_t)Long_val(vc));
  return Val_unit;
}

value caml_rig_remote_link_ready_fn(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)&rail_ready);
}

/* Registers the rail [id] on the link, and is its C state, which
   [rail_ready] takes: [send] and [receive] hold each transfer's src, dst and
   length in turn; [out], [in] and [counts] are its end's areas, which the
   caller keeps reachable until the rail is released. In a child of fork the
   rail is not registered, and its state is never freed. Does not release
   the runtime. */
value caml_rig_remote_link_rail(value vl, value id, value send, value receive,
                                value out, value in, value counts) {
  CAMLparam5(vl, id, send, receive, out);
  CAMLxparam2(in, counts);
  struct rig_remote_link *l = Link_val(vl);
  struct rail *r = calloc(1, sizeof *r);
  if (r == NULL) caml_raise_out_of_memory();
  r->link = l;
  r->id = (uint64_t)Long_val(id);
  r->send = transfers(send);
  r->receive = transfers(receive);
  r->nsend = Wosize_val(send) / 3;
  r->nreceive = Wosize_val(receive) / 3;
  r->out = Caml_ba_data_val(out);
  r->in = Caml_ba_data_val(in);
  r->counts = Caml_ba_data_val(counts);
  r->out_stride = caml_ba_byte_size(Caml_ba_array_val(out)) / 2;
  r->in_stride = caml_ba_byte_size(Caml_ba_array_val(in)) / 2;
  if (rig_remote_forked(l->job)) {
    for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
    CAMLreturn(caml_copy_nativeint((intnat)r));
  }
  pthread_mutex_lock(&l->mu);
  if (atomic_load(&l->failed))
    for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
  r->next = l->rails;
  l->rails = r;
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  CAMLreturn(caml_copy_nativeint((intnat)r));
}

value caml_rig_remote_link_rail_bc(value *argv, int argc) {
  (void)argc;
  return caml_rig_remote_link_rail(argv[0], argv[1], argv[2], argv[3],
                                   argv[4], argv[5], argv[6]);
}

/* Ends the rail [id] on the link once no thread uses it. Releases the
   runtime while it waits. */
value caml_rig_remote_link_release_rail(value vl, value id) {
  CAMLparam2(vl, id);
  struct rig_remote_link *l = Link_val(vl);
  if (rig_remote_forked(l->job)) CAMLreturn(Val_unit);
  uint64_t k = (uint64_t)Long_val(id);
  caml_release_runtime_system();
  pthread_mutex_lock(&l->mu);
  struct rail **p = &l->rails;
  while (*p != NULL && (*p)->id != k) p = &(*p)->next;
  struct rail *r = *p;
  if (r != NULL) {
    while (r->users > 0) pthread_cond_wait(&l->cv, &l->mu);
    p = &l->rails;
    while (*p != r) p = &(*p)->next;
    *p = r->next;
  }
  pthread_mutex_unlock(&l->mu);
  caml_acquire_runtime_system();
  if (r != NULL) {
    free(r->send);
    free(r->receive);
    free(r);
  }
  CAMLreturn(Val_unit);
}
