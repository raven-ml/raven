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

/* Frames: their kinds, as wire.mli lists them, and the header before each
   payload: its length (u64) and its kind (u8). */
enum {
  K_REQUEST = 1,
  K_ANSWER,
  K_HANDOVER,
  K_DROP,
  K_WORD,
  K_BYTES,
  K_RAIL,
  K_BEAT,
  K_ABORT,
  K_CLOSE
};

#define HEADER 9

/* A beat after a second without a send; a failure after ten without a
   byte. */
#define BEAT_NS 1000000000LL
#define SILENCE_MS 10000

/* How often the sending thread reads its rails' [ready] while idle. */
#define POLL_NS 50000LL

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

enum { OPEN, CLOSED, FAILED };

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

struct job;

struct link {
  struct job *job;
  struct link *next;
  rig_remote_sock fd;
  char *name;
  int peer; /* 0 for the controller, i for agent i */
  pthread_mutex_t mu;
  pthread_cond_t cv;
  struct entry *head, *tail;
  size_t queued;
  int sending; /* a frame is being sent, by the thread or an abort */
  int closing, sent_close, got_close, threads, fd_closed;
  _Atomic int failed;
  struct pending *pending, *pending_last;
  struct cmd *cmds, *cmds_last;
  struct rail *rails;
  int64_t sent_ns;
};

struct job {
  pthread_mutex_t mu;
  pthread_cond_t cv; /* the state changed, or a link ended */
  _Atomic int state;
  _Atomic(char *) why; /* set before the state */
  long pid;
  struct link *links;
};

static pthread_mutex_t jobs_mu = PTHREAD_MUTEX_INITIALIZER;
static struct job *last_job;

#define Job_val(v) ((struct job *)Nativeint_val(v))
#define Link_val(v) ((struct link *)Nativeint_val(v))

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

/* Little-endian integers */

static uint64_t get_u64(const unsigned char *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
  return v;
}

static void put_u64(unsigned char *p, uint64_t v) {
  for (int i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (8 * i));
}

static void put_header(unsigned char *p, uint64_t n, int kind) {
  put_u64(p, n);
  p[8] = (unsigned char)kind;
}

static _Atomic uint64_t *count(struct rail *r, int which) {
  return (_Atomic uint64_t *)(r->counts + COUNT_STRIDE * which);
}

/* Reasons */

static char *reason(const char *fmt, const char *name, const char *what) {
  int n = snprintf(NULL, 0, fmt, name, what);
  char *s = malloc((size_t)n + 1);
  if (s != NULL) snprintf(s, (size_t)n + 1, fmt, name, what);
  return s;
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
static char *link_reason(struct link *l, int code) {
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

static void free_entries(struct link *l) {
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
static void close_if_idle(struct link *l) {
  if (l->threads == 0 && !l->sending && !l->fd_closed) {
    l->fd_closed = 1;
    close_sock(l->fd);
  }
}

/* Sends an abort with [why] if the stream takes it at once. */
static void try_abort(struct link *l, const char *why) {
#ifdef _WIN32
  (void)l;
  (void)why;
#else
  size_t n = strlen(why);
  if (n > MAX_WHY) n = MAX_WHY;
  unsigned char buf[HEADER + MAX_WHY];
  put_header(buf, n, K_ABORT);
  memcpy(buf + HEADER, why, n);
  (void)send(l->fd, buf, HEADER + n, MSG_DONTWAIT | RIG_REMOTE_NOSIGNAL);
#endif
}

/* Fails [j] with [why], which it takes, unless it failed or closed. */
static void fail_job(struct job *j, char *why) {
  pthread_mutex_lock(&j->mu);
  if (atomic_load(&j->state) != OPEN) {
    pthread_mutex_unlock(&j->mu);
    free(why);
    return;
  }
  atomic_store(&j->why, why != NULL ? why : (char *)"the job failed");
  atomic_store(&j->state, FAILED);
  for (struct link *l = j->links; l != NULL; l = l->next) {
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
static void link_lost(struct link *l, int code) {
  if (atomic_load(&l->failed)) return;
  fail_job(l->job, link_reason(l, code));
}

/* [1] if this process is a child of the one that made [j], whose job it
   then fails in its own copy, taking no lock. */
static int forked(struct job *j) {
  if (j->pid == (long)getpid()) return 0;
  if (atomic_load(&j->state) == OPEN) {
    atomic_store(&j->why,
                 (char *)"a child of fork does not use its parent's connections");
    atomic_store(&j->state, FAILED);
  }
  return 1;
}

/* Thread ends */

static void thread_ends(struct link *l) {
  struct job *j = l->job;
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
static int send_rails(struct link *l) {
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
      put_u64(h + HEADER, r->id);
      put_u64(h + HEADER + 8, c);
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
  struct link *l = arg;
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
    int64_t until = l->sent_ns + BEAT_NS;
    if (l->rails != NULL && now + POLL_NS < until) until = now + POLL_NS;
    wait_ns(&l->cv, &l->mu, until - now);
  }
  if (atomic_load(&l->failed)) free_entries(l);
  pthread_mutex_unlock(&l->mu);
  thread_ends(l);
  return NULL;
}

/* The receiving thread */

/* Receives [n] bytes into [p]: 0, a socket error, [ENDED] or [SILENT].
   [last] is when the link's last byte came. */
static int recv_all(struct link *l, unsigned char *p, uint64_t n,
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
static int recv_payload(struct link *l, uint64_t n, unsigned char **p,
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
static int recv_rail(struct link *l, uint64_t n, int64_t *last) {
  unsigned char h[16];
  if (n < 16) return MALFORMED;
  int r = recv_all(l, h, 16, last);
  if (r != 0) return r;
  uint64_t id = get_u64(h), c = get_u64(h + 8);
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
static int recv_answer(struct link *l, uint64_t n, int64_t *last) {
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
static int recv_cmd(struct link *l, int kind, uint64_t n, int64_t *last) {
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

/* Handles one frame: 0 to read on, 1 once the peer closed, or a code. */
static int recv_frame(struct link *l, int kind, uint64_t n, int64_t *last) {
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
    unsigned char *p;
    if (n > MAX_WHY) return MALFORMED;
    int r = recv_payload(l, n, &p, last);
    if (r != 0) return r;
    p[n] = 0;
    fail_job(l->job, (char *)p);
    return 1;
  }
  case K_ANSWER:
    return from_controller ? MALFORMED : recv_answer(l, n, last);
  case K_REQUEST:
  case K_HANDOVER:
  case K_DROP:
    return from_controller ? recv_cmd(l, kind, n, last) : MALFORMED;
  default:
    return MALFORMED;
  }
}

static void *receiver(void *arg) {
  struct link *l = arg;
  int64_t last = now_ns();
  for (;;) {
    unsigned char h[HEADER];
    int r = recv_all(l, h, HEADER, &last);
    if (r == 0) r = recv_frame(l, h[8], get_u64(h), &last);
    if (r == 1) break;
    if (r != 0) {
      link_lost(l, r);
      break;
    }
  }
  thread_ends(l);
  return NULL;
}

/* Queueing */

/* Queues the frame of [kind] whose payload is the [n] bytes at [p], waiting
   while the queue is full, and appends [q] to the pending requests if it is
   not NULL. [0]; [-1] if the link failed or closes, the frame then dropped;
   [-2] if memory ran out. A close is the link's last frame. Called without
   the runtime. */
static int queue(struct link *l, int kind, const unsigned char *p, size_t np,
                 struct pending *q) {
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
    pthread_mutex_unlock(&l->mu);
    free(buf);
    free(e);
    return -1;
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

static value job_reason(struct job *j) {
  const char *w = atomic_load(&j->why);
  return caml_copy_string(w != NULL ? w : "");
}

/* A new open job, or 0 if one is open. Does not release the runtime. */
value caml_rig_remote_link_job(value unit) {
  (void)unit;
  pthread_mutex_lock(&jobs_mu);
  if (last_job != NULL && !forked(last_job) &&
      atomic_load(&last_job->state) == OPEN) {
    pthread_mutex_unlock(&jobs_mu);
    return caml_copy_nativeint(0);
  }
  struct job *j = calloc(1, sizeof *j);
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
  struct job *j = Job_val(vj);
  int64_t ms = Long_val(vms);
  if (!forked(j) && atomic_load(&j->state) == OPEN && ms > 0) {
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
  struct job *j = Job_val(vj);
  if (forked(j)) return Val_unit;
  char *w = strdup(String_val(why));
  fail_job(j, w);
  return Val_unit;
}

/* Ends the job in order. Releases the runtime. */
value caml_rig_remote_link_job_close(value vj) {
  CAMLparam1(vj);
  struct job *j = Job_val(vj);
  if (forked(j)) CAMLreturn(Val_unit);
  caml_release_runtime_system();
  pthread_mutex_lock(&j->mu);
  if (atomic_load(&j->state) == OPEN) {
    for (struct link *l = j->links; l != NULL; l = l->next) {
      pthread_mutex_unlock(&j->mu);
      queue(l, K_CLOSE, NULL, 0, NULL);
      pthread_mutex_lock(&j->mu);
    }
    for (;;) {
      if (atomic_load(&j->state) != OPEN) break;
      int ended = 1;
      for (struct link *l = j->links; l != NULL; l = l->next) {
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
  struct job *j = Job_val(vj);
  struct link *l = calloc(1, sizeof *l);
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
  if (forked(j)) {
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
  int e = pthread_create(&ts, &attr, sender, l);
  if (e != 0) l->threads -= 2;
  if (e == 0 && (e = pthread_create(&tr, &attr, receiver, l)) != 0)
    l->threads -= 1;
  pthread_attr_destroy(&attr);
  pthread_mutex_unlock(&j->mu);
  if (e != 0) link_lost(l, e);
  return caml_copy_nativeint((intnat)l);
}

/* Queues the frame of [kind] whose payload is [head]. Releases the runtime
   while the queue is full. */
value caml_rig_remote_link_post(value vl, value vkind, value head) {
  CAMLparam3(vl, vkind, head);
  struct link *l = Link_val(vl);
  if (forked(l->job)) CAMLreturn(Val_unit);
  size_t n = caml_string_length(head);
  unsigned char *b = malloc(n + 1);
  if (b == NULL) caml_raise_out_of_memory();
  memcpy(b, String_val(head), n);
  caml_release_runtime_system();
  int r = queue(l, Int_val(vkind), b, n, NULL);
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
  struct link *l = Link_val(vl);
  if (forked(l->job)) CAMLreturn(Val_unit);
  size_t nh = caml_string_length(head);
  size_t nb = caml_ba_byte_size(Caml_ba_array_val(area));
  unsigned char *b = malloc(nh + nb + 1);
  if (b == NULL) caml_raise_out_of_memory();
  memcpy(b, String_val(head), nh);
  memcpy(b + nh, Caml_ba_data_val(area), nb);
  caml_release_runtime_system();
  int r = queue(l, Int_val(vkind), b, nh + nb, NULL);
  caml_acquire_runtime_system();
  free(b);
  if (r == -2) caml_raise_out_of_memory();
  CAMLreturn(Val_unit);
}

static value area_of(unsigned char *p, size_t n) {
  return caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED,
                            1, p, (intnat)n);
}

static value area_of_string(const char *s) {
  size_t n = strlen(s);
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
  struct link *l = Link_val(vl);
  int code = 2;
  struct pending *q = NULL;
  if (!forked(l->job)) {
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
    int k = queue(l, K_REQUEST, b, n, q);
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
    const char *w = atomic_load(&l->job->why);
    a = area_of_string(w != NULL ? w : "");
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

/* The next command of the controller: (kind, payload), or (0, root cause)
   once the job failed. Releases the runtime. */
value caml_rig_remote_link_next(value vl) {
  CAMLparam1(vl);
  CAMLlocal2(r, a);
  struct link *l = Link_val(vl);
  struct cmd *c = NULL;
  if (!forked(l->job)) {
    caml_release_runtime_system();
    pthread_mutex_lock(&l->mu);
    while (l->cmds == NULL && !atomic_load(&l->failed))
      pthread_cond_wait(&l->cv, &l->mu);
    if (!atomic_load(&l->failed)) {
      c = l->cmds;
      l->cmds = c->next;
      if (l->cmds == NULL) l->cmds_last = NULL;
    }
    pthread_mutex_unlock(&l->mu);
    caml_acquire_runtime_system();
  }
  if (c == NULL) {
    const char *w = atomic_load(&l->job->why);
    a = area_of_string(w != NULL ? w : "");
  } else
    a = area_of(c->p, c->n);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(c == NULL ? 0 : c->kind));
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

/* Registers the rail [id] on the link: [send] and [receive] hold each
   transfer's src, dst and length in turn; [out], [in] and [counts] are its
   end's areas, which the caller keeps reachable until the rail is
   released. Does not release the runtime. */
value caml_rig_remote_link_rail(value vl, value id, value send, value receive,
                                value out, value in, value counts) {
  CAMLparam5(vl, id, send, receive, out);
  CAMLxparam2(in, counts);
  struct link *l = Link_val(vl);
  struct rail *r = calloc(1, sizeof *r);
  if (r == NULL) caml_raise_out_of_memory();
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
  if (forked(l->job)) {
    for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
    free(r->send);
    free(r->receive);
    free(r);
    CAMLreturn(Val_unit);
  }
  pthread_mutex_lock(&l->mu);
  if (atomic_load(&l->failed))
    for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
  r->next = l->rails;
  l->rails = r;
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  CAMLreturn(Val_unit);
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
  struct link *l = Link_val(vl);
  if (forked(l->job)) CAMLreturn(Val_unit);
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
