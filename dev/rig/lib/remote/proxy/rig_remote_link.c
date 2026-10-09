/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Jobs and links: this process's ends of its job's connections.

   A link owns its socket and two threads that never hold the OCaml
   runtime. A frame goes from its writer's thread, which holds the link's
   send claim while it sends: frames leave whole, one after another. The
   sending thread sends the transfers of the link's rails and a beat after
   a second without a send; it waits on nothing but a writer's send and its
   socket. The receiving thread reads frames, places rail transfers,
   answers requests, and queues an agent's commands; it measures silence on
   its own.

   A job fails once. Whoever finds the failure, a thread or a caller, sets
   the root cause under the job's lock, then for each link: marks it
   failed, raises its rails' counts, sends an abort if no frame is being
   sent, ends its stream and wakes every waiter. The receiving thread of a
   failed link discards frames until the peer ends its stream, so that no
   unread byte makes the system reset the connection and drop the abort
   before the peer reads it; a peer silent for the bound ends it too. A
   link's socket closes once both its threads ended and no abort is being
   sent.

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
#define SHUT_SEND SD_SEND
#else
#include <fcntl.h>
#ifdef __linux__
#include <linux/sockios.h>
#include <sys/ioctl.h>
#endif
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/uio.h>
#include <unistd.h>
typedef struct pollfd rig_remote_pollfd;
#define close_sock close
#define SHUT_SEND SHUT_WR
#endif

/* A beat after a second without a send; a failure after ten without a
   byte. */
#define BEAT_NS 1000000000LL
#define SILENCE_MS 10000

/* The seconds a send may wait for its peer to acknowledge a byte. */
#define SEND_S 10

/* The most bytes of a command's memory a link keeps for later commands. */
#define KEPT_BYTES ((size_t)128 << 20)

/* The most bytes of an abort's reason. */
#define MAX_WHY 4096

/* A rail's counts: [ready], [sent] and [arrived], each in a 128-byte line. */
#define READY 0
#define SENT 1
#define ARRIVED 2
#define COUNT_STRIDE 128


/* What a send or a receive answers besides 0 and a socket error. */
enum {
  ENDED = -1,
  SILENT = -2,
  MALFORMED = -3,
  TOO_LARGE = -4,
  STALLED = -5,
};

/* Structures */

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
  int kept; /* [p] is the link's kept memory */
};

struct rail {
  struct rig_remote_link *link;
  struct rail *next;
  uint64_t id;
  uint64_t *send, *receive; /* src, dst and length of each transfer */
  uint64_t nsend, nreceive;
  unsigned char *out, *in, *counts;
  value areas; /* a root holding [out], [in] and [counts] until released */
  size_t out_stride, in_stride;
  uint64_t posted; /* the sending thread's last count sent */
  uint64_t placed; /* the receiving thread's last count placed */
  int users;       /* threads using it outside the link's lock */
  int releasing;   /* [release_rail] waits for its users */
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

/* Sockets

   A link's socket does not block: each send and receive polls for at most
   the time its peer has left, so a peer that stops reading or sending fails
   the job after a bound the same on every system. */

/* Whether the socket error [e] asks to try again. */
static int again(int e) {
#ifdef _WIN32
  return e == WSAEWOULDBLOCK || e == WSAEINTR;
#else
  return e == EAGAIN || e == EWOULDBLOCK || e == EINTR;
#endif
}

/* The bytes [l]'s socket holds that its peer has not acknowledged, or -1
   where the system does not say. */
static int64_t unacked(struct rig_remote_link *l) {
  int q = -1;
#if defined(__linux__)
  if (ioctl(l->fd, SIOCOUTQ, &q) != 0) return -1;
#elif defined(__APPLE__)
  socklen_t n = sizeof q;
  if (getsockopt(l->fd, SOL_SOCKET, SO_NWRITE, &q, &n) != 0) return -1;
#else
  (void)l;
#endif
  return q;
}

/* Sends the [n] bytes at [p] on [l]'s socket: 0, a socket error, or
   [STALLED] once its peer acknowledged no byte for [SEND_S] seconds. A send
   that waits checks for acknowledgements at least once a second: a system
   may take bytes from a peer that reads nothing, so the bytes it takes are
   the measure only where it does not say what it holds unacknowledged.
   Calls nothing of the runtime. */
static int send_all(struct rig_remote_link *l, const void *p, size_t n) {
  const char *c = p;
  int64_t taken = 0; /* the bytes the system took from this call */
  int64_t stuck = 0; /* when the peer last acknowledged, once a send waits */
  int64_t seen = 0;  /* [got] then */
  while (n > 0) {
    int chunk = n > (1u << 30) ? (1 << 30) : (int)n;
    long k = (long)send(l->fd, c, chunk, RIG_REMOTE_NOSIGNAL);
    if (k > 0) {
      c += k;
      n -= (size_t)k;
      taken += k;
      continue;
    }
    int e = k < 0 ? rig_remote_sock_error() : 0;
    if (k < 0 && !again(e)) return e;
    /* Grows by each byte the peer acknowledges, whatever the system takes. */
    int64_t q = unacked(l);
    int64_t got = q < 0 ? taken : taken - q;
    if (stuck == 0 || got > seen) {
      stuck = now_ns();
      seen = got;
    }
    int64_t left = SEND_S * 1000LL - (now_ns() - stuck) / 1000000;
    if (left <= 0) return STALLED;
    rig_remote_pollfd pf = {0};
    pf.fd = l->fd;
    pf.events = POLLOUT;
    if (poll(&pf, 1, left < 1000 ? (int)left : 1000) < 0) {
      e = rig_remote_sock_error();
      if (!again(e)) return e;
    }
  }
  return 0;
}

/* Reasons */

static struct rig_remote_why failed_why = {14, "the job failed"};
static struct rig_remote_why closed_why = {17, "the job is closed"};
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
    snprintf(buf, sizeof buf, "%d", SILENCE_MS / 1000);
    return reason("%s: silent for %s s", l->name, buf);
  case STALLED:
    snprintf(buf, sizeof buf, "%d", SEND_S);
    return reason("%s: read nothing for %s s", l->name, buf);
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

/* Closes [l]'s socket once nothing uses it. Holds [l]'s lock. */
static void close_if_idle(struct rig_remote_link *l) {
  if (l->threads == 0 && !l->sending && !l->fd_closed) {
    l->fd_closed = 1;
    close_sock(l->fd);
  }
}

/* Sends an abort with [why], a string as wire.mli lays it out, after the
   frames sent before. A send that makes no progress for [SEND_S] seconds
   gives up: the peer reads nothing. */
static void send_abort(struct rig_remote_link *l,
                       const struct rig_remote_why *why) {
  unsigned char buf[HEADER + 4 + MAX_WHY];
  put_header(buf, 4 + why->n, K_ABORT);
  for (int i = 0; i < 4; i++)
    buf[HEADER + i] = (unsigned char)(why->n >> (8 * i));
  memcpy(buf + HEADER + 4, why->s, why->n);
  (void)send_all(l, buf, HEADER + 4 + why->n);
}

/* Sends the abort a failure owes [l], then ends its stream; nothing while a
   frame or an abort is being sent, whose sender calls it after. Holds [l]'s
   lock, which it releases while it sends. */
static void abort_link(struct rig_remote_link *l) {
  if (l->fd_closed || l->sending) return;
  if (l->abort_owed && !l->sent_close) {
    l->abort_owed = 0;
    l->sending = 1;
    pthread_mutex_unlock(&l->mu);
    send_abort(l, atomic_load(&l->job->why));
    pthread_mutex_lock(&l->mu);
    l->sending = 0;
  }
  shutdown(l->fd, SHUT_SEND);
  close_if_idle(l);
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
    for (struct rail *r = l->rails; r != NULL; r = r->next)
      for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
    l->abort_owed = !l->sent_close;
    pthread_cond_broadcast(&l->cv);
    pthread_cond_signal(&l->wake);
    pthread_mutex_unlock(&l->mu);
  }
  struct rig_remote_link *links = j->links;
  pthread_cond_broadcast(&j->cv);
  pthread_mutex_unlock(&j->mu);
  /* The root cause reaches every peer whatever its link does: an idle link
     sends it here; a link mid-frame sends it after that frame, from the
     thread that sends it. Links are only ever prepended, so the
     list read under the job's lock stays whole. */
  for (struct rig_remote_link *l = links; l != NULL; l = l->next) {
    pthread_mutex_lock(&l->mu);
    abort_link(l);
    pthread_mutex_unlock(&l->mu);
  }
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

/* Releases the send claim. Holds [l]'s lock. The writers waiting for it
   go first. The sending thread wakes only to act: once the link closed, or
   for a rail that is due once no writer waits, the last writer's release
   waking it then. */
static void release_claim(struct rig_remote_link *l) {
  l->sending = 0;
  l->sent_ns = now_ns();
  if (l->writers > 0) pthread_cond_broadcast(&l->cv);
  if (l->sent_close || (l->due && l->writers == 0))
    pthread_cond_signal(&l->wake);
}

/* Ends a rail's use by a thread outside [l]'s lock. Holds [l]'s lock. */
static void rail_unused(struct rig_remote_link *l, struct rail *r) {
  if (--r->users == 0 && r->releasing) pthread_cond_broadcast(&l->cv);
}

/* The sending thread */

/* Sends the transfers of [l]'s rails whose [ready] advanced. Holds [l]'s
   lock, and releases it while it sends. [0], or [-1] if the link
   failed. */
static int send_rails(struct rig_remote_link *l) {
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
      err = send_all(l, h, sizeof h);
      if (err == 0)
        err = send_all(l, r->out + k * r->out_stride + t[0], (size_t)t[2]);
      if (err == 0) {
        atomic_store_explicit(count(r, SENT), c, memory_order_release);
        r->posted = c;
      }
    }
    pthread_mutex_lock(&l->mu);
    rail_unused(l, r);
    release_claim(l);
    if (err != 0) {
      pthread_mutex_unlock(&l->mu);
      link_lost(l, err);
      pthread_mutex_lock(&l->mu);
      return -1;
    }
  }
  return 0;
}

/* Sends [f]: its own bytes, then its spans. The first span goes in the
   same call as the frame's own bytes, so that a small frame leaves in one
   segment. */
static int send_frame(struct rig_remote_link *l, struct rig_remote_frame *f) {
  size_t own = 0, first = 0; /* bytes of [buf] and of the first span sent */
#ifndef _WIN32
  if (f->nspans > 0) {
    struct iovec v[2] = {{f->buf, f->n},
                         {(void *)f->spans[0].p, f->spans[0].n}};
    struct msghdr m = {0};
    m.msg_iov = v;
    m.msg_iovlen = 2;
    ssize_t k = sendmsg(l->fd, &m, RIG_REMOTE_NOSIGNAL);
    if (k < 0 && !again(errno)) return errno;
    if (k > 0) {
      own = (size_t)k < f->n ? (size_t)k : f->n;
      first = (size_t)k - own;
    }
  }
#endif
  int err = send_all(l, f->buf + own, f->n - own);
  for (int i = 0; err == 0 && i < f->nspans; i++) {
    size_t skip = i == 0 ? first : 0;
    err = send_all(l, f->spans[i].p + skip, f->spans[i].n - skip);
  }
  return err;
}

static void *sender(void *arg) {
  struct rig_remote_link *l = arg;
  pthread_mutex_lock(&l->mu);
  for (;;) {
    if (atomic_load(&l->failed) || l->sent_close) break;
    int claimed = l->sending || l->writers > 0;
    if (!claimed && l->due) {
      l->due = 0;
      if (send_rails(l) < 0) break;
      continue;
    }
    int64_t now = now_ns();
    if (!claimed && now - l->sent_ns >= BEAT_NS) {
      unsigned char h[HEADER];
      put_header(h, 0, K_BEAT);
      l->sending = 1;
      pthread_mutex_unlock(&l->mu);
      int err = send_all(l, h, sizeof h);
      pthread_mutex_lock(&l->mu);
      release_claim(l);
      if (err != 0) {
        pthread_mutex_unlock(&l->mu);
        link_lost(l, err);
        pthread_mutex_lock(&l->mu);
        break;
      }
      continue;
    }
    /* A rail's ready function, the claim's release while a rail is due,
       and the link's end wake it. A claim's release counts as a send, so
       while one is held the beat is a second away. */
    if (l->due)
      pthread_cond_wait(&l->wake, &l->mu);
    else
      wait_ns(&l->wake, &l->mu, claimed ? BEAT_NS : l->sent_ns + BEAT_NS - now);
  }
  if (atomic_load(&l->failed)) abort_link(l);
  pthread_mutex_unlock(&l->mu);
  thread_ends(l);
  return NULL;
}

/* The receiving thread */

/* Receives [n] bytes into [p]: 0, a socket error, [ENDED] or [SILENT].
   [last] is when the link's last byte came before it failed. */
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
      if (again(e)) continue;
      return e;
    }
    if (k == 0) return SILENT;
    int want = n > (1u << 30) ? (1 << 30) : (int)n;
    long m = (long)recv(l->fd, (char *)p, want, 0);
    if (m == 0) return ENDED;
    if (m < 0) {
      int e = rig_remote_sock_error();
      if (again(e)) continue;
      return e;
    }
    if (!atomic_load(&l->failed)) *last = now_ns();
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
  rail_unused(l, rl);
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

/* Queues a command of the controller for [next], received into the link's
   kept memory when it is free and large enough, else into new memory. */
static int recv_cmd(struct rig_remote_link *l, int kind, uint64_t n, int64_t *last) {
  struct cmd *c = malloc(sizeof *c);
  if (c == NULL) return TOO_LARGE;
  pthread_mutex_lock(&l->mu);
  c->kept = l->kept_free && n <= l->kept_n;
  if (c->kept) l->kept_free = 0;
  pthread_mutex_unlock(&l->mu);
  int r;
  if (c->kept) {
    c->p = l->kept_p;
    r = recv_all(l, c->p, n, last);
  } else
    r = recv_payload(l, n, &c->p, last);
  if (r != 0) {
    pthread_mutex_lock(&l->mu);
    if (c->kept) l->kept_free = 1;
    pthread_mutex_unlock(&l->mu);
    free(c);
    return r;
  }
  c->next = NULL;
  c->kind = kind;
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

/* Receives and discards [n] bytes: 0 or a receive's answer. */
static int skip(struct rig_remote_link *l, uint64_t n, int64_t *last) {
  unsigned char b[16384];
  while (n > 0) {
    size_t k = n < sizeof b ? (size_t)n : sizeof b;
    int r = recv_all(l, b, k, last);
    if (r != 0) return r;
    n -= k;
  }
  return 0;
}

/* Handles one frame of a failed link: its peer's abort or close ends the
   stream, and anything else is discarded. */
static int drain_frame(struct rig_remote_link *l, int kind, uint64_t n,
                       int64_t *last) {
  if (kind == K_ABORT || kind == K_CLOSE) return 1;
  return skip(l, n, last);
}

static void *receiver(void *arg) {
  struct rig_remote_link *l = arg;
  int64_t last = now_ns();
  for (;;) {
    unsigned char h[HEADER];
    int r = recv_all(l, h, HEADER, &last);
    if (r == 0) {
      uint64_t n = rig_remote_get_u64(h);
      r = atomic_load(&l->failed) ? drain_frame(l, h[8], n, &last)
                                  : recv_frame(l, h[8], n, &last);
    }
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

/* Sending */

struct rig_remote_frame *rig_remote_frame(int kind, size_t np, int nspans) {
  size_t spans = (size_t)nspans * sizeof(struct rig_remote_span);
  if (np > SIZE_MAX - HEADER - spans - sizeof(struct rig_remote_frame))
    return NULL;
  struct rig_remote_frame *f = malloc(sizeof *f + spans + HEADER + np);
  if (f == NULL) return NULL;
  f->kind = kind;
  f->buf = (unsigned char *)f->spans + spans;
  f->n = HEADER + np;
  f->nspans = nspans;
  return f;
}

/* Appends [q], if not NULL, to the pending requests as the frame takes the
   send claim, so that answers find their requests in the order sent; a
   close makes the frame the link's last. */
int rig_remote_send(struct rig_remote_link *l, struct rig_remote_frame *f,
                    struct pending *q) {
  uint64_t np = f->n - HEADER;
  for (int i = 0; i < f->nspans; i++) np += f->spans[i].n;
  put_header(f->buf, np, f->kind);
  pthread_mutex_lock(&l->mu);
  l->writers++;
  while (!atomic_load(&l->failed) && !l->closing && l->sending)
    pthread_cond_wait(&l->cv, &l->mu);
  l->writers--;
  if (atomic_load(&l->failed) || l->closing) {
    int r = atomic_load(&l->failed) ? -1 : -3;
    pthread_mutex_unlock(&l->mu);
    free(f);
    return r;
  }
  if (q != NULL) {
    if (l->pending_last != NULL)
      l->pending_last->next = q;
    else
      l->pending = q;
    l->pending_last = q;
  }
  if (f->kind == K_CLOSE) l->closing = 1;
  l->sending = 1;
  pthread_mutex_unlock(&l->mu);
  int err = send_frame(l, f);
  pthread_mutex_lock(&l->mu);
  if (err == 0 && f->kind == K_CLOSE) l->sent_close = 1;
  release_claim(l);
  /* A failure meanwhile left the link's abort to this sender. */
  if (atomic_load(&l->failed)) abort_link(l);
  pthread_mutex_unlock(&l->mu);
  free(f);
  /* Sent or not, the frame was taken: the failure reaches its waiters. */
  if (err != 0) link_lost(l, err);
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
      struct rig_remote_frame *f = rig_remote_frame(K_CLOSE, 0, 0);
      if (f != NULL) rig_remote_send(l, f, NULL);
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
  pthread_cond_init(&l->wake, NULL);
  l->kept = l->gave = Val_unit;
  caml_register_generational_global_root(&l->kept);
  caml_register_generational_global_root(&l->gave);
  l->sent_ns = now_ns();
  rig_remote_quiet(l->fd);
  int one = 1;
  (void)setsockopt(l->fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one,
                   sizeof one);
#ifdef _WIN32
  u_long on = 1;
  (void)ioctlsocket(l->fd, FIONBIO, &on);
#else
  (void)fcntl(l->fd, F_SETFL, fcntl(l->fd, F_GETFL) | O_NONBLOCK);
#endif
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
    pthread_cond_destroy(&l->wake);
    caml_remove_generational_global_root(&l->kept);
    caml_remove_generational_global_root(&l->gave);
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

/* Sends the frame of [kind] whose payload is [head]. Releases the
   runtime. */
value caml_rig_remote_link_post(value vl, value vkind, value head) {
  CAMLparam3(vl, vkind, head);
  struct rig_remote_link *l = Link_val(vl);
  if (rig_remote_forked(l->job)) CAMLreturn(Val_unit);
  size_t n = caml_string_length(head);
  struct rig_remote_frame *f = rig_remote_frame(Int_val(vkind), n, 0);
  if (f == NULL) caml_raise_out_of_memory();
  memcpy(f->buf + HEADER, String_val(head), n);
  caml_release_runtime_system();
  rig_remote_send(l, f, NULL);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

/* Sends the frame of [kind] whose payload is [head] then [area]'s bytes,
   which it reads in place: it returns once they are sent, or the job
   failed. Releases the runtime. */
value caml_rig_remote_link_post_area(value vl, value vkind, value head,
                                     value area) {
  CAMLparam4(vl, vkind, head, area);
  struct rig_remote_link *l = Link_val(vl);
  if (rig_remote_forked(l->job)) CAMLreturn(Val_unit);
  size_t n = caml_string_length(head);
  struct rig_remote_frame *f = rig_remote_frame(Int_val(vkind), n, 1);
  if (f == NULL) caml_raise_out_of_memory();
  memcpy(f->buf + HEADER, String_val(head), n);
  f->spans[0].p = Caml_ba_data_val(area);
  f->spans[0].n = caml_ba_byte_size(Caml_ba_array_val(area));
  caml_release_runtime_system();
  rig_remote_send(l, f, NULL);
  caml_acquire_runtime_system();
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
   answer), (1, why) if the agent refused, or (2, why) if the job's close
   began, why saying so, or if the job failed, why its root cause. Releases
   the runtime. */
value caml_rig_remote_link_request(value vl, value head) {
  CAMLparam2(vl, head);
  CAMLlocal2(r, a);
  struct rig_remote_link *l = Link_val(vl);
  int code = 2, closing = 0;
  struct pending *q = NULL;
  if (!rig_remote_forked(l->job)) {
    size_t n = caml_string_length(head);
    struct rig_remote_frame *f = rig_remote_frame(K_REQUEST, n, 0);
    q = calloc(1, sizeof *q);
    if (f == NULL || q == NULL) {
      free(f);
      free(q);
      caml_raise_out_of_memory();
    }
    memcpy(f->buf + HEADER, String_val(head), n);
    caml_release_runtime_system();
    int k = rig_remote_send(l, f, q);
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
    closing = k == -3;
  }
  if (closing)
    a = area_of_bytes(closed_why.s, closed_why.n);
  else if (code == 2) {
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

/* Lends later commands the memory of the area [next] gave last: the kept
   memory once more, or that area's if the kept memory is free and smaller
   and the area holds at most [KEPT_BYTES]. Holds the link's lock and the
   runtime. */
static void lend_back(struct rig_remote_link *l) {
  if (l->gave_kept)
    l->kept_free = 1;
  else if (l->gave != Val_unit && (l->kept == Val_unit || l->kept_free)) {
    size_t n = caml_ba_byte_size(Caml_ba_array_val(l->gave));
    if (n > l->kept_n && n <= KEPT_BYTES) {
      caml_modify_generational_global_root(&l->kept, l->gave);
      l->kept_p = Caml_ba_data_val(l->kept);
      l->kept_n = n;
      l->kept_free = 1;
    }
  }
  caml_modify_generational_global_root(&l->gave, Val_unit);
  l->gave_kept = 0;
}

/* The next command of the controller: (kind, payload, its bytes), the
   payload's first bytes being the command's; (K_CLOSE, "", 0) once it
   closed and every earlier command was read; or (0, root cause, its bytes)
   once the job failed. The payload of the command before is lent to later
   commands. Releases the runtime. */
value caml_rig_remote_link_next(value vl) {
  CAMLparam1(vl);
  CAMLlocal2(r, a);
  struct rig_remote_link *l = Link_val(vl);
  struct cmd *c = NULL;
  int closed = 0;
  if (!rig_remote_forked(l->job)) {
    pthread_mutex_lock(&l->mu);
    lend_back(l);
    pthread_mutex_unlock(&l->mu);
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
  size_t n = 0;
  if (c != NULL) {
    a = c->kept ? l->kept : area_of(c->p, c->n);
    n = c->n;
    pthread_mutex_lock(&l->mu);
    caml_modify_generational_global_root(&l->gave, a);
    l->gave_kept = c->kept;
    pthread_mutex_unlock(&l->mu);
  } else if (closed)
    a = area_of_bytes("", 0);
  else {
    const struct rig_remote_why *w = atomic_load(&l->job->why);
    a = w != NULL ? area_of_bytes(w->s, w->n) : area_of_bytes("", 0);
    n = w != NULL ? w->n : 0;
  }
  r = caml_alloc_tuple(3);
  Store_field(r, 0, Val_int(kind));
  Store_field(r, 1, a);
  Store_field(r, 2, Val_long(n));
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
   sending thread, or, while the send claim is held, has its release wake
   it. Calls nothing of the runtime: compiled host code calls it through
   its address. */
static void rail_ready(void *arg, uint64_t c) {
  struct rail *r = arg;
  struct rig_remote_link *l = r->link;
  atomic_store_explicit(count(r, READY), c, memory_order_release);
  if (rig_remote_forked(l->job)) return;
  pthread_mutex_lock(&l->mu);
  l->due = 1;
  if (!l->sending && l->writers == 0) pthread_cond_signal(&l->wake);
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
  CAMLlocal1(areas);
  struct rig_remote_link *l = Link_val(vl);
  areas = caml_alloc_small(3, 0);
  Field(areas, 0) = out;
  Field(areas, 1) = in;
  Field(areas, 2) = counts;
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
  /* The threads use the areas until the rail is released, whoever else
     holds them. */
  r->areas = areas;
  caml_register_generational_global_root(&r->areas);
  pthread_mutex_lock(&l->mu);
  if (atomic_load(&l->failed))
    for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
  r->next = l->rails;
  l->rails = r;
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
    r->releasing = 1;
    while (r->users > 0) pthread_cond_wait(&l->cv, &l->mu);
    p = &l->rails;
    while (*p != r) p = &(*p)->next;
    *p = r->next;
  }
  pthread_mutex_unlock(&l->mu);
  caml_acquire_runtime_system();
  if (r != NULL) {
    caml_remove_generational_global_root(&r->areas);
    free(r->send);
    free(r->receive);
    free(r);
  }
  CAMLreturn(Val_unit);
}
