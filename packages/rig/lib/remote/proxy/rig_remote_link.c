/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Jobs and links: this process's ends of its job's connections.

   A link owns its socket and two threads that never hold the OCaml
   runtime. A frame goes from its writer's thread, which holds the link's
   send claim while it sends: frames leave whole, one after another. A
   rail's ready function sends a transfer the same way when the claim is
   free, as much of it as the socket takes at once, and hands the rest to
   the sending thread with the claim. The sending thread sends the rest,
   the transfers due while the claim was held, and a beat after a second
   without a send; it wakes only to send, and waits on nothing but a
   writer's send and its socket. The receiving thread reads frames, places
   rail transfers, answers requests, and queues an agent's commands; it
   measures silence on its own.

   A job fails once, in two steps. Whoever finds the failure, a thread or
   a caller, makes each link owe its peer an abort, then sets the root
   cause, under the job's lock; from then on no link starts a frame. A
   link sends its abort at once if no frame is being sent, else the
   frame's sender sends it after, then ends its stream; a link whose send
   failed sends none, its stream cut mid-frame. A sender whose send failed
   begins the failure before it releases the send claim, so no frame
   follows the cut. Once the last abort left, the job is marked failed:
   each link is marked failed, its rails' counts raised, and every waiter
   woken. Nothing in the process learns of the failure before, so a
   process that ends on its failure ends after every peer was sent the
   root cause.

   The receiving thread of a failing link discards frames until the peer
   ends its stream, so that no unread byte makes the system reset the
   connection and drop the abort before the peer reads it; a peer silent
   for the bound ends it too. A link's socket closes once both its threads
   ended and no abort is being sent.

   Reasons are formatted here, because the threads that find a failure hold
   no runtime: a fixed phrase, the system's error, or a peer's abort, each
   after the link's name, in memory never freed.

   A child of fork sees its parent's job failed, in its own copy: every
   entry compares the process id with the job's before it takes a lock. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/callback.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/threads.h>
#include <errno.h>
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

/* A beat after a second without a send. */
#define BEAT_NS 1000000000LL

/* The fewest bytes the receiving thread asks its socket for: a frame up to
   this size comes in one call with its header. */
#define READ_AHEAD 16384

/* The most bytes of a command's memory a link keeps for later commands. */
#define KEPT_BYTES ((size_t)128 << 20)

/* The most bytes of an abort's reason. */
#define MAX_WHY 4096

/* The most bytes one send offers. Winsock takes the whole of a send while
   the socket holds fewer bytes than its buffer's size, so a send of a large
   frame would leave at once, however little the peer reads. */
#define SEND_MAX (1 << 20)

/* A rail's counts: [ready], [sent] and [arrived], each in a 128-byte line. */
#define READY 0
#define SENT 1
#define ARRIVED 2
#define COUNT_STRIDE 128

/* A rail transfer's frame before its bytes: the header, the rail's id and
   the transfer's count. */
#define RAIL_HEADER (HEADER + 16)


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
  struct rail *next, *live_next;
  uint64_t id;
  uint64_t *send, *receive; /* src, dst and length of each transfer */
  uint64_t nsend, nreceive;
  unsigned char *out, *in, *counts;
  value areas; /* a root holding [out], [in] and [counts] until released */
  size_t out_stride, in_stride;
  uint64_t posted; /* the last count sent, by the claim's holders */
  size_t partial;  /* bytes of transfer [posted + 1]'s frame sent */
  uint64_t placed; /* the receiving thread's last count placed */
  int users;       /* threads using it outside the link's lock */
  int releasing;   /* release callers keeping it while its users drain */
};

static pthread_mutex_t jobs_mu = PTHREAD_MUTEX_INITIALIZER;
static struct rig_remote_job *last_job;

#define Job_val(v) ((struct rig_remote_job *)Nativeint_val(v))
#define Link_val(v) ((struct rig_remote_link *)Nativeint_val(v))

/* Ready callbacks find live rails by their counts. Taking this lock before
   a connection's keeps lookup and release from racing over the C state. */
static pthread_mutex_t rails_mu = PTHREAD_MUTEX_INITIALIZER;
static struct rail *live_rails;

rig_remote_link_hook_fn *_Atomic rig_remote_link_hook;

_Atomic int rig_remote_link_bound_s = 10;

/* The seconds a link waits for a byte, and for its peer to acknowledge one
   of a send, before it fails. */
static inline int bound_s(void) {
  return atomic_load_explicit(&rig_remote_link_bound_s, memory_order_relaxed);
}

static inline void hook(enum rig_remote_link_point at, int waiting) {
  rig_remote_link_hook_fn *f =
      atomic_load_explicit(&rig_remote_link_hook, memory_order_acquire);
  if (f != NULL) f(at, waiting);
}

#ifndef _WIN32
static pthread_once_t rails_once = PTHREAD_ONCE_INIT;
static int rails_fork_error;
static void rails_before_fork(void) { pthread_mutex_lock(&rails_mu); }
static void rails_after_fork(void) { pthread_mutex_unlock(&rails_mu); }
static void rails_child(void) {
  live_rails = NULL;
  pthread_mutex_unlock(&rails_mu);
}
static void rails_atfork(void) {
  rails_fork_error = pthread_atfork(rails_before_fork, rails_after_fork, rails_child);
}
#endif

/* Time */

static int64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

/* A condition waits on a clock no step of the wall clock moves: macOS and
   Windows wait for the time left; elsewhere the condition waits on
   CLOCK_MONOTONIC. */
void rig_remote_cond_init(pthread_cond_t *cv) {
#if defined(__APPLE__) || defined(_WIN32)
  pthread_cond_init(cv, NULL);
#else
  pthread_condattr_t monotonic;
  pthread_condattr_init(&monotonic);
  pthread_condattr_setclock(&monotonic, CLOCK_MONOTONIC);
  pthread_cond_init(cv, &monotonic);
  pthread_condattr_destroy(&monotonic);
#endif
}

void rig_remote_wait_ns(pthread_cond_t *cv, pthread_mutex_t *mu, int64_t ns) {
  if (ns < 0) ns = 0;
#if defined(__APPLE__) || defined(_WIN32)
  struct timespec t = {(time_t)(ns / 1000000000LL), (long)(ns % 1000000000LL)};
  pthread_cond_timedwait_relative_np(cv, mu, &t);
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  int64_t at = (int64_t)t.tv_nsec + ns;
  t.tv_sec += (time_t)(at / 1000000000LL);
  t.tv_nsec = (long)(at % 1000000000LL);
  pthread_cond_timedwait(cv, mu, &t);
#endif
}

/* Whether [v], a u64 a frame holds, fits a non-negative OCaml int: wire.mli
   holds one with bit 62 or 63 set malformed. */
static int fits_int(uint64_t v) { return v >> 62 == 0; }

static void put_header(unsigned char *p, uint64_t n, int kind) {
  rig_remote_put_u64(p, n);
  p[8] = (unsigned char)kind;
}

static _Atomic uint64_t *count(struct rail *r, int which) {
  return (_Atomic uint64_t *)(r->counts + COUNT_STRIDE * which);
}

/* A completed transfer or ready callback cannot overwrite failure's count,
   which wakes every waiter even while another thread completes its I/O. */
static void advance_count(_Atomic uint64_t *at, uint64_t next) {
  uint64_t current = atomic_load_explicit(at, memory_order_relaxed);
  while (current != INT64_MAX && current < next &&
         !atomic_compare_exchange_weak_explicit(at, &current, next,
                                               memory_order_release,
                                               memory_order_relaxed)) {}
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
   [STALLED] once its peer acknowledged no byte for [bound_s] seconds. A
   send that waits checks for acknowledgements at least once a second: a
   system may take bytes from a peer that reads nothing, so the bytes it
   takes are the measure only where it does not say what it holds
   unacknowledged. Calls nothing of the runtime. */
static int send_all(struct rig_remote_link *l, const void *p, size_t n) {
  const char *c = p;
  int64_t taken = 0; /* the bytes the system took from this call */
  int64_t stuck = 0; /* when the peer last acknowledged, once a send waits */
  int64_t seen = 0;  /* [got] then */
  while (n > 0) {
    int chunk = n > SEND_MAX ? SEND_MAX : (int)n;
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
    int64_t left = bound_s() * 1000LL - (now_ns() - stuck) / 1000000;
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
  DWORD k = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
      (DWORD)e, 0, buf, (DWORD)n, NULL);
  if (k == 0) snprintf(buf, n, "socket error %d", e);
  /* Windows ends its messages with CRLF. */
  while (k > 0 && (buf[k - 1] == '\r' || buf[k - 1] == '\n')) buf[--k] = '\0';
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
    snprintf(buf, sizeof buf, "%d", bound_s());
    return reason("%s: silent for %s s", l->name, buf);
  case STALLED:
    snprintf(buf, sizeof buf, "%d", bound_s());
    return reason("%s: read nothing for %s s", l->name, buf);
  case MALFORMED:
    return reason("%s: a malformed frame%s", l->name, "");
  case TOO_LARGE:
    return reason("%s: a frame larger than this process can hold%s", l->name,
                  "");
  default:
#ifdef _WIN32
    if (code == WSAECONNRESET || code == WSAECONNABORTED)
      return reason("%s: closed its connection%s", l->name, "");
#else
    if (code == EPIPE || code == ECONNRESET)
      return reason("%s: closed its connection%s", l->name, "");
#endif
    error_text(code, buf, sizeof buf);
    return reason("%s: %s", l->name, buf);
  }
}

/* Failure */

/* Whether [j]'s failure began: no link starts a frame. */
static int failing(struct rig_remote_job *j) {
  return atomic_load(&j->why) != NULL;
}

/* Closes [l]'s socket once nothing uses it. Holds [l]'s lock. */
static void close_if_idle(struct rig_remote_link *l) {
  if (l->threads == 0 && !l->sending && !l->fd_closed) {
    l->fd_closed = 1;
    close_sock(l->fd);
  }
}

/* Sends an abort with [why], a string as wire.mli lays it out, after the
   frames sent before. A send that makes no progress for [bound_s] seconds
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

/* Settles the abort a failure owes [l]: sends it, unless the link sent its
   close or a send of it failed, then ends its stream; nothing while a frame
   or an abort is being sent, whose sender calls it after. Whether it
   settled one, which its caller counts out with [abort_left] once it
   released [l]'s lock. Holds [l]'s lock, which it releases while it
   sends. */
static int abort_link(struct rig_remote_link *l) {
  if (l->sending) return 0;
  int owed = l->abort_owed;
  l->abort_owed = 0;
  if (owed && !l->fd_closed && !l->sent_close && !l->broken) {
    l->sending = 1;
    pthread_mutex_unlock(&l->mu);
    hook(RIG_REMOTE_LINK_ABORTING, 0);
    send_abort(l, atomic_load(&l->job->why));
    pthread_mutex_lock(&l->mu);
    l->sending = 0;
    pthread_cond_signal(&l->wake);
  }
  if (!l->fd_closed) {
    shutdown(l->fd, SHUT_SEND);
    close_if_idle(l);
  }
  return owed;
}

/* Marks [j] and its links failed, raises their rails' counts and wakes
   every waiter. Holds [j]'s lock. The job reads failed first, so that one
   who learns of the failure from a link finds the job failed. */
static void mark_failed(struct rig_remote_job *j) {
  atomic_store(&j->state, FAILED);
  for (struct rig_remote_link *l = j->links; l != NULL; l = l->next) {
    pthread_mutex_lock(&l->mu);
    atomic_store(&l->failed, 1);
    for (struct rail *r = l->rails; r != NULL; r = r->next)
      if (!r->releasing)
        for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
    pthread_cond_broadcast(&l->cv);
    pthread_cond_signal(&l->wake);
    pthread_mutex_unlock(&l->mu);
  }
  pthread_cond_broadcast(&j->cv);
}

/* Counts out one abort [j]'s failure owes; the last marks it failed. */
static void abort_left(struct rig_remote_job *j) {
  pthread_mutex_lock(&j->mu);
  if (--j->owed == 0) mark_failed(j);
  pthread_mutex_unlock(&j->mu);
}

/* Fails [j] with [why], which it takes, unless its failure began or it
   closed. */
static void fail_job(struct rig_remote_job *j, struct rig_remote_why *why) {
  pthread_mutex_lock(&j->mu);
  if (atomic_load(&j->state) != OPEN || failing(j)) {
    pthread_mutex_unlock(&j->mu);
    why_free(why);
    return;
  }
  /* This call's share keeps the count above 0 until every link owes. */
  j->owed = 1;
  for (struct rig_remote_link *l = j->links; l != NULL; l = l->next) {
    pthread_mutex_lock(&l->mu);
    l->abort_owed = !l->fd_closed;
    j->owed += l->abort_owed;
    pthread_mutex_unlock(&l->mu);
  }
  /* Set once every link owes: a link's thread that sees the cause settles
     its abort, and one that found none owed would end the stream without
     it. */
  atomic_store(&j->why, why != NULL ? why : &failed_why);
  struct rig_remote_link *links = j->links;
  pthread_mutex_unlock(&j->mu);
  hook(RIG_REMOTE_LINK_FAILING, 0);
  /* An idle link sends its abort here; a link mid-frame sends it after
     that frame, from the thread that sends it. Links are only ever
     prepended, so the list read under the job's lock stays whole. */
  for (struct rig_remote_link *l = links; l != NULL; l = l->next) {
    pthread_mutex_lock(&l->mu);
    pthread_cond_signal(&l->wake);
    int settled = abort_link(l);
    pthread_mutex_unlock(&l->mu);
    if (settled) abort_left(j);
  }
  abort_left(j);
}

/* Fails [l]'s job with the reason of [code], unless its failure began. */
static void link_lost(struct rig_remote_link *l, int code) {
  if (failing(l->job)) return;
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

/* Sends what [l]'s socket takes at once of the [n] bytes at [p] then of
   [b], in one call that does not wait: the bytes sent, or a negated socket
   error. One call sends a small frame in one segment, so that its peer's
   receiving thread wakes once for it. Where the system has no call for
   two areas, it sends [p]'s bytes alone, or [b]'s once [n] is 0. */
static long send_some(struct rig_remote_link *l, const unsigned char *p,
                      size_t n, struct rig_remote_span b) {
  if (b.n > SEND_MAX) b.n = SEND_MAX;
#ifdef _WIN32
  if (n == 0) {
    p = b.p;
    n = b.n;
  }
  long k = (long)send(l->fd, (const char *)p, n > SEND_MAX ? SEND_MAX : (int)n,
                      RIG_REMOTE_NOSIGNAL);
#else
  struct iovec v[2] = {{(void *)p, n}, {(void *)b.p, b.n}};
  struct msghdr m = {0};
  m.msg_iov = v;
  m.msg_iovlen = 2;
  long k = (long)sendmsg(l->fd, &m, RIG_REMOTE_NOSIGNAL);
#endif
  return k < 0 ? -(long)rig_remote_sock_error() : k;
}

/* Sends the [n] bytes at [p], then [spans], the first span in the same call
   as [p]'s bytes. */
static int send_spans(struct rig_remote_link *l, const unsigned char *p,
                      size_t n, const struct rig_remote_span *spans,
                      int nspans) {
  size_t own = 0, first = 0; /* bytes of [p] and of the first span sent */
  if (nspans > 0) {
    long k = send_some(l, p, n, spans[0]);
    if (k < 0 && !again((int)-k)) return (int)-k;
    if (k > 0) {
      own = (size_t)k < n ? (size_t)k : n;
      first = (size_t)k - own;
    }
  }
  int err = send_all(l, p + own, n - own);
  for (int i = 0; err == 0 && i < nspans; i++) {
    size_t skip = i == 0 ? first : 0;
    err = send_all(l, spans[i].p + skip, spans[i].n - skip);
  }
  return err;
}

/* Releases the send claim after a send that answered [err]. Holds [l]'s
   lock. The writers waiting for it go first. The sending thread wakes only
   to act: once the link closed or its job began to fail, or for a rail that
   is due once no writer waits, the last writer's release waking it then. */
static void release_claim(struct rig_remote_link *l, int err) {
  l->sending = 0;
  l->sent_ns = now_ns();
  if (err != 0) l->broken = 1;
  if (l->writers > 0) pthread_cond_broadcast(&l->cv);
  if (l->sent_close || failing(l->job) || (l->due && l->writers == 0))
    pthread_cond_signal(&l->wake);
}

/* Rail transfers */

/* The frame of [r]'s transfer [c] after its first [at] bytes: the rest of
   its header, which it writes in [h], and of the transfer's bytes. */
static void rail_frame(struct rail *r, uint64_t c, size_t at,
                       unsigned char h[RAIL_HEADER],
                       struct rig_remote_span s[2]) {
  uint64_t j = (c - 1) % r->nsend, k = ((c - 1) / r->nsend) % 2;
  uint64_t *t = r->send + 3 * j;
  put_header(h, 16 + t[2], K_RAIL);
  rig_remote_put_u64(h + HEADER, r->id);
  rig_remote_put_u64(h + HEADER + 8, c);
  size_t in_h = at < RAIL_HEADER ? at : RAIL_HEADER;
  s[0].p = h + in_h;
  s[0].n = RAIL_HEADER - in_h;
  s[1].p = r->out + k * r->out_stride + t[0] + (at - in_h);
  s[1].n = (size_t)t[2] - (at - in_h);
}

/* Marks [r]'s transfer [c] sent. */
static void rail_sent(struct rail *r, uint64_t c) {
  r->partial = 0;
  r->posted = c;
  hook(RIG_REMOTE_LINK_SENT, 0);
  advance_count(count(r, SENT), c);
}

/* Sends [r]'s transfers up to [ready], the first from its byte
   [r->partial]. Once the job began to fail it sends none, but the rest of
   a transfer begun: the abort follows whole frames. Holds the send claim,
   not [l]'s lock. [0], or a socket error. */
static int send_due(struct rig_remote_link *l, struct rail *r,
                    uint64_t ready) {
  while (r->posted < ready && (r->partial > 0 || !failing(l->job))) {
    uint64_t c = r->posted + 1;
    unsigned char h[RAIL_HEADER];
    struct rig_remote_span s[2];
    rail_frame(r, c, r->partial, h, s);
    int err = send_spans(l, s[0].p, s[0].n, &s[1], 1);
    if (err != 0) return err;
    rail_sent(r, c);
  }
  return 0;
}

/* Ends a rail's use by a thread outside [l]'s lock. Holds [l]'s lock. */
static void rail_unused(struct rig_remote_link *l, struct rail *r) {
  if (--r->users == 0 && r->releasing) pthread_cond_broadcast(&l->cv);
}

/* The sending thread */

/* Ends the sending thread's send that failed with [err]: begins the job's
   failure, then releases the send claim, so that no frame follows the cut
   one. Holds [l]'s lock and the claim, and releases the lock while the
   failure begins, which may wait for other links' aborts. */
static void cut(struct rig_remote_link *l, int err) {
  pthread_mutex_unlock(&l->mu);
  link_lost(l, err);
  pthread_mutex_lock(&l->mu);
  release_claim(l, err);
  pthread_mutex_unlock(&l->mu);
  hook(RIG_REMOTE_LINK_CUT, 0);
  pthread_mutex_lock(&l->mu);
}

/* Sends the transfers of [l]'s rails whose [ready] advanced, until the link
   ended. Holds [l]'s lock, and releases it while it sends. [0], or the error
   of a send that failed. */
static int send_rails(struct rig_remote_link *l) {
  for (struct rail *r = l->rails; r != NULL; r = r->next) {
    if (failing(l->job) || l->sent_close) return 0;
    if (r->releasing || r->nsend == 0) continue;
    uint64_t ready = atomic_load_explicit(count(r, READY), memory_order_acquire);
    if (ready <= r->posted) continue;
    r->users++;
    l->sending = 1;
    pthread_mutex_unlock(&l->mu);
    int err = send_due(l, r, ready);
    pthread_mutex_lock(&l->mu);
    rail_unused(l, r);
    if (err != 0) {
      cut(l, err);
      return err;
    }
    release_claim(l, 0);
    /* This rail stays linked while used; releases of later rails updated
       its next pointer under the lock while the send held none. */
  }
  return 0;
}

/* Finishes what the ready function began ([rail_ready]): it handed over the
   send claim and its use of the rail, whose transfer it sent in part or
   not at all, or met a socket error on. Holds [l]'s lock, and releases it
   while it sends. [0], or the error of a send that failed. */
static int resume(struct rig_remote_link *l) {
  struct rail *r = l->resume;
  int err = l->resume_error;
  uint64_t ready = l->resume_ready;
  l->resume = NULL;
  l->resume_error = 0;
  l->resume_ready = 0;
  pthread_mutex_unlock(&l->mu);
  hook(RIG_REMOTE_LINK_SENDING, 0);
  if (err == 0) err = send_due(l, r, ready);
  pthread_mutex_lock(&l->mu);
  rail_unused(l, r);
  if (err != 0)
    cut(l, err);
  else
    release_claim(l, 0);
  return err;
}

static void *sender(void *arg) {
  struct rig_remote_link *l = arg;
  pthread_mutex_lock(&l->mu);
  for (;;) {
    if (l->resume != NULL) {
      if (resume(l) != 0) break;
      continue;
    }
    /* Once the link ends, it waits for the claim's holder: a ready function
       leaves a failure's abort to it. */
    int ended = failing(l->job) || l->sent_close;
    if (ended && !l->sending) break;
    int claimed = l->sending || l->writers > 0;
    if (!ended && !claimed && l->due) {
      l->due = 0;
      if (send_rails(l) != 0) break;
      continue;
    }
    int64_t now = now_ns();
    if (!ended && !claimed && now - l->sent_ns >= BEAT_NS) {
      unsigned char h[HEADER];
      put_header(h, 0, K_BEAT);
      l->sending = 1;
      pthread_mutex_unlock(&l->mu);
      int err = send_all(l, h, sizeof h);
      pthread_mutex_lock(&l->mu);
      if (err != 0) {
        cut(l, err);
        break;
      }
      release_claim(l, 0);
      continue;
    }
    /* A claim handed over, the claim's release while a rail is due or once
       the link ended, and the link's end wake it. A claim's release counts
       as a send, so while one is held the beat is a second away. */
    if (ended || l->due)
      pthread_cond_wait(&l->wake, &l->mu);
    else
      rig_remote_wait_ns(&l->wake, &l->mu,
                         claimed ? BEAT_NS : l->sent_ns + BEAT_NS - now);
  }
  int settled = failing(l->job) ? abort_link(l) : 0;
  pthread_mutex_unlock(&l->mu);
  if (settled) abort_left(l->job);
  thread_ends(l);
  return NULL;
}

/* The receiving thread */

/* The receiving thread's reader: when the link's last byte came, and bytes
   received ahead of the frames that hold them. Silence counts from the last
   byte, before a failure and during its drain alike. */
struct reader {
  int64_t last;
  size_t at, end; /* [b]'s bytes from [at] to [end] are unread */
  unsigned char b[READ_AHEAD];
};

/* Receives [n] bytes into [p]: 0, a socket error, [ENDED] or [SILENT]. It
   asks the socket for [READ_AHEAD] bytes at least, so that a small frame's
   header and payload come in one call, and receives larger payloads in
   place. It polls only once the socket holds nothing. */
static int recv_all(struct rig_remote_link *l, unsigned char *p, uint64_t n,
                    struct reader *rd) {
  for (;;) {
    size_t have = rd->end - rd->at;
    if (have > n) have = (size_t)n;
    memcpy(p, rd->b + rd->at, have);
    rd->at += have;
    p += have;
    n -= have;
    if (n == 0) return 0;
    int ahead = n < READ_AHEAD;
    int want = ahead ? READ_AHEAD : n > (1u << 30) ? (1 << 30) : (int)n;
    long m = (long)recv(l->fd, (char *)(ahead ? rd->b : p), want, 0);
    if (m == 0) return ENDED;
    if (m > 0) {
      rd->last = now_ns();
      if (ahead) {
        rd->at = 0;
        rd->end = (size_t)m;
      } else {
        p += m;
        n -= (uint64_t)m;
      }
      continue;
    }
    int e = rig_remote_sock_error();
    if (!again(e)) return e;
    int64_t left = bound_s() * 1000LL - (now_ns() - rd->last) / 1000000;
    if (left <= 0) return SILENT;
    rig_remote_pollfd pf = {0};
    pf.fd = l->fd;
    pf.events = POLLIN;
    int k = poll(&pf, 1, (int)left);
    if (k == 0) return SILENT;
    if (k < 0 && !again(e = rig_remote_sock_error())) return e;
  }
}

/* Receives a payload of [n] bytes into new memory. */
static int recv_payload(struct rig_remote_link *l, uint64_t n, unsigned char **p,
                        struct reader *rd) {
  if (n > SIZE_MAX - 1) return TOO_LARGE;
  *p = malloc((size_t)n + 1);
  if (*p == NULL) return TOO_LARGE;
  int r = recv_all(l, *p, n, rd);
  if (r != 0) {
    free(*p);
    *p = NULL;
  }
  return r;
}

/* Receives and discards [n] bytes: 0 or a receive's answer. */
static int skip(struct rig_remote_link *l, uint64_t n, struct reader *rd) {
  unsigned char b[16384];
  while (n > 0) {
    size_t k = n < sizeof b ? (size_t)n : sizeof b;
    int r = recv_all(l, b, k, rd);
    if (r != 0) return r;
    n -= k;
  }
  return 0;
}

/* Places a rail transfer of [n] bytes, its rail and count first. A
   transfer to a rail this end does not hold is discarded: the rail ended
   here, where nothing reads it any more, and its peer's end may still
   send until its own machine ends it. */
static int recv_rail(struct rig_remote_link *l, uint64_t n, struct reader *rd) {
  unsigned char h[16];
  if (n < 16) return MALFORMED;
  int r = recv_all(l, h, 16, rd);
  if (r != 0) return r;
  uint64_t id = rig_remote_get_u64(h), c = rig_remote_get_u64(h + 8);
  if (!fits_int(id) || !fits_int(c)) return MALFORMED;
  pthread_mutex_lock(&l->mu);
  struct rail *rl = l->rails;
  while (rl != NULL && rl->id != id) rl = rl->next;
  if (rl != NULL && rl->releasing) rl = NULL;
  if (rl != NULL) rl->users++;
  pthread_mutex_unlock(&l->mu);
  if (rl == NULL) return skip(l, n - 16, rd);
  r = MALFORMED;
  if (rl->nreceive > 0 && c == rl->placed + 1) {
    uint64_t j = (c - 1) % rl->nreceive, k = ((c - 1) / rl->nreceive) % 2;
    uint64_t *t = rl->receive + 3 * j;
    if (t[2] == n - 16) {
      r = recv_all(l, rl->in + k * rl->in_stride + t[1], t[2], rd);
      if (r == 0) {
        rl->placed = c;
        hook(RIG_REMOTE_LINK_ARRIVED, 0);
        advance_count(count(rl, ARRIVED), c);
      }
    }
  }
  pthread_mutex_lock(&l->mu);
  rail_unused(l, rl);
  pthread_mutex_unlock(&l->mu);
  return r;
}

/* Takes the answer to the oldest pending request. */
static int recv_answer(struct rig_remote_link *l, uint64_t n, struct reader *rd) {
  unsigned char *p;
  if (n < 1) return MALFORMED;
  int r = recv_payload(l, n, &p, rd);
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
static int recv_cmd(struct rig_remote_link *l, int kind, uint64_t n, struct reader *rd) {
  struct cmd *c = malloc(sizeof *c);
  if (c == NULL) return TOO_LARGE;
  pthread_mutex_lock(&l->mu);
  c->kept = l->kept_free && n <= l->kept_n;
  if (c->kept) l->kept_free = 0;
  pthread_mutex_unlock(&l->mu);
  int r;
  if (c->kept) {
    c->p = l->kept_p;
    r = recv_all(l, c->p, n, rd);
  } else
    r = recv_payload(l, n, &c->p, rd);
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
static int recv_word(struct rig_remote_link *l, uint64_t n, struct reader *rd) {
  unsigned char h[16];
  if (n != 16) return MALFORMED;
  int r = recv_all(l, h, 16, rd);
  if (r != 0) return r;
  uint64_t id = rig_remote_get_u64(h), v = rig_remote_get_u64(h + 8);
  if (!fits_int(id) || !fits_int(v)) return MALFORMED;
  pthread_mutex_lock(&l->mu);
  struct rig_remote_dev *d = dev_of(l, id);
  /* A stopped proxy's word took its last value already, maybe before this
     report. */
  if (d != NULL && d->stopped && v <= atomic_load(d->word)) {
    pthread_mutex_unlock(&l->mu);
    return 0;
  }
  if (d == NULL || v <= atomic_load(d->word) || v > d->handed ||
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
  atomic_store_explicit(d->word, v, memory_order_release);
  pthread_cond_broadcast(&l->cv);
  pthread_mutex_unlock(&l->mu);
  return 0;
}

/* Writes a copy's bytes where the oldest copy into this process's memory of
   its proxy named. */
static int recv_bytes(struct rig_remote_link *l, uint64_t n, struct reader *rd) {
  unsigned char h[16];
  if (n < 16) return MALFORMED;
  int r = recv_all(l, h, 16, rd);
  if (r != 0) return r;
  uint64_t id = rig_remote_get_u64(h), v = rig_remote_get_u64(h + 8);
  if (!fits_int(id) || !fits_int(v)) return MALFORMED;
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
  r = recv_all(l, c->at, c->bytes, rd);
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
  if (atomic_load(d->word) < d->handed)
    atomic_store_explicit(d->word, d->handed, memory_order_release);
  pthread_cond_broadcast(&l->cv);
}

/* Handles one frame: 0 to read on, 1 once the peer closed, or a code. */
static int recv_frame(struct rig_remote_link *l, int kind, uint64_t n, struct reader *rd) {
  int from_controller = l->peer == 0;
  if (!fits_int(n)) return MALFORMED;
  switch (kind) {
  case K_RAIL:
    return recv_rail(l, n, rd);
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
    int r = recv_payload(l, n, &p, rd);
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
    return from_controller ? MALFORMED : recv_answer(l, n, rd);
  case K_WORD:
    return from_controller ? MALFORMED : recv_word(l, n, rd);
  case K_BYTES:
    return from_controller ? MALFORMED : recv_bytes(l, n, rd);
  case K_REQUEST:
  case K_HANDOVER:
  case K_DROP:
    return from_controller ? recv_cmd(l, kind, n, rd) : MALFORMED;
  default:
    return MALFORMED;
  }
}

/* Handles one frame of a failing link: its peer's abort or close ends the
   stream, and anything else is discarded. */
static int drain_frame(struct rig_remote_link *l, int kind, uint64_t n,
                       struct reader *rd) {
  if (kind == K_ABORT || kind == K_CLOSE) return 1;
  return skip(l, n, rd);
}

static void *receiver(void *arg) {
  struct rig_remote_link *l = arg;
  struct reader rd;
  rd.last = now_ns();
  rd.at = rd.end = 0;
  for (;;) {
    unsigned char h[HEADER];
    int r = recv_all(l, h, HEADER, &rd);
    if (r == 0) {
      uint64_t n = rig_remote_get_u64(h);
      r = failing(l->job) ? drain_frame(l, h[8], n, &rd)
                          : recv_frame(l, h[8], n, &rd);
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
  /* Once the job began to fail, it waits until the failure is known. */
  while (!atomic_load(&l->failed) && !l->closing &&
         (l->sending || failing(l->job)))
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
  int err = send_spans(l, f->buf, f->n, f->spans, f->nspans);
  if (err != 0) link_lost(l, err);
  pthread_mutex_lock(&l->mu);
  if (err == 0 && f->kind == K_CLOSE) l->sent_close = 1;
  release_claim(l, err);
  /* A failure meanwhile, or this send's, left the link's abort to this
     sender. */
  int settled = failing(l->job) ? abort_link(l) : 0;
  pthread_mutex_unlock(&l->mu);
  if (err != 0) hook(RIG_REMOTE_LINK_CUT, 0);
  if (settled) abort_left(l->job);
  free(f);
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
  rig_remote_cond_init(&j->cv);
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
      rig_remote_wait_ns(&j->cv, &j->mu, until - now);
    pthread_mutex_unlock(&j->mu);
    caml_acquire_runtime_system();
  }
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(atomic_load(&j->state)));
  Store_field(r, 1, job_reason(j));
  CAMLreturn(r);
}

/* Fails the job, and returns once its failure is known, whoever began it.
   Releases the runtime: an abort waits up to [bound_s] seconds for a peer
   that reads nothing. */
value caml_rig_remote_link_job_fail(value vj, value why) {
  struct rig_remote_job *j = Job_val(vj);
  if (rig_remote_forked(j)) return Val_unit;
  struct rig_remote_why *w = why_of(String_val(why), caml_string_length(why));
  caml_release_runtime_system();
  fail_job(j, w);
  pthread_mutex_lock(&j->mu);
  while (atomic_load(&j->state) == OPEN) pthread_cond_wait(&j->cv, &j->mu);
  pthread_mutex_unlock(&j->mu);
  caml_acquire_runtime_system();
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
   closed. It takes [fd], which it closes if it makes no link. Releases the
   runtime only to fail the job, if its threads do not start. */
value caml_rig_remote_link_make(value vj, value fd, value name, value peer) {
  struct rig_remote_job *j = Job_val(vj);
  struct rig_remote_link *l = calloc(1, sizeof *l);
  char *nm = strdup(String_val(name));
  if (l == NULL || nm == NULL) {
    free(l);
    free(nm);
    close_sock(rig_remote_sock_val(fd));
    caml_raise_out_of_memory();
  }
  l->job = j;
  l->fd = rig_remote_sock_val(fd);
  l->name = nm;
  l->peer = Int_val(peer);
  pthread_mutex_init(&l->mu, NULL);
  rig_remote_cond_init(&l->cv);
  rig_remote_cond_init(&l->wake);
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
    close_sock(l->fd);
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
  if (failing(j)) {
    /* A failure under way marks it failed with the others. */
    if (atomic_load(&j->state) == FAILED) atomic_store(&l->failed, 1);
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
  if (e != 0) {
    caml_release_runtime_system();
    link_lost(l, e);
    caml_acquire_runtime_system();
  }
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

/* Drops the kept memory once [next] gave the controller's close: no command
   holds it, none comes, and a link is never freed, so an agent serving jobs
   one after another would keep one job's memory each. An area [next] gave
   keeps its own storage. Holds the runtime. */
static void retire_kept(struct rig_remote_link *l) {
  pthread_mutex_lock(&l->mu);
  caml_modify_generational_global_root(&l->kept, Val_unit);
  l->kept_p = NULL;
  l->kept_n = 0;
  l->kept_free = 0;
  pthread_mutex_unlock(&l->mu);
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
  } else if (closed) {
    retire_kept(l);
    a = area_of_bytes("", 0);
  } else {
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
  if (t == NULL) return NULL;
  for (mlsize_t i = 0; i < n; i++) t[i] = (uint64_t)Long_val(Field(a, i));
  return t;
}

/* Advances a rail's [ready] to [c] with release order. If the link sends
   nothing, it sends the rail's next transfer itself, as much of it as the
   socket takes in one call that does not wait, and hands what is left to
   the sending thread with the send claim; else the claim's release wakes
   the sending thread for it. Calls nothing of the runtime: compiled host
   code calls it through its address. */
static void rail_ready(void *counts, uint64_t c) {
  advance_count((_Atomic uint64_t *)counts, c);
  pthread_mutex_lock(&rails_mu);
  struct rail *r = live_rails;
  while (r != NULL && r->counts != counts) r = r->live_next;
  if (r == NULL || r->nsend == 0) {
    pthread_mutex_unlock(&rails_mu);
    return;
  }
  struct rig_remote_link *l = r->link;
  pthread_mutex_lock(&l->mu);
  pthread_mutex_unlock(&rails_mu);
  if (l->sending || l->writers > 0) {
    l->due = 1;
    pthread_mutex_unlock(&l->mu);
    return;
  }
  if (failing(l->job) || l->closing || r->posted >= c) {
    pthread_mutex_unlock(&l->mu);
    return;
  }
  l->sending = 1;
  r->users++;
  pthread_mutex_unlock(&l->mu);
  uint64_t next = r->posted + 1;
  unsigned char h[RAIL_HEADER];
  struct rig_remote_span s[2];
  rail_frame(r, next, r->partial, h, s);
  long k = send_some(l, s[0].p, s[0].n, s[1]);
  int whole = k == (long)(s[0].n + s[1].n);
  if (whole)
    rail_sent(r, next);
  else if (k > 0)
    r->partial += (size_t)k;
  pthread_mutex_lock(&l->mu);
  if (whole) {
    if (r->posted < c) l->due = 1;
    rail_unused(l, r);
    release_claim(l, 0);
  } else {
    l->resume = r;
    l->resume_error = k < 0 && !again((int)-k) ? (int)-k : 0;
    l->resume_ready = c;
    pthread_cond_signal(&l->wake);
  }
  pthread_mutex_unlock(&l->mu);
}

value caml_rig_remote_link_ready(value counts, value vc) {
  CAMLparam2(counts, vc);
  void *p = Caml_ba_data_val(counts);
  uint64_t c = (uint64_t)Long_val(vc);
  caml_release_runtime_system();
  rail_ready(p, c);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

value caml_rig_remote_link_ready_fn(value unit) {
  CAMLparam1(unit);
  CAMLreturn(caml_copy_nativeint((intnat)&rail_ready));
}

/* Frees [r], which no list holds. Holds the runtime. */
static void rail_free(struct rail *r) {
  caml_remove_generational_global_root(&r->areas);
  free(r->send);
  free(r->receive);
  free(r);
}

/* Registers the rail [id] on the link: [send] and [receive] hold each
   transfer's src, dst and length in turn; [out], [in] and [counts] are its
   end's areas, which the caller keeps reachable until the rail is
   released. Does not release the runtime.

   A child of fork runs none of the link's threads, and may have copied the
   link's lock held by a thread it does not have: there the link's rails
   change under [rails_mu] alone, which fork leaves free, and a new rail is
   never live, its counts the failure's. */
value caml_rig_remote_link_rail(value vl, value id, value send, value receive,
                                value out, value in, value counts) {
  CAMLparam5(vl, id, send, receive, out);
  CAMLxparam2(in, counts);
  CAMLlocal1(areas);
  struct rig_remote_link *l = Link_val(vl);
  int forked = rig_remote_forked(l->job);
#ifndef _WIN32
  pthread_once(&rails_once, rails_atfork);
  if (rails_fork_error != 0) caml_failwith("rail: cannot register fork handlers");
#endif
  areas = caml_alloc_small(3, 0);
  Store_field(areas, 0, out);
  Store_field(areas, 1, in);
  Store_field(areas, 2, counts);
  struct rail *r = calloc(1, sizeof *r);
  if (r == NULL) caml_raise_out_of_memory();
  r->link = l;
  r->id = (uint64_t)Long_val(id);
  r->send = transfers(send);
  r->receive = transfers(receive);
  if (r->send == NULL || r->receive == NULL) {
    free(r->send);
    free(r->receive);
    free(r);
    caml_raise_out_of_memory();
  }
  r->nsend = Wosize_val(send) / 3;
  r->nreceive = Wosize_val(receive) / 3;
  r->out = Caml_ba_data_val(out);
  r->in = Caml_ba_data_val(in);
  r->counts = Caml_ba_data_val(counts);
  r->out_stride = caml_ba_byte_size(Caml_ba_array_val(out)) / 2;
  r->in_stride = caml_ba_byte_size(Caml_ba_array_val(in)) / 2;
  /* The threads use the areas until the rail is released, whoever else
     holds them. */
  r->areas = areas;
  caml_register_generational_global_root(&r->areas);
  pthread_mutex_lock(&rails_mu);
  if (!forked) pthread_mutex_lock(&l->mu);
  for (struct rail *live = l->rails; live != NULL; live = live->next) {
    if (live->id != r->id) continue;
    if (!forked) pthread_mutex_unlock(&l->mu);
    pthread_mutex_unlock(&rails_mu);
    rail_free(r);
    caml_invalid_argument("Rig_remote_proxy.Link.rail: the link has this rail id");
  }
  if (forked || atomic_load(&l->failed))
    for (int c = 0; c < 3; c++) atomic_store(count(r, c), INT64_MAX);
  r->next = l->rails;
  l->rails = r;
  if (!forked) {
    r->live_next = live_rails;
    live_rails = r;
    pthread_mutex_unlock(&l->mu);
  }
  pthread_mutex_unlock(&rails_mu);
  CAMLreturn(Val_unit);
}

value caml_rig_remote_link_rail_bc(value *argv, int argc) {
  (void)argc;
  return caml_rig_remote_link_rail(argv[0], argv[1], argv[2], argv[3],
                                   argv[4], argv[5], argv[6]);
}

/* Ends the rail [id] on the link once no thread uses it. Releases the
   runtime while it waits. A child of fork unlinks it under [rails_mu]
   alone, as [caml_rig_remote_link_rail] states. */
value caml_rig_remote_link_release_rail(value vl, value id) {
  CAMLparam2(vl, id);
  struct rig_remote_link *l = Link_val(vl);
  uint64_t k = (uint64_t)Long_val(id);
  struct rail **p, *r;
  if (rig_remote_forked(l->job)) {
    pthread_mutex_lock(&rails_mu);
    for (p = &l->rails; *p != NULL && (*p)->id != k; p = &(*p)->next) {}
    r = *p;
    if (r != NULL) *p = r->next;
    pthread_mutex_unlock(&rails_mu);
    if (r != NULL) rail_free(r);
    CAMLreturn(Val_unit);
  }
  caml_release_runtime_system();
  pthread_mutex_lock(&rails_mu);
  pthread_mutex_lock(&l->mu);
  p = &l->rails;
  while (*p != NULL && (*p)->id != k) p = &(*p)->next;
  r = *p;
  if (r != NULL && r->releasing++ == 0) {
    struct rail **live = &live_rails;
    while (*live != r) live = &(*live)->live_next;
    *live = r->live_next;
  }
  pthread_mutex_unlock(&rails_mu);
  hook(RIG_REMOTE_LINK_RELEASING, r != NULL && r->users > 0);
  if (r != NULL) {
    while (r->users > 0) pthread_cond_wait(&l->cv, &l->mu);
    if (--r->releasing == 0) {
      /* Joining releases all wait for the same users. The last one owns
         unlinking and freeing; until then this id cannot be reused. */
      p = &l->rails;
      while (*p != r) p = &(*p)->next;
      *p = r->next;
    } else {
      r = NULL;
    }
  }
  pthread_mutex_unlock(&l->mu);
  caml_acquire_runtime_system();
  if (r != NULL) rail_free(r);
  CAMLreturn(Val_unit);
}

/* Stubs: guards */

/* A guard: a mutex of its own memory, which a custom block holds and
   destroys once collected. No section takes a second guard, so a thread
   holds at most one: a signal handler, finaliser or Gc.Memprof callback
   that runs while its thread holds a guard and takes one would wait for
   good on its own, or take two in no fixed order. */
#define Guard_val(v) (*(pthread_mutex_t **)Data_custom_val(v))

/* Whether this thread holds a guard. */
static _Thread_local int holding;

static void guard_finalize(value v) {
  pthread_mutex_t *m = Guard_val(v);
  if (m == NULL) return;
  pthread_mutex_destroy(m);
  free(m);
}

static struct custom_operations guard_ops = {
    "rig.remote.proxy.guard",   guard_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

value caml_rig_remote_link_guard(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v);
  v = caml_alloc_custom(&guard_ops, sizeof(pthread_mutex_t *), 0, 1);
  Guard_val(v) = NULL;
  pthread_mutex_t *m = malloc(sizeof *m);
  if (m == NULL) caml_raise_out_of_memory();
  int err = pthread_mutex_init(m, NULL);
  if (err != 0) {
    free(m);
    if (err == ENOMEM || err == EAGAIN) caml_raise_out_of_memory();
    char why[96];
    snprintf(why, sizeof why, "Rig_remote_proxy: a guard: %s", strerror(err));
    caml_failwith(why);
  }
  Guard_val(v) = m;
  CAMLreturn(v);
}

/* Takes the guard [v_g], and is true: by try-lock, and otherwise after
   releasing the runtime without running pending signal handlers, which run
   at the caller's next poll point. Is false, having taken nothing, if this
   thread holds a guard. No exception can end the take, so a caller that
   takes it as the last step before its handler gives it back on every
   exit. */
value caml_rig_remote_link_guard_take(value v_g) {
  if (holding) return Val_false;
  pthread_mutex_t *m = Guard_val(v_g);
  if (pthread_mutex_trylock(m) != 0) {
    caml_enter_blocking_section_no_pending();
    int err = pthread_mutex_lock(m);
    caml_leave_blocking_section();
    if (err != 0) {
      fprintf(stderr, "Rig_remote_proxy: a guard: %s\n", strerror(err));
      abort();
    }
  }
  holding = 1;
  return Val_true;
}

value caml_rig_remote_link_guard_give(value v_g) {
  holding = 0;
  pthread_mutex_unlock(Guard_val(v_g));
  return Val_unit;
}
