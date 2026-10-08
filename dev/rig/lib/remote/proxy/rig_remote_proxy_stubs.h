/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the library's stub files share: sockets as C sees them, frames, and
   the C state of jobs, links and proxies. */

#ifndef RIG_REMOTE_PROXY_STUBS_H
#define RIG_REMOTE_PROXY_STUBS_H

#include <caml/mlvalues.h>
#include <caml/unixsupport.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
typedef SOCKET rig_remote_sock;
#define rig_remote_sock_val(v) Socket_val(v)
#define RIG_REMOTE_NOSIGNAL 0
#else
#include <sys/socket.h>
typedef int rig_remote_sock;
#define rig_remote_sock_val(v) Int_val(v)
#ifdef MSG_NOSIGNAL
#define RIG_REMOTE_NOSIGNAL MSG_NOSIGNAL
#else
#define RIG_REMOTE_NOSIGNAL 0
#endif
#endif

/* The last socket error of the calling thread. */
static inline int rig_remote_sock_error(void) {
#ifdef _WIN32
  return WSAGetLastError();
#else
  return errno;
#endif
}

/* Keeps sends on [s] from raising SIGPIPE where the system needs a socket
   option for that (macOS); elsewhere sends pass RIG_REMOTE_NOSIGNAL. */
static inline void rig_remote_quiet(rig_remote_sock s) {
#ifdef SO_NOSIGPIPE
  int one = 1;
  (void)setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#else
  (void)s;
#endif
}

/* Sends the [n] bytes at [p] on [s], all of them: 0, or the socket error. It
   blocks and calls nothing of the runtime. */
static inline int rig_remote_send_all(rig_remote_sock s, const void *p,
                                      size_t n) {
  const char *c = p;
  while (n > 0) {
    int chunk = n > (1u << 30) ? (1 << 30) : (int)n;
    long k = (long)send(s, c, chunk, RIG_REMOTE_NOSIGNAL);
    if (k < 0) {
      int e = rig_remote_sock_error();
#ifndef _WIN32
      if (e == EINTR) continue;
#endif
      return e;
    }
    c += k;
    n -= (size_t)k;
  }
  return 0;
}

/* Raises Unix.Unix_error for the socket error [e] of [op]. */
static inline void rig_remote_raise_error(int e, const char *op) {
#ifdef _WIN32
  caml_win32_maperr(e);
#else
  errno = e;
#endif
  caml_uerror(op, Nothing);
}

/* Little-endian integers */

static inline uint64_t rig_remote_get_u64(const unsigned char *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
  return v;
}

static inline void rig_remote_put_u64(unsigned char *p, uint64_t v) {
  for (int i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (8 * i));
}

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

/* Jobs, links and proxies (rig_remote_link.c, rig_remote_proxy.c)

   A job and its links are never freed. A job's lock guards its list of
   links; a link's lock guards everything mutable of the link and of its
   proxies' C state. The job's lock is taken before a link's. */

enum { OPEN, CLOSED, FAILED };

struct entry;
struct pending;
struct cmd;
struct rail;
struct rig_remote_dev;

struct rig_remote_job {
  pthread_mutex_t mu;
  pthread_cond_t cv; /* the state changed, or a link ended */
  _Atomic int state;
  _Atomic(char *) why; /* set before the state */
  long pid;
  struct rig_remote_link *links;
};

struct rig_remote_link {
  struct rig_remote_job *job;
  struct rig_remote_link *next;
  rig_remote_sock fd;
  char *name;
  int peer; /* 0 for the controller, i for agent i */
  pthread_mutex_t mu;
  pthread_cond_t cv; /* any change: queue, answers, commands, words */
  struct entry *head, *tail;
  size_t queued;
  int sending; /* a frame is being sent, by the thread or an abort */
  int closing, sent_close, got_close, threads, fd_closed;
  int receiving; /* the receiving thread runs: copies' bytes may land */
  _Atomic int failed;
  struct pending *pending, *pending_last;
  struct cmd *cmds, *cmds_last;
  struct rail *rails;
  struct rig_remote_dev **devs; /* the proxies, by their agent's id */
  size_t ndevs;
  int64_t sent_ns;
};

/* A copy into this process's memory whose bytes are to come. */
struct rig_remote_local {
  struct rig_remote_local *next;
  uint64_t value;
  unsigned char *at;
  uint64_t bytes;
};

/* The work of a value handed over and not reached: its bytes. */
struct rig_remote_flight {
  struct rig_remote_flight *next;
  uint64_t value, bytes;
};

/* A proxy's C state. Every field but [word] is guarded by its link's
   lock; [word] is written there too, and read without it. */
struct rig_remote_dev {
  struct rig_remote_link *link;
  uint64_t id;
  _Atomic uint64_t word; /* the shadow */
  uint64_t handed;       /* the last value handed over */
  uint64_t written;      /* the last value that copies into local memory */
  uint64_t flying;       /* bytes of [flights] */
  int stopped; /* its stop ran: the word takes [handed] once no copy's
                  bytes may land */
  struct rig_remote_local *locals, *locals_last;
  struct rig_remote_flight *flights, *flights_last;
};

/* [1] if the calling process did not make [j], whose job it then fails in
   its own copy, taking no lock. */
int rig_remote_forked(struct rig_remote_job *j);

/* Queues a frame of [kind] whose payload is the [n] bytes at [p], waiting
   while the queue is full: [0]; [-1] if the job failed, or [-3] if the link
   closes, the frame then dropped; [-2] if memory ran out. Called without the runtime, and
   without the link's lock. */
int rig_remote_queue(struct rig_remote_link *l, int kind,
                     const unsigned char *p, size_t n, struct pending *q);

/* Writes [handed] into a stopped proxy's word once no copy into this
   process's memory may still land: none is pending, or the receiving thread
   ended. Holds the link's lock. */
void rig_remote_settle(struct rig_remote_dev *d);

/* Adds [d] to its link's proxies: [0]; [-1] if the link has a proxy of
   [d]'s id; [-2] if memory ran out. */
int rig_remote_add_dev(struct rig_remote_dev *d);

#define Link_c(v) ((struct rig_remote_link *)Nativeint_val(Field(v, 0)))

#endif
