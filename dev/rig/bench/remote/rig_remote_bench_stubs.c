/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The floors of the remote bench, and the caller's half of a rail's run.

   A floor moves the bytes a row moves over a socket set up as a link sets
   up its own, with plain blocking sends and receives and nothing else: no
   frames, no queue, no threads but the one that receives. A rail's run
   stores [ready] and waits for [arrived], as work on either machine does.
   Every call that waits releases the runtime. A failure crosses as a
   negated code: errno, or WSAGetLastError on Windows. */

#define _GNU_SOURCE

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

#ifdef _WIN32
#include <winsock2.h>
typedef SOCKET sock;
#define sock_val(v) Socket_val(v)
#define NOSIGNAL 0
#define SHUT_BOTH SD_BOTH
#else
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
typedef int sock;
#define sock_val(v) Int_val(v)
#ifdef MSG_NOSIGNAL
#define NOSIGNAL MSG_NOSIGNAL
#else
#define NOSIGNAL 0
#endif
#define SHUT_BOTH SHUT_RDWR
#endif

static intnat sock_error(void) {
#ifdef _WIN32
  return -(intnat)WSAGetLastError();
#else
  return -(intnat)errno;
#endif
}

static int interrupted(void) {
#ifdef _WIN32
  return 0;
#else
  return errno == EINTR;
#endif
}

/* Sends the [n] bytes at [p]: 0, or a negated code. */
static intnat send_all(sock s, const char *p, size_t n) {
  while (n > 0) {
    int chunk = n > (1u << 30) ? (1 << 30) : (int)n;
    long k = (long)send(s, p, chunk, NOSIGNAL);
    if (k < 0 && interrupted()) continue;
    if (k < 0) return sock_error();
    p += k;
    n -= (size_t)k;
  }
  return 0;
}

/* Receives [n] bytes into [p]: 0, [1] if the stream ended, or a negated
   code. */
static intnat recv_all(sock s, char *p, size_t n) {
  while (n > 0) {
    int chunk = n > (1u << 30) ? (1 << 30) : (int)n;
    long k = (long)recv(s, p, chunk, 0);
    if (k == 0) return 1;
    if (k < 0 && interrupted()) continue;
    if (k < 0) return sock_error();
    p += k;
    n -= (size_t)k;
  }
  return 0;
}

/* Waits until [*count >= c], read with acquire order. */
static void await(_Atomic uint64_t *count, uint64_t c) {
  while (atomic_load_explicit(count, memory_order_acquire) < c) sched_yield();
}

/* [tune fd] sets on [fd] the options a link sets on its socket: no delay
   for small segments and, where the system needs an option for it, no
   SIGPIPE. Does not release the runtime. */
value rig_remote_bench_tune(value fd) {
  sock s = sock_val(fd);
  int one = 1;
  (void)setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&one,
                   sizeof one);
#ifdef SO_NOSIGPIPE
  (void)setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
  return Val_unit;
}

/* Round trips: a request's bytes out and its answer's back, through the
   bytes of [buf], at least as many as either. */

/* [ask fd buf out back] sends [out] bytes on [fd] and receives [back]: 0, [1]
   if the stream ended, or a negated code. Releases the runtime. */
value rig_remote_bench_ask(value fd, value buf, value vout, value vback) {
  sock s = sock_val(fd);
  char *p = Caml_ba_data_val(buf);
  size_t out = (size_t)Long_val(vout), back = (size_t)Long_val(vback);
  caml_release_runtime_system();
  intnat r = send_all(s, p, out);
  if (r == 0) r = recv_all(s, p, back);
  caml_acquire_runtime_system();
  return Val_long(r);
}

/* [echo fd buf in out] answers each [in] bytes received on [fd] with [out]
   bytes, until the stream ends: [1] then, or a negated code. Releases the
   runtime. */
value rig_remote_bench_echo(value fd, value buf, value vin, value vout) {
  sock s = sock_val(fd);
  char *p = Caml_ba_data_val(buf);
  size_t in = (size_t)Long_val(vin), out = (size_t)Long_val(vout);
  caml_release_runtime_system();
  intnat r = 0;
  while (r == 0) {
    r = recv_all(s, p, in);
    if (r == 0) r = send_all(s, p, out);
  }
  caml_acquire_runtime_system();
  return Val_long(r);
}

/* Streams: one way, [n] bytes per run, a thread receiving each run whole and
   counting it, as a rail's receiving thread places a transfer and stores
   [arrived]. */

struct stream {
  sock out, in;
  size_t n;
  char *src, *dst;
  uint64_t posted;
  _Atomic uint64_t arrived; /* UINT64_MAX once the stream ended */
  pthread_t thread;
};

static void *receive(void *arg) {
  struct stream *st = arg;
  for (;;) {
    if (recv_all(st->in, st->dst, st->n) != 0) break;
    atomic_fetch_add_explicit(&st->arrived, 1, memory_order_release);
  }
  atomic_store_explicit(&st->arrived, UINT64_MAX, memory_order_release);
  return NULL;
}

/* [stream_open out in n] is a stream of [n > 0] bytes a run from [out] to
   [in], the two ends of one connection, or 0 if memory or a thread ran out.
   Does not release the runtime. */
value rig_remote_bench_stream_open(value vout, value vin, value vn) {
  struct stream *st = calloc(1, sizeof *st);
  size_t n = (size_t)Long_val(vn);
  if (st == NULL) return caml_copy_nativeint(0);
  st->out = sock_val(vout);
  st->in = sock_val(vin);
  st->n = n;
  st->src = malloc(n);
  st->dst = malloc(n);
  if (st->src != NULL) memset(st->src, 1, n);
  if (st->src == NULL || st->dst == NULL ||
      pthread_create(&st->thread, NULL, receive, st) != 0) {
    free(st->src);
    free(st->dst);
    free(st);
    return caml_copy_nativeint(0);
  }
  return caml_copy_nativeint((intnat)st);
}

/* [stream_run st] sends a run's bytes and waits until they arrived: 0, [1]
   if the stream ended, or a negated code. Releases the runtime. */
value rig_remote_bench_stream_run(value vst) {
  struct stream *st = (struct stream *)Nativeint_val(vst);
  caml_release_runtime_system();
  intnat r = send_all(st->out, st->src, st->n);
  if (r == 0) {
    st->posted++;
    await(&st->arrived, st->posted);
    if (atomic_load(&st->arrived) == UINT64_MAX) r = 1;
  }
  caml_acquire_runtime_system();
  return Val_long(r);
}

/* [stream_close st] shuts the connection down, ends the receiving thread and
   frees [st]; the caller closes the sockets. Releases the runtime. */
value rig_remote_bench_stream_close(value vst) {
  struct stream *st = (struct stream *)Nativeint_val(vst);
  caml_release_runtime_system();
  shutdown(st->out, SHUT_BOTH);
  shutdown(st->in, SHUT_BOTH);
  pthread_join(st->thread, NULL);
  caml_acquire_runtime_system();
  free(st->src);
  free(st->dst);
  free(st);
  return Val_unit;
}

/* Rails */

/* A rail end's [arrived], at byte 256 of its counts. */
#define ARRIVED 256

/* An end's ready function (Rig_remote_abi.end_'s [ready_fn]). */
typedef void ready_fn(void *arg, uint64_t c);

/* [rail_run fn arg receiver c] advances the sending end's [ready] to [c]
   through its ready function [fn] and [arg], and waits until the receiving
   end's [arrived], in the counts [receiver], reaches [c]: 0, or [1] if the
   job failed meanwhile. Releases the runtime. */
value rig_remote_bench_rail_run(value vfn, value varg, value vreceiver,
                                value vc) {
  ready_fn *fn = (ready_fn *)Nativeint_val(vfn);
  void *arg = (void *)Nativeint_val(varg);
  unsigned char *r = Caml_ba_data_val(vreceiver);
  uint64_t c = (uint64_t)Long_val(vc);
  _Atomic uint64_t *arrived = (_Atomic uint64_t *)(r + ARRIVED);
  caml_release_runtime_system();
  fn(arg, c);
  await(arrived, c);
  caml_acquire_runtime_system();
  return Val_long(atomic_load(arrived) == (uint64_t)INT64_MAX ? 1 : 0);
}
