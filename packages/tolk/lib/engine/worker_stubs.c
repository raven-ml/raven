/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The calling thread's signal mask, for Worker. Unix.sigprocmask runs the
   pending signal handlers before it changes the mask, and its result
   allocates: each is a point at which a handler's exception can cut a call
   short. Blocking here runs no handler and allocates nothing, so that the
   code between two of these calls runs whole. A mask travels as the bits of
   signals 1 to 64, every signal of Linux and macOS. Windows has no signal
   masks: there the mask is 0 and nothing changes. */

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include <stdint.h>

#ifndef _WIN32
#include <pthread.h>
#include <signal.h>

#define SIGNALS 64

/* Every signal but those a fault raises, which a thread that blocks them
   cannot take. */
static sigset_t all;
static pthread_once_t built = PTHREAD_ONCE_INIT;

static void build_all(void)
{
  /* The faults rig's threads leave unblocked too (rig_edge_stubs.c); tolk
     does not depend on rig, so the list is not shared. */
  static const int faults[] = {SIGSEGV, SIGBUS,  SIGFPE, SIGILL,
                               SIGTRAP, SIGABRT, SIGSYS};
  sigfillset(&all);
  for (size_t i = 0; i < sizeof faults / sizeof *faults; i++)
    sigdelset(&all, faults[i]);
}

static int64_t bits_of(const sigset_t *set)
{
  uint64_t bits = 0;
  for (int s = 1; s <= SIGNALS; s++)
    if (sigismember(set, s) == 1) bits |= (uint64_t)1 << (s - 1);
  return (int64_t)bits;
}

static void set_of(int64_t bits, sigset_t *set)
{
  sigemptyset(set);
  for (int s = 1; s <= SIGNALS; s++)
    if ((uint64_t)bits & ((uint64_t)1 << (s - 1))) sigaddset(set, s);
}
#endif

/* [tolk_worker_block ()] blocks every signal but a fault's on the calling
   thread and is its mask before. */
int64_t tolk_worker_block(value unit)
{
  (void)unit;
#ifdef _WIN32
  return 0;
#else
  sigset_t old;
  pthread_once(&built, build_all);
  pthread_sigmask(SIG_BLOCK, &all, &old);
  return bits_of(&old);
#endif
}

value tolk_worker_block_byte(value unit)
{
  return caml_copy_int64(tolk_worker_block(unit));
}

/* [tolk_worker_restore mask] sets the calling thread's mask to [mask]. A
   signal it unblocks that the kernel held is delivered before
   pthread_sigmask returns. One the runtime recorded while the thread blocked
   it was skipped by the poll points since, each clearing what makes the
   next one look: leaving a blocking section sets that again while any is
   recorded. The external is not noalloc, so its return arms the next poll
   point to run the handlers. */
value tolk_worker_restore(int64_t bits)
{
#ifdef _WIN32
  (void)bits;
#else
  sigset_t set;
  set_of(bits, &set);
  pthread_sigmask(SIG_SETMASK, &set, NULL);
  caml_enter_blocking_section_no_pending();
  caml_leave_blocking_section();
#endif
  return Val_unit;
}

value tolk_worker_restore_byte(value bits)
{
  return tolk_worker_restore(Int64_val(bits));
}
