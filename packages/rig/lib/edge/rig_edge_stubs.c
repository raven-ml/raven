/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's clock, the timeline words, the count of the locks a thread
   holds and its signal mask, which rig, its drivers and their C share. */

#define _GNU_SOURCE

#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <signal.h>
#include <time.h>
#endif

#include "rig_edge.h"

uint64_t rig_now_ns(void) {
#if defined(_WIN32)
  LARGE_INTEGER count, frequency;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  uint64_t c = (uint64_t)count.QuadPart, f = (uint64_t)frequency.QuadPart;
  return c / f * 1000000000u + c % f * 1000000000u / f;
#elif defined(__APPLE__)
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
#endif
}

/* The largest page a host has: the array starts on one, so that its pages
   hold nothing else whatever the host's page. */
#define MOST_PAGE ((uintptr_t)1 << 16)

/* The process's zeroed data, resident once a word in a page is used. */
static char lines[RIG_WORDS_BYTES + MOST_PAGE];

_Atomic uint64_t *rig_words(void) {
  return (_Atomic uint64_t *)(((uintptr_t)lines + MOST_PAGE - 1) &
                              ~(MOST_PAGE - 1));
}

/* Locks */

static _Thread_local int holds;

int rig_holds(void) { return holds; }

int rig_hold(void) { return holds++; }

void rig_unhold(void) { holds--; }

/* Signals */

#if !defined(_WIN32)

/* The thread's state: [masks], the rig_mask () calls it has not yet ended,
   and [outer], its mask before the first of them; [shields], the
   rig_shield () calls among them; [window], the depth of the calls at
   which a wait has given [outer] back, or 0. */
static _Thread_local int masks, shields, window;
static _Thread_local sigset_t outer;

/* Every signal but those a fault raises, which a thread that blocks them
   cannot take, built as the library loads, before any thread masks. */
static sigset_t all;

__attribute__((constructor)) static void build_all(void) {
  static const int faults[] = {SIGSEGV, SIGBUS,  SIGFPE, SIGILL,
                               SIGTRAP, SIGABRT, SIGSYS};
  sigfillset(&all);
  for (size_t i = 0; i < sizeof faults / sizeof *faults; i++)
    sigdelset(&all, faults[i]);
}

static void block_all(sigset_t *old) {
  pthread_sigmask(SIG_BLOCK, &all, old);
}

/* Gives the thread back [outer]. A signal still pending in the kernel is
   delivered before pthread_sigmask returns, and the runtime's record of it
   makes the next poll point run its handler. */
static void give_back(void) { pthread_sigmask(SIG_SETMASK, &outer, NULL); }

/* A call that starts inside a wait's window, from a handler or a callback
   that runs there, blocks again, and its end opens the window again. A wait
   of its own closes the outer window for the rest of the outer wait, which
   stays blocked: a signal then waits for the outer wait's end. */
void rig_mask(void) {
  if (masks++ == 0) block_all(&outer);
  else if (window > 0) block_all(NULL);
}

int rig_unmask(void) {
  if (--masks == 0) {
    window = 0;
    give_back();
    return 1;
  }
  if (window == masks) give_back();
  return 0;
}

void rig_shield(void) {
  rig_mask();
  shields++;
}

int rig_unshield(void) {
  shields--;
  return rig_unmask();
}

void rig_unblock(void) {
  if (masks == 0 || shields > 0) return;
  window = masks;
  give_back();
}

void rig_block(void) {
  if (masks == 0 || window != masks) return;
  window = 0;
  block_all(NULL);
}

#else

void rig_mask(void) {}
int rig_unmask(void) { return 0; }
void rig_shield(void) {}
int rig_unshield(void) { return 0; }
void rig_unblock(void) {}
void rig_block(void) {}

#endif
