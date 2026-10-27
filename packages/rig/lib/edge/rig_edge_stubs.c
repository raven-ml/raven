/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's clock, the timeline words, the count of the locks a thread
   holds and its signal mask, which rig, its drivers and their C share. */

#define _GNU_SOURCE

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>

/* rearm () below needs caml_check_pending_signals and
   caml_set_action_pending, which caml/signals.h declares only for the
   runtime's own code and its libraries. memory.h includes signals.h, so it
   comes after. */
#define CAML_INTERNALS
#include <caml/signals.h>
#undef CAML_INTERNALS

#include <caml/memory.h>

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

intnat caml_rig_edge_now(value unit) {
  (void)unit;
  return (intnat)rig_now_ns();
}

value caml_rig_edge_now_byte(value unit) {
  return Val_long(caml_rig_edge_now(unit));
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

/* rig's array of words: its address and its bytes. */
value caml_rig_edge_words(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v);
  v = caml_alloc_tuple(2);
  Store_field(v, 0, Val_long((intnat)rig_words()));
  Store_field(v, 1, Val_long((intnat)RIG_WORDS_BYTES));
  CAMLreturn(v);
}

/* Locks */

static _Thread_local int holds;

int rig_holds(void) { return holds; }

int rig_hold(void) { return holds++; }

void rig_unhold(void) { holds--; }

intnat caml_rig_edge_holds(value unit) {
  (void)unit;
  return holds;
}

value caml_rig_edge_holds_byte(value unit) {
  return Val_long(caml_rig_edge_holds(unit));
}

value caml_rig_edge_hold(value unit) {
  (void)unit;
  rig_hold();
  return Val_unit;
}

value caml_rig_edge_unhold(value unit) {
  (void)unit;
  rig_unhold();
  return Val_unit;
}

/* Signals */

#if !defined(_WIN32)

/* The thread's state: [masks], the rig_mask () calls it has not yet ended,
   and [outer], its mask before the first of them; [shields], the
   rig_shield () calls among them; [window], the depth of the calls at
   which a wait has given [outer] back, or 0. */
static _Thread_local int masks, shields, window;
static _Thread_local sigset_t outer;

/* Blocks every signal but those a fault raises, which a thread that blocks
   them cannot take. */
static void block_all(sigset_t *old) {
  static const int faults[] = {SIGSEGV, SIGBUS,  SIGFPE, SIGILL,
                               SIGTRAP, SIGABRT, SIGSYS};
  sigset_t s;
  sigfillset(&s);
  for (size_t i = 0; i < sizeof faults / sizeof *faults; i++)
    sigdelset(&s, faults[i]);
  pthread_sigmask(SIG_BLOCK, &s, old);
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

/* Ends a pair: whether it was the outermost. */
static int unmasked(void) {
  if (--masks == 0) {
    window = 0;
    give_back();
    return 1;
  }
  if (window == masks) give_back();
  return 0;
}

void rig_unmask(void) { (void)unmasked(); }

void rig_shield(void) {
  rig_mask();
  shields++;
}

static int unshielded(void) {
  shields--;
  return unmasked();
}

void rig_unshield(void) { (void)unshielded(); }

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

static int unmasked(void) { return 0; }
static int unshielded(void) { return 0; }
void rig_mask(void) {}
void rig_unmask(void) {}
void rig_shield(void) {}
void rig_unshield(void) {}
void rig_unblock(void) {}
void rig_block(void) {}

#endif

value caml_rig_edge_mask(value unit) {
  (void)unit;
  rig_mask();
  return Val_unit;
}

/* The outermost end also makes the next poll point look at the signals
   the runtime recorded: one recorded while the thread blocked it was
   skipped by the poll points since, each clearing what makes the next one
   look. rearm () sets that again while any is recorded, the step with which
   caml_leave_blocking_section ends, without releasing the runtime, which
   was there only for that step. systhreads' Thread.yield does the same
   after its switch (otherlibs/systhreads/st_stubs.c, thread_yield). The
   external is not noalloc, so its return arms the next poll point to run
   the handler. A wait's end needs none of this: its blocking step leaves a
   blocking section, and caml_rig_edge_block runs the handlers. */
static void rearm(void) {
  if (caml_check_pending_signals()) caml_set_action_pending(Caml_state);
}

value caml_rig_edge_unmask(value unit) {
  (void)unit;
  if (unmasked()) rearm();
  return Val_unit;
}

value caml_rig_edge_unblock(value unit) {
  (void)unit;
  rig_unblock();
  return Val_unit;
}

/* A wait's end runs the handlers of the signals that came during it, before
   it blocks them again, so that their exceptions end the wait. */
value caml_rig_edge_block(value unit) {
  (void)unit;
  caml_result r = caml_process_pending_actions_res();
  rig_block();
  return caml_get_value_or_raise(r);
}

value caml_rig_edge_shield(value unit) {
  (void)unit;
  rig_shield();
  return Val_unit;
}

value caml_rig_edge_unshield(value unit) {
  (void)unit;
  if (unshielded()) rearm();
  return Val_unit;
}
