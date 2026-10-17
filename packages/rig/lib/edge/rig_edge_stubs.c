/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The host's clock, the timeline words and the count of the locks a thread
   holds, which rig, its drivers and their C share. */

#define _GNU_SOURCE

#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#ifdef _WIN32
#include <windows.h>
#else
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
