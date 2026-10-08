/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Reading buffers from C.

   A library that computes on host memory in C, such as nx's CPU kernels,
   reads a buffer (an OCaml value of type Rig.Buffer.t) through these
   functions. Each reads the value without allocating and may be called with
   the domain lock held only. The address a buffer gives stays valid while the
   value is reachable. A function that releases the domain lock registers
   each buffer it reads as a root before, and claims and releases only while
   holding the lock: the claim lives in the OCaml heap, which a compaction
   moves. */

#ifndef RIG_H
#define RIG_H

#include <stddef.h>
#include <caml/mlvalues.h>

/* The host address of [b]'s first byte, or NULL if the host does not
   address [b]'s memory. */
void *rig_buffer_host(value b);

/* The number of [b]'s bytes. */
size_t rig_buffer_bytes(value b);

/* The reason the consumption of [b] gave if [b] is dead, as a C string that
   lives while [b] is reachable; NULL if [b] is live. */
const char *rig_buffer_why(value b);

/* Claims

   Host code that reads or writes a buffer's bytes claims its memory
   first, as Rig.Claim.read does, so that no donation on another domain
   holds it exclusive while the code runs, and then reads or writes only
   if the claim answers RIG_CLAIMED. Memory the caller holds exclusive
   (Rig.Claim.with_) is claimed already; claiming it again answers
   RIG_EXCLUSIVE. */

enum rig_access { RIG_READ, RIG_READ_WRITE };

enum rig_claim {
  RIG_CLAIMED,   /* claimed */
  RIG_PENDING,   /* not claimed: call Rig.Buffer.wait, then claim again */
  RIG_DEAD,      /* not claimed: rig_buffer_why gives the reason */
  RIG_EXCLUSIVE, /* not claimed: the memory is held exclusive */
  RIG_READ_ONLY  /* not claimed: RIG_READ_WRITE on Read memory */
};

/* Claims [b]'s memory for [access]. It is RIG_CLAIMED once the work
   submitted before the claim that the access must follow is done, as
   Rig.Buffer.wait states it; work submitted after it is the caller's to
   exclude. It claims, then checks [b] under the claim, and undoes the
   claim on every other answer. RIG_PENDING means that a point the access
   must follow is not reached as the host last read its device's word, or
   that [b]'s memory is a lost device's as Rig.Lost states it:
   Rig.Buffer.wait waits for it or raises Lost, and a claim with no work
   submitted since is RIG_CLAIMED. A buffer just made is no exception:
   memory a device reuses carries its earlier uses. */
enum rig_claim rig_buffer_claim(value b, enum rig_access access);

/* Ends a claim that rig_buffer_claim made; [b] may be dead. Ending a
   claim the caller does not hold ends another reader's. */
void rig_buffer_release(value b);

#endif
