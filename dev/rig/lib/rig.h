/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Reading buffers from C.

   A library that computes on host memory in C, such as nx's CPU kernels,
   reads a buffer (an OCaml value of type Rig.Buffer.t) through these
   functions. Each may be called with the domain lock held only, and each
   but rig_buffer_wait reads the value without allocating. The address a
   buffer gives stays valid while the value is reachable. A function that
   releases the domain lock registers each buffer it reads as a root before,
   and claims and releases only while holding the lock: the claim lives in
   the OCaml heap, which a compaction moves. */

#ifndef RIG_H
#define RIG_H

#include <stddef.h>
#include <caml/mlvalues.h>

/* The host address of [b]'s first byte, or NULL if the host does not
   address [b]'s memory. */
void *rig_buffer_host(value b);

/* The number of [b]'s bytes. */
size_t rig_buffer_bytes(value b);

/* The reason the consumption of [b] gave if [b] is dead, as a C string in
   the OCaml heap; NULL if [b] is live. Like String_val's, the pointer holds
   only until the caller next allocates, runs OCaml code or releases the
   domain lock, any of which may move the string. A caller that keeps the
   reason copies its bytes first, into memory of its own or into an OCaml
   string it allocated, reading the reason again after that allocation. */
const char *rig_buffer_why(value b);

/* Claims

   Host code that reads or writes a buffer's bytes claims its memory
   first, as Rig.Claim.read does, so that no donation on another domain
   holds it exclusive while the code runs. It then reads or writes if the
   claim answers RIG_CLAIMED, and after rig_buffer_wait if it answers
   RIG_WAIT. Memory the caller holds exclusive (Rig.Claim.with_) is claimed
   already; claiming it again answers RIG_EXCLUSIVE. */

enum rig_access { RIG_READ, RIG_READ_WRITE };

enum rig_claim {
  RIG_CLAIMED,   /* claimed, and the work the access must follow is done */
  RIG_WAIT,      /* claimed: call rig_buffer_wait before the access */
  RIG_DEAD,      /* not claimed: rig_buffer_why gives the reason */
  RIG_EXCLUSIVE, /* not claimed: the memory is held exclusive */
  RIG_READ_ONLY  /* not claimed: RIG_READ_WRITE on Read memory */
};

/* Claims [b]'s memory for [access]. It claims, then checks [b] under the
   claim, and undoes the claim on RIG_DEAD, RIG_EXCLUSIVE and
   RIG_READ_ONLY. It is RIG_CLAIMED if the work submitted before the claim
   that the access must follow is done, as Rig.Buffer.wait states it; work
   submitted after it is the caller's to exclude. It is RIG_WAIT, holding
   the claim, if a point the access must follow is not reached as the host
   last read its device's word, or if [b]'s memory is a lost device's as
   Rig.Lost states it. A buffer just made is no exception: memory a device
   reuses carries its earlier uses. */
enum rig_claim rig_buffer_claim(value b, enum rig_access access);

/* Waits for the work that an access of [b] for [access] must follow, as
   Rig.Buffer.wait does, for a caller that holds a claim rig_buffer_claim
   answered RIG_WAIT. The wait runs OCaml code: driver calls, signal
   handlers, the collector, other threads. So it is never called from a
   [@@noalloc] external, and every value its caller uses after it, [b]
   included, is a registered root. It answers unit, or the exception the
   wait raised, Rig.Lost or Sys.Break among others; the claim stays held
   either way. */
caml_result rig_buffer_wait(value b, enum rig_access access);

/* Ends a claim that rig_buffer_claim made; [b] may be dead. Ending a
   claim the caller does not hold ends another reader's. */
void rig_buffer_release(value b);

#endif
