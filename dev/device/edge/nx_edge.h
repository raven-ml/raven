/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Handing work to a device from C.

   A device's driver exports two functions in the shapes nx_room_fn and
   nx_submit_fn, and the caller that holds its submissions in C calls them
   with the structures below. [self] is the driver's state for one device.
   Both functions are called one at a time per device, the room check
   before its submission, without the OCaml runtime: they call no function
   of it and read no OCaml value.

   A submission is the work of one value [v] of the device's timeline: an
   array of parts, each on one of the device's queues, and the waits on
   other devices' words it starts after. */

#ifndef NX_EDGE_H
#define NX_EDGE_H

#include <stddef.h>
#include <stdint.h>

/* What a room check answers: the parts fit the device's queues now; they
   fit once one of the device's values is reached; they never fit the
   device's empty queues. */
enum { NX_FITS, NX_LATER, NX_NEVER };

/* What a submission answers: handed to the device; failed, and the device
   is lost. */
enum { NX_OK, NX_FAILED };

/* A wait's kinds: the 64-bit word at the mapped address [at] holds at
   least [value]; [at] is an object of the driver that reaches [value]. */
enum { NX_WORD, NX_OBJECT };

struct nx_wait {
  uint64_t at, value;
  int kind;
};

/* One part of a submission, on the queue at index [queue] of the driver's
   queue list. A part is one of three works, the others' fields zero:
   - words: the [n] words at [words], placed on the queue;
   - a fill: [fill] called with the queue's context, [arg] and [v], which
     returns 0 or a failure. [ring_units] and [segment_bytes] bound what it
     writes, for a driver whose queue is a ring;
   - a copy: [copy_bytes] bytes from [copy_src_offset] bytes into the
     memory whose handle is [copy_src] to [copy_dst_offset] bytes into the
     memory whose handle is [copy_dst].
   The part runs after the parts of the same submission before it on its
   queue, and after the parts whose indices the [nafter] ints at [after]
   list, each below its own: [after] orders parts of different queues. */
struct nx_part {
  int queue;
  const uint32_t *words;
  size_t n;
  int (*fill)(void *queue, void *arg, uint64_t v);
  void *arg;
  size_t ring_units, segment_bytes;
  uint64_t copy_dst, copy_dst_offset, copy_src, copy_src_offset, copy_bytes;
  const int *after;
  int nafter;
};

/* Answers NX_FITS, NX_LATER or NX_NEVER for the [n] parts at [parts]. */
typedef int nx_room_fn(void *self, const struct nx_part *parts, int n);

/* Hands [parts] to the device as the work of [v], the value after the last
   one it received, which starts after [waits]. [handles] lists the memory
   the work uses, for a driver whose submissions name it. Answers NX_OK, or
   NX_FAILED with [*failure] set to the driver's message, which lives as
   long as the device. */
typedef int nx_submit_fn(void *self, uint64_t v, const struct nx_wait *waits,
                         int nwaits, const struct nx_part *parts, int nparts,
                         const uint64_t *handles, int nhandles,
                         const char **failure);

#endif
