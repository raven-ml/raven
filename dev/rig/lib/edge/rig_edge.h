/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Handing work to a device from C.

   A device's driver exports two functions in the shapes rig_room_fn and
   rig_submit_fn, and the caller that holds its submissions in C calls them
   with the structures below. [self] is the driver's state for one device.
   Both functions are called one at a time per device, the room check
   before its submission, without the OCaml runtime: they call no function
   of it and read no OCaml value.

   A submission is the work of one value [v] of the device's timeline: an
   array of parts, each on one of the device's queues, and the waits on
   other devices' words it starts after. */

#ifndef RIG_EDGE_H
#define RIG_EDGE_H

#include <stddef.h>
#include <stdint.h>

/* What a room check answers: the parts fit the device's queues now; they
   fit once one of the device's values is reached; they never fit the
   device's empty queues. */
enum { RIG_FITS, RIG_LATER, RIG_NEVER };

/* What a submission answers: handed to the device; failed, and the device
   is lost. */
enum { RIG_OK, RIG_FAILED };

/* A wait's kinds: the 64-bit word at the mapped address [at] holds at
   least [value]; [at] is an object of the driver that reaches [value]. */
enum { RIG_WORD, RIG_OBJECT };

/* Which side of a copy is memory of the calling process, if any. */
enum { RIG_LOCAL_NONE, RIG_LOCAL_SRC, RIG_LOCAL_DST };

struct rig_wait {
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
     memory whose handle is [copy_dst]. Where [copy_local] is RIG_LOCAL_SRC
     or RIG_LOCAL_DST, that side is memory of the calling process instead:
     its handle field holds the host address of the memory, and its offset
     field the offset into it. Only a driver of a device of another machine
     is handed one.
   The part runs after the parts of the same submission before it on its
   queue, and after the parts whose indices the [nafter] ints at [after]
   list, each below its own: [after] orders parts of different queues. */
struct rig_part {
  int queue;
  const uint32_t *words;
  size_t n;
  int (*fill)(void *queue, void *arg, uint64_t v);
  void *arg;
  size_t ring_units, segment_bytes;
  uint64_t copy_dst, copy_dst_offset, copy_src, copy_src_offset, copy_bytes;
  int copy_local;
  const int *after;
  int nafter;
};

/* Answers RIG_FITS, RIG_LATER or RIG_NEVER for the [n] parts at [parts]. */
typedef int rig_room_fn(void *self, const struct rig_part *parts, int n);

/* Hands [parts] to the device as the work of [v], the value after the last
   one it received, which starts after [waits]. [handles] lists [self]'s
   own memory the work uses, for a driver whose submissions name it; a
   copy's side that [copy_local] names is not in it. Answers RIG_OK, or
   RIG_FAILED with [*failure] set to the driver's message, which lives as
   long as the device. */
typedef int rig_submit_fn(void *self, uint64_t v, const struct rig_wait *waits,
                         int nwaits, const struct rig_part *parts, int nparts,
                         const uint64_t *handles, int nhandles,
                         const char **failure);

#endif
