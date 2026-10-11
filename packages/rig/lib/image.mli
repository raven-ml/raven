(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Images: the implementation of {!Rig.Image}.

    A driver loads a binary in one of two ways ({!Rig_edge.type-code}). It
    loads it itself and returns its image, or it asks for code memory of a
    size and lays the binary out into it. In the second case {!load} allocates
    that memory as [Device] memory of the device ({!Memory.alloc_entry}), so
    it counts in the budget, and places the code after the device's queued
    work: by a copy from [Pinned] memory on the copy engine ({!Copy.queued}),
    or by the host where the device has no copy engine. A failure while laying
    out or placing the code retires the code memory ({!Memory.retire}), after
    unloading the image if the driver made one.

    An image's token puts the driver's image and its code memory on the
    device's release list once the image is collected. A drain then unloads
    the image after the work submitted on the device until then is done, and
    the code memory goes to the cache or back to the driver
    ({!Memory.drain}). A submission keeps the images of its launches
    reachable, since its prepared form names their code. *)

type t = Def.image
(** {!Rig.Image.t}. *)

val load : Def.device -> string -> (t, string) result
(** [load d binary] is {!Rig.Image.load}. It records a [Load] event in the
    profiles being taken. *)

val device : t -> Def.device
(** [device i] is the device [i] is loaded on. *)

val mem : t -> string -> bool
(** [mem i f] is {!Rig.Image.mem}: whether the driver has an entry for [f]
    ({!Memory.kernel_entry}). *)

val entry : t -> string -> int option
(** [entry i f] is {!Rig.Image.entry}: the [code] of the driver's entry for
    [f]. *)

val params : t -> string -> int option
(** [params i f] is {!Rig.Image.params}: the [params] of the driver's entry
    for [f]. *)

val threads : t -> string -> int option
(** [threads i f] is {!Rig.Image.threads}: the [threads] of the driver's entry
    for [f]. *)
