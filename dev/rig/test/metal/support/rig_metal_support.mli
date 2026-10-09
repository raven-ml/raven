(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Metal suite and bench share. *)

(** {1:gpu The GPU} *)

include Rig_gpu_support.Conformance with module D = Rig_metal
(** Metal's device [0]; its second device is device [0] opened again. *)

(** {1:ring A device's ring, by hand}

    The ring of a device's command buffers, driven without Metal: the suite
    plays the submitter, which commits command buffers, and Metal's handler,
    which completes them in any order. Commit [k] (from [0]) takes the value
    after the last one taken when it ends a submission, [0] otherwise, and
    completes with the times [10k + 1] and [10k + 2]. *)

type ring
(** The type for rings driven by hand. *)

val ring : int -> ring
(** [ring n] is a ring of [n] slots, [1 <= n <= 8], whose word reads [0]. *)

val commit : ring -> last:bool -> int
(** [commit r ~last] takes a slot of [r] and commits its command buffer, which
    ends a submission iff [last]. It is the slot's index. The ring has a free
    slot. *)

val complete : ring -> int -> failed:bool -> unit
(** [complete r i ~failed] completes the command buffer of slot [i], which
    failed with the message ["command buffer k failed"] iff [failed]. *)

val word : ring -> int
(** [word r] is the value in [r]'s word. *)

val times : ring -> int -> int * int
(** [times r k] is the start and end times written for commit [k], [(0, 0)]
    until written. *)

val failure : ring -> string option
(** [failure r] is the first failure the ring recorded. *)

val sleep : ring -> string option
(** [sleep r] is what the ring's sleep answers when the word changes at once. *)

val stop : ring -> bool
(** [stop r] is [true] after writing the last value taken into the word, iff no
    slot is taken. *)

(** {1:fills Fills} *)

type fill
(** The type for fills and their arguments, which live as long as the value. *)

val part : fill -> Rig.Submission.part
(** [part f] is [f] as a part of a Metal device's queue, its argument a host
    buffer. *)

val failing : int -> fill
(** [failing code] returns [code] and encodes nothing. *)

val dispatch :
  pipeline:int ->
  ?offset:int ->
  Rig_metal.region ->
  groups:int ->
  threads:int ->
  fill
(** [dispatch ~pipeline ~offset args ~groups ~threads] dispatches [pipeline]
    over [groups] threadgroups of [threads] threads, with [args] at [offset]
    (defaults to [0]) as its kernel buffer [0]. *)

val split : fill -> Rig_metal.t -> int -> times:int -> unit
(** [split f d k ~times] makes the dispatch [f] split [k] times through [d]'s
    [split], dispatching again after each, and write the times of split [i] at
    the 64-bit words [2i] and [2i + 1] at [times], unless [times] is [0]. *)

val execute : Rig_metal_abi.icb -> fill
(** [execute b] runs every command of [b]. *)

val watching : unit -> fill * nativeint
(** [watching ()] is [(f, w)] with [f] a fill that encodes nothing and makes [w]
    a weak reference to its command buffer, read with {!alive}. *)

val resize : nativeint -> groups:int -> threads:int -> unit
(** [resize c ~groups ~threads] makes the indirect compute command [c] dispatch
    [groups] threadgroups of [threads] threads. *)

(** {1:probes Probes} *)

val macos : bool
(** [macos] is [true] iff the suite was built for macOS. *)

val fixture : dir:string -> string -> string
(** [fixture ~dir f] is the metallib [f.metallib] of the directory [dir]. *)

val weak : nativeint -> nativeint
(** [weak o] is a weak reference to the Metal object [o]. *)

val alive : nativeint -> bool
(** [alive w] is [true] iff the object of the weak reference [w] lives. *)

val uptime : unit -> int
(** [uptime ()] is the host clock of Metal's times, in nanoseconds. *)
