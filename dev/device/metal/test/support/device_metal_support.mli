(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Metal suite and bench share. *)

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

val defer : ring -> int
(** [defer r] defers a release to the last slot taken and is its number, from
    [0]. *)

val ran : ring -> int array
(** [ran r] is the releases that ran, in the order they ran. *)

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

(** {1:memory Host memory} *)

val get8 : nativeint -> int -> int
(** [get8 p i] is the byte [i] at [p]. *)

val set8 : nativeint -> int -> int -> unit
(** [set8 p i x] stores [x] as the byte [i] at [p]. *)

val get32 : nativeint -> int -> int
(** [get32 p i] is the unsigned 32-bit word [i] at [p]. *)

val set32 : nativeint -> int -> int -> unit
(** [set32 p i x] stores [x] as the unsigned 32-bit word [i] at [p]. *)

val get64 : nativeint -> int -> int64
(** [get64 p i] is the 64-bit word [i] at [p]. *)

val set64 : nativeint -> int -> int64 -> unit
(** [set64 p i x] stores [x] as the 64-bit word [i] at [p]. *)

val pages : int -> nativeint
(** [pages n] is the address of [n] zeroed bytes of host memory, starting at a
    page, never freed. *)

(** {1:fills Fills} *)

type fill
(** The type for fills and their arguments, which live as long as the value. *)

val part : Device_metal.t -> fill -> Device_metal.part
(** [part d f] is [f] as a part of [d]'s queue. The caller keeps [f] until the
    part's submission returned. *)

val failing : int -> fill
(** [failing code] returns [code] and encodes nothing. *)

val dispatch :
  pipeline:int ->
  ?offset:int ->
  Device_metal.region ->
  groups:int ->
  threads:int ->
  fill
(** [dispatch ~pipeline ~offset args ~groups ~threads] dispatches [pipeline]
    over [groups] threadgroups of [threads] threads, with [args] at [offset]
    (defaults to [0]) as its kernel buffer [0]. *)

val split : fill -> Device_metal.t -> int -> times:nativeint -> unit
(** [split f d k ~times] makes the dispatch [f] split [k] times through [d]'s
    [split], dispatching again after each, and write the times of split [i] at
    the 64-bit words [2i] and [2i + 1] at [times], unless [times] is [0n]. *)

val execute : Device_metal_abi.icb -> pipelines:int array -> fill
(** [execute b ~pipelines] runs every command of [b] after setting each of
    [pipelines] on the encoder. *)

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

val wait : Device_metal.t -> int -> unit
(** [wait d v] returns once [d]'s word, read as host memory, reaches [v]. *)
