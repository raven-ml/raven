(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Metal suite and bench share. *)

(** {1:gpu The GPU} *)

include Rig_gpu_support.Conformance with module D = Rig_metal
(** Metal's device [0]; its second device is device [0] opened again. *)

(** {1:fills Fills} *)

type fill
(** The type for fills and their arguments, which live as long as the value. *)

val part : fill -> Rig_gpu_support.work
(** [part f] is [f] as work on a Metal device’s queue, with its parameters
    copied into the run. *)

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

val room :
  Rig_metal.t -> Rig_edge.entry -> params:int -> [ `Fits | `Later | `Never | `Grow ]
(** [room g e ~params] is what [g]'s room answers for one launch of [e], over
    one thread, with [params] bytes of parameters. *)

val virtual_mac : bool
(** [virtual_mac] is [true] iff this Mac is a virtual machine
    ([kern.hv_vmm_present]), whose paravirtual GPU makes no residency set: the
    GPU suites count no GPU there. *)

val fixture : dir:string -> string -> string
(** [fixture ~dir f] is the metallib [f.metallib] of the directory [dir]. *)

val weak : nativeint -> nativeint
(** [weak o] is a weak reference to the Metal object [o]. *)

val alive : nativeint -> bool
(** [alive w] is [true] iff the object of the weak reference [w] lives. *)

val uptime : unit -> int
(** [uptime ()] is the host clock of Metal's times, in nanoseconds. *)
