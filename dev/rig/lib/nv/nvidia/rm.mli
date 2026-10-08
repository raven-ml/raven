(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's kernel driver: its device files, the escapes to its resource
    manager (RM), its unified memory driver, and the process's client of both
    (private).

    Every failure is an [Error] naming what failed: the RM's status by name, or
    the system call's error. Any domain may call any function. *)

(** {1:files Files and mappings} *)

val open_file : string -> (int, string) result
(** [open_file path] is the descriptor of the device file [path]. *)

val close : int -> unit
(** [close fd] closes [fd]. *)

val register : int -> ctl:int -> (unit, string) result
(** [register fd ~ctl] ties the GPU file [fd] to the control file [ctl]; the
    driver refuses a GPU to a process without it. *)

val map : int -> int -> int -> (unit, string) result
(** [map fd at n] maps [n] bytes of [fd], or anonymous memory if [fd < 0], at
    the address [at] of the process. *)

val unmap : int -> int -> unit
(** [unmap at n] returns the [n] bytes {!map} mapped at [at] to the process's
    reservation of the GPU's addresses. *)

(** {1:client The client} *)

type t = {
  ctl : int;  (** [/dev/nvidiactl]. *)
  uvm : int;  (** [/dev/nvidia-uvm]. *)
  root : int;  (** The client's handle. *)
  layouts : (module Defs.RELEASE);  (** The layouts of the driver's release. *)
  release : int;  (** The release's number, such as [615]. *)
  low : Va.t;  (** The GPU's addresses of memory the host maps too. *)
  main : Va.t;  (** The GPU's addresses of the rest. *)
}
(** The type for the process's client. *)

val client : unit -> (t, string) result
(** [client ()] is the process's client, opened at its first call: the RM's
    client, the unified memory driver, and the GPU's addresses, reserved in the
    process below [2{^40}]. It is [Error] if the kernel driver's release is none
    of {!Defs.releases}. *)

val handle : unit -> int
(** [handle ()] is a new handle for an object the process names itself. *)

(** {1:rm The RM} *)

val escape : int -> int -> Rig_nv.params -> string -> (unit, string) result
(** [escape fd nr p what] runs the escape [nr] on [fd] with [p]. *)

val check : t -> string -> int -> (unit, string) result
(** [check c what s] is [Ok ()] if the status [s] of [what] is [NV_OK]. *)

val alloc :
  t ->
  parent:int ->
  int ->
  Rig_nv.params option ->
  (int * int, string) result
(** [alloc c ~parent cls p] is the RM's status and the handle of a new object of
    class [cls] under [parent], made with [p]. *)

val rm : t -> Rig_nv.rm
(** [rm c] is the RM as the driver calls it. *)

(** {1:uvm Unified memory} *)

val uvm :
  t -> int -> Rig_nv.params -> int * int -> string -> (int, string) result
(** [uvm c cmd p status what] runs the unified memory ioctl [cmd] with [p]: the
    RM status it leaves at the field [status]. *)

val uvm_call :
  t -> int -> Rig_nv.params -> int * int -> string -> (unit, string) result
(** [uvm_call] is {!uvm} that checks the status. *)

(** {1:params Parameters} *)

val params : int -> Rig_nv.params
(** [params n] is [n] zero bytes. *)

val get : Rig_nv.params -> int * int -> int

val set : Rig_nv.params -> int * int -> int -> unit
(** [get] and [set] read and write a field (byte offset, bytes) of 1, 2, 4 or 8
    bytes. *)

val address : Rig_nv.params -> int
(** [address p] is the address of [p]'s first byte. *)
