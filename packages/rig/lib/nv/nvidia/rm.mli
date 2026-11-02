(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA's kernel driver: its device files, the escapes to its resource
    manager (RM), its unified memory driver, and the process's client of both.

    Every failure is an [Error] naming what failed: the RM's status by name, or
    the system call's error. Any domain may call any function. *)

(** {1:files Files and mappings} *)

val open_file : string -> (int, string) result
(** [open_file path] is the descriptor of the device file [path]. *)

val open_spare : string -> (int option, string) result
(** [open_spare path] is {!open_file}[ path], or [Ok None] if the process or the
    system has no file left ([EMFILE], [ENFILE]). Use it for a file opened
    for one call, so that running out of files refuses only that call. *)

val host_page : unit -> int
(** [host_page ()] is the host's page size, in bytes, on Linux. *)

val close : int -> unit
(** [close fd] closes [fd], ignoring a failure. *)

val register : int -> ctl:int -> (unit, string) result
(** [register fd ~ctl] ties the GPU file [fd] to the control file [ctl]; the
    driver refuses a GPU to a process without it. *)

val map : int -> int -> int -> (unit, string) result
(** [map fd at n] maps [n] bytes of [fd], or anonymous memory if [fd < 0], at
    the address [at] of the process. *)

val unmap : int -> int -> unit
(** [unmap at n] returns the [n] bytes {!map} mapped at [at] to the process's
    reservation of the GPU's addresses. *)

val reserve_any : align:int -> int -> (int option, string) result
(** [reserve_any ~align n] is the first of [n] of the process's addresses that
    it reserves, inaccessible, wherever it has room for them, at a multiple of
    [align], a power of two of at least a page; [None] if it has no room. *)

val unreserve : int -> int -> unit
(** [unreserve at n] gives back the [n] addresses from [at] that
    {!reserve_any} reserved. *)

(** {1:client The client} *)

type t = {
  ctl : int;  (** [/dev/nvidiactl]. *)
  root : int;  (** The client's handle. *)
  layouts : (module Defs.RELEASE);  (** The layouts of the driver's release. *)
  release : int;  (** The release's number, such as [615]. *)
}
(** The type for the process's client of the RM. *)

val client : dev:string -> (t, string) result
(** [client ~dev] is the process's client of the RM, opened at its first call
    from the control file in the directory [dev]. Later calls return it
    whatever their [dev], since a process has one client of the machine's
    kernel driver. It is [Error] if the kernel driver's release is none of
    {!Defs.releases}. *)

type unified = {
  uvm : int;  (** [/dev/nvidia-uvm]. *)
  low : Va.t;  (** The GPU's addresses of memory the host maps too. *)
}
(** The type for the process's unified memory. *)

val unified : t -> dev:string -> (unified, string) result
(** [unified c ~dev] is the process's unified memory, opened at its first call
    from the files in the directory [dev]: the unified memory driver, tied to
    the process's address space, and the GPU's addresses of memory the host
    maps too, reserved in the process from 384 GiB to [2{^40}]. Later calls
    return it, as {!client} does. *)

val handle : unit -> int
(** [handle ()] is a handle for an object the process names itself, one no
    object of the client holds: one {!free_handle} gave back, else a new one. *)

val free_handle : int -> unit
(** [free_handle h] gives back [h], a handle {!handle} gave, once the RM freed
    its object or never made one, so that a later {!handle} takes it again. *)

(** {1:rm The RM} *)

val bits : int * int -> int -> int
(** [bits f x] is [x] in the bit field [f], (lowest bit, bits), cut to its
    bits. *)

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

val control :
  t ->
  ?refs:((int * int) * Rig_nv.params) list ->
  int ->
  int ->
  Rig_nv.params ->
  (unit, string) result
(** [control c ~refs obj cmd p] runs the command [cmd] on [obj] with [p],
    checked, once it has written at each field of [p] that [refs] names the
    address of that ref's bytes, which it keeps through the call. *)

val control_status :
  t -> int -> int -> Rig_nv.params -> (int, string) result
(** [control_status c obj cmd p] is {!control}'s, unchecked: the RM's status
    of the command, or the escape's failure. *)

val free : t -> parent:int -> int -> (unit, string) result
(** [free c ~parent obj] frees [obj], a child of [parent], and every object
    under it, or is the RM's refusal ([NV_ESC_RM_FREE]). *)

val rm : t -> Rig_nv.rm
(** [rm c] is the RM as the driver calls it, whose [free] is {!free} and whose
    commands are the RM's controls of their names. *)

val control_request : int
(** [control_request] is the ioctl of a control on the control file
    ([NV_ESC_RM_CONTROL]). *)

(** {1:uvm Unified memory} *)

val uvm :
  unified -> int -> Rig_nv.params -> int * int -> string -> (int, string) result
(** [uvm u cmd p status what] runs the unified memory ioctl [cmd] with [p]: the
    RM status it leaves at the field [status]. *)

val uvm_call :
  t ->
  unified ->
  int ->
  Rig_nv.params ->
  int * int ->
  string ->
  (unit, string) result
(** [uvm_call c u] is {!uvm}[ u] that checks the status by [c]'s release. *)

(** {1:raw Calls that record first}

    These raise [Unix.Unix_error] where the system call fails, and allocate
    nothing once it has succeeded, so that a caller stores what the call
    made in a record made before it, with no allocation between. *)

val open_raw : string -> int
(** [open_raw path] is the descriptor of the device file [path]. *)

val map_raw : int -> int -> int -> unit
(** [map_raw fd at n] is {!map}[ fd at n]. *)

val reserve_any_raw : int -> int -> int
(** [reserve_any_raw n align] is {!reserve_any}[ ~align n], [0] if the process
    has no room. *)

val escape_raw : int -> int -> Rig_nv.params -> unit
(** [escape_raw fd nr p] is {!escape}[ fd nr p], which [p]'s RM status
    answers after. *)

val uvm_raw : unified -> int -> Rig_nv.params -> unit
(** [uvm_raw u cmd p] runs the unified memory ioctl [cmd] with [p], which holds
    its RM status after. *)

val address : Rig_nv.params -> int
(** [address p] is the address of [p]'s bytes, for a field that points to
    other parameters. *)
