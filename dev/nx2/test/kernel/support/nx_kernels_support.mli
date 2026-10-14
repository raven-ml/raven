(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernel libraries as nx.kernel's suite runs them. *)

(** The type for a kind of any arity. *)
type kind =
  | K0 of Nx_kernel.Prog.op0
  | K1 of Nx_kernel.Prog.op1
  | K2 of Nx_kernel.Prog.op2
  | K3 of Nx_kernel.Prog.op3

type backend = {
  name : string;  (** The backend's name in the suite, as ["cpu/base"]. *)
  kernels : (module Nx_kernel.S);  (** Its kernels. *)
  device : Rig.t;  (** The device whose memory its laws compute on. *)
  computes : kind -> Nx_array.Dtype.any -> bool;
      (** [computes k dt] is [true] iff it states it computes [k] at [dt]: the
          result's dtype for [K0], the operand's for [K1] and [K2], and for [K3]
          the second operand's, with a [bool] condition. The suite fails when it
          declines one. *)
  around : 'a. (unit -> 'a) -> 'a;
      (** [around f] is [f ()] run as the backend: nx.cpu's under its target
          table. *)
}
(** The type for kernel libraries in one configuration. *)

val cpus : backend list
(** [cpus] is nx.cpu under each target table the host runs, base first. *)

val backends : backend list
(** [backends] is {!cpus}, then nx.cuda on the machine's NVIDIA GPU 0 where it
    has one, as ["cuda"] through CUDA and as ["cuda/nv"] through rig.nv. *)

val nested : int -> int -> int
(** [nested n cost] runs, through nx.cpu's jobs, a job of [n] units of
    [cost] bytes whose every unit begins a job of [n] units of [cost] bytes
    adding [0] to [n - 1]: the total of the sums. *)

(** {1:guards Guard pages} *)

(** The type for the side of an array's bytes an inaccessible page lies on. *)
type side =
  | Before  (** The first byte of its span lies just after the page. *)
  | After  (** The last byte of its span lies just before the page. *)

val guarded :
  side ->
  ('v, 's) Nx_array.Dtype.t ->
  Nx_array.Layout.t ->
  ('v, 's) Nx_array.t * (unit -> unit)
(** [guarded side dt l] is an array of [dt] over [l] on host memory of its own,
    zeroed, whose span's bytes ({!Nx_array.Layout.span}) lie against an
    inaccessible page on [side], so that a kernel reading a byte past them on
    that side faults; and the function that frees the memory, after which the
    array is not used. [l]'s positions start at the memory's first byte. *)

val guard : side -> ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t * (unit -> unit)
(** [guard side x] is {!guarded} [side] over [x]'s layout, holding the bytes of
    [x]'s span: [x]'s elements, bit for bit, against the page, strided,
    reversed or broadcast as [x] is. With the function that frees it. [x] is
    on host memory. *)
