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
(** [backends] is {!cpus}, then nx.cuda on CUDA's GPU 0 where the machine has
    one, opened under the GPU lock ({!Rig_gpu_lock.hold}). *)

val serial_reduce :
  Nx_kernel.Spec.reduce Nx_kernel.Spec.t ->
  dsts:Nx_array.any array ->
  Nx_array.any array ->
  Nx_array.answer
(** [serial_reduce] is {!Nx_cpu.reduce} on one thread. *)

val serial_scan :
  Nx_kernel.Spec.scan Nx_kernel.Spec.t ->
  dsts:Nx_array.any array ->
  Nx_array.any array ->
  Nx_array.answer
(** [serial_scan] is {!Nx_cpu.scan} on one thread. *)

val nested : int -> int -> int
(** [nested n cost] runs, through nx.cpu's jobs, a job of [n] units of
    [cost] bytes whose every unit begins a job of [n] units of [cost] bytes
    adding [0] to [n - 1]: the total of the sums. *)
