(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernel libraries as nx.kernel's suite runs them. *)

(** The type for kinds of no operand. *)
type op0 =
  | Fill of string  (** A [Const] node of the result's dtype and these bits. *)
  | Iota of int  (** A [Coord] node of this axis. *)

(** The type for a kind of any arity. *)
type kind =
  | K0 of op0
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
(** [backends] is {!cpus}. *)

val x86_64 : bool
(** [x86_64] is [true] iff nx.cpu is compiled for x86-64, whose
    crossovers, as linear algebra's two-stage reductions', differ from
    arm64's. *)

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

(** {1:single One-node maps}

    A kind of the elementwise kernel, {!Nx_kernel.S.map} of the program
    {!Nx_kernel.Prog.of_node} makes of its one node, each operand read
    plain, into one destination. Each raises [Invalid_argument] where the
    kind does not take the operands' dtypes and the result's, as
    {!Nx_kernel.Prog.v} does. *)

val map0 :
  (module Nx_kernel.S) -> op0 -> dst:('v, 's) Nx_array.t -> Nx_array.answer
(** [map0 k op ~dst] stores [op] at each index of [dst]. *)

val map1 :
  (module Nx_kernel.S) ->
  Nx_kernel.Prog.op1 ->
  dst:('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer
(** [map1 k op ~dst x] stores [op] of each element of [x] into [dst]. *)

val map2 :
  (module Nx_kernel.S) ->
  Nx_kernel.Prog.op2 ->
  dst:('v, 's) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer
(** [map2 k op ~dst x y] stores [op] of each element of [x] and [y] into
    [dst]. *)

val map3 :
  (module Nx_kernel.S) ->
  Nx_kernel.Prog.op3 ->
  dst:('v, 's) Nx_array.t ->
  ('c, 'e) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  ('a, 'b) Nx_array.t ->
  Nx_array.answer
(** [map3 k op ~dst c x y] stores [op] of each element of [c], [x] and [y]
    into [dst]. *)

(** {1:memory Address space} *)

val cap_memory : int -> bool
(** [cap_memory n] limits the calling process's address space to [n] bytes
    above its size, so that an allocation past them fails. It is [true] iff
    the limit is set, which only Linux does. The limit stays for the
    process's life: only a forked child calls it. *)
