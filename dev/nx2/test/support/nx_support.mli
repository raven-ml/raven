(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices, kernel libraries and expansions for nx's suites. *)

val memory : int -> Rig.t
(** [memory k] is the memory device [k], [0 <= k < 4], named ["m<k>"]: its
    memory is the host's and its work runs in this process
    ({!Rig.memory_device}). *)

(** nx.cpu, counting each call of its entries, named ["nx.test"]. *)
module Counting : sig
  include Nx_kernel.S

  val calls : unit -> int
  (** [calls ()] is the number of entry calls since the last {!reset}. *)

  val positions : unit -> int
  (** [positions ()] is the number of loop positions that the [map],
      [reduce] and [scan] calls since the last {!reset} computed. *)

  val products : unit -> int
  (** [products ()] is the number of products that the [contract] calls
      since the last {!reset} computed: each one's outputs times its
      contracted extent. *)

  val reset : unit -> unit
end

module Declining : Nx_kernel.S
(** nx.cpu, declining the one-node maps of [Add] and of a constant, named
    ["nx.test"]. *)

module Blocked : Nx_kernel.S
(** nx.cpu with a contraction association of its own, as
    {!Nx_kernel.S.contract} allows a library, named ["nx.blocks"]. A
    contraction of one contracting pair sums each block of 4096 products in
    chain order from [+0]; lane order combines the blocks' sums by lane
    order's tree of blocks, then adds [init]; chain order adds them in turn
    to [init], or to the first. Each sum is in [acc], rounded once into
    [dst]. A contraction of no products is nx.cpu's.

    Raises [Failure] for a contraction over several contracting pairs. *)

(** {1:nans NaN bits}

    An OCaml float holds a NaN quiet, so suites make a signalling NaN
    through its bits. A complex number's parts are each a float. *)

val signalled : ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t
(** [signalled a] is a C-contiguous copy of [a] with each NaN made
    signalling where it stays a NaN: its quiet bit, the top bit of its
    fraction, cleared where the rest of its fraction is not zero. *)

val quieted : ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t
(** [quieted a] is a C-contiguous copy of [a] with each NaN quieted, as an
    operation returns it: its quiet bit set, its sign and the rest of its
    payload kept. float8_e4m3fn's NaNs are quiet already. *)

(** {1:expansions Expansions} *)

val folding :
  int ->
  ('d Nx.int64_t -> 'd Nx.Prim.any list -> 'd Nx.Prim.any list) ->
  'd Nx.Prim.any list ->
  'd Nx.Prim.any list
(** [folding n step init] is [step (n - 1) (… (step 0 init))], each trip's
    index a 0-d int64 constant: the [repeat] of {!Nx.Prim.expand} that
    computes the trips in turn. *)

val expanded : ('r. 'r Nx.Prim.t -> bool) -> (unit -> 'a) -> 'a
(** [expanded which f] is [f ()] in an interpretation that computes each
    operation [which] selects by its expansion ({!Nx.Prim.expand}), its loops
    {!folding}, and every other operation, and one without an expansion, as
    eager execution does. *)
