(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices, kernel libraries and the witnesses nx's suites share. *)

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
    contraction of one contracting pair sums each block of 4096 products as
    nx.cpu sums it from [+0], and adds the blocks' sums in turn to [init],
    or to the first. Each sum is in [acc], rounded once into [dst]. A
    contraction of no products is nx.cpu's.

    Raises [Failure] for a contraction over several contracting pairs. *)

(** {1:nans NaN bits}

    An OCaml float holds a NaN quiet, so suites make a signalling NaN
    through its bits. A complex number's parts are each a float. *)

val signalled : ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t
(** [signalled a] is a C-contiguous copy of [a] with each NaN made
    signalling where it stays a NaN: its quiet bit, the top bit of its
    fraction, cleared where the rest of its fraction is not zero. *)

val quieted : ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t
(** [quieted a] is a C-contiguous copy of [a] with each NaN quieted, as
    {!Nx_array.Dtype.of_float} stores it: its quiet bit set, its sign and the
    rest of its payload kept. float8_e4m3fn's NaNs are quiet already. *)

val signalling :
  ('v, 's) Nx_array.Dtype.t -> int array -> 'v array -> ('v, 's, Nx.host) Nx.t
(** [signalling dt s data] is the host value of shape [s] holding [data] in
    C order, {!signalled}. *)

val signals : ('v, 's) Nx_array.Dtype.t -> 'v array -> bool
(** [signals dt data] is [true] iff [data] holds a NaN that {!signalled}
    makes signalling. *)

(** {1:allocation Allocation} *)

val words : (unit -> 'a) -> float
(** [words f] is the minor words one call of [f] allocates, the mean over a
    thousand calls after a first uncounted one. *)

(** {1:backtraces Backtraces} *)

val raised_in : (unit -> 'a) -> string
(** [raised_in f] is the base name of the file whose code raised what [f ()]
    raises, the first frame of its backtrace, with backtraces recorded while [f]
    runs; ["no exception"] if [f] returns. *)

(** {1:witnesses Witnesses and indices} *)

val pp_ints : Format.formatter -> int array -> unit
(** [pp_ints] formats an int array as [[2; 3]]. *)

val index : int array -> int -> int array
(** [index s k] is the index, one position per axis, of the element at
    position [k] in C order of a value of shape [s], for [0 <= k] below the
    product of [s]. *)

val placement : unit -> 'd Nx.Placement.t Windtrap.testable
(** [placement ()] is placements, equal by {!Nx.Placement.equal} and
    formatted by {!Nx.Placement.pp}. *)

val elements : ('v, 's) Nx_array.Dtype.t -> 'v list Windtrap.testable
(** [elements dt] is lists of [dt]'s elements, equal where they have one
    length and each two elements hold the same bits, any two NaNs equal
    whatever their bits; a complex number's parts compare each as a float. *)

val equal_elements : ('v, 's) Nx_array.Dtype.t -> 'v array -> 'v array -> unit
(** [equal_elements dt expected actual] asserts [equal (elements dt)] over
    the elements of [expected] and [actual]. *)

(** {1:remedies Remedies} *)

val remedied :
  by:string ->
  candidates:(string -> 'd Nx.Placement.t list) ->
  (string * 'd Nx.Placement.t) list ->
  ((string -> 'd Nx.Placement.t) -> 'a) ->
  'a
(** [remedied ~by ~candidates args run] is [run at] for the placements [at]
    of the arguments, starting at [args], once it returns. Each refusal of
    [run at] must read
    [by: name lies at p and is read at q; place (e) name] for an
    argument [name] of [args], its placement [p] at [at], and [q] the format
    of one of [candidates name] other than [p]; [run] is then called again
    with [name] there. It fails the test at a refusal that is not so, or
    that repeats an earlier one. *)

(** {1:bits Bits to place} *)

type bits = Bits : ('v, 's) Nx_array.t -> bits
(** The type for host arrays of any dtype. *)

val bits : bits Windtrap.Gen.t
(** [bits] is host arrays of shape [[r; c]] of a dtype of a byte or more,
    their bytes drawn, NaN payloads and [-0.] among them, laid out plainly,
    transposed, flipped on both axes, at stride 2 along columns, broadcast
    along rows or as windows over one axis, whose rows overlap. [r] and [c]
    are mostly drawn from [0], [1], [2], [3] and [5]; one array in four is
    [128] by [130]. *)

val bytes : ('v, 's) Nx_array.t -> int array
(** [bytes a] is the bytes of [a]'s elements in C order of indices, each
    element's in memory order, read on the host. [a] must lie in memory the
    host addresses. *)
