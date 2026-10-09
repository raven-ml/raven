(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** N-dimensional arrays as values on device sets.

    A value of type [('v, 's, 'd) t] is elements of one dtype, read as ['v] and
    stored as ['s], with a shape and a placement over the device set ['d].
    Nothing changes a value.

    A device set is a module: [module Gpu = (val Nx.devices [ d0; d1 ])] mints a
    brand [Gpu.d], and values of two sets never meet in typed code. *)

(** {1:values Values}

    A value is a shape, a dtype and the devices its elements lie on. *)

type (!'v, !'s, +!'d) t
(** The type for arrays of ['s]-format elements read as ['v], on the device set
    ['d]. *)

type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t
(** The type for dtypes whose elements OCaml reads as ['v] and which store them
    as ['s]. *)

val shape : ('v, 's, 'd) t -> int array
(** [shape x] is [x]'s extents, a fresh array. *)

val dtype : ('v, 's, 'd) t -> ('v, 's) dtype
(** [dtype x] is [x]'s element format. *)

(** {1:dtypes Dtypes}

    A dtype says how a value stores its elements and how OCaml reads them:
    [float32] stores 32-bit floats and reads them as [float], [int8] stores
    8-bit integers and reads them as [int]. The operands of an operation share a
    dtype by their types; {!cast} changes it. There is one value per constructor
    of {!Dtype.t}, named in lowercase. *)

module Dtype = Nx_array.Dtype
(** Dtypes: their constructors, element types and properties. *)

val float64 : (float, Dtype.float64_elt) dtype
val float32 : (float, Dtype.float32_elt) dtype
val float16 : (float, Dtype.float16_elt) dtype
val bfloat16 : (float, Dtype.bfloat16_elt) dtype
val float8_e4m3fn : (float, Dtype.float8_e4m3fn_elt) dtype
val float8_e5m2 : (float, Dtype.float8_e5m2_elt) dtype
val float4_e2m1fn : (float, Dtype.float4_e2m1fn_elt) dtype
val int64 : (int64, Dtype.int64_elt) dtype
val uint64 : (int64, Dtype.uint64_elt) dtype
val int32 : (int32, Dtype.int32_elt) dtype
val uint32 : (int32, Dtype.uint32_elt) dtype
val int16 : (int, Dtype.int16_signed_elt) dtype
val uint16 : (int, Dtype.int16_unsigned_elt) dtype
val int8 : (int, Dtype.int8_signed_elt) dtype
val uint8 : (int, Dtype.int8_unsigned_elt) dtype
val int4 : (int, Dtype.int4_elt) dtype
val uint4 : (int, Dtype.uint4_elt) dtype
val complex128 : (Complex.t, Dtype.complex64_elt) dtype
val complex64 : (Complex.t, Dtype.complex32_elt) dtype
val bool : (bool, Dtype.bool_elt) dtype
val bit : (bool, Dtype.bit_elt) dtype

(** {1:creation Creation and constants}

    A constant, a value made from a dtype, a shape and numbers, belongs to every
    device set: its type is polymorphic in ['d], and it is computed on a set the
    first time an operation there needs it, once per placement. {!zeros_like}
    and {!copy} make a value from another, where it lies. *)

val zeros : ('v, 's) dtype -> int array -> ('v, 's, 'd) t
(** [zeros dt s] is the constant of shape [s] whose every element is zero.

    Raises [Invalid_argument] if an extent is negative. *)

val scalar : ('v, 's) dtype -> 'v -> ('v, 's, 'd) t
(** [scalar dt v] is the 0-d constant [v].

    Raises [Invalid_argument] if [v] is an [int] outside [dt]'s range. *)

val zeros_like : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [zeros_like x] is zeros of [x]'s dtype and shape, where [x] lies. *)

val copy : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [copy x] is [x] stored afresh: equal elements in new memory. *)

(** {1:arith Arithmetic}

    Operations compute where their operands lie and give a new value. Binary
    operations broadcast: aligned at their last axes, two extents are equal or
    one of them is [1], which stretches. *)

val add : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [add a b] is the elementwise sum, integers wrapping.

    Raises [Invalid_argument] if the shapes do not broadcast, or [a]'s dtype is
    a boolean. *)

val mul : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [mul a b] is the elementwise product, as {!add}. *)

val less : ('v, 's, 'd) t -> ('v, 's, 'd) t -> (bool, Dtype.bool_elt, 'd) t
(** [less a b] is [true] where [a]'s element is below [b]'s: integers by value,
    unsigned dtypes unsigned; floats by value, [-0.] not below [+0.];
    [false < true]; complex numbers by real part, then imaginary part. It is
    [false] where either is a NaN, or a complex number with a NaN part.

    Raises [Invalid_argument] if the shapes do not broadcast. *)

val where :
  (bool, Dtype.bool_elt, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [where c x y] is [x]'s element where [c] is [true] and [y]'s elsewhere.

    Raises [Invalid_argument] if the shapes do not broadcast. *)

val cast : ('w, 'r) dtype -> ('v, 's, 'd) t -> ('w, 'r, 'd) t
(** [cast dt x] is [x]'s elements stored in [dt]; [x] itself where [dt] is [x]'s
    dtype.

    A float rounds once to the nearest value of a float format, ties to even.
    Past the format's largest finite value it is an infinity in [float64],
    [float32], [float16] and [bfloat16], and saturates in the formats of a byte
    or less. NaN stays NaN, but in [float4_e2m1fn], where it is [+0.]. Into an
    integer, a float truncates toward zero and saturates to the range, NaN
    giving [0]; an integer is kept modulo the width; a boolean is [0] or [1].
    Into a boolean, any element is [true] where it is not zero, NaN included. An
    integer into a float rounds once from its exact value. Into a complex dtype
    these rules give the real part, and the imaginary part is [0.]; a complex
    number stores part by part into a complex dtype, as [true] into a boolean if
    a part is not zero, and by its real part into any other dtype. *)

val reshape : int array -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [reshape s x] is [x]'s elements in C order of indices, of shape [s].

    Raises [Invalid_argument] unless [s]'s extents are not negative and multiply
    to [x]'s number of elements. *)

(** {1:devices Device sets and placement}

    A value lies on a device set, a module minted by {!devices} whose brand ['d]
    keeps its values apart from other sets'. Within its set a value has a
    placement: whole on one or every device, or cut into windows across them.
    {!place} moves a value between sets, and between placements of one. *)

type host
(** The brand of {!host}. *)

type +'d devices
(** The type for device sets of brand ['d]: distinct devices, and the kernels
    that compute on them. *)

val host : host devices
(** [host] is the process's host, {!Rig.host}, computed by nx.cpu. *)

val rigs : 'd devices -> Rig.t array
(** [rigs s] is [s]'s devices in order, a fresh array. *)

(** Named axes over a set's devices. *)
module Mesh : sig
  type +'d t
  (** The type for named axes over a set of brand ['d]. *)

  val v : 'd devices -> (string * int) list -> 'd t
  (** [v s axes] lays [s]'s devices in row-major order over the named [axes].

      Raises [Invalid_argument] unless the extents are positive and multiply to
      the number of [s]'s devices and the names are distinct. *)
end

(** Where a value's elements lie over its set. *)
module Placement : sig
  type +'d t
  (** The type for placements over a set of brand ['d]. *)

  val host : host t
  (** [host] is the host's one device. *)

  val on : 'd devices -> 'd t
  (** [on s] holds the whole value on every device of [s]. *)

  val split : axis:int -> 'd devices -> 'd t
  (** [split ~axis s] cuts [axis] into equal windows, one per device of [s], in
      order.

      Raises [Invalid_argument] if [axis] is negative or not below
      {!Nx_array.Layout.max_rank}. *)

  val mesh : 'd Mesh.t -> (int * string list) list -> 'd t
  (** [mesh m cuts] cuts each axis [a] of [(a, names)] over the mesh axes
      [names], major first; a mesh axis no cut names holds copies.
      [split ~axis s] is [mesh (Mesh.v s [ ("x", n) ]) [ (axis, [ "x" ]) ]].

      Raises [Invalid_argument] for a name [m] lacks, an axis or a name in two
      cuts, or an axis that is negative or not below
      {!Nx_array.Layout.max_rank}. *)

  val devices : 'd t -> 'd devices
  (** [devices p] is [p]'s set. *)

  val equal : 'd t -> 'd t -> bool
  (** [equal p q] is [true] iff every device holds the same window of a value at
      [p] and at [q]. *)

  val pp : Format.formatter -> 'd t -> unit
  (** [pp] formats a placement on one device as the device's name, others as
      [on [CUDA:0; CUDA:1]], [split ~axis:0 [CUDA:0; CUDA:1]] or a mesh's
      extents and cuts. *)
end

(** The type for device sets as modules: a brand and its placements. *)
module type Devices = sig
  type d
  (** The set's brand. *)

  val v : d devices
  (** [v] is the set. *)

  val on : d Placement.t
  (** [on] is [Placement.on v]. *)

  val split : axis:int -> d Placement.t
  (** [split ~axis] is [Placement.split ~axis v]. *)
end

val devices : ?kernels:(module Nx_kernel.S) -> Rig.t list -> (module Devices)
(** [devices ~kernels ds] mints a new set over [ds], in their order, with a
    brand no other set has, computed eagerly by [kernels]. Without [kernels], a
    set whose every device runs on the host ({!Rig.runs_on_host}) is computed by
    nx.cpu, and any other set by none.

    Raises [Invalid_argument] if [ds] is empty or repeats a device, or if
    [kernels] does not compute on one of [ds]. *)

val place : 'e Placement.t -> ('v, 's, 'd) t -> ('v, 's, 'e) t
(** [place p x] is [x]'s elements at [p]: over [x]'s own memory where [x]
    already lies at [p] or where [p]'s device maps [x]'s memory, a copy
    otherwise. Across sets it changes the brand; within one, the arrangement.

    Raises [Invalid_argument] if [p]'s cuts do not divide [x]'s shape, and
    {!Rig.Lost} for a lost device. *)

val placement : ('v, 's, 'd) t -> 'd Placement.t
(** [placement x] is where [x] lies. *)

(** {1:errors Errors}

    Every misuse raises [Invalid_argument] whose message is the function the
    user called, a colon, the reason and, where they matter, the operands as
    [dtype shape on set], as in
    [Nx.place: axis 0 of extent 6 does not split evenly over 4]. A lost device
    raises {!Rig.Lost}, and a device's memory that runs out
    {!Rig.Out_of_memory}. Host memory that runs out raises OCaml's
    [Out_of_memory]. *)

(** {1:repr Arrays}

    The crossing to the array layer, for libraries that read or make bytes:
    formats, C bindings, compilers. Layouts and buffers are visible here and
    nowhere else in this module. *)

module Repr : sig
  val of_array : 'd devices -> ('v, 's) Nx_array.t -> ('v, 's, 'd) t
  (** [of_array s a] is the value whose bytes are [a]'s.

      Raises [Invalid_argument] naming [Nx.Repr.of_array] unless [a] lies on a
      device of [s]. *)

  val array : ('v, 's, 'd) t -> ('v, 's) Nx_array.t option
  (** [array x] is [Some a] iff [x] lies on one device, [a] its array. [a] may
      be strided or offset. *)

  val of_shards : 'd Placement.t -> ('v, 's) Nx_array.t array -> ('v, 's, 'd) t
  (** [of_shards p arrays] is the value whose bytes are [arrays], one per device
      of [p] in order, each that device's window.

      Raises [Invalid_argument] naming [Nx.Repr.of_shards] unless there is one
      array per device of [p], each on its device, all of one shape, of a rank
      that has every axis [p] cuts. *)

  val shards : ('v, 's, 'd) t -> ('v, 's) Nx_array.t array option
  (** [shards x] is [Some arrays], one per device of [x]'s placement, in order:
      [[| a |]] for a value on one device. *)
end
