(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** N-dimensional arrays.

    [Nx] provides n-dimensional arrays (tensors) with NumPy-like semantics. A
    tensor [('a, 'b) t] holds elements of OCaml type ['a] stored in a buffer
    with element kind ['b].

    {b Tensors, views, and contiguity.} A tensor is a {e view} over a flat
    buffer described by a shape, strides, and an offset. Operations that only
    rearrange metadata ({!reshape}, {!transpose}, {!val-slice}, …) return views
    in O(1) without copying data. Use {!is_c_contiguous} to test whether
    elements are laid out contiguously in row-major order, and {!contiguous} to
    obtain a contiguous copy when needed.

    {b Broadcasting.} Binary operations automatically broadcast operands whose
    shapes differ: dimensions are aligned from the right and each pair must be
    equal or one of them must be 1.

    {b Immutable tensors.} All operations return freshly allocated tensors. *)

(** {1:types Types} *)

type ('a, 'b) t = ('a, 'b) Nx_effect.t
(** The type for tensors with OCaml element type ['a] and buffer element kind
    ['b]. *)

(** {2:elt_kinds Element kinds}

    Witnesses for the buffer element representation. Used as the second type
    parameter of {!type-t}. *)

type float16_elt = Nx_dtype.float16_elt
type float32_elt = Nx_dtype.float32_elt
type float64_elt = Nx_dtype.float64_elt
type bfloat16_elt = Nx_dtype.bfloat16_elt
type float8_e4m3_elt = Nx_dtype.float8_e4m3_elt
type float8_e5m2_elt = Nx_dtype.float8_e5m2_elt
type int4_elt = Nx_dtype.int4_elt
type uint4_elt = Nx_dtype.uint4_elt
type int8_elt = Nx_dtype.int8_elt
type uint8_elt = Nx_dtype.uint8_elt
type int16_elt = Nx_dtype.int16_elt
type uint16_elt = Nx_dtype.uint16_elt
type int32_elt = Nx_dtype.int32_elt
type uint32_elt = Nx_dtype.uint32_elt
type int64_elt = Nx_dtype.int64_elt
type uint64_elt = Nx_dtype.uint64_elt
type complex32_elt = Nx_dtype.complex32_elt
type complex64_elt = Nx_dtype.complex64_elt
type bool_elt = Nx_dtype.bool_elt

(** {2:dtype Data types} *)

type ('a, 'b) dtype = ('a, 'b) Nx_dtype.t =
  | Float16 : (float, float16_elt) dtype
  | Float32 : (float, float32_elt) dtype
  | Float64 : (float, float64_elt) dtype
  | BFloat16 : (float, bfloat16_elt) dtype
  | Float8_e4m3 : (float, float8_e4m3_elt) dtype
  | Float8_e5m2 : (float, float8_e5m2_elt) dtype
  | Int4 : (int, int4_elt) dtype
  | UInt4 : (int, uint4_elt) dtype
  | Int8 : (int, int8_elt) dtype
  | UInt8 : (int, uint8_elt) dtype
  | Int16 : (int, int16_elt) dtype
  | UInt16 : (int, uint16_elt) dtype
  | Int32 : (int32, int32_elt) dtype
  | UInt32 : (int32, uint32_elt) dtype
  | Int64 : (int64, int64_elt) dtype
  | UInt64 : (int64, uint64_elt) dtype
  | Complex64 : (Complex.t, complex32_elt) dtype
  | Complex128 : (Complex.t, complex64_elt) dtype
  | Bool : (bool, bool_elt) dtype
      (** The type for data type descriptors. A [('a, 'b) dtype] links the OCaml
          element type ['a] to its buffer representation ['b].

          [int4] and [uint4] are storage formats: their values move, cast and
          are read and written, and every computation on them raises
          [Invalid_argument]; cast them to a wider integer to compute. [bool]
          values compare, combine logically and bitwise, select ({!where}),
          sort and reduce by {!max} and {!min}; arithmetic on them (sums,
          products, negation, cumulative sums, {!matmul}) raises
          [Invalid_argument]. *)

(** {2:tensor_aliases Tensor aliases} *)

type float16_t = (float, float16_elt) t
type float32_t = (float, float32_elt) t
type float64_t = (float, float64_elt) t
type bfloat16_t = (float, bfloat16_elt) t
type float8_e4m3_t = (float, float8_e4m3_elt) t
type float8_e5m2_t = (float, float8_e5m2_elt) t
type int4_t = (int, int4_elt) t
type uint4_t = (int, uint4_elt) t
type int8_t = (int, int8_elt) t
type uint8_t = (int, uint8_elt) t
type int16_t = (int, int16_elt) t
type uint16_t = (int, uint16_elt) t
type int32_t = (int32, int32_elt) t
type uint32_t = (int32, uint32_elt) t
type int64_t = (int64, int64_elt) t
type uint64_t = (int64, uint64_elt) t
type complex64_t = (Complex.t, complex32_elt) t
type complex128_t = (Complex.t, complex64_elt) t
type bool_t = (bool, bool_elt) t

(** {2:dtype_vals Data type values} *)

val float16 : (float, float16_elt) dtype
val float32 : (float, float32_elt) dtype
val float64 : (float, float64_elt) dtype
val bfloat16 : (float, bfloat16_elt) dtype
val float8_e4m3 : (float, float8_e4m3_elt) dtype
val float8_e5m2 : (float, float8_e5m2_elt) dtype
val int4 : (int, int4_elt) dtype
val uint4 : (int, uint4_elt) dtype
val int8 : (int, int8_elt) dtype
val uint8 : (int, uint8_elt) dtype
val int16 : (int, int16_elt) dtype
val uint16 : (int, uint16_elt) dtype
val int32 : (int32, int32_elt) dtype
val uint32 : (int32, uint32_elt) dtype
val int64 : (int64, int64_elt) dtype
val uint64 : (int64, uint64_elt) dtype
val complex64 : (Complex.t, complex32_elt) dtype
val complex128 : (Complex.t, complex64_elt) dtype
val bool : (bool, bool_elt) dtype

(** {2:index Index specifications} *)

(** The type for index specifications used by {!val-slice} and {!set}. *)
type index =
  | I of int  (** [I i] selects a single index, reducing the dimension. *)
  | L of int list  (** [L [i0; i1; …]] gathers the listed indices. *)
  | R of int * int
      (** [R (start, stop)] selects the half-open range \[[start], [stop]). *)
  | Rs of int * int * int
      (** [Rs (start, stop, step)] selects a strided range. *)
  | A
      (** [A] selects the entire axis. This is the default for axes not covered
          by a {!val-slice} specification. *)
  | M of (bool, bool_elt) t
      (** [M mask] selects, along the axis it addresses, the positions where the
          rank-1 boolean tensor [mask] is [true]. [mask] must have length equal
          to that axis. Equivalent to an [L] gather of the true positions. *)
  | N  (** [N] inserts a new axis of size 1 (does not consume an input axis). *)
  | D of int64_t * int
      (** [D (start, len)] selects the run of [len] positions beginning at the
          run-time value of the scalar tensor [start], clamped into \[[0],
          [size - len]\] so the run always fits. Keeps the axis, like [R]. [len]
          is static because traced shapes are: one compiled program serves every
          position. *)

(** {2:packed Packed tensors} *)

(** The type for tensors of any dtype, such as a file's named tensors, a
    checkpoint's entries or the tensors of {!Ptree.flatten}. Match on [P] for
    the tensor; {!unpack} also fixes its dtype. *)
type packed = Nx_effect.packed =
  | P : ('a, 'b) t -> packed  (** A tensor whose dtype is hidden. *)

val unpack : ('a, 'b) dtype -> packed -> ('a, 'b) t
(** [unpack dtype p] is the tensor in [p] if its dtype is [dtype].

    Raises [Invalid_argument] naming both dtypes otherwise, such as
    ["unpack: expected dtype float32, got uint8"]. *)

(** {1:properties Properties} *)

val shape : ('a, 'b) t -> int array
(** [shape t] is the dimensions of [t]. A scalar tensor has shape [|\||]. *)

val dtype : ('a, 'b) t -> ('a, 'b) dtype
(** [dtype t] is the data type of [t]. *)

val dim : int -> ('a, 'b) t -> int
(** [dim i t] is the size of dimension [i].

    Raises [Invalid_argument] if [i] is out of bounds. *)

val ndim : ('a, 'b) t -> int
(** [ndim t] is the number of dimensions of [t]. *)

val itemsize : ('a, 'b) t -> int
(** [itemsize t] is the number of bytes per element. *)

val numel : ('a, 'b) t -> int
(** [numel t] is the total number of elements in [t]. *)

val nbytes : ('a, 'b) t -> int
(** [nbytes t] is [numel t * itemsize t]. *)

val is_c_contiguous : ('a, 'b) t -> bool
(** [is_c_contiguous t] is [true] iff [t]'s elements are laid out contiguously
    in row-major (C) order.

    See also {!contiguous}. *)

val to_bigarray : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_bigarray t] is a fresh C-layout bigarray of [t]'s shape holding [t]'s
    elements. It always copies, so writing it leaves [t] unchanged. Loop over it
    for element access that allocates nothing per element.

    Raises [Invalid_argument] if [t]'s dtype has no {!Bigarray.kind}: bfloat16,
    the float8 dtypes, uint32, uint64, int4, uint4 and bool. {!bitcast} the
    first five to the integers of their width first, which have one, and
    {!cast} the last three to [uint8].

    See also {!of_bigarray}. *)

val to_array : ('a, 'b) t -> 'a array
(** [to_array t] is a fresh OCaml array containing the elements of [t] in
    row-major order. Always copies.

    {@ocaml[
      # let t =
          create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |]
        in
        to_array t
      - : int32 array = [|1l; 2l; 3l; 4l|]
    ]} *)

(** {1:placement Devices, backends and placement}

    Where a value lives is a value too. A device holds memory; the library that
    owns a device's runtime opens it (for example [Rune.device "METAL"]), and
    {!Device.host} is the host. A backend ({!Nx_backend.t}) is kernels that
    compute nx's operations on arrays. A placement is one device, a list of
    devices each holding a full copy, or a list of devices each holding an
    equal slice along one axis, together with the one backend that computes on
    values there: [Nx_cpu.backend] unless the placement names another.

    The host device holds values in two ways. A value at {!Placement.host}, the
    host device with [Nx_cpu.backend], is an array in host memory: operations
    on it call nx.cpu's kernels directly, which keeps the default path as cheap
    as the kernels allow.
    A value at any other placement that includes the host device, with another
    backend or beside other devices, is a placed value like one on a GPU: its
    storage carries what compiled calls need to bind and consume it. Creating
    at {!Placement.host}, operating on host values only, and placing on
    {!Placement.host} give the first; everything else gives the second, and
    moving between them copies.

    The result of an operation lives where its placed operands live, computed
    by their placement's backend: operands on the host join them, and operands
    on two different device sets, or with two different backends, raise. The
    backend computes on the host, over copies of the operands' elements, and
    the result is placed on its devices. Over
    split operands, an elementwise result keeps their split, which must be the
    same for all of them (copies take it); a reduction over the split axis, and
    a {!take} along it, give a full copy on each device; and an operation along
    the split axis ({!sort}, {!cumsum}, {!pad} or {!concatenate} along it,
    linear algebra on its last two axes, {!fft} over it) raises. A read
    ({!item}, {!to_array}, {!to_bigarray}, {!pp}, a save) copies the elements it
    reads and leaves the value where it is. A value's storage is released when
    no value reaches it.

    The disk ([Device.of_runtime Nx_device.disk]) holds values in files, such as
    the tensors [Nx_io.load_safetensors] loads, and computes nothing: a value on
    it takes part in an operation as a host value, which the host reads in the
    file's pages, and a movement of it stays on the disk. {!place} onto the host
    or a device whose memory is the host's borrows the file's pages, and onto
    another device reads the bytes into it; a placement onto the disk raises.

    Reads and placements keep their source storage in use until they return. If
    a compiled call consumes that storage concurrently, the conflicting
    operation raises [Invalid_argument] instead of waiting. This applies to
    every view of the storage. A consumed value cannot be read or placed again;
    its shape and dtype remain available.

    A movement ({!reshape}, {!transpose}, {!slice} by indices and unit-step
    ranges, {!flip}, {!broadcast_to}, {!sliding_window}) of a placed value is a
    view of the same storage: it copies nothing and stays on the same devices. A
    split value moves shard by shard and stays split: its split axis follows a
    transpose, is kept by a slice of the other axes, and survives a reshape that
    keeps the product of the extents before it, when its new extent divides over
    the devices. A movement that would move elements between devices raises
    [Invalid_argument]: any other reshape, a cut of the split axis across
    shards, a flip of it, or windows along it. Functions built from movements
    ({!roll}, {!flatten}, {!diagonal}, {!array_split}) raise the same way. Place
    the value replicated or on one device first. A cut inside one shard (a row
    of a value split by rows, {!item}) is a view of that shard on its device
    alone, so {!item} reads one element. *)

(** Devices. *)
module Device : sig
  type t = Nx_effect.device
  (** The type for devices. *)

  val host : t
  (** [host] is the host, named ["CPU"]. *)

  val name : t -> string
  (** [name d] is [d]'s name, for example ["METAL"] or ["CUDA:3"]. *)

  val equal : t -> t -> bool
  (** [equal d d'] is [true] iff [d] and [d'] are the same device. Libraries
      that open devices return one value per device. *)

  val compare : t -> t -> int
  (** [compare] is a total order on devices, compatible with {!equal}. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a device's name. *)

  val of_runtime : Nx_device.t -> t
  (** [of_runtime d] is the device that holds placed values in [d]'s buffers:
      the same value for every call with [d], and {!host} for [Nx_device.host].
      It has [d]'s name. {!place} and operations raise {!Out_of_memory} with
      this device when [d] cannot allocate. A value placed on another such
      device is copied into [d]'s buffers straight from that device's, a
      window at a time, when the window is a contiguous run of the value's
      storage, and through the host otherwise. *)

  val runtime : t -> Nx_device.t
  (** [runtime d] is the runtime device whose buffers hold [d]'s placed values:
      [runtime (of_runtime rd)] is [rd], and [runtime host] is
      [Nx_device.host].

      Raises [Invalid_argument] if [d] holds its values in memory of its own,
      as a device another library opens does. *)

  exception Out_of_memory of t * int
  (** Raised by an operation, a {!place} or a compiled call when a device cannot
      allocate the given number of bytes. *)
end

(** Placements. *)
module Placement : sig
  type t = Nx_effect.placement
  (** The type for placements: where each device's window of a value lies, and
      the backend that computes on it. Only the functions below build one, and
      a placement is in normal form: a list of one device is that device, and a
      list never repeats a device. *)

  val host : t
  (** [host] is [device Device.host]: the host device with [Nx_cpu.backend],
      whose values are arrays in host memory (see {{!placement}above}). *)

  val device : ?backend:Nx_backend.t -> Device.t -> t
  (** [device ~backend d] is placement on [d] alone, computed by [backend]
      (defaults to [Nx_cpu.backend]).

      Raises [Invalid_argument] if [backend] does not run on the host, where
      nx computes on every placement's values. *)

  val replicated : ?backend:Nx_backend.t -> Device.t list -> t
  (** [replicated ~backend ds] is a full copy on each device of [ds], computed
      by [backend] (defaults to [Nx_cpu.backend]).

      Raises [Invalid_argument] if [ds] is empty, repeats a device, mixes
      devices whose memories differ (those of {!Device.of_runtime}, the host
      included, and those another library opens), or as {!device}. *)

  val sharded : ?backend:Nx_backend.t -> axis:int -> Device.t list -> t
  (** [sharded ~backend ~axis ds] is equal slices of [axis] on the devices of
      [ds], in order, computed by [backend] (defaults to [Nx_cpu.backend]).

      Raises [Invalid_argument] if [axis] is negative, or as {!replicated}. *)

  val devices : t -> Device.t list
  (** [devices p] is the devices of [p], in the order that decides which window
      each holds. *)

  val backend : t -> Nx_backend.t
  (** [backend p] is the backend that computes on values at [p]. *)

  val window : t -> int array -> Device.t -> (int * int) array
  (** [window p shape d] is the window of a value of shape [shape] that [d]
      holds at [p], as [(start, stop)] per axis, [stop] exclusive, as {!shrink}
      takes it: [shrink (window p (shape x) d) x] is [d]'s part of [x].

      Raises [Invalid_argument] if [d] is not one of [devices p], or if [p]
      splits an axis [shape] does not have or does not divide evenly. *)

  val with_leading_axis : t -> t
  (** [with_leading_axis p] is the placement of a value of one more axis in
      front, split as [p] splits the others. It is for transformations that
      map over a leading axis. *)

  val without_leading_axis : t -> t
  (** [without_leading_axis p] is the placement of a value without its first
      axis: a device axis that split it then holds copies, and the other axes
      shift down by one. It is for transformations that map over a leading
      axis. *)

  val equal : t -> t -> bool
  (** [equal p p'] is [true] iff [p] and [p'] have the same backend and every
      device holds the same window of any value at [p] and at [p']. A list of
      full copies is equal to the same devices in another order. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a placement: its devices and layout, followed by
      [with <name>] when its backend is not [Nx_cpu.backend], as in
      ["CPU with counting"]. *)
end

val place : Placement.t -> ('a, 'b) t -> ('a, 'b) t
(** [place p x] is [x] held at [p]. It equals [x] in shape, dtype and elements,
    and [x] is unchanged and stays where it was. It is [x] itself when [x] is
    already at [p]. When [x] is placed and [p] differs from [x]'s placement
    only in its backend, the result is a view of [x]'s storage and copies
    nothing.

    Raises [Invalid_argument] if [p] splits an axis [x] does not have or does
    not divide evenly, or if [p]'s devices cannot hold [x]'s dtype. *)

val placement : ('a, 'b) t -> Placement.t
(** [placement x] is where [x] lives: {!Placement.host} for a value on the host.
*)

(** {1:creation Creation} *)

val create : ('a, 'b) dtype -> int array -> 'a array -> ('a, 'b) t
(** [create dtype shape data] is a tensor of the given [dtype] and [shape]
    initialised from [data] in row-major order.

    Raises [Invalid_argument] if [Array.length data] does not equal the product
    of [shape].

    {@ocaml[
      # create float32 [| 2; 3 |]
          [| 1.; 2.; 3.; 4.; 5.; 6. |]
      - : (float, float32_elt) t = float32 [2,3] [[1, 2, 3],
                                                  [4, 5, 6]]
    ]} *)

val init : ('a, 'b) dtype -> int array -> (int array -> 'a) -> ('a, 'b) t
(** [init dtype shape f] is a tensor where the element at multi-index [i] is
    [f i].

    {@ocaml[
      # init int32 [| 2; 3 |]
          (fun i -> Int32.of_int (i.(0) + i.(1)))
      - : (int32, int32_elt) t = int32 [2,3] [[0, 1, 2],
                                              [1, 2, 3]]
    ]} *)

val full : ('a, 'b) dtype -> int array -> 'a -> ('a, 'b) t
(** [full dtype shape v] is a tensor filled with [v].

    {@ocaml[
      # full float32 [| 2; 3 |] 3.14
      - : (float, float32_elt) t = float32 [2,3]
      [[3.14, 3.14, 3.14],
       [3.14, 3.14, 3.14]]
    ]} *)

val ones : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [ones dtype shape] is a tensor filled with ones. *)

val zeros : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [zeros dtype shape] is a tensor filled with zeros. *)

val scalar : ('a, 'b) dtype -> 'a -> ('a, 'b) t
(** [scalar dtype v] is a 0-dimensional tensor containing [v]. The result has
    shape [|\||]. *)

val full_like : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [full_like t v] is {!full} with the same dtype and shape as [t], placed
    where [t] is: a split [t] gives a value split the same way. Inside a
    compiled function the value is a constant of the program. *)

val ones_like : ('a, 'b) t -> ('a, 'b) t
(** [ones_like t] is {!full_like} with ones. *)

val zeros_like : ('a, 'b) t -> ('a, 'b) t
(** [zeros_like t] is {!full_like} with zeros. *)

val scalar_like : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [scalar_like t v] is {!scalar} with the same dtype as [t]. *)

val eye : ?m:int -> ?k:int -> ('a, 'b) dtype -> int -> ('a, 'b) t
(** [eye ?m ?k dtype n] is an [n × m] matrix with ones on the [k]-th diagonal
    and zeros elsewhere. [m] defaults to [n]. [k] defaults to [0] (main
    diagonal); positive [k] selects an upper diagonal, negative [k] a lower one.

    {@ocaml[
      # eye int32 3
      - : (int32, int32_elt) t = int32 [3,3] [[1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 1]]
      # eye ~k:1 int32 3
      - : (int32, int32_elt) t = int32 [3,3] [[0, 1, 0],
                                              [0, 0, 1],
                                              [0, 0, 0]]
    ]}

    See also {!diag}. *)

val diag : ?k:int -> ('a, 'b) t -> ('a, 'b) t
(** [diag ?k v] extracts or constructs a diagonal.

    When [v] is 1-D, returns a 2-D tensor with [v] on the [k]-th diagonal. When
    [v] is 2-D, returns the [k]-th diagonal as a 1-D tensor. [k] defaults to
    [0].

    Raises [Invalid_argument] if [v] is not 1-D or 2-D.

    {@ocaml[
      # let v = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        diag v
      - : (int32, int32_elt) t = int32 [3,3] [[1, 0, 0],
                                              [0, 2, 0],
                                              [0, 0, 3]]
      # let x =
          arange int32 0 9 1 |> reshape [| 3; 3 |]
        in
        diag x
      - : (int32, int32_elt) t = [0, 4, 8]
    ]}

    See also {!eye}, {!diagonal}. *)

val arange : ('a, 'b) dtype -> int -> int -> int -> ('a, 'b) t
(** [arange dtype start stop step] is the 1-D tensor of the values
    [start + i * step], [i = 0, 1, ...], from [start] (inclusive) toward [stop]
    (exclusive). It is empty when [stop] does not lie beyond [start] in [step]'s
    direction. A float or complex [dtype] holds each value as {!cast} converts
    an [int64] to it, and [bool] holds [0] as [false] and [1] as [true].

    Raises [Invalid_argument] if [step = 0], if a value does not fit [dtype], or
    if the range holds more than [max_int] values. An integer dtype fits the
    values of its range, [bool] fits [0] and [1], and a float or complex dtype
    fits the values whose magnitude is at most its largest finite value, such
    as [65504] for [float16] and [448] for [float8_e4m3].

    {@ocaml[
      # arange int32 0 10 2
      - : (int32, int32_elt) t = int32 [5] [0, 2, ..., 6, 8]
      # arange int32 5 0 (-1)
      - : (int32, int32_elt) t = int32 [5] [5, 4, ..., 2, 1]
    ]}

    See also {!arange_f}, {!linspace}. *)

val arange_f : (float, 'a) dtype -> float -> float -> float -> (float, 'a) t
(** [arange_f dtype start stop step] is like {!arange} for floating-point
    ranges.

    Raises [Invalid_argument] if [step = 0.0].

    {@ocaml[
      # arange_f float32 0. 1. 0.2
      - : (float, float32_elt) t = float32 [5] [0, 0.2, ..., 0.6, 0.8]
    ]}

    See also {!arange}, {!linspace}. *)

val linspace :
  ('a, 'b) dtype -> ?endpoint:bool -> float -> float -> int -> ('a, 'b) t
(** [linspace dtype ?endpoint start stop n] is [n] values evenly spaced from
    [start] to [stop]. [endpoint] defaults to [true] (include [stop]).

    Raises [Invalid_argument] if [n] is negative.

    {@ocaml[
      # linspace float32 0. 10. 5
      - : (float, float32_elt) t = float32 [5] [0, 2.5, ..., 7.5, 10]
      # linspace float32 ~endpoint:false 0. 10. 5
      - : (float, float32_elt) t = float32 [5] [0, 2, ..., 6, 8]
    ]}

    See also {!logspace}, {!geomspace}. *)

val logspace :
  (float, 'a) dtype ->
  ?endpoint:bool ->
  ?base:float ->
  float ->
  float ->
  int ->
  (float, 'a) t
(** [logspace dtype ?endpoint ?base start stop n] is [n] values evenly spaced on
    a logarithmic scale: [base{^x}] where [x] ranges from [start] to [stop].
    [endpoint] defaults to [true]. [base] defaults to [10.0].

    Raises [Invalid_argument] if [n] is negative.

    {@ocaml[
      # logspace float32 0. 2. 3
      - : (float, float32_elt) t = [1, 10, 100]
      # logspace float32 ~base:2.0 0. 3. 4
      - : (float, float32_elt) t = [1, 2, 4, 8]
    ]}

    See also {!linspace}, {!geomspace}. *)

val geomspace :
  (float, 'a) dtype -> ?endpoint:bool -> float -> float -> int -> (float, 'a) t
(** [geomspace dtype ?endpoint start stop n] is [n] values evenly spaced on a
    geometric (multiplicative) scale. [endpoint] defaults to [true].

    Raises [Invalid_argument] if [start] or [stop] is not positive.

    {@ocaml[
      # geomspace float32 1. 1000. 4
      - : (float, float32_elt) t = [1, 10, 100, 1000]
    ]}

    See also {!linspace}, {!logspace}. *)

val meshgrid :
  ?indexing:[ `xy | `ij ] -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
(** [meshgrid ?indexing x y] is a pair of 2-D coordinate grids built from 1-D
    arrays [x] and [y]. [indexing] defaults to [`xy] (Cartesian: X varies along
    columns, Y along rows). With [`ij] (matrix), X varies along rows, Y along
    columns.

    Raises [Invalid_argument] if [x] or [y] is not 1-D.

    {@ocaml[
      # let x = linspace float32 0. 2. 3 in
        let y = linspace float32 0. 1. 2 in
        meshgrid x y
      - : (float, float32_elt) t * (float, float32_elt) t =
      (float32 [2,3] [[0, 1, 2],
                      [0, 1, 2]], float32 [2,3] [[0, 0, 0],
                                                 [1, 1, 1]])
    ]} *)

val tril : ?k:int -> ('a, 'b) t -> ('a, 'b) t
(** [tril ?k x] is the lower-triangular part of [x] with elements above the
    [k]-th diagonal set to zero. [k] defaults to [0].

    Raises [Invalid_argument] if [x] has fewer than 2 dimensions.

    See also {!triu}. *)

val triu : ?k:int -> ('a, 'b) t -> ('a, 'b) t
(** [triu ?k x] is the upper-triangular part of [x] with elements below the
    [k]-th diagonal set to zero. [k] defaults to [0].

    Raises [Invalid_argument] if [x] has fewer than 2 dimensions.

    See also {!tril}. *)

val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> ('a, 'b) t
(** [of_bigarray ba] is a tensor of [ba]'s shape over [ba]'s memory, without a
    copy, of the dtype of [ba]'s kind. The tensor takes ownership: the caller
    must not write [ba] afterwards. Fill a bigarray, then wrap it, to build a
    tensor element by element. A tensor of a dtype with no {!Bigarray.kind} is
    built from another: a bfloat16, float8, uint32 or uint64 one from the
    integers of its width with {!bitcast}, such as a bfloat16 one from
    [int16_unsigned] elements, and an int4, uint4 or bool one from [uint8]
    elements with {!cast}.

    Raises [Invalid_argument] if [ba]'s kind is [Char], [Int] or [Nativeint],
    or if its first element does not lie at a multiple of its size (of one
    component for complex kinds), as a bigarray that [Unix.map_file] maps from
    an unaligned [pos] may not.

    See also {!to_bigarray}, which always copies. *)

val one_hot : num_classes:int -> ('a, 'b) t -> (int, uint8_elt) t
(** [one_hot ~num_classes indices] is a one-hot encoded tensor.

    Appends a new trailing dimension of size [num_classes]. Values in [indices]
    must lie in \[[0], [num_classes]). Out-of-range indices produce all-zero
    rows.

    Raises [Invalid_argument] if [indices] is not an integer dtype or
    [num_classes <= 0].

    {@ocaml[
      # let idx =
          create int32 [| 3 |] [| 0l; 1l; 3l |]
        in
        one_hot ~num_classes:4 idx
      - : (int, uint8_elt) t = uint8 [3,4] [[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 0, 1]]
    ]} *)

module Ptree = Ptree
(** Structures of tensors: types with one [walk] that walks their parts, which
    Rune's transformations, Vega's optimisers and checkpoints take. *)

(** {1:rng Random number generation}

    One generator, reached two ways. {!module-Rng} holds it: keys, one sampler
    per distribution, and the scope. The {{!section:keyless}keyless samplers}
    below are the same samplers drawing from the ambient scope.

    {v   Rng.with_key (Rng.key 42) (fun () -> rand float32 [| 3 |]) v} *)

module Rng : sig
  (** One splittable random generator, reached two ways.

      Draws come from a Threefry generator whose state is a {e key}. Every
      sampler here is a pure function of the key it is given: the same key,
      dtype and shape give the same values — eagerly, under {!Rune.val-jit}, and
      on every device. Fresh values come from fresh keys, by {!split}ting a key
      into independent subkeys or {!fold_in}ning a counter.

      Naming a key per draw site is precise but verbose, so a
      {{!section:scope} scope} lets one key stand for a region: the keyless
      samplers ([Nx.rand], [Nx.randn], …) draw successive subkeys of its root.
      The two are the same generator — [Nx.rand] is {!uniform} on a subkey of
      the scope — and both are as strong as the key involved.

      Which to reach for is not "explicit under a transform, scope elsewhere". A
      scope rooted at a traced or batched key traces and batches too (see
      {{!section:scope}Scope}). Pass keys when you want a particular draw pinned
      to a particular name, and results that survive inserting a draw elsewhere;
      open a scope when you want the draws in a region decorrelated and would
      rather not thread a subkey to each one. *)

  type ('a, 'b) tensor := ('a, 'b) t

  type t = private (int32, int32_elt) tensor
  (** The type for keys and batches of keys. A key is the generator: its whole
      state, two 32-bit words held in an int32 tensor of shape [[|2|]], from
      which every draw is a pure function. A batch of keys, from {!split_batch},
      puts axes before the words, as in [[|n; 2|]], and each lane of a
      {!Rune.val-vmap} over it sees one key. The samplers take one key.

      Only this module builds keys: arithmetic or slicing on a key gives a
      tensor, which no sampler takes. A key coerces to its tensor,
      [(k :> Nx.int32_t)], to read its words or save them, and {!of_tensor}
      turns saved words back into a key. A key in a structure, a jitted
      function's argument or a mapped argument of {!Rune.val-vmap} is walked
      with {!ptree}. *)

  (** {1:keys Keys} *)

  val key : int -> t
  (** [key seed] is the key for [seed]. Equal seeds give equal keys. *)

  val of_tensor : (int32, int32_elt) tensor -> t
  (** [of_tensor t] is the key, or the batch of keys, whose words are [t]: the
      inverse of the coercion [(k :> Nx.int32_t)], for words that were saved or
      loaded.

      Raises [Invalid_argument] if [t]'s last axis does not have length [2]. *)

  val split : ?n:int -> t -> t array
  (** [split ?n k] is [n] independent subkeys derived from [k] ([n] defaults to
      [2]). Deterministic, and the subkeys are independent of each other; derive
      one subkey per consumer instead of reusing [k].

      Raises [Invalid_argument] if [n < 1]. *)

  val split_batch : n:int -> t -> t
  (** [split_batch ~n k] is [split ~n k] as one batch of keys, of shape
      [[|n; 2|]]: row [i] holds the words of [(split ~n k).(i)]. It is the
      argument that gives each lane of a {!Rune.val-vmap} its own key:

      {v
      Rune.vmap
        Nx.Ptree.(Nx.Rng.ptree @-> returns tensor)
        (fun k -> Nx.Rng.normal k Nx.float32 [| 3 |])
        (Nx.Rng.split_batch ~n:8 key)
      v}

      A sampler takes one key, so it raises on a batch outside the map.

      Raises [Invalid_argument] if [n < 1]. *)

  val fold_in : t -> int -> t
  (** [fold_in k data] is the subkey of [k] indexed by [data]: distinct [data]
      values give independent keys. Use it to derive per-step keys from a root
      key and a loop counter. *)

  val fold_in_tensor : t -> (int32, int32_elt) tensor -> t
  (** [fold_in_tensor k data] is {!fold_in} for a [data] known only at run time
      — a step counter carried through a compiled loop, a device index. [data]
      is a scalar [int32] tensor, and the result agrees with [fold_in k i]
      whenever [data] holds [i].

      Unlike {!fold_in}, this keeps the derivation inside the computation, so
      the subkey tracks a traced counter instead of freezing whatever value it
      held at trace time. *)

  val ptree : t Ptree.t
  (** [ptree] is the structure of a key or a batch of keys: one [int32] tensor,
      at the root path. Rebuilding it checks the tensor as {!of_tensor} does. *)

  (** {1:samplers Explicit samplers}

      Every sampler is pure: the same key and arguments always produce the same
      values. A sampler without parameters takes the dtype and shape of the
      draw. A sampler with parameters takes them as tensors, elementwise, and
      the draw has their shape and dtype: a tensor of rates gives one Poisson
      count per rate, and a parameter that is a traced or batched tensor traces
      or batches the draw with it. A scalar parameter is a broadcast scalar,
      [broadcast_to shape (scalar dtype v)], a view that allocates nothing.
      Location and scale are never parameters: [uniform] is on [\[0, 1)],
      [normal] is standard, [exponential] has rate 1, and the caller's own [add]
      and [mul] place them, the same line for a scalar and a tensor.

      Parameters are data, so their values are not checked: an argument outside
      the distribution's domain gives the result the arithmetic gives, NaN for a
      float draw, as [log] of a negative does. Each sampler states its domain.
      Shapes and dtypes are known when the program is built, so those are still
      checked and raise [Invalid_argument].

      Float draws are computed at float64 for float64 parameters and at float32
      otherwise, then returned at the parameters' dtype. *)

  val bits : t -> int array -> int32_t
  (** [bits k shape] is a tensor of uniformly random 32-bit words, the raw
      output of the generator that every sampler here is built from. For a
      distribution this module does not provide: build it on [bits] and it is as
      pure and as transform-safe as the rest. [uniform] at float32 is the low 24
      bits of these words scaled by [2 ** -24]. *)

  val uniform : t -> (float, 'b) dtype -> int array -> (float, 'b) tensor
  (** [uniform k dtype shape] samples uniformly from [\[0, 1)].

      A draw is a multiple of [2 ** -p], where [p] is the significand width of
      [dtype]; [1] itself is unreachable. *)

  val normal : t -> (float, 'b) dtype -> int array -> (float, 'b) tensor
  (** [normal k dtype shape] samples the standard normal distribution (mean 0,
      variance 1) via the Box-Muller transform over two {!uniform} draws. *)

  val randint :
    t -> ?low:int -> high:int -> int array -> (int32, int32_elt) tensor
  (** [randint k ~high shape] samples integers uniformly from [\[low, high)].
      [low] defaults to [0]. The result is [int32]; cast it for a wider or
      narrower integer.

      The draw comes from a 24-bit {!uniform}, so a range wider than [2 ** 24]
      leaves some values unreachable.

      Raises [Invalid_argument] if [low >= high], or if either bound falls
      outside [int32]. *)

  val bernoulli : t -> (float, 'b) tensor -> (bool, bool_elt) tensor
  (** [bernoulli k p] samples booleans that are [true] with probability [p],
      elementwise. The comparison runs at 24 random bits unless [p] is float64.
      A [p] above [1] is always [true]; below [0], or NaN, always [false]. *)

  val truncated_normal :
    t -> (float, 'b) tensor -> (float, 'b) tensor -> (float, 'b) tensor
  (** [truncated_normal k lower upper] samples the standard normal distribution
      conditioned on landing in [[lower, upper]], elementwise; the bounds
      broadcast against each other and may be given in either order.

      Drawn by inverting the conditioned distribution rather than by rejecting
      out-of-range samples: one draw per element whatever the bounds, so the
      cost does not grow as the interval narrows, and the draw is differentiable
      in both bounds. At float64 the draw carries double precision; at narrower
      dtypes about seven digits, the precision of {!erfinv} there.

      The bounds enter through {!erf}, which reaches [±1] at about 8.3 standard
      deviations at float64 and 5.4 at narrower dtypes: an interval beyond that
      collapses onto its bound nearer zero. *)

  val gumbel : t -> (float, 'b) dtype -> int array -> (float, 'b) tensor
  (** [gumbel k dtype shape] samples the standard Gumbel distribution, the
      limiting distribution of a maximum.

      Adding it to unnormalised log-probabilities and taking the argmax samples
      the distribution they describe, which is what {!categorical} does; a
      softmax in place of the argmax gives the relaxed, differentiable form. *)

  val exponential : t -> (float, 'b) dtype -> int array -> (float, 'b) tensor
  (** [exponential k dtype shape] samples the exponential distribution with rate
      1. Scale by [1 /. rate] for another rate. *)

  val gamma : t -> (float, 'b) tensor -> (float, 'b) tensor
  (** [gamma k concentration] samples the gamma distribution with the given
      concentration (the shape parameter, named to avoid colliding with the
      tensor shape) and unit rate, elementwise; [concentration] must be
      positive. Divide by a rate, or multiply by a scale, for the two-parameter
      family.

      Other distributions follow from it: a chi-square with [k] degrees of
      freedom is [2] times a gamma of concentration [k /. 2], and Student's t is
      a normal over the square root of a chi-square over its degrees of freedom.

      {b This sampler is not exact.} Every algorithm for the gamma rejects, and
      a rejection loop cannot be traced, so a fixed eight attempts are drawn and
      the first acceptance taken. Roughly one element in [1e14] is accepted by
      none and falls back to the distribution's mean. Its derivative in
      [concentration] flows through the accepted proposal alone, without the
      acceptance correction, so it is a biased estimator. *)

  val beta : t -> (float, 'b) tensor -> (float, 'b) tensor -> (float, 'b) tensor
  (** [beta k a b] samples the beta distribution on [[0, 1]] with concentrations
      [a] and [b], in that order (Beta(a, b) is the mirror image of Beta(b, a)),
      elementwise; the two broadcast against each other and must be positive.
      Built from two {!gamma} draws, whose approximation and biased derivative
      it inherits. *)

  val dirichlet : t -> (float, 'b) tensor -> (float, 'b) tensor
  (** [dirichlet k concentration] samples the Dirichlet distribution whose
      components are the last axis of [concentration], one draw per row; the
      result has the shape of [concentration] and every row sums to one. The
      concentrations must be positive.

      Built from one {!gamma} draw, whose approximation and biased derivative it
      inherits.

      Raises [Invalid_argument] if the last axis of [concentration] has fewer
      than two components. *)

  val poisson : t -> (float, 'b) tensor -> int32_t
  (** [poisson k rate] samples the Poisson distribution with the given rate,
      elementwise, at any rate; [rate] must be positive and finite, and a rate
      of zero, a negative rate or NaN gives a count of [0].

      Below 10 the count is read off the cumulative distribution with one
      uniform, exactly. From 10 up it comes from a transformed rejection sampler
      run for a fixed sixteen rounds, so like {!gamma} it is not quite exact:
      about one element in [5e9] is accepted by no round and takes its last
      proposal, a draw from the envelope with mean near [rate]. The cost per
      element is the same at every rate.

      Computed at [rate]'s compute dtype, so a float32 rate compiles on every
      device. Float32 places the proposals exactly up to a rate of about [1e5];
      give a float64 rate beyond that. *)

  val categorical : t -> ?axis:int -> (float, 'a) tensor -> int64_t
  (** [categorical k logits] samples category indices from unnormalised
      log-probabilities: one index per row of [logits] along [axis], which
      defaults to [-1] (the last axis). The result has the shape of [logits]
      with [axis] removed; broadcast [logits] for more draws than rows.

      Raises [Invalid_argument] if [logits] is a float8 type, or if [axis] is
      out of bounds or has length [0]. *)

  val permutation : t -> int -> int64_t
  (** [permutation k n] is a random permutation of \[[0], [n-1]\].

      Raises [Invalid_argument] if [n <= 0]. *)

  val shuffle : t -> ('a, 'b) tensor -> ('a, 'b) tensor
  (** [shuffle k t] is [t] with its first axis randomly permuted. Scalars are
      returned unchanged. *)

  (** {1:scope Scope}

      The scope is where a key enters once instead of at every call: the keyless
      samplers draw successive subkeys of its root, so draws inside it are
      decorrelated without deriving a subkey by hand per site.

      A scope is exactly as strong as its root key. Every draw is {!fold_in} of
      the root, a tensor computation, so a root that is a jitted function's
      input leaf or a {!Rune.val-vmap} mapped axis makes the whole scope traced
      or batched — the keyless samplers then compile and decorrelate just as the
      keyed ones do. A root a transform closes over, such as [with_key (key 42)]
      inside a jitted function, is a constant of that transform, and under
      {!Rune.val-jit} raises rather than freeze one draw into the compiled
      program.

      What a scope gives up against passing keys explicitly is
      order-independence: inserting a draw shifts every draw after it. *)

  val with_key : t -> (unit -> 'a) -> 'a
  (** [with_key k f] runs [f] in a scope rooted at [k]. The keyless samplers
      inside [f] draw successive subkeys of [k], so the same [k] and the same
      draw sequence give the same values. Scopes nest: an inner one replaces the
      outer for its duration.

      For a seeded scope, root it at one: [with_key (key 42) f].

      The scope is an effect handler, so it is per-fiber and per-domain: a draw
      on a domain spawned inside [f] does not see it. *)

  val next_key : unit -> t
  (** [next_key ()] draws a fresh subkey from the current scope. Two calls
      always return different keys. This is what the keyless samplers call.

      Outside any scope the subkey comes from a per-domain generator seeded from
      system entropy, so unscoped draws differ from run to run. Open a scope to
      make them reproducible. *)
end

(** {2:keyless Keyless samplers}

    A closed set: the draws reached for reflexively, taking their subkey from
    the ambient scope (see {!Rng.with_key}) instead of a key argument. [rand] is
    {!Rng.uniform} on \[[0], [1]) and [randn] is {!Rng.normal}, keeping the
    names the ecosystem gives them; the rest match their keyed twin's name and
    take its arguments minus the key, and raise whatever it raises.

    Every other distribution — {!Rng.gamma}, {!Rng.beta}, {!Rng.dirichlet},
    {!Rng.poisson}, {!Rng.gumbel}, {!Rng.exponential} — lives in {!module-Rng}
    only. Reaching for one of those is a deliberate act, by which point a key is
    at hand; [Rng.gamma (Rng.next_key ()) ...] covers the case where it is not.
    Keeping them out of this namespace also leaves [gamma] and [beta] free for
    the special functions of those names, which belong beside {!erf}. *)

val rand : (float, 'b) dtype -> int array -> (float, 'b) t
(** [rand dtype shape] samples uniformly from \[[0], [1]).

    Raises [Invalid_argument] if a shape dimension is negative. *)

val randn : (float, 'b) dtype -> int array -> (float, 'b) t
(** [randn dtype shape] samples from the standard normal distribution (mean 0,
    variance 1).

    Raises [Invalid_argument] if a shape dimension is negative. *)

val randint : ?low:int -> high:int -> int array -> (int32, int32_elt) t
(** [randint ~high shape] samples integers uniformly from \[[low], [high]).
    [low] defaults to [0]. The result is [int32]; cast it for a wider or
    narrower integer.

    Raises [Invalid_argument] if [low >= high], or if either bound falls outside
    [int32]. *)

val bernoulli : (float, 'b) t -> bool_t
(** [bernoulli p] samples booleans that are [true] with probability [p],
    elementwise. See {!Rng.bernoulli}. *)

val truncated_normal : (float, 'b) t -> (float, 'b) t -> (float, 'b) t
(** [truncated_normal lower upper] samples the standard normal conditioned on
    landing in \[[lower], [upper]\], elementwise. See {!Rng.truncated_normal}.
*)

val categorical : ?axis:int -> (float, 'a) t -> int64_t
(** [categorical logits] samples one category index per row of [logits] along
    [axis]. See {!Rng.categorical}.

    Raises [Invalid_argument] if [logits] is a float8 type, or if [axis] is out
    of bounds or has length [0]. *)

val permutation : int -> int64_t
(** [permutation n] is a random permutation of \[[0], [n-1]\].

    Raises [Invalid_argument] if [n <= 0]. *)

val shuffle : ('a, 'b) t -> ('a, 'b) t
(** [shuffle t] is [t] with its first axis randomly permuted. Scalars are
    returned unchanged. *)

(** {1:shape Shape manipulation} *)

val reshape : int array -> ('a, 'b) t -> ('a, 'b) t
(** [reshape shape t] is a view of [t] with the given [shape].

    At most one dimension may be [-1]; it is inferred from the total number of
    elements. The product of [shape] must equal {!numel} [t].

    Raises [Invalid_argument] if [shape] is incompatible, contains more than one
    [-1], or cannot view [t]'s layout, as a transpose's cannot be flattened;
    call {!contiguous} first.

    {@ocaml[
      # create int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
        |> reshape [| 2; 3 |]
      - : (int32, int32_elt) t = int32 [2,3] [[1, 2, 3],
                                              [4, 5, 6]]
      # create int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
        |> reshape [| 3; -1 |]
      - : (int32, int32_elt) t = int32 [3,2] [[1, 2],
                                              [3, 4],
                                              [5, 6]]
    ]}

    See also {!flatten}, {!unflatten}, {!ravel}. *)

val broadcast_to : int array -> ('a, 'b) t -> ('a, 'b) t
(** [broadcast_to shape t] is a view of [t] broadcast to [shape].

    Dimensions are aligned from the right; each dimension of [t] must be [1] or
    equal to the corresponding target dimension. Broadcast dimensions have zero
    byte-stride (no copy).

    Raises [Invalid_argument] if the shapes are incompatible.

    {@ocaml[
      # create int32 [| 1; 3 |] [| 1l; 2l; 3l |]
        |> broadcast_to [| 3; 3 |]
      - : (int32, int32_elt) t = int32 [3,3] [[1, 2, 3],
                                              [1, 2, 3],
                                              [1, 2, 3]]
    ]}

    See also {!broadcasted}, {!expand}. *)

val broadcasted :
  ?reverse:bool -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
(** [broadcasted ?reverse t1 t2] is [(t1', t2')] where both are broadcast to
    their common shape. When [reverse] is [true] (default [false]), returns
    [(t2', t1')].

    Raises [Invalid_argument] if the shapes are incompatible.

    See also {!broadcast_to}, {!broadcast_arrays}. *)

val expand : int array -> ('a, 'b) t -> ('a, 'b) t
(** [expand shape t] is like {!broadcast_to} but [-1] in [shape] preserves the
    corresponding dimension of [t].

    Raises [Invalid_argument] if any dimension in [shape] is negative (other
    than [-1]).

    {@ocaml[
      # ones float32 [| 1; 4; 1 |]
        |> expand [| 3; -1; 5 |] |> shape
      - : int array = [|3; 4; 5|]
    ]}

    See also {!broadcast_to}. *)

val flatten : ?start_dim:int -> ?end_dim:int -> ('a, 'b) t -> ('a, 'b) t
(** [flatten ?start_dim ?end_dim t] collapses dimensions [start_dim] through
    [end_dim] (inclusive) into a single dimension. [start_dim] defaults to [0].
    [end_dim] defaults to [-1] (last). Negative indices count from the end. It
    is a view where the layout allows one, and a copy otherwise.

    Raises [Invalid_argument] if indices are out of bounds.

    {@ocaml[
      # zeros float32 [| 2; 3; 4 |] |> flatten |> shape
      - : int array = [|24|]
      # zeros float32 [| 2; 3; 4; 5 |]
        |> flatten ~start_dim:1 ~end_dim:2 |> shape
      - : int array = [|2; 12; 5|]
    ]}

    See also {!unflatten}, {!ravel}. *)

val unflatten : int -> int array -> ('a, 'b) t -> ('a, 'b) t
(** [unflatten dim sizes t] expands dimension [dim] into multiple dimensions
    given by [sizes]. At most one element of [sizes] may be [-1] (inferred). The
    product of [sizes] must equal the size of dimension [dim].

    Raises [Invalid_argument] if the product mismatches or [dim] is out of
    bounds.

    {@ocaml[
      # zeros float32 [| 2; 12; 5 |]
        |> unflatten 1 [| 3; 4 |] |> shape
      - : int array = [|2; 3; 4; 5|]
    ]}

    See also {!flatten}. *)

val ravel : ('a, 'b) t -> ('a, 'b) t
(** [ravel t] is [t] reshaped to 1-D. Returns a view when possible.

    Raises [Invalid_argument] if [t] cannot be flattened without copying; call
    {!contiguous} first.

    See also {!flatten}, {!contiguous}. *)

val squeeze : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [squeeze ?axes t] removes dimensions of size 1. When [axes] is given, only
    those axes are removed. Negative indices count from the end.

    Raises [Invalid_argument] if a specified axis does not have size 1.

    {@ocaml[
      # ones float32 [| 1; 3; 1; 4 |]
        |> squeeze |> shape
      - : int array = [|3; 4|]
      # ones float32 [| 1; 3; 1; 4 |]
        |> squeeze ~axes:[ 0 ] |> shape
      - : int array = [|3; 1; 4|]
    ]}

    See also {!unsqueeze}. *)

val unsqueeze : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [unsqueeze ?axes t] inserts dimensions of size 1 at the positions listed in
    [axes]. Positions refer to the result tensor.

    Raises [Invalid_argument] if [axes] is not specified, contains duplicates,
    or values are out of bounds.

    {@ocaml[
      # create float32 [| 3 |] [| 1.; 2.; 3. |]
        |> unsqueeze ~axes:[ 0; 2 ] |> shape
      - : int array = [|1; 3; 1|]
    ]}

    See also {!squeeze}. *)

val transpose : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [transpose ?axes t] permutes the dimensions of [t].

    [axes] must be a permutation of [[0; …; ndim t - 1]]. When omitted, reverses
    all dimensions. Returns a view (no copy).

    Raises [Invalid_argument] if [axes] is not a valid permutation.

    {@ocaml[
      # create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
        |> transpose
      - : (int32, int32_elt) t = int32 [3,2] [[1, 4],
                                              [2, 5],
                                              [3, 6]]
    ]}

    See also {!matrix_transpose}, {!moveaxis}, {!swapaxes}. *)

val flip : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [flip ?axes t] reverses elements along the given [axes]. When omitted, flips
    all dimensions.

    Raises [Invalid_argument] if any axis is out of bounds.

    {@ocaml[
      # create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
        |> flip ~axes:[ 1 ]
      - : (int32, int32_elt) t = int32 [2,3] [[3, 2, 1],
                                              [6, 5, 4]]
    ]} *)

val moveaxis : int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [moveaxis src dst t] moves dimension [src] to position [dst].

    Raises [Invalid_argument] if either index is out of bounds.

    See also {!transpose}, {!swapaxes}. *)

val swapaxes : int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [swapaxes a1 a2 t] exchanges dimensions [a1] and [a2].

    Raises [Invalid_argument] if either index is out of bounds.

    See also {!transpose}, {!moveaxis}. *)

val roll : ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [roll ?axis shift t] shifts elements along [axis] by [shift] positions,
    wrapping around. When [axis] is omitted, it shifts the flattened tensor and
    keeps [t]'s shape. Negative [shift] rolls backward.

    Raises [Invalid_argument] if [axis] is out of bounds.

    {@ocaml[
      # create int32 [| 5 |] [| 1l; 2l; 3l; 4l; 5l |]
        |> roll 2
      - : (int32, int32_elt) t = int32 [5] [4, 5, ..., 2, 3]
    ]} *)

val pad : (int * int) array -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [pad widths value t] pads [t] with [value]. [widths.(i)] is
    [(before, after)] for dimension [i].

    Raises [Invalid_argument] if [Array.length widths] does not match {!ndim}
    [t] or any width is negative.

    {@ocaml[
      # create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
        |> pad [| (1, 1); (1, 1) |] 0. |> shape
      - : int array = [|4; 4|]
    ]}

    See also {!shrink}. *)

val shrink : (int * int) array -> ('a, 'b) t -> ('a, 'b) t
(** [shrink ranges t] extracts a slice where [ranges.(i)] is [(start, stop)]
    (exclusive) for dimension [i]. Returns a view.

    {@ocaml[
      # create int32 [| 3; 3 |]
          [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |]
        |> shrink [| (1, 3); (0, 2) |]
      - : (int32, int32_elt) t = int32 [2,2] [[4, 5],
                                              [7, 8]]
    ]}

    See also {!pad}. *)

val tile : int array -> ('a, 'b) t -> ('a, 'b) t
(** [tile reps t] is [t] repeated according to [reps]. [reps.(i)] gives the
    repetition count along dimension [i]. If [reps] is longer than {!ndim} [t],
    dimensions are prepended.

    Raises [Invalid_argument] if any repetition count is negative.

    {@ocaml[
      # create int32 [| 1; 2 |] [| 1l; 2l |]
        |> tile [| 2; 3 |]
      - : (int32, int32_elt) t = int32 [2,6] [[1, 2, ..., 1, 2],
                                              [1, 2, ..., 1, 2]]
    ]}

    See also {!repeat}. *)

val repeat : ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [repeat ?axis n t] repeats each element [n] times along [axis]. When [axis]
    is omitted, operates on the flattened tensor.

    Raises [Invalid_argument] if [n] is negative or [axis] is out of bounds.

    {@ocaml[
      # create int32 [| 3 |] [| 1l; 2l; 3l |]
        |> repeat 2
      - : (int32, int32_elt) t = int32 [6] [1, 1, ..., 3, 3]
    ]}

    See also {!tile}. *)

(** {1:combine Combining and splitting} *)

val concatenate : axis:int -> ('a, 'b) t list -> ('a, 'b) t
(** [concatenate ~axis ts] joins tensors along the existing axis [axis]. All
    tensors must have the same shape except on the concatenation axis. To join
    along a flattened view, ravel each tensor first. Always copies.

    Raises [Invalid_argument] if the list is empty or shapes are incompatible.

    {@ocaml[
      # let a = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        let b = create int32 [| 1; 2 |] [| 5l; 6l |] in
        concatenate ~axis:0 [ a; b ]
      - : (int32, int32_elt) t = int32 [3,2] [[1, 2],
                                              [3, 4],
                                              [5, 6]]
    ]}

    See also {!stack}. *)

val stack : ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
(** [stack ?axis ts] joins tensors along a {e new} axis. All tensors must have
    identical shape. [axis] defaults to [0]. Negative values count from the end
    of the result shape.

    Raises [Invalid_argument] if the list is empty, shapes differ, or [axis] is
    out of bounds.

    {@ocaml[
      # let a = create int32 [| 2 |] [| 1l; 2l |] in
        let b = create int32 [| 2 |] [| 3l; 4l |] in
        stack [ a; b ]
      - : (int32, int32_elt) t = int32 [2,2] [[1, 2],
                                              [3, 4]]
      # let a = create int32 [| 2 |] [| 1l; 2l |] in
        let b = create int32 [| 2 |] [| 3l; 4l |] in
        stack ~axis:1 [ a; b ]
      - : (int32, int32_elt) t = int32 [2,2] [[1, 3],
                                              [2, 4]]
    ]}

    See also {!concatenate}. *)

val broadcast_arrays : ('a, 'b) t list -> ('a, 'b) t list
(** [broadcast_arrays ts] broadcasts every tensor to their common shape. Returns
    views (no copies).

    Raises [Invalid_argument] if shapes are incompatible.

    See also {!broadcast_to}, {!broadcasted}. *)

val array_split :
  axis:int ->
  [< `Count of int | `Indices of int list ] ->
  ('a, 'b) t ->
  ('a, 'b) t list
(** [array_split ~axis spec t] splits [t] into sub-tensors.

    With [`Count n], divides as evenly as possible (first sections absorb extra
    elements). With [`Indices [i0; i1; …]], splits at the given indices
    producing [\[0, i0)], [\[i0, i1)], …, [\[ik, end)].

    Raises [Invalid_argument] if [axis] is out of bounds or [spec] is invalid.

    {@ocaml[
      # create int32 [| 5 |] [| 1l; 2l; 3l; 4l; 5l |]
        |> array_split ~axis:0 (`Count 3)
      - : (int32, int32_elt) t list = [[1, 2]; [3, 4]; [5]]
    ]}

    See also {!split}. *)

val split : axis:int -> int -> ('a, 'b) t -> ('a, 'b) t list
(** [split ~axis n t] splits [t] into [n] equal parts along [axis].

    Raises [Invalid_argument] if the axis size is not divisible by [n].

    See also {!array_split}. *)

(** {1:conversion Type conversion and copying} *)

val cast : ('c, 'd) dtype -> ('a, 'b) t -> ('c, 'd) t
(** [cast dtype t] is [t] with elements converted to [dtype]. It is [t] itself
    when [t] already has that dtype: a tensor is a value, so only a change of
    dtype allocates. Use {!copy} for fresh storage.

    A float becomes an integer by truncation toward zero, held at the ends of
    the integer's range, and NaN becomes [0]. A real value becomes a complex one
    with no imaginary part, and a complex value a real one by dropping its
    imaginary part.

    {@ocaml[
      # create float32 [| 3 |] [| 1.5; 2.7; 3.1 |]
        |> cast int32
      - : (int32, int32_elt) t = [1, 2, 3]
    ]}

    See also {!contiguous}, {!copy}. *)

val bitcast : ('c, 'd) dtype -> ('a, 'b) t -> ('c, 'd) t
(** [bitcast dtype t] reads the bits of each element of [t] as an element of
    [dtype], without conversion: [t]'s shape, each element keeping its place. It
    reinterprets, where {!cast} converts values. The bits are read in the
    machine's byte order, NaN payloads and subnormals included, and the result
    may share [t]'s storage.

    Raises [Invalid_argument] if the two dtypes differ in width, or if either is
    [bool], whose only bytes are 0 and 1, or [int4] or [uint4], whose elements
    are packed in pairs. A compiled function (under [Rune.jit]) refuses a
    bitcast to or from [float8_e4m3] or [float8_e5m2]: the compiler emulates
    those formats through a wider float, which would change subnormal and
    infinite bits.

    Reading a float's bits as an integer of its width gives a key that sorts as
    the float does once negative keys have their other bits flipped:

    {@ocaml[
      # let bits = create float32 [| 3 |] [| -1.5; 0.; 2. |] |> bitcast int32 in
        let flipped = bitwise_xor bits (scalar int32 Int32.max_int) in
        to_array (where (less_s bits 0l) flipped bits)
      - : int32 array = [|-1069547521l; 0l; 1073741824l|]
    ]}

    See also {!cast}. *)

val contiguous : ('a, 'b) t -> ('a, 'b) t
(** [contiguous t] is [t], sharing its storage, if [t] is C-contiguous from the
    start of its storage, or a fresh contiguous copy otherwise.

    See also {!is_c_contiguous}, {!copy}. *)

val copy : ('a, 'b) t -> ('a, 'b) t
(** [copy t] is a deep copy of [t]. Always allocates new memory; the result is
    contiguous.

    {@ocaml[
      # let x = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        is_c_contiguous (copy (transpose x))
      - : bool = true
    ]}

    See also {!contiguous}. *)

val fill : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [fill v t] is a tensor of [t]'s dtype and shape with every element set to
    [v]; the same as {!full_like} [t v], with the value first so it pipes. [t]
    is unchanged. *)

(** {1:indexing Indexing and slicing}

    Indices are {!int64_t}, which reach every element a tensor can have. An
    index outside its axis reads zero, and an update at one is dropped, however
    far outside it lies. A [D] window clamps its start instead. *)

val get : int list -> ('a, 'b) t -> ('a, 'b) t
(** [get indices t] is the sub-tensor at [indices], indexing from the outermost
    dimension inward. Returns a scalar tensor when all dimensions are indexed;
    otherwise a view of the remaining dimensions. Negative indices count from
    the end.

    Raises [Invalid_argument] if any index is out of bounds.

    {@ocaml[
      # let x =
          create int32 [| 2; 3 |]
            [| 1l; 2l; 3l; 4l; 5l; 6l |]
        in
        get [ 1 ] x
      - : (int32, int32_elt) t = [4, 5, 6]
    ]}

    See also {!item}, {!val-slice}. *)

val slice : index list -> ('a, 'b) t -> ('a, 'b) t
(** [slice specs t] extracts a sub-tensor using advanced indexing.

    Each element of [specs] addresses one axis from left to right:
    - [I i] — single index (reduces dimension; negative from end).
    - [L [i0; i1; …]] — gather listed indices.
    - [R (start, stop)] — half-open range \[[start], [stop]).
    - [Rs (start, stop, step)] — strided range.
    - [A] — full axis (default for trailing axes).
    - [M mask] — rank-1 boolean mask selecting the true positions along the
      axis; [mask]'s length must equal that axis.
    - [N] — insert a new axis of size 1.
    - [D (start, len)] — the run of [len] positions from the run-time value of
      the scalar tensor [start], clamped so the run fits.

    Returns a view for [I], [R], [Rs] with step ±1, [A] and [N]; [L], [M] and
    [D] gather. A traced [M] mask raises under [Rune.jit] (its result shape
    depends on data); a traced [D] start is a gather.

    Raises [Invalid_argument] if specs are out of bounds, if step is zero, or if
    a mask is not rank 1 or its length does not match the axis.

    {@ocaml[
      # let x =
          create int32 [| 3; 3 |]
            [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |]
        in
        slice [ R (0, 2); L [ 0; 2 ] ] x
      - : (int32, int32_elt) t = int32 [2,2] [[1, 3],
                                              [4, 6]]
    ]}

    See also {!get}, {!set}. *)

val set : index list -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [set specs v t] is [t] with [v], broadcast to the selection, at the
    positions [specs] select. [t] is unchanged: a tensor is a value, and this is
    the one way to obtain one that differs from another at chosen positions.
    [specs] uses the index forms of {!val-slice}; every selection is injective,
    so an [L] listing a position twice raises.

    The cost is one copy of [t] plus the selection. A mask alone with a [v] that
    has no extent along the mask (a scalar, or one row per masked row) selects
    through {!where}; a window ([I], [R], [Rs] with step ±1, [A], [N], [D]) is
    one backend window write, which a compiler can perform in place; any other
    combination scatters. Both operands differentiate.

    {@ocaml[
      # let x = zeros float32 [| 2; 3 |] in
        set [ I 1; R (1, 3) ] (create float32 [| 2 |] [| 7.; 8. |]) x
      - : (float, float32_elt) t = float32 [2,3] [[0, 0, 0],
                                                  [0, 7, 8]]
    ]}

    Element-by-element construction is not a loop of [set] (each call copies):
    build the values first with {!create}, {!init}, {!stack} or a filled
    {!of_bigarray}.

    Raises [Invalid_argument] if [specs] are out of bounds, [v] does not
    broadcast to the selection, or an [L] repeats a position.

    See also {!val-slice}, {!scatter}. *)

val item : int list -> ('a, 'b) t -> 'a
(** [item indices t] is the scalar value at [indices]. Indices must cover all
    dimensions.

    Each call allocates its index list. To visit every element, use {!iter_item}
    or {!fold_item}; for indexed reads in a hot loop, index the bigarray of
    {!to_bigarray}.

    Raises [Invalid_argument] if the number of indices is wrong or any index is
    out of bounds.

    See also {!get}. *)

val take : ?axis:int -> indices:int64_t -> ('a, 'b) t -> ('a, 'b) t
(** [take ?axis ~indices t] gathers elements from [t] at [indices] along [axis].
    When [axis] is omitted, [t] is flattened first. An index outside \[[0],
    [size]), negative included, reads zero, eagerly and under [Rune.jit] alike;
    wrap indices with [mod_ (add_s i n) n] or clamp them with {!clamp} yourself.
    At an integer dtype the zero read is index [0]: mask with the index's range
    when the gathered values are themselves positions.

    {@ocaml[
      # let x =
          create int32 [| 5 |]
            [| 0l; 1l; 2l; 3l; 4l |]
        in
        take
          ~indices:(create int64 [| 3 |] [| 1L; 3L; 0L |])
          x
      - : (int32, int32_elt) t = [1, 3, 0]
    ]}

    See also {!scatter}, {!take_along_axis}. *)

val take_along_axis : axis:int -> indices:int64_t -> ('a, 'b) t -> ('a, 'b) t
(** [take_along_axis ~axis ~indices t] gathers values from [t] along [axis]
    using [indices]. [indices] must match [t]'s shape except along [axis]. An
    index outside \[[0], [size along axis]) reads zero, as in {!take}. Useful
    for gathering from {!argmax}/{!argmin} results.

    Raises [Invalid_argument] if shapes are incompatible.

    {@ocaml[
      # let x =
          create float32 [| 2; 3 |]
            [| 4.; 1.; 2.; 3.; 5.; 6. |]
        in
        let idx =
          create int64 [| 2; 1 |] [| 1L; 0L |]
        in
        take_along_axis ~axis:1 ~indices:idx x
      - : (float, float32_elt) t = float32 [2,1] [[1],
                                                  [3]]
    ]}

    See also {!take}, {!scatter}. *)

val scatter :
  ?mode:[ `Set | `Add ] ->
  ?unique_indices:bool ->
  axis:int ->
  indices:int64_t ->
  values:('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [scatter ?mode ?unique_indices ~axis ~indices ~values t] is [t] with
    [values] placed at the positions selected by [indices] along [axis]; the
    tensor-indexed form of {!set}. [indices] must match [t]'s shape except along
    [axis], and [values] is broadcast to [indices]' shape. An update whose index
    lies outside \[[0], [size along axis]), negative included, is dropped,
    eagerly and under [Rune.jit] alike, so [-1] addresses nothing; {!take} and
    {!take_along_axis} read zero at such an index.

    [mode] controls how updates combine with [t]: [`Set] (default) overwrites,
    the last update winning at duplicate positions; [`Add] accumulates every
    update into [t]'s value, a float sum as {!sum} describes, so a position
    whose value and updates sum to exactly zero holds [0.].
    [unique_indices = true] promises that no position is selected twice,
    letting backends write the updates in any order. Where the promise is
    broken, a position selected more than once holds an unspecified one of its
    updates under [`Set] and an unspecified value under [`Add]; every other
    position is exact.

    [scatter] differentiates with respect to both [t] and [values]; a dropped
    update's gradient is zero.

    {@ocaml[
      # let x = zeros float32 [| 2; 3 |] in
        let idx =
          create int64 [| 2; 1 |] [| 1L; 0L |]
        in
        scatter ~axis:1 ~indices:idx
          ~values:(create float32 [| 2; 1 |]
                     [| 10.; 20. |])
          x
      - : (float, float32_elt) t = float32 [2,3] [[0, 10, 0],
                                                  [20, 0, 0]]
    ]}

    Raises [Invalid_argument] if shapes are incompatible.

    See also {!set}, {!take_along_axis}. *)

val compress :
  ?axis:int -> condition:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t
(** [compress ?axis ~condition t] selects elements where [condition] is [true]
    along [axis]. [condition] must be 1-D. When [axis] is omitted, [t] is
    flattened first.

    Raises [Invalid_argument] if the condition length is incompatible.

    {@ocaml[
      # let x =
          create int32 [| 5 |]
            [| 1l; 2l; 3l; 4l; 5l |]
        in
        compress
          ~condition:(create bool [| 5 |]
            [| true; false; true; false; true |])
          x
      - : (int32, int32_elt) t = [1, 3, 5]
    ]}

    See also {!extract}, {!nonzero}. *)

val extract : condition:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t
(** [extract ~condition t] is the 1-D tensor of elements of [t] where
    [condition] is [true]. Both are flattened before comparison.

    Raises [Invalid_argument] if sizes differ.

    See also {!compress}, {!nonzero}. *)

val nonzero : ('a, 'b) t -> int64_t array
(** [nonzero t] is an array of 1-D index tensors, one per dimension, giving the
    coordinates of non-zero elements.

    {@ocaml[
      # let x =
          create int32 [| 3; 3 |]
            [| 0l; 1l; 0l;
               2l; 0l; 3l;
               0l; 0l; 4l |]
        in
        let idx = nonzero x in
        idx.(0), idx.(1)
      - : int64_t * int64_t = ([0, 1, 1, 2], [1, 0, 2, 2])
    ]}

    See also {!argwhere}. *)

val argwhere : ('a, 'b) t -> int64_t
(** [argwhere t] is a 2-D tensor of shape [[k; ndim t]] whose rows are the
    coordinates of the [k] non-zero elements.

    See also {!nonzero}. *)

(** {1:arithmetic Arithmetic}

    Element-wise arithmetic with broadcasting. At [float16], [bfloat16] and the
    float8 dtypes, every element-wise operation, the mathematical functions
    below included, computes at [float32] and rounds once. Each operation [op]
    has variants:
    - [op_s t s] — tensor-scalar.
    - [rop_s s t] — scalar-tensor (reversed operands). *)

val add : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [add a b] is the element-wise sum of [a] and [b]. *)

val add_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [add_s t s] adds scalar [s] to each element of [t]. *)

val sub : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sub a b] is the element-wise difference [a - b]. *)

val sub_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [sub_s t s] subtracts scalar [s] from each element. *)

val rsub_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rsub_s s t] is [s - t] element-wise. *)

val mul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [mul a b] is the element-wise product of [a] and [b]. *)

val mul_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [mul_s t s] multiplies each element by scalar [s]. *)

val div : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [div a b] is the element-wise quotient [a / b].

    Float dtypes use true division. Integer dtypes truncate toward zero, and an
    integer divided by zero is zero.

    {@ocaml[
      # let x = create int32 [| 2 |] [| -7l; 8l |] in
        let y = create int32 [| 2 |] [| 2l; 2l |] in
        div x y
      - : (int32, int32_elt) t = [-3, 4]
    ]} *)

val div_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [div_s t s] divides each element by scalar [s]. *)

val rdiv_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rdiv_s s t] is [s / t] element-wise. *)

val pow : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [pow base exp] is [base] raised to [exp] element-wise. *)

val pow_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [pow_s t s] raises each element to scalar power [s]. *)

val rpow_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rpow_s s t] is [s{^t}] element-wise. *)

val mod_ : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [mod_ a b] is the element-wise remainder of [a / b], of the sign of [a]. An
    integer remainder by zero is zero. *)

val mod_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [mod_s t s] is the remainder of each element divided by scalar [s]. *)

val rmod_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rmod_s s t] is [s mod t] element-wise. *)

val neg : ('a, 'b) t -> ('a, 'b) t
(** [neg t] is the element-wise negation of [t]. *)

(** {1:complex Complex numbers}

    A complex tensor stores an interleaved real and imaginary component. These
    functions move between such a tensor and the float tensors holding its
    components. All are element-wise and return a fresh tensor, never a view.

    The ones producing a float tensor take that dtype first. It selects the
    result's storage precision independently of the input's, so the arithmetic
    still happens at the input's precision: [magnitude float64] of a [complex64]
    tensor widens a float32 result rather than recomputing it. The matching
    pairs are [complex64] with [float32], and [complex128] with [float64].

    {2:complex_nonfinite Non-finite components}

    Only {!real} and {!magnitude} read a component directly. The others reach
    the imaginary component by rotating it into the real one, which multiplies
    the two components together, so a non-finite component poisons the other:
    {!imag}, {!angle}, {!val-complex}, and {!conjugate} produce NaN components
    wherever an input component is infinite or NaN. Finite inputs are
    unaffected, including components near the dtype maximum. *)

val real : (float, 'b) dtype -> (Complex.t, 'a) t -> (float, 'b) t
(** [real dt z] is the real component of each element of [z].

    {@ocaml[
      # create complex64 [| 2 |] [| Complex.{ re = 3.; im = 4. }; Complex.i |]
        |> real float32
      - : (float, float32_elt) t = [3, 0]
    ]}

    See also {!imag}, {!magnitude}. *)

val imag : (float, 'b) dtype -> (Complex.t, 'a) t -> (float, 'b) t
(** [imag dt z] is the imaginary component of each element of [z].

    See also {!real}, {!val-complex}. *)

val magnitude : (float, 'b) dtype -> (Complex.t, 'a) t -> (float, 'b) t
(** [magnitude dt z] is the element-wise modulus of [z]. It is named apart from
    {!abs}, which preserves the dtype and so cannot leave the complex domain.

    Computed without intermediate overflow, so components large enough that
    [re² + im²] would saturate still yield a finite magnitude.

    {@ocaml[
      # create complex64 [| 2 |] [| Complex.{ re = 3.; im = 4. }; Complex.i |]
        |> magnitude float32
      - : (float, float32_elt) t = [5, 1]
    ]}

    See also {!angle}, {!val-complex}. *)

val angle : (float, 'b) dtype -> (Complex.t, 'a) t -> (float, 'b) t
(** [angle dt z] is the element-wise argument of [z] in radians, in \[[-π],
    [π]\].

    The negative real axis is a branch cut and the sign of a zero imaginary
    component picks the side: [-1. + 0.i] has argument [π], [-1. - 0.i] has
    argument [-π]. Zero has argument zero.

    See also {!magnitude}, {!val-complex}. *)

val complex :
  (Complex.t, 'c) dtype ->
  re:(float, 'a) t ->
  im:(float, 'a) t ->
  (Complex.t, 'c) t
(** [complex dt ~re ~im] assembles a complex tensor from its components, which
    are broadcast together.

    {@ocaml[
      # complex complex64
          ~re:(create float32 [| 2 |] [| 3.; 0. |])
          ~im:(create float32 [| 2 |] [| 4.; 1. |])
      - : (Complex.t, complex32_elt) t = [(3+4i), (0+1i)]
    ]}

    See also {!real}, {!imag}. *)

val conjugate : ('a, 'b) t -> ('a, 'b) t
(** [conjugate t] negates the imaginary component of each element. Real dtypes
    are returned unchanged. *)

(** {1:math Mathematical functions} *)

(** {2:math_basic Basic} *)

val abs : ('a, 'b) t -> ('a, 'b) t
(** [abs t] is the element-wise absolute value. *)

val sign : ('a, 'b) t -> ('a, 'b) t
(** [sign t] is [-1], [0], or [1] according to the sign of each element. For
    unsigned types, returns [1] for non-zero, [0] for zero.

    {@ocaml[
      # create float32 [| 3 |] [| -2.; 0.; 3.5 |]
        |> sign
      - : (float, float32_elt) t = [-1, 0, 1]
    ]} *)

val square : ('a, 'b) t -> ('a, 'b) t
(** [square t] is the element-wise square. *)

val sqrt : ('a, 'b) t -> ('a, 'b) t
(** [sqrt t] is the element-wise square root. *)

val rsqrt : ('a, 'b) t -> ('a, 'b) t
(** [rsqrt t] is the element-wise reciprocal square root ([1 / sqrt t]). *)

val recip : ('a, 'b) t -> ('a, 'b) t
(** [recip t] is the element-wise reciprocal ([1 / t]). *)

(** {2:math_exp Exponential and logarithmic} *)

val log : ('a, 'b) t -> ('a, 'b) t
(** [log t] is the element-wise natural logarithm. *)

val log2 : ('a, 'b) t -> ('a, 'b) t
(** [log2 t] is the element-wise base-2 logarithm. *)

val exp : ('a, 'b) t -> ('a, 'b) t
(** [exp t] is the element-wise exponential. *)

val exp2 : ('a, 'b) t -> ('a, 'b) t
(** [exp2 t] is [2{^t}] element-wise. *)

(** {2:math_trig Trigonometric} *)

val sin : ('a, 'b) t -> ('a, 'b) t
(** [sin t] is the element-wise sine. *)

val cos : ('a, 'b) t -> ('a, 'b) t
(** [cos t] is the element-wise cosine. *)

val tan : ('a, 'b) t -> ('a, 'b) t
(** [tan t] is the element-wise tangent. *)

val asin : ('a, 'b) t -> ('a, 'b) t
(** [asin t] is the element-wise arcsine. *)

val acos : ('a, 'b) t -> ('a, 'b) t
(** [acos t] is the element-wise arccosine. *)

val atan : ('a, 'b) t -> ('a, 'b) t
(** [atan t] is the element-wise arctangent. *)

val atan2 : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [atan2 y x] is the element-wise two-argument arctangent, returning angles in
    \[[-π], [π]\]. *)

(** {2:math_hyp Hyperbolic} *)

val sinh : ('a, 'b) t -> ('a, 'b) t
(** [sinh t] is the element-wise hyperbolic sine. *)

val cosh : ('a, 'b) t -> ('a, 'b) t
(** [cosh t] is the element-wise hyperbolic cosine. *)

val tanh : ('a, 'b) t -> ('a, 'b) t
(** [tanh t] is the element-wise hyperbolic tangent. *)

val asinh : ('a, 'b) t -> ('a, 'b) t
(** [asinh t] is the element-wise inverse hyperbolic sine. *)

val acosh : ('a, 'b) t -> ('a, 'b) t
(** [acosh t] is the element-wise inverse hyperbolic cosine. *)

val atanh : ('a, 'b) t -> ('a, 'b) t
(** [atanh t] is the element-wise inverse hyperbolic tangent. *)

(** {2:math_round Rounding} *)

val trunc : ('a, 'b) t -> ('a, 'b) t
(** [trunc t] rounds each element toward zero. *)

val ceil : ('a, 'b) t -> ('a, 'b) t
(** [ceil t] rounds each element toward positive infinity. *)

val floor : ('a, 'b) t -> ('a, 'b) t
(** [floor t] rounds each element toward negative infinity. *)

val round : ('a, 'b) t -> ('a, 'b) t
(** [round t] rounds each element to the nearest integer. Ties round away from
    zero (not banker's rounding).

    {@ocaml[
      # create float32 [| 4 |] [| 2.5; 3.5; -2.5; -3.5 |]
        |> round
      - : (float, float32_elt) t = [3, 4, -3, -4]
    ]} *)

(** {2:math_misc Other} *)

val hypot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [hypot x y] is [sqrt(x² + y²)] computed without intermediate overflow.

    {@ocaml[
      # hypot (scalar float32 3.) (scalar float32 4.)
        |> item []
      - : float = 5.
    ]} *)

val lerp : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [lerp a b w] is the linear interpolation [a + w * (b - a)]. [w] is typically
    in \[[0], [1]\].

    {@ocaml[
      # let a = create float32 [| 2 |] [| 1.; 2. |] in
        let b = create float32 [| 2 |] [| 5.; 8. |] in
        lerp a b (scalar float32 0.25)
      - : (float, float32_elt) t = [2, 3.5]
    ]} *)

val isinf : ('a, 'b) t -> (bool, bool_elt) t
(** [isinf t] is [true] where [t] is positive or negative infinity, [false]
    elsewhere. Non-float dtypes always return all [false].

    {@ocaml[
      # create float32 [| 4 |]
          [| 1.; Float.infinity;
             Float.neg_infinity; Float.nan |]
        |> isinf
      - : (bool, bool_elt) t = [false, true, true, false]
    ]}

    See also {!isnan}, {!isfinite}. *)

val isnan : ('a, 'b) t -> (bool, bool_elt) t
(** [isnan t] is [true] where [t] is NaN, [false] elsewhere. Non-float dtypes
    always return all [false].

    See also {!isinf}, {!isfinite}. *)

val isfinite : ('a, 'b) t -> (bool, bool_elt) t
(** [isfinite t] is [true] where [t] is neither infinite nor NaN. Non-float
    dtypes always return all [true].

    See also {!isinf}, {!isnan}. *)

(** {1:comparison Comparison and logic} *)

val less : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [less a b] is [true] where [a < b], [false] elsewhere. *)

val less_s : ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [less_s t s] is [true] where [t < s]. *)

val not_equal : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [not_equal a b] is [true] where [a ≠ b], [false] elsewhere. *)

val not_equal_s : ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [not_equal_s t s] is [true] where [t ≠ s]. *)

val equal : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [equal a b] is [true] where [a = b], [false] elsewhere. *)

val equal_s : ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [equal_s t s] is [true] where [t = s]. *)

val greater : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [greater a b] is [true] where [a > b], [false] elsewhere. *)

val greater_s : ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [greater_s t s] is [true] where [t > s]. *)

val less_equal : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [less_equal a b] is [true] where [a ≤ b], [false] elsewhere. *)

val less_equal_s : ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [less_equal_s t s] is [true] where [t ≤ s]. *)

val greater_equal : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [greater_equal a b] is [true] where [a ≥ b], [false] elsewhere. *)

val greater_equal_s : ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [greater_equal_s t s] is [true] where [t ≥ s]. *)

val array_equal : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [array_equal a b] is a scalar [true] iff all elements of [a] and [b] are
    equal. Returns [false] if shapes differ.

    {@ocaml[
      # let a = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        let b = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        array_equal a b |> item []
      - : bool = true
    ]} *)

val maximum : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [maximum a b] is the element-wise maximum of [a] and [b]. On floats it is
    the IEEE 754 maximum: NaN propagates, and [-0.] is less than [0.], so the
    maximum of [-0.] and [0.] is [0.] in either order. *)

val maximum_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [maximum_s t s] is the element-wise maximum of [t] and scalar [s]. *)

val minimum : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [minimum a b] is the element-wise minimum of [a] and [b]. On floats it is
    the IEEE 754 minimum: NaN propagates, and [-0.] is less than [0.], so the
    minimum of [-0.] and [0.] is [-0.] in either order. *)

val minimum_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [minimum_s t s] is the element-wise minimum of [t] and scalar [s]. *)

val logical_and : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_and a b] is the element-wise logical AND: one of the dtype where
    both are non-zero, zero elsewhere. *)

val logical_or : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_or a b] is the element-wise logical OR: one where either is
    non-zero, zero elsewhere. *)

val logical_xor : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_xor a b] is the element-wise logical XOR: one where exactly one is
    non-zero, zero elsewhere. *)

val logical_not : ('a, 'b) t -> ('a, 'b) t
(** [logical_not t] is the element-wise logical NOT: non-zero becomes [0], zero
    becomes [1]. *)

val where : (bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [where cond if_true if_false] selects elements from [if_true] where [cond]
    is [true] and from [if_false] elsewhere. All three inputs broadcast to a
    common shape.

    {@ocaml[
      # let x =
          create float32 [| 4 |] [| -1.; 2.; -3.; 4. |]
        in
        where
          (greater x (scalar float32 0.))
          x (scalar float32 0.)
      - : (float, float32_elt) t = [0, 2, 0, 4]
    ]} *)

val clamp : ?min:'a -> ?max:'a -> ('a, 'b) t -> ('a, 'b) t
(** [clamp ?min ?max t] clamps elements to \[[min], [max]\]. Either bound may be
    omitted. *)

(** {1:bitwise Bitwise operations} *)

val bitwise_xor : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_xor a b] is the element-wise bitwise XOR. *)

val bitwise_or : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_or a b] is the element-wise bitwise OR. *)

val bitwise_and : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_and a b] is the element-wise bitwise AND. *)

val bitwise_not : ('a, 'b) t -> ('a, 'b) t
(** [bitwise_not t] is the element-wise bitwise NOT. *)

val lshift : ('a, 'b) t -> int -> ('a, 'b) t
(** [lshift t n] left-shifts each element by [n] bits.

    Raises [Invalid_argument] if [n] is negative or the dtype is not an integer
    type.

    {@ocaml[
      # create int32 [| 3 |] [| 1l; 2l; 3l |]
        |> Fun.flip lshift 2
      - : (int32, int32_elt) t = [4, 8, 12]
    ]}

    See also {!rshift}. *)

val rshift : ('a, 'b) t -> int -> ('a, 'b) t
(** [rshift t n] right-shifts each element by [n] bits, keeping the sign of a
    signed integer: [t / 2{^n}] rounded toward negative infinity.

    Raises [Invalid_argument] if [n] is negative or the dtype is not an integer
    type.

    See also {!lshift}. *)

(** {1:infix Infix operators} *)

module Infix : sig
  (** {2:infix_arith Element-wise arithmetic} *)

  val ( + ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a + b] is {!add} [a b]. *)

  val ( - ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a - b] is {!sub} [a b]. *)

  val ( ~- ) : ('a, 'b) t -> ('a, 'b) t
  (** [-t] is {!neg} [t]. *)

  val ( * ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a * b] is {!mul} [a b]. *)

  val ( / ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a / b] is {!div} [a b]. *)

  val ( ** ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a ** b] is {!pow} [a b]. *)

  (** {2:infix_scalar Scalar arithmetic} *)

  val ( +$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  (** [t +$ s] is {!add_s} [t s]. *)

  val ( -$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  (** [t -$ s] is {!sub_s} [t s]. *)

  val ( *$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  (** [t *$ s] is {!mul_s} [t s]. *)

  val ( /$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  (** [t /$ s] is {!div_s} [t s]. *)

  val ( **$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  (** [t **$ s] is {!pow_s} [t s]. *)

  (** {2:infix_cmp Comparisons} *)

  val ( < ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a < b] is {!less} [a b]. *)

  val ( <> ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a <> b] is {!not_equal} [a b]. *)

  val ( = ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a = b] is {!equal} [a b]. *)

  val ( > ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a > b] is {!greater} [a b]. *)

  val ( <= ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a <= b] is {!less_equal} [a b]. *)

  val ( >= ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a >= b] is {!greater_equal} [a b]. *)

  (** {2:infix_scalar_cmp Scalar comparisons} *)

  val ( =$ ) : ('a, 'b) t -> 'a -> (bool, bool_elt) t
  (** [t =$ s] is {!equal_s} [t s]. *)

  val ( <>$ ) : ('a, 'b) t -> 'a -> (bool, bool_elt) t
  (** [t <>$ s] is {!not_equal_s} [t s]. *)

  val ( <$ ) : ('a, 'b) t -> 'a -> (bool, bool_elt) t
  (** [t <$ s] is {!less_s} [t s]. *)

  val ( >$ ) : ('a, 'b) t -> 'a -> (bool, bool_elt) t
  (** [t >$ s] is {!greater_s} [t s]. *)

  val ( <=$ ) : ('a, 'b) t -> 'a -> (bool, bool_elt) t
  (** [t <=$ s] is {!less_equal_s} [t s]. *)

  val ( >=$ ) : ('a, 'b) t -> 'a -> (bool, bool_elt) t
  (** [t >=$ s] is {!greater_equal_s} [t s]. *)

  (** {2:infix_bitwise Bitwise} *)

  val ( lxor ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a lxor b] is {!bitwise_xor} [a b]. *)

  val ( lor ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a lor b] is {!bitwise_or} [a b]. *)

  val ( land ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a land b] is {!bitwise_and} [a b]. *)

  (** {2:infix_mod Modulo} *)

  val ( % ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a % b] is {!mod_} [a b]. *)

  val ( mod ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a mod b] is {!mod_} [a b]. *)

  val ( %$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  (** [t %$ s] is {!mod_s} [t s]. *)

  (** {2:infix_logic Logical} *)

  val ( && ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a && b] is {!logical_and} [a b]. *)

  val ( || ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a || b] is {!logical_or} [a b]. *)

  (** {2:infix_linalg Linear algebra} *)

  val ( *@ ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a *@ b] is {!matmul} [a b]. *)

  val ( /@ ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a /@ b] is {!solve} [a b]. *)

  val ( **@ ) : ('a, 'b) t -> int -> ('a, 'b) t
  (** [t **@ n] is {!matrix_power} [t n]. *)

  (** {2:infix_index Indexing} *)

  val ( .%{} ) : ('a, 'b) t -> int list -> ('a, 'b) t
  (** [t.%\{i\}] is {!get} [i t]. *)

  val ( .${} ) : ('a, 'b) t -> index list -> ('a, 'b) t
  (** [t.$\{s\}] is {!val-slice} [s t]. *)
end

(** {1:reduction Reductions} *)

val sum : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [sum ?axes ?keepdims t] sums elements along [axes]. When [axes] is omitted,
    reduces all axes (returns a scalar). When [keepdims] is [true], reduced axes
    are kept with size 1. [keepdims] defaults to [false]. Negative axes count
    from the end.

    A float sum is [0.] plus its terms in an unspecified association, so a sum
    that is exactly zero is [0.], never [-0.], and a sum of nothing is [0.]. It
    is deterministic for a given input layout on a given machine and does not
    depend on the thread count; the same values in another layout can differ in
    rounding, and at overflow in whether a term overflows. {!mean} and the
    contraction of {!matmul} and the products built on it sum the same way.
    Today each output of a product of vectors, or of a single row or column,
    also has the bits of {!inner} of its row and column, which this contract
    does not promise.

    {@ocaml[
      # create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
        |> sum |> item []
      - : float = 10.
      # create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
        |> sum ~axes:[ 0 ]
      - : (float, float32_elt) t = [4, 6]
      # create float32 [| 1; 2 |] [| 1.; 2. |]
        |> sum ~axes:[ 1 ] ~keepdims:true
      - : (float, float32_elt) t = float32 [1,1] [[3]]
    ]} *)

val max : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [max ?axes ?keepdims t] is the maximum along [axes], as {!maximum} orders
    elements: NaN propagates and [-0.] is less than [0.]. The result does not
    depend on the order the elements are combined in. [keepdims] defaults to
    [false].

    {@ocaml[
      # create float32 [| 2; 3 |]
          [| 1.; 2.; 3.; 4.; 5.; 6. |]
        |> max |> item []
      - : float = 6.
    ]} *)

val min : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [min ?axes ?keepdims t] is the minimum along [axes], as {!minimum} orders
    elements: NaN propagates and [-0.] is less than [0.]. The result does not
    depend on the order the elements are combined in. [keepdims] defaults to
    [false]. *)

val prod : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [prod ?axes ?keepdims t] is the product along [axes]. [keepdims] defaults to
    [false].

    {@ocaml[
      # create int32 [| 3 |] [| 2l; 3l; 4l |]
        |> prod |> item []
      - : int32 = 24l
    ]} *)

val cumsum : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [cumsum ?axis t] is the inclusive cumulative sum along [axis]. Each running
    sum is a float sum as {!sum} describes, [0.] plus the terms so far: the
    first element of [cumsum] of [[-0.]] is [0.]. When [axis] is omitted, it
    accumulates the flattened tensor and keeps [t]'s shape.

    See also {!cumprod}. *)

val cumprod : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [cumprod ?axis t] is the inclusive cumulative product along [axis]. When
    [axis] is omitted, it accumulates the flattened tensor and keeps [t]'s
    shape.

    See also {!cumsum}. *)

val cummax : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [cummax ?axis t] is the inclusive cumulative maximum along [axis], as
    {!maximum} orders elements: NaN propagates and [-0.] is less than [0.]. When
    [axis] is omitted, it accumulates the flattened tensor and keeps [t]'s
    shape.

    See also {!cummin}. *)

val cummin : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [cummin ?axis t] is the inclusive cumulative minimum along [axis], as
    {!minimum} orders elements: NaN propagates and [-0.] is less than [0.]. When
    [axis] is omitted, it accumulates the flattened tensor and keeps [t]'s
    shape.

    See also {!cummax}. *)

val mean : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [mean ?axes ?keepdims t] is the arithmetic mean along [axes]. NaN
    propagates. [keepdims] defaults to [false].

    {@ocaml[
      # create float32 [| 4 |] [| 1.; 2.; 3.; 4. |]
        |> mean |> item []
      - : float = 2.5
    ]} *)

val var :
  ?axes:int list -> ?keepdims:bool -> ?ddof:int -> ('a, 'b) t -> ('a, 'b) t
(** [var ?axes ?keepdims ?ddof t] is the variance along [axes]. [ddof] (delta
    degrees of freedom) defaults to [0] (population variance); use [1] for
    sample variance. Computed as [E[(X - E[X])²] / (N - ddof)]. [keepdims]
    defaults to [false].

    Raises [Invalid_argument] if [ddof >= N].

    {@ocaml[
      # create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
        |> var |> item []
      - : float = 2.
      # create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
        |> var ~ddof:1 |> item []
      - : float = 2.5
    ]}

    See also {!std}. *)

val std :
  ?axes:int list -> ?keepdims:bool -> ?ddof:int -> ('a, 'b) t -> ('a, 'b) t
(** [std ?axes ?keepdims ?ddof t] is the standard deviation:
    [sqrt({!var} ~ddof t)]. [ddof] defaults to [0]. [keepdims] defaults to
    [false].

    See also {!var}. *)

val all : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> (bool, bool_elt) t
(** [all ?axes ?keepdims t] is [true] iff every element along [axes] is
    non-zero. [keepdims] defaults to [false].

    {@ocaml[
      # create int32 [| 3 |] [| 1l; 2l; 3l |]
        |> all |> item []
      - : bool = true
      # create int32 [| 3 |] [| 1l; 0l; 3l |]
        |> all |> item []
      - : bool = false
    ]}

    See also {!any}. *)

val any : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> (bool, bool_elt) t
(** [any ?axes ?keepdims t] is [true] iff at least one element along [axes] is
    non-zero. [keepdims] defaults to [false].

    See also {!all}. *)

val argmax : ?axis:int -> ?keepdims:bool -> ('a, 'b) t -> int64_t
(** [argmax ?axis ?keepdims t] is the index of the maximum along [axis]: the
    first index holding the element {!max} returns, so [-0.] and [0.] do not
    tie and the argmax of [[-0.; 0.]] is [1]. A NaN counts as the maximum: the
    result is the index of the first NaN. When [axis] is omitted, operates on
    the flattened tensor. [keepdims] defaults to [false].

    Raises [Invalid_argument] if [axis] is out of bounds, or if the reduced
    axis, all of [t] when [axis] is omitted, has no element.

    {@ocaml[
      # create int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |]
        |> argmax |> item []
      - : int64 = 4L
    ]}

    See also {!argmin}. *)

val argmin : ?axis:int -> ?keepdims:bool -> ('a, 'b) t -> int64_t
(** [argmin ?axis ?keepdims t] is the index of the minimum along [axis]: the
    first index holding the element {!min} returns, so the argmin of
    [[0.; -0.]] is [1]. A NaN counts as the minimum: the result is the index of
    the first NaN. When [axis] is omitted, operates on the flattened tensor.
    [keepdims] defaults to [false].

    Raises [Invalid_argument] as {!argmax} does.

    See also {!argmax}. *)

(** {1:sorting Sorting and searching} *)

val sort : ?descending:bool -> ?axis:int -> ('a, 'b) t -> ('a, 'b) t * int64_t
(** [sort ?descending ?axis t] sorts elements along [axis] and returns
    [(sorted, indices)] where [indices] maps sorted positions back to originals.
    [descending] defaults to [false]. [axis] defaults to [-1] (last).

    [sorted] is [take_along_axis ~axis ~indices t], bit for bit. The sort is
    stable: equal elements keep their input order, NaNs included. [-0.] sorts
    before [0.], and NaN sorts to the end in either direction.

    Raises [Invalid_argument] if [axis] is out of bounds.

    {@ocaml[
      # create int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |]
        |> sort
      - : (int32, int32_elt) t * int64_t =
      (int32 [5] [1, 1, ..., 4, 5], int64 [5] [1, 3, ..., 2, 4])
    ]}

    See also {!argsort}. *)

val argsort : ?descending:bool -> ?axis:int -> ('a, 'b) t -> int64_t
(** [argsort ?descending ?axis t] is [snd (sort ?descending ?axis t)].

    See also {!sort}. *)

val top_k : k:int -> ?axis:int -> ('a, 'b) t -> ('a, 'b) t * int64_t
(** [top_k ~k ?axis t] is [(values, indices)]: the [k] greatest entries along
    [axis], greatest first, and their positions. Both have [t]'s shape with
    [axis] of extent [k]. [axis] defaults to [-1] (last).

    It is the first [k] entries of [sort ~descending:true ?axis t]: equal
    entries come lowest position first, and NaN comes after every number.
    [values] is [take_along_axis ~axis ~indices t], so it differentiates with
    respect to [t].

    Up to [k = 8] the cost is [k] passes over [axis], each after the one before
    it, which suits a router choosing a few of many. A greater [k] sorts [axis]
    when it has at most 2048 entries. A longer [axis] is not sorted: the [k]th
    greatest entry is found by radix select, one pass over [axis] for every four
    bits of the dtype (eight for [float32]; every two bits past [2{^20}] entries
    over all rows), and only the [k] entries kept are put in order, by [k * k]
    comparisons, or by a sort of them once those number more than [2{^24}] over
    all rows. Eagerly a call holds about 30 bytes per entry at once, mostly the
    running counts that place the kept entries: 64 rows of 131072 [float32] hold
    about 250 MB, where a sort of them holds about 100 MB, its values and their
    [int64] positions.

    Raises [Invalid_argument] if [t] has no dimension, [axis] is out of bounds,
    [k] is outside \[[1], extent of [axis]\], or [t] is complex.

    {@ocaml[
      # let values, indices =
          create float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] |> top_k ~k:2
        in
        (to_array values, to_array indices)
      - : float array * int64 array = ([|5.; 4.|], [|4L; 2L|])
    ]}

    See also {!sort}, {!argmax}. *)

(** {1:linalg Linear algebra} *)

exception
  Linalg_error of {
    op : string;
    kind : [ `Not_positive_definite | `Singular | `No_convergence ];
  }
(** Raised by a linear-algebra operation when the numeric computation fails,
    carrying the operation name and a failure [kind]. Precondition violations
    (non-square input, wrong dtype) raise [Invalid_argument] instead.

    This is [Nx_backend.Linalg_error]; catching it here catches the exception
    raised by the backend. *)

(** {2:linalg_products Products} *)

val dot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [dot a b] is the generalised dot product.

    Contracts the last axis of [a] with:
    - the only axis of [b] when [b] is 1-D,
    - the second-to-last axis of [b] otherwise.

    Dimension rules:
    - 1-D × 1-D → scalar (inner product).
    - 2-D × 2-D → matrix multiplication.
    - N-D × M-D → contraction; output axes are the non-contracted axes of [a]
      followed by those of [b].

    {b Note.} Unlike {!matmul}, [dot] does {e not} broadcast batch dimensions—it
    concatenates them. It computes at {!matmul}'s precision.

    Raises [Invalid_argument] if contraction axes differ in size or either input
    is 0-D.

    {@ocaml[
      # let a = create float32 [| 2 |] [| 1.; 2. |] in
        let b = create float32 [| 2 |] [| 3.; 4. |] in
        dot a b |> item []
      - : float = 11.
      # dot (ones float32 [| 3; 4; 5 |])
            (ones float32 [| 5; 6 |]) |> shape
      - : int array = [|3; 4; 6|]
    ]}

    See also {!matmul}, {!vdot}, {!vecdot}. *)

val matmul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [matmul a b] is the matrix product of [a] and [b] with batch broadcasting.

    Dimension rules:
    - 1-D × 1-D → scalar (inner product).
    - 1-D × N-D → [a] is treated as a row vector.
    - N-D × 1-D → [b] is treated as a column vector.
    - N-D × M-D → matrix multiply on last two axes; leading axes are broadcast.

    At [float16], [bfloat16] and the float8 dtypes, the operands are widened to
    [float32], multiplied and summed at [float32], and each element of the
    result is rounded once to the operands' dtype. The contraction sums as
    {!sum} describes: an output whose products sum to exactly zero is [0.], and
    an empty contraction gives [0.].

    Raises [Invalid_argument] if inputs are 0-D or inner dimensions mismatch.

    {@ocaml[
      # let a =
          create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
        in
        let b = create float32 [| 2 |] [| 5.; 6. |] in
        matmul a b
      - : (float, float32_elt) t = [17, 39]
      # matmul (ones float32 [| 1; 3; 4 |])
               (ones float32 [| 5; 4; 2 |]) |> shape
      - : int array = [|5; 3; 2|]
    ]}

    See also {!dot}, {!multi_dot}. *)

val diagonal :
  ?offset:int -> ?axis1:int -> ?axis2:int -> ('a, 'b) t -> ('a, 'b) t
(** [diagonal ?offset ?axis1 ?axis2 t] extracts diagonals from 2-D planes
    defined by [axis1] and [axis2]. [offset] defaults to [0]. [axis1] and
    [axis2] default to the last two axes.

    Raises [Invalid_argument] if [axis1 = axis2] or either is out of bounds.

    See also {!diag}, {!trace}. *)

val matrix_transpose : ('a, 'b) t -> ('a, 'b) t
(** [matrix_transpose t] swaps the last two axes: [[…; m; n]] → [[…; n; m]]. For
    1-D tensors, returns [t] unchanged.

    See also {!transpose}. *)

val vdot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [vdot a b] is the dot product of two vectors. Both inputs are flattened; for
    complex dtypes, [a] is conjugated first. Always returns a scalar. It
    computes at {!matmul}'s precision.

    Raises [Invalid_argument] if the inputs have different numbers of elements.

    See also {!dot}, {!vecdot}. *)

val vecdot : ?axis:int -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [vecdot ?axis a b] is the dot product of [a] and [b] along [axis] with
    broadcasting. [axis] defaults to [-1]. It computes at {!matmul}'s precision.

    Raises [Invalid_argument] if the specified axis dimensions differ.

    See also {!vdot}, {!dot}. *)

val inner : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [inner a b] is the inner product over the last axes of [a] and [b]. It
    computes at {!matmul}'s precision.

    Raises [Invalid_argument] if the last dimensions differ.

    See also {!dot}, {!outer}. *)

val outer : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [outer a b] is the outer product. Inputs are flattened to 1-D; the result
    has shape [[numel a; numel b]].

    See also {!inner}. *)

val tensordot :
  ?axes:int list * int list -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [tensordot ?axes a b] contracts [a] and [b] along the specified axis pairs.
    [axes] defaults to contracting the last axis of [a] with the first axis of
    [b].

    Raises [Invalid_argument] if the contracted axes have different sizes. *)

val einsum : string -> ('a, 'b) t array -> ('a, 'b) t
(** [einsum subscripts operands] evaluates Einstein summation.

    {@ocaml[
      # let a =
          create float32 [| 2; 3 |]
            [| 1.; 2.; 3.; 4.; 5.; 6. |]
        in
        let b =
          create float32 [| 3; 2 |]
            [| 1.; 2.; 3.; 4.; 5.; 6. |]
        in
        einsum "ij,jk->ik" [| a; b |] |> shape
      - : int array = [|2; 2|]
    ]}

    See also {!matmul}, {!tensordot}. *)

val kron : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [kron a b] is the Kronecker product. The result has shape
    [[a.shape.(i) * b.shape.(i)]] for each [i]. *)

val multi_dot : ('a, 'b) t array -> ('a, 'b) t
(** [multi_dot ts] is the chained matrix product of [ts], automatically choosing
    the association order that minimises computation.

    Raises [Invalid_argument] if the array is empty, shapes are incompatible, or
    dtypes are not floating-point or complex.

    See also {!matmul}. *)

val matrix_power : ('a, 'b) t -> int -> ('a, 'b) t
(** [matrix_power t n] raises square matrix [t] to integer power [n]. [n = 0]
    returns the identity; [n < 0] uses the inverse.

    Raises {!Linalg_error} with kind [`Singular] if [n < 0] and [t] is singular.
    Raises [Invalid_argument] if [t] is not square or the dtype is not
    floating-point or complex. *)

val cross : ?axis:int -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [cross ?axis a b] is the cross product of 3-element vectors along [axis].
    [axis] defaults to [-1].

    Raises [Invalid_argument] if the axis dimension is not 3. *)

(** {2:linalg_decomp Decompositions} *)

val cholesky : ?upper:bool -> ('a, 'b) t -> ('a, 'b) t
(** [cholesky ?upper a] is the Cholesky factor of positive-definite matrix [a],
    real symmetric or complex Hermitian. When [upper] is [true], returns the
    upper-triangular factor [U] such that [a = Uᴴ U]; otherwise (default)
    returns the lower-triangular factor [L] such that [a = L Lᴴ], where [ᴴ] is
    the conjugate transpose, the transpose on real matrices. Whichever factor
    is returned, only the lower triangle of [a] and the real part of its
    diagonal are read; the strictly upper triangle and the diagonal's imaginary
    parts may hold anything.

    Raises {!Linalg_error} with kind [`Not_positive_definite] if [a] is not
    positive-definite. Raises [Invalid_argument] if [a] is not square or the
    dtype is not floating-point or complex.

    See also {!solve}. *)

val qr : ?mode:[ `Complete | `Reduced ] -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
(** [qr ?mode a] is [(Q, R)] where [a = Q R], the columns of [Q] are orthonormal
    ([Q] is unitary on complex matrices), and [R] is upper-triangular. [mode]
    defaults to [`Reduced].

    Raises {!Linalg_error} with kind [`No_convergence] if the factorization does
    not converge. Raises [Invalid_argument] if the dtype is not floating-point
    or complex.

    See also {!svd}, {!lu}. *)

val lu : ('a, 'b) t -> int64_t * ('a, 'b) t * ('a, 'b) t
(** [lu a] is [(perm, l, u)] where row [i] of [l *@ u] is row [perm.(i)] of [a],
    [l] is lower-triangular with ones on its diagonal, and [u] is
    upper-triangular. For [a] of shape [·.., m, n] and [k = min m n], [perm] is
    [·.., m], [l] is [·.., m, k] and [u] is [·.., k, n].

    The factorization pivots partially: at each column the row of largest
    magnitude at or below the diagonal ([|re| + |im|] for complex) moves up to
    the diagonal, the first such row on a tie, so for a real [a] every entry of
    [L] has magnitude at most 1. A singular [a] is not an error: its zero pivots
    stay on the diagonal of [U].

    Raises [Invalid_argument] if [a] has fewer than two dimensions or the dtype
    is not floating-point or complex.

    See also {!det}, {!solve}. *)

val svd :
  ?full_matrices:bool ->
  ('a, 'b) t ->
  ('a, 'b) t * (float, float64_elt) t * ('a, 'b) t
(** [svd ?full_matrices a] is [(U, S, Vh)] where [a = U diag(S) Vh]. [S]
    contains the singular values in descending order, each non-negative: a zero
    one is [+0], whatever the signs of [a]'s zeros. [full_matrices] defaults to
    [false] (economy decomposition).

    Raises [Invalid_argument] if the dtype is not floating-point or complex.

    See also {!svdvals}, {!qr}. *)

val svdvals : ('a, 'b) t -> (float, float64_elt) t
(** [svdvals a] is the singular values of [a] in descending order, as {!svd}
    gives them. More efficient than {!svd} when only the values are needed.

    Raises [Invalid_argument] if the dtype is not floating-point or complex. *)

(** {2:linalg_eig Eigenvalues and eigenvectors} *)

val eig :
  ('a, 'b) t -> (Complex.t, complex64_elt) t * (Complex.t, complex64_elt) t
(** [eig a] is [(eigenvalues, eigenvectors)] of general square matrix [a].
    Results are complex since real matrices may have complex eigenvalues.

    Raises [Invalid_argument] if [a] is not square or the dtype is not
    floating-point or complex.

    See also {!eigh}, {!eigvals}. *)

val eigh :
  ?uplo:[ `U | `L ] -> ('a, 'b) t -> (float, float64_elt) t * ('a, 'b) t
(** [eigh ?uplo a] is [(w, v)], the eigenvalues and eigenvectors of the real
    symmetric or complex Hermitian matrix [a]: [w] holds the eigenvalues, real,
    in ascending order, and the columns of [v], of [a]'s dtype, are orthonormal
    eigenvectors, [a v = v diag(w)]. Only the triangle [uplo] names, the lower
    one by default ([`L]) or the upper one ([`U]), and the real part of the
    diagonal are read; the other triangle and the diagonal's imaginary parts may
    hold anything. More efficient than {!eig} for symmetric matrices.

    Raises [Invalid_argument] if [a] is not square or the dtype is not
    floating-point or complex.

    See also {!eig}, {!eigvalsh}. *)

val eigvals : ('a, 'b) t -> (Complex.t, complex64_elt) t
(** [eigvals a] is the eigenvalues of general square matrix [a]. More efficient
    than {!eig} when eigenvectors are not needed.

    Raises [Invalid_argument] if [a] is not square or the dtype is not
    floating-point or complex.

    See also {!eig}, {!eigvalsh}. *)

val eigvalsh : ?uplo:[ `U | `L ] -> ('a, 'b) t -> (float, float64_elt) t
(** [eigvalsh ?uplo a] is the eigenvalues of the real symmetric or complex
    Hermitian matrix [a], real, in ascending order. Only the triangle [uplo]
    names and the real part of the diagonal are read, as in {!eigh}.

    Raises [Invalid_argument] if [a] is not square or the dtype is not
    floating-point or complex.

    See also {!eigh}, {!eigvals}. *)

(** {2:linalg_norms Norms and invariants} *)

val norm :
  ?ord:
    [ `Fro
    | `Nuc
    | `One
    | `Two
    | `Inf
    | `NegOne
    | `NegTwo
    | `NegInf
    | `P of float ] ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [norm ?ord ?axes ?keepdims t] is the matrix or vector norm. [ord] defaults
    to Frobenius for matrices, 2-norm for vectors. [keepdims] defaults to
    [false].

    - [`Fro] — Frobenius norm.
    - [`Nuc] — nuclear norm (sum of singular values).
    - [`One] — max absolute column sum (matrix) or 1-norm (vector).
    - [`Two] — largest singular value (matrix) or 2-norm (vector).
    - [`Inf] — max absolute row sum (matrix) or ∞-norm (vector).
    - [`P p] — p-norm (vectors only).
    - [`NegOne], [`NegTwo], [`NegInf] — corresponding minimum norms.

    Raises [Invalid_argument] if [ord] requires a floating-point or complex
    dtype. *)

val cond :
  ?p:[ `One | `Two | `Inf | `NegOne | `NegTwo | `NegInf | `Fro ] ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [cond ?p a] is the condition number of [a] in the [p]-norm. [p] defaults to
    [`Two].

    Raises [Invalid_argument] if the dtype is not floating-point or complex. *)

val det : ('a, 'b) t -> ('a, 'b) t
(** [det a] is the determinant of square matrix [a], in [a]'s dtype: the product
    of the diagonal of {!lu}'s [U], negated once per row interchange. A matrix
    of size 0 has determinant 1.

    Raises [Invalid_argument] if [a] is not square or the dtype is not
    floating-point or complex.

    See also {!slogdet}. *)

val slogdet : ('a, 'b) t -> ('a, 'b) t * (float, float64_elt) t
(** [slogdet a] is [(sign, log_abs_det)] with [det a = sign * exp(log_abs_det)].
    [sign] is in [a]'s dtype: [1] or [-1] for a real matrix, a complex number of
    modulus 1 for a complex one, and [0] for a singular matrix, whose
    [log_abs_det] is [neg_infinity]. [log_abs_det] is float64, summed from the
    logarithms of {!lu}'s pivots, so it holds determinants far beyond the range
    of {!det}.

    Raises [Invalid_argument] if [a] is not square or the dtype is not
    floating-point or complex. *)

val matrix_rank :
  ?tol:float -> ?rtol:float -> ?hermitian:bool -> ('a, 'b) t -> int
(** [matrix_rank ?tol ?rtol ?hermitian a] is the rank of [a], counting singular
    values above the tolerance. [rtol] defaults to [max(M, N) * ε * σ_max]. When
    [hermitian] is [true] (default [false]), uses a more efficient
    eigenvalue-based algorithm.

    Raises [Invalid_argument] if the dtype is not floating-point or complex. *)

val trace : ?offset:int -> ('a, 'b) t -> ('a, 'b) t
(** [trace ?offset t] is the sum along the [offset]-th diagonal. [offset]
    defaults to [0].

    Raises [Invalid_argument] if [t] has fewer than 2 dimensions.

    See also {!diagonal}. *)

(** {2:linalg_solve Solving} *)

val solve_triangular :
  ?upper:bool ->
  ?transpose:bool ->
  ?unit_diag:bool ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [solve_triangular ?upper ?transpose ?unit_diag a b] solves the triangular
    system [a *@ x = b] for [x], exploiting that [a] is triangular instead of
    factoring.

    [a] is upper-triangular when [upper] is [true] (default [false]); its
    diagonal is assumed to be all ones — and is not read — when [unit_diag] is
    [true] (default [false]). When [transpose] is [true], the system solved is
    [transpose a *@ x = b] (the conjugate transpose for complex [a]). Only the
    triangle [upper] names is read: [a] is not checked for triangularity, and
    the rest is silently ignored. Passing a pre-factorized or otherwise
    pre-triangularized [a] skips the factorization cost of {!solve}.

    [b] is one right-hand-side vector of shape [·.., n] or [nrhs] right-hand
    sides stacked as [·.., n, nrhs] (with [n] the size of [a]), sharing the
    batch dimensions of [a]; the result has the shape of [b].

    Raises [Invalid_argument] if the dtype is not floating-point or complex, [a]
    is not square, [a] and [b] differ in dtype, or [b]'s shape does not match
    [a]'s. Raises {!Linalg_error} with kind [`Singular] if a diagonal entry of
    [a] is zero and [unit_diag] is [false].

    See also {!solve}. *)

val solve : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [solve a b] is [x] such that [a *@ x = b].

    [x] comes from {!lu}'s factors of [a] and two triangular solves.

    Raises {!Linalg_error} with kind [`Singular] if [a] is singular: a pivot of
    its LU factorization lies below tolerance. Raises [Invalid_argument] if [a]
    is not square or the dtype is not floating-point or complex.

    See also {!solve_triangular}, {!lstsq}, {!inv}. *)

val lstsq :
  ?rcond:float ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t * ('a, 'b) t * int * (float, float64_elt) t
(** [lstsq ?rcond a b] is [(x, residuals, rank, sv)] — the least-squares
    solution to [a *@ x ≈ b]. [rcond] defaults to machine precision.

    Raises [Invalid_argument] if the dtype is not floating-point or complex.

    See also {!solve}. *)

val inv : ('a, 'b) t -> ('a, 'b) t
(** [inv a] is the inverse of square matrix [a].

    Raises {!Linalg_error} with kind [`Singular] if [a] is singular. Raises
    [Invalid_argument] if [a] is not square or the dtype is not floating-point
    or complex.

    See also {!pinv}, {!solve}. *)

val pinv : ?rtol:float -> ?hermitian:bool -> ('a, 'b) t -> ('a, 'b) t
(** [pinv ?rtol ?hermitian a] is the Moore–Penrose pseudoinverse of [a]. Handles
    non-square and singular matrices. [hermitian] defaults to [false].

    Raises [Invalid_argument] if the dtype is not floating-point or complex.

    See also {!inv}. *)

val tensorsolve : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [tensorsolve ?axes a b] solves the tensor equation [tensordot a x axes = b]
    for [x].

    Raises [Invalid_argument] if shapes are incompatible or the dtype is not
    floating-point or complex. *)

val tensorinv : ?ind:int -> ('a, 'b) t -> ('a, 'b) t
(** [tensorinv ?ind a] is the tensor inverse such that
    [tensordot a (tensorinv a) ind] is the identity. [ind] defaults to [2].

    Raises [Invalid_argument] if the result is not square in the specified
    dimensions or the dtype is not floating-point or complex. *)

(** {1:fft Fourier transforms}

    Every transform raises [Invalid_argument] if an axis is out of range or if
    [s] and [axes] differ in length. *)

type fft_norm = [ `Backward | `Forward | `Ortho ]
(** FFT normalisation mode.
    - [`Backward] — normalise by [1/n] on the inverse (default).
    - [`Forward] — normalise by [1/n] on the forward.
    - [`Ortho] — normalise by [1/√n] on both. *)

val fft :
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [fft ?axis ?n ?norm x] is the 1-D discrete Fourier transform along [axis].
    [axis] defaults to [-1]. [n] truncates or zero-pads the input. [norm]
    defaults to [`Backward]. The transform of no points ([n = 0], or an empty
    axis) is empty.

    See also {!ifft}, {!rfft}. *)

val ifft :
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [ifft ?axis ?n ?norm x] is the inverse of {!fft}.

    See also {!fft}, {!irfft}. *)

val fft2 :
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [fft2 ?axes ?s ?norm x] is the 2-D FFT. [axes] defaults to the last two.

    Raises [Invalid_argument] if the input has fewer than 2 dimensions or if
    [axes] does not name exactly two axes. So do {!ifft2}, {!rfft2} and
    {!irfft2}.

    See also {!ifft2}, {!fft}. *)

val ifft2 :
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [ifft2 ?axes ?s ?norm x] is the inverse of {!fft2}. *)

val fftn :
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [fftn ?axes ?s ?norm x] is the N-D FFT. [axes] defaults to all.

    See also {!ifftn}. *)

val ifftn :
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [ifftn ?axes ?s ?norm x] is the inverse of {!fftn}. *)

val rfft :
  (Complex.t, 'b) dtype ->
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (Complex.t, 'b) t
(** [rfft dtype ?axis ?n ?norm x] is the 1-D FFT of real input, stored as
    [dtype]. Returns only the non-redundant positive frequencies; the output
    size along the transformed axis is [n/2 + 1]. For no points that is one bin,
    the empty sum, zero under every [norm].

    [dtype] selects the storage precision of the result, independent of the
    input's: the natural pairings are [float32] with [complex64] and [float64]
    with [complex128], but any pairing is accepted and a narrowing one rounds
    the spectrum. The result is accurate to at least [dtype]'s precision.

    The normalisation implied by [norm] is applied after the result has been
    stored as [dtype], so a spectrum whose unnormalised magnitudes overflow
    [dtype] stays infinite. Transform at the wider dtype and {!cast} afterwards
    if the input's dynamic range approaches that limit.

    {@ocaml[
      # create float64 [| 4 |] [| 0.; 1.; 2.; 3. |]
        |> rfft complex128 |> shape
      - : int array = [|3|]
    ]}

    See also {!irfft}, {!fft}. *)

val irfft :
  (float, 'b) dtype ->
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (float, 'b) t
(** [irfft dtype ?axis ?n ?norm x] is the inverse of {!rfft}, producing real
    output stored as [dtype]. Assumes Hermitian symmetry: it reads only the real
    part of the DC bin and, for an even [n], of the Nyquist bin. [n] defaults to
    [2 * (m - 1)] for [m] bins along [axis], and the bins are cropped or
    zero-padded to the [n/2 + 1] it reads: one bin inverts, by default, the
    {!rfft} of no points and gives an empty result, and [~n:1] gives its
    constant signal. As with {!rfft}, [dtype] selects the storage precision
    independent of the input's.

    See also {!rfft}. *)

val rfft2 :
  (Complex.t, 'b) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (Complex.t, 'b) t
(** [rfft2 dtype ?axes ?s ?norm x] is the 2-D FFT of real input, stored as
    [dtype]. [axes] defaults to the last two. As with {!rfftn}, the intermediate
    spectrum is carried at [dtype] between the two axis passes.

    See also {!irfft2}, {!rfft}. *)

val irfft2 :
  (float, 'b) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (float, 'b) t
(** [irfft2 dtype ?axes ?s ?norm x] is the inverse of {!rfft2}. *)

val rfftn :
  (Complex.t, 'b) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (Complex.t, 'b) t
(** [rfftn dtype ?axes ?s ?norm x] is the N-D FFT of real input, stored as
    [dtype]. [axes] defaults to all.

    The real-to-complex pass runs along the last of [axes] and the remaining
    axes are then transformed in place, so with more than one axis the
    intermediate spectrum is carried at [dtype] rather than at a wider working
    precision. A narrow [dtype] therefore rounds once per axis. Pass
    [complex128] and {!cast} afterwards to keep the intermediates wide.

    See also {!irfftn}, {!rfft}. *)

val irfftn :
  (float, 'b) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (float, 'b) t
(** [irfftn dtype ?axes ?s ?norm x] is the inverse of {!rfftn}: {!ifftn} along
    every axis of [axes] but the last, then {!irfft} along the last. [axes]
    defaults to all. [s] defaults to the lengths of the leading axes and
    [2 * (m - 1)] for the last. *)

val hfft :
  (float, 'b) dtype ->
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (float, 'b) t
(** [hfft dtype ?axis ?n ?norm x] is the FFT of the length-[n] signal with
    Hermitian symmetry whose first [n/2 + 1] samples are [x], as {!irfft} reads
    them, producing real output stored as [dtype]. [n] defaults to [2 * (m - 1)]
    for [m] samples along [axis]. *)

val ihfft :
  (Complex.t, 'b) dtype ->
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (Complex.t, 'b) t
(** [ihfft dtype ?axis ?n ?norm x] is the inverse of {!hfft}, producing the
    [n/2 + 1] complex samples stored as [dtype]. [n] defaults to the length of
    [axis]. *)

val dct :
  ?type_:int -> ?axis:int -> ?norm:fft_norm -> (float, 'a) t -> (float, 'a) t
(** [dct ?type_ ?axis ?norm x] is the discrete cosine transform along [axis].
    [type_] selects DCT type I, II, III, or IV and defaults to II. [axis]
    defaults to [-1]; [norm] defaults to [`Backward]. The result has the same
    shape and dtype as [x].

    [`Forward] scales by the reciprocal of the logical transform length,
    [`Backward] leaves the forward transform unscaled, and [`Ortho] produces an
    orthonormal transform. The logical length is [2 * (n - 1)] for type I and
    [2 * n] otherwise.

    Raises [Invalid_argument] unless [x] is float32 or float64, if [type_] is
    outside [[1; 4]], if [axis] is invalid, if the transformed axis is empty, or
    if type I is requested for an axis of length one.

    See also {!idct}, {!dctn}, {!dst}. *)

val idct :
  ?type_:int -> ?axis:int -> ?norm:fft_norm -> (float, 'a) t -> (float, 'a) t
(** [idct ?type_ ?axis ?norm x] is the inverse discrete cosine transform. Types
    I and IV are self-inverse up to normalisation; types II and III are inverse
    pairs. Calling [idct] with the same [type_] and [norm] as {!dct} recovers
    the input up to floating-point error.

    See also {!dct}, {!idctn}. *)

val dctn :
  ?type_:int ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (float, 'a) t
(** [dctn ?type_ ?axes ?norm x] applies {!dct} successively along [axes]. [axes]
    defaults to all axes. Negative axes are accepted; repeated axes are
    rejected. An empty [axes] list returns [x] unchanged.

    See also {!idctn}, {!dct}. *)

val idctn :
  ?type_:int ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (float, 'a) t
(** [idctn ?type_ ?axes ?norm x] is the inverse of {!dctn}. *)

val dst :
  ?type_:int -> ?axis:int -> ?norm:fft_norm -> (float, 'a) t -> (float, 'a) t
(** [dst ?type_ ?axis ?norm x] is the discrete sine transform along [axis].
    [type_] selects DST type I, II, III, or IV and defaults to II. [axis]
    defaults to [-1]; [norm] defaults to [`Backward]. The result has the same
    shape and dtype as [x].

    The normalisation modes have the same meaning as for {!dct}. The logical
    transform length is [2 * (n + 1)] for type I and [2 * n] otherwise.

    Raises [Invalid_argument] unless [x] is float32 or float64, if [type_] is
    outside [[1; 4]], if [axis] is invalid, or if the transformed axis is empty.

    See also {!idst}, {!dstn}, {!dct}. *)

val idst :
  ?type_:int -> ?axis:int -> ?norm:fft_norm -> (float, 'a) t -> (float, 'a) t
(** [idst ?type_ ?axis ?norm x] is the inverse discrete sine transform. Types I
    and IV are self-inverse up to normalisation; types II and III are inverse
    pairs. Calling [idst] with the same [type_] and [norm] as {!dst} recovers
    the input up to floating-point error.

    See also {!dst}, {!idstn}. *)

val dstn :
  ?type_:int ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (float, 'a) t
(** [dstn ?type_ ?axes ?norm x] applies {!dst} successively along [axes]. [axes]
    defaults to all axes. Negative axes are accepted; repeated axes are
    rejected. An empty [axes] list returns [x] unchanged.

    See also {!idstn}, {!dst}. *)

val idstn :
  ?type_:int ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (float, 'a) t
(** [idstn ?type_ ?axes ?norm x] is the inverse of {!dstn}. *)

val fftfreq : (float, 'a) dtype -> ?d:float -> int -> (float, 'a) t
(** [fftfreq dtype ?d n] is the DFT sample frequencies for window length [n] and
    sample spacing [d] (default [1.0]), as a tensor of dtype [dtype]. Pair it
    with the dtype the spectrum was transformed at so the frequency axis and the
    magnitudes combine without a {!cast}.

    {@ocaml[
      # fftfreq float64 4
      - : (float, float64_elt) t = [0, 0.25, -0.5, -0.25]
    ]}

    See also {!rfftfreq}. *)

val rfftfreq : (float, 'a) dtype -> ?d:float -> int -> (float, 'a) t
(** [rfftfreq dtype ?d n] is the positive DFT sample frequencies:
    [[0, 1, …, n/2] / (d * n)], as a tensor of dtype [dtype]. This is the
    frequency axis matching {!rfft}'s output.

    See also {!fftfreq}. *)

val fftshift : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [fftshift ?axes t] shifts the zero-frequency component to the centre. [axes]
    defaults to all.

    {@ocaml[
      # fftfreq float64 5 |> fftshift
      - : (float, float64_elt) t = float64 [5] [-0.4, -0.2, ..., 0.2, 0.4]
    ]}

    See also {!ifftshift}. *)

val ifftshift : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [ifftshift ?axes t] is the inverse of {!fftshift}. *)

(** {2:shorttime Short-time analysis} *)

val hann : (float, 'a) dtype -> int -> (float, 'a) t
(** [hann dtype n] is the length-[n] Hann taper
    [0.5 - 0.5 * cos (2 * pi * i / n)].

    This is the periodic form: the sample that would open the next period is
    dropped rather than repeated, which is what lets shifted copies sum to a
    constant and keeps {!istft} well conditioned. Pass it as {!stft}'s [win].

    Raises [Invalid_argument] if [n < 1].

    {@ocaml[
      # hann float64 4
      - : (float, float64_elt) t = [0, 0.5, 1, 0.5]
    ]} *)

val stft :
  (Complex.t, 'c) dtype ->
  window:int ->
  ?step:int ->
  ?win:(float, 'a) t ->
  (float, 'a) t ->
  (Complex.t, 'c) t
(** [stft dtype ~window ?step ?win t] is the short-time Fourier transform of [t]
    along its last axis: [t] is cut into frames of [window] samples every [step]
    samples, each frame is multiplied by [win], and each is transformed with
    {!rfft}.

    {b The result is time-major.} [[…; n]] becomes [[…; frames; window / 2 + 1]]
    with [frames = (n - window) / step + 1], so a spectrogram indexes as
    [t.{frame, bin}] and feeds a sequence model directly. Most signal-processing
    libraries return the transpose of this; swap the last two axes with
    {!transpose} if you want frequency-major. Samples past the last whole frame
    are dropped.

    [win] defaults to a periodic {!hann} taper of length [window] in [t]'s
    dtype, and must otherwise have shape [[|window|]] and [t]'s dtype. That
    default is deliberate rather than neutral: framing with no taper multiplies
    the signal by a rectangle, and a rectangle's own spectrum smears every
    component across every bin. Pass [~win:(ones (dtype t) [| window |])] for an
    untapered framing.

    [step] defaults to [window / 4], and to [1] for a window under 4. Framing is
    a view, so no framed copy of [t] is allocated; only the transform's own
    output is.

    Raises [Invalid_argument] if [window < 1], [step < 1], [window] exceeds the
    last axis, [t] is 0-d, or [win] has the wrong shape.

    {@ocaml[
      # stft complex128 ~window:16 ~step:8 (zeros float64 [| 64 |]) |> shape
      - : int array = [|7; 9|]
    ]}

    See also {!istft}, {!hann}. *)

val istft :
  (float, 'a) dtype ->
  window:int ->
  ?step:int ->
  ?win:(float, 'a) t ->
  ?length:int ->
  (Complex.t, 'c) t ->
  (float, 'a) t
(** [istft dtype ~window ?step ?win ?length z] reconstructs a signal from the
    frames [z] produced by {!stft} with the same [window], [step], and [win].

    Each frame is inverted with {!irfft}, multiplied by [win] again, and summed
    back at the position it was taken from; the sum is then divided by the
    overlap envelope the windows themselves produce. Dividing by the measured
    envelope rather than assuming a constant one makes reconstruction exact for
    every [step] that covers the signal, not only for the hops at which a given
    window happens to sum flat.

    [[…; frames; window / 2 + 1]] becomes [[…; (frames - 1) * step + window]],
    or exactly [length] samples when given — truncated, or zero-extended when
    the frames end early. Samples that [win] multiplied by zero carry no
    information and come back as [0]; under the default taper that is sample
    [0], which a periodic Hann sends to zero and no later frame reaches.

    [win] and [step] default as in {!stft}, so a pair called with the same
    arguments round-trips.

    Raises [Invalid_argument] if [step] is outside [[1, window]] (a wider step
    leaves gaps no frame covers), if [z] has fewer than 2 dimensions, if its
    last axis is not [window / 2 + 1], if [win] has the wrong shape, or if
    [length < 1].

    {@ocaml[
      # let x = init float64 [| 64 |] (fun i -> float_of_int i.(0)) in
        let w = hann float64 16 in
        let y =
          stft complex128 ~window:16 ~step:4 ~win:w x
          |> istft float64 ~window:16 ~step:4 ~win:w
        in
        (shape y, item [] (max (abs (sub y x))) < 1e-12)
      - : int array * bool = ([|64|], true)
    ]}

    See also {!stft}. *)

(** {1:activation Activation functions} *)

val relu : ('a, 'b) t -> ('a, 'b) t
(** [relu t] is [max(0, t)] element-wise.

    {@ocaml[
      # create float32 [| 5 |]
          [| -2.; -1.; 0.; 1.; 2. |]
        |> relu
      - : (float, float32_elt) t = float32 [5] [0, 0, ..., 1, 2]
    ]} *)

val sigmoid : ('a, 'b) t -> ('a, 'b) t
(** [sigmoid t] is [1 / (1 + exp(-t))] element-wise, in [[0, 1]]: an element is
    [0] or [1] only where its exact value rounds there.

    {@ocaml[
      # sigmoid (scalar float32 0.) |> item []
      - : float = 0.5
    ]} *)

val softmax : ?axes:int list -> ?scale:float -> ('a, 'b) t -> ('a, 'b) t
(** [softmax ?axes ?scale t] is the softmax normalisation
    [exp(scale * (t - max t)) / Σ exp(scale * (t - max t))]. [axes] defaults to
    [[-1]]. [scale] defaults to [1.0]. Output sums to [1] along the specified
    axes.

    {@ocaml[
      # create float32 [| 3 |] [| 1.; 2.; 3. |]
        |> softmax |> sum |> item []
      - : float = 1.
    ]}

    See also {!log_softmax}. *)

val log_softmax : ?axes:int list -> ?scale:float -> ('a, 'b) t -> ('a, 'b) t
(** [log_softmax ?axes ?scale t] is the natural logarithm of {!softmax}. Same
    defaults as {!softmax}.

    See also {!softmax}, {!logsumexp}. *)

val logsumexp : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [logsumexp ?axes ?keepdims t] is [log(Σ exp(t))] computed in a numerically
    stable way. [axes] defaults to all. [keepdims] defaults to [false].

    See also {!logmeanexp}, {!log_softmax}. *)

val logmeanexp : ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [logmeanexp ?axes ?keepdims t] is [log(mean(exp(t)))]: {!logsumexp} minus
    [log N]. [axes] defaults to all. [keepdims] defaults to [false].

    See also {!logsumexp}. *)

val standardize :
  ?axes:int list ->
  ?mean:('a, 'b) t ->
  ?variance:('a, 'b) t ->
  ?epsilon:float ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [standardize ?axes ?mean ?variance ?epsilon t] is
    [(t - mean) / sqrt(variance + epsilon)]. When [mean] or [variance] are
    omitted, they are computed along [axes] (default all). [epsilon] defaults to
    [1e-5]. *)

val erf : ('a, 'b) t -> ('a, 'b) t
(** [erf t] is the error function [erf(x) = (2/√π) ∫₀ˣ e^{-u²} du].

    {@ocaml[
      # erf (scalar float32 0.) |> item []
      - : float = 0.
    ]} *)

val erfinv : (float, 'b) t -> (float, 'b) t
(** [erfinv t] is the inverse of {!erf} on \[[-1], [1]\]: [erf (erfinv x) = x].
    It is [±infinity] at [±1] and NaN outside the interval.

    At float64 the result carries double precision over the whole interval, to
    the last representable value before [±1]. At narrower dtypes it carries
    about seven digits.

    {@ocaml[
      # erfinv (create float64 [| 3 |] [| -0.5; 0.; 0.5 |])
      - : (float, float64_elt) t = [-0.476936, 0, 0.476936]
    ]} *)

(** {1:windows Sliding windows} *)

(** {2:views Views} *)

val sliding_window :
  ?axis:int -> window:int -> ?step:int -> ('a, 'b) t -> ('a, 'b) t
(** [sliding_window ?axis ~window ?step t] is a {e view} of [t] with sliding
    windows of length [window] along [axis], taken every [step] elements. [axis]
    defaults to the last axis; negative indices count from the end. [step]
    defaults to [1].

    The size of [axis] becomes [(size - window) / step + 1] and a trailing axis
    of length [window] is appended. No data is copied, so framing a signal costs
    nothing: {!stft} is this view, a taper and one batched {!rfft}. Windows
    closer together than they are wide share storage, which nothing can observe:
    a tensor is a value. Under [Rune.jit] the windows are materialized.

    @raise Invalid_argument
      if [window < 1], [step < 1], [window] exceeds the size of [axis], or
      [axis] is out of bounds

    Framing a signal into overlapping windows:
    {[
      # create int32 [| 5 |] [| 1l; 2l; 3l; 4l; 5l |]
        |> sliding_window ~window:3
      - : (int32, int32_elt) t = int32 [3,3] [[1, 2, 3],
                                              [2, 3, 4],
                                              [3, 4, 5]]
    ]}

    Stepping by the window width gives disjoint blocks:
    {[
      # create int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
        |> sliding_window ~window:2 ~step:2
      - : (int32, int32_elt) t = int32 [3,2] [[1, 2],
                                              [3, 4],
                                              [5, 6]]
    ]}

    See also {!extract_patches}, which gathers a copy and so is always writable.
*)

(** {2:patches Patches} *)

val extract_patches :
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [extract_patches ~kernel_size ~stride ~dilation ~padding t] extracts sliding
    windows from the last [K] spatial dimensions where
    [K = Array.length kernel_size].

    Input: [[leading…; spatial…]]. Output: [[leading…; prod(kernel_size); L]],
    where [L] is the product over the spatial axes of the number of windows,
    [(n + before + after - (dilation (kernel - 1) + 1)) / stride + 1] for an
    axis of [n] elements, or [0] where the dilated kernel is longer than the
    padded axis.

    {@ocaml[
      # arange_f float32 0. 16. 1.
        |> reshape [| 1; 1; 4; 4 |]
        |> extract_patches
             ~kernel_size:[| 2; 2 |]
             ~stride:[| 1; 1 |]
             ~dilation:[| 1; 1 |]
             ~padding:[| (0, 0); (0, 0) |]
        |> shape
      - : int array = [|1; 1; 4; 9|]
    ]}

    Raises [Invalid_argument] if [kernel_size] is empty, if [stride],
    [dilation] and [padding] do not have one entry per kernel axis, if a size,
    stride or dilation is not positive or a padding negative, or if [t] has
    fewer axes than [kernel_size].

    See also {!combine_patches}. *)

val combine_patches :
  output_size:int array ->
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding t] is
    the inverse of {!extract_patches}: [t], of shape
    [[leading…; prod(kernel_size); L]], is placed back into a tensor of shape
    [[leading…; output_size…]]. Overlapping values are summed, and an element
    no window covers is zero.

    Raises [Invalid_argument] as {!extract_patches} does on the geometry, if
    [output_size] does not have one non-negative entry per kernel axis, or if
    [t]'s last two axes are not those an {!extract_patches} to [output_size]
    gives.

    See also {!extract_patches}. *)

(** {2:correlate Cross-correlation and convolution} *)

val correlate :
  ?padding:[ `Full | `Same | `Valid ] -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [correlate ?padding x kernel] is the N-D cross-correlation (no kernel flip).
    Spatial dimensions [K = ndim kernel]. Leading dimensions of [x] beyond [K]
    are batch dimensions. [padding] defaults to [`Valid].

    Along each spatial axis, of size [n] in [x] and [k] in [kernel], the full
    correlation has [n + k - 1] values, and [padding] keeps some of them. With
    [m = min n k]:
    - [`Full] keeps them all;
    - [`Valid] keeps the [max n k - m + 1] where the shorter of [x] and
      [kernel] lies within the longer, from index [m - 1];
    - [`Same] keeps [max n k], from index [(m - 1) / 2] if [k <= n] and from
      index [m / 2] if [k > n]. With [k <= n], value [i] correlates [kernel]
      with the elements of [x] from [i - k / 2] to [i + (k - 1) / 2], those
      outside [x] being zero: an odd kernel is centred on element [i].

    For [x = [|1.; 2.; 3.; 4.; 5.; 6.; 7.|]] and
    [kernel = [|1.; 10.; 100.; 1000.|]], [`Same] gives
    [[|2100.; 3210.; 4321.; 5432.; 6543.; 7654.; 765.|]].

    See also {!convolve}. *)

val convolve :
  ?padding:[ `Full | `Same | `Valid ] -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [convolve ?padding x kernel] is the N-D convolution: {!correlate} with the
    kernel flipped along all spatial axes, [padding] keeping the values of the
    full convolution from the same indices, except that [`Same] keeps them
    from index [(m - 1) / 2] whether [k] is shorter or longer than [n]. The
    convolution of two arrays without batch dimensions is commutative.

    See also {!correlate}. *)

(** {2:filters Filters} *)

val maximum_filter :
  kernel_size:int array -> ?stride:int array -> ('a, 'b) t -> ('a, 'b) t
(** [maximum_filter ~kernel_size ?stride t] is the sliding-window maximum over
    the last [K] dimensions. [stride] defaults to [kernel_size].

    See also {!minimum_filter}, {!uniform_filter}. *)

val minimum_filter :
  kernel_size:int array -> ?stride:int array -> ('a, 'b) t -> ('a, 'b) t
(** [minimum_filter ~kernel_size ?stride t] is the sliding-window minimum over
    the last [K] dimensions. [stride] defaults to [kernel_size].

    See also {!maximum_filter}. *)

val uniform_filter :
  kernel_size:int array -> ?stride:int array -> (float, 'b) t -> (float, 'b) t
(** [uniform_filter ~kernel_size ?stride t] is the sliding-window mean over the
    last [K] dimensions. [stride] defaults to [kernel_size].

    See also {!maximum_filter}, {!minimum_filter}. *)

(** {1:iteration Iteration} *)

val map_item : ('a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
(** [map_item f t] applies [f] to each scalar element of [t] and returns a fresh
    tensor of the results. *)

val iter_item : ('a -> unit) -> ('a, 'b) t -> unit
(** [iter_item f t] applies [f] to each scalar element of [t] for its side
    effects. *)

val fold_item : ('a -> 'b -> 'a) -> 'a -> ('b, 'c) t -> 'a
(** [fold_item f init t] folds [f] over the scalar elements of [t] in row-major
    order, starting with [init]. *)

(** {1:pp Formatting} *)

val pp : Format.formatter -> ('a, 'b) t -> unit
(** [pp ppf t] formats [t] compactly. Multidimensional or truncated tensors
    include their dtype and shape. *)

val to_string : ('a, 'b) t -> string
(** [to_string t] is [t] formatted with {!pp}. *)

val print : ('a, 'b) t -> unit
(** [print t] formats [t] with {!pp}, followed by a newline, on standard output.
*)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape ppf shape] formats [shape] as a bracketed, comma-separated list of
    dimensions, for example [[2,3,4]]. *)

val pp_dtype : Format.formatter -> ('a, 'b) dtype -> unit
(** [pp_dtype ppf dtype] formats [dtype] by name (e.g. [float32]). *)

(** {1:low_level For transformations and file formats}

    What a transformation and a file format need of nx's values: the operations
    as values, their interpretation, and the representation of a value. Programs
    never need this section. *)

(** Operations as values.

    Every operation nx computes or answers is a constructor of {!t}: those
    computed by the placement's backend, and {!Move}, {!Place} and {!Read},
    which nx answers itself. A transformation is an interpreter of these values,
    installed with {!intercept}. *)
module Op : sig
  type move = Nx_effect.move =
    | Reshape of int array
    | Expand of int array
    | Permute of int array
    | Shrink of (int * int) array
    | Flip of bool array
    | Window of { axis : int; size : int; step : int }
        (** The type for movements: views of a value's storage. *)

  type conversion = Nx_effect.conversion =
    | Cast
    | Bitcast
        (** The type for dtype conversions: [Cast] converts values, [Bitcast]
            reads the bits of each element as one of another dtype of its width.
        *)

  type 'r t = 'r Nx_effect.Op.t =
    | Unary : Nx_backend.unary * ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t t
    | Binary :
        Nx_backend.binary * ('a, 'b) Nx_effect.t * ('a, 'b) Nx_effect.t
        -> ('a, 'b) Nx_effect.t t
    | Compare :
        Nx_backend.compare * ('a, 'b) Nx_effect.t * ('a, 'b) Nx_effect.t
        -> (bool, Nx_dtype.bool_elt) Nx_effect.t t
    | Where :
        (bool, Nx_dtype.bool_elt) Nx_effect.t
        * ('a, 'b) Nx_effect.t
        * ('a, 'b) Nx_effect.t
        -> ('a, 'b) Nx_effect.t t
    | Reduce :
        Nx_backend.reduce * int array * ('a, 'b) Nx_effect.t
        -> ('a, 'b) Nx_effect.t t
    | Scan :
        Nx_backend.reduce * int * ('a, 'b) Nx_effect.t
        -> ('a, 'b) Nx_effect.t t
    | Arg_reduce :
        Nx_backend.arg_reduce * int * ('a, 'b) Nx_effect.t
        -> (int64, Nx_dtype.int64_elt) Nx_effect.t t
    | Sort : {
        descending : bool;
        axis : int;
        x : ('a, 'b) Nx_effect.t;
      }
        -> ('a, 'b) Nx_effect.t t
    | Argsort : {
        descending : bool;
        axis : int;
        x : ('a, 'b) Nx_effect.t;
      }
        -> (int64, Nx_dtype.int64_elt) Nx_effect.t t
    | Pad :
        (int * int) array * 'a * ('a, 'b) Nx_effect.t
        -> ('a, 'b) Nx_effect.t t
    | Cat : int * ('a, 'b) Nx_effect.t list -> ('a, 'b) Nx_effect.t t
    | Convert :
        conversion * ('c, 'd) Nx_dtype.t * ('a, 'b) Nx_effect.t
        -> ('c, 'd) Nx_effect.t t
    | Threefry :
        (int32, Nx_dtype.int32_elt) Nx_effect.t
        * (int32, Nx_dtype.int32_elt) Nx_effect.t
        -> (int32, Nx_dtype.int32_elt) Nx_effect.t t
    | Gather :
        int * (int64, Nx_dtype.int64_elt) Nx_effect.t * ('a, 'b) Nx_effect.t
        -> ('a, 'b) Nx_effect.t t
    | Scatter : {
        mode : [ `Set | `Add ];
        unique : bool;
        axis : int;
        indices : (int64, Nx_dtype.int64_elt) Nx_effect.t;
        updates : ('a, 'b) Nx_effect.t;
        into : ('a, 'b) Nx_effect.t;
      }
        -> ('a, 'b) Nx_effect.t t
    | Update :
        ('a, 'b) Nx_effect.t
        * (int64, Nx_dtype.int64_elt) Nx_effect.t
        * ('a, 'b) Nx_effect.t
        -> ('a, 'b) Nx_effect.t t
    | Unfold : {
        kernel_size : int array;
        stride : int array;
        dilation : int array;
        padding : (int * int) array;
        x : ('a, 'b) Nx_effect.t;
      }
        -> ('a, 'b) Nx_effect.t t
    | Fold : {
        output_size : int array;
        kernel_size : int array;
        stride : int array;
        dilation : int array;
        padding : (int * int) array;
        x : ('a, 'b) Nx_effect.t;
      }
        -> ('a, 'b) Nx_effect.t t
    | Matmul :
        ('a, 'b) Nx_effect.t * ('a, 'b) Nx_effect.t
        -> ('a, 'b) Nx_effect.t t
    | Fft : {
        inverse : bool;
        axes : int array;
        x : (Complex.t, 'b) Nx_effect.t;
      }
        -> (Complex.t, 'b) Nx_effect.t t
    | Rfft : {
        dtype : (Complex.t, 'c) Nx_dtype.t;
        axes : int array;
        x : (float, 'b) Nx_effect.t;
      }
        -> (Complex.t, 'c) Nx_effect.t t
    | Irfft : {
        dtype : (float, 'c) Nx_dtype.t;
        axes : int array;
        s : int array option;
        x : (Complex.t, 'b) Nx_effect.t;
      }
        -> (float, 'c) Nx_effect.t t
    | Contiguous : ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t t
    | Cholesky : {
        upper : bool;
        x : ('a, 'b) Nx_effect.t;
      }
        -> ('a, 'b) Nx_effect.t t
    | Qr : {
        reduced : bool;
        x : ('a, 'b) Nx_effect.t;
      }
        -> (('a, 'b) Nx_effect.t * ('a, 'b) Nx_effect.t) t
    | Lu :
        ('a, 'b) Nx_effect.t
        -> (('a, 'b) Nx_effect.t
           * (int64, Nx_dtype.int64_elt) Nx_effect.t
           * (int64, Nx_dtype.int64_elt) Nx_effect.t)
           t
    | Svd : {
        full_matrices : bool;
        x : ('a, 'b) Nx_effect.t;
      }
        -> (('a, 'b) Nx_effect.t
           * (float, Nx_dtype.float64_elt) Nx_effect.t
           * ('a, 'b) Nx_effect.t)
           t
    | Eig : {
        vectors : bool;
        x : ('a, 'b) Nx_effect.t;
      }
        -> ((Complex.t, Nx_dtype.complex64_elt) Nx_effect.t
           * (Complex.t, Nx_dtype.complex64_elt) Nx_effect.t option)
           t
    | Eigh : {
        vectors : bool;
        x : ('a, 'b) Nx_effect.t;
      }
        -> ((float, Nx_dtype.float64_elt) Nx_effect.t
           * ('a, 'b) Nx_effect.t option)
           t
    | Solve_triangular : {
        upper : bool;
        transpose : bool;
        unit_diag : bool;
        a : ('a, 'b) Nx_effect.t;
        b : ('a, 'b) Nx_effect.t;
      }
        -> ('a, 'b) Nx_effect.t t
    | Move : ('a, 'b) Nx_effect.t * move -> ('a, 'b) Nx_effect.t t
    | Place : Placement.t * ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t t
    | Read : ('a, 'b) Nx_effect.t -> Nx_device.Buffer.t t

  (** The type for operations whose result is ['r]. *)

  val eval : 'r t -> 'r
  (** [eval op] is [op]'s result in the current interpretation: delivered to the
      interpreter around the caller, or computed when there is none. *)

  val placement : 'r t -> Placement.t
  (** [placement op] is where [op]'s result lives, as {!eval} with no
      interpreter places it: host operands and values on the disk join any
      placement, and a traced operand joins as its placement says. The result of
      {!Read} is on the host.

      Raises [Invalid_argument] as {!eval} does when the operands cannot meet:
      placed operands with different backends or device sets, operands split
      differently, an operation along a split axis, or a movement that would
      move elements between devices. *)

  val shape : ('a, 'b) Nx_effect.t t -> int array
  (** [shape op] is the shape of [op]'s result, without computing it, for an
      [op] that {!eval} accepts. nx allocates every result at this shape.

      Raises [Invalid_argument] for a concatenation of no value. *)

  val dtype : ('a, 'b) Nx_effect.t t -> ('a, 'b) Nx_dtype.t
  (** [dtype op] is the dtype of [op]'s result, without computing it, for an
      [op] that {!eval} accepts.

      Raises [Invalid_argument] for a concatenation of no value. *)

  val operands : 'r t -> packed list
  (** [operands op] is [op]'s value operands, in order. *)

  val name : 'r t -> string
  (** [name op] is [op]'s name, as messages print it. *)

  val pp : Format.formatter -> 'r t -> unit
  (** [pp] formats an operation with its operands' dtypes and shapes, as in
      [mul float32[3] float32[3]]. *)

  type interpreter = Nx_effect.interpreter = {
    run : 'r. 'r t -> 'r;  (** [run op] is [op]'s result. *)
    claims : 'r. 'r t -> bool;
        (** [claims op] is [true] iff the interpreter takes [op], typically when
            an operand is one of its own values. *)
  }
  (** The type for interpreters of operations. *)

  val intercept : interpreter -> (unit -> 'a) -> 'a
  (** [intercept i f] is [f ()] with every operation [f]'s fiber performs
      within its extent and [i] claims delivered to [i.run]. [i.run] runs
      above every handler [f] installs: an operation it issues reaches the
      interpretation around [intercept], and an effect it performs the handlers
      around [intercept]. An operation [i] does not claim reaches the
      interpretation around [intercept] as if [i] were not installed, without
      being performed again. Fibers, threads and domains [f] starts are outside
      the extent. *)

  val intercepted : unit -> bool
  (** [intercepted ()] is [true] iff the calling fiber is inside the extent of
      an {!intercept}, outside its interpreter. *)
end

(** The representation of values.

    A value is an array on the host, a placed value over the storage of its
    devices, or a traced value that an interpreter made, which has no bytes. *)
module Repr : sig
  type ('a, 'b) node = ('a, 'b) Nx_effect.node = ..
  (** The type for the payload of traced values, which the interpreter that
      makes them extends. *)

  (** Storage: one runtime buffer per device of a placement. *)
  module Storage : sig
    type t = Nx_effect.cell
    (** The type for storage. *)

    val v : Placement.t -> Nx_device.Buffer.t list -> t
    (** [v p buffers] is the storage of [buffers], one per device of [p], in
        order, each of the same length and in the memory of its device.

        Raises [Invalid_argument] otherwise. *)

    val buffers : t -> Nx_device.Buffer.t list
    (** [buffers s] is [s]'s buffers, one per device.

        Raises [Invalid_argument] if [s] was consumed, or if its devices hold it
        in memory of their own. *)

    val placement : t -> Placement.t
    (** [placement s] is where [s] lives. *)

    (** {2:claims Claims}

        A compiled call claims the storage its arguments reach. It borrows each
        storage for reading, upgrades the borrow of each storage it consumes to
        an exclusive claim, consumes it, and finishes. A program that binds a
        storage pins it for as long as the program lives. Consumed storage is
        retired once no call holds it and no program pins it. *)

    val borrow : t -> unit
    (** [borrow s] claims [s] for reading, beside other readers.

        Raises [Invalid_argument] if a call holds [s] exclusively. *)

    val release : t -> unit
    (** [release s] ends a {!borrow} of [s].

        Raises [Invalid_argument] if [s] has no borrow to end. *)

    val upgrade : t -> unit
    (** [upgrade s] turns the caller's borrow of [s] into an exclusive claim.

        Raises [Invalid_argument] if another reader or call holds [s]. *)

    val consume : t -> path:string -> unit
    (** [consume s ~path] consumes [s] at [path], the consumed leaf's path in a
        compiled call's arguments: every later read or use of a value over [s]
        raises [Invalid_argument] naming [path], while shapes and dtypes stay
        readable. Nothing unconsumes [s].

        Raises [Invalid_argument] unless the caller holds [s] exclusively. *)

    val finish : t -> bool
    (** [finish s] turns the exclusive claim on [s] back into the caller's
        borrow, which it then {!release}s. It is [true] iff [s] is consumed and
        no program pins it: its buffers can be retired. *)

    val pin : t -> unit
    (** [pin s] records one more program that binds [s]. *)

    val unpin : t -> bool
    (** [unpin s] records one program fewer that binds [s]. It is [true] iff
        that was the last, [s] is consumed and no call holds it exclusively: its
        buffers can be retired. *)

    val live : t -> bool
    (** [live s] is [true] iff [s] was not consumed. *)

    val pins : t -> int
    (** [pins s] is the number of programs that bind [s]. *)
  end

  (** Placed values. *)
  module Placed : sig
    type ('a, 'b) t = ('a, 'b) Nx_effect.resident
    (** The type for placed values. *)

    val v :
      Placement.t ->
      ('a, 'b) dtype ->
      Nx_array.View.t ->
      Storage.t ->
      ('a, 'b) Nx_effect.t
    (** [v p dtype view s] is the value of [dtype] at [p] whose elements, on
        each device, are those [view] reaches in its storage [s].

        Raises [Invalid_argument] if [p] is {!Placement.host}, if [view] reaches
        an element outside [s], or if [s]'s devices do not hold [p]'s. *)

    val id : ('a, 'b) t -> int
    (** [id x] is [x]'s identity, for hashing: every placed value has its own.
    *)

    val view : ('a, 'b) t -> Nx_array.View.t
    (** [view x] is the view each device has of its storage. *)

    val storage : ('a, 'b) t -> Storage.t
    (** [storage x] is [x]'s storage, shared with every view of it. *)
  end

  (** Traced values. *)
  module Traced : sig
    type ('a, 'b) t = ('a, 'b) Nx_effect.traced
    (** The type for traced values. *)

    val v :
      context:Placement.t ->
      Placement.t ->
      ('a, 'b) dtype ->
      int array ->
      ('a, 'b) node ->
      ('a, 'b) Nx_effect.t
    (** [v ~context p dtype shape node] is a traced value of [dtype] and [shape]
        at [p], whose payload is [node]. A value made beside it is made at
        [context]. *)

    val id : ('a, 'b) t -> int
    (** [id x] is [x]'s identity: every traced value has its own, and a value
        traced later has a larger one. *)

    val node : ('a, 'b) t -> ('a, 'b) node
    (** [node x] is [x]'s payload. *)
  end

  type ('a, 'b) t = ('a, 'b) Nx_effect.t =
    | Host : ('a, 'b) Nx_array.t -> ('a, 'b) t
    | Placed : ('a, 'b) Placed.t -> ('a, 'b) t
    | Traced : ('a, 'b) Traced.t -> ('a, 'b) t
        (** The type for the representation of values. *)

  val v : ('a, 'b) Nx_effect.t -> ('a, 'b) t
  (** [v x] is [x]'s representation. *)

  val host : ('a, 'b) Nx_array.t -> ('a, 'b) Nx_effect.t
  (** [host a] is the value at {!Placement.host} of array [a].

      Raises [Invalid_argument] if [a]'s buffer is not on the host or not of
      [a]'s dtype, or if [a]'s view reaches an element outside it. *)

  val context : ('a, 'b) Nx_effect.t -> Placement.t
  (** [context x] is where a value made beside [x] is made: the host for a host
      value or one on the disk, a copy on each of [x]'s devices for a placed
      one, and a traced value's context. *)
end
