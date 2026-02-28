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

    {b The [?out] convention.} Many operations accept an optional [?out] tensor.
    When provided, the result is written into [out] instead of allocating a
    fresh tensor; the shape of [out] must match the result shape. *)

(** {1:types Types} *)

type ('a, 'b) t = ('a, 'b) Nx_effect.t
(** The type for tensors with OCaml element type ['a] and buffer element kind
    ['b]. *)

(** {2:elt_kinds Element kinds}

    Witnesses for the buffer element representation. Used as the second type
    parameter of {!type-t}. *)

type float16_elt = Nx_buffer.float16_elt
type float32_elt = Nx_buffer.float32_elt
type float64_elt = Nx_buffer.float64_elt
type bfloat16_elt = Nx_buffer.bfloat16_elt
type float8_e4m3_elt = Nx_buffer.float8_e4m3_elt
type float8_e5m2_elt = Nx_buffer.float8_e5m2_elt
type int4_elt = Nx_buffer.int4_signed_elt
type uint4_elt = Nx_buffer.int4_unsigned_elt
type int8_elt = Nx_buffer.int8_signed_elt
type uint8_elt = Nx_buffer.int8_unsigned_elt
type int16_elt = Nx_buffer.int16_signed_elt
type uint16_elt = Nx_buffer.int16_unsigned_elt
type int32_elt = Nx_buffer.int32_elt
type uint32_elt = Nx_buffer.uint32_elt
type int64_elt = Nx_buffer.int64_elt
type uint64_elt = Nx_buffer.uint64_elt
type complex32_elt = Nx_buffer.complex32_elt
type complex64_elt = Nx_buffer.complex64_elt
type bool_elt = Nx_buffer.bool_elt

(** {2:dtype Data types} *)

type ('a, 'b) dtype = ('a, 'b) Nx_core.Dtype.t =
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
          element type ['a] to its buffer representation ['b]. *)

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

(** The type for index specifications used by {!val-slice} and {!set_slice}. *)
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
      (** [M mask] selects positions where [mask] is [true]. *)
  | N  (** [N] inserts a new axis of size 1 (does not consume an input axis). *)

(** {1:properties Properties} *)

val data : ('a, 'b) t -> ('a, 'b) Nx_buffer.t
(** [data t] is the underlying flat buffer of [t].

    The buffer is shared: mutations through the buffer are visible through [t]
    and vice-versa. The buffer may be larger than the tensor's logical extent
    when [t] is a strided view. *)

val shape : ('a, 'b) t -> int array
(** [shape t] is the dimensions of [t]. A scalar tensor has shape [|\||]. *)

val dtype : ('a, 'b) t -> ('a, 'b) dtype
(** [dtype t] is the data type of [t]. *)

val strides : ('a, 'b) t -> int array
(** [strides t] is the byte stride for each dimension of [t].

    Raises [Invalid_argument] if [t] does not have computable strides (e.g.
    after certain non-contiguous view operations). Use {!is_c_contiguous} or
    call {!contiguous} first.

    See also {!stride}. *)

val stride : int -> ('a, 'b) t -> int
(** [stride i t] is the byte stride of dimension [i].

    Raises [Invalid_argument] if [i] is out of bounds or [t] does not have
    computable strides.

    See also {!strides}. *)

val dims : ('a, 'b) t -> int array
(** [dims t] is {!shape}. *)

val dim : int -> ('a, 'b) t -> int
(** [dim i t] is the size of dimension [i].

    Raises [Invalid_argument] if [i] is out of bounds. *)

val ndim : ('a, 'b) t -> int
(** [ndim t] is the number of dimensions of [t]. *)

val itemsize : ('a, 'b) t -> int
(** [itemsize t] is the number of bytes per element. *)

val size : ('a, 'b) t -> int
(** [size t] is the total number of elements. *)

val numel : ('a, 'b) t -> int
(** [numel t] is {!size}. *)

val nbytes : ('a, 'b) t -> int
(** [nbytes t] is [size t * itemsize t]. *)

val offset : ('a, 'b) t -> int
(** [offset t] is the element offset of [t] in its underlying buffer. *)

val is_c_contiguous : ('a, 'b) t -> bool
(** [is_c_contiguous t] is [true] iff [t]'s elements are laid out contiguously
    in row-major (C) order.

    See also {!contiguous}. *)

val to_bigarray : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_bigarray t] is a contiguous bigarray with the same shape and data as
    [t]. Always copies.

    Raises [Invalid_argument] if [t]'s dtype is an extended type not supported
    by [Bigarray].

    See also {!of_bigarray}. *)

val to_buffer : ('a, 'b) t -> ('a, 'b) Nx_buffer.t
(** [to_buffer t] is a flat, contiguous buffer of [t]'s data.

    Returns the underlying buffer directly when [t] is already contiguous with
    zero offset and matching size; copies otherwise. *)

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

(** {1:creation Creation} *)

val create : ('a, 'b) dtype -> int array -> 'a array -> ('a, 'b) t
(** [create dtype shape data] is a tensor of the given [dtype] and [shape]
    initialised from [data] in row-major order.

    Raises [Invalid_argument] if [Array.length data] does not equal the product
    of [shape].

    {@ocaml[
      # create float32 [| 2; 3 |]
          [| 1.; 2.; 3.; 4.; 5.; 6. |]
      - : (float, float32_elt) t = float32 [2; 3] [[1, 2, 3],
                                                   [4, 5, 6]]
    ]} *)

val init : ('a, 'b) dtype -> int array -> (int array -> 'a) -> ('a, 'b) t
(** [init dtype shape f] is a tensor where the element at multi-index [i] is
    [f i].

    {@ocaml[
      # init int32 [| 2; 3 |]
          (fun i -> Int32.of_int (i.(0) + i.(1)))
      - : (int32, int32_elt) t = int32 [2; 3] [[0, 1, 2],
                                               [1, 2, 3]]
    ]} *)

val empty : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [empty dtype shape] is an uninitialized tensor.

    {b Warning.} Elements contain arbitrary values until written. *)

val full : ('a, 'b) dtype -> int array -> 'a -> ('a, 'b) t
(** [full dtype shape v] is a tensor filled with [v].

    {@ocaml[
      # full float32 [| 2; 3 |] 3.14
      - : (float, float32_elt) t = float32 [2; 3]
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

val empty_like : ('a, 'b) t -> ('a, 'b) t
(** [empty_like t] is {!empty} with the same dtype and shape as [t]. *)

val full_like : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [full_like t v] is {!full} with the same dtype and shape as [t]. *)

val ones_like : ('a, 'b) t -> ('a, 'b) t
(** [ones_like t] is {!ones} with the same dtype and shape as [t]. *)

val zeros_like : ('a, 'b) t -> ('a, 'b) t
(** [zeros_like t] is {!zeros} with the same dtype and shape as [t]. *)

val scalar_like : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [scalar_like t v] is {!scalar} with the same dtype as [t]. *)

val eye : ?m:int -> ?k:int -> ('a, 'b) dtype -> int -> ('a, 'b) t
(** [eye ?m ?k dtype n] is an [n × m] matrix with ones on the [k]-th diagonal
    and zeros elsewhere. [m] defaults to [n]. [k] defaults to [0] (main
    diagonal); positive [k] selects an upper diagonal, negative [k] a lower one.

    {@ocaml[
      # eye int32 3
      - : (int32, int32_elt) t = int32 [3; 3] [[1, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 1]]
      # eye ~k:1 int32 3
      - : (int32, int32_elt) t = int32 [3; 3] [[0, 1, 0],
                                               [0, 0, 1],
                                               [0, 0, 0]]
    ]}

    See also {!identity}, {!diag}. *)

val identity : ('a, 'b) dtype -> int -> ('a, 'b) t
(** [identity dtype n] is [eye dtype n]. *)

val diag : ?k:int -> ('a, 'b) t -> ('a, 'b) t
(** [diag ?k v] extracts or constructs a diagonal.

    When [v] is 1-D, returns a 2-D tensor with [v] on the [k]-th diagonal. When
    [v] is 2-D, returns the [k]-th diagonal as a 1-D tensor. [k] defaults to
    [0].

    Raises [Invalid_argument] if [v] is not 1-D or 2-D.

    {@ocaml[
      # let v = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        diag v
      - : (int32, int32_elt) t = int32 [3; 3] [[1, 0, 0],
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
(** [arange dtype start stop step] is a 1-D tensor of values from [start]
    (inclusive) to [stop] (exclusive) with stride [step].

    Raises [Invalid_argument] if [step = 0].

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
      (float32 [2; 3] [[0, 1, 2],
                       [0, 1, 2]], float32 [2; 3] [[0, 0, 0],
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
(** [of_bigarray ba] is a tensor sharing memory with [ba].

    Zero-copy: mutations through either are visible to both.

    See also {!to_bigarray}. *)

val of_buffer : ('a, 'b) Nx_buffer.t -> shape:int array -> ('a, 'b) t
(** [of_buffer buf ~shape] is a tensor viewing [buf] with the given [shape]. The
    product of [shape] must equal the buffer length. *)

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
      - : (int, uint8_elt) t = uint8 [3; 4]
      [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1]]
    ]} *)

(** {1:rng Random number generation}

    Sampling functions use the implicit RNG state managed by {!module-Rng}. Wrap
    calls in {!Rng.run} for reproducibility:

    {v
      Rng.run ~seed:42 (fun () -> rand float32 [| 3 |])
    v} *)

module Rng : sig
  (** Splittable RNG keys and implicit key management.

      Keys are deterministic integers that can be split to derive independent
      subkeys. {!run} and {!with_key} install an effect handler that provides
      implicit key threading via {!next_key}; outside any handler a domain-local
      auto-seeded generator is used as a convenient fallback. *)

  (** {1:keys Keys} *)

  type key = int
  (** The type for RNG keys. *)

  val key : int -> key
  (** [key seed] is a normalized 31-bit non-negative key derived from [seed]. *)

  val split : ?n:int -> key -> key array
  (** [split ?n k] deterministically derives [n] subkeys from [k].

      [n] defaults to [2]. *)

  val fold_in : key -> int -> key
  (** [fold_in k data] mixes [data] into [k] and returns the derived key. *)

  val to_int : key -> int
  (** [to_int k] is [k] as an integer. *)

  (** {1:implicit Implicit key management} *)

  val next_key : unit -> key
  (** [next_key ()] returns a fresh subkey from the current RNG scope.

      Inside a {!run} or {!with_key} block, each call returns a
      deterministically derived key. Outside any scope, falls back to a
      domain-local auto-seeded generator (convenient but non-reproducible).

      Two calls to [next_key ()] always return different keys. *)

  val run : seed:int -> (unit -> 'a) -> 'a
  (** [run ~seed f] executes [f] in an RNG scope seeded by [seed].

      Every {!next_key} call within [f] returns a deterministically derived key.
      The same [seed] and the same sequence of [next_key] calls produce the same
      keys. Scopes nest: an inner [run] replaces the outer scope for its
      duration. *)

  val with_key : key -> (unit -> 'a) -> 'a
  (** [with_key k f] executes [f] in an RNG scope initialized from [k].

      This is the explicit-key equivalent of [run]: useful when you have an
      existing key from a split and want to establish a scope for a
      sub-computation (e.g. in layer composition). *)
end

val rand : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [rand dtype shape] samples uniformly from \[[0], [1]).

    Raises [Invalid_argument] if [dtype] is not a float type. *)

val randn : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [randn dtype shape] samples from the standard normal distribution (mean 0,
    variance 1) via the Box–Muller transform.

    Raises [Invalid_argument] if [dtype] is not a float type. *)

val randint : ('a, 'b) dtype -> ?high:int -> int array -> int -> ('a, 'b) t
(** [randint dtype ?high shape low] samples integers uniformly from \[[low],
    [high]). [high] defaults to [10].

    Raises [Invalid_argument] if [dtype] is not an integer type or
    [low >= high]. *)

val bernoulli : p:float -> int array -> bool_t
(** [bernoulli ~p shape] samples booleans that are [true] with probability [p].

    Raises [Invalid_argument] if [p] is not in \[[0], [1]\]. *)

val permutation : int -> int32_t
(** [permutation n] is a random permutation of \[[0], [n-1]\].

    Raises [Invalid_argument] if [n <= 0]. *)

val shuffle : ('a, 'b) t -> ('a, 'b) t
(** [shuffle t] is a copy of [t] with the first axis randomly permuted. No-op on
    scalars. *)

val categorical : ?axis:int -> ?shape:int array -> (float, 'a) t -> int32_t
(** [categorical ?axis ?shape logits] samples category indices from unnormalised
    log-probabilities using the Gumbel-max trick. [axis] defaults to [-1] (last
    axis). [shape] prepends extra batch dimensions.

    Raises [Invalid_argument] if [logits] is not a float type or [axis] is out
    of bounds. *)

val truncated_normal :
  ('a, 'b) dtype -> lower:float -> upper:float -> int array -> ('a, 'b) t
(** [truncated_normal dtype ~lower ~upper shape] samples from a standard normal
    distribution truncated to \[[lower], [upper]\].

    Raises [Invalid_argument] if [dtype] is not a float type or
    [lower >= upper]. *)

(** {1:shape Shape manipulation} *)

val reshape : int array -> ('a, 'b) t -> ('a, 'b) t
(** [reshape shape t] is a view of [t] with the given [shape].

    At most one dimension may be [-1]; it is inferred from the total number of
    elements. The product of [shape] must equal {!size} [t].

    Raises [Invalid_argument] if [shape] is incompatible or contains more than
    one [-1].

    {@ocaml[
      # create int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
        |> reshape [| 2; 3 |]
      - : (int32, int32_elt) t = int32 [2; 3] [[1, 2, 3],
                                               [4, 5, 6]]
      # create int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
        |> reshape [| 3; -1 |]
      - : (int32, int32_elt) t = int32 [3; 2] [[1, 2],
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
      - : (int32, int32_elt) t = int32 [3; 3] [[1, 2, 3],
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
    [end_dim] defaults to [-1] (last). Negative indices count from the end.

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

    See also {!squeeze}, {!expand_dims}. *)

val squeeze_axis : int -> ('a, 'b) t -> ('a, 'b) t
(** [squeeze_axis i t] removes dimension [i] if its size is 1.

    Raises [Invalid_argument] if dimension [i] is not 1.

    See also {!squeeze}. *)

val unsqueeze_axis : int -> ('a, 'b) t -> ('a, 'b) t
(** [unsqueeze_axis i t] inserts a dimension of size 1 at position [i].

    See also {!unsqueeze}. *)

val expand_dims : int list -> ('a, 'b) t -> ('a, 'b) t
(** [expand_dims axes t] is {!unsqueeze} [~axes t]. *)

val transpose : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [transpose ?axes t] permutes the dimensions of [t].

    [axes] must be a permutation of [[0; …; ndim t - 1]]. When omitted, reverses
    all dimensions. Returns a view (no copy).

    Raises [Invalid_argument] if [axes] is not a valid permutation.

    {@ocaml[
      # create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
        |> transpose
      - : (int32, int32_elt) t = int32 [3; 2] [[1, 4],
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
      - : (int32, int32_elt) t = int32 [2; 3] [[3, 2, 1],
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
    wrapping around. When [axis] is omitted, operates on the flattened tensor.
    Negative [shift] rolls backward.

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
      - : (int32, int32_elt) t = int32 [2; 2] [[4, 5],
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
      - : (int32, int32_elt) t = int32 [2; 6] [[1, 2, ..., 1, 2],
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

val concatenate : ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
(** [concatenate ?axis ts] joins tensors along an existing axis. All tensors
    must have the same shape except on the concatenation axis. When [axis] is
    omitted, every tensor is flattened first. Always copies.

    Raises [Invalid_argument] if the list is empty or shapes are incompatible.

    {@ocaml[
      # let a = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        let b = create int32 [| 1; 2 |] [| 5l; 6l |] in
        concatenate ~axis:0 [ a; b ]
      - : (int32, int32_elt) t = int32 [3; 2] [[1, 2],
                                               [3, 4],
                                               [5, 6]]
    ]}

    See also {!stack}, {!vstack}, {!hstack}. *)

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
      - : (int32, int32_elt) t = int32 [2; 2] [[1, 2],
                                               [3, 4]]
      # let a = create int32 [| 2 |] [| 1l; 2l |] in
        let b = create int32 [| 2 |] [| 3l; 4l |] in
        stack ~axis:1 [ a; b ]
      - : (int32, int32_elt) t = int32 [2; 2] [[1, 3],
                                               [2, 4]]
    ]}

    See also {!concatenate}. *)

val vstack : ('a, 'b) t list -> ('a, 'b) t
(** [vstack ts] stacks vertically (along axis 0). 1-D tensors are treated as row
    vectors (shape [[1; n]]).

    Raises [Invalid_argument] if shapes are incompatible.

    {@ocaml[
      # let a = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        let b = create int32 [| 3 |] [| 4l; 5l; 6l |] in
        vstack [ a; b ]
      - : (int32, int32_elt) t = int32 [2; 3] [[1, 2, 3],
                                               [4, 5, 6]]
    ]}

    See also {!hstack}, {!dstack}, {!concatenate}. *)

val hstack : ('a, 'b) t list -> ('a, 'b) t
(** [hstack ts] stacks horizontally. 1-D tensors are concatenated directly;
    higher-D tensors concatenate along axis 1.

    Raises [Invalid_argument] if shapes are incompatible.

    {@ocaml[
      # let a = create int32 [| 2; 1 |] [| 1l; 2l |] in
        let b = create int32 [| 2; 1 |] [| 3l; 4l |] in
        hstack [ a; b ]
      - : (int32, int32_elt) t = int32 [2; 2] [[1, 3],
                                               [2, 4]]
    ]}

    See also {!vstack}, {!dstack}, {!concatenate}. *)

val dstack : ('a, 'b) t list -> ('a, 'b) t
(** [dstack ts] stacks depth-wise (along axis 2). Tensors are reshaped to at
    least 3-D before concatenation: 1-D [[n]] → [[1; n; 1]], 2-D [[m; n]] →
    [[m; n; 1]].

    Raises [Invalid_argument] if the resulting shapes are incompatible.

    See also {!vstack}, {!hstack}, {!concatenate}. *)

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
(** [cast dtype t] is a copy of [t] with elements converted to [dtype].

    {@ocaml[
      # create float32 [| 3 |] [| 1.5; 2.7; 3.1 |]
        |> cast int32
      - : (int32, int32_elt) t = [1, 2, 3]
    ]}

    See also {!contiguous}, {!copy}. *)

val astype : ('a, 'b) dtype -> ('c, 'd) t -> ('a, 'b) t
(** [astype dtype t] is {!cast}. *)

val contiguous : ('a, 'b) t -> ('a, 'b) t
(** [contiguous t] is [t] if it is already C-contiguous, or a fresh contiguous
    copy otherwise.

    See also {!is_c_contiguous}, {!copy}. *)

val copy : ('a, 'b) t -> ('a, 'b) t
(** [copy t] is a deep copy of [t]. Always allocates new memory; the result is
    contiguous.

    {@ocaml[
      # let x = create float32 [| 3 |] [| 1.; 2.; 3. |] in
        let y = copy x in
        set_item [ 0 ] 999. y;
        x, y
      - : (float, float32_elt) t * (float, float32_elt) t =
      ([1, 2, 3], [999, 2, 3])
    ]}

    See also {!contiguous}. *)

val blit : ('a, 'b) t -> ('a, 'b) t -> unit
(** [blit src dst] copies the elements of [src] into [dst] in-place. Shapes must
    match exactly.

    Raises [Invalid_argument] if shapes differ. *)

val fill : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [fill v t] is a fresh copy of [t] with every element set to [v]. Does not
    mutate [t]. *)

(** {1:indexing Indexing and slicing} *)

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

val set : int list -> ('a, 'b) t -> ('a, 'b) t -> unit
(** [set indices t v] writes [v] at the position given by [indices].

    Raises [Invalid_argument] if indices are out of bounds. *)

val slice : index list -> ('a, 'b) t -> ('a, 'b) t
(** [slice specs t] extracts a sub-tensor using advanced indexing.

    Each element of [specs] addresses one axis from left to right:
    - [I i] — single index (reduces dimension; negative from end).
    - [L [i0; i1; …]] — gather listed indices.
    - [R (start, stop)] — half-open range \[[start], [stop]).
    - [Rs (start, stop, step)] — strided range.
    - [A] — full axis (default for trailing axes).
    - [M mask] — boolean mask selecting positions where [mask] is [true].
    - [N] — insert a new axis of size 1.

    Returns a view when possible.

    Raises [Invalid_argument] if specs are out of bounds, if step is zero, or if
    a mask spec is used (not yet supported).

    {@ocaml[
      # let x =
          create int32 [| 3; 3 |]
            [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |]
        in
        slice [ R (0, 2); L [ 0; 2 ] ] x
      - : (int32, int32_elt) t = int32 [2; 2] [[1, 3],
                                               [4, 6]]
    ]}

    See also {!get}, {!set_slice}. *)

val set_slice : index list -> ('a, 'b) t -> ('a, 'b) t -> unit
(** [set_slice specs t v] writes [v] into the region of [t] selected by [specs].
    [v] is broadcast if needed.

    Raises [Invalid_argument] if [N] (new-axis) specs are used (not supported
    for writes).

    See also {!val-slice}. *)

val item : int list -> ('a, 'b) t -> 'a
(** [item indices t] is the scalar value at [indices]. Indices must cover all
    dimensions.

    Raises [Invalid_argument] if the number of indices is wrong or any index is
    out of bounds.

    See also {!get}, {!set_item}. *)

val set_item : int list -> 'a -> ('a, 'b) t -> unit
(** [set_item indices v t] sets the element at [indices] to [v] in-place.
    Indices must cover all dimensions.

    Raises [Invalid_argument] if the number of indices is wrong or any index is
    out of bounds.

    See also {!item}. *)

val take :
  ?axis:int ->
  ?mode:[ `raise | `wrap | `clip ] ->
  (int32, int32_elt) t ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [take ?axis ?mode indices t] gathers elements from [t] at [indices] along
    [axis]. When [axis] is omitted, [t] is flattened first. [mode] controls
    out-of-bounds indices: [`raise] (default) raises, [`wrap] uses modular
    indexing, [`clip] clamps to bounds.

    Raises [Invalid_argument] if [mode] is [`raise] and any index is out of
    bounds.

    {@ocaml[
      # let x =
          create int32 [| 5 |]
            [| 0l; 1l; 2l; 3l; 4l |]
        in
        take
          (create int32 [| 3 |] [| 1l; 3l; 0l |])
          x
      - : (int32, int32_elt) t = [1, 3, 0]
    ]}

    See also {!put}, {!take_along_axis}. *)

val take_along_axis :
  axis:int -> (int32, int32_elt) t -> ('a, 'b) t -> ('a, 'b) t
(** [take_along_axis ~axis indices t] gathers values from [t] along [axis] using
    [indices]. [indices] must match [t]'s shape except along [axis]. Useful for
    gathering from {!argmax}/{!argmin} results.

    Raises [Invalid_argument] if shapes are incompatible.

    {@ocaml[
      # let x =
          create float32 [| 2; 3 |]
            [| 4.; 1.; 2.; 3.; 5.; 6. |]
        in
        let idx =
          create int32 [| 2; 1 |] [| 1l; 0l |]
        in
        take_along_axis ~axis:1 idx x
      - : (float, float32_elt) t = float32 [2; 1] [[1],
                                                   [3]]
    ]}

    See also {!take}, {!put_along_axis}. *)

val put :
  ?axis:int ->
  indices:(int32, int32_elt) t ->
  values:('a, 'b) t ->
  ?mode:[ `raise | `wrap | `clip ] ->
  ('a, 'b) t ->
  unit
(** [put ?axis ~indices ~values ?mode t] writes [values] into [t] at positions
    given by [indices]. When [axis] is omitted, [t] is flattened first. [mode]
    defaults to [`raise]. Modifies [t] in-place.

    Raises [Invalid_argument] if [mode] is [`raise] and any index is out of
    bounds.

    See also {!take}, {!put_along_axis}, {!index_put}. *)

val index_put :
  indices:(int32, int32_elt) t array ->
  values:('a, 'b) t ->
  ?mode:[ `raise | `wrap | `clip ] ->
  ('a, 'b) t ->
  unit
(** [index_put ~indices ~values ?mode t] writes [values] into [t] at the
    coordinates given by [indices].

    [indices] contains one index tensor per axis of [t]; they are broadcast to a
    common shape that determines the number of updates. [values] is broadcast to
    the same shape. Duplicate coordinates overwrite. [mode] defaults to
    [`raise].

    Raises [Invalid_argument] if the number of index tensors does not match
    {!ndim} [t].

    {@ocaml[
      # let t = zeros float32 [| 3; 3 |] in
        let rows =
          create int32 [| 3 |] [| 0l; 2l; 1l |]
        in
        let cols =
          create int32 [| 3 |] [| 1l; 0l; 2l |]
        in
        index_put ~indices:[| rows; cols |]
          ~values:(create float32 [| 3 |]
                     [| 10.; 20.; 30. |])
          t;
        t
      - : (float, float32_elt) t = float32 [3; 3]
      [[0, 10, 0],
       [0, 0, 30],
       [20, 0, 0]]
    ]}

    See also {!put}. *)

val put_along_axis :
  axis:int ->
  indices:(int32, int32_elt) t ->
  values:('a, 'b) t ->
  ('a, 'b) t ->
  unit
(** [put_along_axis ~axis ~indices ~values t] writes [values] into [t] at
    positions selected by [indices] along [axis]. Modifies [t] in-place.

    Raises [Invalid_argument] if shapes are incompatible.

    See also {!take_along_axis}, {!put}. *)

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

val nonzero : ('a, 'b) t -> (int32, int32_elt) t array
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
      - : (int32, int32_elt) t * (int32, int32_elt) t =
      ([0, 1, 1, 2], [1, 0, 2, 2])
    ]}

    See also {!argwhere}. *)

val argwhere : ('a, 'b) t -> (int32, int32_elt) t
(** [argwhere t] is a 2-D tensor of shape [[k; ndim t]] whose rows are the
    coordinates of the [k] non-zero elements.

    See also {!nonzero}. *)

(** {1:arithmetic Arithmetic}

    Element-wise arithmetic with broadcasting. Each operation [op] has variants:
    - [op_s t s] — tensor-scalar.
    - [rop_s s t] — scalar-tensor (reversed operands). *)

val add : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [add ?out a b] is the element-wise sum of [a] and [b]. [out] defaults to a
    fresh allocation. *)

val add_s : ?out:('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [add_s ?out t s] adds scalar [s] to each element of [t]. [out] defaults to a
    fresh allocation. *)

val radd_s : ?out:('a, 'b) t -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [radd_s ?out s t] is [add_s ?out t s]. *)

val sub : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sub ?out a b] is the element-wise difference [a - b]. [out] defaults to a
    fresh allocation. *)

val sub_s : ?out:('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [sub_s ?out t s] subtracts scalar [s] from each element. [out] defaults to a
    fresh allocation. *)

val rsub_s : ?out:('a, 'b) t -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rsub_s ?out s t] is [s - t] element-wise. [out] defaults to a fresh
    allocation. *)

val mul : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [mul ?out a b] is the element-wise product of [a] and [b]. [out] defaults to
    a fresh allocation. *)

val mul_s : ?out:('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [mul_s ?out t s] multiplies each element by scalar [s]. [out] defaults to a
    fresh allocation. *)

val rmul_s : ?out:('a, 'b) t -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rmul_s ?out s t] is [mul_s ?out t s]. *)

val div : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [div ?out a b] is the element-wise quotient [a / b]. [out] defaults to a
    fresh allocation.

    Float dtypes use true division. Integer dtypes truncate toward zero.

    {@ocaml[
      # let x = create int32 [| 2 |] [| -7l; 8l |] in
        let y = create int32 [| 2 |] [| 2l; 2l |] in
        div x y
      - : (int32, int32_elt) t = [-3, 4]
    ]} *)

val div_s : ?out:('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [div_s ?out t s] divides each element by scalar [s]. [out] defaults to a
    fresh allocation. *)

val rdiv_s : ?out:('a, 'b) t -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rdiv_s ?out s t] is [s / t] element-wise. [out] defaults to a fresh
    allocation. *)

val pow : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [pow ?out base exp] is [base] raised to [exp] element-wise. [out] defaults
    to a fresh allocation. *)

val pow_s : ?out:('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [pow_s ?out t s] raises each element to scalar power [s]. [out] defaults to
    a fresh allocation. *)

val rpow_s : ?out:('a, 'b) t -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rpow_s ?out s t] is [s{^t}] element-wise. [out] defaults to a fresh
    allocation. *)

val mod_ : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [mod_ ?out a b] is the element-wise remainder of [a / b]. [out] defaults to
    a fresh allocation. *)

val mod_s : ?out:('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [mod_s ?out t s] is the remainder of each element divided by scalar [s].
    [out] defaults to a fresh allocation. *)

val rmod_s : ?out:('a, 'b) t -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rmod_s ?out s t] is [s mod t] element-wise. [out] defaults to a fresh
    allocation. *)

val neg : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [neg ?out t] is the element-wise negation of [t]. [out] defaults to a fresh
    allocation. *)

val conjugate : ('a, 'b) t -> ('a, 'b) t
(** [conjugate t] is the complex conjugate of [t]. For complex dtypes, negates
    the imaginary part. For real dtypes, returns [t] unchanged. *)

(** {1:math Mathematical functions} *)

(** {2:math_basic Basic} *)

val abs : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [abs ?out t] is the element-wise absolute value. [out] defaults to a fresh
    allocation. *)

val sign : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sign ?out t] is [-1], [0], or [1] according to the sign of each element.
    For unsigned types, returns [1] for non-zero, [0] for zero. [out] defaults
    to a fresh allocation.

    {@ocaml[
      # create float32 [| 3 |] [| -2.; 0.; 3.5 |]
        |> sign
      - : (float, float32_elt) t = [-1, 0, 1]
    ]} *)

val square : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [square ?out t] is the element-wise square. [out] defaults to a fresh
    allocation. *)

val sqrt : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sqrt ?out t] is the element-wise square root. [out] defaults to a fresh
    allocation. *)

val rsqrt : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [rsqrt ?out t] is the element-wise reciprocal square root ([1 / sqrt t]).
    [out] defaults to a fresh allocation. *)

val recip : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [recip ?out t] is the element-wise reciprocal ([1 / t]). [out] defaults to a
    fresh allocation. *)

(** {2:math_exp Exponential and logarithmic} *)

val log : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [log ?out t] is the element-wise natural logarithm. [out] defaults to a
    fresh allocation. *)

val log2 : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [log2 ?out t] is the element-wise base-2 logarithm. [out] defaults to a
    fresh allocation. *)

val exp : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [exp ?out t] is the element-wise exponential. [out] defaults to a fresh
    allocation. *)

val exp2 : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [exp2 ?out t] is [2{^t}] element-wise. [out] defaults to a fresh allocation.
*)

(** {2:math_trig Trigonometric} *)

val sin : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sin ?out t] is the element-wise sine. [out] defaults to a fresh allocation.
*)

val cos : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [cos ?out t] is the element-wise cosine. [out] defaults to a fresh
    allocation. *)

val tan : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [tan ?out t] is the element-wise tangent. [out] defaults to a fresh
    allocation. *)

val asin : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [asin ?out t] is the element-wise arcsine. [out] defaults to a fresh
    allocation. *)

val acos : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [acos ?out t] is the element-wise arccosine. [out] defaults to a fresh
    allocation. *)

val atan : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [atan ?out t] is the element-wise arctangent. [out] defaults to a fresh
    allocation. *)

val atan2 : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [atan2 ?out y x] is the element-wise two-argument arctangent, returning
    angles in \[[-π], [π]\]. [out] defaults to a fresh allocation. *)

(** {2:math_hyp Hyperbolic} *)

val sinh : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sinh ?out t] is the element-wise hyperbolic sine. [out] defaults to a fresh
    allocation. *)

val cosh : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [cosh ?out t] is the element-wise hyperbolic cosine. [out] defaults to a
    fresh allocation. *)

val tanh : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [tanh ?out t] is the element-wise hyperbolic tangent. [out] defaults to a
    fresh allocation. *)

val asinh : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [asinh ?out t] is the element-wise inverse hyperbolic sine. [out] defaults
    to a fresh allocation. *)

val acosh : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [acosh ?out t] is the element-wise inverse hyperbolic cosine. [out] defaults
    to a fresh allocation. *)

val atanh : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [atanh ?out t] is the element-wise inverse hyperbolic tangent. [out]
    defaults to a fresh allocation. *)

(** {2:math_round Rounding} *)

val trunc : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [trunc ?out t] rounds each element toward zero. [out] defaults to a fresh
    allocation. *)

val ceil : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [ceil ?out t] rounds each element toward positive infinity. [out] defaults
    to a fresh allocation. *)

val floor : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [floor ?out t] rounds each element toward negative infinity. [out] defaults
    to a fresh allocation. *)

val round : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [round ?out t] rounds each element to the nearest integer. Ties round away
    from zero (not banker's rounding). [out] defaults to a fresh allocation.

    {@ocaml[
      # create float32 [| 4 |] [| 2.5; 3.5; -2.5; -3.5 |]
        |> round
      - : (float, float32_elt) t = [3, 4, -3, -4]
    ]} *)

(** {2:math_misc Other} *)

val hypot : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [hypot ?out x y] is [sqrt(x² + y²)] computed without intermediate overflow.
    [out] defaults to a fresh allocation.

    {@ocaml[
      # hypot (scalar float32 3.) (scalar float32 4.)
        |> item []
      - : float = 5.
    ]} *)

val lerp :
  ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [lerp ?out a b w] is the linear interpolation [a + w * (b - a)]. [w] is
    typically in \[[0], [1]\]. [out] defaults to a fresh allocation.

    {@ocaml[
      # let a = create float32 [| 2 |] [| 1.; 2. |] in
        let b = create float32 [| 2 |] [| 5.; 8. |] in
        lerp a b (scalar float32 0.25)
      - : (float, float32_elt) t = [2, 3.5]
    ]} *)

val lerp_scalar_weight :
  ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [lerp_scalar_weight ?out a b w] is like {!lerp} with a scalar weight. [out]
    defaults to a fresh allocation. *)

val isinf : ?out:(bool, bool_elt) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [isinf ?out t] is [true] where [t] is positive or negative infinity, [false]
    elsewhere. Non-float dtypes always return all [false]. [out] defaults to a
    fresh allocation.

    {@ocaml[
      # create float32 [| 4 |]
          [| 1.; Float.infinity;
             Float.neg_infinity; Float.nan |]
        |> isinf
      - : (bool, bool_elt) t = [false, true, true, false]
    ]}

    See also {!isnan}, {!isfinite}. *)

val isnan : ?out:(bool, bool_elt) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [isnan ?out t] is [true] where [t] is NaN, [false] elsewhere. Non-float
    dtypes always return all [false]. [out] defaults to a fresh allocation.

    See also {!isinf}, {!isfinite}. *)

val isfinite : ?out:(bool, bool_elt) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [isfinite ?out t] is [true] where [t] is neither infinite nor NaN. Non-float
    dtypes always return all [true]. [out] defaults to a fresh allocation.

    See also {!isinf}, {!isnan}. *)

(** {1:comparison Comparison and logic} *)

val cmplt :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [cmplt ?out a b] is [true] where [a < b], [false] elsewhere. [out] defaults
    to a fresh allocation. *)

val less :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [less a b] is {!cmplt}. *)

val less_s : ?out:(bool, bool_elt) t -> ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [less_s ?out t s] is [true] where [t < s]. [out] defaults to a fresh
    allocation. *)

val cmpne :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [cmpne ?out a b] is [true] where [a ≠ b], [false] elsewhere. [out] defaults
    to a fresh allocation. *)

val not_equal :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [not_equal a b] is {!cmpne}. *)

val not_equal_s :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [not_equal_s ?out t s] is [true] where [t ≠ s]. [out] defaults to a fresh
    allocation. *)

val cmpeq :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [cmpeq ?out a b] is [true] where [a = b], [false] elsewhere. [out] defaults
    to a fresh allocation. *)

val equal :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [equal a b] is {!cmpeq}. *)

val equal_s : ?out:(bool, bool_elt) t -> ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [equal_s ?out t s] is [true] where [t = s]. [out] defaults to a fresh
    allocation. *)

val cmpgt :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [cmpgt ?out a b] is [true] where [a > b], [false] elsewhere. [out] defaults
    to a fresh allocation. *)

val greater :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [greater a b] is {!cmpgt}. *)

val greater_s :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [greater_s ?out t s] is [true] where [t > s]. [out] defaults to a fresh
    allocation. *)

val cmple :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [cmple ?out a b] is [true] where [a ≤ b], [false] elsewhere. [out] defaults
    to a fresh allocation. *)

val less_equal :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [less_equal a b] is {!cmple}. *)

val less_equal_s :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [less_equal_s ?out t s] is [true] where [t ≤ s]. [out] defaults to a fresh
    allocation. *)

val cmpge :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [cmpge ?out a b] is [true] where [a ≥ b], [false] elsewhere. [out] defaults
    to a fresh allocation. *)

val greater_equal :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [greater_equal a b] is {!cmpge}. *)

val greater_equal_s :
  ?out:(bool, bool_elt) t -> ('a, 'b) t -> 'a -> (bool, bool_elt) t
(** [greater_equal_s ?out t s] is [true] where [t ≥ s]. [out] defaults to a
    fresh allocation. *)

val array_equal : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
(** [array_equal a b] is a scalar [true] iff all elements of [a] and [b] are
    equal. Returns [false] if shapes differ.

    {@ocaml[
      # let a = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        let b = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        array_equal a b |> item []
      - : bool = true
    ]} *)

val maximum : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [maximum ?out a b] is the element-wise maximum of [a] and [b]. [out]
    defaults to a fresh allocation. *)

val maximum_s : ?out:('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [maximum_s ?out t s] is the element-wise maximum of [t] and scalar [s].
    [out] defaults to a fresh allocation. *)

val rmaximum_s : ?out:('a, 'b) t -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rmaximum_s ?out s t] is [maximum_s ?out t s]. *)

val minimum : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [minimum ?out a b] is the element-wise minimum of [a] and [b]. [out]
    defaults to a fresh allocation. *)

val minimum_s : ?out:('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [minimum_s ?out t s] is the element-wise minimum of [t] and scalar [s].
    [out] defaults to a fresh allocation. *)

val rminimum_s : ?out:('a, 'b) t -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rminimum_s ?out s t] is [minimum_s ?out t s]. *)

val logical_and : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_and ?out a b] is the element-wise logical AND. Non-zero is [true].
    [out] defaults to a fresh allocation. *)

val logical_or : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_or ?out a b] is the element-wise logical OR. [out] defaults to a
    fresh allocation. *)

val logical_xor : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_xor ?out a b] is the element-wise logical XOR. [out] defaults to a
    fresh allocation. *)

val logical_not : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_not ?out t] is the element-wise logical NOT: non-zero becomes [0],
    zero becomes [1]. [out] defaults to a fresh allocation. *)

val where :
  ?out:('a, 'b) t ->
  (bool, bool_elt) t ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [where ?out cond if_true if_false] selects elements from [if_true] where
    [cond] is [true] and from [if_false] elsewhere. All three inputs broadcast
    to a common shape. [out] defaults to a fresh allocation.

    {@ocaml[
      # let x =
          create float32 [| 4 |] [| -1.; 2.; -3.; 4. |]
        in
        where
          (cmpgt x (scalar float32 0.))
          x (scalar float32 0.)
      - : (float, float32_elt) t = [0, 2, 0, 4]
    ]} *)

val clamp : ?out:('a, 'b) t -> ?min:'a -> ?max:'a -> ('a, 'b) t -> ('a, 'b) t
(** [clamp ?out ?min ?max t] clamps elements to \[[min], [max]\]. Either bound
    may be omitted. [out] defaults to a fresh allocation.

    See also {!clip}. *)

val clip : ?out:('a, 'b) t -> ?min:'a -> ?max:'a -> ('a, 'b) t -> ('a, 'b) t
(** [clip ?out ?min ?max t] is {!clamp}. *)

(** {1:bitwise Bitwise operations} *)

val bitwise_xor : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_xor ?out a b] is the element-wise bitwise XOR. [out] defaults to a
    fresh allocation. *)

val bitwise_or : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_or ?out a b] is the element-wise bitwise OR. [out] defaults to a
    fresh allocation. *)

val bitwise_and : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_and ?out a b] is the element-wise bitwise AND. [out] defaults to a
    fresh allocation. *)

val bitwise_not : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_not ?out t] is the element-wise bitwise NOT. [out] defaults to a
    fresh allocation. *)

val invert : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [invert ?out t] is {!bitwise_not}. *)

val lshift : ?out:('a, 'b) t -> ('a, 'b) t -> int -> ('a, 'b) t
(** [lshift ?out t n] left-shifts each element by [n] bits. [out] defaults to a
    fresh allocation.

    Raises [Invalid_argument] if [n] is negative or the dtype is not an integer
    type.

    {@ocaml[
      # create int32 [| 3 |] [| 1l; 2l; 3l |]
        |> Fun.flip lshift 2
      - : (int32, int32_elt) t = [4, 8, 12]
    ]}

    See also {!rshift}. *)

val rshift : ?out:('a, 'b) t -> ('a, 'b) t -> int -> ('a, 'b) t
(** [rshift ?out t n] right-shifts each element by [n] bits. [out] defaults to a
    fresh allocation.

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
  (** [a < b] is {!cmplt} [a b]. *)

  val ( <> ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a <> b] is {!cmpne} [a b]. *)

  val ( = ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a = b] is {!cmpeq} [a b]. *)

  val ( > ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a > b] is {!cmpgt} [a b]. *)

  val ( <= ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a <= b] is {!cmple} [a b]. *)

  val ( >= ) : ('a, 'b) t -> ('a, 'b) t -> (bool, bool_elt) t
  (** [a >= b] is {!cmpge} [a b]. *)

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

  val ( ^ ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a ^ b] is {!logical_xor} [a b]. *)

  val ( && ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a && b] is {!logical_and} [a b]. *)

  val ( || ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a || b] is {!logical_or} [a b]. *)

  val ( ~- ) : ('a, 'b) t -> ('a, 'b) t
  (** [~-t] is {!logical_not} [t]. *)

  (** {2:infix_linalg Linear algebra} *)

  val ( @@ ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a @@ b] is {!matmul} [a b]. *)

  val ( /@ ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a /@ b] is {!solve} [a b]. *)

  val ( **@ ) : ('a, 'b) t -> int -> ('a, 'b) t
  (** [t **@ n] is {!matrix_power} [t n]. *)

  val ( <.> ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a <.> b] is {!dot} [a b]. *)

  (** {2:infix_concat Concatenation} *)

  val ( @= ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a @= b] is {!vstack} [[a; b]]. *)

  val ( @|| ) : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [a @|| b] is {!hstack} [[a; b]]. *)

  (** {2:infix_index Indexing} *)

  val ( .%{} ) : ('a, 'b) t -> int list -> ('a, 'b) t
  (** [t.%\{i\}] is {!get} [i t]. *)

  val ( .%{}<- ) : ('a, 'b) t -> int list -> ('a, 'b) t -> unit
  (** [t.%\{i\} <- v] is {!set} [i t v]. *)

  val ( .${} ) : ('a, 'b) t -> index list -> ('a, 'b) t
  (** [t.$\{s\}] is {!val-slice} [s t]. *)

  val ( .${}<- ) : ('a, 'b) t -> index list -> ('a, 'b) t -> unit
  (** [t.$\{s\} <- v] is {!set_slice} [s t v]. *)
end

(** {1:reduction Reductions} *)

val sum :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [sum ?out ?axes ?keepdims t] sums elements along [axes]. When [axes] is
    omitted, reduces all axes (returns a scalar). When [keepdims] is [true],
    reduced axes are kept with size 1. [keepdims] defaults to [false]. Negative
    axes count from the end. [out] defaults to a fresh allocation.

    {@ocaml[
      # create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
        |> sum |> item []
      - : float = 10.
      # create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]
        |> sum ~axes:[ 0 ]
      - : (float, float32_elt) t = [4, 6]
      # create float32 [| 1; 2 |] [| 1.; 2. |]
        |> sum ~axes:[ 1 ] ~keepdims:true
      - : (float, float32_elt) t = float32 [1; 1] [[3]]
    ]} *)

val max :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [max ?out ?axes ?keepdims t] is the maximum along [axes]. NaN propagates.
    [keepdims] defaults to [false]. [out] defaults to a fresh allocation.

    {@ocaml[
      # create float32 [| 2; 3 |]
          [| 1.; 2.; 3.; 4.; 5.; 6. |]
        |> max |> item []
      - : float = 6.
    ]} *)

val min :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [min ?out ?axes ?keepdims t] is the minimum along [axes]. NaN propagates.
    [keepdims] defaults to [false]. [out] defaults to a fresh allocation. *)

val prod :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [prod ?out ?axes ?keepdims t] is the product along [axes]. [keepdims]
    defaults to [false]. [out] defaults to a fresh allocation.

    {@ocaml[
      # create int32 [| 3 |] [| 2l; 3l; 4l |]
        |> prod |> item []
      - : int32 = 24l
    ]} *)

val cumsum : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [cumsum ?axis t] is the inclusive cumulative sum along [axis]. When [axis]
    is omitted, operates on the flattened tensor.

    See also {!cumprod}. *)

val cumprod : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [cumprod ?axis t] is the inclusive cumulative product along [axis]. When
    [axis] is omitted, operates on the flattened tensor.

    See also {!cumsum}. *)

val cummax : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [cummax ?axis t] is the inclusive cumulative maximum along [axis]. NaN
    propagates for floating-point dtypes. When [axis] is omitted, operates on
    the flattened tensor.

    See also {!cummin}. *)

val cummin : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [cummin ?axis t] is the inclusive cumulative minimum along [axis]. NaN
    propagates for floating-point dtypes. When [axis] is omitted, operates on
    the flattened tensor.

    See also {!cummax}. *)

val mean :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [mean ?out ?axes ?keepdims t] is the arithmetic mean along [axes]. NaN
    propagates. [keepdims] defaults to [false]. [out] defaults to a fresh
    allocation.

    {@ocaml[
      # create float32 [| 4 |] [| 1.; 2.; 3.; 4. |]
        |> mean |> item []
      - : float = 2.5
    ]} *)

val var :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ?ddof:int ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [var ?out ?axes ?keepdims ?ddof t] is the variance along [axes]. [ddof]
    (delta degrees of freedom) defaults to [0] (population variance); use [1]
    for sample variance. Computed as [E[(X - E[X])²] / (N - ddof)]. [keepdims]
    defaults to [false]. [out] defaults to a fresh allocation.

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
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ?ddof:int ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [std ?out ?axes ?keepdims ?ddof t] is the standard deviation:
    [sqrt({!var} ~ddof t)]. [ddof] defaults to [0]. [keepdims] defaults to
    [false]. [out] defaults to a fresh allocation.

    See also {!var}. *)

val all :
  ?out:(bool, bool_elt) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  (bool, bool_elt) t
(** [all ?out ?axes ?keepdims t] is [true] iff every element along [axes] is
    non-zero. [keepdims] defaults to [false]. [out] defaults to a fresh
    allocation.

    {@ocaml[
      # create int32 [| 3 |] [| 1l; 2l; 3l |]
        |> all |> item []
      - : bool = true
      # create int32 [| 3 |] [| 1l; 0l; 3l |]
        |> all |> item []
      - : bool = false
    ]}

    See also {!any}. *)

val any :
  ?out:(bool, bool_elt) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  (bool, bool_elt) t
(** [any ?out ?axes ?keepdims t] is [true] iff at least one element along [axes]
    is non-zero. [keepdims] defaults to [false]. [out] defaults to a fresh
    allocation.

    See also {!all}. *)

val argmax : ?axis:int -> ?keepdims:bool -> ('a, 'b) t -> (int32, int32_elt) t
(** [argmax ?axis ?keepdims t] is the index of the maximum along [axis]. Returns
    the first occurrence for ties. When [axis] is omitted, operates on the
    flattened tensor. [keepdims] defaults to [false].

    Raises [Invalid_argument] if [axis] is out of bounds.

    {@ocaml[
      # create int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |]
        |> argmax |> item []
      - : int32 = 4l
    ]}

    See also {!argmin}. *)

val argmin : ?axis:int -> ?keepdims:bool -> ('a, 'b) t -> (int32, int32_elt) t
(** [argmin ?axis ?keepdims t] is the index of the minimum along [axis]. Returns
    the first occurrence for ties. When [axis] is omitted, operates on the
    flattened tensor. [keepdims] defaults to [false].

    Raises [Invalid_argument] if [axis] is out of bounds.

    See also {!argmax}. *)

(** {1:sorting Sorting and searching} *)

val sort :
  ?descending:bool ->
  ?axis:int ->
  ('a, 'b) t ->
  ('a, 'b) t * (int32, int32_elt) t
(** [sort ?descending ?axis t] sorts elements along [axis] and returns
    [(sorted, indices)] where [indices] maps sorted positions back to originals.
    [descending] defaults to [false]. [axis] defaults to [-1] (last).

    The sort is stable (equal elements preserve their relative order). NaN sorts
    to the end in ascending order and to the beginning in descending order.

    Raises [Invalid_argument] if [axis] is out of bounds.

    {@ocaml[
      # create int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |]
        |> sort
      - : (int32, int32_elt) t * (int32, int32_elt) t =
      (int32 [5] [1, 1, ..., 4, 5], int32 [5] [1, 3, ..., 2, 4])
    ]}

    See also {!argsort}. *)

val argsort :
  ?descending:bool -> ?axis:int -> ('a, 'b) t -> (int32, int32_elt) t
(** [argsort ?descending ?axis t] is [snd (sort ?descending ?axis t)].

    See also {!sort}. *)

(** {1:linalg Linear algebra} *)

(** {2:linalg_products Products} *)

val dot : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [dot ?out a b] is the generalised dot product. [out] defaults to a fresh
    allocation.

    Contracts the last axis of [a] with:
    - the only axis of [b] when [b] is 1-D,
    - the second-to-last axis of [b] otherwise.

    Dimension rules:
    - 1-D × 1-D → scalar (inner product).
    - 2-D × 2-D → matrix multiplication.
    - N-D × M-D → contraction; output axes are the non-contracted axes of [a]
      followed by those of [b].

    {b Note.} Unlike {!matmul}, [dot] does {e not} broadcast batch dimensions—it
    concatenates them.

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

val matmul : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [matmul ?out a b] is the matrix product of [a] and [b] with batch
    broadcasting. [out] defaults to a fresh allocation; ignored when either
    input is 1-D.

    Dimension rules:
    - 1-D × 1-D → scalar (inner product).
    - 1-D × N-D → [a] is treated as a row vector.
    - N-D × 1-D → [b] is treated as a column vector.
    - N-D × M-D → matrix multiply on last two axes; leading axes are broadcast.

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
    complex dtypes, [a] is conjugated first. Always returns a scalar.

    Raises [Invalid_argument] if the inputs have different numbers of elements.

    See also {!dot}, {!vecdot}. *)

val vecdot : ?axis:int -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [vecdot ?axis a b] is the dot product of [a] and [b] along [axis] with
    broadcasting. [axis] defaults to [-1].

    Raises [Invalid_argument] if the specified axis dimensions differ.

    See also {!vdot}, {!dot}. *)

val inner : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [inner a b] is the inner product over the last axes of [a] and [b].

    Raises [Invalid_argument] if the last dimensions differ.

    See also {!dot}, {!outer}. *)

val outer : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [outer ?out a b] is the outer product. Inputs are flattened to 1-D; the
    result has shape [[numel a; numel b]]. [out] defaults to a fresh allocation.

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

    Raises [Invalid_argument] if [t] is not square, the dtype is not
    floating-point or complex, or [n < 0] and [t] is singular. *)

val cross :
  ?out:('a, 'b) t -> ?axis:int -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [cross ?out ?axis a b] is the cross product of 3-element vectors along
    [axis]. [axis] defaults to [-1]. [out] defaults to a fresh allocation.

    Raises [Invalid_argument] if the axis dimension is not 3. *)

(** {2:linalg_decomp Decompositions} *)

val cholesky : ?upper:bool -> ('a, 'b) t -> ('a, 'b) t
(** [cholesky ?upper a] is the Cholesky factor of positive- definite matrix [a].
    When [upper] is [true], returns the upper-triangular factor [U] such that
    [a = Uᵀ U]; otherwise (default) returns the lower-triangular factor [L] such
    that [a = L Lᵀ].

    Raises [Invalid_argument] if [a] is not positive-definite or the dtype is
    not floating-point or complex.

    See also {!solve}. *)

val qr : ?mode:[ `Complete | `Reduced ] -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
(** [qr ?mode a] is [(Q, R)] where [a = Q R], [Q] is orthogonal, and [R] is
    upper-triangular. [mode] defaults to [`Reduced].

    Raises [Invalid_argument] if the dtype is not floating-point or complex.

    See also {!svd}. *)

val svd :
  ?full_matrices:bool ->
  ('a, 'b) t ->
  ('a, 'b) t * (float, float64_elt) t * ('a, 'b) t
(** [svd ?full_matrices a] is [(U, S, Vh)] where [a = U diag(S) Vh]. [S]
    contains the singular values in descending order. [full_matrices] defaults
    to [false] (economy decomposition).

    Raises [Invalid_argument] if the dtype is not floating-point or complex.

    See also {!svdvals}, {!qr}. *)

val svdvals : ('a, 'b) t -> (float, float64_elt) t
(** [svdvals a] is the singular values of [a] in descending order. More
    efficient than {!svd} when only the values are needed.

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
(** [eigh ?uplo a] is [(eigenvalues, eigenvectors)] of symmetric / Hermitian
    matrix [a] in ascending eigenvalue order. [uplo] defaults to [`L]. More
    efficient than {!eig} for symmetric matrices.

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
(** [eigvalsh ?uplo a] is the eigenvalues of symmetric / Hermitian matrix [a] in
    ascending order. [uplo] defaults to [`L].

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
(** [det a] is the determinant of square matrix [a].

    Raises [Invalid_argument] if [a] is not square or the dtype is not
    floating-point or complex. *)

val slogdet : ('a, 'b) t -> (float, float32_elt) t * (float, float32_elt) t
(** [slogdet a] is [(sign, log_abs_det)] where
    [det a = sign * exp(log_abs_det)]. More numerically stable than {!det} for
    matrices with very large or small determinants.

    Raises [Invalid_argument] if [a] is not square or the dtype is not
    floating-point or complex. *)

val matrix_rank :
  ?tol:float -> ?rtol:float -> ?hermitian:bool -> ('a, 'b) t -> int
(** [matrix_rank ?tol ?rtol ?hermitian a] is the rank of [a], counting singular
    values above the tolerance. [rtol] defaults to [max(M, N) * ε * σ_max]. When
    [hermitian] is [true] (default [false]), uses a more efficient
    eigenvalue-based algorithm.

    Raises [Invalid_argument] if the dtype is not floating-point or complex. *)

val trace : ?out:('a, 'b) t -> ?offset:int -> ('a, 'b) t -> ('a, 'b) t
(** [trace ?out ?offset t] is the sum along the [offset]-th diagonal. [offset]
    defaults to [0]. [out] defaults to a fresh allocation.

    Raises [Invalid_argument] if [t] has fewer than 2 dimensions.

    See also {!diagonal}. *)

(** {2:linalg_solve Solving} *)

val solve : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [solve a b] is [x] such that [a @@ x = b].

    Raises [Invalid_argument] if [a] is singular or the dtype is not
    floating-point or complex.

    See also {!lstsq}, {!inv}. *)

val lstsq :
  ?rcond:float ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t * ('a, 'b) t * int * (float, float64_elt) t
(** [lstsq ?rcond a b] is [(x, residuals, rank, sv)] — the least-squares
    solution to [a @@ x ≈ b]. [rcond] defaults to machine precision.

    Raises [Invalid_argument] if the dtype is not floating-point or complex.

    See also {!solve}. *)

val inv : ('a, 'b) t -> ('a, 'b) t
(** [inv a] is the inverse of square matrix [a].

    Raises [Invalid_argument] if [a] is singular, not square, or the dtype is
    not floating-point or complex.

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

(** {1:fft Fourier transforms} *)

type fft_norm = [ `Backward | `Forward | `Ortho ]
(** FFT normalisation mode.
    - [`Backward] — normalise by [1/n] on the inverse (default).
    - [`Forward] — normalise by [1/n] on the forward.
    - [`Ortho] — normalise by [1/√n] on both. *)

val fft :
  ?out:(Complex.t, 'a) t ->
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [fft ?out ?axis ?n ?norm x] is the 1-D discrete Fourier transform along
    [axis]. [axis] defaults to [-1]. [n] truncates or zero-pads the input.
    [norm] defaults to [`Backward]. [out] defaults to a fresh allocation.

    See also {!ifft}, {!rfft}. *)

val ifft :
  ?out:(Complex.t, 'a) t ->
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [ifft ?out ?axis ?n ?norm x] is the inverse of {!fft}. [out] defaults to a
    fresh allocation.

    See also {!fft}, {!irfft}. *)

val fft2 :
  ?out:(Complex.t, 'a) t ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [fft2 ?out ?axes ?s ?norm x] is the 2-D FFT. [axes] defaults to the last
    two. [out] defaults to a fresh allocation.

    Raises [Invalid_argument] if the input has fewer than 2 dimensions.

    See also {!ifft2}, {!fft}. *)

val ifft2 :
  ?out:(Complex.t, 'a) t ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [ifft2 ?out ?axes ?s ?norm x] is the inverse of {!fft2}. [out] defaults to a
    fresh allocation. *)

val fftn :
  ?out:(Complex.t, 'a) t ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [fftn ?out ?axes ?s ?norm x] is the N-D FFT. [axes] defaults to all. [out]
    defaults to a fresh allocation.

    See also {!ifftn}. *)

val ifftn :
  ?out:(Complex.t, 'a) t ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (Complex.t, 'a) t
(** [ifftn ?out ?axes ?s ?norm x] is the inverse of {!fftn}. [out] defaults to a
    fresh allocation. *)

val rfft :
  ?out:(Complex.t, complex64_elt) t ->
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (Complex.t, complex64_elt) t
(** [rfft ?out ?axis ?n ?norm x] is the 1-D FFT of real input. Returns only the
    non-redundant positive frequencies; the output size along the transformed
    axis is [n/2 + 1]. [out] defaults to a fresh allocation.

    {@ocaml[
      # create float64 [| 4 |] [| 0.; 1.; 2.; 3. |]
        |> rfft |> shape
      - : int array = [|3|]
    ]}

    See also {!irfft}, {!fft}. *)

val irfft :
  ?out:(float, float64_elt) t ->
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (float, float64_elt) t
(** [irfft ?out ?axis ?n ?norm x] is the inverse of {!rfft}, producing real
    output. Assumes Hermitian symmetry. [out] defaults to a fresh allocation.

    See also {!rfft}. *)

val rfft2 :
  ?out:(Complex.t, complex64_elt) t ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (Complex.t, complex64_elt) t
(** [rfft2 ?out ?axes ?s ?norm x] is the 2-D FFT of real input. [out] defaults
    to a fresh allocation.

    See also {!irfft2}, {!rfft}. *)

val irfft2 :
  ?out:(float, float64_elt) t ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (float, float64_elt) t
(** [irfft2 ?out ?axes ?s ?norm x] is the inverse of {!rfft2}. [out] defaults to
    a fresh allocation. *)

val rfftn :
  ?out:(Complex.t, complex64_elt) t ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (Complex.t, complex64_elt) t
(** [rfftn ?out ?axes ?s ?norm x] is the N-D FFT of real input. [out] defaults
    to a fresh allocation.

    See also {!irfftn}, {!rfft}. *)

val irfftn :
  ?out:(float, float64_elt) t ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (float, float64_elt) t
(** [irfftn ?out ?axes ?s ?norm x] is the inverse of {!rfftn}. [out] defaults to
    a fresh allocation. *)

val hfft :
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a) t ->
  (float, float64_elt) t
(** [hfft ?axis ?n ?norm x] is the FFT of a signal with Hermitian symmetry,
    producing real output. *)

val ihfft :
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (float, 'a) t ->
  (Complex.t, complex64_elt) t
(** [ihfft ?axis ?n ?norm x] is the inverse of {!hfft}. *)

val fftfreq : ?d:float -> int -> (float, float64_elt) t
(** [fftfreq ?d n] is the DFT sample frequencies for window length [n] and
    sample spacing [d] (default [1.0]).

    {@ocaml[
      # fftfreq 4
      - : (float, float64_elt) t = [0, 0.25, -0.5, -0.25]
    ]}

    See also {!rfftfreq}. *)

val rfftfreq : ?d:float -> int -> (float, float64_elt) t
(** [rfftfreq ?d n] is the positive DFT sample frequencies:
    [[0, 1, …, n/2] / (d * n)].

    See also {!fftfreq}. *)

val fftshift : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [fftshift ?axes t] shifts the zero-frequency component to the centre. [axes]
    defaults to all.

    {@ocaml[
      # fftfreq 5 |> fftshift
      - : (float, float64_elt) t = float64 [5] [-0.4, -0.2, ..., 0.2, 0.4]
    ]}

    See also {!ifftshift}. *)

val ifftshift : ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
(** [ifftshift ?axes t] is the inverse of {!fftshift}. *)

(** {1:activation Activation functions} *)

val relu : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [relu ?out t] is [max(0, t)] element-wise. [out] defaults to a fresh
    allocation.

    {@ocaml[
      # create float32 [| 5 |]
          [| -2.; -1.; 0.; 1.; 2. |]
        |> relu
      - : (float, float32_elt) t = float32 [5] [0, 0, ..., 1, 2]
    ]} *)

val sigmoid : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sigmoid ?out t] is [1 / (1 + exp(-t))] element-wise. Output in [(0, 1)].
    [out] defaults to a fresh allocation.

    {@ocaml[
      # sigmoid (scalar float32 0.) |> item []
      - : float = 0.5
    ]} *)

val softmax :
  ?out:('a, 'b) t -> ?axes:int list -> ?scale:float -> ('a, 'b) t -> ('a, 'b) t
(** [softmax ?out ?axes ?scale t] is the softmax normalisation
    [exp(scale * (t - max t)) / Σ exp(scale * (t - max t))]. [axes] defaults to
    [[-1]]. [scale] defaults to [1.0]. Output sums to [1] along the specified
    axes. [out] defaults to a fresh allocation.

    {@ocaml[
      # create float32 [| 3 |] [| 1.; 2.; 3. |]
        |> softmax |> sum |> item []
      - : float = 1.
    ]}

    See also {!log_softmax}. *)

val log_softmax :
  ?out:('a, 'b) t -> ?axes:int list -> ?scale:float -> ('a, 'b) t -> ('a, 'b) t
(** [log_softmax ?out ?axes ?scale t] is the natural logarithm of {!softmax}.
    Same defaults as {!softmax}. [out] defaults to a fresh allocation.

    See also {!softmax}, {!logsumexp}. *)

val logsumexp :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [logsumexp ?out ?axes ?keepdims t] is [log(Σ exp(t))] computed in a
    numerically stable way. [axes] defaults to all. [keepdims] defaults to
    [false]. [out] defaults to a fresh allocation.

    See also {!logmeanexp}, {!log_softmax}. *)

val logmeanexp :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [logmeanexp ?out ?axes ?keepdims t] is [log(mean(exp(t)))]: {!logsumexp}
    minus [log N]. [axes] defaults to all. [keepdims] defaults to [false]. [out]
    defaults to a fresh allocation.

    See also {!logsumexp}. *)

val standardize :
  ?out:('a, 'b) t ->
  ?axes:int list ->
  ?mean:('a, 'b) t ->
  ?variance:('a, 'b) t ->
  ?epsilon:float ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [standardize ?out ?axes ?mean ?variance ?epsilon t] is
    [(t - mean) / sqrt(variance + epsilon)]. When [mean] or [variance] are
    omitted, they are computed along [axes] (default all). [epsilon] defaults to
    [1e-5]. [out] defaults to a fresh allocation. *)

val erf : ?out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [erf ?out t] is the error function [erf(x) = (2/√π) ∫₀ˣ e^{-u²} du]. [out]
    defaults to a fresh allocation.

    {@ocaml[
      # erf (scalar float32 0.) |> item []
      - : float = 0.
    ]} *)

(** {1:windows Sliding windows} *)

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

    Input: [[leading…; spatial…]]. Output: [[leading…; prod(kernel_size); L]].

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
    the inverse of {!extract_patches}. Overlapping values are summed.

    See also {!extract_patches}. *)

(** {2:correlate Cross-correlation and convolution} *)

val correlate :
  ?padding:[ `Full | `Same | `Valid ] -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [correlate ?padding x kernel] is the N-D cross-correlation (no kernel flip).
    Spatial dimensions [K = ndim kernel]. Leading dimensions of [x] beyond [K]
    are batch dimensions. [padding] defaults to [`Valid].

    See also {!convolve}. *)

val convolve :
  ?padding:[ `Full | `Same | `Valid ] -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [convolve ?padding x kernel] is like {!correlate} but flips the kernel along
    all spatial axes before correlating.

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

val map : (('a, 'b) t -> ('a, 'b) t) -> ('a, 'b) t -> ('a, 'b) t
(** [map f t] applies tensor function [f] to each element of [t], presented as a
    scalar tensor.

    See also {!map_item}. *)

val iter : (('a, 'b) t -> unit) -> ('a, 'b) t -> unit
(** [iter f t] applies tensor function [f] to each element of [t], presented as
    a scalar tensor.

    See also {!iter_item}. *)

val fold : ('a -> ('b, 'c) t -> 'a) -> 'a -> ('b, 'c) t -> 'a
(** [fold f init t] folds tensor function [f] over the elements of [t], each
    presented as a scalar tensor.

    See also {!fold_item}. *)

(** {1:pp Formatting} *)

val pp_data : Format.formatter -> ('a, 'b) t -> unit
(** [pp_data fmt t] formats the data of [t]. *)

val format_to_string : (Format.formatter -> 'a -> unit) -> 'a -> string
(** [format_to_string pp x] is the string produced by [pp]. *)

val print_with_formatter : (Format.formatter -> 'a -> unit) -> 'a -> unit
(** [print_with_formatter pp x] prints [x] to stdout using [pp]. *)

val data_to_string : ('a, 'b) t -> string
(** [data_to_string t] is the data of [t] as a string. *)

val print_data : ('a, 'b) t -> unit
(** [print_data t] prints the data of [t] to stdout. *)

val pp_dtype : Format.formatter -> ('a, 'b) dtype -> unit
(** [pp_dtype fmt dt] formats [dt]. *)

val dtype_to_string : ('a, 'b) dtype -> string
(** [dtype_to_string dt] is [dt] as a string. *)

val shape_to_string : int array -> string
(** [shape_to_string s] formats [s] as ["[2x3x4]"]. *)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape fmt s] formats shape [s]. *)

val pp : Format.formatter -> ('a, 'b) t -> unit
(** [pp fmt t] formats [t] for debugging (dtype, shape, and data). *)

val print : ('a, 'b) t -> unit
(** [print t] prints [t] to stdout. *)

val to_string : ('a, 'b) t -> string
(** [to_string t] is [t] formatted as a string (dtype, shape, and data). *)
