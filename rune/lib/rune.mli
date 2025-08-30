(** N-dimensional array operations for OCaml.

    This module provides NumPy-style tensor operations. Tensors are immutable
    views over mutable buffers, supporting broadcasting, slicing, and efficient
    memory layout transformations.

    {2 Type System}

    The type [('a, 'b, 'dev) t] represents a tensor where ['a] is the OCaml type
    of elements and ['b] is the bigarray element type. For example,
    [(float, float32_elt) t] is a tensor of 32-bit floats.

    {2 Broadcasting}

    Operations automatically broadcast compatible shapes: each dimension must be
    equal or one of them must be 1. Shape [|3; 1; 5|] broadcasts with
    [|1; 4; 5|] to [|3; 4; 5|].

    {2 Memory Layout}

    Tensors can be C-contiguous or strided. Operations return views when
    possible (O(1)), otherwise copy (O(n)). Use {!is_contiguous} to check layout
    and {!contiguous} to ensure contiguity. *)

(** {2 Type Definitions} *)

type ('a, 'b, 'dev) t
(** [('a, 'b, 'dev) t] is a tensor with OCaml type ['a] and bigarray type ['b]
    stored on device ['dev]. *)

type float16_elt = Bigarray_ext.float16_elt
type float32_elt = Bigarray_ext.float32_elt
type float64_elt = Bigarray_ext.float64_elt
type int8_elt = Bigarray_ext.int8_signed_elt
type uint8_elt = Bigarray_ext.int8_unsigned_elt
type int16_elt = Bigarray_ext.int16_signed_elt
type uint16_elt = Bigarray_ext.int16_unsigned_elt
type int32_elt = Bigarray_ext.int32_elt
type int64_elt = Bigarray_ext.int64_elt
type int_elt = Bigarray_ext.int_elt
type nativeint_elt = Bigarray_ext.nativeint_elt
type complex32_elt = Bigarray_ext.complex32_elt
type complex64_elt = Bigarray_ext.complex64_elt
type bfloat16_elt = Bigarray_ext.bfloat16_elt
type bool_elt = Bigarray_ext.bool_elt
type int4_elt = Bigarray_ext.int4_signed_elt
type uint4_elt = Bigarray_ext.int4_unsigned_elt
type float8_e4m3_elt = Bigarray_ext.float8_e4m3_elt
type float8_e5m2_elt = Bigarray_ext.float8_e5m2_elt
type complex16_elt = Bigarray_ext.complex16_elt
type qint8_elt = Bigarray_ext.qint8_elt
type quint8_elt = Bigarray_ext.quint8_elt

type ('a, 'b) dtype = ('a, 'b) Nx_core.Dtype.t =
  | Float16 : (float, float16_elt) dtype
  | Float32 : (float, float32_elt) dtype
  | Float64 : (float, float64_elt) dtype
  | Int8 : (int, int8_elt) dtype
  | UInt8 : (int, uint8_elt) dtype
  | Int16 : (int, int16_elt) dtype
  | UInt16 : (int, uint16_elt) dtype
  | Int32 : (int32, int32_elt) dtype
  | Int64 : (int64, int64_elt) dtype
  | Int : (int, int_elt) dtype
  | NativeInt : (nativeint, nativeint_elt) dtype
  | Complex32 : (Complex.t, complex32_elt) dtype
  | Complex64 : (Complex.t, complex64_elt) dtype
  | BFloat16 : (float, bfloat16_elt) dtype
  | Bool : (bool, bool_elt) dtype
  | Int4 : (int, int4_elt) dtype
  | UInt4 : (int, uint4_elt) dtype
  | Float8_e4m3 : (float, float8_e4m3_elt) dtype
  | Float8_e5m2 : (float, float8_e5m2_elt) dtype
  | Complex16 : (Complex.t, complex16_elt) dtype
  | QInt8 : (int, qint8_elt) dtype
  | QUInt8 : (int, quint8_elt) dtype
      (** Data type specification. Links OCaml types to bigarray element types.
      *)

type 'dev float16_t = (float, float16_elt, 'dev) t
type 'dev float32_t = (float, float32_elt, 'dev) t
type 'dev float64_t = (float, float64_elt, 'dev) t
type 'dev int8_t = (int, int8_elt, 'dev) t
type 'dev uint8_t = (int, uint8_elt, 'dev) t
type 'dev int16_t = (int, int16_elt, 'dev) t
type 'dev uint16_t = (int, uint16_elt, 'dev) t
type 'dev int32_t = (int32, int32_elt, 'dev) t
type 'dev int64_t = (int64, int64_elt, 'dev) t
type 'dev std_int_t = (int, int_elt, 'dev) t
type 'dev std_nativeint_t = (nativeint, nativeint_elt, 'dev) t
type 'dev complex32_t = (Complex.t, complex32_elt, 'dev) t
type 'dev complex64_t = (Complex.t, complex64_elt, 'dev) t

val float16 : (float, float16_elt) dtype
val float32 : (float, float32_elt) dtype
val float64 : (float, float64_elt) dtype
val int8 : (int, int8_elt) dtype
val uint8 : (int, uint8_elt) dtype
val int16 : (int, int16_elt) dtype
val uint16 : (int, uint16_elt) dtype
val int32 : (int32, int32_elt) dtype
val int64 : (int64, int64_elt) dtype
val int : (int, int_elt) dtype
val nativeint : (nativeint, nativeint_elt) dtype
val complex32 : (Complex.t, complex32_elt) dtype
val complex64 : (Complex.t, complex64_elt) dtype

(** {2 Device Management}

    Functions to manage tensor devices and contexts. *)

type 'a device

val ocaml : [ `ocaml ] device
(** [ocaml] represents CPU device with pure OCaml implementation with no FFI
    dependency. *)

val c : [ `c ] device
(** [c] represents CPU device with C FFI for performance. *)

val metal : unit -> [ `metal ] device
(** [metal ()] returns Metal device for GPU tensors.

    Requires Metal backend support. Use for GPU-accelerated computations on
    Apple devices. *)

val device : ('a, 'b, 'dev) t -> 'dev device
(** [device t] returns device where tensor [t] is stored.

    Returns [native] for CPU tensors, [metal] for Metal tensors, and [nctive_c]
    for CPU tensors with C FFI. *)

val is_device_available : [< `ocaml | `c | `metal ] -> bool
(** [is_device_available dev] checks if the specified device is available.

    Returns true if the device can be used for tensor operations. *)

(** {2 Array Properties}

    Functions to inspect array dimensions, memory layout, and data access. *)

val data :
  ('a, 'b, 'dev) t -> ('a, 'b, Bigarray_ext.c_layout) Bigarray_ext.Array1.t
(** [data t] returns underlying bigarray buffer.

    Buffer may contain data beyond tensor bounds for strided views. Direct
    access requires careful index computation using strides and offset.

    @raise Invalid_argument if tensor not C-contiguous when safety matters *)

val shape : ('a, 'b, 'dev) t -> int array
(** [shape t] returns dimensions. Empty array for scalars. *)

val dtype : ('a, 'b, 'dev) t -> ('a, 'b) dtype
(** [dtype t] returns data type. *)

val strides : ('a, 'b, 'dev) t -> int array
(** [strides t] returns byte strides for each dimension. *)

val stride : int -> ('a, 'b, 'dev) t -> int
(** [stride i t] returns byte stride for dimension [i].

    @raise Invalid_argument if [i] out of bounds *)

val dims : ('a, 'b, 'dev) t -> int array
(** [dims t] is synonym for {!shape}. *)

val dim : int -> ('a, 'b, 'dev) t -> int
(** [dim i t] returns size of dimension [i].

    @raise Invalid_argument if [i] out of bounds *)

val ndim : ('a, 'b, 'dev) t -> int
(** [ndim t] returns number of dimensions. *)

val itemsize : ('a, 'b, 'dev) t -> int
(** [itemsize t] returns bytes per element. *)

val size : ('a, 'b, 'dev) t -> int
(** [size t] returns total number of elements. *)

val numel : ('a, 'b, 'dev) t -> int
(** [numel t] is synonym for {!size}. *)

val nbytes : ('a, 'b, 'dev) t -> int
(** [nbytes t] returns [size t * itemsize t]. *)

val offset : ('a, 'b, 'dev) t -> int
(** [offset t] returns element offset in underlying buffer. *)

val is_c_contiguous : ('a, 'b, 'dev) t -> bool
(** [is_c_contiguous t] returns true if elements are contiguous in C order. *)

val to_bigarray :
  ('a, 'b, 'dev) t -> ('a, 'b, Bigarray_ext.c_layout) Bigarray_ext.Genarray.t
(** [to_bigarray t] converts to bigarray_ext.

    Always returns contiguous copy with same shape. Use for interop with
    libraries expecting bigarrays.

    {@ocaml[
      # let t = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
      val t : (float, float32_elt) t = [[1, 2, 3],
                                        [4, 5, 6]]
      # Bigarray_ext.Genarray.dims (to_bigarray t) = shape t
      - : bool = true
    ]} *)

val to_array : ('a, 'b, 'dev) t -> 'a array
(** [to_array t] converts to OCaml array.

    Flattens tensor to 1-D array in row-major (C) order. Always copies.

    {@ocaml[
      # let t = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |]
      val t : (int32, int32_elt, 'dev) t = [[1, 2],
                                      [3, 4]]
      # to_array t
      - : int32 array = [|1l; 2l; 3l; 4l|]
    ]} *)

(** {2 Array Creation}

    Functions to create and initialize arrays. *)

val create :
  'dev device -> ('a, 'b) dtype -> int array -> 'a array -> ('a, 'b, 'dev) t
(** [create dtype shape data] creates tensor from array [data].

    Length of [data] must equal product of [shape].

    @raise Invalid_argument if array size doesn't match shape

    {@ocaml[
      # create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
      - : (float, float32_elt) t = [[1, 2, 3],
                                    [4, 5, 6]]
    ]} *)

val init :
  'dev device ->
  ('a, 'b) dtype ->
  int array ->
  (int array -> 'a) ->
  ('a, 'b, 'dev) t
(** [init dtype shape f] creates tensor where element at indices [i] has value
    [f i].

    Function [f] receives array of indices for each position. Useful for
    creating position-dependent values.

    {@ocaml[
      # init int32 [| 2; 3 |] (fun i -> Int32.of_int (i.(0) + i.(1)))
      - : (int32, int32_elt, 'dev) t = [[0, 1, 2],
                                  [1, 2, 3]]

      # init float32 [| 3; 3 |] (fun i -> if i.(0) = i.(1) then 1. else 0.)
      - : (float, float32_elt) t = [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]]
    ]} *)

val empty : 'dev device -> ('a, 'b) dtype -> int array -> ('a, 'b, 'dev) t
(** [empty dtype shape] allocates uninitialized tensor. *)

val full : 'dev device -> ('a, 'b) dtype -> int array -> 'a -> ('a, 'b, 'dev) t
(** [full dtype shape value] creates tensor filled with [value].

    {@ocaml[
      # full float32 [| 2; 3 |] 3.14
      - : (float, float32_elt) t = [[3.14, 3.14, 3.14],
                                    [3.14, 3.14, 3.14]]
    ]} *)

val ones : 'dev device -> ('a, 'b) dtype -> int array -> ('a, 'b, 'dev) t
(** [ones dtype shape] creates tensor filled with ones. *)

val zeros : 'dev device -> ('a, 'b) dtype -> int array -> ('a, 'b, 'dev) t
(** [zeros dtype shape] creates tensor filled with zeros. *)

val scalar : 'dev device -> ('a, 'b) dtype -> 'a -> ('a, 'b, 'dev) t
(** [scalar dtype value] creates scalar tensor containing [value]. *)

val empty_like : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [empty_like t] creates uninitialized tensor with same shape and dtype as
    [t]. *)

val full_like : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [full_like t value] creates tensor shaped like [t] filled with [value]. *)

val ones_like : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [ones_like t] creates tensor shaped like [t] filled with ones. *)

val zeros_like : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [zeros_like t] creates tensor shaped like [t] filled with zeros. *)

val scalar_like : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [scalar_like t value] creates scalar with same dtype as [t]. *)

val eye :
  'dev device -> ?m:int -> ?k:int -> ('a, 'b) dtype -> int -> ('a, 'b, 'dev) t
(** [eye ?m ?k dtype n] creates matrix with ones on k-th diagonal.

    Default [m = n] (square), [k = 0] (main diagonal). Positive [k] shifts
    diagonal above main, negative below.

    {@ocaml[
      # eye int32 3
      - : (int32, int32_elt, 'dev) t = [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]
      # eye ~k:1 int32 3
      - : (int32, int32_elt, 'dev) t = [[0, 1, 0],
                                  [0, 0, 1],
                                  [0, 0, 0]]
      # eye ~m:2 ~k:(-1) int32 3
      - : (int32, int32_elt, 'dev) t = [[0, 0, 0],
                                  [1, 0, 0]]
    ]} *)

val identity : 'dev device -> ('a, 'b) dtype -> int -> ('a, 'b, 'dev) t
(** [identity dtype n] creates n×n identity matrix.

    Equivalent to [eye dtype n]. Square matrix with ones on main diagonal, zeros
    elsewhere.

    {@ocaml[
      # identity int32 3
      - : (int32, int32_elt, 'dev) t = [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]
    ]} *)

val arange :
  'dev device -> ('a, 'b) dtype -> int -> int -> int -> ('a, 'b, 'dev) t
(** [arange dtype start stop step] generates values from [start] to [\[stop)].

    Step must be non-zero. Result length is [(stop - start) / step] rounded
    toward zero.

    @raise Failure if [step = 0]

    {@ocaml[
      # arange int32 0 10 2
      - : (int32, int32_elt, 'dev) t = [0, 2, 4, 6, 8]
      # arange int32 5 0 (-1)
      - : (int32, int32_elt, 'dev) t = [5, 4, 3, 2, 1]
    ]} *)

val arange_f :
  'dev device ->
  (float, 'a) dtype ->
  float ->
  float ->
  float ->
  (float, 'a, 'dev) t
(** [arange_f dtype start stop step] generates float values from [start] to
    [\[stop)].

    Like {!arange} but for floating-point ranges. Handles fractional steps. Due
    to floating-point precision, final value may differ slightly from expected.

    @raise Failure if [step = 0.0]

    {@ocaml[
      # arange_f float32 0. 1. 0.2
      - : (float, float32_elt) t = [0, 0.2, 0.4, 0.6, 0.8]
      # arange_f float32 1. 0. (-0.25)
      - : (float, float32_elt) t = [1, 0.75, 0.5, 0.25]
    ]} *)

val linspace :
  'dev device ->
  ('a, 'b) dtype ->
  ?endpoint:bool ->
  float ->
  float ->
  int ->
  ('a, 'b, 'dev) t
(** [linspace dtype ?endpoint start stop count] generates [count] evenly spaced
    values from [start] to [stop].

    If [endpoint] is true (default), [stop] is included.

    {@ocaml[
      # linspace float32 ~endpoint:true 0. 10. 5
      - : (float, float32_elt) t = [0, 2.5, 5, 7.5, 10]
      # linspace float32 ~endpoint:false 0. 10. 5
      - : (float, float32_elt) t = [0, 2, 4, 6, 8]
    ]} *)

val logspace :
  'dev device ->
  (float, 'a) dtype ->
  ?endpoint:bool ->
  ?base:float ->
  float ->
  float ->
  int ->
  (float, 'a, 'dev) t
(** [logspace dtype ?endpoint ?base start_exp stop_exp count] generates values
    evenly spaced on log scale.

    Returns [base ** x] where x ranges from [start_exp] to [stop_exp]. Default
    [base = 10.0].

    {@ocaml[
      # logspace float32 0. 2. 3
      - : (float, float32_elt) t = [1, 10, 100]
      # logspace float32 ~base:2.0 0. 3. 4
      - : (float, float32_elt) t = [1, 2, 4, 8]
    ]} *)

val geomspace :
  'dev device ->
  (float, 'a) dtype ->
  ?endpoint:bool ->
  float ->
  float ->
  int ->
  (float, 'a, 'dev) t
(** [geomspace dtype ?endpoint start stop count] generates values evenly spaced
    on geometric (multiplicative) scale.

    @raise Invalid_argument if [start <= 0.] or [stop <= 0.]

    {@ocaml[
      # geomspace float32 1. 1000. 4
      - : (float, float32_elt) t = [1, 10, 100, 1000]
    ]} *)

val meshgrid :
  ?indexing:[ `xy | `ij ] ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * ('a, 'b, 'dev) t
(** [meshgrid ?indexing x y] creates coordinate grids from 1D arrays.

    Returns (X, Y) where X and Y are 2D arrays representing grid coordinates.

    - [`xy] (default): Cartesian indexing - X changes along columns, Y changes
      along rows
    - [`ij]: Matrix indexing - X changes along rows, Y changes along columns

    @raise Invalid_argument if x or y are not 1D

    {@ocaml[
      # let x = linspace float32 0. 2. 3 in
        let y = linspace float32 0. 1. 2 in
        meshgrid x y
      - : (float, float32_elt) t * (float, float32_elt) t =
      ([[0, 1, 2],
        [0, 1, 2]], [[0, 0, 0],
                     [1, 1, 1]])
    ]} *)

val tril : ?k:int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [tril ?k x] returns lower triangular part of matrix.

    Elements above the k-th diagonal are zeroed.
    - [k = 0] (default): main diagonal
    - [k > 0]: include k diagonals above main
    - [k < 0]: exclude |k| diagonals below main

    @raise Invalid_argument if x has less than 2 dimensions *)

val triu : ?k:int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [triu ?k x] returns upper triangular part of matrix.

    Elements below the k-th diagonal are zeroed.
    - [k = 0] (default): main diagonal
    - [k > 0]: exclude k diagonals above main
    - [k < 0]: include |k| diagonals below main

    @raise Invalid_argument if x has less than 2 dimensions *)

val of_bigarray :
  'dev device ->
  ('a, 'b, Bigarray_ext.c_layout) Bigarray_ext.Genarray.t ->
  ('a, 'b, 'dev) t
(** [of_bigarray ba] creates tensor from bigarray_ext.

    Zero-copy when bigarray is contiguous. Creates view sharing same memory.
    Modifications to either affect both.

    {@ocaml[
      # let ba = Bigarray_ext.Array2.create Float32 C_layout 2 3 in
        let t = of_bigarray (Bigarray_ext.genarray_of_array2 ba) in
        t
      - : (float, float32_elt) t = [[0, 0, 0],
                                    [0, 0, 0]]
    ]} *)

val of_nx : 'dev device -> ('a, 'b) Nx.t -> ('a, 'b, 'dev) t
(** [of_nx dev nx_tensor] creates a Rune tensor from an Nx tensor.

    Converts an Nx tensor to a Rune tensor on the specified device. The
    underlying bigarray is shared when possible (zero-copy).

    {@ocaml[
      # let nx_arr = Nx.randn Nx.float32 [| 2; 3 |] in
        let rune_arr = of_nx c nx_arr in
        rune_arr
      - : (float, float32_elt, [`c]) t
    ]} *)

val to_nx : ('a, 'b, 'dev) t -> ('a, 'b) Nx.t
(** [to_nx tensor] converts a Rune tensor to an Nx tensor.

    Converts a Rune tensor to an Nx tensor. The underlying bigarray is shared
    when possible (zero-copy).

    {@ocaml[
      # let rune_arr = randn c ~dtype:Float32 [| 2; 3 |] in
        let nx_arr = to_nx rune_arr in
        nx_arr
      - : (float, float32_elt) Nx.t
    ]} *)

(** {2 Random Number Generation}

    Functions to generate arrays with random values. *)

val rand :
  'dev device -> ('a, 'b) dtype -> ?seed:int -> int array -> ('a, 'b, 'dev) t
(** [rand dtype ?seed shape] generates uniform random values in [\[0, 1)].

    Only supports float dtypes. Same seed produces same sequence.

    @raise Invalid_argument if non-float dtype *)

val randn :
  'dev device -> ('a, 'b) dtype -> ?seed:int -> int array -> ('a, 'b, 'dev) t
(** [randn dtype ?seed shape] generates standard normal random values.

    Mean 0, variance 1. Uses Box-Muller transform for efficiency. Only supports
    float dtypes. Same seed produces same sequence.

    @raise Invalid_argument if non-float dtype *)

val randint :
  'dev device ->
  ('a, 'b) dtype ->
  ?seed:int ->
  ?high:int ->
  int array ->
  int ->
  ('a, 'b, 'dev) t
(** [randint dtype ?seed ?high shape low] generates integers in [\[low, high)].

    Uniform distribution over range. Default [high = 10]. Note: [high] is
    exclusive (NumPy convention).

    @raise Invalid_argument if non-integer dtype or [low >= high] *)

(** {2 Shape Manipulation}

    Functions to reshape, transpose, and rearrange arrays. *)

val reshape : int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [reshape shape t] returns view with new shape.

    At most one dimension can be -1 (inferred from total elements). Product of
    dimensions must match total elements. Returns view when possible (O(1)),
    copies if tensor is not contiguous and cannot be viewed.

    @raise Invalid_argument if shape incompatible or multiple -1 dimensions

    {@ocaml[
      # let t = create int32 [|2; 3|] [|1l; 2l; 3l; 4l; 5l; 6l|] in
        reshape [|6|] t
      - : (int32, int32_elt, 'dev) t = [1, 2, 3, 4, 5, 6]
      # let t = create int32 [|6|] [|1l; 2l; 3l; 4l; 5l; 6l|] in
        reshape [|3; -1|] t
      - : (int32, int32_elt, 'dev) t = [[1, 2],
                                  [3, 4],
                                  [5, 6]]
    ]} *)

val broadcast_to : int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [broadcast_to shape t] broadcasts tensor to target shape.

    Shapes must be broadcast-compatible: dimensions align from right, each must
    be equal or source must be 1. Returns view (no copy) with zero strides for
    broadcast dimensions.

    @raise Invalid_argument if shapes incompatible

    {@ocaml[
      # let t = create int32 [|1; 3|] [|1l; 2l; 3l|] in
        broadcast_to [|3; 3|] t
      - : (int32, int32_elt, 'dev) t = [[1, 2, 3],
                                  [1, 2, 3],
                                  [1, 2, 3]]
      # let t = ones float32 [|3; 1|] in
        shape (broadcast_to [|2; 3; 4|] t)
      - : int array = [|2; 3; 4|]
    ]} *)

val broadcasted :
  ?reverse:bool ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * ('a, 'b, 'dev) t
(** [broadcasted ?reverse t1 t2] broadcasts tensors to common shape.

    Returns views of both tensors broadcast to compatible shape. If [reverse] is
    true, returns [(t2', t1')] instead of [(t1', t2')]. Useful before
    element-wise operations.

    @raise Invalid_argument if shapes incompatible

    {@ocaml[
      # let t1 = ones float32 [|3;1|] in
        let t2 = ones float32 [|1;5|] in
        let t1', t2' = broadcasted t1 t2 in
        shape t1', shape t2'
      - : int array * int array = ([|3; 5|], [|3; 5|])
    ]} *)

val expand : int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [expand shape t] broadcasts tensor where [-1] keeps original dimension.

    Like {!broadcast_to} but [-1] preserves existing dimension size. Adds
    dimensions on left if needed.

    {@ocaml[
      # let t = ones float32 [|1; 4; 1|] in
        shape (expand [|3; -1; 5|] t)
      - : int array = [|3; 4; 5|]
      # let t = ones float32 [|5; 5|] in
        shape (expand [|-1; -1|] t)
      - : int array = [|5; 5|]
    ]} *)

val flatten :
  ?start_dim:int -> ?end_dim:int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [flatten ?start_dim ?end_dim t] collapses dimensions into single dimension.

    Default [start_dim = 0], [end_dim = -1] (last). Negative indices count from
    end. Dimensions [start_dim] through [end_dim] inclusive are flattened.

    @raise Invalid_argument if indices out of bounds

    {@ocaml[
      # flatten (zeros float32 [| 2; 3; 4 |]) |> shape
      - : int array = [|24|]
      # flatten ~start_dim:1 ~end_dim:2 (zeros float32 [| 2; 3; 4; 5 |]) |> shape
      - : int array = [|2; 12; 5|]
    ]} *)

val unflatten : int -> int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [unflatten dim sizes t] expands dimension [dim] into multiple dimensions.

    Product of [sizes] must equal size of dimension [dim]. At most one dimension
    can be -1 (inferred). Inverse of {!flatten}.

    @raise Invalid_argument if product mismatch or dim out of bounds

    {@ocaml[
      # unflatten 1 [| 3; 4 |] (zeros float32 [| 2; 12; 5 |]) |> shape
      - : int array = [|2; 3; 4; 5|]
      # unflatten 0 [| -1; 2 |] (ones float32 [| 6; 5 |]) |> shape
      - : int array = [|3; 2; 5|]
    ]} *)

val ravel : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [ravel t] returns contiguous 1-D view.

    Equivalent to [flatten t] but always returns contiguous result. Use when you
    need both flattening and contiguity.

    {@ocaml[
      # let x = create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
        ravel x
      - : (int32, int32_elt, 'dev) t = [1, 2, 3, 4, 5, 6]
      # let t = transpose (ones float32 [| 3; 4 |]) in
        is_c_contiguous t
      - : bool = false
      # let t_ravel = ravel t in
        is_c_contiguous t_ravel
      - : bool = true
    ]} *)

val squeeze : ?axes:int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [squeeze ?axes t] removes dimensions of size 1.

    If [axes] specified, only removes those dimensions. Negative indices count
    from end. Returns view when possible.

    @raise Invalid_argument if specified axis doesn't have size 1

    {@ocaml[
      # squeeze (ones float32 [| 1; 3; 1; 4 |]) |> shape
      - : int array = [|3; 4|]
      # squeeze ~axes:[| 0; 2 |] (ones float32 [| 1; 3; 1; 4 |]) |> shape
      - : int array = [|3; 4|]
      # squeeze ~axes:[| -1 |] (ones float32 [| 3; 4; 1 |]) |> shape
      - : int array = [|3; 4|]
    ]} *)

val unsqueeze : ?axes:int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [unsqueeze ?axes t] inserts dimensions of size 1 at specified positions.

    Axes refer to positions in result tensor. Must be in range [0, ndim].

    @raise Invalid_argument
      if [axes] not specified, out of bounds, or contains duplicates

    {@ocaml[
      # unsqueeze ~axes:[| 0; 2 |] (create float32 [| 3 |] [| 1.; 2.; 3. |]) |> shape
      - : int array = [|1; 3; 1|]
      # unsqueeze ~axes:[| 1 |] (create float32 [| 2 |] [| 5.; 6. |]) |> shape
      - : int array = [|2; 1|]
    ]} *)

val squeeze_axis : int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [squeeze_axis axis t] removes dimension [axis] if size is 1.

    @raise Invalid_argument if dimension size is not 1 *)

val unsqueeze_axis : int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [unsqueeze_axis axis t] inserts dimension of size 1 at [axis]. *)

val expand_dims : int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [expand_dims axes t] is synonym for {!unsqueeze}. *)

val transpose : ?axes:int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [transpose ?axes t] permutes dimensions.

    Default reverses all dimensions. [axes] must be permutation of [0..ndim-1].
    Returns view (no copy) with adjusted strides.

    @raise Invalid_argument if [axes] not valid permutation

    {@ocaml[
      # let x = create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
        transpose x
      - : (int32, int32_elt, 'dev) t = [[1, 4],
                                  [2, 5],
                                  [3, 6]]
      # transpose ~axes:[| 2; 0; 1 |] (zeros float32 [| 2; 3; 4 |]) |> shape
      - : int array = [|4; 2; 3|]
      # let id = transpose ~axes:[| 1; 0 |] in
        id == transpose
      - : bool = false
    ]} *)

val flip : ?axes:int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [flip ?axes t] reverses order along specified dimensions.

    Default flips all dimensions.

    {@ocaml[
      # let x = create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
        flip x
      - : (int32, int32_elt, 'dev) t = [[6, 5, 4],
                                  [3, 2, 1]]
      # let x = create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
        flip ~axes:[| 1 |] x
      - : (int32, int32_elt, 'dev) t = [[3, 2, 1],
                                  [6, 5, 4]]
    ]} *)

val moveaxis : int -> int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [moveaxis src dst t] moves dimension from [src] to [dst].

    @raise Invalid_argument if indices out of bounds *)

val swapaxes : int -> int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [swapaxes axis1 axis2 t] exchanges two dimensions.

    @raise Invalid_argument if indices out of bounds *)

val roll : ?axis:int -> int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [roll ?axis shift t] shifts elements along axis.

    Elements shifted beyond last position wrap to beginning. If [axis] not
    specified, shifts flattened tensor. Negative shift rolls backward.

    @raise Invalid_argument if axis out of bounds

    {@ocaml[
      # let x = create int32 [| 5 |] [| 1l; 2l; 3l; 4l; 5l |] in
        roll 2 x
      - : (int32, int32_elt, 'dev) t = [4, 5, 1, 2, 3]
      # let x = create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
        roll ~axis:1 1 x
      - : (int32, int32_elt, 'dev) t = [[3, 1, 2],
                                  [6, 4, 5]]
      # let x = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        roll ~axis:0 (-1) x
      - : (int32, int32_elt, 'dev) t = [[3, 4],
                                  [1, 2]]
    ]} *)

val pad : (int * int) array -> 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [pad padding value t] pads tensor with [value].

    [padding] specifies (before, after) for each dimension. Length must match
    tensor dimensions. Negative padding not allowed.

    @raise Invalid_argument if padding length wrong or negative values

    {@ocaml[
      # let x = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        pad [| (1, 1); (2, 2) |] 0l x
      - : (int32, int32_elt, 'dev) t =
      [[0, 0, 0, 0, 0, 0],
       [0, 0, 1, 2, 0, 0],
       [0, 0, 3, 4, 0, 0],
       [0, 0, 0, 0, 0, 0]]
    ]} *)

val shrink : (int * int) array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [shrink ranges t] extracts slice from [start] to [stop] (exclusive) for each
    dimension.

    {@ocaml[
      # let x = create int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |] in
        shrink [| (1, 3); (0, 2) |] x
      - : (int32, int32_elt, 'dev) t = [[4, 5],
                                  [7, 8]]
    ]} *)

val tile : int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [tile reps t] constructs tensor by repeating [t].

    [reps] specifies repetitions per dimension. If longer than ndim, prepends
    dimensions. Zero repetitions create empty tensor.

    @raise Invalid_argument if [reps] contains negative values

    {@ocaml[
      # let x = create int32 [| 1; 2 |] [| 1l; 2l |] in
        tile [| 2; 3 |] x
      - : (int32, int32_elt, 'dev) t = [[1, 2, 1, 2, 1, 2],
                                  [1, 2, 1, 2, 1, 2]]
      # let x = create int32 [| 2 |] [| 1l; 2l |] in
        tile [| 2; 1; 3 |] x |> shape
      - : int array = [|2; 1; 6|]
    ]} *)

val repeat : ?axis:int -> int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [repeat ?axis count t] repeats elements [count] times.

    If [axis] not specified, repeats flattened tensor.

    {@ocaml[
      # let x = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        repeat 2 x
      - : (int32, int32_elt, 'dev) t = [1, 1, 2, 2, 3, 3]
      # let x = create int32 [| 1; 2 |] [| 1l; 2l |] in
        repeat ~axis:0 3 x
      - : (int32, int32_elt, 'dev) t = [[1, 2],
                                  [1, 2],
                                  [1, 2]]
    ]} *)

(** {2 Array Combination and Splitting}

    Functions to join and split arrays. *)

val concatenate : ?axis:int -> ('a, 'b, 'dev) t list -> ('a, 'b, 'dev) t
(** [concatenate ?axis ts] joins tensors along existing axis.

    All tensors must have same shape except on concatenation axis. If [axis] not
    specified, flattens all tensors then concatenates. Returns contiguous
    result.

    @raise Invalid_argument if empty list or shape mismatch

    {@ocaml[
      # let x1 = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        let x2 = create int32 [| 1; 2 |] [| 5l; 6l |] in
        concatenate ~axis:0 [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[1, 2],
                                  [3, 4],
                                  [5, 6]]
      # let x1 = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        let x2 = create int32 [| 1; 2 |] [| 5l; 6l |] in
        concatenate [x1; x2]
      - : (int32, int32_elt, 'dev) t = [1, 2, 3, 4, 5, 6]
    ]} *)

val stack : ?axis:int -> ('a, 'b, 'dev) t list -> ('a, 'b, 'dev) t
(** [stack ?axis ts] joins tensors along new axis.

    All tensors must have identical shape. Result rank is input rank + 1.
    Default axis=0. Negative axis counts from end of result shape.

    @raise Invalid_argument if empty list, shape mismatch, or axis out of bounds

    {@ocaml[
      # let x1 = create int32 [| 2 |] [| 1l; 2l |] in
        let x2 = create int32 [| 2 |] [| 3l; 4l |] in
        stack [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[1, 2],
                                  [3, 4]]
      # let x1 = create int32 [| 2 |] [| 1l; 2l |] in
        let x2 = create int32 [| 2 |] [| 3l; 4l |] in
        stack ~axis:1 [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[1, 3],
                                  [2, 4]]
      # stack ~axis:(-1) [ones float32 [| 2; 3 |]; zeros float32 [| 2; 3 |]] |> shape
      - : int array = [|2; 3; 2|]
    ]} *)

val vstack : ('a, 'b, 'dev) t list -> ('a, 'b, 'dev) t
(** [vstack ts] stacks tensors vertically (row-wise).

    1-D tensors are treated as row vectors (shape [1;n]). Higher-D tensors
    concatenate along axis 0. All tensors must have same shape except possibly
    first dimension.

    @raise Invalid_argument if incompatible shapes

    {@ocaml[
      # let x1 = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        let x2 = create int32 [| 3 |] [| 4l; 5l; 6l |] in
        vstack [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[1, 2, 3],
                                  [4, 5, 6]]
      # let x1 = create int32 [| 1; 2 |] [| 1l; 2l |] in
        let x2 = create int32 [| 2; 2 |] [| 3l; 4l; 5l; 6l |] in
        vstack [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[1, 2],
                                  [3, 4],
                                  [5, 6]]
    ]} *)

val hstack : ('a, 'b, 'dev) t list -> ('a, 'b, 'dev) t
(** [hstack ts] stacks tensors horizontally (column-wise).

    1-D tensors concatenate directly. Higher-D tensors concatenate along axis 1.
    For 1-D arrays of different lengths, use vstack to make 2-D first.

    @raise Invalid_argument if incompatible shapes or <2D with different axis 0

    {@ocaml[
      # let x1 = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        let x2 = create int32 [| 3 |] [| 4l; 5l; 6l |] in
        hstack [x1; x2]
      - : (int32, int32_elt, 'dev) t = [1, 2, 3, 4, 5, 6]
      # let x1 = create int32 [| 2; 1 |] [| 1l; 2l |] in
        let x2 = create int32 [| 2; 1 |] [| 3l; 4l |] in
        hstack [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[1, 3],
                                  [2, 4]]
      # let x1 = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        let x2 = create int32 [| 2; 1 |] [| 5l; 6l |] in
        hstack [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[1, 2, 5],
                                  [3, 4, 6]]
    ]} *)

val dstack : ('a, 'b, 'dev) t list -> ('a, 'b, 'dev) t
(** [dstack ts] stacks tensors depth-wise (along third axis).

    Tensors are reshaped to at least 3-D before concatenation:
    - 1-D shape [n] → [1;n;1]
    - 2-D shape [m;n] → [m;n;1]
    - 3-D+ unchanged

    @raise Invalid_argument if resulting shapes incompatible

    {@ocaml[
      # let x1 = create int32 [| 2 |] [| 1l; 2l |] in
        let x2 = create int32 [| 2 |] [| 3l; 4l |] in
        dstack [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[[1, 3],
                                   [2, 4]]]
      # let x1 = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        let x2 = create int32 [| 2; 2 |] [| 5l; 6l; 7l; 8l |] in
        dstack [x1; x2]
      - : (int32, int32_elt, 'dev) t = [[[1, 5],
                                   [2, 6]],
                                  [[3, 7],
                                   [4, 8]]]
    ]} *)

val broadcast_arrays : ('a, 'b, 'dev) t list -> ('a, 'b, 'dev) t list
(** [broadcast_arrays ts] broadcasts all tensors to common shape.

    Finds the common broadcast shape and returns list of views with that shape.
    Broadcasting rules: dimensions align right, each must be 1 or equal.

    @raise Invalid_argument if shapes incompatible or empty list

    {@ocaml[
      # let x1 = ones float32 [| 3; 1 |] in
        let x2 = ones float32 [| 1; 5 |] in
        broadcast_arrays [x1; x2] |> List.map shape
      - : int array list = [[|3; 5|]; [|3; 5|]]
      # let x1 = scalar float32 5. in
        let x2 = ones float32 [| 2; 3; 4 |] in
        broadcast_arrays [x1; x2] |> List.map shape
      - : int array list = [[|2; 3; 4|]; [|2; 3; 4|]]
    ]} *)

val array_split :
  axis:int ->
  [< `Count of int | `Indices of int list ] ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t list
(** [array_split ~axis sections t] splits tensor into multiple parts.

    [`Count n] divides into n parts as evenly as possible. Extra elements go to
    first parts. [`Indices [i1;i2;...]] splits at indices creating [start:i1],
    [i1:i2], [i2:end].

    @raise Invalid_argument if axis out of bounds or invalid sections

    {@ocaml[
      # let x = create int32 [| 5 |] [| 1l; 2l; 3l; 4l; 5l |] in
        array_split ~axis:0 (`Count 3) x
      - : (int32, int32_elt, 'dev) t list = [[1, 2]; [3, 4]; [5]]
      # let x = create int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
        array_split ~axis:0 (`Indices [ 2; 4 ]) x
      - : (int32, int32_elt, 'dev) t list = [[1, 2]; [3, 4]; [5, 6]]
    ]} *)

val split : axis:int -> int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t list
(** [split ~axis sections t] splits into equal parts.

    @raise Invalid_argument if axis size not divisible by sections

    {@ocaml[
      # let x = create int32 [| 4; 2 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l |] in
        split ~axis:0 2 x
      - : (int32, int32_elt, 'dev) t list = [[[1, 2],
                                        [3, 4]]; [[5, 6],
                                                  [7, 8]]]
    ]} *)

(** {2 Type Conversion and Copying}

    Functions to convert between types and create copies. *)

val cast : ('c, 'd) dtype -> ('a, 'b, 'dev) t -> ('c, 'd, 'dev) t
(** [cast dtype t] converts elements to new dtype.

    Returns copy with same values in new type.

    {@ocaml[
      # let x = create float32 [| 3 |] [| 1.5; 2.7; 3.1 |] in
        cast int32 x
      - : (int32, int32_elt, 'dev) t = [1, 2, 3]
    ]} *)

val astype : ('a, 'b) dtype -> ('c, 'd, 'dev) t -> ('a, 'b, 'dev) t
(** [astype dtype t] is synonym for {!cast}. *)

val contiguous : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [contiguous t] returns C-contiguous tensor.

    Returns [t] unchanged if already contiguous (O(1)), otherwise creates
    contiguous copy (O(n)). Use before operations requiring direct memory
    access.

    {@ocaml[
      # let t = transpose (ones float32 [| 3; 4 |]) in
        is_c_contiguous (contiguous t)
      - : bool = true
    ]} *)

val copy : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [copy t] returns deep copy.

    Always allocates new memory and copies data. Result is contiguous.

    {@ocaml[
      # let x = create float32 [| 3 |] [| 1.; 2.; 3. |] in
        let y = copy x in
        set_item [ 0 ] 999. y;
        x, y
      - : (float, float32_elt) t * (float, float32_elt) t =
      ([1, 2, 3], [999, 2, 3])
    ]} *)

val blit : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> unit
(** [blit src dst] copies [src] into [dst].

    Shapes must match exactly. Handles broadcasting internally. Modifies [dst]
    in-place.

    @raise Invalid_argument if shape mismatch

    {@ocaml[
      let dst = zeros float32 [| 3; 3 |] in
      blit (ones float32 [| 3; 3 |]) dst
      (* dst now contains all 1s *)
    ]} *)

val fill : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [fill value t] sets all elements to [value].

    Modifies [t] in-place and returns it for chaining.

    {@ocaml[
      # let x = zeros float32 [| 2; 3 |] in
        let y = fill 5. x in
        y == x
      - : bool = true
    ]} *)

(** {2 Element Access and Slicing}

    Functions to access and modify array elements. *)

val get : int list -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [get indices t] returns subtensor at indices.

    Indexes from outermost dimension. Returns scalar tensor if all dimensions
    indexed, otherwise returns view of remaining dimensions.

    @raise Invalid_argument if indices out of bounds

    {@ocaml[
      # let x = create int32 [| 2; 2; 2 |] [| 0l; 1l; 2l; 3l; 4l; 5l; 6l; 7l |] in
        get [ 1; 1; 1 ] x
      - : (int32, int32_elt, 'dev) t = 7
      # let x = create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
        get [ 1 ] x
      - : (int32, int32_elt, 'dev) t = [4, 5, 6]
    ]} *)

val set : int list -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> unit
(** [set indices t value] assigns [value] at indices.

    @raise Invalid_argument if indices out of bounds *)

type index =
  | I of int (* Single index *)
  | L of int list (* List of indices *)
  | R of int * int (* Range [start, stop) *)
  | Rs of int * int * int (* Range with step *)
  | A (* All indices *)
  | M of (int, uint8_elt, [ `ocaml | `c | `metal ]) t (* Boolean mask *)
  | N (* New axis *)

val slice : index list -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [slice specs t] extracts subtensor using advanced indexing.

    Use for mixed indexing types beyond simple ranges:
    - [A]: full axis
    - [I i]: single index (reduces dimension; negative from end)
    - [R (start, stop)]: range [start, stop) with step 1
    - [Rs (start, stop, step)]: range with step
    - [L [i1;i2;...]]: fancy indexing with list of indices
    - [M bool_tensor]: boolean mask (shape must match axis)
    - [N]: insert new dimension of size 1

    Specs list corresponds to axes from left. Missing specs = [A].
    Returns view when possible.

    @raise Invalid_argument if specs out of bounds or incompatible

    {@ocaml[
      # let x = create int32 [| 2; 4 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l |] in
        slice [ I 1 ] x
      - : (int32, int32_elt, 'dev) t = [5, 6, 7, 8]
      # let x = create int32 [| 5 |] [| 0l; 1l; 2l; 3l; 4l |] in
        slice [ R (1, 3) ] x
      - : (int32, int32_elt, 'dev) t = [1, 2]
      # let x = create int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |] in
        slice [ R (0, 2); L [0; 2] ] x
      - : (int32, int32_elt, 'dev) t = [[1, 3],
                                  [4, 6]]
      # let x = create int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |] in
        slice [ M (greater_s x 5l) ] x  (* Elements >5 *)
      - : (int32, int32_elt, 'dev) t = [6, 7, 8, 9]
      # slice [ A; N ] x  (* Add new axis *)
      - : (int32, int32_elt, 'dev) t = [[[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]]
    ]} *)

val set_slice : index list -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> unit
(** [set_slice specs t value] assigns [value] to indexed region.

    Like {!index} but modifies t in-place. Value is broadcast if needed.

    @raise Invalid_argument if specs incompatible *)

val item : int list -> ('a, 'b, 'dev) t -> 'a
(** [item indices t] returns scalar value at indices.

    Must provide indices for all dimensions.

    @raise Invalid_argument if wrong number of indices or out of bounds *)

val set_item : int list -> 'a -> ('a, 'b, 'dev) t -> unit
(** [set_item indices value t] sets scalar value at indices.

    Must provide indices for all dimensions. Modifies tensor in-place.

    @raise Invalid_argument if wrong number of indices or out of bounds *)

val take :
  ?axis:int ->
  ?mode:[ `raise | `wrap | `clip ] ->
  (int32, int32_elt, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [take ?axis ?mode indices  t] takes elements from t using indices.

    Equivalent to t[indices] in NumPy along the specified axis. If [axis] is
    None, flattens t first. [indices] is an integer tensor of indices to take.
    [mode] handles out-of-bounds indices: `raise (default), `wrap (modulo),
    `clip (clamp to bounds).

    Returns a new tensor with shape based on indices and t's shape.

    @raise Invalid_argument if indices out of bounds and mode=`raise

    {@ocaml[
      # let x = create int32 [| 5 |] [| 0l; 1l; 2l; 3l; 4l |] in
        take (create int32 [| 3 |] [| 1l; 3l; 0l |]) x
      - : (int32, int32_elt, 'dev) t = [1, 3, 0]
      # let x = create int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |] in
        take ~axis:1 (create int32 [| 3 |] [| 0l; 2l; 1l |]) x
      - : (int32, int32_elt, 'dev) t = [[1, 3, 2],
                                  [4, 6, 5],
                                  [7, 9, 8]]
      # take ~mode:`clip (create int32 [| 2 |] [| -1l; 5l |]) x  (* Clamps to [0,4] *)
      - : (int32, int32_elt, 'dev) t = [0, 4]
    ]} *)

val take_along_axis :
  axis:int -> (int32, int32_elt, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [take_along_axis ~axis indices t] takes values along the specified axis
    using indices.

    Equivalent to NumPy's take_along_axis. [indices] must have the same shape as
    t except along the specified axis, where it matches the output size. Useful
    for gathering from argmax/argmin results.

    @raise Invalid_argument if shapes incompatible

    {@ocaml[
      # let x = create float32 [| 2; 3 |] [| 4.; 1.; 2.; 3.; 5.; 6. |] in
        let indices = create int32 [| 2; 1 |] [| 1l; 0l |] in  (* Per row max indices *)
        take_along_axis ~axis:1 indices x
      - : (float, float32_elt) t = [[1, 4],
                                    [5, 3]]
      # let x = create float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |] in
        let indices = argmax ~axis:0 x in
        take_along_axis ~axis:0 indices x  (* Max per column *)
      - : (float, float32_elt) t = [7, 8, 9]
    ]} *)

val put :
  ?axis:int ->
  indices:(int32, int32_elt, 'dev) t ->
  values:('a, 'b, 'dev) t ->
  ?mode:[ `raise | `wrap | `clip ] ->
  ('a, 'b, 'dev) t ->
  unit
(** [put ?axis indices values ?mode t] sets elements in t at positions specified
    by indices to values.

    Equivalent to NumPy's put (in-place version of take for setting). If [axis]
    is None, flattens t first. [indices] is an integer tensor of positions to
    set. [values] must match the number of indices (broadcasted if needed).
    [mode] handles out-of-bounds indices: `raise (default), `wrap (modulo),
    `clip (clamp).

    Modifies t in-place.

    @raise Invalid_argument
      if shapes incompatible or indices out of bounds with mode=`raise

    {@ocaml[
      # let x = zeros int32 [| 5 |] in
        put (create int32 [| 3 |] [| 1l; 3l; 0l |]) (create int32 [| 3 |] [| 10l; 20l; 30l |]) x;
        x
      - : (int32, int32_elt, 'dev) t = [30, 10, 0, 20, 0]
      # let x = create int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |] in
        put ~axis:1 (create int32 [| 3 |] [| 0l; 2l; 1l |]) (create int32 [| 3 |] [| 10l; 20l; 30l |]) x;
        x
      - : (int32, int32_elt, 'dev) t = [[10, 2, 3],
                                  [4, 30, 6],
                                  [7, 8, 20]]
      # put ~mode:`clip (create int32 [| 2 |] [| -1l; 5l |]) (create int32 [| 2 |] [| 99l; 99l |]) x  (* Clamps to [0,4] *)
    ]} *)

val put_along_axis :
  axis:int ->
  indices:(int32, int32_elt, 'dev) t ->
  values:('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  unit
(** [put_along_axis ~axis indices values t] sets values along the specified axis
    using indices.

    Equivalent to NumPy's put_along_axis. [indices] must have the same shape as
    t except along the axis (where it matches values' size along that axis).
    [values] is broadcasted to match the selection shape. Useful for scattering
    to argmax/argmin positions.

    Modifies t in-place.

    @raise Invalid_argument if shapes incompatible

    {@ocaml[
      # let x = zeros float32 [| 2; 3 |] in
        let indices = create int32 [| 2; 1 |] [| 1l; 0l |] in  (* Per row positions *)
        put_along_axis ~axis:1 indices (create float32 [| 2; 1 |] [| 10.; 20. |]) x;
        x
      - : (float, float32_elt) t = [[0, 10, 0],
                                    [20, 0, 0]]
      # let x = create float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |] in
        let indices = argmax ~axis:0 x in  (* Max positions per column *)
        put_along_axis ~axis:0 indices (ones float32 [| 3 |]) x  (* Set max per column to 1 *)
    ]} *)

val compress :
  ?axis:int ->
  condition:(int, uint8_elt, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [compress ?axis condition t] selects elements where condition is true.

    Equivalent to NumPy's compress. [condition] is a 1D boolean array. If [axis]
    is None, flattens t first. Otherwise, compresses along the specified axis
    (condition length must match t's dim along axis).

    Returns a new tensor with reduced size along the axis/flattened.

    @raise Invalid_argument if condition length incompatible

    {@ocaml[
      # let x = create int32 [| 5 |] [| 1l; 2l; 3l; 4l; 5l |] in
        compress (create bool [| 5 |] [| true; false; true; false; true |]) x
      - : (int32, int32_elt, 'dev) t = [1, 3, 5]
      # let x = create int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |] in
        compress ~axis:0 (create bool [| 3 |] [| false; true; true |]) x
      - : (int32, int32_elt, 'dev) t = [[4, 5, 6],
                                  [7, 8, 9]]
    ]} *)

val extract :
  condition:(int, uint8_elt, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [extract condition t] flattens and selects elements where condition is true.

    Equivalent to NumPy's extract (1D compress after flatten). [condition] must
    have the same shape and size as t (element-wise).

    Returns a 1D tensor with selected elements.

    @raise Invalid_argument if shapes incompatible

    {@ocaml[
      # let x = create int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |] in
        extract (greater_s x 5l) x
      - : (int32, int32_elt, 'dev) t = [6, 7, 8, 9]
    ]} *)

val nonzero : ('a, 'b, 'dev) t -> (int32, int32_elt, 'dev) t array
(** [nonzero t] returns indices of non-zero elements.

    Equivalent to NumPy's nonzero. Treats non-zero as true for bool tensors.
    Returns an array of 1D tensors, one per dimension, with coordinates of
    non-zeros.

    For example, for a 2D tensor, returns [| rows; cols |] where rows[i],
    cols[i] is the position of the i-th non-zero.

    {@ocaml[
      # let x = create int32 [| 3; 3 |] [| 0l; 1l; 0l; 2l; 0l; 3l; 0l; 0l; 4l |] in
        let rows, cols = nonzero x in
        rows, cols
      - : (int32, int32_elt, 'dev) t * (int32, int32_elt, 'dev) t = ([0, 1, 1, 2], [1, 0, 2, 2])
    ]} *)

val argwhere : ('a, 'b, 'dev) t -> (int32, int32_elt, 'dev) t
(** [argwhere t] returns indices of non-zero elements as a 2D tensor.

    Equivalent to NumPy's argwhere. Each row is a coordinate [dim0; dim1; ...]
    of a non-zero element. Shape is [num_nonzeros; ndim].

    {@ocaml[
      # let x = create int32 [| 3; 3 |] [| 0l; 1l; 0l; 2l; 0l; 3l; 0l; 0l; 4l |] in
        argwhere x
      - : (int32, int32_elt, 'dev) t = [[0, 1],
                                  [1, 0],
                                  [1, 2],
                                  [2, 2]]
    ]} *)

(** {2 Basic Arithmetic Operations}

    Element-wise arithmetic operations and their variants. *)

val add : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [add t1 t2] computes element-wise sum with broadcasting.

    @raise Invalid_argument if shapes incompatible *)

val add_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [add_s t scalar] adds scalar to each element. *)

val iadd : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [iadd target value] adds [value] to [target] in-place.

    Returns modified [target]. *)

val radd_s : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [radd_s scalar t] is [add_s t scalar]. *)

val iadd_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [iadd_s t scalar] adds scalar to [t] in-place. *)

val sub : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [sub t1 t2] computes element-wise difference with broadcasting. *)

val sub_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [sub_s t scalar] subtracts scalar from each element. *)

val rsub_s : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [rsub_s scalar t] computes [scalar - t]. *)

val isub : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [isub target value] subtracts [value] from [target] in-place. *)

val isub_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [isub_s t scalar] subtracts scalar from [t] in-place. *)

val mul : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [mul t1 t2] computes element-wise product with broadcasting. *)

val mul_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [mul_s t scalar] multiplies each element by scalar. *)

val rmul_s : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [rmul_s scalar t] is [mul_s t scalar]. *)

val imul : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [imul target value] multiplies [target] by [value] in-place. *)

val imul_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [imul_s t scalar] multiplies [t] by scalar in-place. *)

val div : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [div t1 t2] computes element-wise division.

    True division for floats (result is float). Integer division for integers
    (truncates toward zero). Complex division follows standard rules.

    {@ocaml[
      # let x = create float32 [| 3 |] [| 7.; 8.; 9. |] in
        let y = create float32 [| 3 |] [| 2.; 2.; 2. |] in
        div x y
      - : (float, float32_elt) t = [3.5, 4, 4.5]
      # let x = create int32 [| 3 |] [| 7l; 8l; 9l |] in
        let y = create int32 [| 3 |] [| 2l; 2l; 2l |] in
        div x y
      - : (int32, int32_elt, 'dev) t = [3, 4, 4]
      # let x = create int32 [| 2 |] [| -7l; 8l |] in
        let y = create int32 [| 2 |] [| 2l; 2l |] in
        div x y
      - : (int32, int32_elt, 'dev) t = [-3, 4]
    ]} *)

val div_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [div_s t scalar] divides each element by scalar. *)

val rdiv_s : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [rdiv_s scalar t] computes [scalar / t]. *)

val idiv : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [idiv target value] divides [target] by [value] in-place. *)

val idiv_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [idiv_s t scalar] divides [t] by scalar in-place. *)

val pow : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [pow base exponent] computes element-wise power. *)

val pow_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [pow_s t scalar] raises each element to scalar power. *)

val rpow_s : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [rpow_s scalar t] computes [scalar ** t]. *)

val ipow : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [ipow target exponent] raises [target] to [exponent] in-place. *)

val ipow_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [ipow_s t scalar] raises [t] to scalar power in-place. *)

val mod_ : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [mod_ t1 t2] computes element-wise modulo. *)

val mod_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [mod_s t scalar] computes modulo scalar for each element. *)

val rmod_s : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [rmod_s scalar t] computes [scalar mod t]. *)

val imod : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [imod target divisor] computes modulo in-place. *)

val imod_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [imod_s t scalar] computes modulo scalar in-place. *)

val neg : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [neg t] negates all elements. *)

(** {2 Mathematical Functions}

    Unary mathematical operations and special functions. *)

val abs : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [abs t] computes absolute value. *)

val sign : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [sign t] returns -1, 0, or 1 based on sign.

    For unsigned types, returns 1 for all non-zero values, 0 for zero.

    {@ocaml[
      # let x = create float32 [| 3 |] [| -2.; 0.; 3.5 |] in
        sign x
      - : (float, float32_elt) t = [-1, 0, 1]
    ]} *)

val square : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [square t] computes element-wise square. *)

val sqrt : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [sqrt t] computes element-wise square root. *)

val rsqrt : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [rsqrt t] computes reciprocal square root. *)

val recip : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [recip t] computes element-wise reciprocal. *)

val log : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [log t] computes natural logarithm. *)

val log2 : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [log2 t] computes base-2 logarithm. *)

val exp : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [exp t] computes exponential. *)

val exp2 : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [exp2 t] computes 2^x. *)

val sin : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [sin t] computes sine. *)

val cos : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [cos t] computes cosine. *)

val tan : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [tan t] computes tangent. *)

val asin : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [asin t] computes arcsine. *)

val acos : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [acos t] computes arccosine. *)

val atan : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [atan t] computes arctangent. *)

val atan2 : (float, 'a, 'dev) t -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [atan2 y x] computes arctangent of y/x using signs to determine quadrant.

    Returns angle in radians in range [-π, π]. Handles x=0 correctly.

    {@ocaml[
      # let y = scalar float32 1. in
        let x = scalar float32 1. in
        atan2 y x |> item [] |> Float.round
      - : float = 1.
      # let y = scalar float32 1. in
        let x = scalar float32 0. in
        atan2 y x |> item [] |> Float.round
      - : float = 2.
      # let y = scalar float32 0. in
        let x = scalar float32 0. in
        atan2 y x |> item []
      - : float = 0.
    ]} *)

val sinh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [sinh t] computes hyperbolic sine. *)

val cosh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [cosh t] computes hyperbolic cosine. *)

val tanh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [tanh t] computes hyperbolic tangent. *)

val asinh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [asinh t] computes inverse hyperbolic sine. *)

val acosh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [acosh t] computes inverse hyperbolic cosine. *)

val atanh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [atanh t] computes inverse hyperbolic tangent. *)

val hypot : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [hypot x y] computes sqrt(x² + y²) avoiding overflow.

    Uses numerically stable algorithm: max * sqrt(1 + (min/max)²).

    {@ocaml[
      # let x = scalar float32 3. in
        let y = scalar float32 4. in
        hypot x y |> item []
      - : float = 5.
      # let x = scalar float64 1e200 in
        let y = scalar float64 1e200 in
        hypot x y |> item [] < Float.infinity
      - : bool = true
    ]} *)

val trunc : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [trunc t] rounds toward zero.

    Removes fractional part. Positive values round down, negative round up.

    {@ocaml[
      # let x = create float32 [| 3 |] [| 2.7; -2.7; 2.0 |] in
        trunc x
      - : (float, float32_elt) t = [2, -2, 2]
    ]} *)

val ceil : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [ceil t] rounds up to nearest integer.

    Smallest integer not less than input.

    {@ocaml[
      # let x = create float32 [| 4 |] [| 2.1; 2.9; -2.1; -2.9 |] in
        ceil x
      - : (float, float32_elt) t = [3, 3, -2, -2]
    ]} *)

val floor : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [floor t] rounds down to nearest integer.

    Largest integer not greater than input.

    {@ocaml[
      # let x = create float32 [| 4 |] [| 2.1; 2.9; -2.1; -2.9 |] in
        floor x
      - : (float, float32_elt) t = [2, 2, -3, -3]
    ]} *)

val round : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [round t] rounds to nearest integer (half away from zero).

    Ties round away from zero (not banker's rounding).

    {@ocaml[
      # let x = create float32 [| 4 |] [| 2.5; 3.5; -2.5; -3.5 |] in
        round x
      - : (float, float32_elt) t = [3, 4, -3, -4]
    ]} *)

val lerp :
  ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [lerp start end_ weight] computes linear interpolation.

    Returns start + weight * (end_ - start). Weight typically in [0, 1].

    {@ocaml[
      # let start = scalar float32 0. in
        let end_ = scalar float32 10. in
        let weight = scalar float32 0.3 in
        lerp start end_ weight |> item []
      - : float = 3.
      # let start = create float32 [| 2 |] [| 1.; 2. |] in
        let end_ = create float32 [| 2 |] [| 5.; 8. |] in
        let weight = create float32 [| 2 |] [| 0.25; 0.5 |] in
        lerp start end_ weight
      - : (float, float32_elt) t = [2, 5]
    ]} *)

val lerp_scalar_weight :
  ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [lerp_scalar_weight start end_ weight] interpolates with scalar weight. *)

(** {2 Comparison and Logical Operations}

    Element-wise comparisons and logical operations. *)

val cmplt : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmplt t1 t2] returns 1 where t1 < t2, 0 elsewhere. *)

val less : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [less t1 t2] is synonym for {!cmplt}. *)

val less_s : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
(** [less_s t scalar] checks if each element is less than scalar. *)

val cmpne : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmpne t1 t2] returns 1 where t1 ≠ t2, 0 elsewhere. *)

val not_equal : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [not_equal t1 t2] is synonym for {!cmpne}. *)

val not_equal_s : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
(** [not_equal_s t scalar] compares each element with scalar for inequality. *)

val cmpeq : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmpeq t1 t2] returns 1 where t1 = t2, 0 elsewhere. *)

val equal : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [equal t1 t2] is synonym for {!cmpeq}. *)

val equal_s : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
(** [equal_s t scalar] compares each element with scalar for equality. *)

val cmpgt : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmpgt t1 t2] returns 1 where t1 > t2, 0 elsewhere. *)

val greater : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [greater t1 t2] is synonym for {!cmpgt}. *)

val greater_s : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
(** [greater_s t scalar] checks if each element is greater than scalar. *)

val cmple : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmple t1 t2] returns 1 where t1 ≤ t2, 0 elsewhere. *)

val less_equal :
  ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [less_equal t1 t2] is synonym for {!cmple}. *)

val less_equal_s : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
(** [less_equal_s t scalar] checks if each element is less than or equal to
    scalar. *)

val cmpge : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmpge t1 t2] returns 1 where t1 ≥ t2, 0 elsewhere. *)

val greater_equal :
  ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [greater_equal t1 t2] is synonym for {!cmpge}. *)

val greater_equal_s : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
(** [greater_equal_s t scalar] checks if each element is greater than or equal
    to scalar. *)

val array_equal :
  ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [array_equal t1 t2] returns scalar 1 if all elements equal, 0 otherwise.

    Broadcasts inputs before comparison. Returns 0 if shapes incompatible.

    {@ocaml[
      # let x = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        let y = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        array_equal x y |> item []
      - : int = 1
      # let x = create int32 [| 2 |] [| 1l; 2l |] in
        let y = create int32 [| 2 |] [| 1l; 3l |] in
        array_equal x y |> item []
      - : int = 0
    ]} *)

val maximum : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [maximum t1 t2] returns element-wise maximum. *)

val maximum_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [maximum_s t scalar] returns maximum of each element and scalar. *)

val rmaximum_s : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [rmaximum_s scalar t] is [maximum_s t scalar]. *)

val imaximum : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [imaximum target value] computes maximum in-place. *)

val imaximum_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [imaximum_s t scalar] computes maximum with scalar in-place. *)

val minimum : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [minimum t1 t2] returns element-wise minimum. *)

val minimum_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [minimum_s t scalar] returns minimum of each element and scalar. *)

val rminimum_s : 'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [rminimum_s scalar t] is [minimum_s t scalar]. *)

val iminimum : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [iminimum target value] computes minimum in-place. *)

val iminimum_s : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
(** [iminimum_s t scalar] computes minimum with scalar in-place. *)

val logical_and : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [logical_and t1 t2] computes element-wise AND.

    Non-zero values are true. *)

val logical_or : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [logical_or t1 t2] computes element-wise OR. *)

val logical_xor : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [logical_xor t1 t2] computes element-wise XOR. *)

val logical_not : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [logical_not t] computes element-wise NOT.

    Returns 1 - x. Non-zero values become 0, zero becomes 1.

    {@ocaml[
      # let x = create int32 [| 3 |] [| 0l; 1l; 5l |] in
        logical_not x
      - : (int32, int32_elt, 'dev) t = [1, 0, -4]
    ]} *)

val isinf : (float, 'a, 'dev) t -> (int, uint8_elt, 'dev) t
(** [isinf t] returns 1 where infinite, 0 elsewhere.

    Detects both positive and negative infinity. Non-float types return all 0s.

    {@ocaml[
      # let x = create float32 [| 4 |] [| 1.; Float.infinity; Float.neg_infinity; Float.nan |] in
        isinf x
      - : (int, uint8_elt, 'dev) t = [0, 1, 1, 0]
    ]} *)

val isnan : ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [isnan t] returns 1 where NaN, 0 elsewhere.

    NaN is the only value that doesn't equal itself. Non-float types return all
    0s.

    {@ocaml[
      # let x = create float32 [| 3 |] [| 1.; Float.nan; Float.infinity |] in
        isnan x
      - : (int, uint8_elt, 'dev) t = [0, 1, 0]
    ]} *)

val isfinite : (float, 'a, 'dev) t -> (int, uint8_elt, 'dev) t
(** [isfinite t] returns 1 where finite, 0 elsewhere.

    Finite means not inf, -inf, or NaN. Non-float types return all 1s.

    {@ocaml[
      # let x = create float32 [| 4 |] [| 1.; Float.infinity; Float.nan; -0. |] in
        isfinite x
      - : (int, uint8_elt, 'dev) t = [1, 0, 0, 1]
    ]} *)

val where :
  (int, uint8_elt, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [where cond if_true if_false] selects elements based on condition.

    Returns [if_true] where [cond] is non-zero, [if_false] elsewhere. All three
    inputs broadcast to common shape.

    @raise Invalid_argument if shapes incompatible for broadcasting

    {@ocaml[
      # let cond = create uint8 [| 3 |] [| 1; 0; 1 |] in
        let if_true = create int32 [| 3 |] [| 2l; 3l; 4l |] in
        let if_false = create int32 [| 3 |] [| 5l; 6l; 7l |] in
        where cond if_true if_false
      - : (int32, int32_elt, 'dev) t = [2, 6, 4]
      # let x = create float32 [| 4 |] [| -1.; 2.; -3.; 4. |] in
        where (cmpgt x (scalar float32 0.)) x (scalar float32 0.)
      - : (float, float32_elt) t = [0, 2, 0, 4]
    ]} *)

val clamp : ?min:'a -> ?max:'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [clamp ?min ?max t] limits values to range.

    Elements below [min] become [min], above [max] become [max]. *)

val clip : ?min:'a -> ?max:'a -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [clip ?min ?max t] is synonym for {!clamp}. *)

(** {2 Bitwise Operations}

    Bitwise operations on integer arrays. *)

val bitwise_xor : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [bitwise_xor t1 t2] computes element-wise XOR. *)

val bitwise_or : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [bitwise_or t1 t2] computes element-wise OR. *)

val bitwise_and : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [bitwise_and t1 t2] computes element-wise AND. *)

val bitwise_not : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [bitwise_not t] computes element-wise NOT. *)

val invert : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [invert t] is synonym for {!bitwise_not}. *)

val lshift : ('a, 'b, 'dev) t -> int -> ('a, 'b, 'dev) t
(** [lshift t shift] left-shifts elements by [shift] bits.

    Equivalent to multiplication by 2^shift. Overflow wraps around.

    @raise Invalid_argument if shift negative or non-integer dtype

    {@ocaml[
      # let x = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        lshift x 2
      - : (int32, int32_elt, 'dev) t = [4, 8, 12]
    ]} *)

val rshift : ('a, 'b, 'dev) t -> int -> ('a, 'b, 'dev) t
(** [rshift t shift] right-shifts elements by [shift] bits.

    Equivalent to integer division by 2^shift (rounds toward zero).

    @raise Invalid_argument if shift negative or non-integer dtype

    {@ocaml[
      # let x = create int32 [| 3 |] [| 8l; 9l; 10l |] in
        rshift x 2
      - : (int32, int32_elt, 'dev) t = [2, 2, 2]
    ]} *)

(** Infix operators *)
module Infix : sig
  (** {3 Elementwise Arithmetic} *)

  val ( + ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 + t2] is a synonym for {!add}. *)

  val ( - ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 - t2] is a synonym for {!sub}. *)

  val ( * ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 * t2] is a synonym for {!mul}. *)

  val ( / ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 / t2] is a synonym for {!div}. *)

  val ( ** ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 ** t2] is a synonym for {!pow}. *)

  (** {3 Scalar-right Arithmetic} *)

  val ( +$ ) : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
  (** [t +$ scalar] is a synonym for {!add_s}. *)

  val ( -$ ) : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
  (** [t -$ scalar] is a synonym for {!sub_s}. *)

  val ( *$ ) : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
  (** [t *$ scalar] is a synonym for {!mul_s}. *)

  val ( /$ ) : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
  (** [t /$ scalar] is a synonym for {!div_s}. *)

  val ( **$ ) : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
  (** [t **$ scalar] is a synonym for {!pow_s}. *)

  (** {3 Comparisons} *)

  val ( < ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
  (** [t1 < t2] is a synonym for {!less} *)

  val ( <> ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
  (** [t1 <> t2] is a synonym for {!not_equal}. *)

  val ( = ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
  (** [t1 = t1] is a synonym for {!equal}. *)

  val ( > ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
  (** [t1 > t2] is a synonym for {!greater}. *)

  val ( <= ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
  (** [t1 <= t2] is a synonym for {!less_equal}. *)

  val ( >= ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
  (** [t1 >= t2] is a synonym for {!greater_equal}. *)

  (** {3 Scalar Comparisons} *)

  val ( =$ ) : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
  (** [t =$ scalar] compares each element with scalar for equality. *)

  val ( <>$ ) : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
  (** [t <>$ scalar] compares each element with scalar for inequality. *)

  val ( <$ ) : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
  (** [t <$ scalar] checks if each element is less than scalar. *)

  val ( >$ ) : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
  (** [t >$ scalar] checks if each element is greater than scalar. *)

  val ( <=$ ) : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
  (** [t <=$ scalar] checks if each element is less than or equal to scalar. *)

  val ( >=$ ) : ('a, 'b, 'dev) t -> 'a -> (int, uint8_elt, 'dev) t
  (** [t >=$ scalar] checks if each element is greater than or equal to scalar.
  *)

  (** {3 Bitwise Operations} *)

  val ( lxor ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 lxor t2] is a synonym for {!bitwise_xor}. *)

  val ( lor ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 lor t2] is a synonym for {!bitwise_or}. *)

  val ( land ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 land t2] is a synonym for {!bitwise_and}. *)

  (** {3 Modulo Operations} *)

  val ( % ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 % t2] is a synonym for {!mod_}. *)

  val ( mod ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 mod t2] is a synonym for {!mod_}. *)

  val ( %$ ) : ('a, 'b, 'dev) t -> 'a -> ('a, 'b, 'dev) t
  (** [t %$ scalar] is a synonym for {!mod_s}. *)

  (** {3 Boolean Mask Logic} *)

  val ( ^ ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 ^ t2] is a synonym for {!logical_xor}. *)

  val ( && ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 && t2] is a synonym for {!logical_and}. *)

  val ( || ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 || t2] is a synonym for {!logical_or}. *)

  val ( ~- ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [~-t] is a synonym for {!logical_not}. *)

  (** {3 Linear Algebra} *)

  val ( @@ ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 @@ t2] is a synonym for {!matmul}. *)

  val ( /@ ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 /@ t2] solves the linear system t1 * x = t2 for x. *)

  val ( **@ ) : ('a, 'b, 'dev) t -> int -> ('a, 'b, 'dev) t
  (** [t **@ n] computes matrix power (t raised to the nth power). *)

  val ( <.> ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 <.> t2] is a synonym for {!dot}. *)

  (** {3 Concatenation} *)

  val ( @= ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 @= t2] concatenates t1 and t2 vertically (along axis 0). *)

  val ( @|| ) : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** [t1 @|| t2] concatenates t1 and t2 horizontally (along axis 1). *)

  (** {3 Indexing and Slicing} *)

  val ( .%{} ) : ('a, 'b, 'dev) t -> int list -> ('a, 'b, 'dev) t
  (** [t.%{indices}] is a synonym for {!get}. *)

  val ( .%{}<- ) : ('a, 'b, 'dev) t -> int list -> ('a, 'b, 'dev) t -> unit
  (** [t.%{indices} <- value] is a synonym for {!set}. *)

  val ( .${} ) : ('a, 'b, 'dev) t -> index list -> ('a, 'b, 'dev) t
  (** [t.${slice}] is a synonym for {!slice}. *)

  val ( .${}<- ) : ('a, 'b, 'dev) t -> index list -> ('a, 'b, 'dev) t -> unit
  (** [t.${slice} <- value] is a synonym for {!set_slice}. *)
end

(** {2 Reduction Operations}

    Functions that reduce array dimensions. *)

val sum :
  ?axes:int array -> ?keepdims:bool -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [sum ?axes ?keepdims t] sums elements along specified axes.

    Default sums all axes (returns scalar). If [keepdims] is true, retains
    reduced dimensions with size 1. Negative axes count from end.

    @raise Invalid_argument if any axis is out of bounds

    {@ocaml[
      # let x = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        sum x |> item []
      - : float = 10.
      # let x = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        sum ~axes:[| 0 |] x
      - : (float, float32_elt) t = [4, 6]
      # let x = create float32 [| 1; 2 |] [| 1.; 2. |] in
        sum ~axes:[| 1 |] ~keepdims:true x
      - : (float, float32_elt) t = [[3]]
      # let x = create float32 [| 1; 3 |] [| 1.; 2.; 3. |] in
        sum ~axes:[| -1 |] x
      - : (float, float32_elt) t = [6]
    ]} *)

val max :
  ?axes:int array -> ?keepdims:bool -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [max ?axes ?keepdims t] finds maximum along axes.

    Default reduces all axes. NaN propagates (any NaN input gives NaN output).

    {@ocaml[
      # let x = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        max x |> item []
      - : float = 6.
      # let x = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        max ~axes:[| 0 |] x
      - : (float, float32_elt) t = [3, 4]
      # let x = create float32 [| 1; 2 |] [| 1.; 2. |] in
        max ~axes:[| 1 |] ~keepdims:true x
      - : (float, float32_elt) t = [[2]]
    ]} *)

val min :
  ?axes:int array -> ?keepdims:bool -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [min ?axes ?keepdims t] finds minimum along axes.

    Default reduces all axes. NaN propagates (any NaN input gives NaN output).

    {@ocaml[
      # let x = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        min x |> item []
      - : float = 1.
      # let x = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        min ~axes:[| 0 |] x
      - : (float, float32_elt) t = [1, 2]
    ]} *)

val prod :
  ?axes:int array -> ?keepdims:bool -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [prod ?axes ?keepdims t] computes product along axes.

    Default multiplies all elements. Empty axes give 1.

    {@ocaml[
      # let x = create int32 [| 3 |] [| 2l; 3l; 4l |] in
        prod x |> item []
      - : int32 = 24l
      # let x = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
        prod ~axes:[| 0 |] x
      - : (int32, int32_elt, 'dev) t = [3, 8]
    ]} *)

val mean :
  ?axes:int array -> ?keepdims:bool -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [mean ?axes ?keepdims t] computes arithmetic mean along axes.

    Sum of elements divided by count. NaN propagates.

    {@ocaml[
      # let x = create float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
        mean x |> item []
      - : float = 2.5
      # let x = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        mean ~axes:[| 1 |] x
      - : (float, float32_elt) t = [2, 5]
    ]} *)

val var :
  ?axes:int array ->
  ?keepdims:bool ->
  ?ddof:int ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [var ?axes ?keepdims ?ddof t] computes variance along axes.

    [ddof] is delta degrees of freedom. Default 0 (population variance). Use 1
    for sample variance. Variance = E[(X - E[X])²] / (N - ddof).

    @raise Invalid_argument if ddof >= number of elements

    {@ocaml[
      # let x = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        var x |> item []
      - : float = 2.
      # let x = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        var ~ddof:1 x |> item []
      - : float = 2.5
    ]} *)

val std :
  ?axes:int array ->
  ?keepdims:bool ->
  ?ddof:int ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [std ?axes ?keepdims ?ddof t] computes standard deviation.

    Square root of variance: sqrt(var(t, ddof)). See {!var} for ddof meaning.

    {@ocaml[
      # let x = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        std x |> item [] |> Float.round
      - : float = 1.
      # let x = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        std ~ddof:1 x |> item [] |> Float.round
      - : float = 2.
    ]} *)

val all :
  ?axes:int array ->
  ?keepdims:bool ->
  ('a, 'b, 'dev) t ->
  (int, uint8_elt, 'dev) t
(** [all ?axes ?keepdims t] tests if all elements are true (non-zero).

    Returns 1 if all elements along axes are non-zero, 0 otherwise.

    {@ocaml[
      # let x = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        all x |> item []
      - : int = 1
      # let x = create int32 [| 3 |] [| 1l; 0l; 3l |] in
        all x |> item []
      - : int = 0
      # let x = create int32 [| 2; 2 |] [| 1l; 0l; 1l; 1l |] in
        all ~axes:[| 1 |] x
      - : (int, uint8_elt, 'dev) t = [0, 1]
    ]} *)

val any :
  ?axes:int array ->
  ?keepdims:bool ->
  ('a, 'b, 'dev) t ->
  (int, uint8_elt, 'dev) t
(** [any ?axes ?keepdims t] tests if any element is true (non-zero).

    Returns 1 if any element along axes is non-zero, 0 if all are zero.

    {@ocaml[
      # let x = create int32 [| 3 |] [| 0l; 0l; 1l |] in
        any x |> item []
      - : int = 1
      # let x = create int32 [| 3 |] [| 0l; 0l; 0l |] in
        any x |> item []
      - : int = 0
      # let x = create int32 [| 2; 2 |] [| 0l; 0l; 0l; 1l |] in
        any ~axes:[| 1 |] x
      - : (int, uint8_elt, 'dev) t = [0, 1]
    ]} *)

val argmax :
  ?axis:int -> ?keepdims:bool -> ('a, 'b, 'dev) t -> (int32, int32_elt, 'dev) t
(** [argmax ?axis ?keepdims t] finds indices of maximum values.

    Returns index of first occurrence for ties. If [axis] not specified,
    operates on flattened tensor and returns scalar.

    {@ocaml[
      # let x = create int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |] in
        argmax x |> item []
      - : int32 = 4l
      # let x = create int32 [| 2; 3 |] [| 1l; 5l; 3l; 2l; 4l; 6l |] in
        argmax ~axis:1 x
      - : (int32, int32_elt, 'dev) t = [1, 2]
    ]} *)

val argmin :
  ?axis:int -> ?keepdims:bool -> ('a, 'b, 'dev) t -> (int32, int32_elt, 'dev) t
(** [argmin ?axis ?keepdims t] finds indices of minimum values.

    Returns index of first occurrence for ties. If [axis] not specified,
    operates on flattened tensor and returns scalar.

    {@ocaml[
      # let x = create int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |] in
        argmin x |> item []
      - : int32 = 1l
      # let x = create int32 [| 2; 3 |] [| 5l; 2l; 3l; 1l; 4l; 0l |] in
        argmin ~axis:1 x
      - : (int32, int32_elt, 'dev) t = [1, 2]
    ]} *)

(** {2 Sorting and Searching}

    Functions for sorting arrays and finding indices. *)

val sort :
  ?descending:bool ->
  ?axis:int ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * (int32, int32_elt, 'dev) t
(** [sort ?descending ?axis t] sorts elements along axis.

    Returns (sorted_values, indices) where indices map sorted positions to
    original positions. Default sorts last axis in ascending order.

    Algorithm: Bitonic sort (parallel-friendly, stable)
    - Pads to power of 2 with inf/-inf for correctness
    - O(n log² n) comparisons, O(log² n) depth
    - Stable: preserves relative order of equal elements
    - First occurrence wins for duplicate values

    Special values:
    - NaN: sorted to end (ascending) or beginning (descending)
    - inf/-inf: sorted normally
    - For integers: uses max/min values for padding

    @raise Invalid_argument if axis out of bounds

    {@ocaml[
      # let x = create int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |] in
        sort x
      - : (int32, int32_elt, 'dev) t * (int32, int32_elt, 'dev) t =
      ([1, 1, 3, 4, 5], [1, 3, 0, 2, 4])
      # let x = create int32 [| 2; 2 |] [| 3l; 1l; 1l; 4l |] in
        sort ~descending:true ~axis:0 x
      - : (int32, int32_elt, 'dev) t * (int32, int32_elt, 'dev) t =
      ([[3, 4],
        [1, 1]], [[0, 1],
                  [1, 0]])
      # let x = create float32 [| 4 |] [| Float.nan; 1.; 2.; Float.nan |] in
        let v, _ = sort x in
        v
      - : (float, float32_elt) t = [1, 2, nan, nan]
    ]} *)

val argsort :
  ?descending:bool ->
  ?axis:int ->
  ('a, 'b, 'dev) t ->
  (int32, int32_elt, 'dev) t
(** [argsort ?descending ?axis t] returns indices that would sort tensor.

    Equivalent to [snd (sort ?descending ?axis t)]. Returns indices such that
    taking elements at these indices yields sorted array.

    For 1-D: result[i] is the index of the i-th smallest element. For N-D: sorts
    along specified axis independently.

    @raise Invalid_argument if axis out of bounds

    {@ocaml[
      # let x = create int32 [| 5 |] [| 3l; 1l; 4l; 1l; 5l |] in
        argsort x
      - : (int32, int32_elt, 'dev) t = [1, 3, 0, 2, 4]
      # let x = create int32 [| 2; 3 |] [| 3l; 1l; 4l; 2l; 5l; 0l |] in
        argsort ~axis:1 x
      - : (int32, int32_elt, 'dev) t = [[1, 0, 2],
                                  [2, 0, 1]]
    ]} *)

(** {2 Linear Algebra}

    Matrix operations and linear algebra functions.

    Most linear algebra functions require floating-point or complex tensors.
    Functions will raise [Invalid_argument] if given integer tensors. *)

val dot : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [dot a b] computes generalized dot product.

    Important: [dot] has different broadcasting behavior than [matmul]:
    - [matmul] broadcasts batch dimensions
    - [dot] does NOT broadcast; it concatenates non-contracted dimensions

    For N-D × M-D arrays: [dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])] This
    can result in much larger output arrays than [matmul].

    Contracts last axis of [a] with:
    - 1-D [b]: the only axis (axis 0)
    - N-D [b]: second-to-last axis (axis -2)

    Dimension rules:
    - 1-D × 1-D: inner product, returns scalar
    - 2-D × 2-D: matrix multiplication
    - N-D × M-D: batched contraction over all but contracted axes

    Supports broadcasting on batch dimensions. Result shape is concatenation of:
    - Broadcasted batch dims
    - Remaining dims from [a] (except last)
    - Remaining dims from [b] (except contracted axis)

    @raise Invalid_argument
      if contraction axes have different sizes or inputs are 0-D

    {@ocaml[
      # let a = create float32 [| 2 |] [| 1.; 2. |] in
        let b = create float32 [| 2 |] [| 3.; 4. |] in
        dot a b |> get_item []
      - : float = 11.
      # let a = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = create float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
        dot a b
      - : (float, float32_elt) t = [[19, 22],
                                    [43, 50]]
      # dot (ones float32 [| 3; 4; 5 |]) (ones float32 [| 5; 6 |]) |> shape
      - : int array = [|3; 4; 6|]
      # dot (ones float32 [| 2; 3; 4; 5 |]) (ones float32 [| 3; 5; 6 |]) |> shape
      - : int array = [|2; 3; 4; 6|]
    ]} *)

val matmul : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [matmul a b] computes matrix multiplication with broadcasting.

    Follows NumPy's \@ operator semantics:
    - 1-D × 1-D: inner product (returns scalar tensor)
    - 1-D × N-D: treated as [1 × k] \@ [... × k × n] → [... × n]
    - N-D × 1-D: treated as [... × m × k] \@ [k × 1] → [... × m]
    - N-D × M-D: batched matrix multiply on last 2 dimensions

    Broadcasting rules:
    - All dimensions except last 2 are broadcast together
    - For 1-D inputs, dimension is temporarily added then removed
    - Inner dimensions must match: a.shape[-1] == b.shape[-2]

    Result shape:
    - Batch dims: broadcast(a.shape[:-2], b.shape[:-2])
    - Matrix dims: [..., a.shape[-2], b.shape[-1]]
    - 1-D adjustments applied after

    @raise Invalid_argument if inputs are 0-D or inner dimensions mismatch

    {@ocaml[
      # let a = create float32 [| 3 |] [| 1.; 2.; 3. |] in
        let b = create float32 [| 3 |] [| 4.; 5.; 6. |] in
        matmul a b |> get_item []
      - : float = 32.
      # let a = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        let b = create float32 [| 2 |] [| 5.; 6. |] in
        matmul a b
      - : (float, float32_elt) t = [17, 39]
      # let a = create float32 [| 2 |] [| 1.; 2. |] in
        let b = create float32 [| 2; 3 |] [| 3.; 4.; 5.; 6.; 7.; 8. |] in
        matmul a b
      - : (float, float32_elt) t = [15, 18, 21]
      # matmul (ones float32 [| 10; 3; 4 |]) (ones float32 [| 10; 4; 5 |]) |> shape
      - : int array = [|10; 3; 5|]
      # matmul (ones float32 [| 1; 3; 4 |]) (ones float32 [| 5; 4; 2 |]) |> shape
      - : int array = [|5; 3; 2|]
    ]} *)

val diagonal :
  ?offset:int ->
  ?axis1:int ->
  ?axis2:int ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [diagonal ?offset ?axis1 ?axis2 a] extracts diagonal from 2-D planes.

    - [offset]: diagonal offset (0=main, positive=above, negative=below)
    - [axis1], [axis2]: axes of 2-D planes (default: last two axes)

    For 2-D array, returns 1-D array of diagonal elements. For N-D array,
    returns array with diagonals from each 2-D subarray.

    @raise Invalid_argument if axis1 = axis2 or axes out of bounds *)

val matrix_transpose : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [matrix_transpose a] transposes matrix dimensions.

    Swaps last two axes: [..., M, N] -> [..., N, M]. For 1-D arrays, returns
    unchanged.

    This is specifically for matrix operations, unlike general [transpose] which
    can permute any axes. *)

val vdot : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [vdot a b] returns dot product of two vectors.

    For complex vectors, conjugates first vector before multiplication. Always
    returns scalar tensor regardless of input shapes. Flattens inputs before
    computation.

    @raise Invalid_argument if inputs have different number of elements *)

val vecdot :
  ?axis:int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [vecdot ?axis x1 x2] computes vector dot product along an axis.

    - [axis]: axis along which to compute dot product (default: -1)

    Unlike [vdot] which always flattens, [vecdot] computes dot products along
    specified axis with broadcasting support.

    @raise Invalid_argument if specified axis dimensions differ *)

val inner : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [inner a b] computes inner product over last axes.

    For 1-D arrays, this is ordinary inner product. For higher dimensions, sums
    products over last axes of a and b.

    @raise Invalid_argument if last dimensions differ *)

val outer : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [outer a b] computes outer product of two vectors.

    Given vectors a[i] and b[j], produces matrix M[i,j] = a[i] * b[j]. Input
    tensors are flattened if not already 1-D.

    {@ocaml[
      # outer (create float32 [|2|] [|1.; 2.|]) (create float32 [|3|] [|3.; 4.; 5.|])
      - : (float, float32_elt) t = [[3, 4, 5],
                                    [6, 8, 10]]
    ]} *)

val tensordot :
  ?axes:int array * int array ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [tensordot ?axes a b] computes tensor contraction along specified axes.

    - [axes]: pair of axis arrays to contract (default: last of a, first of b)

    Generalizes matrix multiplication to arbitrary dimensions.

    @raise Invalid_argument if specified axes have different sizes *)

val einsum : string -> ('a, 'b, 'dev) t array -> ('a, 'b, 'dev) t
(** [einsum subscripts operands] evaluates Einstein summation convention.

    Subscripts string specifies contraction, e.g., "ij,jk->ik" for matmul.
    Repeated indices are summed, free indices form output dimensions.

    {@ocaml[
      # einsum "ij,jk->ik" [|a; b|]  (* matrix multiplication *)
      # einsum "ii->i" [|a|]         (* diagonal *)
      # einsum "ij->ji" [|a|]        (* transpose *)
    ]} *)

val kron : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [kron a b] computes Kronecker product.

    Result has shape [a.shape[i] * b.shape[i] for i in range(ndim)]. Each
    element a[i,j] is replaced by a[i,j] * b. *)

val multi_dot : ('a, 'b, 'dev) t array -> ('a, 'b, 'dev) t
(** [multi_dot arrays] computes chained matrix multiplication optimally.

    Automatically selects the association order that minimizes computational
    cost. Much more efficient than repeated [matmul] for chains of 3+ matrices.

    @raise Invalid_argument if array is empty or shapes incompatible
    @raise Invalid_argument if inputs are not float or complex *)

val matrix_power : ('a, 'b, 'dev) t -> int -> ('a, 'b, 'dev) t
(** [matrix_power a n] raises square matrix to integer power.
    
    - n > 0: a @ a @ ... @ a (n times)
    - n = 0: identity matrix
    - n < 0: inv(a) @ inv(a) @ ... @ inv(a) (|n| times)
    
    @raise Invalid_argument if matrix is not square
    @raise Invalid_argument if input is not float or complex
    @raise Invalid_argument if n < 0 and matrix is singular *)

val cross :
  ?axis:int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [cross ?axis a b] returns cross product of 3-element vectors.

    - [axis]: axis containing vectors (default: last axis)

    @raise Invalid_argument if axis dimension is not 3 *)

(** {3 Matrix Decompositions} *)

val cholesky : ?upper:bool -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [cholesky ?upper a] computes Cholesky decomposition.
    
    - [upper]: return upper triangular if true (default: false)
    
    Returns L (or U) such that a = L @ L.T (or U.T @ U).
    
    @raise Invalid_argument if matrix is not positive-definite
    @raise Invalid_argument if input is not float or complex *)

val qr :
  ?mode:[ `Complete | `Reduced ] ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * ('a, 'b, 'dev) t
(** [qr ?mode a] computes QR decomposition.
    
    - [mode]: [`Reduced] for economy mode (default), [`Complete] for full
    
    Returns (Q, R) where a = Q @ R, Q is orthogonal, R is upper triangular.
    
    @raise Invalid_argument if input is not float or complex *)

val svd :
  ?full_matrices:bool ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * (float, float64_elt, 'dev) t * ('a, 'b, 'dev) t
(** [svd ?full_matrices a] computes singular value decomposition.
    
    - [full_matrices]: compute full U, V matrices (default: false)
    
    Returns (U, S, Vh) where a = U @ diag(S) @ Vh.
    S is 1-D array of singular values in descending order.
    
    @raise Invalid_argument if input is not float or complex *)

val svdvals : ('a, 'b, 'dev) t -> (float, float64_elt, 'dev) t
(** [svdvals a] returns singular values only.

    More efficient than [svd] when only singular values are needed.

    @raise Invalid_argument if input is not float or complex *)

(** {3 Eigenvalues and Eigenvectors} *)

val eig :
  ('a, 'b, 'dev) t ->
  (Complex.t, complex64_elt, 'dev) t * (Complex.t, complex64_elt, 'dev) t
(** [eig a] computes eigenvalues and right eigenvectors.

    Returns (eigenvalues, eigenvectors) for general square matrix. For real
    float32/float64 inputs, outputs are complex32/complex64 since real matrices
    can have complex eigenvalues.

    @raise Invalid_argument if matrix is not square
    @raise Invalid_argument if input is not float or complex *)

val eigh :
  ?uplo:[ `U | `L ] ->
  ('a, 'b, 'dev) t ->
  (float, float64_elt, 'dev) t * ('a, 'b, 'dev) t
(** [eigh ?uplo a] computes eigenvalues for symmetric/Hermitian matrix.

    - [uplo]: use upper (`U) or lower (`L) triangle (default: `L`)

    Returns (eigenvalues, eigenvectors) in ascending order. For real symmetric
    matrices, eigenvalues are guaranteed real. More efficient than [eig] for
    symmetric matrices.

    @raise Invalid_argument if matrix is not square
    @raise Invalid_argument if input is not float or complex *)

val eigvals : ('a, 'b, 'dev) t -> (Complex.t, complex64_elt, 'dev) t
(** [eigvals a] computes eigenvalues only.

    For real inputs, returns complex tensor since eigenvalues may be complex.
    More efficient than [eig] when eigenvectors not needed.

    @raise Invalid_argument if matrix is not square
    @raise Invalid_argument if input is not float or complex *)

val eigvalsh :
  ?uplo:[ `U | `L ] -> ('a, 'b, 'dev) t -> (float, float64_elt, 'dev) t
(** [eigvalsh ?uplo a] computes eigenvalues for symmetric/Hermitian matrix.

    For real symmetric inputs, returns real eigenvalues. More efficient than
    [eigvals] for symmetric matrices.

    @raise Invalid_argument if matrix is not square
    @raise Invalid_argument if input is not float or complex *)

(** {3 Norms and Condition Numbers} *)

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
  ?axes:int array ->
  ?keepdims:bool ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [norm ?ord ?axes ?keepdims x] computes matrix or vector norm.

    - [ord]: norm type (default: Frobenius for matrices, 2-norm for vectors)
    - [`Fro]: Frobenius norm
    - [`Nuc]: nuclear norm (sum of singular values)
    - [`One]: max column sum (for matrices)
    - [`Two]: spectral norm (largest singular value)
    - [`Inf]: max row sum (for matrices)
    - [`NegOne]: min column sum
    - [`NegTwo]: smallest singular value
    - [`NegInf]: min row sum
    - [`P p]: p-norm for vectors
    - [axes]: axes to compute norm over. For matrix norms, must be 2-element
      array
    - [keepdims]: keep reduced dimensions as size 1

    @raise Invalid_argument if ord requires float/complex input *)

val cond :
  ?p:[ `One | `Two | `Inf | `NegOne | `NegTwo | `NegInf | `Fro ] ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [cond ?p x] computes condition number.

    - [p]: norm to use (default: 2-norm)
    - [`One]: 1-norm (max column sum)
    - [`Two]: 2-norm (max singular value)
    - [`Inf]: infinity norm (max row sum)
    - [`NegOne]: -1 norm (min column sum)
    - [`NegTwo]: -2 norm (min singular value)
    - [`NegInf]: -infinity norm (min row sum)
    - [`Fro]: Frobenius norm

    Returns ratio of largest to smallest norm.

    @raise Invalid_argument if input is not float or complex *)

val det : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [det a] computes determinant of square matrix.

    @raise Invalid_argument if matrix is not square
    @raise Invalid_argument if input is not float or complex *)

val slogdet :
  ('a, 'b, 'dev) t ->
  (float, float32_elt, 'dev) t * (float, float32_elt, 'dev) t
(** [slogdet a] computes sign and log of determinant.

    Returns (sign, logdet) where det(a) = sign * exp(logdet). More stable than
    [det] for matrices with very small/large determinants.

    @raise Invalid_argument if matrix is not square
    @raise Invalid_argument if input is not float or complex *)

val matrix_rank :
  ?tol:float -> ?rtol:float -> ?hermitian:bool -> ('a, 'b, 'dev) t -> int
(** [matrix_rank ?tol ?rtol ?hermitian a] returns rank of matrix.

    - [tol]: absolute tolerance for small singular values
    - [rtol]: relative tolerance (default: max(M,N) * eps *
      largest_singular_value)
    - [hermitian]: if true, use more efficient algorithm for Hermitian matrices

    Counts singular values greater than tolerance.

    @raise Invalid_argument if input is not float or complex *)

val trace : ?offset:int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [trace ?offset a] returns sum along diagonal.

    - [offset]: diagonal offset (default: 0, positive for upper diagonals) *)

(** {3 Solving Linear Systems} *)

val solve : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [solve a b] solves linear system a @ x = b for x.
    
    Supports batched operations when a, b have compatible batch dimensions.
    
    @raise Invalid_argument if a is singular
    @raise Invalid_argument if input is not float or complex *)

val lstsq :
  ?rcond:float ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * ('a, 'b, 'dev) t * int * (float, float64_elt, 'dev) t
(** [lstsq ?rcond a b] computes least-squares solution to a @ x = b.
    
    - [rcond]: cutoff for small singular values (default: machine precision)
    
    Returns (solution, residuals, rank, singular_values).
    Handles over/under-determined systems.
    
    @raise Invalid_argument if input is not float or complex *)

val inv : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [inv a] computes inverse of square matrix.

    @raise Invalid_argument if matrix is singular
    @raise Invalid_argument if matrix is not square
    @raise Invalid_argument if input is not float or complex *)

val pinv :
  ?rtol:float -> ?hermitian:bool -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [pinv ?rtol ?hermitian a] computes Moore-Penrose pseudoinverse.

    - [rtol]: relative tolerance for small singular values
    - [hermitian]: if true, use more efficient algorithm for Hermitian matrices

    Handles non-square and singular matrices.

    @raise Invalid_argument if input is not float or complex *)

val tensorsolve :
  ?axes:int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [tensorsolve ?axes a b] solves tensor equation a x = b for x.

    - [axes]: axes in [a] to reorder to end (default: product of b.ndim
      rightmost axes)

    Solves for x such that tensordot(a, x, axes) = b.

    @raise Invalid_argument if shapes incompatible
    @raise Invalid_argument if input is not float or complex *)

val tensorinv : ?ind:int -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [tensorinv ?ind a] computes 'inverse' of N-D array.

    - [ind]: number of first indices involved in inverse sum (default: 2)

    Result is such that tensordot(a, a_inv, ind) = I.

    @raise Invalid_argument if input is not square in specified dimensions
    @raise Invalid_argument if input is not float or complex *)

(** {2 Fourier Transform}

    Fast Fourier Transform (FFT) and related signal processing functions. *)

type fft_norm = [ `Backward | `Forward | `Ortho ]
(** FFT normalization mode:
    - [`Backward]: normalize by 1/n on inverse transform (default)
    - [`Forward]: normalize by 1/n on forward transform
    - [`Ortho]: normalize by 1/√n on both transforms *)

val fft :
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (Complex.t, 'a, 'dev) t
(** [fft x ?axis ?n ?norm] computes discrete Fourier transform over specified
    axis.

    Transforms the last axis if [axis] is [None]. Forward transform is unscaled,
    inverse transform applies 1/N normalization.

    Computing 1D FFT of a signal:
    {@ocaml[
      # let real = Nx.create Nx.float32 [|4|] [|0.; 1.; 2.; 3.|] in
        let imag = Nx.zeros Nx.float32 [|4|] in
        let x = Nx.complex ~real ~imag in
        let result = Nx.fft x ~axes:(Some [|0|]) in
        Nx.real result
      - : (float, float32_elt) t = [6, -2, -2, -2]
    ]} *)

val ifft :
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (Complex.t, 'a, 'dev) t
(** [ifft x ?axis ?n ?norm] computes inverse discrete Fourier transform.

    Divides by product of sizes along transformed axes. *)

val fft2 :
  ?s:int array ->
  ?axes:int array ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (Complex.t, 'a, 'dev) t
(** [fft2 ?s ?axes x] computes 2-dimensional FFT.

    Transforms last two axes by default. Truncates or pads to shape [s] if
    given.

    @raise Invalid_argument if input has less than 2 dimensions

    Computing 2D FFT of a 2x2 matrix:
    {@ocaml[
      # let real = Nx.create Nx.float32 [|2; 2|] [|1.; 2.; 3.; 4.|] in
        let imag = Nx.zeros Nx.float32 [|2; 2|] in
        let x = Nx.complex ~real ~imag in
        Nx.shape (Nx.fft2 x)
      - : int array = [|2; 2|]
    ]} *)

val ifft2 :
  ?s:int array ->
  ?axes:int array ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (Complex.t, 'a, 'dev) t
(** [ifft2 ?s ?axes x] computes 2-dimensional inverse FFT.

    @raise Invalid_argument if input has less than 2 dimensions *)

val fftn :
  ?s:int array ->
  ?axes:int array ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (Complex.t, 'a, 'dev) t
(** [fftn ?s ?axes x] computes N-dimensional FFT.

    Transforms all axes by default. *)

val ifftn :
  ?s:int array ->
  ?axes:int array ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (Complex.t, 'a, 'dev) t
(** [ifftn ?s ?axes x] computes N-dimensional inverse FFT. *)

val rfft :
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (float, 'a, 'dev) t ->
  (Complex.t, complex64_elt, 'dev) t
(** [rfft x ?axis ?n ?norm] computes FFT of real input.

    Returns only non-redundant positive frequencies. Output size along last
    transformed axis is n/2+1 where n is input size.

    Computing real FFT:
    {@ocaml[
      # let x = Nx.create Nx.float32 [|4|] [|0.; 1.; 2.; 3.|] in
        let result = Nx.rfft x ~axes:(Some [|0|]) in
        Nx.shape result
      - : int array = [|3|]
    ]} *)

val irfft :
  ?axis:int ->
  ?n:int ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (float, float64_elt, 'dev) t
(** [irfft x ?axis ?n ?norm] computes inverse FFT returning real output.

    Assumes Hermitian symmetry. Shape [s] specifies output size along
    transformed axes. *)

val rfft2 :
  ?s:int array ->
  ?axes:int array ->
  ?norm:fft_norm ->
  (float, 'a, 'dev) t ->
  (Complex.t, complex64_elt, 'dev) t
(** [rfft2 ?s ?axes x] computes 2D FFT of real input.

    @raise Invalid_argument if input has less than 2 dimensions *)

val irfft2 :
  ?axes:int array ->
  ?s:int array ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (float, float64_elt, 'dev) t
(** [irfft2 ?s ?axes x] computes 2D inverse FFT returning real output.

    @raise Invalid_argument
      if input has less than 2 dimensions or if [s] not specified *)

val rfftn :
  ?s:int array ->
  ?axes:int array ->
  ?norm:fft_norm ->
  (float, 'a, 'dev) t ->
  (Complex.t, complex64_elt, 'dev) t
(** [rfftn ?s ?axes x] computes N-dimensional FFT of real input. *)

val irfftn :
  ?axes:int array ->
  ?s:int array ->
  ?norm:fft_norm ->
  (Complex.t, 'a, 'dev) t ->
  (float, float64_elt, 'dev) t
(** [irfftn ?s ?axes x] computes N-dimensional inverse FFT returning real
    output.

    @raise Invalid_argument if [s] not specified *)

val hfft :
  n:int -> axis:int -> (Complex.t, 'a, 'dev) t -> (float, float64_elt, 'dev) t
(** [hfft x ~n ~axis] computes FFT of Hermitian signal.

    Interprets input as positive frequencies of Hermitian signal. *)

val ihfft :
  n:int -> axis:int -> (float, 'a, 'dev) t -> (Complex.t, complex64_elt, 'dev) t
(** [ihfft x ~n ~axis] computes inverse FFT for Hermitian output. *)

val fftfreq : 'dev device -> ?d:float -> int -> (float, float64_elt, 'dev) t
(** [fftfreq ctx device ?d n] returns DFT sample frequencies.

    For window length [n] and sample spacing [d], returns frequencies
    [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n) if n is even.

    Getting frequencies for 4-point FFT:
    {@ocaml[
      # let ctx = Nx_native.create_context () in
        Nx.fftfreq ctx 4 ()
      - : (float, float64_elt) t = [0, 0.25, -0.5, -0.25]
    ]} *)

val rfftfreq : 'dev device -> ?d:float -> int -> (float, float64_elt, 'dev) t
(** [rfftfreq ctx device n ?d ()] returns positive DFT frequencies.

    Returns [0, 1, ..., n/2] / (d*n). *)

val fftshift : ?axes:int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [fftshift x ?axes] shifts zero-frequency component to center.

    Shifts all axes by default. For visualization of frequency spectra.

    Centering frequency spectrum:
    {@ocaml[
      # let ctx = Nx_native.create_context () in
        let freqs = Nx.fftfreq ctx 5 () in
        Nx.fftshift freqs
      - : (float, float64_elt) t = [-0.4, -0.2, 0, 0.2, 0.4]
    ]} *)

val ifftshift : ?axes:int array -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [ifftshift x ?axes] undoes fftshift. *)

(** {2 Activation Functions}

    Neural network activation functions. *)

val relu : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [relu t] applies Rectified Linear Unit: max(0, x).

    {@ocaml[
      # let x = create float32 [| 5 |] [| -2.; -1.; 0.; 1.; 2. |] in
        relu x
      - : (float, float32_elt) t = [0, 0, 0, 1, 2]
    ]} *)

val relu6 : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [relu6 t] applies ReLU6: min(max(0, x), 6).

    Bounded ReLU used in mobile networks. Clips to [0, 6] range.

    {@ocaml[
      # let x = create float32 [| 3 |] [| -1.; 3.; 8. |] in
        relu6 x
      - : (float, float32_elt) t = [0, 3, 6]
    ]} *)

val sigmoid : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [sigmoid t] applies logistic sigmoid: 1 / (1 + exp(-x)).

    Output in range (0, 1). Symmetric around x=0 where sigmoid(0) = 0.5.

    {@ocaml[
      # sigmoid (scalar float32 0.) |> item []
      - : float = 0.5
      # sigmoid (scalar float32 10.) |> item [] |> Float.round
      - : float = 1.
      # sigmoid (scalar float32 (-10.)) |> item [] |> Float.round
      - : float = 0.
    ]} *)

val hard_sigmoid :
  ?alpha:float -> ?beta:float -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [hard_sigmoid ?alpha ?beta t] applies piecewise linear sigmoid
    approximation.

    Default [alpha = 1/6], [beta = 0.5]. *)

val softplus : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [softplus t] applies smooth ReLU: log(1 + exp(x)).

    Smooth approximation to ReLU. Always positive, differentiable everywhere.

    {@ocaml[
      # softplus (scalar float32 0.) |> item [] |> Float.round
      - : float = 1.
      # softplus (scalar float32 100.) |> item [] |> Float.round
      - : float = infinity
    ]} *)

val silu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [silu t] applies Sigmoid Linear Unit: x * sigmoid(x).

    Also called Swish. Smooth, non-monotonic activation.

    {@ocaml[
      # silu (scalar float32 0.) |> item []
      - : float = 0.
      # silu (scalar float32 1.) |> item [] |> Float.round
      - : float = 1.
      # silu (scalar float32 (-1.)) |> item [] |> Float.round
      - : float = -0.
    ]} *)

val hard_silu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [hard_silu t] applies x * hard_sigmoid(x).

    Piecewise linear approximation of SiLU. More efficient than SiLU.

    {@ocaml[
      # let x = create float32 [| 3 |] [| -3.; 0.; 3. |] in
        hard_silu x
      - : (float, float32_elt) t = [-0, 0, 3]
    ]} *)

val log_sigmoid : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [log_sigmoid t] computes log(sigmoid(x)).

    Numerically stable version of log(1/(1+exp(-x))). Always negative.

    {@ocaml[
      # log_sigmoid (scalar float32 0.) |> item [] |> Float.round
      - : float = -1.
      # log_sigmoid (scalar float32 100.) |> item [] |> Float.abs |> (fun x -> x < 0.001)
      - : bool = true
    ]} *)

val leaky_relu :
  ?negative_slope:float -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [leaky_relu ?negative_slope t] applies Leaky ReLU.

    Default [negative_slope = 0.01]. Returns x if x > 0, else negative_slope *
    x. *)

val hard_tanh : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [hard_tanh t] clips values to [-1, 1].

    Linear in [-1, 1], saturates outside. Cheaper than tanh.

    {@ocaml[
      # let x = create float32 [| 5 |] [| -2.; -0.5; 0.; 0.5; 2. |] in
        hard_tanh x
      - : (float, float32_elt) t = [-1, -0.5, 0, 0.5, 1]
    ]} *)

val elu : ?alpha:float -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [elu ?alpha t] applies Exponential Linear Unit.

    Default [alpha = 1.0]. Returns x if x > 0, else alpha * (exp(x) - 1). Smooth
    for x < 0, helps with vanishing gradients.

    {@ocaml[
      # elu (scalar float32 1.) |> item []
      - : float = 1.
      # elu (scalar float32 0.) |> item []
      - : float = 0.
      # elu (scalar float32 (-1.)) |> item [] |> Float.round
      - : float = -1.
    ]} *)

val selu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [selu t] applies Scaled ELU with fixed alpha=1.67326, lambda=1.0507.

    Self-normalizing activation. Preserves mean 0 and variance 1 in deep
    networks under certain conditions.

    {@ocaml[
      # selu (scalar float32 0.) |> item []
      - : float = 0.
      # selu (scalar float32 1.) |> item [] |> Float.round
      - : float = 1.
    ]} *)

val softmax : ?axes:int array -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [softmax ?axes t] applies softmax normalization.

    Default axis -1. Computes exp(x - max) / sum(exp(x - max)) for numerical
    stability. Output sums to 1 along specified axes.

    {@ocaml[
      # let x = create float32 [| 3 |] [| 1.; 2.; 3. |] in
        softmax x |> to_array |> Array.map Float.round
      - : float array = [|0.; 0.; 1.|]
      # let x = create float32 [| 3 |] [| 1.; 2.; 3. |] in
        sum (softmax x) |> item []
      - : float = 1.
    ]} *)

val gelu_approx : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [gelu_approx t] applies Gaussian Error Linear Unit approximation.

    Smooth activation: x * Φ(x) where Φ is Gaussian CDF. This uses tanh
    approximation for efficiency.

    {@ocaml[
      # gelu_approx (scalar float32 0.) |> item []
      - : float = 0.
      # gelu_approx (scalar float32 1.) |> item [] |> Float.round
      - : float = 1.
    ]} *)

val softsign : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [softsign t] computes x / (|x| + 1).

    Similar to tanh but computationally cheaper. Range (-1, 1).

    {@ocaml[
      # let x = create float32 [| 3 |] [| -10.; 0.; 10. |] in
        softsign x
      - : (float, float32_elt) t = [-0.909091, 0, 0.909091]
    ]} *)

val mish : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [mish t] applies Mish activation: x * tanh(softplus(x)).

    Self-regularizing non-monotonic activation. Smoother than ReLU.

    {@ocaml[
      # mish (scalar float32 0.) |> item [] |> Float.abs |> (fun x -> x < 0.001)
      - : bool = true
      # mish (scalar float32 (-10.)) |> item [] |> Float.round
      - : float = -0.
    ]} *)

(** {2 Convolution and Pooling}

    Neural network convolution and pooling operations. *)

val correlate1d :
  ?groups:int ->
  ?stride:int ->
  ?padding_mode:[ `Full | `Same | `Valid ] ->
  ?dilation:int ->
  ?fillvalue:float ->
  ?bias:(float, 'a, 'dev) t ->
  (float, 'a, 'dev) t ->
  (float, 'a, 'dev) t ->
  (float, 'a, 'dev) t
(** [correlate1d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w]
    computes 1D cross-correlation (no kernel flip).

    - [x]: input [batch_size; channels_in; width]
    - [w]: weights [channels_out; channels_in/groups; kernel_width]
    - [bias]: optional per-channel bias [channels_out]
    - [groups]: split input/output channels into groups (default 1)
    - [stride]: step between windows (default 1)
    - [padding_mode]: `Valid (no pad), `Same (preserve size), `Full (all
      overlaps)
    - [dilation]: spacing between kernel elements (default 1)
    - [fillvalue]: padding value (default 0.0)

    Output width depends on padding:
    - `Valid: (width - dilation*(kernel-1) - 1)/stride + 1
    - `Same: width/stride (rounded up)
    - `Full: (width + dilation*(kernel-1) - 1)/stride + 1

    @raise Invalid_argument if channels_in not divisible by groups

    {@ocaml[
      # let x = create float32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        let w = create float32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
        correlate1d x w |> shape
      - : int array = [|1; 1; 3|]
    ]} *)

val correlate2d :
  ?groups:int ->
  ?stride:int * int ->
  ?padding_mode:[ `Full | `Same | `Valid ] ->
  ?dilation:int * int ->
  ?fillvalue:float ->
  ?bias:(float, 'a, 'dev) t ->
  (float, 'a, 'dev) t ->
  (float, 'a, 'dev) t ->
  (float, 'a, 'dev) t
(** [correlate2d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w]
    computes 2D cross-correlation (no kernel flip).

    - [x]: input [batch; channels_in; height; width]
    - [w]: weights [channels_out; channels_in/groups; kernel_h; kernel_w]
    - [bias]: optional per-channel bias [channels_out]
    - [stride]: (stride_h, stride_w) step between windows (default (1,1))
    - [dilation]: (dilation_h, dilation_w) kernel spacing (default (1,1))
    - [padding_mode]: `Valid (no pad), `Same (preserve size), `Full (all
      overlaps)

    Uses Winograd F(4,3) for 3×3 kernels with stride 1 when beneficial. For
    `Same` with even kernels, pads more on bottom/right (SciPy convention).

    @raise Invalid_argument if channels_in not divisible by groups

    {@ocaml[
      # let image = ones float32 [| 1; 1; 5; 5 |] in
        let sobel_x = create float32 [| 1; 1; 3; 3 |] [| 1.; 0.; -1.; 2.; 0.; -2.; 1.; 0.; -1. |] in
        correlate2d image sobel_x |> shape
      - : int array = [|1; 1; 3; 3|]
    ]} *)

val convolve1d :
  ?groups:int ->
  ?stride:int ->
  ?padding_mode:[< `Full | `Same | `Valid > `Valid ] ->
  ?dilation:int ->
  ?fillvalue:'a ->
  ?bias:('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [convolve1d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w]
    computes 1D convolution (flips kernel before correlation).

    Same parameters as {!correlate1d} but flips kernel. For `Same` with even
    kernels, pads more on left (NumPy convention).

    {@ocaml[
      # let x = create float32 [| 1; 1; 3 |] [| 1.; 2.; 3. |] in
        let w = create float32 [| 1; 1; 2 |] [| 4.; 5. |] in
        convolve1d x w
      - : (float, float32_elt) t = [[[13, 22]]]
    ]} *)

val convolve2d :
  ?groups:int ->
  ?stride:int * int ->
  ?padding_mode:[< `Full | `Same | `Valid > `Valid ] ->
  ?dilation:int * int ->
  ?fillvalue:'a ->
  ?bias:('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [convolve2d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w]
    computes 2D convolution (flips kernel before correlation).

    Same parameters as {!correlate2d} but flips kernel horizontally and
    vertically. For `Same` with even kernels, pads more on top/left.

    {@ocaml[
      # let image = ones float32 [| 1; 1; 5; 5 |] in
        let gaussian = create float32 [| 1; 1; 3; 3 |] [| 1.; 2.; 1.; 2.; 4.; 2.; 1.; 2.; 1. |] in
        convolve2d image (mul_s gaussian (1. /. 16.)) |> shape
      - : int array = [|1; 1; 3; 3|]
    ]} *)

val im2col :
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [im2col ~kernel_size ~stride ~dilation ~padding t] extracts sliding local
    blocks from tensor.

    Extracts patches of size kernel_size from the input tensor at the specified
    stride and dilation.

    - [kernel_size]: size of sliding blocks to extract
    - [stride]: step between consecutive blocks
    - [dilation]: spacing between kernel elements
    - [padding]: (before, after) padding for each spatial dimension

    For a 4D input [batch; channels; height; width], produces output shape
    [batch; channels * kh * kw; num_patches_h; num_patches_w] where kh, kw are
    kernel dimensions and num_patches depends on stride and padding.

    {@ocaml[
      # let x = arange_f float32 0. 16. 1. |> reshape [| 1; 1; 4; 4 |] in
        im2col ~kernel_size:[|2; 2|] ~stride:[|1; 1|]
               ~dilation:[|1; 1|] ~padding:[|(0, 0); (0, 0)|] x |> shape
      - : int array = [|1; 4; 3; 3|]
    ]} *)

val col2im :
  output_size:int array ->
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [col2im ~output_size ~kernel_size ~stride ~dilation ~padding t] combines
    sliding local blocks into tensor.

    This is the inverse of {!im2col}. Accumulates values from the unfolded
    representation back into spatial dimensions. Overlapping regions are summed.

    - [output_size]: target spatial dimensions [height; width]
    - [kernel_size]: size of sliding blocks
    - [stride]: step between consecutive blocks
    - [dilation]: spacing between kernel elements
    - [padding]: (before, after) padding for each spatial dimension

    For input shape [batch; channels * kh * kw; num_patches_h; num_patches_w],
    produces output [batch; channels; height; width].

    {@ocaml[
      # let unfolded = create float32 [| 1; 4; 3; 3 |] (Array.init 36 Float.of_int) in
        col2im ~output_size:[|4; 4|] ~kernel_size:[|2; 2|]
                    ~stride:[|1; 1|] ~dilation:[|1; 1|]
                    ~padding:[|(0, 0); (0, 0)|] unfolded |> shape
      - : int array = [|1; 1; 4; 4|]
    ]} *)

val avg_pool1d :
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?count_include_pad:bool ->
  (float, 'a, 'dev) t ->
  (float, 'a, 'dev) t
(** [avg_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
     ?count_include_pad x] applies 1D average pooling.

    - [kernel_size]: pooling window size
    - [stride]: step between windows (default: kernel_size)
    - [dilation]: spacing between kernel elements (default 1)
    - [padding_spec]: same as convolution padding modes
    - [ceil_mode]: use ceiling for output size calculation (default false)
    - [count_include_pad]: include padding in average (default true)

    Input shape: [batch; channels; width] Output width: (width + 2*pad -
    dilation*(kernel-1) - 1)/stride + 1

    {@ocaml[
      # let x = create float32 [| 1; 1; 4 |] [| 1.; 2.; 3.; 4. |] in
        avg_pool1d ~kernel_size:2 x
      - : (float, float32_elt) t = [[[1.5, 3.5]]]
    ]} *)

val avg_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?count_include_pad:bool ->
  (float, 'a, 'dev) t ->
  (float, 'a, 'dev) t
(** [avg_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
     ?count_include_pad x] applies 2D average pooling.

    - [kernel_size]: (height, width) of pooling window
    - [stride]: (stride_h, stride_w) (default: kernel_size)
    - [dilation]: (dilation_h, dilation_w) (default (1,1))
    - [count_include_pad]: whether padding contributes to denominator

    Input shape: [batch; channels; height; width]

    {@ocaml[
      # let x = create float32 [| 1; 1; 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        avg_pool2d ~kernel_size:(2, 2) x
      - : (float, float32_elt) t = [[[[2.5]]]]
    ]} *)

val max_pool1d :
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?return_indices:bool ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * (int32, int32_elt, 'dev) t option
(** [max_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
     ?return_indices x] applies 1D max pooling.

    - [return_indices]: if true, also returns indices of max values for
      unpooling
    - Other parameters same as {!avg_pool1d}

    Returns (pooled_values, Some indices) if return_indices=true, otherwise
    (pooled_values, None). Indices are flattened positions in input.

    {@ocaml[
      # let x = create float32 [| 1; 1; 4 |] [| 1.; 3.; 2.; 4. |] in
        let vals, idx = max_pool1d ~kernel_size:2 ~return_indices:true x in
        vals, idx
      - : (float, float32_elt) t * (int32, int32_elt, 'dev) t option =
      ([[[3, 4]]], Some [[[1, 1]]])
    ]} *)

val max_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?return_indices:bool ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * (int32, int32_elt, 'dev) t option
(** [max_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
     ?return_indices x] applies 2D max pooling.

    Parameters same as {!max_pool1d} but for 2D. Indices encode flattened
    position within each pooling window.

    {@ocaml[
      # let x = create float32 [| 1; 1; 4; 4 |]
          [| 1.; 2.; 5.; 6.; 3.; 4.; 7.; 8.; 9.; 10.; 13.; 14.; 11.; 12.; 15.; 16. |] in
        let vals, _ = max_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) x in
        vals
      - : (float, float32_elt) t = [[[[4, 8],
                                      [12, 16]]]]
    ]} *)

val min_pool1d :
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?return_indices:bool ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * (int32, int32_elt, 'dev) t option
(** [min_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
     ?return_indices x] applies 1D min pooling.

    - [return_indices]: if true, also returns indices of min values (currently
      returns None)
    - Other parameters same as {!avg_pool1d}

    Returns (pooled_values, None). Index tracking not yet implemented.

    {@ocaml[
      # let x = create float32 [| 1; 1; 4 |] [| 4.; 2.; 3.; 1. |] in
        let vals, _ = min_pool1d ~kernel_size:2 x in
        vals
      - : (float, float32_elt) t = [[[2, 1]]]
    ]} *)

val min_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?return_indices:bool ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t * (int32, int32_elt, 'dev) t option
(** [min_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
     ?return_indices x] applies 2D min pooling.

    Parameters same as {!min_pool1d} but for 2D. Commonly used for morphological
    erosion operations in image processing.

    {@ocaml[
      # let x = create float32 [| 1; 1; 4; 4 |]
          [| 1.; 2.; 5.; 6.; 3.; 4.; 7.; 8.; 9.; 10.; 13.; 14.; 11.; 12.; 15.; 16. |] in
        let vals, _ = min_pool2d ~kernel_size:(2, 2) ~stride:(2, 2) x in
        vals
      - : (float, float32_elt) t = [[[[1, 5],
                                      [9, 13]]]]
    ]} *)

val max_unpool1d :
  (int, uint8_elt, 'dev) t ->
  ('a, 'b, 'dev) t ->
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?output_size_opt:int array ->
  unit ->
  (int, uint8_elt, 'dev) t
(** [max_unpool1d indices values ~kernel_size ?stride ?dilation ?padding_spec
     ?output_size_opt ()] reverses max pooling.

    - [indices]: indices from max_pool1d with return_indices=true
    - [values]: pooled values to place at indexed positions
    - [kernel_size], [stride], [dilation], [padding_spec]: must match original
      pool
    - [output_size_opt]: exact output shape (inferred if not provided)

    Places values at positions indicated by indices, fills rest with zeros.
    Output size computed from input unless explicitly specified.

    @raise Invalid_argument if indices out of bounds

    {@ocaml[
      # let x = create float32 [| 1; 1; 4 |] [| 1.; 3.; 2.; 4. |] in
        let pooled, _ = max_pool1d ~kernel_size:2 x in
        pooled
      - : (float, float32_elt) t = [[[3, 4]]]
    ]} *)

val max_unpool2d :
  (int, uint8_elt, 'dev) t ->
  ('a, 'b, 'dev) t ->
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?output_size_opt:int array ->
  unit ->
  (int, uint8_elt, 'dev) t
(** [max_unpool2d indices values ~kernel_size ?stride ?dilation ?padding_spec
     ?output_size_opt ()] reverses 2D max pooling.

    Same as {!max_unpool1d} but for 2D. Indices encode position within each
    pooling window. Useful for architectures like segmentation networks that
    need to "remember" where maxima came from.

    @raise Invalid_argument if indices out of bounds or shape mismatch

    {@ocaml[
      # let x = create float32 [| 1; 1; 4; 4 |]
          [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.;
             9.; 10.; 11.; 12.; 13.; 14.; 15.; 16. |] in
        let pooled, _ = max_pool2d ~kernel_size:(2,2) x in
        pooled
      - : (float, float32_elt) t = [[[[6, 8],
                                      [14, 16]]]]
    ]} *)

val one_hot : num_classes:int -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [one_hot ~num_classes indices] creates one-hot encoding.

    Adds new last dimension of size [num_classes]. Values must be in
    [\[0, num_classes)]. Out-of-range indices produce zero vectors.

    @raise Invalid_argument if indices not integer type or num_classes <= 0

    {@ocaml[
      # let indices = create int32 [| 3 |] [| 0l; 1l; 3l |] in
        one_hot ~num_classes:4 indices
      - : (int, uint8_elt, 'dev) t = [[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]]
      # let indices = create int32 [| 2; 2 |] [| 0l; 2l; 1l; 0l |] in
        one_hot ~num_classes:3 indices |> shape
      - : int array = [|2; 2; 3|]
    ]} *)

(** {2 Iteration and Mapping}

    Functions to iterate over and transform arrays. *)

val map_item : ('a -> 'a) -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [map_item f t] applies [f] to each element.

    Operates on contiguous data directly. Type-preserving only. *)

val iter_item : ('a -> unit) -> ('a, 'b, 'dev) t -> unit
(** [iter_item f t] applies [f] to each element for side effects. *)

val fold_item : ('a -> 'b -> 'a) -> 'a -> ('b, 'c, 'dev) t -> 'a
(** [fold_item f init t] folds [f] over elements. *)

val map :
  (('a, 'b, 'dev) t -> ('a, 'b, 'dev) t) -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [map f t] applies tensor function [f] to each element as scalar tensor. *)

val iter : (('a, 'b, 'dev) t -> unit) -> ('a, 'b, 'dev) t -> unit
(** [iter f t] applies tensor function [f] to each element. *)

val fold : ('a -> ('b, 'c, 'dev) t -> 'a) -> 'a -> ('b, 'c, 'dev) t -> 'a
(** [fold f init t] folds tensor function over elements. *)

(** {2 Printing and Display}

    Functions to display arrays and convert to strings. *)

val pp_data : Format.formatter -> ('a, 'b, 'dev) t -> unit
(** [pp_data fmt t] pretty-prints tensor data. *)

val format_to_string : (Format.formatter -> 'a -> unit) -> 'a -> string
(** [format_to_string pp x] converts using pretty-printer. *)

val print_with_formatter : (Format.formatter -> 'a -> unit) -> 'a -> unit
(** [print_with_formatter pp x] prints using formatter. *)

val data_to_string : ('a, 'b, 'dev) t -> string
(** [data_to_string t] converts tensor data to string. *)

val print_data : ('a, 'b, 'dev) t -> unit
(** [print_data t] prints tensor data to stdout. *)

val pp_dtype : Format.formatter -> ('a, 'b) dtype -> unit
(** [pp_dtype fmt dt] pretty-prints dtype. *)

val dtype_to_string : ('a, 'b) dtype -> string
(** [dtype_to_string dt] converts dtype to string. *)

val shape_to_string : int array -> string
(** [shape_to_string shape] formats shape as "[2x3x4]". *)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape fmt shape] pretty-prints shape. *)

val pp : Format.formatter -> ('a, 'b, 'dev) t -> unit
(** [pp fmt t] pretty-prints tensor info and data. *)

val print : ('a, 'b, 'dev) t -> unit
(** [print t] prints tensor info and data to stdout. *)

val to_string : ('a, 'b, 'dev) t -> string
(** [to_string t] converts tensor info and data to string. *)

(** {2 Automatic Differentiation}

    Functions for automatic differentiation and gradient computation. *)

val grad :
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t) -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [grad f t] computes the gradient of [f] with respect to [t].

    Returns a tensor of the same shape as [t] containing the gradient values.

    {@ocaml[
      # let x = create float32 [| 2 |] [| 3.; 4. |] in
        let f t = sum (mul_s t 2.) in
        grad f x |> item []
      - : float = 2.
    ]} *)

val grads :
  (('a, 'b, 'dev) t list -> ('c, 'd, 'dev) t) ->
  ('a, 'b, 'dev) t list ->
  ('a, 'b, 'dev) t list
(** [grads f ts] computes gradients of [f] with respect to each tensor in [ts].

    Returns a list of gradients, one for each input tensor.

    {@ocaml[
      # let xs = [create float32 [| 2 |] [| 3. |]; create float32 [| 2 |] [| 4. |]] in
        let f ts = sum (mul_s (List.hd ts) 2.) +. sum (mul_s (List.nth ts 1) 3.) in
        grads f xs |> List.map (fun t -> item t [])
      - : float list = [6.; 12.]
    ]} *)

val value_and_grad :
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t) ->
  ('a, 'b, 'dev) t ->
  ('c, 'd, 'dev) t * ('a, 'b, 'dev) t
(** [value_and_grad f t] computes both the value of [f] and the gradient with
    respect to [t].

    Returns a tuple of the function value and the gradient tensor.

    {@ocaml[
      # let x = create float32 [| 2 |] [| 3. |] in
        let f t = sum (mul_s t 2.) in
        value_and_grad f x |> (fun (v, g) -> (item v [], item g []))
      - : float * float = (6., 2.)
    ]} *)

val value_and_grads :
  (('a, 'b, 'dev) t list -> ('c, 'd, 'dev) t) ->
  ('a, 'b, 'dev) t list ->
  ('c, 'd, 'dev) t * ('a, 'b, 'dev) t list
(** [value_and_grads f ts] computes both the value of [f] and the gradients with
    respect to each tensor in [ts].

    Returns a tuple of the function value and a list of gradient tensors.

    {@ocaml[
      # let xs = [create float32 [| 2 |] [| 3. |]; create float32 [| 2 |] [| 4. |]] in
        let f ts = sum (mul_s (List.hd ts) 2.) +. sum (mul_s (List.nth ts 1) 3.) in
        value_and_grads f xs |> (fun (v, gs) -> (item v [], List.map (fun g -> item g []) gs))
      - : float * float list = (18., [6.; 12.])
    ]} *)

val jvp :
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t) ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('c, 'd, 'dev) t * ('c, 'd, 'dev) t
(** [jvp f primals tangents] computes a Jacobian-vector product (forward-mode
    AD).

    Returns a tuple of (primal_output, tangent_output) where:
    - primal_output = f(primals)
    - tangent_output = Jf(primals) · tangents

    {@ocaml[
      # let x = scalar float32 2. in
        let v = scalar float32 1. in
        let f x = mul x x in
        jvp f x v |> (fun (p, t) -> (item p [], item t []))
      - : float * float = (4., 4.)
    ]} *)

val jvp_aux :
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t * 'e) ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  ('c, 'd, 'dev) t * ('c, 'd, 'dev) t * 'e
(** [jvp_aux f primals tangents] like [jvp] but for functions with auxiliary
    output.

    Returns (primal_output, tangent_output, aux) where aux is the auxiliary
    data.

    {@ocaml[
      # let x = scalar float32 2. in
        let v = scalar float32 1. in
        let f x = (mul x x, shape x) in
        jvp_aux f x v |> (fun (p, t, aux) -> (item p [], item t [], aux))
      - : float * float * int array = (4., 4., [||])
    ]} *)

val jvps :
  (('a, 'b, 'dev) t list -> ('c, 'd, 'dev) t) ->
  ('a, 'b, 'dev) t list ->
  ('a, 'b, 'dev) t list ->
  ('c, 'd, 'dev) t * ('c, 'd, 'dev) t
(** [jvps f primals tangents] computes JVP for functions with multiple inputs.

    Returns (primal_output, tangent_output) for the list of inputs.

    {@ocaml[
      # let xs = [scalar float32 3.; scalar float32 2.] in
        let vs = [scalar float32 1.; scalar float32 0.5] in
        let f inputs = mul (List.hd inputs) (List.nth inputs 1) in
        jvps f xs vs |> (fun (p, t) -> (item p [], item t []))
      - : float * float = (6., 3.5)
    ]} *)

(** {2 Gradient Checking} *)

type method_ = [ `Central | `Forward | `Backward ]
(** Finite difference method to use:
    - [`Central]: (f(x+h) - f(x-h)) / 2h (most accurate)
    - [`Forward]: (f(x+h) - f(x)) / h
    - [`Backward]: (f(x) - f(x-h)) / h *)

val finite_diff :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t) ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [finite_diff ?eps ?method_ f x] computes the gradient of scalar-valued
    function [f] with respect to input [x] using finite differences. The
    function [f] must return a scalar tensor.

    @param eps Step size for finite differences (default: 1e-5)
    @param method_ Finite difference method (default: `Central)
    @param f Function to differentiate (must return scalar)
    @param x Input tensor at which to compute gradient (must be float type)
    @return Gradient tensor with same shape as [x] *)

val finite_diff_jacobian :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t) ->
  ('a, 'b, 'dev) t ->
  ('c, 'd, 'dev) t
(** [finite_diff_jacobian ?eps ?method_ f x] computes the Jacobian matrix of
    function [f] with respect to input [x] using finite differences.

    @param eps Step size for finite differences (default: 1e-5)
    @param method_ Finite difference method (default: `Central)
    @param f Function to differentiate
    @param x Input tensor at which to compute Jacobian (must be float type)
    @return
      Jacobian matrix of shape [output_size × input_size] if f returns
      non-scalar, or gradient vector with same shape as [x] if f returns scalar
*)

type gradient_check_result = {
  max_abs_error : float;
      (** Maximum absolute error between autodiff and finite difference
          gradients *)
  max_rel_error : float;
      (** Maximum relative error between autodiff and finite difference
          gradients *)
  mean_abs_error : float;
      (** Mean absolute error across all checked elements *)
  mean_rel_error : float;
      (** Mean relative error across all checked elements *)
  failed_indices : (int array * float * float * float) list;
      (** List of (index, autodiff_value, finite_diff_value, absolute_error) for
          failed elements *)
  passed : bool;  (** Whether all checked elements passed the tolerance tests *)
  num_checked : int;  (** Total number of elements checked *)
  num_failed : int;  (** Number of elements that failed the tolerance tests *)
}

val check_gradient :
  ?eps:float ->
  ?rtol:float ->
  ?atol:float ->
  ?verbose:bool ->
  ?check_indices:int list option ->
  ?method_:[ `Central | `Forward | `Backward ] ->
  ((float, 'a, 'dev) t -> ('b, 'c, 'dev) t) ->
  (float, 'a, 'dev) t ->
  [ `Pass of gradient_check_result | `Fail of gradient_check_result ]
(** [check_gradient ?eps ?rtol ?atol ?verbose ?check_indices ?method_ f x]
    compares the gradient of [f] at [x] computed via automatic differentiation
    against finite differences.

    @param eps Step size for finite differences (default: 1e-5)
    @param rtol Relative tolerance for comparison (default: 1e-3)
    @param atol Absolute tolerance for comparison (default: 1e-5)
    @param verbose Whether to print detailed error information (default: false)
    @param check_indices Optional list of indices to check (default: all)
    @param method_ Finite difference method (default: `Central)
    @param f Function to check (must return scalar)
    @param x Input tensor at which to check gradient
    @return
      [`Pass result] if all gradients match within tolerance, [`Fail result]
      otherwise

    The check passes if for each element either:
    - absolute_error <= atol, or
    - relative_error <= rtol *)

val check_gradients :
  ?eps:float ->
  ?rtol:float ->
  ?atol:float ->
  ?verbose:bool ->
  ?method_:[ `Central | `Forward | `Backward ] ->
  ((float, 'a, 'dev) t list -> ('b, 'c, 'dev) t) ->
  (float, 'a, 'dev) t list ->
  [ `Pass of gradient_check_result list | `Fail of gradient_check_result list ]
(** [check_gradients ?eps ?rtol ?atol ?verbose ?method_ f xs] compares the
    gradients of [f] with respect to each input in [xs] computed via automatic
    differentiation against finite differences.

    @param eps Step size for finite differences (default: 1e-5)
    @param rtol Relative tolerance for comparison (default: 1e-3)
    @param atol Absolute tolerance for comparison (default: 1e-5)
    @param verbose Whether to print detailed error information (default: false)
    @param method_ Finite difference method (default: `Central)
    @param f Function to check (must return scalar)
    @param xs List of input tensors at which to check gradients
    @return
      [`Pass results] if all gradients match within tolerance, [`Fail results]
      otherwise

    Returns a list of results, one for each input tensor. *)

(** {2 Vectorizing Map (vmap)}

    Functions for mapping computations over batch dimensions. *)

type axis_spec = Vmap.axis_spec =
  | Map of int  (** Map over this axis index *)
  | NoMap  (** Don't map this axis *)

type 'a in_axes_spec = 'a Vmap.in_axes_spec =
  | Single of axis_spec
  | Container of 'a

type 'a out_axes_spec = 'a Vmap.out_axes_spec =
  | OutSingle of int option
  | OutContainer of 'a

val vmap :
  ?in_axes:'a in_axes_spec ->
  ?out_axes:'b out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd, 'dev) t -> ('e, 'f, 'dev) t) ->
  ('c, 'd, 'dev) t ->
  ('e, 'f, 'dev) t
(** [vmap ?in_axes ?out_axes ?axis_name ?axis_size f] creates a vectorized
    version of function [f].

    @param in_axes
      Specifies which input array axes to map over. Default: Single (Map 0) -
      maps over the first axis.
    @param out_axes
      Specifies where the mapped axis should appear in output. Default:
      OutSingle (Some 0) - mapped axis at position 0. Use None to not include
      mapped axis in output.
    @param axis_name
      Optional name for the mapped axis (for collective operations).
    @param axis_size
      Optional size of the mapped axis. Required when in_axes is NoMap.
    @param f The function to be mapped.

    {@ocaml[
      # let batch_x = create float32 [| 10; 3; 3 |] (Array.init 90 float_of_int) in
        let w = create float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let batched_matmul = vmap (fun x -> matmul x w) in
        batched_matmul batch_x |> shape
      - : int array = [| 10; 3; 2 |]
    ]} *)

val vmaps :
  ?in_axes:Vmap.axis_spec list ->
  ?out_axes:'b Vmap.out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd, 'dev) t list -> ('e, 'f, 'dev) t) ->
  ('c, 'd, 'dev) t list ->
  ('e, 'f, 'dev) t
(** [vmaps ?in_axes ?out_axes ?axis_name ?axis_size f] creates a vectorized
    version of function [f] that takes multiple tensor arguments.

    Similar to {!vmap} but for functions taking multiple arguments.

    Examples:
    {[
      let x = create float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      let y = create float32 [| 3; 2 |] [| 10.; 20.; 30.; 40.; 50.; 60. |] in
      let batched_add = vmaps (fun [x; y] -> add x y) in
      batched_add [x; y] |> to_float1
      - : float array = [| 11.; 22.; 33.; 44.; 55.; 66. |]
    ]} *)

(** {2 Random Number Generation}

    JAX-style splittable PRNG for reproducible random number generation. *)

module Rng : sig
  type key
  (** PRNG key type representing the random state *)

  val key : int -> key
  (** Create a PRNG key from a seed value.

      The seed is used to initialize the random state. Same seed produces same
      sequence of random numbers. *)

  val split : ?n:int -> key -> key array
  (** Split a PRNG key into multiple new keys.

      [split key n] returns an array of [n] new independent keys derived from
      the input key. The original key should not be reused after splitting to
      maintain statistical guarantees.

      @param key The key to split
      @param n Number of new keys to generate (default: 2)
      @return Array of new independent keys *)

  val fold_in : key -> int -> key
  (** Fold data into a key to derive a new key.

      [fold_in key data] combines a key with integer data to produce a new key.
      Useful for deriving keys based on iteration indices or other structured
      data.

      @param key The base key
      @param data Integer data to fold into the key
      @return New derived key *)

  val to_int : key -> int
  (** Convert key to integer representation for debugging.

      [to_int key] returns the internal integer representation of the key. This
      is mainly useful for debugging and should not be used to recreate keys. *)

  val uniform :
    key -> 'dev device -> ('a, 'b) dtype -> int array -> ('a, 'b, 'dev) t
  (** Generate uniform random values in \[0, 1).
      
      [uniform key dtype shape] generates a tensor of the given shape with
      values uniformly distributed in the half-open interval \[0, 1).
      
      @param key PRNG key for random generation
      @param dtype Data type of the output tensor
      @param shape Shape of the output tensor
      @return Tensor with uniform random values *)

  val normal :
    key -> 'dev device -> ('a, 'b) dtype -> int array -> ('a, 'b, 'dev) t
  (** Generate standard normal random values.

      [normal key dtype shape] generates a tensor of the given shape with values
      sampled from a standard normal distribution (mean=0, std=1).

      @param key PRNG key for random generation
      @param dtype Data type of the output tensor
      @param shape Shape of the output tensor
      @return Tensor with normal random values *)

  val randint :
    key -> 'dev device -> min:int -> max:int -> int array -> 'dev int32_t
  (** Generate random integers in a range.
      
      [randint key ~min ~max shape] generates a tensor of integers uniformly
      distributed in the half-open interval \[min, max).
      
      @param key PRNG key for random generation
      @param min Minimum value (inclusive)
      @param max Maximum value (exclusive)
      @param shape Shape of the output tensor
      @return Tensor with random integer values *)

  val bernoulli : key -> 'dev device -> p:float -> int array -> 'dev uint8_t
  (** Generate Bernoulli random values.

      [bernoulli key ~p shape] generates a tensor of boolean values where each
      element is true with probability [p].

      @param key PRNG key for random generation
      @param p Probability of true (must be in \[0, 1])
      @param shape Shape of the output tensor
      @return Tensor with boolean random values *)

  val permutation : key -> 'dev device -> int -> 'dev int32_t
  (** Generate random permutation.

      [permutation key n] generates a random permutation of integers from 0 to
      n-1.

      @param key PRNG key for random generation
      @param n Number of elements to permute
      @return 1-D tensor containing a random permutation *)

  val shuffle : key -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
  (** Randomly shuffle the first dimension of a tensor.

      [shuffle key x] returns a copy of tensor [x] with its first dimension
      randomly shuffled.

      @param key PRNG key for random generation
      @param x Tensor to shuffle
      @return Shuffled tensor *)

  val categorical : key -> ?axis:int -> ('a, 'b, 'dev) t -> 'dev int32_t
  (** Sample from a categorical distribution.

      [categorical key logits ?axis] samples indices from a categorical
      distribution defined by logits along the specified axis.

      @param key PRNG key for random generation
      @param logits Unnormalized log probabilities
      @param axis Axis along which to sample (default: -1)
      @return Tensor of sampled indices *)

  val truncated_normal :
    key ->
    'dev device ->
    ('a, 'b) dtype ->
    lower:'a ->
    upper:'a ->
    int array ->
    ('a, 'b, 'dev) t
  (** Generate random values from a truncated normal distribution.

      [truncated_normal key dtype ~lower ~upper shape] generates values from a
      normal distribution truncated to the range [lower, upper].

      @param key PRNG key for random generation
      @param dtype Data type of the output tensor
      @param lower Lower truncation bound
      @param upper Upper truncation bound
      @param shape Shape of the output tensor
      @return Tensor with truncated normal random values *)
end

(** {2 Debugging}

    Functions for debugging, JIT compilation, and gradient computation. *)

val debug : ('a -> 'b) -> 'a -> 'b
(** [debug f x] applies [f] to [x] and prints debug information.

    Useful for inspecting intermediate values during development. *)

val debug_with_context : string -> (unit -> 'a) -> 'a
(** [debug_with_context context f] runs [f] with a debug context.

    Prints the context name before executing [f]. Useful for tracing specific
    computation paths. *)

val debug_push_context : string -> unit
(** [debug_push_context context] pushes a new debug context.

    Use this to mark the start of a specific computation section. The context
    will be printed in debug messages. *)

val debug_pop_context : unit -> unit
(** [debug_pop_context ()] pops the last debug context.

    Use this to mark the end of a specific computation section. The context will
    be removed from the debug stack. *)

(** {2 Just-In-Time Compilation}

    Functions for JIT compilation of tensor operations. *)

val jit :
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t) -> ('a, 'b, 'dev) t -> ('c, 'd, 'dev) t
(** [jit f t] compiles the function [f] for efficient execution on [t].

    Returns a compiled version of [f] that can be called with tensors of the
    same shape and type as [t]. This can significantly speed up repeated calls.

    {@ocaml[
      # let x = create float32 [| 2 |] [| 3. |] in
        let f t = sum (mul_s t 2.) in
        let compiled_f = jit f x in
        compiled_f x |> item []
      - : float = 6.
    ]} *)

val xla :
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t) -> ('a, 'b, 'dev) t -> ('c, 'd, 'dev) t
(** [xla f t] compiles and executes the function [f] using the XLA (Accelerated
    Linear Algebra) compiler.

    This is an alternative to the standard (work-in-progress) [jit] function
    that uses XLA's optimizing compiler. XLA can provide better performance for
    certain workloads, especially those involving many linear algebra
    operations.

    Currently only single-input, single-output functions are supported. The
    compiled function will execute on CPU by default.

    {@ocaml[
      # let x = create float32 [| 100 |] in
        let f t = sin (mul t t) in
        xla f x
      - : (float, float32_elt, 'dev) t = <tensor>
    ]} *)
