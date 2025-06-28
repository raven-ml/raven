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

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_elt = Bigarray.int8_signed_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type int16_elt = Bigarray.int16_signed_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type int_elt = Bigarray.int_elt
type nativeint_elt = Bigarray.nativeint_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt

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

(** Index specification for slicing *)
type index =
  | I of int  (** Single index *)
  | L of int list  (** List of indices *)
  | R of int list  (** Range [start; stop; step] where stop is exclusive *)

(** {2 Device Management}

    Functions to manage tensor devices and contexts. *)

type 'a device

val native : [ `cpu ] device
(** [native] represents CPU device. Used for tensors stored in native OCaml
    memory. *)

val metal : unit -> [ `metal ] device
(** [metal ()] returns Metal device for GPU tensors.

    Requires Metal backend support. Use for GPU-accelerated computations on
    Apple devices. *)

val cblas : [ `cblas ] device
(** [cblas] represents CBLAS device for tensors stored in CBLAS memory.

    Used for interoperability with CBLAS libraries. *)

val device : ('a, 'b, 'dev) t -> 'dev device
(** [device t] returns device where tensor [t] is stored.

    Returns [native] for CPU tensors, [metal] for Metal tensors, and [cblas] for
    CBLAS tensors. *)

(** {2 Array Properties}

    Functions to inspect array dimensions, memory layout, and data access. *)

val unsafe_data :
  ('a, 'b, 'dev) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [unsafe_data t] returns underlying bigarray buffer.

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

val unsafe_to_bigarray :
  ('a, 'b, 'dev) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
(** [unsafe_to_bigarray t] converts to bigarray.

    Always returns contiguous copy with same shape. Use for interop with
    libraries expecting bigarrays.

    {@ocaml[
      # let t = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
      val t : (float, float32_elt) t = [[1, 2, 3],
                                        [4, 5, 6]]
      # Bigarray.Genarray.dims (unsafe_to_bigarray t) = shape t
      - : bool = true
    ]} *)

val unsafe_to_array : ('a, 'b, 'dev) t -> 'a array
(** [unsafe_to_array t] converts to OCaml array.

    Flattens tensor to 1-D array in row-major (C) order. Always copies.

    {@ocaml[
      # let t = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |]
      val t : (int32, int32_elt, 'dev) t = [[1, 2],
                                      [3, 4]]
      # unsafe_to_array t
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

val of_bigarray :
  'dev device ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t ->
  ('a, 'b, 'dev) t
(** [of_bigarray ba] creates tensor from bigarray.

    Zero-copy when bigarray is contiguous. Creates view sharing same memory.
    Modifications to either affect both.

    {@ocaml[
      # let ba = Bigarray.Array2.create Float32 C_layout 2 3 in
        let t = of_bigarray (Bigarray.genarray_of_array2 ba) in
        t
      - : (float, float32_elt) t = [[0, 0, 0],
                                    [0, 0, 0]]
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
        unsafe_set [ 0 ] 999. y;
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

val slice : index list -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [slice indices t] extracts subtensor.

    - [I n]: select index n (reduces dimension)
    - [L [i;j;k]]: fancy indexing - select indices i, j, k
    - [R [start;stop;step]]: range [\[start, stop)] with step

    Stop is exclusive. Negative indices count from end. Missing indices select
    all. Returns view when possible.

    @raise Invalid_argument if indices out of bounds

    {@ocaml[
      # let x = create int32 [| 2; 4 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l |] in
        slice [ I 1 ] x
      - : (int32, int32_elt, 'dev) t = [5, 6, 7, 8]
      # let x = create int32 [| 5 |] [| 0l; 1l; 2l; 3l; 4l |] in
        slice [ R [ 1; 3 ] ] x
      - : (int32, int32_elt, 'dev) t = [1, 2]
    ]} *)

val set_slice : index list -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> unit
(** [set_slice indices t value] assigns [value] to slice.

    @raise Invalid_argument if shapes incompatible *)

val slice_ranges :
  ?steps:int list ->
  int list ->
  int list ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t
(** [slice_ranges ?steps starts stops t] extracts ranges.

    Equivalent to [slice [R[s0;e0;st0]; R[s1;e1;st1]; ...] t]. Lists must have
    same length ≤ ndim. Default step is 1. Missing dimensions select all.

    @raise Invalid_argument if list lengths differ or indices out of bounds

    {@ocaml[
      # let x = create int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |] in
        slice_ranges [ 0; 1 ] [ 2; 3 ] x
      - : (int32, int32_elt, 'dev) t = [[2, 3],
                                  [5, 6]]
      # slice_ranges ~steps:[ 2; 1 ] [ 0; 0 ] [ 4; 2 ] (eye int32 4)
      - : (int32, int32_elt, 'dev) t = [[1, 0],
                                  [0, 0]]
    ]} *)

val set_slice_ranges :
  ?steps:int list ->
  int list ->
  int list ->
  ('a, 'b, 'dev) t ->
  ('a, 'b, 'dev) t ->
  unit
(** [set_slice_ranges ?steps starts stops t value] assigns to ranges.

    Like {!slice_ranges} but assigns [value] to selected region. Value is
    broadcast to target shape if needed.

    @raise Invalid_argument if shapes incompatible after slicing

    {@ocaml[
      # let x = zeros float32 [| 3; 3 |] in
        set_slice_ranges [ 1; 2 ] [ 2; 3 ] x (ones float32 [| 1; 1 |]);
        unsafe_get [ 1; 2 ] x
      - : float = 1.
    ]} *)

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

val unsafe_get : int list -> ('a, 'b, 'dev) t -> 'a
(** [unsafe_get indices t] returns scalar value at indices.

    Must provide indices for all dimensions.

    @raise Invalid_argument if wrong number of indices or out of bounds *)

val unsafe_set : int list -> 'a -> ('a, 'b, 'dev) t -> unit
(** [unsafe_set indices value t] sets scalar value at indices.

    Must provide indices for all dimensions. Modifies tensor in-place.

    @raise Invalid_argument if wrong number of indices or out of bounds *)

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
        atan2 y x |> unsafe_get [] |> Float.round
      - : float = 1.
      # let y = scalar float32 1. in
        let x = scalar float32 0. in
        atan2 y x |> unsafe_get [] |> Float.round
      - : float = 2.
      # let y = scalar float32 0. in
        let x = scalar float32 0. in
        atan2 y x |> unsafe_get []
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
        hypot x y |> unsafe_get []
      - : float = 5.
      # let x = scalar float64 1e200 in
        let y = scalar float64 1e200 in
        hypot x y |> unsafe_get [] < Float.infinity
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
        lerp start end_ weight |> unsafe_get []
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

val cmpne : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmpne t1 t2] returns 1 where t1 ≠ t2, 0 elsewhere. *)

val not_equal : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [not_equal t1 t2] is synonym for {!cmpne}. *)

val cmpeq : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmpeq t1 t2] returns 1 where t1 = t2, 0 elsewhere. *)

val equal : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [equal t1 t2] is synonym for {!cmpeq}. *)

val cmpgt : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmpgt t1 t2] returns 1 where t1 > t2, 0 elsewhere. *)

val greater : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [greater t1 t2] is synonym for {!cmpgt}. *)

val cmple : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmple t1 t2] returns 1 where t1 ≤ t2, 0 elsewhere. *)

val less_equal :
  ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [less_equal t1 t2] is synonym for {!cmple}. *)

val cmpge : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [cmpge t1 t2] returns 1 where t1 ≥ t2, 0 elsewhere. *)

val greater_equal :
  ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [greater_equal t1 t2] is synonym for {!cmpge}. *)

val array_equal :
  ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> (int, uint8_elt, 'dev) t
(** [array_equal t1 t2] returns scalar 1 if all elements equal, 0 otherwise.

    Broadcasts inputs before comparison. Returns 0 if shapes incompatible.

    {@ocaml[
      # let x = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        let y = create int32 [| 3 |] [| 1l; 2l; 3l |] in
        array_equal x y |> unsafe_get []
      - : int = 1
      # let x = create int32 [| 2 |] [| 1l; 2l |] in
        let y = create int32 [| 2 |] [| 1l; 3l |] in
        array_equal x y |> unsafe_get []
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
        sum x |> unsafe_get []
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
        max x |> unsafe_get []
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
        min x |> unsafe_get []
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
        prod x |> unsafe_get []
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
        mean x |> unsafe_get []
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
        var x |> unsafe_get []
      - : float = 2.
      # let x = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        var ~ddof:1 x |> unsafe_get []
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
        std x |> unsafe_get [] |> Float.round
      - : float = 1.
      # let x = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
        std ~ddof:1 x |> unsafe_get [] |> Float.round
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
        all x |> unsafe_get []
      - : int = 1
      # let x = create int32 [| 3 |] [| 1l; 0l; 3l |] in
        all x |> unsafe_get []
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
        any x |> unsafe_get []
      - : int = 1
      # let x = create int32 [| 3 |] [| 0l; 0l; 0l |] in
        any x |> unsafe_get []
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
        argmax x |> unsafe_get []
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
        argmin x |> unsafe_get []
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

    Matrix operations and linear algebra functions. *)

val dot : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [dot a b] computes generalized dot product.

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
        dot a b |> unsafe_get []
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
        matmul a b |> unsafe_get []
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
      # sigmoid (scalar float32 0.) |> unsafe_get []
      - : float = 0.5
      # sigmoid (scalar float32 10.) |> unsafe_get [] |> Float.round
      - : float = 1.
      # sigmoid (scalar float32 (-10.)) |> unsafe_get [] |> Float.round
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
      # softplus (scalar float32 0.) |> unsafe_get [] |> Float.round
      - : float = 1.
      # softplus (scalar float32 100.) |> unsafe_get [] |> Float.round
      - : float = infinity
    ]} *)

val silu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [silu t] applies Sigmoid Linear Unit: x * sigmoid(x).

    Also called Swish. Smooth, non-monotonic activation.

    {@ocaml[
      # silu (scalar float32 0.) |> unsafe_get []
      - : float = 0.
      # silu (scalar float32 1.) |> unsafe_get [] |> Float.round
      - : float = 1.
      # silu (scalar float32 (-1.)) |> unsafe_get [] |> Float.round
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
      # log_sigmoid (scalar float32 0.) |> unsafe_get [] |> Float.round
      - : float = -1.
      # log_sigmoid (scalar float32 100.) |> unsafe_get [] |> Float.abs |> (fun x -> x < 0.001)
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
      # elu (scalar float32 1.) |> unsafe_get []
      - : float = 1.
      # elu (scalar float32 0.) |> unsafe_get []
      - : float = 0.
      # elu (scalar float32 (-1.)) |> unsafe_get [] |> Float.round
      - : float = -1.
    ]} *)

val selu : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [selu t] applies Scaled ELU with fixed alpha=1.67326, lambda=1.0507.

    Self-normalizing activation. Preserves mean 0 and variance 1 in deep
    networks under certain conditions.

    {@ocaml[
      # selu (scalar float32 0.) |> unsafe_get []
      - : float = 0.
      # selu (scalar float32 1.) |> unsafe_get [] |> Float.round
      - : float = 1.
    ]} *)

val softmax : ?axes:int array -> (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [softmax ?axes t] applies softmax normalization.

    Default axis -1. Computes exp(x - max) / sum(exp(x - max)) for numerical
    stability. Output sums to 1 along specified axes.

    {@ocaml[
      # let x = create float32 [| 3 |] [| 1.; 2.; 3. |] in
        softmax x |> unsafe_to_array |> Array.map Float.round
      - : float array = [|0.; 0.; 1.|]
      # let x = create float32 [| 3 |] [| 1.; 2.; 3. |] in
        sum (softmax x) |> unsafe_get []
      - : float = 1.
    ]} *)

val gelu_approx : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
(** [gelu_approx t] applies Gaussian Error Linear Unit approximation.

    Smooth activation: x * Φ(x) where Φ is Gaussian CDF. This uses tanh
    approximation for efficiency.

    {@ocaml[
      # gelu_approx (scalar float32 0.) |> unsafe_get []
      - : float = 0.
      # gelu_approx (scalar float32 1.) |> unsafe_get [] |> Float.round
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
      # mish (scalar float32 0.) |> unsafe_get [] |> Float.abs |> (fun x -> x < 0.001)
      - : bool = true
      # mish (scalar float32 (-10.)) |> unsafe_get [] |> Float.round
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

val unsafe_map : ('a -> 'a) -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
(** [unsafe_map f t] applies [f] to each element.

    Operates on contiguous data directly. Type-preserving only. *)

val unsafe_iter : ('a -> unit) -> ('a, 'b, 'dev) t -> unit
(** [unsafe_iter f t] applies [f] to each element for side effects. *)

val unsafe_fold : ('a -> 'b -> 'a) -> 'a -> ('b, 'c, 'dev) t -> 'a
(** [unsafe_fold f init t] folds [f] over elements. *)

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
        grad f x |> unsafe_get []
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
        grads f xs |> List.map (fun t -> unsafe_get t [])
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
        value_and_grad f x |> (fun (v, g) -> (unsafe_get v [], unsafe_get g []))
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
        value_and_grads f xs |> (fun (v, gs) -> (unsafe_get v [], List.map (fun g -> unsafe_get g []) gs))
      - : float * float list = (18., [6.; 12.])
    ]} *)

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
        compiled_f x |> unsafe_get []
      - : float = 6.
    ]} *)
