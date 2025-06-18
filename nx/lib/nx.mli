(** N-dimensional array operations for OCaml.

    This module provides NumPy-style tensor operations. Tensors are immutable
    views over mutable buffers, supporting broadcasting, slicing, and efficient
    memory layout transformations.

    {2 Type System}

    The type [('a, 'b) t] represents a tensor where ['a] is the OCaml type of
    elements and ['b] is the bigarray element type. For example,
    [(float, float32_elt) t] is a tensor of 32-bit floats.

    {2 Broadcasting}

    Operations automatically broadcast compatible shapes: each dimension must be
    equal or one of them must be 1. Shape [|3; 1; 5|] broadcasts with
    [|1; 4; 5|] to [|3; 4; 5|].

    {2 Memory Layout}

    Tensors can be C-contiguous or strided. Operations return views when
    possible (O(1)), otherwise copy (O(n)). Use {!is_c_contiguous} to check
    layout and {!contiguous} to ensure contiguity. *)

(** {2 Type Definitions}

    Core types, element types, data type specifications, and type aliases. *)

type ('a, 'b) t = ('a, 'b) Nx_native.t
(** [('a, 'b) t] is a tensor with OCaml type ['a] and bigarray type ['b]. *)

type context = Nx_native.context
(** Backend-specific context for tensor operations. *)

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

type float16_t = (float, float16_elt) t
type float32_t = (float, float32_elt) t
type float64_t = (float, float64_elt) t
type int8_t = (int, int8_elt) t
type uint8_t = (int, uint8_elt) t
type int16_t = (int, int16_elt) t
type uint16_t = (int, uint16_elt) t
type int32_t = (int32, int32_elt) t
type int64_t = (int64, int64_elt) t
type std_int_t = (int, int_elt) t
type std_nativeint_t = (nativeint, nativeint_elt) t
type complex32_t = (Complex.t, complex32_elt) t
type complex64_t = (Complex.t, complex64_elt) t

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

type index =
  | I of int
  | L of int list
  | R of int list
      (** Index specification. [I n] for single index, [L [i;j;k]] for fancy
          indexing, [R [start;stop;step]] for ranges (stop is exclusive). *)

(** {2 Array Properties}

    Functions to inspect array dimensions, memory layout, and data access. *)

val data : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [data t] returns underlying bigarray buffer. Buffer may contain data beyond
    tensor bounds for strided views. *)

val shape : ('a, 'b) t -> int array
(** [shape t] returns dimensions. Empty array for scalars. *)

val dtype : ('a, 'b) t -> ('a, 'b) dtype
(** [dtype t] returns data type. *)

val strides : ('a, 'b) t -> int array
(** [strides t] returns byte strides for each dimension. *)

val stride : int -> ('a, 'b) t -> int
(** [stride i t] returns byte stride for dimension [i].

    @raise Invalid_argument if [i] out of bounds *)

val dims : ('a, 'b) t -> int array
(** [dims t] is synonym for {!shape}. *)

val dim : int -> ('a, 'b) t -> int
(** [dim i t] returns size of dimension [i].

    @raise Invalid_argument if [i] out of bounds *)

val ndim : ('a, 'b) t -> int
(** [ndim t] returns number of dimensions. *)

val itemsize : ('a, 'b) t -> int
(** [itemsize t] returns bytes per element. *)

val size : ('a, 'b) t -> int
(** [size t] returns total number of elements. *)

val numel : ('a, 'b) t -> int
(** [numel t] is synonym for {!size}. *)

val nbytes : ('a, 'b) t -> int
(** [nbytes t] returns [size t * itemsize t]. *)

val offset : ('a, 'b) t -> int
(** [offset t] returns element offset in underlying buffer. *)

val is_c_contiguous : ('a, 'b) t -> bool
(** [is_c_contiguous t] returns true if elements are contiguous in C order. *)

val to_bigarray : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_bigarray t] converts to bigarray. Makes contiguous copy. *)

val to_array : ('a, 'b) t -> 'a array
(** [to_array t] converts to OCaml array. Makes contiguous copy. *)

(** {2 Array Creation}

    Functions to create and initialize arrays. *)

val create : ('a, 'b) dtype -> int array -> 'a array -> ('a, 'b) t
(** [create dtype shape data] creates tensor from array [data].

    Length of [data] must equal product of [shape].

    @raise Invalid_argument if array size doesn't match shape

    {[
      create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
      (* [[1.;2.;3.];[4.;5.;6.]] *)
    ]} *)

val init : ('a, 'b) dtype -> int array -> (int array -> 'a) -> ('a, 'b) t
(** [init dtype shape f] creates tensor where element at indices [i] has value
    [f i].

    {[
      init int32 [| 2; 3 |] (fun i -> Int32.of_int (i.(0) + i.(1)))
      (* [[0l;1l;2l];[1l;2l;3l]] *)
    ]} *)

val empty : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [empty dtype shape] allocates uninitialized tensor. *)

val full : ('a, 'b) dtype -> int array -> 'a -> ('a, 'b) t
(** [full dtype shape value] creates tensor filled with [value].

    {[
      full float32 [| 2; 3 |] 3.14
      = [ [ 3.14; 3.14; 3.14 ]; [ 3.14; 3.14; 3.14 ] ]
    ]} *)

val ones : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [ones dtype shape] creates tensor filled with ones. *)

val zeros : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [zeros dtype shape] creates tensor filled with zeros. *)

val scalar : ('a, 'b) dtype -> 'a -> ('a, 'b) t
(** [scalar dtype value] creates scalar tensor containing [value]. *)

val empty_like : ('a, 'b) t -> ('a, 'b) t
(** [empty_like t] creates uninitialized tensor with same shape and dtype as
    [t]. *)

val full_like : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [full_like t value] creates tensor shaped like [t] filled with [value]. *)

val ones_like : ('a, 'b) t -> ('a, 'b) t
(** [ones_like t] creates tensor shaped like [t] filled with ones. *)

val zeros_like : ('a, 'b) t -> ('a, 'b) t
(** [zeros_like t] creates tensor shaped like [t] filled with zeros. *)

val scalar_like : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [scalar_like t value] creates scalar with same dtype as [t]. *)

val eye : ?m:int -> ?k:int -> ('a, 'b) dtype -> int -> ('a, 'b) t
(** [eye ?m ?k dtype n] creates matrix with ones on k-th diagonal.

    Default [m = n] (square), [k = 0] (main diagonal).

    {[
      eye int32 3
      = [ [ 1l; 0l; 0l ]; [ 0l; 1l; 0l ]; [ 0l; 0l; 1l ] ] eye ~k:1 int32 3
      = [ [ 0l; 1l; 0l ]; [ 0l; 0l; 1l ]; [ 0l; 0l; 0l ] ]
    ]} *)

val identity : ('a, 'b) dtype -> int -> ('a, 'b) t
(** [identity dtype n] creates n×n identity matrix. *)

val arange : ('a, 'b) dtype -> int -> int -> int -> ('a, 'b) t
(** [arange dtype start stop step] generates values from [start] to [stop).

    Step must be non-zero. Result length is [(stop - start) / step] rounded
    toward zero.

    @raise Failure if [step = 0]

    {[
      arange int32 0 10 2 = [|0l;2l;4l;6l;8l|]
      arange int32 5 0 (-1) = [|5l;4l;3l;2l;1l|]
    ]} *)

val arange_f : (float, 'a) dtype -> float -> float -> float -> (float, 'a) t
(** [arange_f dtype start stop step] generates float values from [start] to [stop).

    @raise Failure if [step = 0.0] *)

val linspace :
  ('a, 'b) dtype -> ?endpoint:bool -> float -> float -> int -> ('a, 'b) t
(** [linspace dtype ?endpoint start stop count] generates [count] evenly spaced
    values from [start] to [stop].

    If [endpoint] is true (default), [stop] is included.

    {[
      linspace float32 ~endpoint:true 0. 10. 5
      = [| 0.; 2.5; 5.; 7.5; 10. |] linspace float32 ~endpoint:false 0. 10. 5
      = [| 0.; 2.; 4.; 6.; 8. |]
    ]} *)

val logspace :
  (float, 'a) dtype ->
  ?endpoint:bool ->
  ?base:float ->
  float ->
  float ->
  int ->
  (float, 'a) t
(** [logspace dtype ?endpoint ?base start_exp stop_exp count] generates values
    evenly spaced on log scale.

    Returns [base ** x] where x ranges from [start_exp] to [stop_exp]. Default
    [base = 10.0].

    {[
      logspace float32 0. 2. 3
      = [| 1.; 10.; 100. |] logspace float32 ~base:2.0 0. 3. 4
      = [| 1.; 2.; 4.; 8. |]
    ]} *)

val geomspace :
  (float, 'a) dtype -> ?endpoint:bool -> float -> float -> int -> (float, 'a) t
(** [geomspace dtype ?endpoint start stop count] generates values evenly spaced
    on geometric (multiplicative) scale.

    @raise Invalid_argument if [start <= 0.] or [stop <= 0.]

    {[
      geomspace float32 1. 1000. 4 = [| 1.; 10.; 100.; 1000. |]
    ]} *)

val meshgrid :
  ?indexing:[ `xy | `ij ] -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
(** [meshgrid ?indexing x y] creates coordinate grids from 1D arrays.

    Returns (X, Y) where X and Y are 2D arrays representing grid coordinates.

    - [`xy] (default): Cartesian indexing - X changes along columns, Y changes
      along rows
    - [`ij]: Matrix indexing - X changes along rows, Y changes along columns

    @raise Invalid_argument if x or y are not 1D

    {[
      let x = linspace float32 0. 2. 3 in  (* [0.; 1.; 2.] *)
      let y = linspace float32 0. 1. 2 in  (* [0.; 1.] *)
      let xx, yy = meshgrid x y in
      (* xx = [[0.; 1.; 2.];
               [0.; 1.; 2.]] *)
      (* yy = [[0.; 0.; 0.];
               [1.; 1.; 1.]] *)
    ]} *)

val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> ('a, 'b) t
(** [of_bigarray ba] creates tensor from bigarray. Zero-copy if possible. *)

(** {2 Random Number Generation}

    Functions to generate arrays with random values. *)

val rand : ('a, 'b) dtype -> ?seed:int -> int array -> ('a, 'b) t
(** [rand dtype ?seed shape] generates uniform random values in [0, 1).

    Only supports float dtypes.

    @raise Invalid_argument if non-float dtype *)

val randn : ('a, 'b) dtype -> ?seed:int -> int array -> ('a, 'b) t
(** [randn dtype ?seed shape] generates standard normal random values.

    Only supports float dtypes. Uses Box-Muller transform.

    @raise Invalid_argument if non-float dtype *)

val randint :
  ('a, 'b) dtype -> ?seed:int -> ?high:int -> int array -> int -> ('a, 'b) t
(** [randint dtype ?seed ?high shape low] generates integers from [low] to [high).

    Default [high = 10]. Only supports integer dtypes.

    @raise Invalid_argument if non-integer dtype or [low >= high] *)

(** {2 Shape Manipulation}

    Functions to reshape, transpose, and rearrange arrays. *)

val reshape : int array -> ('a, 'b) t -> ('a, 'b) t
(** [reshape shape t] returns view with new shape.

    At most one dimension can be -1 (inferred). Product of dimensions must match
    total elements.

    @raise Invalid_argument if shape incompatible

    {[
      reshape [| 6 |] [ [ 1; 2; 3 ]; [ 4; 5; 6 ] ]
      = [| 1; 2; 3; 4; 5; 6 |] reshape [| 3; -1 |] [| 1; 2; 3; 4; 5; 6 |]
      = [ [ 1; 2 ]; [ 3; 4 ]; [ 5; 6 ] ]
    ]} *)

val broadcast_to : int array -> ('a, 'b) t -> ('a, 'b) t
(** [broadcast_to shape t] broadcasts tensor to target shape.

    Shapes must be broadcast-compatible: each dimension must be equal or one of
    them must be 1.

    @raise Invalid_argument if shapes incompatible

    {[
      broadcast_to [| 3; 3 |] [ [ 1; 2; 3 ] ]
      = [ [ 1; 2; 3 ]; [ 1; 2; 3 ]; [ 1; 2; 3 ] ]
    ]} *)

val broadcasted :
  ?reverse:bool -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
(** [broadcasted ?reverse t1 t2] broadcasts tensors to common shape.

    If [reverse] is true, returns [(t2', t1')] instead of [(t1', t2')]. *)

val expand : int array -> ('a, 'b) t -> ('a, 'b) t
(** [expand shape t] broadcasts tensor where [-1] keeps original dimension.

    {[
      expand [| 3; -1; 5 |] (ones float32 [| 1; 4; 1 |])
      (* Shape: [|3;4;5|] *)
    ]} *)

val flatten : ?start_dim:int -> ?end_dim:int -> ('a, 'b) t -> ('a, 'b) t
(** [flatten ?start_dim ?end_dim t] collapses dimensions from [start_dim] to
    [end_dim] into single dimension.

    Default [start_dim = 0], [end_dim = -1] (last).

    {[
      flatten ~start_dim:1 ~end_dim:2 (zeros float32 [| 2; 3; 4; 5 |])
      (* Shape: [|2;12;5|] *)
    ]} *)

val unflatten : int -> int array -> ('a, 'b) t -> ('a, 'b) t
(** [unflatten dim sizes t] expands dimension [dim] into multiple dimensions.

    Product of [sizes] must equal size of dimension [dim]. One dimension can be
    -1 (inferred).

    @raise Invalid_argument if product mismatch

    {[
      unflatten 1 [| 3; 4 |] (zeros float32 [| 2; 12; 5 |])
      (* Shape: [|2;3;4;5|] *)
    ]} *)

val ravel : ('a, 'b) t -> ('a, 'b) t
(** [ravel t] returns contiguous 1-D view. *)

val squeeze : ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
(** [squeeze ?axes t] removes dimensions of size 1.

    If [axes] specified, only removes those dimensions.

    @raise Invalid_argument if specified axis doesn't have size 1

    {[
      squeeze
        (ones float32 [| 1; 3; 1; 4 |]) (* Shape: [|3;4|] *)
        squeeze ~axes:[| 0; 2 |]
        (ones float32 [| 1; 3; 1; 4 |])
      (* Shape: [|3;4|] *)
    ]} *)

val unsqueeze : ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
(** [unsqueeze ?axes t] inserts dimensions of size 1 at specified positions.

    @raise Invalid_argument if [axes] not specified or contains duplicates

    {[
      unsqueeze ~axes:[| 0; 2 |] [| 1; 2; 3 |] (* Shape: [|1;3;1|] *)
    ]} *)

val squeeze_axis : int -> ('a, 'b) t -> ('a, 'b) t
(** [squeeze_axis axis t] removes dimension [axis] if size is 1.

    @raise Invalid_argument if dimension size is not 1 *)

val unsqueeze_axis : int -> ('a, 'b) t -> ('a, 'b) t
(** [unsqueeze_axis axis t] inserts dimension of size 1 at [axis]. *)

val expand_dims : int array -> ('a, 'b) t -> ('a, 'b) t
(** [expand_dims axes t] is synonym for {!unsqueeze}. *)

val transpose : ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
(** [transpose ?axes t] permutes dimensions.

    Default reverses all dimensions. [axes] must be permutation of [0..ndim-1].

    @raise Invalid_argument if [axes] invalid

    {[
      transpose [ [ 1; 2; 3 ]; [ 4; 5; 6 ] ]
      = [ [ 1; 4 ]; [ 2; 5 ]; [ 3; 6 ] ]
          transpose ~axes:[| 2; 0; 1 |]
          (zeros float32 [| 2; 3; 4 |])
      (* Shape: [|4;2;3|] *)
    ]} *)

val flip : ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
(** [flip ?axes t] reverses order along specified dimensions.

    Default flips all dimensions.

    {[
      flip [ [ 1; 2; 3 ]; [ 4; 5; 6 ] ]
      = [ [ 6; 5; 4 ]; [ 3; 2; 1 ] ]
          flip ~axes:[| 1 |]
          [ [ 1; 2; 3 ]; [ 4; 5; 6 ] ]
      = [ [ 3; 2; 1 ]; [ 6; 5; 4 ] ]
    ]} *)

val moveaxis : int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [moveaxis src dst t] moves dimension from [src] to [dst].

    @raise Invalid_argument if indices out of bounds *)

val swapaxes : int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [swapaxes axis1 axis2 t] exchanges two dimensions.

    @raise Invalid_argument if indices out of bounds *)

val roll : ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [roll ?axis shift t] shifts elements along axis.

    Elements shifted beyond last position wrap around. If [axis] not specified,
    shifts flattened tensor.

    {[
      roll 2 [| 1; 2; 3; 4; 5 |]
      = [| 4; 5; 1; 2; 3 |] roll ~axis:1 1 [ [ 1; 2; 3 ]; [ 4; 5; 6 ] ]
      = [ [ 3; 1; 2 ]; [ 6; 4; 5 ] ]
    ]} *)

val pad : (int * int) array -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [pad padding value t] pads tensor with [value].

    [padding] specifies (before, after) for each dimension.

    {[
      pad [| (1, 1); (2, 2) |] 0 [ [ 1; 2 ]; [ 3; 4 ] ]
      (* [[0;0;0;0;0;0]; [0;0;1;2;0;0]; [0;0;3;4;0;0]; [0;0;0;0;0;0]] *)
    ]} *)

val shrink : (int * int) array -> ('a, 'b) t -> ('a, 'b) t
(** [shrink ranges t] extracts slice from [start] to [stop] (exclusive) for each
    dimension.

    {[
      shrink [| (1, 3); (0, 2) |] [ [ 1; 2; 3 ]; [ 4; 5; 6 ]; [ 7; 8; 9 ] ]
      (* [[4;5];[7;8]] *)
    ]} *)

val tile : int array -> ('a, 'b) t -> ('a, 'b) t
(** [tile reps t] constructs tensor by repeating [t].

    [reps] specifies repetitions per dimension. Length must be >= ndim.

    @raise Invalid_argument if [reps] too short or contains negatives

    {[
      tile [| 2; 3 |] [ [ 1; 2 ] ]
      = [ [ 1; 2; 1; 2; 1; 2 ]; [ 1; 2; 1; 2; 1; 2 ] ]
    ]} *)

val repeat : ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [repeat ?axis count t] repeats elements [count] times.

    If [axis] not specified, repeats flattened tensor.

    {[
      repeat 2 [| 1; 2; 3 |]
      = [| 1; 1; 2; 2; 3; 3 |] repeat ~axis:0 3 [ [ 1; 2 ] ]
      = [ [ 1; 2 ]; [ 1; 2 ]; [ 1; 2 ] ]
    ]} *)

(** {2 Array Combination and Splitting}

    Functions to join and split arrays. *)

val concatenate : ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
(** [concatenate ?axis ts] joins tensors along existing axis.

    All tensors must have same shape except on concatenation axis. If [axis] not
    specified, flattens then concatenates.

    @raise Invalid_argument if empty list or shape mismatch

    {[
      concatenate ~axis:0 [ [ 1; 2 ]; [ 3; 4 ] ] [ [ 5; 6 ] ]
      = [ [ 1; 2 ]; [ 3; 4 ]; [ 5; 6 ] ]
          concatenate
          [ [ 1; 2 ]; [ 3; 4 ] ]
          [ [ 5; 6 ] ]
      = [| 1; 2; 3; 4; 5; 6 |]
    ]} *)

val stack : ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
(** [stack ?axis ts] joins tensors along new axis.

    All tensors must have identical shape. Result rank is input rank + 1.

    @raise Invalid_argument if empty list or shape mismatch

    {[
      stack [ [ 1; 2 ]; [ 3; 4 ] ] [ [ 5; 6 ]; [ 7; 8 ] ]
      (* [[[1;2];[3;4]];[[5;6];[7;8]]] shape [|2;2;2|] *)
    ]} *)

val vstack : ('a, 'b) t list -> ('a, 'b) t
(** [vstack ts] stacks tensors vertically (row-wise).

    1-D tensors become rows. Higher-D tensors concatenate along axis 0.

    {[
      vstack [ [| 1; 2; 3 |]; [| 4; 5; 6 |] ] = [ [ 1; 2; 3 ]; [ 4; 5; 6 ] ]
    ]} *)

val hstack : ('a, 'b) t list -> ('a, 'b) t
(** [hstack ts] stacks tensors horizontally (column-wise).

    1-D tensors concatenate. Higher-D tensors concatenate along axis 1.

    {[
      hstack [ [| 1; 2; 3 |]; [| 4; 5; 6 |] ]
      = [| 1; 2; 3; 4; 5; 6 |] hstack [ [ [ 1 ]; [ 2 ] ]; [ [ 3 ]; [ 4 ] ] ]
      = [ [ 1; 3 ]; [ 2; 4 ] ]
    ]} *)

val dstack : ('a, 'b) t list -> ('a, 'b) t
(** [dstack ts] stacks tensors depth-wise (along third axis).

    Tensors broadcast to at least 3-D before concatenation.

    {[
      dstack [ [ [ 1; 2 ]; [ 3; 4 ] ]; [ [ 5; 6 ]; [ 7; 8 ] ] ]
      (* Shape: [|2;2;2|] *)
    ]} *)

val broadcast_arrays : ('a, 'b) t list -> ('a, 'b) t list
(** [broadcast_arrays ts] broadcasts all tensors to common shape.

    @raise Invalid_argument if shapes incompatible *)

val array_split :
  axis:int ->
  [< `Count of int | `Indices of int list ] ->
  ('a, 'b) t ->
  ('a, 'b) t list
(** [array_split ~axis sections t] splits tensor into multiple parts.

    [`Count n] divides into n parts (possibly unequal). [`Indices [i1;i2;...]]
    splits before each index.

    {[
      array_split ~axis:0 (`Count 3) [| 1; 2; 3; 4; 5 |]
        (* [[|1;2|];[|3;4|];[|5|]] *)
        array_split ~axis:0
        (`Indices [ 2; 4 ])
        [| 1; 2; 3; 4; 5; 6 |]
      (* [[|1;2|];[|3;4|];[|5;6|]] *)
    ]} *)

val split : axis:int -> int -> ('a, 'b) t -> ('a, 'b) t list
(** [split ~axis sections t] splits into equal parts.

    @raise Invalid_argument if axis size not divisible by sections

    {[
      split ~axis:0 2 [ [ 1; 2 ]; [ 3; 4 ]; [ 5; 6 ]; [ 7; 8 ] ]
      (* [[[1;2];[3;4]];[[5;6];[7;8]]] *)
    ]} *)

(** {2 Type Conversion and Copying}

    Functions to convert between types and create copies. *)

val cast : ('c, 'd) dtype -> ('a, 'b) t -> ('c, 'd) t
(** [cast dtype t] converts elements to new dtype.

    Returns copy with same values in new type.

    {[
      cast int32 [ 1.5; 2.7; 3.1 ] = [| 1l; 2l; 3l |]
    ]} *)

val astype : ('a, 'b) dtype -> ('c, 'd) t -> ('a, 'b) t
(** [astype dtype t] is synonym for {!cast}. *)

val contiguous : ('a, 'b) t -> ('a, 'b) t
(** [contiguous t] returns C-contiguous tensor.

    Returns [t] if already contiguous, otherwise copies. *)

val copy : ('a, 'b) t -> ('a, 'b) t
(** [copy t] returns deep copy (always allocates). *)

val blit : ('a, 'b) t -> ('a, 'b) t -> unit
(** [blit src dst] copies [src] into [dst].

    Shapes must match. Modifies [dst] in-place.

    @raise Invalid_argument if shape mismatch *)

val fill : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [fill value t] sets all elements to [value].

    Modifies [t] in-place and returns it. *)

(** {2 Element Access and Slicing}

    Functions to access and modify array elements. *)

val slice : index list -> ('a, 'b) t -> ('a, 'b) t
(** [slice indices t] extracts subtensor.

    - [I n]: select index n
    - [L [i;j;k]]: select indices i, j, k
    - [R [start;stop;step]]: select range with step

    Stop is exclusive. Negative indices count from end.

    {[
      slice [ I 1; R [ 0; -1; 2 ] ] [ [ 1; 2; 3; 4 ]; [ 5; 6; 7; 8 ] ]
      (* [|5;7|] *)
    ]} *)

val set_slice : index list -> ('a, 'b) t -> ('a, 'b) t -> unit
(** [set_slice indices t value] assigns [value] to slice.

    @raise Invalid_argument if shapes incompatible *)

val slice_ranges :
  ?steps:int list -> int list -> int list -> ('a, 'b) t -> ('a, 'b) t
(** [slice_ranges ?steps starts stops t] extracts ranges.

    Equivalent to multiple [R] indices in {!slice}. *)

val set_slice_ranges :
  ?steps:int list -> int list -> int list -> ('a, 'b) t -> ('a, 'b) t -> unit
(** [set_slice_ranges ?steps starts stops t value] assigns to ranges. *)

val get : int list -> ('a, 'b) t -> ('a, 'b) t
(** [get indices t] returns subtensor at indices.

    Returns scalar tensor if all dimensions indexed.

    @raise Invalid_argument if indices out of bounds *)

val set : int list -> ('a, 'b) t -> ('a, 'b) t -> unit
(** [set indices t value] assigns [value] at indices.

    @raise Invalid_argument if indices out of bounds *)

val get_item : int list -> ('a, 'b) t -> 'a
(** [get indices t] returns scalar value at indices. *)

val set_item : int list -> 'a -> ('a, 'b) t -> unit
(** [set indices value t] sets scalar value at indices. *)

(** {2 Basic Arithmetic Operations}

    Element-wise arithmetic operations and their variants. *)

val add : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [add t1 t2] computes element-wise sum with broadcasting.

    @raise Invalid_argument if shapes incompatible *)

val add_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [add_s t scalar] adds scalar to each element. *)

val iadd : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [iadd target value] adds [value] to [target] in-place.

    Returns modified [target]. *)

val radd_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [radd_s scalar t] is [add_s t scalar]. *)

val iadd_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [iadd_s t scalar] adds scalar to [t] in-place. *)

val sub : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sub t1 t2] computes element-wise difference with broadcasting. *)

val sub_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [sub_s t scalar] subtracts scalar from each element. *)

val rsub_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rsub_s scalar t] computes [scalar - t]. *)

val isub : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [isub target value] subtracts [value] from [target] in-place. *)

val isub_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [isub_s t scalar] subtracts scalar from [t] in-place. *)

val mul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [mul t1 t2] computes element-wise product with broadcasting. *)

val mul_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [mul_s t scalar] multiplies each element by scalar. *)

val rmul_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rmul_s scalar t] is [mul_s t scalar]. *)

val imul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [imul target value] multiplies [target] by [value] in-place. *)

val imul_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [imul_s t scalar] multiplies [t] by scalar in-place. *)

val div : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [div t1 t2] computes element-wise division.

    True division for floats, integer division for integers. *)

val div_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [div_s t scalar] divides each element by scalar. *)

val rdiv_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rdiv_s scalar t] computes [scalar / t]. *)

val idiv : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [idiv target value] divides [target] by [value] in-place. *)

val idiv_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [idiv_s t scalar] divides [t] by scalar in-place. *)

val pow : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [pow base exponent] computes element-wise power. *)

val pow_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [pow_s t scalar] raises each element to scalar power. *)

val rpow_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rpow_s scalar t] computes [scalar ** t]. *)

val ipow : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [ipow target exponent] raises [target] to [exponent] in-place. *)

val ipow_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [ipow_s t scalar] raises [t] to scalar power in-place. *)

val mod_ : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [mod_ t1 t2] computes element-wise modulo. *)

val mod_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [mod_s t scalar] computes modulo scalar for each element. *)

val rmod_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rmod_s scalar t] computes [scalar mod t]. *)

val imod : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [imod target divisor] computes modulo in-place. *)

val imod_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [imod_s t scalar] computes modulo scalar in-place. *)

val neg : ('a, 'b) t -> ('a, 'b) t
(** [neg t] negates all elements. *)

(** {2 Mathematical Functions}

    Unary mathematical operations and special functions. *)

val abs : ('a, 'b) t -> ('a, 'b) t
(** [abs t] computes absolute value. *)

val sign : ('a, 'b) t -> ('a, 'b) t
(** [sign t] returns -1, 0, or 1 based on sign.

    For unsigned types, returns 1 for all non-zero values. *)

val square : ('a, 'b) t -> ('a, 'b) t
(** [square t] computes element-wise square. *)

val sqrt : ('a, 'b) t -> ('a, 'b) t
(** [sqrt t] computes element-wise square root. *)

val rsqrt : ('a, 'b) t -> ('a, 'b) t
(** [rsqrt t] computes reciprocal square root. *)

val recip : ('a, 'b) t -> ('a, 'b) t
(** [recip t] computes element-wise reciprocal. *)

val log : (float, 'a) t -> (float, 'a) t
(** [log t] computes natural logarithm. *)

val log2 : ('a, 'b) t -> ('a, 'b) t
(** [log2 t] computes base-2 logarithm. *)

val exp : (float, 'a) t -> (float, 'a) t
(** [exp t] computes exponential. *)

val exp2 : ('a, 'b) t -> ('a, 'b) t
(** [exp2 t] computes 2^x. *)

val sin : ('a, 'b) t -> ('a, 'b) t
(** [sin t] computes sine. *)

val cos : (float, 'a) t -> (float, 'a) t
(** [cos t] computes cosine. *)

val tan : (float, 'a) t -> (float, 'a) t
(** [tan t] computes tangent. *)

val asin : (float, 'a) t -> (float, 'a) t
(** [asin t] computes arcsine. *)

val acos : (float, 'a) t -> (float, 'a) t
(** [acos t] computes arccosine. *)

val atan : (float, 'a) t -> (float, 'a) t
(** [atan t] computes arctangent. *)

val atan2 : (float, 'a) t -> (float, 'a) t -> (float, 'a) t
(** [atan2 y x] computes arctangent of y/x using signs to determine quadrant. *)

val sinh : (float, 'a) t -> (float, 'a) t
(** [sinh t] computes hyperbolic sine. *)

val cosh : (float, 'a) t -> (float, 'a) t
(** [cosh t] computes hyperbolic cosine. *)

val tanh : (float, 'a) t -> (float, 'a) t
(** [tanh t] computes hyperbolic tangent. *)

val asinh : (float, 'a) t -> (float, 'a) t
(** [asinh t] computes inverse hyperbolic sine. *)

val acosh : (float, 'a) t -> (float, 'a) t
(** [acosh t] computes inverse hyperbolic cosine. *)

val atanh : (float, 'a) t -> (float, 'a) t
(** [atanh t] computes inverse hyperbolic tangent. *)

val hypot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [hypot x y] computes sqrt(x² + y²) avoiding overflow. *)

val trunc : ('a, 'b) t -> ('a, 'b) t
(** [trunc t] rounds toward zero. *)

val ceil : (float, 'a) t -> (float, 'a) t
(** [ceil t] rounds up to nearest integer. *)

val floor : (float, 'a) t -> (float, 'a) t
(** [floor t] rounds down to nearest integer. *)

val round : (float, 'a) t -> (float, 'a) t
(** [round t] rounds to nearest integer (half away from zero). *)

val lerp : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [lerp start end_ weight] computes linear interpolation: start + weight *
    (end_ - start). *)

val lerp_scalar_weight : ('a, 'b) t -> ('a, 'b) t -> 'a -> ('a, 'b) t
(** [lerp_scalar_weight start end_ weight] interpolates with scalar weight. *)

(** {2 Comparison and Logical Operations}

    Element-wise comparisons and logical operations. *)

val cmplt : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [cmplt t1 t2] returns 1 where t1 < t2, 0 elsewhere. *)

val less : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [less t1 t2] is synonym for {!cmplt}. *)

val cmpne : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [cmpne t1 t2] returns 1 where t1 ≠ t2, 0 elsewhere. *)

val not_equal : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [not_equal t1 t2] is synonym for {!cmpne}. *)

val cmpeq : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [cmpeq t1 t2] returns 1 where t1 = t2, 0 elsewhere. *)

val equal : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [equal t1 t2] is synonym for {!cmpeq}. *)

val cmpgt : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [cmpgt t1 t2] returns 1 where t1 > t2, 0 elsewhere. *)

val greater : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [greater t1 t2] is synonym for {!cmpgt}. *)

val cmple : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [cmple t1 t2] returns 1 where t1 ≤ t2, 0 elsewhere. *)

val less_equal : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [less_equal t1 t2] is synonym for {!cmple}. *)

val cmpge : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [cmpge t1 t2] returns 1 where t1 ≥ t2, 0 elsewhere. *)

val greater_equal : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [greater_equal t1 t2] is synonym for {!cmpge}. *)

val array_equal : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [array_equal t1 t2] returns scalar 1 if all elements equal, 0 otherwise.

    Handles broadcasting. Returns 0 if shapes incompatible. *)

val maximum : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [maximum t1 t2] returns element-wise maximum. *)

val maximum_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [maximum_s t scalar] returns maximum of each element and scalar. *)

val rmaximum_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rmaximum_s scalar t] is [maximum_s t scalar]. *)

val imaximum : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [imaximum target value] computes maximum in-place. *)

val imaximum_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [imaximum_s t scalar] computes maximum with scalar in-place. *)

val minimum : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [minimum t1 t2] returns element-wise minimum. *)

val minimum_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [minimum_s t scalar] returns minimum of each element and scalar. *)

val rminimum_s : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [rminimum_s scalar t] is [minimum_s t scalar]. *)

val iminimum : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [iminimum target value] computes minimum in-place. *)

val iminimum_s : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [iminimum_s t scalar] computes minimum with scalar in-place. *)

val logical_and : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_and t1 t2] computes element-wise AND.

    Non-zero values are true. *)

val logical_or : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_or t1 t2] computes element-wise OR. *)

val logical_xor : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [logical_xor t1 t2] computes element-wise XOR. *)

val logical_not : ('a, 'b) t -> ('a, 'b) t
(** [logical_not t] computes element-wise NOT.

    Returns 1 - x for boolean tensors. *)

val isinf : (float, 'a) t -> (int, uint8_elt) t
(** [isinf t] returns 1 where infinite, 0 elsewhere. *)

val isnan : ('a, 'b) t -> (int, uint8_elt) t
(** [isnan t] returns 1 where NaN, 0 elsewhere. *)

val isfinite : (float, 'a) t -> (int, uint8_elt) t
(** [isfinite t] returns 1 where finite, 0 elsewhere. *)

val where : (int, uint8_elt) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [where cond if_true if_false] selects elements based on condition.

    Returns [if_true] where [cond] is non-zero, [if_false] elsewhere. Broadcasts
    all three inputs to common shape.

    {[
      where [| 1; 0; 1 |] [| 2; 3; 4 |] [| 5; 6; 7 |] = [| 2; 6; 4 |]
    ]} *)

val clamp : ?min:'a -> ?max:'a -> ('a, 'b) t -> ('a, 'b) t
(** [clamp ?min ?max t] limits values to range.

    Elements below [min] become [min], above [max] become [max]. *)

val clip : ?min:'a -> ?max:'a -> ('a, 'b) t -> ('a, 'b) t
(** [clip ?min ?max t] is synonym for {!clamp}. *)

(** {2 Bitwise Operations}

    Bitwise operations on integer arrays. *)

val bitwise_xor : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_xor t1 t2] computes element-wise XOR. *)

val bitwise_or : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_or t1 t2] computes element-wise OR. *)

val bitwise_and : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [bitwise_and t1 t2] computes element-wise AND. *)

val bitwise_not : ('a, 'b) t -> ('a, 'b) t
(** [bitwise_not t] computes element-wise NOT. *)

val invert : ('a, 'b) t -> ('a, 'b) t
(** [invert t] is synonym for {!bitwise_not}. *)

val lshift : ('a, 'b) t -> int -> ('a, 'b) t
(** [lshift t shift] left-shifts elements by [shift] bits.

    @raise Invalid_argument if shift negative or non-integer dtype *)

val rshift : ('a, 'b) t -> int -> ('a, 'b) t
(** [rshift t shift] right-shifts elements by [shift] bits.

    @raise Invalid_argument if shift negative or non-integer dtype *)

(** {2 Reduction Operations}

    Functions that reduce array dimensions. *)

val sum : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [sum ?axes ?keepdims t] sums elements along specified axes.

    Default sums all axes. If [keepdims] is true, retains reduced dimensions
    with size 1.

    @raise Invalid_argument if any axis is out of bounds

    {[
      sum [ [ 1.; 2. ]; [ 3.; 4. ] ]
      = 10. sum ~axes:[| 0 |] [ [ 1.; 2. ]; [ 3.; 4. ] ]
      = [| 4.; 6. |] sum ~axes:[| 1 |] ~keepdims:true [ [ 1.; 2. ] ]
      = [ [ 3. ] ]
    ]} *)

val max : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [max ?axes ?keepdims t] finds maximum along axes. *)

val min : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [min ?axes ?keepdims t] finds minimum along axes. *)

val prod : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [prod ?axes ?keepdims t] computes product along axes. *)

val mean : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [mean ?axes ?keepdims t] computes arithmetic mean along axes. *)

val var :
  ?axes:int array -> ?keepdims:bool -> ?ddof:int -> ('a, 'b) t -> ('a, 'b) t
(** [var ?axes ?keepdims ?ddof t] computes variance along axes.

    [ddof] is delta degrees of freedom. Default 0 (population variance). Use 1
    for sample variance. *)

val std :
  ?axes:int array -> ?keepdims:bool -> ?ddof:int -> ('a, 'b) t -> ('a, 'b) t
(** [std ?axes ?keepdims ?ddof t] computes standard deviation. *)

val all : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> (int, uint8_elt) t
(** [all ?axes ?keepdims t] tests if all elements are true (non-zero). *)

val any : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> (int, uint8_elt) t
(** [any ?axes ?keepdims t] tests if any element is true (non-zero). *)

val argmax : ?axis:int -> ?keepdims:bool -> ('a, 'b) t -> (int32, int32_elt) t
(** [argmax ?axis ?keepdims t] finds indices of maximum values.

    Returns index of first occurrence for ties. If [axis] not specified,
    operates on flattened tensor. *)

val argmin : ?axis:int -> ?keepdims:bool -> ('a, 'b) t -> (int32, int32_elt) t
(** [argmin ?axis ?keepdims t] finds indices of minimum values. *)

(** {2 Sorting and Searching}

    Functions for sorting arrays and finding indices. *)

val sort :
  ?descending:bool ->
  ?axis:int ->
  ('a, 'b) t ->
  ('a, 'b) t * (int32, int32_elt) t
(** [sort ?descending ?axis t] sorts elements along axis.

    Returns (sorted_values, indices). Default sorts last axis in ascending
    order. Uses stable bitonic sort.

    {[
      let values, indices = sort [| 3; 1; 4; 1; 5 |]
      (* values = [|1;1;3;4;5|], indices = [|1l;3l;0l;2l;4l|] *)
    ]} *)

val argsort :
  ?descending:bool -> ?axis:int -> ('a, 'b) t -> (int32, int32_elt) t
(** [argsort ?descending ?axis t] returns indices that would sort tensor. *)

(** {2 Linear Algebra}

    Matrix operations and linear algebra functions. *)

val dot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [dot a b] computes generalized dot product.

    For 1-D tensors, returns inner product (scalar). For 2-D, performs matrix
    multiplication. Otherwise, contracts last axis of [a] with second-last of
    [b].

    @raise Invalid_argument if contraction axes have different sizes

    {[
      dot [| 1.; 2. |] [| 3.; 4. |]
      = 11. dot [ [ 1.; 2. ]; [ 3.; 4. ] ] [ [ 5.; 6. ]; [ 7.; 8. ] ]
      = [ [ 19.; 22. ]; [ 43.; 50. ] ]
    ]} *)

val matmul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [matmul a b] computes matrix multiplication with broadcasting.

    For 2-D tensors, standard matrix multiply. For N-D, last two dimensions are
    matrix dimensions, others are batch.

    @raise Invalid_argument if inputs are scalars or inner dimensions mismatch

    {[
      matmul [| 1.; 2.; 3. |] [| 4.; 5.; 6. |]
      = 32. (* 1-D: inner product *)
          matmul
          [ [ 1.; 2. ]; [ 3.; 4. ] ]
          [ [ 5. ]; [ 6. ] ]
      = [ [ 17. ]; [ 39. ] ]
      (* 2-D @ 2-D *)
    ]} *)

(** {2 Activation Functions}

    Neural network activation functions. *)

val relu : ('a, 'b) t -> ('a, 'b) t
(** [relu t] applies Rectified Linear Unit: max(0, x). *)

val relu6 : (float, 'a) t -> (float, 'a) t
(** [relu6 t] applies ReLU6: min(max(0, x), 6). *)

val sigmoid : (float, 'a) t -> (float, 'a) t
(** [sigmoid t] applies logistic sigmoid: 1 / (1 + exp(-x)). *)

val hard_sigmoid : ?alpha:float -> ?beta:float -> (float, 'a) t -> (float, 'a) t
(** [hard_sigmoid ?alpha ?beta t] applies piecewise linear sigmoid
    approximation.

    Default [alpha = 1/6], [beta = 0.5]. *)

val softplus : (float, 'a) t -> (float, 'a) t
(** [softplus t] applies smooth ReLU: log(1 + exp(x)). *)

val silu : (float, 'a) t -> (float, 'a) t
(** [silu t] applies Sigmoid Linear Unit: x * sigmoid(x). *)

val hard_silu : (float, 'a) t -> (float, 'a) t
(** [hard_silu t] applies x * hard_sigmoid(x). *)

val log_sigmoid : (float, 'a) t -> (float, 'a) t
(** [log_sigmoid t] computes log(sigmoid(x)). *)

val leaky_relu : ?negative_slope:float -> (float, 'a) t -> (float, 'a) t
(** [leaky_relu ?negative_slope t] applies Leaky ReLU.

    Default [negative_slope = 0.01]. Returns x if x > 0, else negative_slope *
    x. *)

val hard_tanh : (float, 'a) t -> (float, 'a) t
(** [hard_tanh t] clips values to [-1, 1]. *)

val elu : ?alpha:float -> (float, 'a) t -> (float, 'a) t
(** [elu ?alpha t] applies Exponential Linear Unit.

    Default [alpha = 1.0]. Returns x if x > 0, else alpha * (exp(x) - 1). *)

val selu : (float, 'a) t -> (float, 'a) t
(** [selu t] applies Scaled ELU with fixed alpha=1.67326, lambda=1.0507. *)

val softmax : ?axes:int array -> (float, 'a) t -> (float, 'a) t
(** [softmax ?axes t] applies softmax normalization.

    Default axis -1. Computes exp(x - max) / sum(exp(x - max)). *)

val gelu_approx : (float, 'a) t -> (float, 'a) t
(** [gelu_approx t] applies Gaussian Error Linear Unit approximation. *)

val softsign : (float, 'a) t -> (float, 'a) t
(** [softsign t] computes x / (|x| + 1). *)

val mish : (float, 'a) t -> (float, 'a) t
(** [mish t] applies Mish activation: x * tanh(softplus(x)). *)

(** {2 Convolution and Pooling}

    Neural network convolution and pooling operations. *)

val correlate1d :
  ?groups:int ->
  ?stride:int ->
  ?padding_mode:[ `Full | `Same | `Valid ] ->
  ?dilation:int ->
  ?fillvalue:float ->
  ?bias:(float, 'a) t ->
  (float, 'a) t ->
  (float, 'a) t ->
  (float, 'a) t
(** [correlate1d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w]
    computes 1D cross-correlation.

    - [x]: input (batch_size, channels_in, width)
    - [w]: weights (channels_out, channels_in/groups, kernel_width)
    - [bias]: optional bias (channels_out)

    Default [groups=1], [stride=1], [padding_mode=`Valid], [dilation=1]. *)

val correlate2d :
  ?groups:int ->
  ?stride:int * int ->
  ?padding_mode:[ `Full | `Same | `Valid ] ->
  ?dilation:int * int ->
  ?fillvalue:float ->
  ?bias:(float, 'a) t ->
  (float, 'a) t ->
  (float, 'a) t ->
  (float, 'a) t
(** [correlate2d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w]
    computes 2D cross-correlation.

    - [x]: input (batch_size, channels_in, height, width)
    - [w]: weights (channels_out, channels_in/groups, kernel_height,
      kernel_width)

    Uses Winograd F(4,3) for 3×3 kernels with stride 1 when beneficial. *)

val convolve1d :
  ?groups:int ->
  ?stride:int ->
  ?padding_mode:[< `Full | `Same | `Valid > `Valid ] ->
  ?dilation:int ->
  ?fillvalue:'a ->
  ?bias:('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [convolve1d] computes 1D convolution (flips kernel). *)

val convolve2d :
  ?groups:int ->
  ?stride:int * int ->
  ?padding_mode:[< `Full | `Same | `Valid > `Valid ] ->
  ?dilation:int * int ->
  ?fillvalue:'a ->
  ?bias:('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [convolve2d] computes 2D convolution (flips kernel). *)

val avg_pool1d :
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?count_include_pad:bool ->
  (float, 'a) t ->
  (float, 'a) t
(** [avg_pool1d ~kernel_size] applies 1D average pooling.

    Default [stride=kernel_size]. If [ceil_mode], use ceiling instead of floor
    for output size. If [count_include_pad], include padding in average. *)

val avg_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?count_include_pad:bool ->
  (float, 'a) t ->
  (float, 'a) t
(** [avg_pool2d ~kernel_size] applies 2D average pooling. *)

val max_pool1d :
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?return_indices:bool ->
  ('a, 'b) t ->
  ('a, 'b) t * (int32, int32_elt) t option
(** [max_pool1d ~kernel_size] applies 1D max pooling.

    Returns (output, indices) if [return_indices] is true. *)

val max_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?ceil_mode:bool ->
  ?return_indices:bool ->
  ('a, 'b) t ->
  ('a, 'b) t * (int32, int32_elt) t option
(** [max_pool2d ~kernel_size] applies 2D max pooling. *)

val max_unpool1d :
  (int, uint8_elt) t ->
  ('a, 'b) t ->
  kernel_size:int ->
  ?stride:int ->
  ?dilation:int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?output_size_opt:int array ->
  unit ->
  (int, uint8_elt) t
(** [max_unpool1d indices values ~kernel_size] reverses max pooling. *)

val max_unpool2d :
  (int, uint8_elt) t ->
  ('a, 'b) t ->
  kernel_size:int * int ->
  ?stride:int * int ->
  ?dilation:int * int ->
  ?padding_spec:[< `Full | `Same | `Valid > `Valid ] ->
  ?output_size_opt:int array ->
  unit ->
  (int, uint8_elt) t
(** [max_unpool2d indices values ~kernel_size] reverses 2D max pooling. *)

val one_hot : num_classes:int -> ('a, 'b) t -> (int, uint8_elt) t
(** [one_hot ~num_classes indices] creates one-hot encoding.

    Adds new last dimension of size [num_classes].

    @raise Invalid_argument if indices not integer type

    {[
      one_hot ~num_classes:4 [| 0; 1; 3 |]
      (* [[1;0;0;0];[0;1;0;0];[0;0;0;1]] *)
    ]} *)

(** {2 Iteration and Mapping}

    Functions to iterate over and transform arrays. *)

val map_item : ('a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
(** [map_item f t] applies [f] to each element.

    Operates on contiguous data directly. Type-preserving only. *)

val iter_item : ('a -> unit) -> ('a, 'b) t -> unit
(** [iter_item f t] applies [f] to each element for side effects. *)

val fold_item : ('a -> 'b -> 'a) -> 'a -> ('b, 'c) t -> 'a
(** [fold_item f init t] folds [f] over elements. *)

val map : (('a, 'b) t -> ('a, 'b) t) -> ('a, 'b) t -> ('a, 'b) t
(** [map f t] applies tensor function [f] to each element as scalar tensor. *)

val iter : (('a, 'b) t -> unit) -> ('a, 'b) t -> unit
(** [iter f t] applies tensor function [f] to each element. *)

val fold : ('a -> ('b, 'c) t -> 'a) -> 'a -> ('b, 'c) t -> 'a
(** [fold f init t] folds tensor function over elements. *)

(** {2 Printing and Display}

    Functions to display arrays and convert to strings. *)

val pp_data : Format.formatter -> ('a, 'b) t -> unit
(** [pp_data fmt t] pretty-prints tensor data. *)

val format_to_string : (Format.formatter -> 'a -> unit) -> 'a -> string
(** [format_to_string pp x] converts using pretty-printer. *)

val print_with_formatter : (Format.formatter -> 'a -> unit) -> 'a -> unit
(** [print_with_formatter pp x] prints using formatter. *)

val data_to_string : ('a, 'b) t -> string
(** [data_to_string t] converts tensor data to string. *)

val print_data : ('a, 'b) t -> unit
(** [print_data t] prints tensor data to stdout. *)

val pp_dtype : Format.formatter -> ('a, 'b) dtype -> unit
(** [pp_dtype fmt dt] pretty-prints dtype. *)

val dtype_to_string : ('a, 'b) dtype -> string
(** [dtype_to_string dt] converts dtype to string. *)

val shape_to_string : int array -> string
(** [shape_to_string shape] formats shape as "[2x3x4]". *)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape fmt shape] pretty-prints shape. *)

val pp : Format.formatter -> ('a, 'b) t -> unit
(** [pp fmt t] pretty-prints tensor info and data. *)

val print : ('a, 'b) t -> unit
(** [print t] prints tensor info and data to stdout. *)

val to_string : ('a, 'b) t -> string
(** [to_string t] converts tensor info and data to string. *)
