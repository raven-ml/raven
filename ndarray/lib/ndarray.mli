(** N-dimensional array operations in OCaml. *)

(** {1 Core Types and Data Types}

    Basic definitions for tensors and their element types. *)

type ('a, 'b) t
(** An N-dimensional array (tensor). The first type parameter ['a] is the OCaml
    type of elements (e.g., [float], [int], [int32]), and the second ['b] is
    internal storage type (e.g. [float], [double], [uint32], etc.). *)

type layout =
  | C_contiguous
  | Strided
      (** The memory layout of a tensor. [C_contiguous] indicates that the
          tensor is stored in a contiguous block of memory, while [Strided]
          indicates that the tensor's elements are stored with strides (gaps)
          between them. This affects how the data is accessed and manipulated.
      *)

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_elt = Bigarray.int8_signed_elt
type int16_elt = Bigarray.int16_signed_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt

type ('a, 'b) dtype =
  | Float16 : (float, float16_elt) dtype
  | Float32 : (float, float32_elt) dtype
  | Float64 : (float, float64_elt) dtype
  | Int8 : (int, int8_elt) dtype
  | Int16 : (int, int16_elt) dtype
  | Int32 : (int32, int32_elt) dtype
  | Int64 : (int64, int64_elt) dtype
  | UInt8 : (int, uint8_elt) dtype
  | UInt16 : (int, uint16_elt) dtype
  | Complex32 : (Complex.t, complex32_elt) dtype
  | Complex64 : (Complex.t, complex64_elt) dtype
      (** The element data type of a tensor, specifying both the OCaml type ['a]
          and the underlying C type ['b] as per Bigarray conventions. *)

val float16 : (float, float16_elt) dtype
val float32 : (float, float32_elt) dtype
val float64 : (float, float64_elt) dtype
val int8 : (int, int8_elt) dtype
val int16 : (int, int16_elt) dtype
val int32 : (int32, int32_elt) dtype
val int64 : (int64, int64_elt) dtype
val uint8 : (int, uint8_elt) dtype
val uint16 : (int, uint16_elt) dtype
val complex32 : (Complex.t, complex32_elt) dtype
val complex64 : (Complex.t, complex64_elt) dtype

type float16_t = (float, float16_elt) t
type float32_t = (float, float32_elt) t
type float64_t = (float, float64_elt) t
type int8_t = (int, int8_elt) t
type int16_t = (int, int16_elt) t
type int32_t = (int32, int32_elt) t
type int64_t = (int64, int64_elt) t
type uint8_t = (int, uint8_elt) t
type uint16_t = (int, uint16_elt) t
type complex32_t = (Complex.t, complex32_elt) t
type complex64_t = (Complex.t, complex64_elt) t

(** {1 Creating Ndarrays}

    Functions to construct tensors with specific shapes and initial values. *)

val create : ('a, 'b) dtype -> int array -> 'a array -> ('a, 'b) t
(** [create dtype shape data].

    Returns a new tensor with type [dtype] and dimensions [shape], populated
    with values from [data].

    {2 Parameters}
    - [dtype]: element data type
    - [shape]: array specifying the size of each dimension
    - [data]: OCaml array of length equal to [product shape]

    {2 Returns}
    - a fresh tensor with copied data and [C_contiguous] layout

    {2 Raises}
    - [Invalid_argument] if the length of [data] does not match [product shape]

    {2 Examples}
    {[
      let t = create float32 [|2;3|] [|1.;2.;3.;4.;5.;6.|] in
      (* t has shape [|2;3|] and contains the given elements *)
    ]} *)

val init : ('a, 'b) dtype -> int array -> (int array -> 'a) -> ('a, 'b) t
(** [init dtype shape f].

    Creates a new tensor of type [dtype] and shape [shape], where each element
    at index [idx] is initialized by [f idx].

    {2 Parameters}
    - [dtype]: element data type
    - [shape]: array specifying dimensions
    - [f]: function mapping multi-dimensional index to element value

    {2 Returns}
    - a fresh tensor whose element at index [idx] is [f idx]

    {2 Examples}
    {[
      let t = init float32 [|2;2|] (fun [|i; j|] -> float_of_int (i + j)) in
      (* t = [[0.;1.];[1.;2.]] *)
    ]} *)

val scalar : ('a, 'b) dtype -> 'a -> ('a, 'b) t
(** [scalar dtype v].

    Creates a 0-D tensor (scalar) of type [dtype] with value [v]. The resulting
    tensor has no dimensions.

    {2 Parameters}
    - [dtype]: element data type
    - [v]: scalar value

    {2 Returns}
    - a fresh scalar tensor with the given value

    {2 Examples}
    {[
      let s = scalar float32 3.14 in
      (* s is a 0-D tensor with value 3.14 *)
    ]} *)

val copy : ('a, 'b) t -> ('a, 'b) t
(** [copy t].

    Returns a new tensor with the same shape and contents as [t]. Allocates a
    fresh buffer and copies all elements.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - a new tensor identical to [t]

    {2 Notes}
    - If [t] is C_contiguous with offset zero, uses a single buffer blit;
      otherwise falls back to element-wise copy.

    {2 Examples}
    {[
      let b = copy a in
      (* modifications to [b] do not affect [a] *)
    ]} *)

val fill : 'a -> ('a, 'b) t -> unit
(** [fill v t].

    Sets every element of [t] to [v] in place. No new allocation.

    {2 Parameters}
    - [v]: value to assign
    - [t]: tensor to modify

    {2 Notes}
    - If [t] is C_contiguous with offset zero, performs bulk fill (O(N));
      otherwise iterates element-wise respecting strides.

    {2 Examples}
    {[
      fill 0 a
      (* now all elements in [a] are zero *)
    ]} *)

val blit : ('a, 'b) t -> ('a, 'b) t -> unit
(** [blit src dst].

    Copies values from [src] into [dst] (in-place). Requires [src] and [dst]
    have the same shape and dtype; respects strides and offsets.

    {2 Parameters}
    - [src]: source tensor
    - [dst]: destination tensor

    {2 Raises}
    - [Invalid_argument] if tensors differ in rank or shape

    {2 Examples}
    {[
      let a = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
      let b = zeros float32 [| 2; 2 |] in
      blit a b
      (* now b contains the same values as a *)
    ]} *)

val full : ('a, 'b) dtype -> int array -> 'a -> ('a, 'b) t
(** [full dtype shape v].

    Creates a new tensor of type [dtype] and shape [shape], filled with [v].

    {2 Parameters}
    - [dtype]: element data type
    - [shape]: dimensions of the tensor
    - [v]: fill value

    {2 Returns}
    - a fresh tensor where each element equals [v]

    {2 Examples}
    {[
      let m = full float32 [| 2; 3 |] 1.
      (* m = [[1.;1.;1.];[1.;1.;1.]] *)
    ]} *)

val full_like : 'a -> ('a, 'b) t -> ('a, 'b) t
(** [full_like v t].

    Returns a new tensor of the same shape and dtype as [t], filled with [v].

    {2 Parameters}
    - [v]: fill value
    - [t]: reference tensor whose shape and dtype are used

    {2 Returns}
    - a fresh tensor where each element equals [v]

    {2 Examples}
    {[
      let a = create float32 [|2;3|] data in
      let b = full_like 5. a;
      (* b has same shape as a, all elements = 5. *)
    ]} *)

val empty : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [empty dtype shape].

    Returns a new tensor of type [dtype] and shape [shape] with uninitialized
    contents. Allocates a buffer of size [product shape]; contents are
    undefined.

    {2 Parameters}
    - [dtype]: element data type
    - [shape]: dimensions of the tensor

    {2 Returns}
    - a fresh tensor with undefined contents

    {2 Examples}
    {[
      let x = empty float32 [| 2; 2 |]
      (* x contains arbitrary values *)
    ]} *)

val empty_like : ('a, 'b) t -> ('a, 'b) t
(** [empty_like t].

    Returns a new tensor with the same shape and dtype as [t], with
    uninitialized contents. Allocates a buffer of size [size t]; contents
    undefined.

    {2 Parameters}
    - [t]: reference tensor

    {2 Returns}
    - a fresh tensor with undefined contents matching [t]'s shape and dtype

    {2 Examples}
    {[
      let b = empty_like a
    ]} *)

val zeros : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [zeros dtype shape].

    Returns a new tensor of type [dtype] and shape [shape], filled with zeros.

    {2 Parameters}
    - [dtype]: element data type
    - [shape]: dimensions of the tensor

    {2 Returns}
    - a fresh tensor where every element is zero

    {2 Examples}
    {[
      let z = zeros float32 [| 3; 3 |]
    ]} *)

val zeros_like : ('a, 'b) t -> ('a, 'b) t
(** [zeros_like t].

    Returns a new tensor of the same shape and dtype as [t], filled with zeros.

    {2 Parameters}
    - [t]: reference tensor

    {2 Returns}
    - a fresh zero-filled tensor matching [t]'s shape and dtype

    {2 Examples}
    {[
      let z = zeros_like a
    ]} *)

val ones : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [ones dtype shape].

    Returns a new tensor of type [dtype] and shape [shape], filled with ones.

    {2 Parameters}
    - [dtype]: element data type
    - [shape]: dimensions of the tensor

    {2 Returns}
    - a fresh tensor where every element is one

    {2 Examples}
    {[
      let o = ones float32 [| 2; 2 |]
    ]} *)

val ones_like : ('a, 'b) t -> ('a, 'b) t
(** [ones_like t].

    Returns a new tensor of the same shape and dtype as [t], filled with ones.

    {2 Parameters}
    - [t]: reference tensor

    {2 Returns}
    - a fresh one-filled tensor matching [t]'s shape and dtype

    {2 Examples}
    {[
      let o = ones_like a
    ]} *)

val identity : ('a, 'b) dtype -> int -> ('a, 'b) t
(** [identity dtype n].

    Returns a 2-D identity matrix of type [dtype] with shape [|n; n|], i.e.,
    ones on the main diagonal and zeros elsewhere.

    {2 Parameters}
    - [dtype]: element data type
    - [n]: number of rows and columns

    {2 Returns}
    - a fresh identity matrix tensor

    {2 Examples}
    {[
      let I = identity float32 3
      (* I = [[1.;0.;0.];[0.;1.;0.];[0.;0.;1.]] *)
    ]} *)

val eye : ?m:int -> ?k:int -> ('a, 'b) dtype -> int -> ('a, 'b) t
(** [eye ?m ?k dtype n].

    Returns a 2-D tensor of type [dtype] with shape [|m; n|] (m defaults to n),
    containing ones on the k-th diagonal and zeros elsewhere. Positive [k]
    shifts right, negative shifts down.

    {2 Parameters}
    - [?m]: number of rows (default [n])
    - [?k]: diagonal offset (default 0)
    - [dtype]: element data type
    - [n]: number of columns

    {2 Returns}
    - a fresh tensor with ones on the k-th diagonal

    {2 Examples}
    {[
      let E = eye ~m:2 ~k:1 float32 3
      (* E = [[0.;1.;0.];[0.;0.;1.]] *)
    ]} *)

val arange : ('a, 'b) dtype -> int -> int -> int -> ('a, 'b) t
(** [arange dtype start stop step].

    Returns a 1-D tensor of type [dtype] with values starting at [start], then
    incremented by [step], stopping before crossing [stop] (if [step]>0) or
    after crossing [stop] (if [step]<0).

    {2 Parameters}
    - [dtype]: element data type
    - [start]: first value
    - [stop]: exclusive bound
    - [step]: increment (must be non-zero)

    {2 Returns}
    - 1-D tensor of length [max 0 ((stop - start + step - 1) / step)]

    {2 Raises}
    - [Failure] if [step] is zero

    {2 Examples}
    {[
      let v = arange float32 0 5 2 in
      (* v = [|0.;2.;4.|] *)
    ]} *)

val arange_f : (float, 'b) dtype -> float -> float -> float -> (float, 'b) t
(** [arange_f dtype start stop step].

    Returns a 1-D tensor of type [dtype] with values starting at [start], then
    incremented by [step], stopping before crossing [stop] (if [step]>0) or
    after crossing [stop] (if [step]<0). Similar to [arange], but for floating-
    point ranges with careful handling of float precision.

    {2 Parameters}
    - [dtype]: element data type
    - [start]: first value (float)
    - [stop]: exclusive bound (float)
    - [step]: increment (must be non-zero)

    {2 Returns}
    - 1-D tensor of length computed as
      [max 0 (floor((stop - start - ε) / step) + 1)] where ε accounts for
      floating-point round-off

    {2 Raises}
    - [Failure] if [step] is zero

    {2 Examples}
    {[
      let v = arange_f float32 0. 1. 0.3 in
      (* v ≈ [|0.;0.3;0.6;0.9|] *)
    ]} *)

val linspace :
  ('a, 'b) dtype -> ?endpoint:bool -> float -> float -> int -> ('a, 'b) t
(** [linspace dtype ?endpoint start stop count].

    Returns a 1-D tensor of [count] evenly spaced values over the interval from
    [start] to [stop]. If [endpoint=true] (default), includes [stop]; otherwise
    excludes it.

    {2 Parameters}
    - [dtype]: element data type
    - [?endpoint]: include [stop] (default [true])
    - [start]: start of interval
    - [stop]: end of interval
    - [count]: number of samples (must be >= 1)

    {2 Returns}
    - 1-D tensor of length [count] with linearly spaced values

    {2 Raises}
    - [Invalid_argument] if [count] < 1

    {2 Examples}
    {[
      let v = linspace float32 0. 1. 5 in
      (* v = [|0.;0.25;0.5;0.75;1.|] *)
    ]} *)

val logspace :
  (float, 'b) dtype ->
  ?endpoint:bool ->
  ?base:float ->
  float ->
  float ->
  int ->
  (float, 'b) t
(** [logspace dtype ?endpoint ?base start stop count].

    Returns a 1-D tensor of [count] values spaced evenly on a log scale, i.e.
    each element is [base ** x] for x linearly spaced from [start] to [stop]. If
    [?endpoint] is [true] (default), includes [base ** stop]; otherwise excludes
    it. Base defaults to 10.0.

    {2 Parameters}
    - [dtype]: element data type
    - [?endpoint]: include last value (default [true])
    - [?base]: logarithmic base (default [10.0])
    - [start]: starting exponent
    - [stop]: final exponent
    - [count]: number of samples (must be non-negative)

    {2 Returns}
    - 1-D tensor of length [count] with values [base ** exponent]

    {2 Raises}
    - [Invalid_argument] if [count] < 0

    {2 Examples}
    {[
      let v = logspace float32 ~base:10. 0. 2. 3 in
      (* v = [|1.;10.;100.|] *)
    ]} *)

val geomspace :
  (float, 'b) dtype -> ?endpoint:bool -> float -> float -> int -> (float, 'b) t
(** [geomspace dtype ?endpoint start stop count].

    Returns a 1-D tensor of [count] values spaced evenly on a geometric
    progression, i.e. values are [exp(x)] where x are linearly spaced from
    [log start] to [log stop]. If [?endpoint] is [true] (default), includes
    [stop]; otherwise excludes it. Both [start] and [stop] must be positive.

    {2 Parameters}
    - [dtype]: element data type
    - [?endpoint]: include last value (default [true])
    - [start]: starting value (must be > 0)
    - [stop]: final value (must be > 0)
    - [count]: number of samples (must be non-negative)

    {2 Returns}
    - 1-D tensor of length [count] with geometrically spaced values from [start]
      to [stop]

    {2 Raises}
    - [Invalid_argument] if [start] <= 0 or [stop] <= 0 or [count] < 0

    {2 Examples}
    {[
      let v = geomspace float32 1. 1000. 4 in
      (* v = [|1.;10.;100.;1000.|] *)
    ]} *)

(** {1 Array Properties}

    Query metadata such as shape, strides, storage size, etc. *)

val data : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [data t].

    Return the raw host buffer backing [t] as a 1-D Bigarray (C layout). No data
    is copied; mutating the returned buffer (respecting [offset] and [strides])
    will affect [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - underlying Bigarray.Array1.t buffer

    {2 Notes}
    - For views with non-zero [offset] or non-unit [strides], buffer access must
      account for these metadata. *)

val ndim : ('a, 'b) t -> int
(** [ndim t].

    Return the number of dimensions (rank) of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - rank of [t] (0 for scalar) *)

val shape : ('a, 'b) t -> int array
(** [shape t].

    Return the dimensions of [t] as an int array of length [ndim t]. No copy.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - array of sizes for each dimension *)

val dim : int -> ('a, 'b) t -> int
(** [dim i t].

    Return the length of [t] along axis [i].

    {2 Parameters}
    - [i]: axis index (0 <= i < ndim t)
    - [t]: input tensor

    {2 Returns}
    - size of dimension [i]

    {2 Raises}
    - [Invalid_argument] if [i] is out of bounds *)

val dims : ('a, 'b) t -> int array
(** [dims t].

    Alias for [shape], returning the dimensions of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - array of sizes for each dimension *)

val dtype : ('a, 'b) t -> ('a, 'b) dtype
(** [dtype t].

    Return the data type descriptor of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - dtype of [t] *)

val nbytes : ('a, 'b) t -> int
(** [nbytes t].

    Return the total buffer size in bytes for [t]. For 0-D tensors, yields 0.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - number of bytes (size t * itemsize) *)

val size : ('a, 'b) t -> int
(** [size t].

    Return the total number of elements in [t] (product of dimensions). Returns
    1 for 0-D tensors.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - element count of [t] *)

val stride : int -> ('a, 'b) t -> int
(** [stride i t].

    Return the stride (in elements) for axis [i] of [t].

    {2 Parameters}
    - [i]: axis index (0 <= i < ndim t)
    - [t]: input tensor

    {2 Returns}
    - step size in elements along axis [i]

    {2 Raises}
    - [Invalid_argument] if [i] is out of bounds *)

val strides : ('a, 'b) t -> int array
(** [strides t].

    Return the per-axis strides (in elements) of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - array of strides for each dimension *)

val itemsize : ('a, 'b) t -> int
(** [itemsize t].

    Return the size in bytes of a single element in [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - byte size per element (e.g. 4 for float32) *)

val offset : ('a, 'b) t -> int
(** [offset t].

    Return the buffer offset (in elements) of the first logical element of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - element index in the buffer for the first element *)

val layout : ('a, 'b) t -> layout
(** [layout t].

    Return the memory layout of [t], either [C_contiguous] (row-major) or
    [Strided] (non-contiguous).

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - tensor layout *)

(** {1 Element Access and Views}

    Get, set, and create indexed views of tensors. *)

val get_item : int array -> ('a, 'b) t -> 'a
(** [get_item idx t].

    Returns the element of [t] at multi-dimensional index [idx]. Performs bounds
    checks for each axis; no new allocation.

    {2 Parameters}
    - [idx]: array of indices, one per dimension
    - [t]: input tensor

    {2 Returns}
    - the scalar element at position [idx]

    {2 Raises}
    - [Invalid_argument] if [Array.length idx] <> [ndim t] or any index out of
      bounds

    {2 Examples}
    {[
      let a = create int32 [|2;2|] [|1l;2l;3l;4l|] in
      let x = get_item [|1;0|] a in
      (* x = 3l *)
    ]} *)

val set_item : int array -> 'a -> ('a, 'b) t -> unit
(** [set_item idx v t].

    Sets the element of [t] at multi-dimensional index [idx] to [v] in place.
    Performs bounds checks for each axis; no new allocation.

    {2 Parameters}
    - [idx]: array of indices, one per dimension
    - [v]: new value to assign
    - [t]: tensor to modify

    {2 Raises}
    - [Invalid_argument] if [Array.length idx] <> [ndim t] or any index out of
      bounds

    {2 Examples}
    {[
      let a = create float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
      set_item [| 0; 1 |] 10. a
      (* now a.(0,1) = 10. *)
    ]} *)

val get : int array -> ('a, 'b) t -> ('a, 'b) t
(** [get indices t].

    Returns a view of [t] by indexing the first [length indices] dimensions at
    the given [indices]. The resulting tensor has dimensionality
    [ndim t - length indices] and shares storage with [t]; no data is copied.

    {2 Parameters}
    - [indices]: array of indices for each of the first dimensions
    - [t]: input tensor

    {2 Returns}
    - a tensor view with specified axes removed

    {2 Raises}
    - [Invalid_argument] if more indices are provided than dimensions of [t], or
      if any index is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;3;4|] data in
      let sub = get [|1;2|] a in
      (* sub has shape [|4|] corresponding to a.(1,2, :) *)
    ]} *)

val set : int array -> ('a, 'b) t -> ('a, 'b) t -> unit
(** [set indices src t].

    Assigns elements from [src] into [t] at positions specified by [indices] on
    the first [length indices] dimensions. Let [view] be the sub-array of [t]
    obtained by indexing those dimensions; [src] must have the same shape as
    [view]. Data is copied element-wise in-place.

    {2 Parameters}
    - [indices]: array of indices for each of the first dimensions
    - [src]: source tensor of values to assign
    - [t]: target tensor to modify

    {2 Raises}
    - [Invalid_argument] if too many indices, mismatched dtypes, or shape
      mismatch between [src] and the target view

    {2 Examples}
    {[
      let a = zeros float32 [| 2; 3 |] in
      let v = full float32 [| 3 |] 5. in
      set [| 1 |] v a
      (* now a.(1, :) = [|5.;5.;5.|] *)
    ]} *)

val slice :
  ?steps:int array -> int array -> int array -> ('a, 'b) t -> ('a, 'b) t
(** [slice ?steps starts stops t].

    Returns a view of [t] defined by slicing each axis from [starts.(i)]
    (inclusive) to [stops.(i)] (exclusive) with optional [steps.(i)]. No data is
    copied; allocation is O(1).

    {2 Parameters}
    - [steps]: optional strides array per axis; default all ones
    - [starts]: start indices for each dimension
    - [stops]: stop indices (exclusive) for each dimension
    - [t]: input tensor

    {2 Returns}
    - a view tensor with shape derived from [starts], [stops], and [steps]

    {2 Raises}
    - [Invalid_argument] if lengths of [starts], [stops], or [steps] (if
      provided) mismatch [ndim t], if any step is zero, or if computed indices
      are invalid

    {2 Examples}
    {[
      let a = create float32 [|3;3|] [|0.;1.;2.;3.;4.;5.;6.;7.;8.|] in
      let b = slice ~steps:[|1;2|] [|0;0|] [|3;3|] a in
      (* b has shape [|3;2|] with rows [[0.;2.];[3.;5.];[6.;8.]] *)
    ]} *)

val set_slice :
  ?steps:int array -> int array -> int array -> ('a, 'b) t -> ('a, 'b) t -> unit
(** [set_slice ?steps starts stops src dst].

    Assigns elements from [src] into [dst] at positions specified by slicing
    each axis from [starts.(i)] (inclusive) to [stops.(i)] (exclusive) with
    optional [steps.(i)]. The shape of [src] must match the view of [dst]
    defined by the slice. Data is copied element-wise in-place.

    {2 Parameters}
    - [steps]: optional strides array per axis; default all ones
    - [starts]: start indices for each dimension
    - [stops]: stop indices (exclusive) for each dimension
    - [src]: source tensor of values to assign
    - [dst]: target tensor to modify

    {2 Raises}
    - [Invalid_argument] if lengths of [starts], [stops], or [steps] (if
      provided) mismatch [ndim dst], if any step is zero, or if computed indices
      are invalid

    {2 Examples}
    {[
      let a = create float32 [| 3; 3 |] data in
      let b = zeros float32 [| 3; 3 |] in
      set_slice ~steps:[| 1; 2 |] [| 0; 0 |] [| 3; 3 |] a b
      (* now b has rows [[0.;2.];[3.;5.];[6.;8.]] *)
    ]} *)

(** {1 Array Manipulation}

    Reshaping, slicing, stacking, padding, broadcasting, tiling, repeating,
    flipping, rolling, and axis movement. *)

val flatten : ('a, 'b) t -> ('a, 'b) t
(** [flatten t].

    Return a 1-D view of [t] by collapsing all dimensions into one, sharing the
    underlying buffer. Does not allocate or copy data.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - a 1-D tensor view of [t]

    {2 Examples}
    {[
      let a = create float32 [|2;3|] [|1.;2.;3.;4.;5.;6.|] in
      let b = flatten a in
      (* b = [|1.;2.;3.;4.;5.;6.|] *)
    ]} *)

val ravel : ('a, 'b) t -> ('a, 'b) t
(** [ravel t].

    Alias for [flatten t]; returns a 1-D view of [t] sharing the same data.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - a 1-D tensor view of [t]

    {2 Examples}
    {[
      let b = ravel a (* equivalent to flatten a *)
    ]} *)

val reshape : int array -> ('a, 'b) t -> ('a, 'b) t
(** [reshape new_shape t].

    Returns a tensor with shape [new_shape] sharing [t]’s data when possible. If
    [t] is [C_contiguous] or strides permit, returns a view without copying;
    otherwise allocates a new buffer and returns a copy reshaped to [new_shape].

    {2 Parameters}
    - [new_shape]: array of dimensions whose product matches [size t]
    - [t]: input tensor

    {2 Returns}
    - a view on the original data if no copy is needed (O(1) time), else a new
      tensor with copied data

    {2 Raises}
    - [Invalid_argument] if the total number of elements differs

    {2 Examples}
    {[
      let a = create float32 [|2;3|] [|1.;2.;3.;4.;5.;6.|] in
      let b = reshape [|3;2|] a in
      (* b has shape [|3;2|] and shares data if contiguous *)
    ]} *)

val transpose : ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
(** [transpose ?axes t].

    Returns a view of [t] with axes permuted. Does not allocate new data. If
    [axes] is provided, uses it as the new axis order; otherwise reverses the
    dimension order.

    {2 Parameters}
    - [axes]: optional array of length [ndim t] specifying axis order
    - [t]: input tensor

    {2 Returns}
    - a view with permuted axes sharing the original data

    {2 Raises}
    - [Invalid_argument] if [axes] length or contents do not match tensor rank

    {2 Examples}
    {[
      let t = create float32 [|2;3;4|] data in
      let u = transpose ~axes:[|1;0;2|] t in
      (* u.(i,j,k) = t.(j,i,k) *)
    ]} *)

val squeeze : ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
(** [squeeze ?axes t].

    Remove singleton dimensions from [t]. If [?axes] is provided, only the
    specified axes (where size = 1) are removed; otherwise all size-1 axes are
    squeezed. Returns a view sharing data; no copy.

    {2 Parameters}
    - [?axes]: array of axes to remove (default: all axes with size 1)
    - [t]: input tensor

    {2 Returns}
    - a view of [t] with specified singleton dimensions removed

    {2 Raises}
    - [Invalid_argument] if any axis in [?axes] is out of bounds or has size ≠ 1

    {2 Examples}
    {[
      let a = create float32 [|1;3;1;4|] data in
      let b = squeeze a in
      (* b has shape [|3;4|] *)
    ]} *)

(* *)

val split : ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t list
(** [split ?axis sections t].

    Divide [t] into [sections] equal-sized sub-tensors along [axis]. The size of
    [t] along [axis] must be divisible by [sections]. Each part is a view
    sharing the original buffer; no data is copied.

    {2 Parameters}
    - [?axis]: axis along which to split (default: 0)
    - [sections]: number of equal parts to create (must divide size along axis)
    - [t]: input tensor

    {2 Returns}
    - a list of [sections] tensor views, each of equal shape except along [axis]

    {2 Raises}
    - [Invalid_argument] if [sections] ≤ 0 or size along [axis] is not divisible

    {2 Examples}
    {[
      let a = of_array float32 [|4;2|] data in
      let [b; c] = split ~axis:0 2 a in
      (* b, c each have shape [|2;2|] *)
    ]} *)

val array_split : ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t list
(** [array_split ?axis sections t].

    Divide [t] into [sections] parts along [axis], distributing elements as
    evenly as possible. If the size along [axis] is not divisible by [sections],
    the first [rem = size mod sections] parts have one extra element. Returns
    views sharing the original buffer; no data is copied.

    {2 Parameters}
    - [?axis]: axis along which to split (default: 0)
    - [sections]: number of parts to create (must be > 0)
    - [t]: input tensor

    {2 Returns}
    - a list of [sections] tensor views with shapes varying by at most one

    {2 Raises}
    - [Invalid_argument] if [sections] ≤ 0

    {2 Examples}
    {[
      let a = of_array float32 [|5|] [|0.;1.;2.;3.;4.|] in
      let parts = array_split ~axis:0 3 a in
      (* parts shapes: [|2|], [|2|], [|1|] *)
    ]} *)

(* *)

val concatenate : ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
(** [concatenate ?axis ts].

    Joins the list of tensors [ts] along axis [axis], returning a new tensor.
    Allocates a fresh buffer and copies each input into the result.

    {2 Parameters}
    - [axis]: dimension along which to concatenate; default is 0
    - [ts]: non-empty list of tensors with same [dtype] and matching shapes
      except at [axis]

    {2 Returns}
    - a new tensor whose shape is equal to inputs except at [axis], where sizes
      are summed

    {2 Raises}
    - [Invalid_argument] if [ts] is empty, axes out of bounds, or tensors have
      mismatched ranks, dtypes, or shapes on non-concatenation axes

    {2 Examples}
    {[
      let a = create float32 [|2;2|] [|1.;2.;3.;4.|] in
      let b = create float32 [|1;2|] [|5.;6.|] in
      let c = concatenate ~axis:0 [a; b] in
      (* c has shape [|3;2|] with elements [[1.;2.];[3.;4.];[5.;6.]] *)
    ]} *)

val stack : ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
(** [stack ?axis ts].

    Stacks the list of tensors [ts] along a new axis [axis], increasing the
    tensor rank by one. Internally expands each tensor with [expand_dims] then
    concatenates. Allocates a fresh buffer.

    {2 Parameters}
    - [axis]: position to insert new axis (default 0)
    - [ts]: non-empty list of tensors of equal shape and dtype

    {2 Returns}
    - a new tensor of rank [ndim ts + 1] with shape updated at [axis]

    {2 Raises}
    - [Invalid_argument] if [ts] is empty or tensors have mismatched
      shapes/dtypes

    {2 Examples}
    {[
      let a = create float32 [|2;3|] data1 in
      let b = create float32 [|2;3|] data2 in
      let m = stack ~axis:0 [a; b];
      (* m has shape [|2;2;3|] after stacking *)
    ]} *)

val vstack : ('a, 'b) t list -> ('a, 'b) t
(** [vstack ts].

    Stack tensors in [ts] vertically (row-wise). 1-D tensors are promoted to 2-D
    by prepending a new axis. Internally concatenates along axis 0 and allocates
    a fresh buffer.

    {2 Parameters}
    - [ts]: non-empty list of tensors of matching dtype and compatible shapes

    {2 Returns}
    - a new tensor with shape updated by summing sizes along the first axis

    {2 Raises}
    - [Invalid_argument] if [ts] is empty or tensors have mismatched dtypes or
      incompatible shapes

    {2 Examples}
    {[
      let a = create float32 [|2;2|] data1 in
      let b = create float32 [|2;2|] data2 in
      let m = vstack [a; b] in
      (* m has shape [|4;2|] *)
    ]} *)

val hstack : ('a, 'b) t list -> ('a, 'b) t
(** [hstack ts].

    Stack tensors in [ts] horizontally (column-wise). 1-D tensors are promoted
    to 2-D by appending a new axis. Internally chooses axis 1 for 2-D inputs or
    axis 0 for 1-D inputs, then concatenates and allocates a fresh buffer.

    {2 Parameters}
    - [ts]: non-empty list of tensors of matching dtype and compatible shapes

    {2 Returns}
    - a new tensor with shape updated by summing sizes along the chosen axis

    {2 Raises}
    - [Invalid_argument] if [ts] is empty or tensors have mismatched dtypes or
      incompatible shapes

    {2 Examples}
    {[
      let a = create float32 [|2;3|] ... in
      let b = create float32 [|2;3|] ... in
      let m = hstack [a; b] in
      (* m has shape [|2;6|] *)
    ]} *)

val dstack : ('a, 'b) t list -> ('a, 'b) t
(** [dstack ts].

    Stack tensors in [ts] depth-wise (along the third axis). Scalars and 1-D
    tensors are promoted by inserting new axes as needed. Internally
    concatenates along axis 2 and allocates a fresh buffer.

    {2 Parameters}
    - [ts]: non-empty list of tensors of matching dtype and compatible shapes

    {2 Returns}
    - a new tensor with shape updated by summing sizes along the third axis

    {2 Raises}
    - [Invalid_argument] if [ts] is empty or tensors have mismatched dtypes or
      incompatible shapes

    {2 Examples}
    {[
      let a = create float32 [|2;2|] ... in
      let b = create float32 [|2;2|] ... in
      let m = dstack [a; b] in
      (* m has shape [|2;2;2|] *)
    ]} *)

(* *)

val pad : (int * int) array -> 'a -> ('a, 'b) t -> ('a, 'b) t
(** [pad padding value t].

    Pad tensor [t] with [value] according to [padding] on each axis. For axis
    [i], [padding.(i)] = (before, after) gives numbers of values added before
    and after the existing data. Allocates a new tensor, fills it with [value],
    and copies [t] into the central region.

    {2 Parameters}
    - [padding]: array of (before, after) pad widths; length = ndim t
    - [value]: scalar fill value
    - [t]: input tensor

    {2 Returns}
    - a new tensor with shape
      [Array.map2 (fun dim (b,a) -> dim + b + a) (shape t) padding]

    {2 Raises}
    - [Invalid_argument] if [padding] length ≠ ndim t or any pad width is
      negative

    {2 Examples}
    {[
      let a = of_array float32 [|2;2|] [|1.;2.;3.;4.|] in
      let b = pad [|(1,1);(2,2)|] 0. a in
      (* b has shape [|4;6|] with zeros border *)
    ]} *)

val expand_dims : int -> ('a, 'b) t -> ('a, 'b) t
(** [expand_dims axis t].

    Insert a new axis of length 1 at position [axis], increasing the rank of [t]
    by one. Returns a view sharing the same data; no data is copied.

    {2 Parameters}
    - [axis]: position to insert new dimension (0 ≤ axis ≤ ndim t)
    - [t]: input tensor

    {2 Returns}
    - a view of [t] with rank [ndim t + 1] and updated shape

    {2 Raises}
    - [Invalid_argument] if [axis] is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;3|] data in
      let b = expand_dims 0 a in  (* shape [|1;2;3|] *)
      let c = expand_dims 2 a in  (* shape [|2;1;3|] *)
    ]} *)

val broadcast_to : int array -> ('a, 'b) t -> ('a, 'b) t
(** [broadcast_to shape t].

    Returns a view of [t] broadcast to shape [shape] by adjusting strides. No
    data is copied; runs in O(1).

    {2 Parameters}
    - [shape]: target dimensions (each must equal original size or be 1)
    - [t]: input tensor

    {2 Returns}
    - a view sharing the same data with updated strides and layout

    {2 Raises}
    - [Failure] if shapes are not compatible for broadcasting

    {2 Examples}
    {[
      let a = create float32 [|1;3|] [|1.;2.;3.|] in
      let b = broadcast_to [|4;3|] a;
      (* b has shape [|4;3|], with each row equal to a's row *)
    ]} *)

val broadcast_arrays : ('a, 'b) t list -> ('a, 'b) t list
(** [broadcast_arrays ts].

    Broadcast a list of tensors [ts] to a common shape using NumPy-style
    broadcasting rules. All tensors must have the same dtype. Returns a list of
    views sharing the original data; no data is copied.

    {2 Parameters}
    - [ts]: list of tensors to broadcast

    {2 Returns}
    - list of tensor views all having the broadcast shape

    {2 Raises}
    - [Failure] if tensors cannot be broadcast to a common shape

    {2 Examples}
    {[
      let a = create float32 [|3;1|] [|1.;2.;3.|] in
      let b = create float32 [|1;4|] [|10.;20.;30.;40.|] in
      let [a'; b'] = broadcast_arrays [a; b] in
      (* a'.shape = [|3;4|], b'.shape = [|3;4|] *)
    ]} *)

val tile : int array -> ('a, 'b) t -> ('a, 'b) t
(** [tile reps t].

    Constructs a new tensor by repeating [t]’s contents according to [reps] per
    axis. Allocates a new buffer (O(N * Π reps_i)).

    {2 Parameters}
    - [reps]: array of repetition counts for each dimension of [t]
    - [t]: input tensor

    {2 Returns}
    - a fresh tensor with shape [Array.map2 ( * ) (shape t) reps]

    {2 Raises}
    - [Invalid_argument] if [Array.length reps] <> [ndim t]

    {2 Examples}
    {[
      let a = create float32 [|2;1|] [|1.;2.|] in
      let b = tile [|2;3|] a;
      (* b has shape [|4;3|], repeating a along axes *)
    ]} *)

val repeat : ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [repeat ?axis count t].

    Repeat elements of [t] [count] times along the specified [axis]. If [?axis]
    is not provided, [t] is flattened and repeated along axis 0. Allocates a
    fresh buffer for the result.

    {2 Parameters}
    - [?axis]: axis along which to repeat (default: flatten and axis 0)
    - [count]: number of repetitions for each element (must be >= 0)
    - [t]: input tensor

    {2 Returns}
    - new tensor with repeated elements along [axis]

    {2 Raises}
    - [Invalid_argument] if [count] < 0 or [axis] is out of bounds

    {2 Examples}
    {[
      let a = of_array int32 [|3|] [|1l;2l;3l|] in
      let b = repeat ~axis:0 2 a in
      (* b = [|1l;1l;2l;2l;3l;3l|] *)
    ]} *)

val flip : ?axes:int array -> ('a, 'b) t -> ('a, 'b) t
(** [flip ?axes t].

    Reverse the order of elements in [t] along the given [axes]. If [?axes] is
    not provided, all axes are reversed. Returns a view sharing data; no data is
    copied.

    {2 Parameters}
    - [?axes]: array of axes to flip (default: all axes)
    - [t]: input tensor

    {2 Returns}
    - a view with elements flipped along specified axes

    {2 Raises}
    - [Invalid_argument] if any axis is out of bounds

    {2 Examples}
    {[
      let a = of_array float32 [|2;3|] [|1.;2.;3.;4.;5.;6.|] in
      let b = flip ~axes:[|1|] a in
      (* b = [[3.;2.;1.]; [6.;5.;4.]] *)
    ]} *)

val roll : ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [roll ?axis shift t].

    Circularly shift elements of [t] along [axis] by [shift] positions. If
    [?axis] is not provided, operates on the flattened tensor. Elements that
    roll beyond the last position wrap around to the first. Returns a tensor
    constructed via views and concatenation.

    {2 Parameters}
    - [?axis]: axis to roll (default: flatten and axis 0)
    - [shift]: number of places to shift (can be negative)
    - [t]: input tensor

    {2 Returns}
    - tensor with elements circularly shifted along [axis]

    {2 Raises}
    - [Invalid_argument] if [axis] is out of bounds

    {2 Examples}
    {[
      let a = of_array int32 [|5|] [|1l;2l;3l;4l;5l|] in
      let b = roll ~axis:0 2 a in
      (* b = [|4l;5l;1l;2l;3l|] *)
    ]} *)

(* *)

val moveaxis : int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [moveaxis src dst t].

    Move axis [src] to position [dst] in [t], shifting remaining axes
    accordingly. Returns a view sharing the original data; no data is copied.

    {2 Parameters}
    - [src]: original axis index
    - [dst]: destination axis index
    - [t]: input tensor

    {2 Returns}
    - a view of [t] with axes reordered

    {2 Raises}
    - [Invalid_argument] if [src] or [dst] is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;3;4|] data in
      let b = moveaxis 0 2 a in   (* b.shape = [|3;4;2|] *)
    ]} *)

val swapaxes : int -> int -> ('a, 'b) t -> ('a, 'b) t
(** [swapaxes axis1 axis2 t].

    Swap two axes [axis1] and [axis2] in [t], equivalent to a transpose with
    those axes swapped. Returns a view; no data is copied.

    {2 Parameters}
    - [axis1]: first axis index
    - [axis2]: second axis index
    - [t]: input tensor

    {2 Returns}
    - a view of [t] with the two axes swapped

    {2 Raises}
    - [Invalid_argument] if either axis is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;3;4|] data in
      let b = swapaxes 0 1 a in   (* b.shape = [|3;2;4|] *)
    ]} *)

(** {1 Conversion}

    Change data layout or dtype, or extract to a plain OCaml array. *)

val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> ('a, 'b) t
(** [of_bigarray ga].

    Create a tensor view from a C-layout Bigarray Genarray [ga]. Shares the
    underlying buffer; no data is copied. The tensor’s shape, dtype, strides,
    and offset correspond to [ga].

    {2 Parameters}
    - [ga]: input Bigarray.Genarray with C layout

    {2 Returns}
    - tensor view reflecting [ga]

    {2 Examples}
    {[
      let ga = Bigarray.Genarray.create Bigarray.float32 Bigarray.c_layout [|2;3|] in
      let t = of_bigarray ga
    ]} *)

val to_bigarray : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_bigarray t].

    Return a Bigarray Genarray in C layout containing the same elements as [t].
    If [t] is C-contiguous, this is O(1) and shares the underlying buffer;
    otherwise allocates a new Genarray and copies data.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - Bigarray.Genarray reflecting [t]’s content

    {2 Examples}
    {[
      let ga = to_bigarray t
    ]} *)

val to_array : ('a, 'b) t -> 'a array
(** [to_array t].

    Extract all elements of [t] into a new OCaml array in row-major order.
    Allocates an array of length [size t] and copies every element.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - OCaml array of elements in row-major order

    {2 Examples}
    {[
      let arr = to_array t
    ]} *)

val astype : ('c, 'd) dtype -> ('a, 'b) t -> ('c, 'd) t
(** [astype dtype t].

    Returns a new tensor with element type [dtype], containing values of [t]
    cast to the target type. Allocates a new buffer of the same shape.

    {2 Parameters}
    - [dtype]: desired output data type
    - [t]: input tensor

    {2 Returns}
    - a fresh tensor with values cast to [dtype]

    {2 Examples}
    {[
      let a = create float32 [|2|] [|1.;2.|] in
      let b = astype int32 a in
      (* b : int32_t with values [|1l;2l|] *)
    ]} *)

(** {1 Arithmetic and Element-wise Operations}

    Basic +, –, ×, ÷, powers, and pointwise math. *)

val add : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [add t1 t2].

    Compute the element-wise sum of [t1] and [t2], following broadcasting rules.

    {2 Parameters}
    - [t1]: first input tensor
    - [t2]: second input tensor

    {2 Returns}
    - fresh tensor where each element = t1 + t2

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let c = add a b
    ]} *)

val add_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [add_inplace t1 t2].

    Add [t2] into [t1] element-wise in place, following broadcasting rules.
    Modifies [t1] and returns it.

    {2 Parameters}
    - [t1]: tensor to be updated (lhs)
    - [t2]: tensor to add (rhs)

    {2 Returns}
    - the updated [t1]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = add_inplace a b
    ]} *)

val add_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [add_scalar t v].

    Add scalar [v] to each element of [t], returning a new tensor. Allocates a
    fresh buffer of the same shape.

    {2 Parameters}
    - [t]: input tensor
    - [v]: scalar value to add

    {2 Returns}
    - new tensor where each element = original + v

    {2 Examples}
    {[
      let b = add_scalar a 5
    ]} *)

val sub : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sub t1 t2].

    Compute the element-wise difference [t1 - t2], following broadcasting rules.

    {2 Parameters}
    - [t1]: minuend tensor
    - [t2]: subtrahend tensor

    {2 Returns}
    - fresh tensor where each element = t1 - t2

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let c = sub a b
    ]} *)

val sub_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [sub_inplace t1 t2].

    Subtract [t2] from [t1] element-wise in place, following broadcasting rules.
    Modifies [t1] and returns it.

    {2 Parameters}
    - [t1]: tensor to be updated (lhs)
    - [t2]: tensor to subtract (rhs)

    {2 Returns}
    - the updated [t1]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = sub_inplace a b
    ]} *)

val sub_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [sub_scalar t v].

    Subtract scalar [v] from each element of [t], returning a new tensor.

    {2 Parameters}
    - [t]: input tensor
    - [v]: scalar value to subtract

    {2 Returns}
    - new tensor where each element = original - v

    {2 Examples}
    {[
      let b = sub_scalar a 3
    ]} *)

val mul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [mul t1 t2].

    Compute the element-wise product of [t1] and [t2], following broadcasting
    rules.

    {2 Parameters}
    - [t1]: first factor tensor
    - [t2]: second factor tensor

    {2 Returns}
    - fresh tensor where each element = t1 * t2

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let c = mul a b
    ]} *)

val mul_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [mul_inplace t1 t2].

    Multiply [t1] by [t2] element-wise in place, following broadcasting rules.
    Modifies [t1] and returns it.

    {2 Parameters}
    - [t1]: tensor to be updated (lhs)
    - [t2]: tensor to multiply (rhs)

    {2 Returns}
    - the updated [t1]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = mul_inplace a b
    ]} *)

val mul_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [mul_scalar t v].

    Multiply each element of [t] by scalar [v], returning a new tensor.

    {2 Parameters}
    - [t]: input tensor
    - [v]: scalar multiplier

    {2 Returns}
    - new tensor where each element = original * v

    {2 Examples}
    {[
      let b = mul_scalar a 10
    ]} *)

val div : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [div t1 t2].

    Compute the element-wise quotient [t1 / t2], following broadcasting rules.

    {2 Parameters}
    - [t1]: dividend tensor
    - [t2]: divisor tensor

    {2 Returns}
    - fresh tensor where each element = t1 / t2

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let c = div a b
    ]} *)

val div_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [div_inplace t1 t2].

    Divide [t1] by [t2] element-wise in place, using broadcasting rules.
    Modifies [t1] and returns it.

    {2 Parameters}
    - [t1]: dividend tensor to update
    - [t2]: divisor tensor

    {2 Returns}
    - the updated [t1]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = div_inplace a b
    ]} *)

val div_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [div_scalar t v].

    Divide each element of [t] by scalar [v], returning a new tensor.

    {2 Parameters}
    - [t]: input tensor
    - [v]: scalar divisor

    {2 Returns}
    - new tensor where each element = original / v

    {2 Raises}
    - [Invalid_argument] if [v] is zero

    {2 Examples}
    {[
      let b = div_scalar a 2
    ]} *)

val rem : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [rem t1 t2].

    Compute element-wise remainder of [t1] mod [t2], following broadcasting
    rules.

    {2 Parameters}
    - [t1]: dividend tensor
    - [t2]: divisor tensor

    {2 Returns}
    - fresh tensor where each element = t1 mod t2

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let c = rem a b
    ]} *)

val rem_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [rem_inplace t1 t2].

    Compute element-wise remainder of [t1] mod [t2] in place, following
    broadcasting rules. Modifies [t1] and returns it.

    {2 Parameters}
    - [t1]: tensor to be updated (lhs)
    - [t2]: divisor tensor (rhs)

    {2 Returns}
    - the updated [t1]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = rem_inplace a b
    ]} *)

val rem_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [rem_scalar t v].

    Compute element-wise remainder of each element of [t] mod scalar [v].

    {2 Parameters}
    - [t]: input tensor
    - [v]: scalar divisor

    {2 Returns}
    - new tensor where each element = original mod v

    {2 Raises}
    - [Invalid_argument] if v = 0

    {2 Examples}
    {[
      let b = rem_scalar a 3
    ]} *)

val pow : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [pow t1 t2].

    Compute element-wise power [t1 ** t2], following broadcasting rules.

    {2 Parameters}
    - [t1]: base tensor
    - [t2]: exponent tensor

    {2 Returns}
    - fresh tensor where each element = t1 raised to t2

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let c = pow a b
    ]} *)

val pow_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [pow_inplace t1 t2].

    Compute element-wise power [t1 ** t2] in place, following broadcasting
    rules. Modifies [t1] and returns it.

    {2 Parameters}
    - [t1]: tensor to be updated (base)
    - [t2]: exponent tensor

    {2 Returns}
    - the updated [t1]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = pow_inplace a b
    ]} *)

val pow_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [pow_scalar t v].

    Compute element-wise power of elements of [t] raised to scalar exponent [v].

    {2 Parameters}
    - [t]: base tensor
    - [v]: scalar exponent

    {2 Returns}
    - new tensor where each element = original raised to v

    {2 Examples}
    {[
      let b = pow_scalar a 2
    ]} *)

val maximum : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [maximum t1 t2].

    Compute the element-wise maximum of [t1] and [t2], following broadcasting
    rules.

    {2 Parameters}
    - [t1]: first input tensor
    - [t2]: second input tensor

    {2 Returns}
    - fresh tensor where each element = max(t1, t2)

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let c = maximum a b
    ]} *)

val maximum_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [maximum_inplace t1 t2].

    Compute the element-wise maximum of [t1] and [t2] in place, following
    broadcasting rules. Modifies [t1] to hold the maximum and returns it.

    {2 Parameters}
    - [t1]: tensor to be updated
    - [t2]: tensor to compare

    {2 Returns}
    - the updated [t1]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = maximum_inplace a b
    ]} *)

val maximum_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [maximum_scalar t v].

    Compute the element-wise maximum of elements of [t] and scalar [v].

    {2 Parameters}
    - [t]: input tensor
    - [v]: scalar to compare

    {2 Returns}
    - new tensor where each element = max(original, v)

    {2 Examples}
    {[
      let b = maximum_scalar a 5
    ]} *)

val minimum : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [minimum t1 t2].

    Compute the element-wise minimum of [t1] and [t2], following broadcasting
    rules.

    {2 Parameters}
    - [t1]: first input tensor
    - [t2]: second input tensor

    {2 Returns}
    - fresh tensor where each element = min(t1, t2)

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let c = minimum a b
    ]} *)

val minimum_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [minimum_inplace t1 t2].

    Compute the element-wise minimum of [t1] and [t2] in place, following
    broadcasting rules. Modifies [t1] to hold the minimum and returns it.

    {2 Parameters}
    - [t1]: tensor to be updated
    - [t2]: tensor to compare

    {2 Returns}
    - the updated [t1]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = minimum_inplace a b
    ]} *)

val minimum_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [minimum_scalar t v].

    Compute the element-wise minimum of elements of [t] and scalar [v].

    {2 Parameters}
    - [t]: input tensor
    - [v]: scalar to compare

    {2 Returns}
    - new tensor where each element = min(original, v)

    {2 Examples}
    {[
      let b = minimum_scalar a 5
    ]} *)

val fma : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [fma a b c].

    Compute element-wise fused multiply-add: [a * b + c], following broadcasting
    rules.

    {2 Parameters}
    - [a]: first factor tensor
    - [b]: second factor tensor
    - [c]: tensor to add

    {2 Returns}
    - fresh tensor where each element = a * b + c

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let d = fma x y z
    ]} *)

val fma_inplace : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [fma_inplace a b c].

    Perform fused multiply-add into [a]: element-wise compute [a := a * b + c],
    following broadcasting rules. Modifies [a] and returns it.

    {2 Parameters}
    - [a]: tensor to update (base and accumulator)
    - [b]: factor tensor
    - [c]: addend tensor

    {2 Returns}
    - the updated [a]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let () = fma_inplace x y z
    ]} *)

(** {2 Unary Mathematical Functions}

    Exponentials, logarithms, trig, hyperbolic, etc. *)

val exp : ('a, 'b) t -> ('a, 'b) t
(** [exp t].

    Compute the element-wise exponential of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = exp(original)

    {2 Examples}
    {[
      let b = exp a
    ]} *)

val log : ('a, 'b) t -> ('a, 'b) t
(** [log t].

    Compute the element-wise natural logarithm of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = log(original)

    {2 Examples}
    {[
      let b = log a
    ]} *)

val abs : ('a, 'b) t -> ('a, 'b) t
(** [abs t].

    Compute the element-wise absolute value of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = abs(original)

    {2 Examples}
    {[
      let b = abs a
    ]} *)

val neg : ('a, 'b) t -> ('a, 'b) t
(** [neg t].

    Compute the element-wise negation of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = - original

    {2 Examples}
    {[
      let b = neg a
    ]} *)

val sign : ('a, 'b) t -> ('a, 'b) t
(** [sign t].

    Compute the element-wise sign of [t], returning -1 for negative, 0 for zero,
    and 1 for positive values.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = sign(original)

    {2 Examples}
    {[
      let b = sign a
    ]} *)

val sqrt : (float, 'b) t -> (float, 'b) t
(** [sqrt t].

    Compute the element-wise square root of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = sqrt(original)

    {2 Examples}
    {[
      let b = sqrt a
    ]} *)

val square : ('a, 'b) t -> ('a, 'b) t
(** [square t].

    Compute the element-wise square of [t] (x * x).

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = original * original

    {2 Examples}
    {[
      let b = square a
    ]} *)

val sin : (float, 'b) t -> (float, 'b) t
(** [sin t].

    Compute the element-wise sine of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = sin(original)

    {2 Examples}
    {[
      let b = sin a
    ]} *)

val cos : (float, 'b) t -> (float, 'b) t
(** [cos t].

    Compute the element-wise cosine of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = cos(original)

    {2 Examples}
    {[
      let b = cos a
    ]} *)

val tan : (float, 'b) t -> (float, 'b) t
(** [tan t].

    Compute the element-wise tangent of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = tan(original)

    {2 Examples}
    {[
      let b = tan a
    ]} *)

val asin : (float, 'b) t -> (float, 'b) t
(** [asin t].

    Compute the element-wise arcsine of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = asin(original)

    {2 Examples}
    {[
      let b = asin a
    ]} *)

val acos : (float, 'b) t -> (float, 'b) t
(** [acos t].

    Compute the element-wise arccosine of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = acos(original)

    {2 Examples}
    {[
      let b = acos a
    ]} *)

val atan : (float, 'b) t -> (float, 'b) t
(** [atan t].

    Compute the element-wise arctangent of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = atan(original)

    {2 Examples}
    {[
      let b = atan a
    ]} *)

val sinh : (float, 'b) t -> (float, 'b) t
(** [sinh t].

    Compute the element-wise hyperbolic sine of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = sinh(original)

    {2 Examples}
    {[
      let b = sinh a
    ]} *)

val cosh : (float, 'b) t -> (float, 'b) t
(** [cosh t].

    Compute the element-wise hyperbolic cosine of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = cosh(original)

    {2 Examples}
    {[
      let b = cosh a
    ]} *)

val tanh : (float, 'b) t -> (float, 'b) t
(** [tanh t].

    Compute the element-wise hyperbolic tangent of [t].

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = tanh(original)

    {2 Examples}
    {[
      let b = tanh a
    ]} *)

val asinh : (float, 'b) t -> (float, 'b) t
(** [asinh t].

    Compute the element-wise inverse hyperbolic sine of [t]. Allocates a new
    tensor.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = asinh(original)

    {2 Examples}
    {[
      let b = asinh a
    ]} *)

val acosh : (float, 'b) t -> (float, 'b) t
(** [acosh t].

    Compute the element-wise inverse hyperbolic cosine of [t]. Allocates a new
    tensor.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = acosh(original)

    {2 Examples}
    {[
      let b = acosh a
    ]} *)

val atanh : (float, 'b) t -> (float, 'b) t
(** [atanh t].

    Compute the element-wise inverse hyperbolic tangent of [t]. Allocates a new
    tensor.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = atanh(original)

    {2 Examples}
    {[
      let b = atanh a
    ]} *)

(** Evenly round, floor, or ceiling element‑wise. *)

val round : (float, 'b) t -> (float, 'b) t
(** [round t].

    Round each element of [t] to the nearest integer.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = original rounded to nearest integer

    {2 Examples}
    {[
      let b = round a
    ]} *)

val floor : (float, 'b) t -> (float, 'b) t
(** [floor t].

    Compute the element-wise floor of [t], rounding toward negative infinity.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = original rounded toward negative infinity

    {2 Examples}
    {[
      let b = floor a
    ]} *)

val ceil : (float, 'b) t -> (float, 'b) t
(** [ceil t].

    Compute the element-wise ceiling of [t], rounding toward positive infinity.

    {2 Parameters}
    - [t]: input tensor

    {2 Returns}
    - new tensor where each element = original rounded toward positive infinity

    {2 Examples}
    {[
      let b = ceil a
    ]} *)

(* *)

val clip : min:'a -> max:'a -> ('a, 'b) t -> ('a, 'b) t
(** [clip ~min ~max t].

    Clip each element of [t] to the interval [`min` .. `max`]. Allocates a new
    tensor.

    {2 Parameters}
    - [~min]: lower bound
    - [~max]: upper bound
    - [t]: input tensor

    {2 Returns}
    - new tensor where elements < min set to min, elements > max set to max,
      others unchanged

    {2 Examples}
    {[
      let b = clip ~min:0 ~max:1 a
    ]} *)

(** {1 Comparison Operations}

    Element‑wise comparisons producing 0/1 masks. *)

val equal : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [equal t1 t2].

    Compute element-wise equality comparison between [t1] and [t2]. Allocates a
    new mask tensor.

    {2 Parameters}
    - [t1]: first input tensor
    - [t2]: second input tensor

    {2 Returns}
    - mask tensor of type [uint8] with 1 where elements are equal, 0 otherwise

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let m = equal a b
    ]} *)

val greater : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [greater t1 t2].

    Compute element-wise greater-than comparison between [t1] and [t2].
    Allocates a new mask tensor.

    {2 Parameters}
    - [t1]: first input tensor
    - [t2]: second input tensor

    {2 Returns}
    - mask tensor of type [uint8] with 1 where t1 > t2, 0 otherwise

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let m = greater a b
    ]} *)

val greater_equal : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [greater_equal t1 t2].

    Compute element-wise greater-or-equal comparison between [t1] and [t2].
    Allocates a new mask tensor.

    {2 Parameters}
    - [t1]: first input tensor
    - [t2]: second input tensor

    {2 Returns}
    - mask tensor of type [uint8] with 1 where t1 >= t2, 0 otherwise

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let m = greater_equal a b
    ]} *)

val less : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [less t1 t2].

    Compute element-wise less-than comparison between [t1] and [t2]. Allocates a
    new mask tensor.

    {2 Parameters}
    - [t1]: first input tensor
    - [t2]: second input tensor

    {2 Returns}
    - mask tensor of type [uint8] with 1 where t1 < t2, 0 otherwise

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let m = less a b
    ]} *)

val less_equal : ('a, 'b) t -> ('a, 'b) t -> (int, uint8_elt) t
(** [less_equal t1 t2].

    Compute element-wise less-or-equal comparison between [t1] and [t2].
    Allocates a new mask tensor.

    {2 Parameters}
    - [t1]: first input tensor
    - [t2]: second input tensor

    {2 Returns}
    - mask tensor of type [uint8] with 1 where t1 <= t2, 0 otherwise

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes differ

    {2 Examples}
    {[
      let m = less_equal a b
    ]} *)

(** {1 Bitwise Operations}

    Bitwise AND, OR, XOR, NOT, shifts, and masks. *)

val bitwise_and : (int, 'b) t -> (int, 'b) t -> (int, 'b) t
(** [bitwise_and t1 t2].

    Compute element-wise bitwise AND of [t1] and [t2].

    {2 Parameters}
    - [t1]: first input integer tensor
    - [t2]: second input integer tensor

    {2 Returns}
    - new tensor where each element = bitwise AND of corresponding elements

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible

    {2 Examples}
    {[
      let c = bitwise_and a b
    ]} *)

val bitwise_or : (int, 'b) t -> (int, 'b) t -> (int, 'b) t
(** [bitwise_or t1 t2].

    Compute element-wise bitwise OR of [t1] and [t2].

    {2 Parameters}
    - [t1]: first input integer tensor
    - [t2]: second input integer tensor

    {2 Returns}
    - new tensor where each element = bitwise OR of corresponding elements

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible

    {2 Examples}
    {[
      let c = bitwise_or a b
    ]} *)

val bitwise_xor : (int, 'b) t -> (int, 'b) t -> (int, 'b) t
(** [bitwise_xor t1 t2].

    Compute element-wise bitwise XOR of [t1] and [t2].

    {2 Parameters}
    - [t1]: first input integer tensor
    - [t2]: second input integer tensor

    {2 Returns}
    - new tensor where each element = bitwise XOR of corresponding elements

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible

    {2 Examples}
    {[
      let c = bitwise_xor a b
    ]} *)

val invert : (int, 'b) t -> (int, 'b) t
(** [invert t].

    Compute the element-wise bitwise complement (NOT) of [t]. Allocates a new
    tensor.

    {2 Parameters}
    - [t]: input integer tensor

    {2 Returns}
    - new tensor where each element = bitwise NOT of original

    {2 Examples}
    {[
      let b = invert a
    ]} *)

(** {1 Reductions and Statistical Functions}

    Sum, mean, variance, max/min, count, median, percentiles, etc. *)

val sum : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [sum ?axes ?keepdims t].

    Computes the sum of elements in [t] along specified [axes]. Allocates a new
    tensor of the reduced shape.

    {2 Parameters}
    - [axes]: array of axes to reduce; default is all axes
    - [keepdims]: if [true], retains reduced dimensions with size 1
    - [t]: input tensor

    {2 Returns}
    - a new tensor containing summed values of the same [dtype] as [t] and shape
      determined by [axes] and [keepdims]

    {2 Raises}
    - [Invalid_argument] if any axis is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;2|] [|1.;2.;3.;4.|] in
      let total = sum a in
      (* total = 10. *)
      let v = sum ~axes:[|0|] a in
      (* v = [|4.;6.|] shape [|2|] *)
    ]} *)

val prod : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [prod ?axes ?keepdims t].

    Compute the product of elements in [t] along specified [axes]. Allocates a
    new tensor of the reduced shape.

    {2 Parameters}
    - [axes]: array of axes to reduce; default is all axes
    - [keepdims]: if [true], retains reduced dimensions with size 1
    - [t]: input tensor

    {2 Returns}
    - a new tensor containing the product of elements with same [dtype] as [t]
      and shape determined by [axes] and [keepdims]

    {2 Raises}
    - [Invalid_argument] if any axis is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;2|] [|1.;2.;3.;4.|] in
      let total = prod a in
      (* total = 24. *)
      let v = prod ~axes:[|0|] a in
      (* v = [|3.;8.|] shape [|2|] *)
    ]} *)

val max : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [max ?axes ?keepdims t].

    Compute the maximum value of elements in [t] along specified [axes].
    Allocates a new tensor of the reduced shape.

    {2 Parameters}
    - [axes]: array of axes to reduce; default is all axes
    - [keepdims]: if [true], retains reduced dimensions with size 1
    - [t]: input tensor

    {2 Returns}
    - a new tensor containing the maximum of elements with same [dtype] as [t]
      and shape determined by [axes] and [keepdims]

    {2 Raises}
    - [Invalid_argument] if any axis is out of bounds
    - [Invalid_argument] if [t] has no elements to reduce

    {2 Examples}
    {[
      let a = create float32 [|2;2|] [|1.;4.;2.;3.|] in
      let m = max a in
      (* m = 4. *)
      let v = max ~axes:[|1|] a in
      (* v = [|4.;3.|] shape [|2|] *)
    ]} *)

val min : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [min ?axes ?keepdims t].

    Compute the minimum value of elements in [t] along specified [axes].
    Allocates a new tensor of the reduced shape.

    {2 Parameters}
    - [axes]: array of axes to reduce; default is all axes
    - [keepdims]: if [true], retains reduced dimensions with size 1
    - [t]: input tensor

    {2 Returns}
    - a new tensor containing the minimum of elements with same [dtype] as [t]
      and shape determined by [axes] and [keepdims]

    {2 Raises}
    - [Invalid_argument] if any axis is out of bounds
    - [Invalid_argument] if [t] has no elements to reduce

    {2 Examples}
    {[
      let a = create float32 [|2;2|] [|1.;4.;2.;3.|] in
      let m = min a in
      (* m = 1. *)
      let v = min ~axes:[|1|] a in
      (* v = [|1.;2.|] shape [|2|] *)
    ]} *)

(** {1 Statistics}

    Variance, standard deviation, and histograms. *)

val mean : ?axes:int array -> ?keepdims:bool -> (float, 'b) t -> (float, 'b) t
(** [mean ?axes ?keepdims t].

    Compute the arithmetic mean of elements in [t] along specified [axes].
    Allocates a new tensor of the reduced shape.

    {2 Parameters}
    - [axes]: array of axes to reduce; default is all axes
    - [keepdims]: if [true], retains reduced dimensions with size 1
    - [t]: input float tensor

    {2 Returns}
    - a new tensor containing mean values with same float [dtype] as [t] and
      shape determined by [axes] and [keepdims]

    {2 Raises}
    - [Invalid_argument] if any axis is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;2|] [|1.;2.;3.;4.|] in
      let m = mean a in
      (* m = 2.5. *)
      let v = mean ~axes:[|0|] a in
      (* v = [|2.;3.|] shape [|2|] *)
    ]} *)

val var : ?axes:int array -> ?keepdims:bool -> (float, 'b) t -> (float, 'b) t
(** [var ?axes ?keepdims t].

    Compute the variance (average of squared deviations) of elements in [t]
    along specified [axes]. Allocates a new tensor of the reduced shape.

    {2 Parameters}
    - [axes]: array of axes to reduce; default is all axes
    - [keepdims]: if [true], retains reduced dimensions with size 1
    - [t]: input float tensor

    {2 Returns}
    - a new tensor containing variance values with same float [dtype] as [t] and
      shape determined by [axes] and [keepdims]

    {2 Raises}
    - [Invalid_argument] if any axis is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;2|] [|1.;2.;3.;4.|] in
      let v = var a in
      (* v = 1.25. *)
      let w = var ~axes:[|0|] a in
      (* w = [|1.;1.|] shape [|2|] *)
    ]} *)

val std : ?axes:int array -> ?keepdims:bool -> (float, 'b) t -> (float, 'b) t
(** [std ?axes ?keepdims t].

    Compute the standard deviation (square root of variance) of elements in [t]
    along specified [axes]. Allocates a new tensor of the reduced shape.

    {2 Parameters}
    - [axes]: array of axes to reduce; default is all axes
    - [keepdims]: if [true], retains reduced dimensions with size 1
    - [t]: input float tensor

    {2 Returns}
    - a new tensor containing standard deviation values with same float [dtype]
      as [t] and shape determined by [axes] and [keepdims]

    {2 Raises}
    - [Invalid_argument] if any axis is out of bounds

    {2 Examples}
    {[
      let a = create float32 [|2;2|] [|1.;2.;3.;4.|] in
      let s = std a in
      (* s = 1.1180. *)
      let v = std ~axes:[|0|] a in
      (* v = [|1.;1.|] shape [|2|] *)
    ]} *)

(** {1 Linear Algebra and Matrix Operations}

    Matrix products, inverses, eigen‐ and singular‐value decompositions,
    convolutions, determinants, norms, traces, etc. *)

val dot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [dot a b].

    Computes the generalized dot product:
    - If both [a] and [b] are 1-D, returns their inner product (scalar tensor).
    - If both are 2-D, performs matrix multiplication.
    - Otherwise, sums over the last axis of [a] and second‑last axis of [b],
      applying broadcasting to remaining axes.

    {2 Parameters}
    - [a], [b]: input tensors

    {2 Returns}
    - a new tensor holding the dot product result

    {2 Raises}
    - [Invalid_argument] if operand shapes are not aligned for dot operations

    {2 Examples}
    {[
      let x = create float32 [|2|] [|1.;2.|] in
      let y = create float32 [|2|] [|3.;4.|] in
      let s = dot x y in
      (* s is a scalar tensor equal to 11. *)
    ]} *)

val matmul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [matmul a b].

    Perform generalized matrix multiplication of [a] and [b], applying
    broadcasting over leading dimensions. Treat the last two dimensions of [a]
    and [b] as matrices of shapes (... × m × k) and (... × k × n), respectively.
    Supports N-D tensors: any preceding dimensions are broadcast as batch
    dimensions. For 1-D tensor inputs, promotes to 1×N or N×1 matrices and
    squeezes the result accordingly.

    {2 Parameters}
    - [a]: left operand tensor of shape [..., m, k]
    - [b]: right operand tensor of shape [..., k, n]

    {2 Returns}
    - a new tensor of shape [..., m, n] with same [dtype] as inputs

    {2 Raises}
    - [Invalid_argument] if input tensors are scalars (0-D)
    - [Invalid_argument] if inner dimensions are not aligned
    - [Invalid_argument] if batch dimensions cannot be broadcast

    {2 Examples}
    {[
      let a = of_array float32 [|2;3|] [|1.;2.;3.;4.;5.;6.|] in
      let b = of_array float32 [|3;2|] [|7.;8.;9.;10.;11.;12.|] in
      let c = matmul a b in
      (* c = [| [58.; 64.]; [139.; 154.] |] shape [|2;2|] *)
    ]} *)

val convolve1d :
  ?mode:[ `Full | `Valid | `Same ] -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [convolve1d ?mode a v].

    Compute the 1-D discrete convolution (cross-correlation) of tensor [a] with
    kernel [v], returning an output tensor of length determined by [mode].

    The output length is:
    - `Full: n + m - 1
    - `Valid: max(n - m + 1, 0)
    - `Same: n where n = length of [a] and m = length of [v].

    {2 Parameters}
    - [mode]: convolution mode; `Full (default), `Valid, or `Same
    - [a]: input tensor of shape [|n|]
    - [v]: kernel tensor of shape [|m|]

    {2 Returns}
    - a new tensor of shape [|out_len|] with same [dtype] as inputs

    {2 Raises}
    - [Invalid_argument] if [a] or [v] is not 1-D

    {2 Examples}
    {[
      let a = of_array float32 [|5|] [|1.;2.;3.;4.;5.|] in
      let v = of_array float32 [|3|] [|2.;1.;0.|] in
      let full = convolve1d ~mode:`Full a v in
      (* full = [|2.;5.;8.;11.;14.;13.;0.|] *)
      let valid = convolve1d ~mode:`Valid a v in
      (* valid = [|8.;11.;14.|] *)
      let same = convolve1d ~mode:`Same a v in
      (* same = [|2.;5.;8.;11.;14.|] *)
    ]} *)

val inv : (float, 'b) t -> (float, 'b) t
(** [inv a].

    Compute the inverse of the square matrix [a] using Gaussian elimination with
    partial pivoting.

    {2 Parameters}
    - [a]: input 2-D square tensor of floating-point dtype

    {2 Returns}
    - a new 2-D tensor of same shape and [dtype] containing the inverse

    {2 Raises}
    - [Invalid_argument] if [a] is not 2-D square
    - [Invalid_argument] if [a] is singular (non-invertible)

    {2 Examples}
    {[
      let id = eye float64 3 in
      let inv_id = inv id in
      (* inv_id = identity *)
    ]} *)

val solve : (float, 'b) t -> (float, 'b) t -> (float, 'b) t
(** [solve a b].

    Solve the linear system [a] * x = [b] using singular value decomposition.
    Supports least-squares solutions for over- or under-determined systems.

    {2 Parameters}
    - [a]: coefficient matrix of shape [|m; n|], 2-D float tensor
    - [b]: right-hand side tensor of shape [|m; k|] or vector of length [m]

    {2 Returns}
    - solution tensor [x] of shape [|n; k|] or vector of length [n], same dtype
      as [a]

    {2 Raises}
    - [Invalid_argument] if [a] is not 2-D

    {2 Examples}
    {[
      let a = of_array float64 [|2;2|] [|3.;1.;1.;2.|] in
      let b = of_array float64 [|2|] [|9.;8.|] in
      let x = solve a b in
      (* solves 3x + y = 9, x + 2y = 8 *)
    ]} *)

val svd : (float, 'b) t -> (float, 'b) t * (float, 'b) t * (float, 'b) t
(** [svd t].

    Compute the singular value decomposition of a 2-D tensor [t], so that [t] =
    U * diag(s) * Vᵀ. Only real-valued tensors supported. Allocates new tensors
    for components.

    {2 Parameters}
    - [t]: input 2-D float tensor of shape [|m; n|]

    {2 Returns}
    - a triple [(u, s, v)] where:
    - [u]: tensor of shape [|m; r|] with orthonormal columns
    - [s]: 1-D tensor of singular values of length [r], where r = min(m,n)
    - [v]: tensor of shape [|n; n|] with orthonormal columns

    {2 Raises}
    - [Invalid_argument] if [t] is not 2-D

    {2 Examples}
    {[
      let t = of_array float64 [|2;2|] [|1.;0.;0.;-1.|] in
      let (u, s, v) = svd t in
      (* t = u * diag s * v^T *)
    ]} *)

val eig : (float, 'b) t -> (float, 'b) t * (float, 'b) t
(** [eig a].

    Compute the eigenvalue decomposition of a square matrix [a], so that [a] = v
    * diag(w) * v⁻¹. Uses unshifted QR algorithm; only real-valued eigenvalues
    supported.

    {2 Parameters}
    - [a]: input 2-D square float tensor

    {2 Returns}
    - a pair [(w, v)] where:
    - [w]: 1-D tensor of eigenvalues (length n)
    - [v]: tensor of shape [|n; n|] whose columns are eigenvectors

    {2 Raises}
    - [Invalid_argument] if [a] is not square 2-D

    {2 Examples}
    {[
      let a = of_array float64 [|2;2|] [|2.;1.;1.;2.|] in
      let (w, v) = eig a in
      (* a * v = v * diag w *)
    ]} *)

val eigh : (float, 'b) t -> (float, 'b) t * (float, 'b) t
(** [eigh a].

    Compute the eigenvalue decomposition of a symmetric matrix [a], so that [a]
    = v * diag(w) * vᵀ. Uses Jacobi rotations; only real symmetric inputs.

    {2 Parameters}
    - [a]: input 2-D symmetric float tensor

    {2 Returns}
    - a pair [(w, v)] where:
    - [w]: 1-D tensor of eigenvalues (length n)
    - [v]: tensor of shape [|n; n|] whose columns are orthonormal eigenvectors

    {2 Raises}
    - [Invalid_argument] if [a] is not square 2-D

    {2 Examples}
    {[
      let a = of_array float64 [|2;2|] [|2.;1.;1.;2.|] in
      let (w, v) = eigh a in
      (* a = v * diag w * v^T *)
    ]} *)

(** {1 Sorting, Searching, and Unique}

    Order, locate, and dedupe elements. *)

val where : (int, uint8_elt) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [where mask a b].

    Returns a new tensor by selecting elements from [a] or [b], depending on
    [mask]: for each position, if [mask] element != 0 picks [a], else picks [b].
    Broadcasts [mask], [a], and [b] to a common shape before selection.

    {2 Parameters}
    - [mask]: uint8 tensor of 0/1 values
    - [a], [b]: tensors of the same dtype as output; must be
      broadcast-compatible with [mask]

    {2 Returns}
    - a fresh tensor of broadcast shape with elements chosen from [a] or [b]

    {2 Raises}
    - [Invalid_argument] if shapes are not broadcast-compatible or dtypes
      mismatch

    {2 Examples}
    {[
      let m = create uint8 [|2|] [|1;0|] in
      let a = create float32 [|2|] [|1.;2.|] in
      let b = create float32 [|2|] [|3.;4.|] in
      let c = where m a b in
      (* c = [|1.;4.|] *)
    ]} *)

val sort : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
(** [sort ?axis t].

    Return a new tensor with the elements of [t] sorted along [axis].

    {2 Parameters}
    - [axis]: axis to sort along; default is first axis (0)
    - [t]: input tensor

    {2 Returns}
    - a new tensor of same shape and dtype as [t], with elements sorted along
      the specified axis

    {2 Raises}
    - [Invalid_argument] if [axis] is out of bounds

    {2 Examples}
    {[
      let a = of_array float32 [|2;3|] [|3.;1.;2.;6.;4.;5.|] in
      let b = sort ~axis:1 a in
      (* b = [|[1.;2.;3.];[4.;5.;6.]|] *)
    ]} *)

val argsort : ?axis:int -> ('a, 'b) t -> (int64, int64_elt) t
(** [argsort ?axis t].

    Return the indices that would sort [t] along [axis].

    {2 Parameters}
    - [axis]: axis to sort along; default is first axis (0)
    - [t]: input tensor

    {2 Returns}
    - an [int64] tensor of same shape as [t], containing indices into [t] such
      that taking elements in that order along [axis] yields a sorted tensor

    {2 Raises}
    - [Invalid_argument] if [axis] is out of bounds

    {2 Examples}
    {[
      let a = of_array float32 [|3|] [|3.;1.;2.|] in
      let idx = argsort a in
      (* idx = [|1;2;0|] *)
    ]} *)

val argmax : ?axis:int -> ('a, 'b) t -> (int64, int64_elt) t
(** [argmax ?axis t].

    Return the indices of the maximum values of [t] along [axis].

    {2 Parameters}
    - [axis]: axis to reduce; default is first axis (0)
    - [t]: input tensor

    {2 Returns}
    - an [int64] tensor of rank one less than [t], containing the indices of the
      maximum values along the specified axis

    {2 Raises}
    - [Invalid_argument] if [axis] is out of bounds

    {2 Examples}
    {[
      let a = of_array float32 [|2;3|] [|1.;5.;2.;4.;3.;6.|] in
      let m = argmax a in
      (* m = [|1;2|] shape [|3|] *)
    ]} *)

val argmin : ?axis:int -> ('a, 'b) t -> (int64, int64_elt) t
(** [argmin ?axis t].

    Return the indices of the minimum values of [t] along [axis].

    {2 Parameters}
    - [axis]: axis to reduce; default is first axis (0)
    - [t]: input tensor

    {2 Returns}
    - an [int64] tensor of rank one less than [t], containing the indices of the
      minimum values along the specified axis

    {2 Raises}
    - [Invalid_argument] if [axis] is out of bounds

    {2 Examples}
    {[
      let a = of_array float32 [|2;3|] [|1.;5.;2.;4.;3.;6.|] in
      let m = argmin a in
      (* m = [|0;0|] shape [|3|] *)
    ]} *)

(* *)

val unique : ('a, 'b) t -> ('a, 'b) t
(** [unique t].

    Return a 1-D tensor of the unique elements in [t], sorted in ascending
    order. Flattens the input before extracting unique values.

    {2 Parameters}
    - [t]: input tensor of any shape

    {2 Returns}
    - a 1-D tensor of dtype same as [t], containing the unique values

    {2 Examples}
    {[
      let a = of_array int32 [|5|] [|3;1;2;3;2|] in
      let u = unique a in
      (* u = [|1;2;3|] *)
    ]} *)

val nonzero : ('a, 'b) t -> (int64, int64_elt) t
(** [nonzero t].

    Return the indices of non-zero elements in [t] as a 2-D int64 tensor. The
    first dimension corresponds to axes of [t], and the second dimension indexes
    the non-zero elements.

    {2 Parameters}
    - [t]: input tensor of any shape

    {2 Returns}
    - a 2-D [int64] tensor of shape [|d; k|], where d = number of dimensions of
      [t] and k = number of non-zero elements. Row i contains the indices along
      axis i.

    {2 Examples}
    {[
      let a = of_array int32 [|2;2|] [|1;0;2;0|] in
      let nz = nonzero a in
      (* nz = [|[0;1]; [0;0]|] *)
    ]} *)

(** {1 Random Sampling and Distributions}

    Draw random values and shuffle data. *)

val rand : (float, 'b) dtype -> ?seed:int -> int array -> (float, 'b) t
(** [rand dtype ?seed shape].

      Generate a tensor of the given [shape] with random values sampled uniformly
      from [0., 1.) of the specified floating-point [dtype].

      {2 Parameters}
      - [dtype]: floating-point dtype (Float32 or Float64)
      - [seed]: optional random seed; default uses system entropy
      - [shape]: array of dimensions for output tensor

      {2 Returns}
      - a new tensor of shape [shape] and dtype [dtype] with uniform random values

      {2 Examples}
      {[
        let a = rand Float32 [|2;3|] in
      ]} *)

val randn : (float, 'b) dtype -> ?seed:int -> int array -> (float, 'b) t
(** [randn dtype ?seed shape].

    Generate a tensor of the given [shape] with random values sampled from the
    standard normal distribution (mean 0, variance 1) of the specified
    floating-point [dtype].

    {2 Parameters}
    - [dtype]: floating-point dtype (Float32 or Float64)
    - [seed]: optional random seed; default uses system entropy
    - [shape]: array of dimensions for output tensor

    {2 Returns}
    - a new tensor of shape [shape] and dtype [dtype] with normal random values

    {2 Examples}
    {[
      let a = randn Float64 [|100|] in
    ]} *)

val randint :
  ('a, 'b) dtype -> ?seed:int -> ?high:int -> int array -> int -> ('a, 'b) t
(** [randint dtype ?seed ?high shape low].

    Generate a tensor of the given [shape] with random integer values drawn
    uniformly from the half-open interval \[low, high). If [high] is omitted,
    values are drawn from \[0, low).

    {2 Parameters}
    - [dtype]: integer dtype for output (Int8, Int16, Int32, Int64, etc.)
    - [seed]: optional random seed; default uses system entropy
    - [high]: optional exclusive upper bound; if omitted, [low] is used as the
      upper bound and 0 is the lower bound
    - [shape]: array of dimensions for output tensor
    - [low]: inclusive lower bound, or if [high] is omitted, this is treated as
      the upper bound

    {2 Returns}
    - a new tensor of shape [shape] and dtype [dtype] with random integers

    {2 Raises}
    - [Invalid_argument] if the specified range is not positive

    {2 Examples}
    {[
      let a = randint Int32 ~seed:42 [|3;3|] 10 in
      (* values uniformly in [0,10) *)
      let b = randint Int64 ~high:20 [|5|] 5 in
      (* values uniformly in [5,20) *)
    ]} *)

(** {1 Logical and Masking Operations}

    Boolean logic, NaN/Inf tests, and masking utilities. *)

val logical_and : (int, uint8_elt) t -> (int, uint8_elt) t -> (int, uint8_elt) t
(** [logical_and a b].

    Compute the elementwise logical AND of masks [a] and [b], treating non-zero
    values as true. Broadcasts inputs to a common shape.

    {2 Parameters}
    - [a], [b]: input uint8 tensors of 0/1 values; must be broadcast-compatible

    {2 Returns}
    - a new uint8 tensor of broadcast shape where each element is 1 if both
      corresponding elements of [a] and [b] are non-zero, else 0

    {2 Examples}
    {[
      let m1 = of_array uint8 [|2|] [|1;0|] in
      let m2 = of_array uint8 [|2|] [|0;1|] in
      let out = logical_and m1 m2 in
      (* out = [|0;0|] *)
    ]} *)

val logical_or : (int, uint8_elt) t -> (int, uint8_elt) t -> (int, uint8_elt) t
(** [logical_or a b].

    Compute the elementwise logical OR of masks [a] and [b], treating non-zero
    values as true. Broadcasts inputs to a common shape.

    {2 Parameters}
    - [a], [b]: input uint8 tensors of 0/1 values; must be broadcast-compatible

    {2 Returns}
    - a new uint8 tensor of broadcast shape where each element is 1 if either
      corresponding element of [a] or [b] is non-zero, else 0

    {2 Examples}
    {[
      let m1 = of_array uint8 [|3|] [|0;1;0|] in
      let m2 = of_array uint8 [|3|] [|1;0;0|] in
      let out = logical_or m1 m2 in
      (* out = [|1;1;0|] *)
    ]} *)

val logical_not : (int, uint8_elt) t -> (int, uint8_elt) t
(** [logical_not a].

    Compute the elementwise logical NOT of mask [a], treating non-zero values as
    true. Returns a new mask of the same shape.

    {2 Parameters}
    - [a]: input uint8 tensor of 0/1 values

    {2 Returns}
    - a new uint8 tensor of same shape where each element is 0 if [a] element is
      non-zero, else 1

    {2 Examples}
    {[
      let m = of_array uint8 [|3|] [|1;0;1|] in
      let out = logical_not m in
      (* out = [|0;1;0|] *)
    ]} *)

val logical_xor : (int, uint8_elt) t -> (int, uint8_elt) t -> (int, uint8_elt) t
(** [logical_xor a b].

    Compute the elementwise logical XOR of masks [a] and [b], treating non-zero
    values as true. Broadcasts inputs to a common shape.

    {2 Parameters}
    - [a], [b]: input uint8 tensors of 0/1 values; must be broadcast-compatible

    {2 Returns}
    - a new uint8 tensor of broadcast shape where each element is 1 if exactly
      one of the corresponding elements of [a] or [b] is non-zero, else 0

    {2 Examples}
    {[
      let m1 = of_array uint8 [|2|] [|1;1|] in
      let m2 = of_array uint8 [|2|] [|1;0|] in
      let out = logical_xor m1 m2 in
      (* out = [|0;1|] *)
    ]} *)

val isnan : (float, 'b) t -> (int, uint8_elt) t
(** [isnan t].

    Return a mask indicating NaN elements of [t]. For each position, output is 1
    if [t] element is NaN, else 0.

    {2 Parameters}
    - [t]: input float tensor

    {2 Returns}
    - a uint8 tensor of same shape with 1 for NaN elements and 0 otherwise

    {2 Examples}
    {[
      let a = of_array float32 [|3|] [|nan; 1.; nan|] in
      let m = isnan a in
      (* m = [|1;0;1|] *)
    ]} *)

val isinf : (float, 'b) t -> (int, uint8_elt) t
(** [isinf t].

    Return a mask indicating infinite elements of [t]. For each position, output
    is 1 if [t] element is +∞ or -∞, else 0.

    {2 Parameters}
    - [t]: input float tensor

    {2 Returns}
    - a uint8 tensor of same shape with 1 for infinite elements and 0 otherwise

    {2 Examples}
    {[
      let a = of_array float64 [|4|] [|1.; infinity; -1.; neg_infinity|] in
      let m = isinf a in
      (* m = [|0;1;0;1|] *)
    ]} *)

val isfinite : (float, 'b) t -> (int, uint8_elt) t
(** [isfinite t].

    Return a mask indicating finite elements of [t]. For each position, output
    is 1 if [t] element is neither NaN nor infinite, else 0.

    {2 Parameters}
    - [t]: input float tensor

    {2 Returns}
    - a uint8 tensor of same shape with 1 for finite elements and 0 otherwise

    {2 Examples}
    {[
      let a = of_array float64 [|3|] [|nan; 1.; infinity|] in
      let m = isfinite a in
      (* m = [|0;1;0|] *)
    ]} *)

val array_equal : ('a, 'b) t -> ('a, 'b) t -> bool
(** [array_equal a b].

    Test whether two tensors [a] and [b] have the same shape and all
    corresponding elements equal. Returns a boolean result.

    {2 Parameters}
    - [a], [b]: input tensors of same dtype

    {2 Returns}
    - [true] if [a] and [b] have identical shape and elementwise equality; else
      [false]

    {2 Examples}
    {[
      let x = of_array float32 [| 2 |] [| 1.; 2. |] in
      let y = of_array float32 [| 2 |] [| 1.; 2. |] in
      let z = of_array float32 [| 2 |] [| 2.; 1. |] in
      assert (array_equal x y);
      assert (not (array_equal x z))
    ]} *)

(** {1 Functional and Higher‑order Operations}

    Map, fold, iterate, and apply along axes. *)

val map : ('a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
(** [map f t].

    Apply function [f] to each element of tensor [t] and return a new tensor of
    the same shape and dtype. Executes in row-major order.

    {2 Parameters}
    - [f]: unary mapping function
    - [t]: input tensor

    {2 Returns}
    - a new tensor where each element is [f] applied to the corresponding
      element of [t]

    {2 Examples}
    {[
      let a = of_array float32 [|3|] [|1.;2.;3.|] in
      let b = map (fun x -> x *. 2.) a in
      (* b = [|2.;4.;6.|] *)
    ]} *)

val iter : ('a -> unit) -> ('a, 'b) t -> unit
(** [iter f t].

    Apply function [f] to each element of tensor [t] for side effects, in
    row-major order. Does not allocate a new tensor.

    {2 Parameters}
    - [f]: function to apply to each element
    - [t]: input tensor

    {2 Examples}
    {[
      let sum = ref 0. in
      iter (fun x -> sum := !sum +. x) a
      (* sum = total of elements in [a] *)
    ]} *)

val fold : ('a -> 'a -> 'a) -> 'a -> ('a, 'b) t -> 'a
(** [fold f acc t].

    Reduce elements of tensor [t] into a single value by applying binary
    function [f] with initial accumulator [acc], in row-major order.

    {2 Parameters}
    - [f]: accumulator function of type [acc -> element -> acc]
    - [acc]: initial accumulator value
    - [t]: input tensor

    {2 Returns}
    - final accumulated result after processing all elements

    {2 Examples}
    {[
      let total = fold ( +. ) 0.0 a in
      (* total = sum of elements in [a] *)
    ]} *)

(** {1 Utilities: Printing and Debugging}

    Pretty‑printing, to‑string conversion, and formatters. *)

val pp_dtype : Format.formatter -> ('a, 'b) dtype -> unit
(** [pp_dtype fmt dtype].

    Pretty-print the data type [dtype] to formatter [fmt].

    {2 Parameters}
    - [fmt]: output formatter
    - [dtype]: data type to print *)

val dtype_to_string : ('a, 'b) dtype -> string
(** [dtype_to_string dtype].

    Return a string representation of data type [dtype], e.g. "float32".

    {2 Parameters}
    - [dtype]: data type to convert

    {2 Returns}
    - string naming the data type *)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape fmt shape].

    Pretty-print the shape [shape] to formatter [fmt] as e.g. "[2x3x4]".

    {2 Parameters}
    - [fmt]: output formatter
    - [shape]: array of dimensions *)

val shape_to_string : int array -> string
(** [shape_to_string shape].

    Return a string representation of [shape], formatted as "[d0xd1x...]".

    {2 Parameters}
    - [shape]: array of dimensions

    {2 Returns}
    - string representation of the shape *)

val pp : Format.formatter -> ('a, 'b) t -> unit
(** [pp fmt t].

    Pretty-print tensor [t] to formatter [fmt] using nested brackets.

    {2 Parameters}
    - [fmt]: output formatter
    - [t]: tensor to print *)

val to_string : ('a, 'b) t -> string
(** [to_string t].

    Return a string containing the pretty-printed tensor [t], equivalent to
    printing with [pp] into a buffer.

    {2 Parameters}
    - [t]: tensor to convert

    {2 Returns}
    - string representation of [t] *)

val print : ('a, 'b) t -> unit
(** [print t].

    Print tensor [t] to standard output using [pp], followed by a newline.

    {2 Parameters}
    - [t]: tensor to print *)

val pp_info : Format.formatter -> ('a, 'b) t -> unit
(** [pp_info fmt t].

    Pretty-print metadata of tensor [t] (shape, dtype, strides, offset, size) to
    formatter [fmt].

    {2 Parameters}
    - [fmt]: output formatter
    - [t]: tensor whose metadata to print *)

val to_string_info : ('a, 'b) t -> string
(** [to_string_info t].

    Return a string containing metadata of tensor [t], similar to [pp_info].

    {2 Parameters}
    - [t]: tensor whose metadata to convert

    {2 Returns}
    - string representation of [t]’s metadata *)

val print_info : ('a, 'b) t -> unit
(** [print_info t].

    Print metadata of tensor [t] to standard output using [pp_info], followed by
    a newline.

    {2 Parameters}
    - [t]: tensor whose metadata to print *)
