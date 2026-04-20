(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Backend interface for Nx tensor operations.

    This module type defines the contract between Nx's frontend and its
    pluggable backends. Backends may execute operations eagerly (C backend),
    raise effects for JIT compilation (Rune), build computation graphs, or
    implement other execution strategies.

    {1 Design Philosophy}

    Operations exist at the level of C standard library functions: every
    operation that maps to a C stdlib call is a backend primitive, avoiding the
    overhead of composing multiple operations in eager mode. Rune's JIT pipeline
    can decompose these into lower primitives when building computation graphs.

    {1 Frontend/Backend Contract}

    The frontend is responsible for:
    - Broadcasting inputs to matching shapes before calling binary operations.
    - Promoting dtypes to compatible types before calling operations.
    - Validating parameters (axes in range, shapes compatible, etc.).

    The backend can assume all inputs are well-formed. It is responsible for:
    - Executing the operation correctly for all supported dtypes.
    - Handling strided (non-contiguous) inputs via the view metadata.
    - Returning tensors with correct view metadata.

    {1 Conventions}

    - All compute operations allocate and return their result. The frontend
      passes pre-broadcasted, pre-validated inputs and receives the result
      tensor.
    - Movement operations manipulate view metadata (shape, strides, offset)
      without copying data when possible. *)
module type S = sig
  (** {1 Types} *)

  type ('a, 'b) t
  (** ['a] is the OCaml element type (e.g., [float], [int32]). ['b] is a phantom
      type that tags the dtype for type safety. *)

  type context
  (** Backend execution context.

      Carries backend-specific state such as memory pools, device handles,
      command queues, or computation graphs. *)

  (** {1 Tensor Properties} *)

  val view : ('a, 'b) t -> View.t
  (** [view t] returns the strided view metadata describing [t]'s logical layout
      (shape, strides, offset) over its underlying buffer. *)

  val dtype : ('a, 'b) t -> ('a, 'b) Dtype.t
  (** [dtype t] returns the element type of [t]. *)

  val context : ('a, 'b) t -> context
  (** [context t] returns the execution context that owns [t]. *)

  val to_host : ('a, 'b) t -> ('a, 'b) Nx_buffer.t
  (** [to_host t] returns [t]'s data as a flat, C-contiguous host buffer.

      Use {!view} to interpret the logical structure. CPU backends may return a
      direct reference (zero-copy); GPU backends copy from device to host. *)

  (** {1 Tensor Creation} *)

  val buffer : context -> ('a, 'b) Dtype.t -> int array -> ('a, 'b) t
  (** [buffer ctx dtype shape] allocates an uninitialized tensor.

      Contents are undefined. Used internally by backends to allocate output
      tensors.

      {b Backend must:} return a tensor with the given shape and dtype whose
      view is C-contiguous. *)

  val full : context -> ('a, 'b) Dtype.t -> int array -> 'a -> ('a, 'b) t
  (** [full ctx dtype shape value] creates a tensor where every element is
      [value].

      For scalars, [shape] is [[||]]. Subsumes zeros, ones, and constant fill.

      {b Backend must:} return a C-contiguous tensor of the given shape and
      dtype with all elements set to [value]. *)

  val from_host : context -> ('a, 'b) Nx_buffer.t -> ('a, 'b) t
  (** [from_host ctx buf] creates a tensor from a flat, C-contiguous host
      buffer.

      CPU backends may share the buffer directly (zero-copy). GPU backends copy
      from host to device.

      {b Frontend guarantees:} [buf] is C-contiguous. *)

  (** {1 Element-wise Binary Operations}

      {b Frontend guarantees:} [a] and [b] have identical shapes (after
      broadcasting) and compatible dtypes (after promotion).

      {b Backend must:} allocate a C-contiguous output tensor with the correct
      shape and write the result.

      {2 Arithmetic} *)

  val add : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [add a b] is the element-wise sum of [a] and [b]. *)

  val sub : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [sub a b] is the element-wise difference of [a] and [b]. *)

  val mul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [mul a b] is the element-wise product of [a] and [b]. *)

  val div : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [div a b] is the element-wise quotient of [a] and [b].

      Integer dtypes use truncation toward zero (C division). Floating-point
      dtypes use IEEE 754 division. *)

  val mod_ : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [mod_ a b] is the element-wise remainder of [a / b].

      Integers use C's [%] operator (truncated division). Floats use [fmod]. The
      sign of the result follows the dividend [a]. *)

  val pow : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [pow base exponent] is the element-wise power [base ^ exponent]. *)

  val atan2 : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [atan2 y x] is the element-wise arc tangent of [y / x].

      Returns the angle in radians in [(-π, π\]], handling all quadrants. *)

  (** {2 Comparison}

      Comparison operations produce boolean tensors. *)

  val cmpeq : ('a, 'b) t -> ('a, 'b) t -> (bool, Dtype.bool_elt) t
  (** [cmpeq a b] is the element-wise equality test of [a] and [b]. *)

  val cmpne : ('a, 'b) t -> ('a, 'b) t -> (bool, Dtype.bool_elt) t
  (** [cmpne a b] is the element-wise inequality test of [a] and [b]. *)

  val cmplt : ('a, 'b) t -> ('a, 'b) t -> (bool, Dtype.bool_elt) t
  (** [cmplt a b] is the element-wise less-than test of [a] and [b]. *)

  val cmple : ('a, 'b) t -> ('a, 'b) t -> (bool, Dtype.bool_elt) t
  (** [cmple a b] is the element-wise less-or-equal test of [a] and [b]. *)

  (** {2 Min/Max} *)

  val max : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [max a b] is the element-wise maximum of [a] and [b]. *)

  val min : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [min a b] is the element-wise minimum of [a] and [b]. *)

  (** {2 Bitwise}

      Operate on the binary representation of integer and boolean dtypes. For
      booleans, these are equivalent to logical AND/OR/XOR. *)

  val xor : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [xor a b] is the element-wise bitwise XOR of [a] and [b]. *)

  val or_ : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [or_ a b] is the element-wise bitwise OR of [a] and [b]. *)

  val and_ : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [and_ a b] is the element-wise bitwise AND of [a] and [b]. *)

  (** {1 Element-wise Unary Operations}

      {b Frontend guarantees:} [x] has compatible dtype.

      {b Backend must:} allocate a C-contiguous output tensor with the correct
      shape and write the result.

      {2 Arithmetic} *)

  val neg : ('a, 'b) t -> ('a, 'b) t
  (** [neg x] is the element-wise negation of [x]. *)

  val recip : ('a, 'b) t -> ('a, 'b) t
  (** [recip x] is the element-wise reciprocal of [x]. *)

  val abs : ('a, 'b) t -> ('a, 'b) t
  (** [abs x] is the element-wise absolute value of [x]. *)

  val sqrt : ('a, 'b) t -> ('a, 'b) t
  (** [sqrt x] is the element-wise square root of [x]. *)

  val sign : ('a, 'b) t -> ('a, 'b) t
  (** [sign x] is the element-wise sign of [x]: [-1] for negative, [0] for zero,
      [1] for positive. Returns NaN for floating-point NaN inputs. *)

  (** {2 Exponential and Logarithm} *)

  val exp : ('a, 'b) t -> ('a, 'b) t
  (** [exp x] is the element-wise exponential of [x]. *)

  val log : ('a, 'b) t -> ('a, 'b) t
  (** [log x] is the element-wise natural logarithm of [x]. *)

  (** {2 Trigonometric}

      All inputs are in radians. *)

  val sin : ('a, 'b) t -> ('a, 'b) t
  (** [sin x] is the element-wise sine of [x]. *)

  val cos : ('a, 'b) t -> ('a, 'b) t
  (** [cos x] is the element-wise cosine of [x]. *)

  val tan : ('a, 'b) t -> ('a, 'b) t
  (** [tan x] is the element-wise tangent of [x]. *)

  val asin : ('a, 'b) t -> ('a, 'b) t
  (** [asin x] is the element-wise arc sine of [x].

      Returns values in [[-π/2, π/2]]. *)

  val acos : ('a, 'b) t -> ('a, 'b) t
  (** [acos x] is the element-wise arc cosine of [x].

      Returns values in [[0, π]]. *)

  val atan : ('a, 'b) t -> ('a, 'b) t
  (** [atan x] is the element-wise arc tangent of [x].

      Returns values in [[-π/2, π/2]]. *)

  (** {2 Hyperbolic} *)

  val sinh : ('a, 'b) t -> ('a, 'b) t
  (** [sinh x] is the element-wise hyperbolic sine of [x]. *)

  val cosh : ('a, 'b) t -> ('a, 'b) t
  (** [cosh x] is the element-wise hyperbolic cosine of [x]. *)

  val tanh : ('a, 'b) t -> ('a, 'b) t
  (** [tanh x] is the element-wise hyperbolic tangent of [x]. *)

  (** {2 Rounding}

      For integer dtypes, all rounding operations are the identity. *)

  val trunc : ('a, 'b) t -> ('a, 'b) t
  (** [trunc x] rounds each element toward zero. *)

  val ceil : ('a, 'b) t -> ('a, 'b) t
  (** [ceil x] rounds each element toward positive infinity. *)

  val floor : ('a, 'b) t -> ('a, 'b) t
  (** [floor x] rounds each element toward negative infinity. *)

  val round : ('a, 'b) t -> ('a, 'b) t
  (** [round x] rounds each element to nearest integer, half away from zero (C's
      [round]). *)

  (** {2 Special Functions} *)

  val erf : ('a, 'b) t -> ('a, 'b) t
  (** [erf x] computes the error function [erf(x) = 2/√π ∫₀ˣ e^(-t²) dt]. *)

  (** {1 Ternary Operations} *)

  val where : (bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [where cond if_true if_false] selects elements: [if_true.{i}] where
      [cond.{i}] is true, [if_false.{i}] otherwise.

      {b Frontend guarantees:} all three input tensors have identical shapes.
      [cond] is boolean. [if_true] and [if_false] share the same dtype. *)

  (** {1 Reduction Operations}

      Reductions aggregate values along one or more axes.

      {b Frontend guarantees:} [axes] contains valid, non-negative, deduplicated
      axis indices. *)

  val reduce_sum : axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [reduce_sum ~axes ~keepdims x] sums elements of [x] along [axes]. *)

  val reduce_prod : axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [reduce_prod ~axes ~keepdims x] multiplies elements of [x] along [axes].
  *)

  val reduce_max : axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [reduce_max ~axes ~keepdims x] finds the maximum of [x] along [axes]. *)

  val reduce_min : axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [reduce_min ~axes ~keepdims x] finds the minimum of [x] along [axes]. *)

  val argmax :
    axis:int -> keepdims:bool -> ('a, 'b) t -> (int32, Dtype.int32_elt) t
  (** [argmax ~axis ~keepdims x] returns int32 indices of maximum values of [x]
      along [axis]. For ties, returns the first occurrence.

      {b Frontend guarantees:} [axis] is valid and non-negative. *)

  val argmin :
    axis:int -> keepdims:bool -> ('a, 'b) t -> (int32, Dtype.int32_elt) t
  (** [argmin ~axis ~keepdims x] returns int32 indices of minimum values of [x]
      along [axis]. For ties, returns the first occurrence.

      {b Frontend guarantees:} [axis] is valid and non-negative. *)

  val associative_scan :
    axis:int -> op:[ `Sum | `Prod | `Max | `Min ] -> ('a, 'b) t -> ('a, 'b) t
  (** [associative_scan ~axis ~op x] computes an inclusive prefix scan of [x]
      along [axis]. [`Sum] for cumulative sum, [`Prod] for cumulative product,
      [`Max]/[`Min] for running max/min.

      {b Frontend guarantees:} [axis] is valid and non-negative. *)

  (** {1 Sort Operations}

      {b Frontend guarantees:} [axis] is valid and non-negative. *)

  val sort : axis:int -> descending:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [sort ~axis ~descending x] sorts elements of [x] along [axis]. NaN values
      are placed at the end regardless of sort direction. *)

  val argsort :
    axis:int -> descending:bool -> ('a, 'b) t -> (int32, Dtype.int32_elt) t
  (** [argsort ~axis ~descending x] returns int32 indices that would sort
      elements of [x] along [axis]. *)

  (** {1 Movement Operations}

      Movement operations manipulate view metadata (shape, strides, offset)
      without copying data when possible. They return new tensor handles sharing
      the underlying buffer.

      {b Frontend guarantees:} all parameters are validated (axes in range,
      shapes compatible, bounds within limits).

      {b Backend must:} return a tensor with the correct view metadata. May
      share the underlying buffer (zero-copy) or allocate if necessary. *)

  val expand : ('a, 'b) t -> int array -> ('a, 'b) t
  (** [expand t shape] broadcasts dimensions of size 1 to match [shape] by
      setting their stride to 0. Non-singleton dimensions must already match.
      Zero-copy. *)

  val reshape : ('a, 'b) t -> int array -> ('a, 'b) t
  (** [reshape t shape] changes the logical shape, preserving element count.

      Zero-copy when [t] is C-contiguous or the reshape is compatible with the
      current strides. May copy if [t] is non-contiguous. *)

  val permute : ('a, 'b) t -> int array -> ('a, 'b) t
  (** [permute t axes] reorders dimensions according to [axes], which must be a
      permutation of [[0, ..., ndim-1]]. Zero-copy. *)

  val shrink : ('a, 'b) t -> (int * int) array -> ('a, 'b) t
  (** [shrink t ranges] extracts a contiguous slice. [ranges.(i)] is
      [(start, stop)] with exclusive [stop]. Zero-copy (adjusts offset and
      shape). *)

  val flip : ('a, 'b) t -> bool array -> ('a, 'b) t
  (** [flip t axes] reverses dimensions where [axes.(i) = true] by negating
      strides. Zero-copy. *)

  val pad : ('a, 'b) t -> (int * int) array -> 'a -> ('a, 'b) t
  (** [pad t padding fill_value] extends [t] with [fill_value]. [padding.(i)] is
      [(before, after)] for dimension [i].

      {b Backend must:} allocate a new buffer and copy data. *)

  val cat : ('a, 'b) t list -> axis:int -> ('a, 'b) t
  (** [cat tensors ~axis] concatenates [tensors] along [axis].

      {b Frontend guarantees:} all tensors have the same shape except along
      [axis]. [axis] is valid. The list is non-empty. *)

  (** {1 Type Conversion and Memory} *)

  val cast : dtype:('c, 'd) Dtype.t -> ('a, 'b) t -> ('c, 'd) t
  (** [cast ~dtype x] converts elements of [x] to [dtype].

      Float-to-int truncates toward zero. Int-to-float may lose precision for
      large values. *)

  val contiguous : ('a, 'b) t -> ('a, 'b) t
  (** [contiguous t] returns a C-contiguous version of [t].

      May return [t] unchanged if already C-contiguous. Otherwise allocates and
      copies.

      {b Backend must:} return a C-contiguous tensor with the same data. *)

  val copy : ('a, 'b) t -> ('a, 'b) t
  (** [copy t] creates an independent copy with its own buffer.

      {b Backend must:} always allocate a new buffer, even if [t] is already
      contiguous. *)

  val assign : ('a, 'b) t -> ('a, 'b) t -> unit
  (** [assign dst src] copies elements from [src] into [dst] in-place.

      {b Frontend guarantees:} [dst] and [src] have matching shapes and dtypes.

      {b Backend must:} write [src]'s data into [dst]'s buffer, respecting both
      tensors' strides. *)

  (** {1 Random Number Generation} *)

  val threefry :
    (int32, Dtype.int32_elt) t ->
    (int32, Dtype.int32_elt) t ->
    (int32, Dtype.int32_elt) t
  (** [threefry key counter] applies the Threefry-2x32 hash function.

      {b Frontend guarantees:} [key] and [counter] are int32 tensors with
      compatible shapes. *)

  (** {1 Indexed Access Operations} *)

  val gather :
    ('a, 'b) t -> (int32, Dtype.int32_elt) t -> axis:int -> ('a, 'b) t
  (** [gather data indices ~axis] selects elements from [data] along [axis]
      using [indices].

      {b Frontend guarantees:} [rank data = rank indices]. [axis] is valid.
      Index values are in range for [data]'s size along [axis]. *)

  val scatter :
    ?mode:[ `Set | `Add ] ->
    ?unique_indices:bool ->
    ('a, 'b) t ->
    indices:(int32, Dtype.int32_elt) t ->
    updates:('a, 'b) t ->
    axis:int ->
    ('a, 'b) t
  (** [scatter ?mode ?unique_indices template ~indices ~updates ~axis] places
      [updates] into a tensor shaped like [template] along [axis].

      [`Set] (default) uses the last update for duplicate indices. [`Add]
      accumulates. [unique_indices = true] hints that indices are unique.

      {b Frontend guarantees:} [rank indices = rank updates]. [axis] is valid.
      [template] has the desired output shape.

      {b Backend must:} allocate and return the result tensor, initialized from
      [template]'s data. *)

  (** {1 Window Operations}

      Sliding-window extraction and its inverse. Used to implement convolution
      as [unfold + reshape + matmul] and pooling as [unfold + reduce]. *)

  val unfold :
    ('a, 'b) t ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) t
  (** [unfold t ~kernel_size ~stride ~dilation ~padding] extracts sliding
      windows from the last [K] spatial dimensions, where
      [K = Array.length kernel_size].

      Input shape [(leading..., spatial...)] produces
      [(leading..., prod(kernel_size), L)] where [L] is the number of windows.
      All dimensions before the last [K] are preserved as-is.

      {b Frontend guarantees:} all array parameters have length [K]. Values are
      positive. Input has at least [K] dimensions.

      {b Backend must:} allocate and return the result tensor. *)

  val fold :
    ('a, 'b) t ->
    output_size:int array ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) t
  (** [fold t ~output_size ~kernel_size ~stride ~dilation ~padding] combines
      sliding windows (inverse of {!unfold}). Overlapping values are summed.

      Input shape [(leading..., prod(kernel_size), L)] produces
      [(leading..., output_size...)].

      {b Frontend guarantees:} parameters are consistent with a valid unfold
      configuration.

      {b Backend must:} allocate and return the result tensor. *)

  (** {1 Matrix Operations} *)

  val matmul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [matmul a b] computes matrix multiplication [a × b].

      For 2D inputs: standard matrix multiply. For higher dimensions: batched
      multiply on the last two dimensions, with broadcasting via strides.

      {b Frontend guarantees:} [a]'s last dim equals [b]'s second-to-last dim.

      {b Backend must:} allocate and return the result. May use BLAS for
      performance. [a] and [b] may be non-contiguous. *)

  (** {1 Fourier Transforms}

      {b Frontend guarantees:} [axes] contains valid, non-negative axis indices.
      Input tensors have compatible complex or real dtypes. *)

  val fft :
    ?out:(Complex.t, 'b) t ->
    (Complex.t, 'b) t ->
    axes:int array ->
    (Complex.t, 'b) t
  (** [fft ?out t ~axes] computes the forward DFT along [axes]. *)

  val ifft :
    ?out:(Complex.t, 'b) t ->
    (Complex.t, 'b) t ->
    axes:int array ->
    (Complex.t, 'b) t
  (** [ifft ?out t ~axes] computes the inverse DFT along [axes]. *)

  val rfft :
    ?out:(Complex.t, 'b) t ->
    (float, 'a) t ->
    dtype:(Complex.t, 'b) Dtype.t ->
    axes:int array ->
    (Complex.t, 'b) t
  (** [rfft ?out t ~dtype ~axes] computes the real-input DFT along [axes].

      Exploits conjugate symmetry to return only the non-redundant half of the
      spectrum along the last transformed axis. *)

  val irfft :
    ?out:(float, 'b) t ->
    ?s:int array ->
    (Complex.t, 'a) t ->
    dtype:(float, 'b) Dtype.t ->
    axes:int array ->
    (float, 'b) t
  (** [irfft ?out ?s t ~dtype ~axes] computes the inverse real-input DFT along
      [axes].

      Takes conjugate-symmetric complex input, returns real output. [s]
      specifies output sizes along the transformed axes; [None] infers sizes
      from the input. *)

  val dct :
    (float, 'a) t ->
    dct_type:int ->
    ortho:bool ->
    axes:int array ->
    (float, 'a) t
  (** [dct t ~dct_type ~ortho ~axes] computes the Discrete Cosine Transform
      along [axes]. [dct_type] is 1, 2, 3, or 4. *)

  val dst :
    (float, 'a) t ->
    dst_type:int ->
    ortho:bool ->
    axes:int array ->
    (float, 'a) t
  (** [dst t ~dst_type ~ortho ~axes] computes the Discrete Sine Transform
      along [axes]. [dst_type] is 1, 2, 3, or 4. *)

  (** {1 Linear Algebra}

      All linalg operations support batching: the last two dimensions are the
      matrix dimensions, earlier dimensions are batch dimensions.

      {b Frontend guarantees:} input matrices have compatible shapes (square
      where required, matching dimensions for solves).

      {b Backend must:} allocate and return result tensors. Typically delegates
      to LAPACK. *)

  val cholesky : upper:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [cholesky ~upper t] computes the Cholesky factorization of a
      positive-definite matrix. Returns [L] (lower) or [U] (upper) such that
      [A = L·Lᵀ] or [A = Uᵀ·U].

      @raise Failure if not positive-definite. *)

  val qr : reduced:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
  (** [qr ~reduced t] returns [(Q, R)] where [Q] is orthogonal and [R] is upper
      triangular. [reduced = true] returns economy-size factorization. *)

  val svd :
    full_matrices:bool ->
    ('a, 'b) t ->
    ('a, 'b) t * (float, Dtype.float64_elt) t * ('a, 'b) t
  (** [svd ~full_matrices t] returns [(U, S, Vᴴ)]. [S] is a 1D float64 vector of
      singular values in descending order. [full_matrices = false] returns thin
      SVD. *)

  val eig :
    vectors:bool ->
    ('a, 'b) t ->
    (Complex.t, Dtype.complex64_elt) t
    * (Complex.t, Dtype.complex64_elt) t option
  (** [eig ~vectors t] computes eigenvalues (and optionally eigenvectors) of a
      square matrix. Returns complex64 results. *)

  val eigh :
    vectors:bool ->
    ('a, 'b) t ->
    (float, Dtype.float64_elt) t * ('a, 'b) t option
  (** [eigh ~vectors t] computes eigenvalues (and optionally eigenvectors) of a
      symmetric/Hermitian matrix. Eigenvalues are float64. *)

  val triangular_solve :
    upper:bool ->
    transpose:bool ->
    unit_diag:bool ->
    ('a, 'b) t ->
    ('a, 'b) t ->
    ('a, 'b) t
  (** [triangular_solve ~upper ~transpose ~unit_diag a b] solves [A·x = b] or
      [Aᵀ·x = b] where [A] is triangular.

      [upper]: [A] is upper triangular. [transpose]: solve [Aᵀ·x = b].
      [unit_diag]: assume diagonal is all ones. *)
end
