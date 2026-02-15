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
    - Allocating output tensors with the correct shape and dtype.

    The backend can assume all inputs are well-formed. It is responsible for:
    - Executing the operation correctly for all supported dtypes.
    - Handling strided (non-contiguous) inputs via the view metadata.
    - Returning tensors with correct view metadata.

    {1 Conventions}

    - Binary, unary, reduction, and other compute operations write results to a
      caller-provided [~out] buffer for memory reuse. The frontend controls all
      allocation.
    - Movement operations manipulate view metadata (shape, strides, offset)
      without copying data when possible.
    - Operations that must allocate by nature ([copy], [contiguous], [pad],
      [scatter]) return new tensor handles. *)
module type S = sig
  (** {1 Types} *)

  type ('a, 'b) t
  (** ['a] is the OCaml element type (e.g., [float], [int32]). ['b] is a
      phantom type that tags the dtype for type safety. *)

  type context
  (** Backend execution context.

      Carries backend-specific state such as memory pools, device handles,
      command queues, or computation graphs. *)

  (** {1 Tensor Properties} *)

  val view : ('a, 'b) t -> View.t
  (** [view t] returns the strided view metadata describing [t]'s logical
      layout (shape, strides, offset) over its underlying buffer. *)

  val dtype : ('a, 'b) t -> ('a, 'b) Dtype.t
  (** [dtype t] returns the element type of [t]. *)

  val context : ('a, 'b) t -> context
  (** [context t] returns the execution context that owns [t]. *)

  val to_host : ('a, 'b) t -> ('a, 'b, Nx_buffer.c_layout) Nx_buffer.Array1.t
  (** [to_host t] returns [t]'s data as a flat, C-contiguous host buffer.

      Use {!view} to interpret the logical structure. CPU backends may return a
      direct reference (zero-copy); GPU backends copy from device to host. *)

  (** {1 Tensor Creation} *)

  val buffer : context -> ('a, 'b) Dtype.t -> int array -> ('a, 'b) t
  (** [buffer ctx dtype shape] allocates an uninitialized tensor.

      Contents are undefined. Used internally by the frontend to pre-allocate
      [~out] buffers before calling operations.

      {b Backend must:} return a tensor with the given shape and dtype whose
      view is C-contiguous. *)

  val full : context -> ('a, 'b) Dtype.t -> int array -> 'a -> ('a, 'b) t
  (** [full ctx dtype shape value] creates a tensor where every element is
      [value].

      For scalars, [shape] is [\[||\]]. Subsumes zeros, ones, and constant fill.

      {b Backend must:} return a C-contiguous tensor of the given shape and
      dtype with all elements set to [value]. *)

  val from_host :
    context -> ('a, 'b, Nx_buffer.c_layout) Nx_buffer.Array1.t -> ('a, 'b) t
  (** [from_host ctx buf] creates a tensor from a flat, C-contiguous host
      buffer.

      CPU backends may share the buffer directly (zero-copy). GPU backends copy
      from host to device.

      {b Frontend guarantees:} [buf] is C-contiguous. *)

  (** {1 Element-wise Binary Operations}

      {b Frontend guarantees:} [out], [a], and [b] have identical shapes (after
      broadcasting) and compatible dtypes (after promotion). [out] is
      C-contiguous and pre-allocated with the correct shape.

      {b Backend must:} write exactly [numel] elements to [out], respecting the
      strides of [a] and [b] (which may be non-contiguous or broadcast).

      {2 Arithmetic} *)

  val add : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [add ~out a b] computes [out.{i} <- a.{i} + b.{i}]. *)

  val sub : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [sub ~out a b] computes [out.{i} <- a.{i} - b.{i}]. *)

  val mul : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [mul ~out a b] computes [out.{i} <- a.{i} * b.{i}]. *)

  val div : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [div ~out a b] computes [out.{i} <- a.{i} / b.{i}].

      Integer dtypes use truncation toward zero (C division). Floating-point
      dtypes use IEEE 754 division. *)

  val mod_ : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [mod_ ~out a b] computes the remainder of [a / b].

      Integers use C's [%] operator (truncated division). Floats use [fmod].
      The sign of the result follows the dividend [a]. *)

  val pow : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [pow ~out base exponent] computes [out.{i} <- base.{i} ^ exponent.{i}]. *)

  val atan2 : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [atan2 ~out y x] computes [out.{i} <- atan2(y.{i}, x.{i})].

      Returns the angle in radians in [(-π, π\]], handling all quadrants. *)

  (** {2 Comparison}

      Comparison operations produce boolean tensors.

      {b Frontend guarantees:} [out] is a [(bool, bool_elt)] tensor with the
      same shape as [a] and [b]. *)

  val cmpeq :
    out:(bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [cmpeq ~out a b] computes [out.{i} <- (a.{i} = b.{i})]. *)

  val cmpne :
    out:(bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [cmpne ~out a b] computes [out.{i} <- (a.{i} <> b.{i})]. *)

  val cmplt :
    out:(bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [cmplt ~out a b] computes [out.{i} <- (a.{i} < b.{i})]. *)

  val cmple :
    out:(bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [cmple ~out a b] computes [out.{i} <- (a.{i} <= b.{i})]. *)

  (** {2 Min/Max} *)

  val max : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [max ~out a b] computes [out.{i} <- max(a.{i}, b.{i})]. *)

  val min : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [min ~out a b] computes [out.{i} <- min(a.{i}, b.{i})]. *)

  (** {2 Bitwise}

      Operate on the binary representation of integer and boolean dtypes. For
      booleans, these are equivalent to logical AND/OR/XOR. *)

  val xor : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [xor ~out a b] computes bitwise XOR. *)

  val or_ : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [or_ ~out a b] computes bitwise OR. *)

  val and_ : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [and_ ~out a b] computes bitwise AND. *)

  (** {1 Element-wise Unary Operations}

      {b Frontend guarantees:} [out] and [x] have the same shape and dtype.
      [out] is C-contiguous.

      {b Backend must:} write exactly [numel] elements to [out], respecting
      the strides of [x].

      {2 Arithmetic} *)

  val neg : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [neg ~out x] computes [out.{i} <- -x.{i}]. *)

  val recip : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [recip ~out x] computes [out.{i} <- 1 / x.{i}]. *)

  val abs : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [abs ~out x] computes [out.{i} <- |x.{i}|]. *)

  val sqrt : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [sqrt ~out x] computes [out.{i} <- √x.{i}]. *)

  val sign : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [sign ~out x] computes the sign function: [-1] for negative, [0] for
      zero, [1] for positive. Returns NaN for floating-point NaN inputs. *)

  (** {2 Exponential and Logarithm} *)

  val exp : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [exp ~out x] computes [out.{i} <- eˣ⁽ⁱ⁾]. *)

  val log : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [log ~out x] computes [out.{i} <- ln(x.{i})]. *)

  (** {2 Trigonometric}

      All inputs are in radians. *)

  val sin : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [sin ~out x] computes [out.{i} <- sin(x.{i})]. *)

  val cos : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [cos ~out x] computes [out.{i} <- cos(x.{i})]. *)

  val tan : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [tan ~out x] computes [out.{i} <- tan(x.{i})]. *)

  val asin : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [asin ~out x] computes [out.{i} <- arcsin(x.{i})].

      Returns values in [[-π/2, π/2]]. *)

  val acos : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [acos ~out x] computes [out.{i} <- arccos(x.{i})].

      Returns values in [[0, π]]. *)

  val atan : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [atan ~out x] computes [out.{i} <- arctan(x.{i})].

      Returns values in [[-π/2, π/2]]. *)

  (** {2 Hyperbolic} *)

  val sinh : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [sinh ~out x] computes [out.{i} <- sinh(x.{i})]. *)

  val cosh : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [cosh ~out x] computes [out.{i} <- cosh(x.{i})]. *)

  val tanh : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [tanh ~out x] computes [out.{i} <- tanh(x.{i})]. *)

  (** {2 Rounding}

      For integer dtypes, all rounding operations are the identity. *)

  val trunc : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [trunc ~out x] rounds toward zero. *)

  val ceil : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [ceil ~out x] rounds toward positive infinity. *)

  val floor : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [floor ~out x] rounds toward negative infinity. *)

  val round : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [round ~out x] rounds to nearest integer, half away from zero (C's
      [round]). *)

  (** {2 Special Functions} *)

  val erf : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [erf ~out x] computes the error function
      [erf(x) = 2/√π ∫₀ˣ e^(-t²) dt]. *)

  (** {1 Ternary Operations} *)

  val where :
    out:('a, 'b) t ->
    (bool, Dtype.bool_elt) t ->
    ('a, 'b) t ->
    ('a, 'b) t ->
    unit
  (** [where ~out cond if_true if_false] selects elements: [if_true.{i}] where
      [cond.{i}] is true, [if_false.{i}] otherwise.

      {b Frontend guarantees:} all four tensors have identical shapes. [cond]
      is boolean. [out], [if_true], [if_false] share the same dtype. *)

  (** {1 Reduction Operations}

      Reductions aggregate values along one or more axes.

      {b Frontend guarantees:} [axes] contains valid, non-negative,
      deduplicated axis indices. [out] is pre-allocated with the correct shape:
      reduced axes are either removed or kept as size-1 dimensions depending on
      [keepdims]. *)

  val reduce_sum :
    out:('a, 'b) t -> axes:int array -> keepdims:bool -> ('a, 'b) t -> unit
  (** [reduce_sum ~out ~axes ~keepdims x] sums elements along [axes]. *)

  val reduce_prod :
    out:('a, 'b) t -> axes:int array -> keepdims:bool -> ('a, 'b) t -> unit
  (** [reduce_prod ~out ~axes ~keepdims x] multiplies elements along [axes]. *)

  val reduce_max :
    out:('a, 'b) t -> axes:int array -> keepdims:bool -> ('a, 'b) t -> unit
  (** [reduce_max ~out ~axes ~keepdims x] finds maximum along [axes]. *)

  val reduce_min :
    out:('a, 'b) t -> axes:int array -> keepdims:bool -> ('a, 'b) t -> unit
  (** [reduce_min ~out ~axes ~keepdims x] finds minimum along [axes]. *)

  val argmax :
    out:(int32, Dtype.int32_elt) t ->
    axis:int ->
    keepdims:bool ->
    ('a, 'b) t ->
    unit
  (** [argmax ~out ~axis ~keepdims x] writes int32 indices of maximum values
      along [axis] to [out]. For ties, returns the first occurrence.

      {b Frontend guarantees:} [axis] is valid and non-negative. [out] has the
      correct reduced shape with int32 dtype. *)

  val argmin :
    out:(int32, Dtype.int32_elt) t ->
    axis:int ->
    keepdims:bool ->
    ('a, 'b) t ->
    unit
  (** [argmin ~out ~axis ~keepdims x] writes int32 indices of minimum values
      along [axis] to [out]. For ties, returns the first occurrence.

      {b Frontend guarantees:} [axis] is valid and non-negative. [out] has the
      correct reduced shape with int32 dtype. *)

  val associative_scan :
    out:('a, 'b) t ->
    axis:int ->
    op:[ `Sum | `Prod | `Max | `Min ] ->
    ('a, 'b) t ->
    unit
  (** [associative_scan ~out ~axis ~op x] computes an inclusive prefix scan
      along [axis]. [`Sum] for cumulative sum, [`Prod] for cumulative product,
      [`Max]/[`Min] for running max/min.

      {b Frontend guarantees:} [axis] is valid and non-negative. [out] has the
      same shape as [x]. *)

  (** {1 Sort Operations}

      {b Frontend guarantees:} [axis] is valid and non-negative. [out] is
      pre-allocated with the correct shape and dtype. *)

  val sort : out:('a, 'b) t -> axis:int -> descending:bool -> ('a, 'b) t -> unit
  (** [sort ~out ~axis ~descending x] sorts elements along [axis]. NaN values
      are placed at the end regardless of sort direction.

      {b Frontend guarantees:} [out] has the same shape and dtype as [x]. *)

  val argsort :
    out:(int32, Dtype.int32_elt) t ->
    axis:int ->
    descending:bool ->
    ('a, 'b) t ->
    unit
  (** [argsort ~out ~axis ~descending x] writes int32 indices that would sort
      elements along [axis] to [out].

      {b Frontend guarantees:} [out] has the same shape as [x] with int32
      dtype. *)

  (** {1 Movement Operations}

      Movement operations manipulate view metadata (shape, strides, offset)
      without copying data when possible. They return new tensor handles
      sharing the underlying buffer.

      {b Frontend guarantees:} all parameters are validated (axes in range,
      shapes compatible, bounds within limits).

      {b Backend must:} return a tensor with the correct view metadata. May
      share the underlying buffer (zero-copy) or allocate if necessary. *)

  val expand : ('a, 'b) t -> Symbolic_shape.t -> ('a, 'b) t
  (** [expand t shape] broadcasts dimensions of size 1 to match [shape] by
      setting their stride to 0. Non-singleton dimensions must already match.
      Zero-copy. *)

  val reshape : ('a, 'b) t -> Symbolic_shape.t -> ('a, 'b) t
  (** [reshape t shape] changes the logical shape, preserving element count.

      Zero-copy when [t] is C-contiguous or the reshape is compatible with the
      current strides. May copy if [t] is non-contiguous. *)

  val permute : ('a, 'b) t -> int array -> ('a, 'b) t
  (** [permute t axes] reorders dimensions according to [axes], which must be
      a permutation of [\[0, ..., ndim-1\]]. Zero-copy. *)

  val shrink : ('a, 'b) t -> (int * int) array -> ('a, 'b) t
  (** [shrink t ranges] extracts a contiguous slice. [ranges.(i)] is
      [(start, stop)] with exclusive [stop]. Zero-copy (adjusts offset and
      shape). *)

  val flip : ('a, 'b) t -> bool array -> ('a, 'b) t
  (** [flip t axes] reverses dimensions where [axes.(i) = true] by negating
      strides. Zero-copy. *)

  val pad : ('a, 'b) t -> (int * int) array -> 'a -> ('a, 'b) t
  (** [pad t padding fill_value] extends [t] with [fill_value]. [padding.(i)]
      is [(before, after)] for dimension [i].

      {b Backend must:} allocate a new buffer and copy data. *)

  val cat : out:('a, 'b) t -> ('a, 'b) t list -> axis:int -> unit
  (** [cat ~out tensors ~axis] concatenates [tensors] along [axis] into [out].

      {b Frontend guarantees:} all tensors have the same shape except along
      [axis]. [axis] is valid. The list is non-empty. [out] is pre-allocated
      with the correct concatenated shape. *)

  (** {1 Type Conversion and Memory} *)

  val cast : out:('c, 'd) t -> ('a, 'b) t -> unit
  (** [cast ~out x] converts elements of [x] to the dtype of [out].

      Float-to-int truncates toward zero. Int-to-float may lose precision for
      large values.

      {b Frontend guarantees:} [out] is pre-allocated with the correct shape
      and target dtype. *)

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

      {b Backend must:} write [src]'s data into [dst]'s buffer, respecting
      both tensors' strides. *)

  (** {1 Random Number Generation} *)

  val threefry :
    out:(int32, Dtype.int32_elt) t ->
    (int32, Dtype.int32_elt) t ->
    (int32, Dtype.int32_elt) t ->
    unit
  (** [threefry ~out key counter] applies the Threefry-2x32 hash function.

      {b Frontend guarantees:} [key] and [counter] are int32 tensors with
      compatible shapes. [out] is pre-allocated with the same shape as
      [counter]. *)

  (** {1 Indexed Access Operations} *)

  val gather :
    out:('a, 'b) t ->
    ('a, 'b) t ->
    (int32, Dtype.int32_elt) t ->
    axis:int ->
    unit
  (** [gather ~out data indices ~axis] selects elements from [data] along
      [axis] using [indices] and writes them to [out].

      {b Frontend guarantees:} [rank data = rank indices]. [axis] is valid.
      Index values are in range for [data]'s size along [axis]. [out] has the
      same shape as [indices] and the same dtype as [data]. *)

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

      Used to implement convolution as [unfold + matmul + fold]. *)

  val unfold :
    ?out:('a, 'b) t ->
    ('a, 'b) t ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) t
  (** [unfold ?out t ~kernel_size ~stride ~dilation ~padding] extracts sliding
      local blocks (im2col).

      Input shape [(N, C, ...spatial)] produces [(N, C * prod(kernel_size), L)]
      where [L] is the number of windows. Works for any number of spatial
      dimensions.

      {b Frontend guarantees:} all array parameters have length equal to the
      number of spatial dimensions. Values are positive.

      {b Backend must:} write results to [out] if provided, otherwise
      allocate. *)

  val fold :
    ?out:('a, 'b) t ->
    ('a, 'b) t ->
    output_size:int array ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) t
  (** [fold ?out t ~output_size ~kernel_size ~stride ~dilation ~padding]
      combines sliding local blocks (col2im). Inverse of {!unfold}. Overlapping
      values are summed.

      Input shape [(N, C * prod(kernel_size), L)] produces
      [(N, C, ...output_size)].

      {b Frontend guarantees:} parameters are consistent with a valid unfold
      configuration.

      {b Backend must:} write results to [out] if provided, otherwise
      allocate. *)

  (** {1 Matrix Operations} *)

  val matmul : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [matmul ~out a b] computes matrix multiplication [a × b].

      For 2D inputs: standard matrix multiply. For higher dimensions: batched
      multiply on the last two dimensions, with broadcasting via strides.

      {b Frontend guarantees:} [a]'s last dim equals [b]'s second-to-last dim.
      [out] is C-contiguous with the correct output shape.

      {b Backend must:} write the result to [out]. May use BLAS for
      performance. [a] and [b] may be non-contiguous. *)

  (** {1 Fourier Transforms}

      {b Frontend guarantees:} [axes] contains valid, non-negative axis
      indices. Input tensors have compatible complex or real dtypes. *)

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
  (** [qr ~reduced t] returns [(Q, R)] where [Q] is orthogonal and [R] is
      upper triangular. [reduced = true] returns economy-size factorization. *)

  val svd :
    full_matrices:bool ->
    ('a, 'b) t ->
    ('a, 'b) t * (float, Dtype.float64_elt) t * ('a, 'b) t
  (** [svd ~full_matrices t] returns [(U, S, Vᴴ)]. [S] is a 1D float64
      vector of singular values in descending order. [full_matrices = false]
      returns thin SVD. *)

  val eig :
    vectors:bool ->
    ('a, 'b) t ->
    (Complex.t, Dtype.complex64_elt) t
    * (Complex.t, Dtype.complex64_elt) t option
  (** [eig ~vectors t] computes eigenvalues (and optionally eigenvectors) of
      a square matrix. Returns complex64 results. *)

  val eigh :
    vectors:bool ->
    ('a, 'b) t ->
    (float, Dtype.float64_elt) t * ('a, 'b) t option
  (** [eigh ~vectors t] computes eigenvalues (and optionally eigenvectors) of
      a symmetric/Hermitian matrix. Eigenvalues are float64. *)

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
