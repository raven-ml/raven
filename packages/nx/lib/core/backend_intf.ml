(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

exception
  Linalg_error of {
    op : string;
    kind : [ `Not_positive_definite | `Singular | `No_convergence ];
  }
(** Raised by a linear-algebra operation when the numeric computation fails
    (as opposed to a precondition violation such as a non-square or wrong-dtype
    input, which raises [Invalid_argument] or [Failure]).

    [op] names the operation that failed (e.g. ["cholesky"]). [kind] classifies
    the failure:
    - [`Not_positive_definite]: a factorization requiring a positive-definite
      matrix was given one that is not (Cholesky).
    - [`Singular]: a solve was given a singular coefficient matrix.
    - [`No_convergence]: an iterative decomposition did not converge (QR, SVD,
      eigen routines). *)

let () =
  Printexc.register_printer (function
    | Linalg_error { op; kind } ->
        let detail =
          match kind with
          | `Not_positive_definite -> "matrix is not positive-definite"
          | `Singular -> "matrix is singular"
          | `No_convergence -> "algorithm failed to converge"
        in
        Some (Printf.sprintf "Nx.Linalg_error(%s): %s" op detail)
    | _ -> None)

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
      without copying data when possible.

    {1 Extended backend operations}

    The mandatory contract is exactly the operations declared below. The effect
    layer ([nx.effect]) implements all of them and adds an extended tier that
    Rune relies on but a conforming backend need not provide: [const_scalar]
    (materialize a scalar without a host round-trip), [to_device], [psum], and
    the [Deferred] tensor handle (a device-resident result forced to host on
    first data access). Those live outside this module type. *)
module type S = sig
  (** {1 Types} *)

  type ('a, 'b) t
  (** ['a] is the OCaml element type (e.g., [float], [int32]). ['b] is a phantom
      type that tags the dtype for type safety. *)

  type context
  (** Backend execution context.

      Carries backend-specific state such as memory pools, device handles,
      command queues, or computation graphs.

      Construction is absent from this module type: [S] describes operations
      over an existing context, and the frontend never builds one. Engines that
      implement the [nx.backend] virtual library additionally provide
      [create_context : unit -> context] (see [backend/nx_backend.mli], and the
      open question its TODO records). *)

  (** {1 Tensor Properties} *)

  val view : ('a, 'b) t -> View.t
  (** [view t] returns the strided view metadata describing [t]'s logical layout
      (shape, strides, offset) over its underlying buffer. *)

  val dtype : ('a, 'b) t -> ('a, 'b) Dtype.t
  (** [dtype t] returns the element type of [t]. *)

  val context : ('a, 'b) t -> context
  (** [context t] returns the execution context that owns [t]. *)

  val to_host : ('a, 'b) t -> ('a, 'b) Nx_buffer.t
  (** [to_host t] returns [t]'s underlying storage buffer, shared with [t]:
      mutations through the buffer are visible through [t] and vice versa.

      The buffer is {e not} necessarily contiguous nor sized to the logical
      element count. Interpret it through {!view} (offset and strides): for a
      strided view it may exceed the tensor's logical extent and be laid out
      non-contiguously. CPU backends return the storage directly (zero-copy);
      device backends copy it out to host memory. *)

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

  val fdiv : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [fdiv a b] is the element-wise IEEE 754 quotient of [a] and [b].

      {b Frontend guarantees:} [a] and [b] are float or complex dtypes. The
      frontend selects between {!fdiv} and {!idiv} by dtype; the backend never
      inspects the dtype domain here. *)

  val idiv : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** [idiv a b] is the element-wise integer quotient of [a] and [b], truncated
      toward zero (C division).

      {b Frontend guarantees:} [a] and [b] are integer dtypes. *)

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

  val reduce :
    op:[ `Sum | `Prod | `Max | `Min ] ->
    axes:int array ->
    ('a, 'b) t ->
    ('a, 'b) t
  (** [reduce ~op ~axes x] aggregates elements of [x] along [axes]: [`Sum] adds,
      [`Prod] multiplies, [`Max]/[`Min] take the running extremum.

      The result always drops the reduced axes (the [keepdims:false] shape). The
      frontend reinserts size-1 axes when the caller requests kept dimensions,
      so the backend never handles [keepdims]. *)

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

      Two kinds live here. {e Pure-view} movement — {!expand}, {!reshape},
      {!permute}, {!shrink}, {!flip} — only rewrites view metadata (shape,
      strides, offset) and returns a handle sharing the input's buffer, with no
      copy ({!reshape} never copies either: it raises when the existing strides
      cannot express the new shape). {e Materializing assembly} — {!pad} and
      {!cat} — cannot be expressed as a view and must allocate a fresh buffer and
      copy: {!pad} widens each dimension around a fill value, and {!cat} is the
      contract's only variable-arity operation, joining a list of tensors along
      an axis.

      {b Frontend guarantees:} all parameters are validated (axes in range,
      shapes compatible, bounds within limits).

      {b Backend must:} return a tensor with the correct view metadata, sharing
      the underlying buffer for pure-view operations and allocating for
      assembly. *)

  val expand : ('a, 'b) t -> int array -> ('a, 'b) t
  (** [expand t shape] broadcasts dimensions of size 1 to match [shape] by
      setting their stride to 0. Non-singleton dimensions must already match.
      Zero-copy. *)

  val reshape : ('a, 'b) t -> int array -> ('a, 'b) t
  (** [reshape t shape] changes the logical shape, preserving element count.

      Always zero-copy: the result is a view over [t]'s buffer. Raises
      [Invalid_argument] when the current strides cannot express [shape] (some
      non-contiguous layouts); the caller materializes with {!contiguous}
      first. *)

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
  (** [threefry key counter] applies the Threefry-2x32 counter-based hash.

      This is normative, not merely illustrative: the algorithm is Threefry-2x32
      run for 20 rounds, with the standard rotation constants and key schedule of
      that construction. A given [(key, counter)] pair must therefore produce
      bit-identical output in every backend and under every lowering — eager and
      jit results are held equal by rune's [test_rng.ml].

      {b Frontend guarantees:} [key] and [counter] are int32 tensors with
      compatible shapes. *)

  (** {1 Indexed Access Operations}

      Index tensors are uniformly [int32] across the whole contract: {!argmax},
      {!argmin}, and {!argsort} produce [int32], and {!gather}/{!scatter} consume
      it. Axes longer than 2{^31} - 1 (the [int32] maximum) are therefore
      unsupported, and the limit is not checked. *)

  val gather :
    ('a, 'b) t -> (int32, Dtype.int32_elt) t -> axis:int -> ('a, 'b) t
  (** [gather data indices ~axis] selects elements from [data] along [axis]
      using [indices].

      {b Frontend guarantees:} [rank data = rank indices]. [axis] is valid.
      Index values are in range for [data]'s size along [axis]. *)

  val scatter :
    mode:[ `Set | `Add ] ->
    unique_indices:bool ->
    ('a, 'b) t ->
    indices:(int32, Dtype.int32_elt) t ->
    updates:('a, 'b) t ->
    axis:int ->
    ('a, 'b) t
  (** [scatter ~mode ~unique_indices template ~indices ~updates ~axis] places
      [updates] into a tensor shaped like [template] along [axis].

      [`Set] uses the last update for duplicate indices; [`Add] accumulates
      every update into the template's value. [unique_indices = true] hints that
      indices are unique.

      Backend-contract operations take no optional arguments: user-facing
      defaults (here [`Set] and [false]) live on the frontend, which passes both
      labels explicitly.

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

  val fft : (Complex.t, 'b) t -> axes:int array -> (Complex.t, 'b) t
  (** [fft t ~axes] computes the forward DFT along [axes]. *)

  val ifft : (Complex.t, 'b) t -> axes:int array -> (Complex.t, 'b) t
  (** [ifft t ~axes] computes the inverse DFT along [axes]. *)

  val rfft :
    (float, 'a) t ->
    dtype:(Complex.t, 'b) Dtype.t ->
    axes:int array ->
    (Complex.t, 'b) t
  (** [rfft t ~dtype ~axes] computes the real-input DFT along [axes].

      Exploits conjugate symmetry to return only the non-redundant half of the
      spectrum along the last transformed axis. *)

  val irfft :
    ?s:int array ->
    (Complex.t, 'a) t ->
    dtype:(float, 'b) Dtype.t ->
    axes:int array ->
    (float, 'b) t
  (** [irfft ?s t ~dtype ~axes] computes the inverse real-input DFT along
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
      to LAPACK.

      Numeric failures (a matrix that is not positive-definite, a decomposition
      that fails to converge) raise {!Linalg_error}. Precondition violations
      that slip past the frontend (unsupported dtype, degenerate shape) keep
      their [Invalid_argument] or [Failure] form. *)

  val cholesky : upper:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [cholesky ~upper t] computes the Cholesky factorization of a
      positive-definite matrix. Returns [L] (lower) or [U] (upper) such that
      [A = L·Lᵀ] or [A = Uᵀ·U].

      @raise Linalg_error
        with kind [`Not_positive_definite] if [t] is not positive-definite. *)

  val qr : reduced:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
  (** [qr ~reduced t] returns [(Q, R)] where [Q] is orthogonal and [R] is upper
      triangular. [reduced = true] returns economy-size factorization.

      @raise Linalg_error
        with kind [`No_convergence] if the factorization does not converge. *)

  val svd :
    full_matrices:bool ->
    ('a, 'b) t ->
    ('a, 'b) t * (float, Dtype.float64_elt) t * ('a, 'b) t
  (** [svd ~full_matrices t] returns [(U, S, Vᴴ)]. [S] is a 1D float64 vector of
      singular values in descending order. [full_matrices = false] returns thin
      SVD.

      May raise {!Linalg_error} with kind [`No_convergence] if the underlying
      routine does not converge. *)

  val eigvals : ('a, 'b) t -> (Complex.t, Dtype.complex64_elt) t
  (** [eigvals t] computes the eigenvalues of a general square matrix. Returns
      complex64 results.

      This is the values-only variant: it drives the cheaper LAPACK path that
      does not accumulate eigenvectors. Use {!eig} when the vectors are needed.

      May raise {!Linalg_error} with kind [`No_convergence] if the eigenvalue
      iteration does not converge. *)

  val eig :
    ('a, 'b) t ->
    (Complex.t, Dtype.complex64_elt) t * (Complex.t, Dtype.complex64_elt) t
  (** [eig t] computes the eigenvalues and eigenvectors of a general square
      matrix, returned as [(values, vectors)]. Both are complex64.

      May raise {!Linalg_error} with kind [`No_convergence] if the eigenvalue
      iteration does not converge. *)

  val eigvalsh : ('a, 'b) t -> (float, Dtype.float64_elt) t
  (** [eigvalsh t] computes the eigenvalues of a symmetric/Hermitian matrix.
      Eigenvalues are float64.

      This is the values-only variant: it drives the cheaper LAPACK path that
      does not accumulate eigenvectors. Use {!eigh} when the vectors are needed.

      May raise {!Linalg_error} with kind [`No_convergence] if the eigenvalue
      iteration does not converge. *)

  val eigh : ('a, 'b) t -> (float, Dtype.float64_elt) t * ('a, 'b) t
  (** [eigh t] computes the eigenvalues and eigenvectors of a symmetric/Hermitian
      matrix, returned as [(values, vectors)]. Eigenvalues are float64;
      eigenvectors carry the input dtype.

      May raise {!Linalg_error} with kind [`No_convergence] if the eigenvalue
      iteration does not converge. *)

  val triangular_solve :
    upper:bool ->
    transpose:bool ->
    unit_diag:bool ->
    ('a, 'b) t ->
    ('a, 'b) t ->
    ('a, 'b) t
  (** [triangular_solve ~upper ~transpose ~unit_diag a b] solves [A·x = b] or
      [Aᴴ·x = b] where [A] is triangular.

      [upper]: [A] is upper triangular. [transpose]: solve [Aᴴ·x = b] — the
      conjugate transpose for complex dtypes, the plain transpose for real
      ones. [unit_diag]: assume diagonal is all ones.

      May raise {!Linalg_error} with kind [`Singular] if [A] is singular (a zero
      on the diagonal when [unit_diag] is false). *)
end
