(** Backend interface that every Nx backend must implement.

    This module type defines the contract between Nx's frontend and its pluggable
    backends. Backends may execute operations eagerly (C backend), raise effects
    for JIT compilation (Rune), build computation graphs, or implement other
    execution strategies.

    The frontend handles broadcasting, shape validation, and dtype promotion before
    invoking backend operations, ensuring backends receive well-formed inputs.

    {1 Design Philosophy}

    Inspired by tinygrad's minimalism, but operating at an abstraction level closer
    to XLA for reasonable eager CPU performance. Rune's JIT pipeline deconstructs
    these operations into lower primitives, so this interface sits at a higher level
    than JIT operations.

    Binary and unary operations write to caller-provided output buffers for memory
    reuse. Movement operations manipulate view metadata without copying data when
    possible. *)
module type S = sig
  (** {1 Types} *)

  type ('a, 'b) t
  (** Opaque tensor handle.

      ['a] is the OCaml element type (e.g., [float], [int32]). ['b] is a phantom
      type that tags the dtype for type safety. *)

  type context
  (** Backend execution context.

      Carries backend-specific state such as memory pools, device handles, command
      queues, or computation graphs. *)

  (** {1 Tensor Properties} *)

  val view : ('a, 'b) t -> View.t
  (** [view t] returns the view metadata for [t].

      The view describes the logical shape, strides, and offset into the
      underlying buffer. *)

  val dtype : ('a, 'b) t -> ('a, 'b) Dtype.t
  (** [dtype t] returns the element type of [t]. *)

  val context : ('a, 'b) t -> context
  (** [context t] returns the execution context of [t]. *)

  val data : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  (** [data t] returns the raw buffer of [t].

      The buffer is a flat, contiguous C-layout Bigarray. Use {!view} to
      interpret the logical structure. *)

  (** {1 Buffer Allocation} *)

  val op_buffer : context -> ('a, 'b) Dtype.t -> int -> ('a, 'b) t
  (** [op_buffer context dtype size_in_elements] allocates an uninitialized buffer.

      Returns a tensor with [size_in_elements] elements of [dtype]. The buffer
      contents are undefined. *)

  val op_const_scalar : context -> 'a -> ('a, 'b) Dtype.t -> ('a, 'b) t
  (** [op_const_scalar context value dtype] creates a scalar tensor.

      Returns a tensor containing the single value [value] with type [dtype]. *)

  val op_const_array :
    context -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> ('a, 'b) t
  (** [op_const_array context array] creates a tensor from [array].

      The input [array] must be C-contiguous. The resulting tensor may share the
      buffer or copy it, depending on the backend. *)

  (** {1 Element-wise Binary Operations}

      All binary operations write results to a caller-provided [out] buffer. The
      frontend ensures inputs are broadcast to the same shape and cast to
      compatible dtypes before invocation.

      {2 Arithmetic Operations} *)

  val op_add : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_add ~out a b] computes [a + b] element-wise, writing to [out]. *)

  val op_sub : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_sub ~out a b] computes [a - b] element-wise, writing to [out]. *)

  val op_mul : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_mul ~out a b] computes [a * b] element-wise, writing to [out]. *)

  val op_idiv : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_idiv ~out a b] computes integer division [a / b] with truncation toward
      zero, writing to [out]. *)

  val op_fdiv : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_fdiv ~out a b] computes floating-point division [a / b], writing to
      [out]. *)

  val op_mod : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_mod ~out a b] computes modulus [a mod b], writing to [out].

      For integers, uses C's [%] operator (truncated division). For floats, uses
      [fmod]. In both cases, the sign of the result follows the sign of the
      dividend [a]. *)

  val op_pow : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_pow ~out base exp] computes [base ^ exp] element-wise, writing to
      [out]. *)

  (** {2 Comparison Operations} *)

  val op_cmpeq : out:(bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_cmpeq ~out a b] computes [a = b] element-wise, writing bool result to
      [out]. *)

  val op_cmpne : out:(bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_cmpne ~out a b] computes [a <> b] element-wise, writing bool result to
      [out]. *)

  val op_cmplt : out:(bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_cmplt ~out a b] computes [a < b] element-wise, writing bool result to
      [out]. *)

  val op_cmple : out:(bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_cmple ~out a b] computes [a <= b] element-wise, writing bool result to
      [out]. *)

  (** {2 Min/Max Operations} *)

  val op_max : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_max ~out a b] computes element-wise maximum, writing to [out]. *)

  val op_min : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_min ~out a b] computes element-wise minimum, writing to [out]. *)

  (** {2 Bitwise Operations} *)

  val op_xor : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_xor ~out a b] computes bitwise XOR, writing to [out]. *)

  val op_or : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_or ~out a b] computes bitwise OR, writing to [out]. *)

  val op_and : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_and ~out a b] computes bitwise AND, writing to [out]. *)

  (** {1 Element-wise Unary Operations}

      All unary operations write results to a caller-provided [out] buffer.

      {2 Arithmetic Operations} *)

  val op_neg : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_neg ~out x] computes negation [-x] element-wise, writing to [out].

      For boolean inputs, computes logical NOT. *)

  val op_recip : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_recip ~out x] computes reciprocal [1 / x] element-wise, writing to
      [out]. *)

  val op_abs : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_abs ~out x] computes absolute value element-wise, writing to [out]. *)

  val op_sqrt : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_sqrt ~out x] computes square root element-wise, writing to [out]. *)

  (** {2 Exponential and Logarithm Operations} *)

  val op_exp : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_exp ~out x] computes natural exponential [e ^ x] element-wise, writing
      to [out]. *)

  val op_log : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_log ~out x] computes natural logarithm [ln(x)] element-wise, writing to
      [out]. *)

  (** {2 Trigonometric Operations} *)

  val op_sin : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_sin ~out x] computes sine element-wise, writing to [out].

      Input is in radians. *)

  val op_cos : out:('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_cos ~out x] computes cosine element-wise, writing to [out].

      Input is in radians. *)

  (** {1 Ternary Operations} *)

  val op_where :
    out:('a, 'b) t ->
    (bool, Dtype.bool_elt) t ->
    ('a, 'b) t ->
    ('a, 'b) t ->
    unit
  (** [op_where ~out cond if_true if_false] selects elements conditionally,
      writing to [out].

      For each element position, selects from [if_true] where [cond] is true,
      otherwise from [if_false]. All inputs must have the same shape. *)

  (** {1 Reduction Operations}

      Reduction operations aggregate values along specified axes. When
      [keepdims] is true, reduced axes have size 1 in the output; when false,
      they are removed entirely. *)

  val op_reduce_sum :
    out:('a, 'b) t -> axes:int array -> keepdims:bool -> ('a, 'b) t -> unit
  (** [op_reduce_sum ~out ~axes ~keepdims x] sums elements over [axes], writing
      to [out]. *)

  val op_reduce_prod :
    out:('a, 'b) t -> axes:int array -> keepdims:bool -> ('a, 'b) t -> unit
  (** [op_reduce_prod ~out ~axes ~keepdims x] multiplies elements over [axes],
      writing to [out]. *)

  val op_reduce_max :
    out:('a, 'b) t -> axes:int array -> keepdims:bool -> ('a, 'b) t -> unit
  (** [op_reduce_max ~out ~axes ~keepdims x] finds maximum elements over [axes],
      writing to [out]. *)

  val op_reduce_min :
    out:('a, 'b) t -> axes:int array -> keepdims:bool -> ('a, 'b) t -> unit
  (** [op_reduce_min ~out ~axes ~keepdims x] finds minimum elements over [axes],
      writing to [out]. *)

  val op_associative_scan :
    axis:int -> op:[ `Sum | `Prod | `Max | `Min ] -> ('a, 'b) t -> ('a, 'b) t
  (** [op_associative_scan ~axis ~op x] computes an inclusive scan along [axis]
      using [op].

      Returns a new tensor with the same shape as [x]. For [op = `Sum], computes
      cumulative sum; for [op = `Prod], cumulative product; for [op = `Max] and
      [`Min], running maximum and minimum. *)

  (** {1 Movement Operations}

      Movement operations manipulate tensor view metadata (shape, strides, offset)
      without copying data when possible. They return new tensor handles with
      updated views over the same or new buffers. *)

  val op_expand : ('a, 'b) t -> Symbolic_shape.t -> ('a, 'b) t
  (** [op_expand t shape] broadcasts dimensions of size 1 to [shape].

      Returns a new view with expanded dimensions. The underlying buffer is
      shared. Dimensions of size 1 in [t] are repeated to match [shape]; other
      dimensions must already match. *)

  val op_reshape : ('a, 'b) t -> Symbolic_shape.t -> ('a, 'b) t
  (** [op_reshape t shape] changes the logical shape to [shape].

      Returns a new view with the specified shape. The total number of elements
      must remain constant. May require copying if [t] is not contiguous. *)

  val op_permute : ('a, 'b) t -> int array -> ('a, 'b) t
  (** [op_permute t axes] reorders dimensions according to [axes].

      Returns a new view with permuted dimensions. [axes] must be a permutation
      of [0, 1, ..., ndim-1]. The underlying buffer is shared. *)

  val op_shrink : ('a, 'b) t -> (int * int) array -> ('a, 'b) t
  (** [op_shrink t ranges] extracts a slice using [ranges].

      Returns a new view restricted to the specified ranges. [ranges.(i)] is
      [(start, stop)] for dimension [i], where [stop] is exclusive. The
      underlying buffer is shared. *)

  val op_flip : ('a, 'b) t -> bool array -> ('a, 'b) t
  (** [op_flip t axes] reverses dimensions where [axes] is [true].

      Returns a new view with flipped dimensions. [axes.(i) = true] reverses
      dimension [i]. The underlying buffer is shared. *)

  val op_pad : ('a, 'b) t -> (int * int) array -> 'a -> ('a, 'b) t
  (** [op_pad t padding fill_value] pads [t] with [fill_value].

      Returns a new tensor with padding applied. [padding.(i)] is [(before, after)]
      for dimension [i], specifying the number of elements to add. Requires
      allocating a new buffer. *)

  val op_cat : ('a, 'b) t list -> int -> ('a, 'b) t
  (** [op_cat tensors axis] concatenates [tensors] along [axis].

      Returns a new tensor containing all tensors joined along [axis]. All
      tensors must have the same shape except along [axis]. Requires allocating
      a new buffer. *)

  (** {1 Type Conversion and Memory Operations} *)

  val op_cast : ('a, 'b) t -> ('c, 'd) Dtype.t -> ('c, 'd) t
  (** [op_cast t target_dtype] converts elements of [t] to [target_dtype].

      Returns a new tensor with converted elements. The conversion follows
      standard casting rules (e.g., float to int truncates toward zero). *)

  val op_contiguous : ('a, 'b) t -> ('a, 'b) t
  (** [op_contiguous t] returns a C-contiguous version of [t].

      If [t] is already C-contiguous, may return [t] unchanged. Otherwise,
      allocates a new buffer and copies data. *)

  val op_copy : ('a, 'b) t -> ('a, 'b) t
  (** [op_copy t] duplicates [t].

      Returns a new tensor with its own buffer containing a copy of [t]'s data. *)

  val op_assign : ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_assign dst src] copies elements from [src] into [dst].

      Shapes must match. Mutates [dst] in place. *)

  val op_as_strided :
    ('a, 'b) t -> Symbolic_shape.t -> int array -> int -> ('a, 'b) t
  (** [op_as_strided t shape strides offset] creates a view with custom
      [strides] and [offset].

      Returns a new view over [t]'s buffer with the specified shape, strides
      (in elements), and offset (in elements). Backends supporting arbitrary
      strided views implement this as zero-copy. Other backends may copy data
      if necessary.

      @raise Invalid_argument if the view would access out-of-bounds memory. *)

  (** {1 Random Number Generation} *)

  val op_threefry :
    (int32, Dtype.int32_elt) t ->
    (int32, Dtype.int32_elt) t ->
    (int32, Dtype.int32_elt) t
  (** [op_threefry key counter] applies the Threefry-2x32 random number generator.

      Takes a [key] and [counter] tensor, both int32, and returns pseudo-random
      int32 values with the same shape as [counter]. Used as the foundation for
      Nx's random number generation. *)

  (** {1 Indexed Access Operations}

      These operations provide lazy, graph-based element access that avoids
      premature realization and CPU-device transfers. They are primarily used
      internally by Nx's slice and put_slice implementations. *)

  val op_gather :
    ('a, 'b) t -> (int32, Dtype.int32_elt) t -> int -> ('a, 'b) t
  (** [op_gather data indices axis] gathers elements from [data] along [axis]
      using [indices].

      Returns a new tensor with shape matching [indices]. The ranks of [data]
      and [indices] must be equal. Each index value in [indices] selects an
      element along [axis] from [data]. *)

  val op_scatter :
    ?mode:[ `Set | `Add ] ->
    ?unique_indices:bool ->
    ('a, 'b) t ->
    (int32, Dtype.int32_elt) t ->
    ('a, 'b) t ->
    int ->
    ('a, 'b) t
  (** [op_scatter ?mode ?unique_indices data_template indices updates axis]
      scatters [updates] into a tensor shaped like [data_template] along [axis]
      using [indices].

      Returns a new tensor with the same shape as [data_template]. The shapes of
      [indices] and [updates] must match. Each index value in [indices]
      specifies where to place the corresponding element from [updates] along
      [axis].

      The [mode] parameter controls duplicate index handling: [`Set] (default)
      uses the last update; [`Add] accumulates updates. Setting [unique_indices]
      to [true] hints that indices are unique, enabling optimizations. *)

  val op_unfold :
    ?out:('a, 'b) t ->
    ('a, 'b) t ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) t
  (** [op_unfold ?out t ~kernel_size ~stride ~dilation ~padding] extracts sliding
      local blocks (im2col operation).

      For input shape [(N, C, ...spatial_dims)], returns shape
      [(N, C * prod(kernel_size), L)] where [L] is the number of blocks. Works
      for any number of spatial dimensions. Used to implement convolution via
      matrix multiplication.

      @param out Optional pre-allocated output tensor. *)

  val op_fold :
    ?out:('a, 'b) t ->
    ('a, 'b) t ->
    output_size:int array ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) t
  (** [op_fold ?out t ~output_size ~kernel_size ~stride ~dilation ~padding] combines
      sliding local blocks into a tensor (col2im operation).

      For input shape [(N, C * prod(kernel_size), L)], returns shape
      [(N, C, ...output_size)]. Inverse of {!op_unfold}. Overlapping values are
      summed. Works for any number of spatial dimensions.

      @param out Optional pre-allocated output tensor. *)

  (** {1 Matrix Operations} *)

  val op_matmul : out:('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  (** [op_matmul ~out a b] computes matrix multiplication [a * b], writing to
      [out].

      For 2D tensors, computes standard matrix multiplication. For higher
      dimensions, performs batched matrix multiplication on the last two
      dimensions, using strides to handle broadcasting without copying data.

      The caller must:
      - Compute the correct output shape.
      - Allocate [out] with matching shape and dtype.
      - Ensure [out] is contiguous.

      Precondition: The last dimension of [a] must equal the second-to-last
      dimension of [b]. *)

  (** {1 Fourier Transforms} *)

  val op_fft :
    ?out:(Complex.t, 'b) t ->
    (Complex.t, 'b) t ->
    axes:int array ->
    (Complex.t, 'b) t
  (** [op_fft ?out t ~axes] computes the discrete Fourier transform (DFT) along
      [axes].

      Returns a complex tensor with the same shape as [t]. The transform is
      applied independently along each axis in [axes].

      @param out Optional pre-allocated output tensor. *)

  val op_ifft :
    ?out:(Complex.t, 'b) t ->
    (Complex.t, 'b) t ->
    axes:int array ->
    (Complex.t, 'b) t
  (** [op_ifft ?out t ~axes] computes the inverse discrete Fourier transform
      (IDFT) along [axes].

      Returns a complex tensor with the same shape as [t]. The inverse transform
      is applied independently along each axis in [axes].

      @param out Optional pre-allocated output tensor. *)

  val op_rfft :
    ?out:(Complex.t, 'b) t ->
    (float, 'a) t ->
    dtype:(Complex.t, 'b) Dtype.t ->
    axes:int array ->
    (Complex.t, 'b) t
  (** [op_rfft ?out t ~dtype ~axes] computes the real-valued discrete Fourier
      transform (RDFT) along [axes].

      Takes a real input and returns a complex output with [dtype]. Exploits
      conjugate symmetry to compute only the non-redundant half of the spectrum
      along the last transformed axis.

      @param out Optional pre-allocated output tensor. *)

  val op_irfft :
    ?out:(float, 'b) t ->
    (Complex.t, 'a) t ->
    dtype:(float, 'b) Dtype.t ->
    axes:int array ->
    s:int array option ->
    (float, 'b) t
  (** [op_irfft ?out t ~dtype ~axes ~s] computes the inverse real-valued discrete
      Fourier transform (IRDFT) along [axes].

      Takes a complex input (assumed conjugate-symmetric) and returns a real
      output with [dtype]. The parameter [s] specifies output sizes along
      transformed axes; if [None], sizes are inferred from [t].

      @param out Optional pre-allocated output tensor. *)

  (** {1 Linear Algebra Operations}

      Linear algebra operations support batching: the last two dimensions contain
      the matrices, and earlier dimensions are treated as batch dimensions. *)

  val op_cholesky : upper:bool -> ('a, 'b) t -> ('a, 'b) t
  (** [op_cholesky ~upper t] computes the Cholesky decomposition of [t].

      For a positive-definite matrix [A], returns triangular factor [L] or [U]
      such that [A = L * L^T] (when [upper = false]) or [A = U^T * U] (when
      [upper = true]).

      Precondition: [t] must contain square, positive-definite matrices.

      @raise Failure if [t] is not positive-definite. *)

  val op_qr : reduced:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
  (** [op_qr ~reduced t] computes the QR decomposition of [t].

      For an [m × n] matrix [A], returns [(Q, R)] where [A = Q * R], [Q] is
      orthogonal, and [R] is upper triangular. When [reduced = true], returns
      economy-size QR; when [reduced = false], returns full QR. *)

  val op_svd :
    full_matrices:bool ->
    ('a, 'b) t ->
    ('a, 'b) t * (float, Dtype.float64_elt) t * ('a, 'b) t
  (** [op_svd ~full_matrices t] computes the singular value decomposition of [t].

      For an [m × n] matrix [A], returns [(U, S, V^H)] where [A = U * S * V^H].
      [S] is a 1D vector of singular values in descending order, always float64.
      When [full_matrices = false], returns thin SVD; when [full_matrices = true],
      returns full SVD. *)

  val op_eig :
    vectors:bool ->
    ('a, 'b) t ->
    (Complex.t, Dtype.complex64_elt) t
    * (Complex.t, Dtype.complex64_elt) t option
  (** [op_eig ~vectors t] computes eigenvalues and optionally eigenvectors of [t].

      Returns [(eigenvalues, optional eigenvectors)] as complex64. When
      [vectors = true], computes eigenvectors; when [vectors = false], returns
      [None] for eigenvectors.

      Precondition: [t] must contain square matrices. *)

  val op_eigh :
    vectors:bool ->
    ('a, 'b) t ->
    (float, Dtype.float64_elt) t * ('a, 'b) t option
  (** [op_eigh ~vectors t] computes eigenvalues and optionally eigenvectors of
      symmetric/Hermitian [t].

      Returns [(eigenvalues, optional eigenvectors)]. Eigenvalues are float64;
      eigenvectors match the input dtype. When [vectors = true], computes
      eigenvectors; when [vectors = false], returns [None] for eigenvectors.

      Precondition: [t] must contain symmetric (for real) or Hermitian (for
      complex) matrices. *)

  val op_triangular_solve :
    upper:bool ->
    transpose:bool ->
    unit_diag:bool ->
    ('a, 'b) t ->
    ('a, 'b) t ->
    ('a, 'b) t
  (** [op_triangular_solve ~upper ~transpose ~unit_diag a b] solves the
      triangular system [A * x = b] or [A^T * x = b].

      Returns solution [x]. When [upper = true], [A] is upper triangular;
      otherwise lower. When [transpose = true], solves [A^T * x = b]; otherwise
      [A * x = b]. When [unit_diag = true], assumes the diagonal of [A] is all
      ones and does not access it.

      Precondition: [a] must contain triangular matrices compatible with [b]. *)
end
