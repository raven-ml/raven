(* Primitive tensor operations that every Nx backend must implement. *)

(** Backend interface.

    The [`op_*`] functions mirror tinygrad's UOps. A backend can execute them
    eagerly, raise effects for a JIT, build a computation graph, etc.

    The frontend handles broadcasting and shape validation, so each operation
    simply produces a fresh tensor. *)
module type S = sig
  type ('a, 'b) t
  (** Opaque tensor handle.

      ['a] is the OCaml element type; ['b] tags the dtype. *)

  type context
  (** Backend execution context. Carries any state required by the
      implementation (memory pools, command queues, ...). *)

  (* lenses *)

  val view : ('a, 'b) t -> Lazy_view.t
  (** Return the view tracker for [t]. *)

  val dtype : ('a, 'b) t -> ('a, 'b) Dtype.t
  (** Element type of [t]. *)

  val context : ('a, 'b) t -> context
  (** Execution context of [t]. *)

  val data : ('a, 'b) t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  (** Return the raw buffer of [t]. *)

  (* ops: mirrors tinygrad UOps *)

  val op_buffer :
    context -> ('a, 'b) Dtype.t -> int (* size_in_elements *) -> ('a, 'b) t
  (** Allocate a buffer of [size_in_elements] elements of [dtype]. *)

  val op_const_scalar : context -> 'a -> ('a, 'b) Dtype.t -> ('a, 'b) t
  (** Tensor containing a single scalar [value]. *)

  val op_const_array :
    context -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> ('a, 'b) t
  (** Tensor containing the elements of [array]. The array must be contiguous.
  *)

  (* Element-wise Binary Ops *)

  (* These ops assume inputs have been broadcast to the same shape and cast to
     the same compatible dtype by the frontend. The output tensor will also have
     this common dtype. *)

  val op_add : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Element-wise addition. *)

  val op_mul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Element-wise multiplication. *)

  val op_idiv : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Integer division, truncating. *)

  val op_fdiv : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Floating-point division. *)

  val op_max : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Element-wise maximum. *)

  val op_mod : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Integer modulus. *)

  val op_pow : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Raise [base] to [exponent]. *)

  val op_cmplt : ('a, 'b) t -> ('a, 'b) t -> (int, Dtype.uint8_elt) t
  (** Compare [<]. Returns 0 or 1 as uint8. *)

  val op_cmpne : ('a, 'b) t -> ('a, 'b) t -> (int, Dtype.uint8_elt) t
  (** Compare [<>]. Returns 0 or 1 as uint8. *)

  val op_xor : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Bitwise XOR. *)

  val op_or : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Bitwise OR. *)

  val op_and : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Bitwise AND. *)

  (* Element-wise Unary Ops *)

  val op_neg : ('a, 'b) t -> ('a, 'b) t
  (** Negation (logical not for bools). *)

  val op_log2 : ('a, 'b) t -> ('a, 'b) t
  (** Base-2 logarithm. *)

  val op_exp2 : ('a, 'b) t -> ('a, 'b) t
  (** Exponential base 2. *)

  val op_sin : ('a, 'b) t -> ('a, 'b) t
  (** Sine. *)

  val op_sqrt : ('a, 'b) t -> ('a, 'b) t
  (** Square root. *)

  val op_recip : ('a, 'b) t -> ('a, 'b) t
  (** Reciprocal. *)

  (* Ternary Op *)

  val op_where :
    (int, Dtype.uint8_elt) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Select from [if_true] or [if_false] based on a boolean tensor. *)

  (* Reduction Ops *)

  val op_reduce_sum :
    axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** Sum over [axes]. Keeps reduced dimensions if [keepdims] is true. *)

  val op_reduce_max :
    axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** Maximum over [axes]. Keeps reduced dimensions if [keepdims] is true. *)

  val op_reduce_prod :
    axes:int array -> keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
  (** Product over [axes]. Keeps reduced dimensions if [keepdims] is true. *)

  (* Movement Ops - manipulate view metadata *)

  val op_expand : ('a, 'b) t -> Symbolic_shape.t -> ('a, 'b) t
  (** Broadcast dimensions of size 1 to a new shape. *)

  val op_reshape : ('a, 'b) t -> Symbolic_shape.t -> ('a, 'b) t
  (** Change the logical shape without moving data. *)

  val op_permute : ('a, 'b) t -> int array -> ('a, 'b) t
  (** Reorder dimensions according to [axes]. *)

  val op_shrink : ('a, 'b) t -> (int * int) array -> ('a, 'b) t
  (** Slice according to the given start/stop pairs. *)

  val op_flip : ('a, 'b) t -> bool array -> ('a, 'b) t
  (** Flip dimensions where the boolean array is [true]. *)

  val op_pad : ('a, 'b) t -> (int * int) array -> 'a -> ('a, 'b) t
  (** Pad with [fill_value] using the given configuration. *)

  val op_cat : ('a, 'b) t list -> int -> ('a, 'b) t
  (** Concatenate tensors along [axis]. *)

  (* Other Ops *)

  val op_cast : ('a, 'b) t -> ('c, 'd) Dtype.t -> ('c, 'd) t
  (** Cast elements to [target_dtype]. *)

  val op_contiguous : ('a, 'b) t -> ('a, 'b) t
  (** Return a C-contiguous tensor. May copy. *)

  val op_copy : ('a, 'b) t -> ('a, 'b) t
  (** Duplicate [t]. Result has its own buffer. *)

  val op_assign : ('a, 'b) t -> ('a, 'b) t -> unit
  (** Store [src] into [dst] at the given logical indices. *)

  val op_threefry :
    (int32, Dtype.int32_elt) t ->
    (int32, Dtype.int32_elt) t ->
    (int32, Dtype.int32_elt) t
  (** Threefry random number generator. *)

  (* Element Access Ops *)

  (* These operations enable lazy, graph-based element access (get/set). This
     differs from tinygrad's eager realization for __getitem__/__setitem__. We
     opt for lazy ops to avoid premature realization and CPU<->device transfers.
     These are primarily for internal Nx slice/put_slice implementations and
     their direct backend exposure might be refined later. *)

  val op_gather :
    ('a, 'b) t (* data *) ->
    (int32, Dtype.int32_elt) t (* indices *) ->
    int (* axis *) ->
    ('a, 'b) t
  (** Gather elements from [data] along [axis] using [indices]. Output shape
      matches [indices]. Ranks of [data] and [indices] must match. Sizes of
      [indices] dims != [axis] must be <= [data] corresponding dims. *)

  val op_scatter :
    ?mode:[ `Set | `Add ] ->
    ?unique_indices:bool ->
    ('a, 'b) t (* data_template *) ->
    (int32, Dtype.int32_elt) t (* indices *) ->
    ('a, 'b) t (* updates *) ->
    int (* axis *) ->
    ('a, 'b) t
  (** Scatter [updates] into a new tensor shaped like [data_template] along
      [axis] using [indices]. Returns a new tensor.
      - [mode] specifies how to handle duplicate indices:
      - [`Set] (default): last update wins
      - [`Add]: accumulate updates at duplicate indices
      - [unique_indices]: hint that indices are unique (optimization) *)

  val op_unfold :
    ('a, 'b) t ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) t
  (** Unfold (im2col) operation. Extracts sliding local blocks from a batched
      input tensor. For an input of shape (N, C, *spatial_dims), produces output
      of shape (N, C * prod(kernel_size), L) where L is the number of blocks.
      Works for any number of spatial dimensions. *)

  val op_fold :
    ('a, 'b) t ->
    output_size:int array ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) t
  (** Fold (col2im) operation. Combines an array of sliding local blocks into a
      tensor. For an input of shape (N, C * prod(kernel_size), L), produces
      output of shape (N, C, *output_size). Inverse of unfold. Overlapping
      values are summed. Works for any number of spatial dimensions. *)

  val op_matmul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (** Matrix multiplication. For 2D tensors, computes standard matrix
      multiplication. For higher dimensions, performs batched matrix
      multiplication on the last two dimensions, broadcasting batch dimensions
      as needed. The last dimension of the first tensor must match the
      second-to-last dimension of the second tensor. *)

  (* Fourier transforms *)

  val op_fft :
    (Complex.t, 'b) t ->
    axes:int array ->
    s:int array option ->
    (Complex.t, 'b) t
  (** Compute the discrete Fourier transform (DFT) of the input tensor. *)

  val op_ifft :
    (Complex.t, 'b) t ->
    axes:int array ->
    s:int array option ->
    (Complex.t, 'b) t
  (** Compute the inverse discrete Fourier transform (IDFT) of the input tensor.
  *)

  val op_rfft :
    (float, 'a) t ->
    dtype:(Complex.t, 'b) Dtype.t ->
    axes:int array ->
    s:int array option ->
    (Complex.t, 'b) t
  (** Compute the real-valued discrete Fourier transform (RDFT) of the input
      tensor. *)

  val op_irfft :
    (Complex.t, 'a) t ->
    dtype:(float, 'b) Dtype.t ->
    axes:int array ->
    s:int array option ->
    (float, 'b) t
  (** Compute the inverse real-valued discrete Fourier transform (IRDFT) of the
      input tensor. *)

  (* Linear algebra operations *)

  val op_cholesky : upper:bool -> ('a, 'b) t -> ('a, 'b) t
  (** Cholesky decomposition of a positive-definite matrix.
      - [upper]: If true, returns upper triangular factor; else lower (default).
      - Input: Square matrix A (batched).
      - Output: Triangular factor L or U such that A = L*L^T or A = U^T*U.
      - Raises if input is not positive-definite. *)

  val op_qr : reduced:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
  (** QR decomposition.
      - [reduced]: If true (default), returns economy/reduced QR; else full QR.
      - Input: m x n matrix A (batched).
      - Output: (Q, R) where A = Q*R, Q orthogonal, R upper triangular. *)

  val op_svd :
    full_matrices:bool ->
    ('a, 'b) t ->
    ('a, 'b) t * (float, Dtype.float64_elt) t * ('a, 'b) t
  (** Singular value decomposition.
      - [full_matrices]: If false (default), returns thin SVD; else full.
      - Input: m x n matrix A (batched).
      - Output: (U, S, V^H) where A = U*S*V^H.
      - S is 1D vector of singular values in descending order, always float64.
  *)

  val op_eig :
    vectors:bool ->
    ('a, 'b) t ->
    (Complex.t, Dtype.complex64_elt) t
    * (Complex.t, Dtype.complex64_elt) t option
  (** General eigenvalue decomposition.
      - [vectors]: If true (default), computes eigenvectors.
      - Input: Square matrix A (batched).
      - Output: (eigenvalues, optional eigenvectors) always as complex64. *)

  val op_eigh :
    vectors:bool ->
    ('a, 'b) t ->
    (float, Dtype.float64_elt) t * ('a, 'b) t option
  (** Symmetric/Hermitian eigenvalue decomposition.
      - [vectors]: If true (default), computes eigenvectors.
      - Input: Symmetric (real) or Hermitian (complex) matrix A (batched).
      - Output: (eigenvalues as float64, eigenvectors same type as input). *)

  val op_triangular_solve :
    upper:bool ->
    transpose:bool ->
    unit_diag:bool ->
    ('a, 'b) t ->
    ('a, 'b) t ->
    ('a, 'b) t
  (** Solve triangular system A*x = b or A^T*x = b.
      - [upper]: If true, A is upper triangular; else lower.
      - [transpose]: If true, solve A^T*x = b; else A*x = b.
      - [unit_diag]: If true, assume diagonal of A is all 1s.
      - Input: Triangular matrix A, right-hand side b (batched).
      - Output: Solution x. *)

  val op_as_strided :
    ('a, 'b) t ->
    Symbolic_shape.t ->
    int array ->
    int ->
    ('a, 'b) t
  (** Create a strided view of the input tensor with the given shape, strides
      (in elements), and offset (in elements). Backends that support arbitrary
      strided views (e.g., native with Bigarray) can implement this as
      zero-copy. Other backends may fall back to copying data if necessary.
      Raises if the view would access out-of-bounds memory. *)
end
