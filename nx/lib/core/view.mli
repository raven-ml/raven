(** Strided views of tensor data.

    Views describe how to interpret a linear buffer as a multi-dimensional array
    through shape, strides, and offset. They enable zero-copy operations like
    transpose, slice, and reshape by manipulating metadata instead of copying
    data.

    {1 Key Concepts}

    {2 Strides and Layout}

    Strides determine memory layout. C-contiguous (row-major) layout stores the
    last dimension contiguously in memory. For shape [[|2; 3; 4|]], C-contiguous
    strides are [[|12; 4; 1|]].

    Zero strides enable broadcasting: a dimension with stride 0 repeats the same
    memory location. For example, expanding a scalar to shape [[|3; 3|]] uses
    strides [[|0; 0|]].

    Negative strides reverse iteration order without copying data. Flipping a
    dimension inverts its stride and adjusts the offset.

    {2 Symbolic Shapes}

    Views support symbolic dimensions for shape-polymorphic operations. When a
    view contains symbolic dimensions, some operations like {!shrink}, {!flip},
    and {!pad} require concrete values and will fail until dimensions are bound.
    Operations like {!expand}, and {!reshape} (for C-contiguous views) work with
    symbolic shapes.

    {2 Masks}

    Masks restrict valid regions per dimension using [(start, end)] pairs with
    half-open intervals \[start, end) (NumPy convention). Padding operations
    create masks to mark extended regions as invalid. A masked view cannot
    produce standard strides until the mask is removed (typically by
    materializing the tensor). *)

type t
(** A view encapsulating tensor layout information.

    Views are immutable. Operations return new views without modifying the
    original. *)

(** {2 Creation} *)

val create :
  ?offset:int ->
  ?strides:int array ->
  ?mask:(int * int) array ->
  Symbolic_shape.t ->
  t
(** [create ?offset ?strides ?mask shape] constructs a view.

    Creates a view describing how to interpret a buffer as a multi-dimensional
    array. The view contains metadata only; no buffer allocation occurs.

    @param offset Starting position in the buffer. Defaults to [0].

    @param strides
      Step size per dimension for computing linear indices. Defaults to
      C-contiguous strides (row-major layout). For symbolic shapes without
      explicit strides, unit placeholder strides are used (all strides set to 1)
      since true C-contiguous strides cannot be computed without concrete
      dimension values.

    @param mask
      Valid ranges per dimension as [(start, end)] pairs with exclusive ends
      (NumPy half-open interval convention: \[start, end)). A dimension of size
      [n] with mask [(0, n)] is fully valid. Defaults to [None] (all elements
      valid). Masks with full coverage are automatically removed.

    @param shape
      Dimension sizes, which may contain symbolic dimensions for
      shape-polymorphic operations.

    If any dimension is zero-size, the offset is forced to [0] and masks are
    removed. For zero-size tensors, the view represents an empty array with the
    specified shape.

    The view is marked C-contiguous if offset is [0], mask is [None], and
    strides match the expected C-contiguous pattern for the given shape. *)

(** {2 Properties} *)

val shape : t -> Symbolic_shape.t
(** [shape view] returns the dimension sizes.

    The shape may contain symbolic dimensions. Use {!Symbolic_shape.eval} to
    obtain concrete values when all dimensions are bound. *)

val strides : t -> int array
(** [strides view] returns the element strides per dimension.

    Strides are always present, even for views with symbolic shapes or masks.
    For masked views, strides describe the underlying layout but access must
    respect mask bounds.

    To check if a view can be represented as a standard strided layout without
    masks, use {!can_get_strides} or {!strides_opt}. *)

val offset : t -> int
(** [offset view] returns the starting position in the buffer.

    The offset is added to all computed linear indices. Zero-size tensors always
    have offset [0]. *)

val ndim : t -> int
(** [ndim view] returns the number of dimensions.

    Equivalent to [Symbolic_shape.rank (shape view)]. Scalars have [ndim = 0].
*)

val numel : t -> Symbolic_shape.dim
(** [numel view] returns the total number of elements as a symbolic dimension.

    Returns a {!Symbolic_shape.dim} (not [int option]), making the result
    compositional with other symbolic operations. The result is symbolic if any
    dimension is symbolic. Scalars return [static 1]. *)

val offset_dim : t -> Symbolic_shape.dim
(** [offset_dim view] returns the offset as a symbolic dimension.

    Converts the integer offset to a {!Symbolic_shape.dim} for use in symbolic
    expressions. *)

val dim : int -> t -> Symbolic_shape.dim
(** [dim axis view] returns the size of dimension [axis].

    Negative indices are not supported; use non-negative axis values.

    @raise Invalid_argument if [axis < 0] or [axis >= ndim view]. *)

val stride : int -> t -> int
(** [stride axis view] returns the stride of dimension [axis].

    Negative indices are not supported.

    @raise Invalid_argument if [axis < 0] or [axis >= ndim view]. *)

val mask : t -> (int * int) array option
(** [mask view] returns the valid bounds per dimension if the view is masked.

    Returns [Some bounds] where each [(start, end)] pair specifies valid indices
    in the half-open interval \[start, end) (exclusive end). Returns [None] if
    the view is unmasked (all elements are valid). *)

val is_c_contiguous : t -> bool
(** [is_c_contiguous view] tests for C-contiguous (row-major) layout.

    Returns [true] if the view has zero offset, no mask, and strides matching
    the C-contiguous pattern for its shape. C-contiguous views enable efficient
    bulk memory operations. *)

val strides_opt : t -> int array option
(** [strides_opt view] returns element strides when the view can be represented
    as a standard strided layout.

    Returns [None] when the view has a mask that restricts regions, requiring
    materialization before producing standard strides. Calls {!simplify}
    internally, so users don't need to simplify the view first. *)

val can_get_strides : t -> bool
(** [can_get_strides view] is [true] when {!strides_opt} would return [Some _].

    Use this function to check if a view has a mask that prevents standard
    stride representation. *)

val is_materializable : t -> bool
(** [is_materializable view] indicates whether the view can be materialized
    without resolving symbolic dimensions.

    Returns [true] if the shape is fully static and the view can produce
    standard strides. Returns [false] for views with symbolic dimensions or
    partial masks. *)

(** {2 Index Operations} *)

val linear_index : t -> int array -> int
(** [linear_index view indices] computes the linear buffer position for
    multi-dimensional indices.

    Computes [offset + sum(indices[i] * strides[i])] to map from logical
    coordinates to a buffer offset. This is the fundamental operation for
    element access in strided arrays.

    The result includes the view's offset. For a view with shape [|2; 3|],
    strides [|3; 1|], and offset [5], accessing [[1; 2]] yields
    [5 + 1*3 + 2*1 = 10].

    Time complexity: O(ndim).

    @raise Invalid_argument if [Array.length indices <> ndim view].
    @raise Failure
      if the shape is symbolic and computation requires concrete values. *)

val is_valid : t -> int array -> bool
(** [is_valid view indices] checks whether indices fall within mask bounds.

    Returns [true] if the view has no mask, or if all indices satisfy
    [start <= idx < end] for their respective dimension bounds. Returns [false]
    if indices are out of bounds or the indices array length mismatches the
    view's rank.

    Use this function to validate indices before accessing masked views. *)

(** {2 Transformations} *)

val reshape : t -> Symbolic_shape.t -> t
(** [reshape view new_shape] changes the logical shape without copying data.

    Attempts to reinterpret the view's data with a new shape while preserving
    element order. Succeeds when the reshape can be expressed through stride
    manipulation alone.

    Handles the following cases:
    - C-contiguous views: Always succeeds if total elements match
    - Size-1 dimension changes: Adding or removing singleton dimensions
    - Dimension merging: Combining contiguous dimensions (e.g., [|2; 3; 4|] to
      [|6; 4|])
    - Dimension splitting: Dividing dimensions (e.g., [|6|] to [|2; 3|])
    - All-zero strides: Broadcast views can reshape freely
    - Symbolic shapes: C-contiguous symbolic views reshape to other symbolic
      shapes (non-C-contiguous symbolic views always fail)

    Returns the same view if shapes are already equal.

    @raise Invalid_argument
      if total elements differ, unless either the original or new shape is
      zero-size.
    @raise Failure if the view has a mask (masks complicate reshape semantics).
    @raise Failure
      if strides are incompatible with the new shape. Non-contiguous views (like
      transposed tensors) cannot reshape without data reordering. The error
      message indicates expected versus actual strides and suggests calling
      [contiguous()] first. Non-C-contiguous symbolic views always fail because
      stride compatibility cannot be determined without concrete dimensions.
    @raise Failure for symbolic shapes that are not C-contiguous. *)

val expand : t -> Symbolic_shape.t -> t
(** [expand view new_shape] broadcasts singleton dimensions to larger sizes.

    Expands dimensions of size 1 to size [n] by setting their stride to 0,
    causing the same memory location to be read for all positions along that
    dimension. This is the mechanism underlying NumPy-style broadcasting.

    Scalar views (rank 0) can expand to any shape with all strides set to 0.
    Zero-size tensors create a new zero-size view with the new shape.

    For symbolic shapes without concrete values, preserves existing strides and
    metadata.

    @raise Invalid_argument
      if [rank new_shape <> ndim view] for non-scalar views.
    @raise Invalid_argument
      if attempting to expand a non-singleton dimension (size [> 1]). Only
      dimensions of size 1 can be broadcast to larger sizes. *)

val permute : t -> int array -> t
(** [permute view axes] reorders dimensions by permuting shape and strides.

    Creates a new view with dimensions arranged according to [axes], where
    [axes[i]] specifies which original dimension becomes the new dimension [i].
    For example, [permute view [|1; 0|]] transposes a 2D view.

    The permutation is validated to ensure it's a valid permutation: all values
    must be unique and in range \[0, ndim).

    Time complexity: O(ndim).

    @raise Invalid_argument if [Array.length axes <> ndim view].
    @raise Invalid_argument if [axes] contains duplicate values.
    @raise Invalid_argument
      if any axis is out of bounds ([< 0] or [>= ndim view]). *)

val shrink : t -> (int * int) array -> t
(** [shrink view bounds] restricts the view to a sub-region by slicing.

    Each [(start, end)] pair specifies the range to keep for a dimension, with
    exclusive end (NumPy convention). Adjusts offset to point to the first
    element of the sub-region and updates shape to reflect the new sizes.

    For a dimension with size [n], valid bounds are [0 <= start < end <= n].
    Bounds covering the entire dimension [(0, n)] for all dimensions return the
    view unchanged.

    Requires a fully concrete shape.

    @raise Invalid_argument if [Array.length bounds <> ndim view].
    @raise Failure if the shape contains symbolic dimensions.
    @raise Invalid_argument if any bound violates [0 <= start < end <= size]. *)

val pad : t -> (int * int) array -> t
(** [pad view padding] virtually extends dimensions by adding padding regions.

    Each [(before, after)] pair specifies how many elements to add before and
    after the existing data along a dimension. The extended regions are marked
    invalid using a mask. Reading from padded regions produces unspecified
    values; use operations like [where] to handle padding explicitly.

    Padding is virtual: no data copying occurs. The shape increases by
    [before + after] per dimension, the offset decreases by the sum of
    [before_i * stride_i] across all dimensions to account for the virtual
    prefix, and a mask records valid bounds.

    Padding with all zeros returns the view unchanged.

    Requires a fully concrete shape.

    @raise Invalid_argument if [Array.length padding <> ndim view].
    @raise Invalid_argument if any padding value is negative.
    @raise Failure if the shape contains symbolic dimensions. *)

val flip : t -> bool array -> t
(** [flip view axes_to_flip] reverses specified dimensions using negative
    strides.

    For each dimension [i] where [axes_to_flip[i] = true], negates the stride
    and adjusts the offset to point to the last element along that dimension.
    This achieves reversal without copying data.

    If the view has a mask, the mask bounds are also flipped to reflect the
    reversed order.

    Requires a fully concrete shape.

    @raise Invalid_argument if [Array.length axes_to_flip <> ndim view].
    @raise Failure if the shape contains symbolic dimensions. *)
