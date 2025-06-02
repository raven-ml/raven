(** Tensor view: strided view of tensor data.

    Views describe how to interpret a linear buffer as a multi-dimensional array
    through shape, strides, and offset. Supports non-contiguous layouts and
    masked regions. *)

type t
(** View encapsulating tensor layout information. *)

(** {2 Creation} *)

val create :
  ?offset:int ->
  ?strides:int array ->
  ?mask:(int * int) array ->
  Symbolic_shape.t ->
  t
(** [create ?offset ?strides ?mask shape] constructs view.

    Default offset is 0. Default strides are C-contiguous. Mask specifies valid
    ranges per dimension as [(start, end)] pairs. *)

(** {2 Properties} *)

val shape : t -> Symbolic_shape.t
(** [shape view] returns dimension sizes. *)

val strides : t -> int array
(** [strides view] returns element strides per dimension. *)

val offset : t -> int
(** [offset view] returns starting position in buffer. *)

val ndim : t -> int
(** [ndim view] returns number of dimensions. *)

val numel : t -> Symbolic_shape.dim
(** [numel view] returns total elements (may be symbolic). *)

val dim : int -> t -> Symbolic_shape.dim
(** [dim axis view] returns size of dimension [axis].

    @raise Invalid_argument if axis out of bounds *)

val stride : int -> t -> int
(** [stride axis view] returns stride of dimension [axis].

    @raise Invalid_argument if axis out of bounds *)

val mask : t -> (int * int) array option
(** [mask view] returns valid bounds per dimension if masked. *)

val is_c_contiguous : t -> bool
(** [is_c_contiguous view] tests for row-major contiguous layout. *)

(** {2 Index Operations} *)

val linear_index : t -> int array -> int
(** [linear_index view indices] computes buffer position.

    Includes view's offset in result.

    @raise Invalid_argument if rank mismatch
    @raise Failure if shape is symbolic and non-contiguous *)

val is_valid : t -> int array -> bool
(** [is_valid view indices] checks mask bounds.

    Returns true if no mask or indices within all bounds. *)

val can_be_strided : t -> bool

(** {2 Transformations} *)

val reshape : t -> Symbolic_shape.t -> t
(** [reshape view new_shape] changes dimensions.

    Returns view if possible, fails if requires reordering. Handles -1
    dimensions and size-1 squeezing/unsqueezing.

    @raise Invalid_argument if size mismatch
    @raise Failure if cannot reshape strided view *)

val expand : t -> Symbolic_shape.t -> t
(** [expand view new_shape] broadcasts singleton dimensions.

    Size-1 dimensions become size-n with stride 0.

    @raise Invalid_argument if non-singleton expansion attempted *)

val permute : t -> int array -> t
(** [permute view axes] reorders dimensions.

    @raise Invalid_argument if invalid permutation *)

val shrink : t -> (int * int) array -> t
(** [shrink view bounds] restricts to sub-region.

    Bounds are [(start, end)] pairs per dimension.

    @raise Invalid_argument if bounds invalid *)

val pad : t -> (int * int) array -> t
(** [pad view padding] extends dimensions virtually.

    Padding is [(before, after)] pairs. Creates mask for valid region.

    @raise Invalid_argument if negative padding *)

val flip : t -> bool array -> t
(** [flip view axes_to_flip] reverses specified dimensions.

    Adjusts strides to negative and updates offset.

    @raise Invalid_argument if array length mismatch *)

(** {2 View Composition} *)

val merge : t -> t -> t option
(** [merge view1 view2] attempts to compose two views.

    Returns a single view representing the composition of view1 followed by
    view2, or None if the views cannot be merged. *)

val simplify : t -> t
(** [simplify view] attempts to simplify the view representation.

    May return the same view if no simplification is possible. *)
