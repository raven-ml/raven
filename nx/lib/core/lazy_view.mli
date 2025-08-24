(** View tracking for lazy tensor operations.

    A view tracker maintains a list of views that represent a sequence of
    transformations. This allows movement operations (reshape, permute, etc.) to
    be performed without copying data. *)

type t
(** A view tracker maintaining a sequence of view transformations. *)

(** {2 Creation} *)

val create : Symbolic_shape.t -> t
(** [create shape] creates a view tracker with initial contiguous view. *)

val create_strided : Symbolic_shape.t -> strides:int array -> offset:int -> t
(** [create_strided shape ~strides ~offset] creates a view tracker with custom
    strides (in elements) and offset (in elements). This is used for advanced
    indexing operations like as_strided. *)

(** {2 Properties} *)

val shape : t -> Symbolic_shape.t
val ndim : t -> int
val numel : t -> Symbolic_shape.dim
val offset : t -> Symbolic_shape.dim
val strides : t -> int array option
val is_contiguous : t -> bool

(** {2 Movement Operations} *)

val reshape : Symbolic_shape.t -> t -> t
(** [reshape new_shape tracker] changes dimensions without copying.

    @raise Invalid_argument if total elements don't match *)

val permute : int array -> t -> t
(** [permute axes tracker] reorders dimensions.

    @raise Invalid_argument if invalid permutation *)

val expand : Symbolic_shape.t -> t -> t
(** [expand new_shape tracker] broadcasts singleton dimensions.

    @raise Invalid_argument if non-singleton expansion attempted *)

val shrink : (int * int) array -> t -> t
(** [shrink bounds tracker] restricts to sub-region.

    @raise Invalid_argument if bounds invalid *)

val pad : (int * int) array -> t -> t
(** [pad padding tracker] extends dimensions virtually.

    @raise Invalid_argument if negative padding *)

val flip : bool array -> t -> t
(** [flip axes_to_flip tracker] reverses specified dimensions.

    @raise Invalid_argument if array length mismatch *)

(** {2 Analysis} *)

val simplify : t -> t
val compose : t -> View.t option
val can_get_strides : t -> bool
val is_materializable : t -> bool
