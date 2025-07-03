(** OCaml bindings to XLA (Accelerated Linear Algebra) *)

val initialize : unit -> unit
(** Initialize the XLA library. Must be called before using any other functions.
*)

module Element_type = Element_type
(** XLA element types *)

module Client : sig
  type t

  val cpu : unit -> t
  (** Create a CPU client *)

  val gpu : ?device_id:int -> unit -> t
  (** Create a GPU client if available *)
end

module Shape : sig
  type t

  val create : ?layout:int array -> int array -> t
  (** Create a shape with dimensions and optional layout *)

  val dimensions : t -> int array
  (** Get dimensions of a shape *)

  val rank : t -> int
  (** Get the rank (number of dimensions) *)

  val element_count : t -> int
  (** Get total number of elements *)
end

module Literal : sig
  type t

  val create_r0_f32 : float -> t
  (** Create a rank-0 (scalar) float32 literal *)

  val create_r1_f32 : float array -> t
  (** Create a rank-1 float32 literal from array *)

  val create_r2_f32 : float array array -> t
  (** Create a rank-2 float32 literal from 2D array *)

  val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t -> t
  (** Create a literal from a bigarray *)

  val to_bigarray :
    t ->
    ('a, 'b) Bigarray.kind ->
    ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
  (** Convert a literal to a bigarray with the specified kind *)

  val shape : t -> Shape.t
  (** Get the shape of a literal *)
end

module Computation : sig
  type t
  type executable

  val compile : Client.t -> t -> executable
  (** Compile a computation for a specific client *)

  val execute : executable -> Literal.t list -> Literal.t list
  (** Execute a compiled computation *)
end

module Builder : sig
  type t
  type op

  val create : string -> t
  (** Create a new computation builder with a name *)

  val parameter : t -> int -> Shape.t -> string -> op
  (** Add a parameter to the computation *)

  val constant : t -> Literal.t -> op
  (** Add a constant to the computation *)

  val add : t -> op -> op -> op
  (** Element-wise addition *)

  val multiply : t -> op -> op -> op
  (** Element-wise multiplication *)

  val subtract : t -> op -> op -> op
  (** Element-wise subtraction *)

  val divide : t -> op -> op -> op
  (** Element-wise division *)

  val remainder : t -> op -> op -> op
  (** Element-wise remainder *)

  val max : t -> op -> op -> op
  (** Element-wise maximum *)

  val min : t -> op -> op -> op
  (** Element-wise minimum *)

  val pow : t -> op -> op -> op
  (** Element-wise power *)

  val dot : t -> op -> op -> op
  (** Matrix multiplication *)

  val and_ : t -> op -> op -> op
  (** Bitwise AND *)

  val or_ : t -> op -> op -> op
  (** Bitwise OR *)

  val xor : t -> op -> op -> op
  (** Bitwise XOR *)

  val neg : t -> op -> op
  (** Negation *)

  val abs : t -> op -> op
  (** Absolute value *)

  val exp : t -> op -> op
  (** Exponential *)

  val log : t -> op -> op
  (** Natural logarithm *)

  val sqrt : t -> op -> op
  (** Square root *)

  val sin : t -> op -> op
  (** Sine *)

  val cos : t -> op -> op
  (** Cosine *)

  val tanh : t -> op -> op
  (** Hyperbolic tangent *)

  val eq : t -> op -> op -> op
  (** Element-wise equality *)

  val ne : t -> op -> op -> op
  (** Element-wise not equal *)

  val lt : t -> op -> op -> op
  (** Element-wise less than *)

  val le : t -> op -> op -> op
  (** Element-wise less than or equal *)

  val gt : t -> op -> op -> op
  (** Element-wise greater than *)

  val ge : t -> op -> op -> op
  (** Element-wise greater than or equal *)

  val select : t -> op -> op -> op -> op
  (** Select elements from two tensors based on a condition *)

  val convert_element_type : t -> op -> Element_type.t -> op
  (** Convert tensor elements to a different type *)

  val pad :
    t ->
    op ->
    op ->
    low_padding:int array ->
    high_padding:int array ->
    interior_padding:int array ->
    op
  (** Pad a tensor with a given value *)

  val reverse : t -> op -> int array -> op
  (** Reverse a tensor along specified dimensions *)

  val slice :
    t ->
    op ->
    start_indices:int array ->
    limit_indices:int array ->
    strides:int array ->
    op
  (** Slice a tensor *)

  val concatenate : t -> op array -> int -> op
  (** Concatenate tensors along a dimension *)

  val reshape : t -> op -> int array -> op
  (** Reshape operation *)

  val transpose : t -> op -> int array -> op
  (** Transpose with permutation *)

  val broadcast : t -> op -> int array -> op
  (** Broadcast to new shape *)

  val reduce_sum : t -> op -> dims:int array -> keep_dims:bool -> op
  (** Reduce sum along dimensions *)

  val reduce_max : t -> op -> dims:int array -> keep_dims:bool -> op
  (** Reduce max along dimensions *)

  val reduce_min : t -> op -> dims:int array -> keep_dims:bool -> op
  (** Reduce min along dimensions *)

  val gather : t -> op -> op -> axis:int -> op
  (** Gather elements from operand using indices along the specified axis *)

  val conv2d :
    t ->
    op ->
    op ->
    strides:int array ->
    padding:(int * int) array ->
    ?dilation:int array ->
    unit ->
    op
  (** 2D convolution operation (NCHW format) with optional dilation *)

  val scatter :
    t -> op -> op -> op -> axis:int -> update_computation:Computation.t -> op
  (** Scatter updates into operand at indices along the specified axis *)

  val build : t -> op -> Computation.t
  (** Build the computation with the given root operation *)
end
