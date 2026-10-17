(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The frontend: the implementation of {!Nx}, which includes it.

    {!Nx} documents each value it exports from here. A creation function
    here takes first the context it creates at, which {!Nx} fixes to
    {!Placement.host}. The values that {!Nx} does not export, used by
    {!Samplers} and {!Special}, have docs of their own. *)

module B = Entry
(** The entry functions, which {!Samplers} calls by this name. *)

val err : string -> ('a, unit, string, 'b) format4 -> 'a
(** [err op fmt ...] raises [Invalid_argument] with the message [fmt ...]
    prefixed by [op ^ ": "]. *)

type ('a, 'b) t = ('a, 'b) Value.t
(** {!Nx.t}, which {!Nx.Repr} reads. *)

type context = Value.context
(** {!Value.context}. *)

type float16_elt = Nx_dtype.float16_elt
type float32_elt = Nx_dtype.float32_elt
type float64_elt = Nx_dtype.float64_elt
type bfloat16_elt = Nx_dtype.bfloat16_elt
type float8_e4m3_elt = Nx_dtype.float8_e4m3_elt
type float8_e5m2_elt = Nx_dtype.float8_e5m2_elt
type int4_elt = Nx_dtype.int4_elt
type uint4_elt = Nx_dtype.uint4_elt
type int8_elt = Nx_dtype.int8_elt
type uint8_elt = Nx_dtype.uint8_elt
type int16_elt = Nx_dtype.int16_elt
type uint16_elt = Nx_dtype.uint16_elt
type int32_elt = Nx_dtype.int32_elt
type uint32_elt = Nx_dtype.uint32_elt
type int64_elt = Nx_dtype.int64_elt
type uint64_elt = Nx_dtype.uint64_elt
type complex32_elt = Nx_dtype.complex32_elt
type complex64_elt = Nx_dtype.complex64_elt
type bool_elt = Nx_dtype.bool_elt
type bit_elt = Nx_dtype.bit_elt
type ('a, 'b) dtype = ('a, 'b) Nx_dtype.t =
  | Float16 : (float, float16_elt) dtype
  | Float32 : (float, float32_elt) dtype
  | Float64 : (float, float64_elt) dtype
  | BFloat16 : (float, bfloat16_elt) dtype
  | Float8_e4m3 : (float, float8_e4m3_elt) dtype
  | Float8_e5m2 : (float, float8_e5m2_elt) dtype
  | Int4 : (int, int4_elt) dtype
  | UInt4 : (int, uint4_elt) dtype
  | Int8 : (int, int8_elt) dtype
  | UInt8 : (int, uint8_elt) dtype
  | Int16 : (int, int16_elt) dtype
  | UInt16 : (int, uint16_elt) dtype
  | Int32 : (int32, int32_elt) dtype
  | UInt32 : (int32, uint32_elt) dtype
  | Int64 : (int64, int64_elt) dtype
  | UInt64 : (int64, uint64_elt) dtype
  | Complex64 : (Complex.t, complex32_elt) dtype
  | Complex128 : (Complex.t, complex64_elt) dtype
  | Bool : (bool, bool_elt) dtype
  | Bit : (bool, bit_elt) dtype
type float16_t = (float, float16_elt) t
type float32_t = (float, float32_elt) t
type float64_t = (float, float64_elt) t
type int8_t = (int, int8_elt) t
type uint8_t = (int, uint8_elt) t
type int16_t = (int, int16_elt) t
type uint16_t = (int, uint16_elt) t
type int32_t = (int32, int32_elt) t
type int64_t = (int64, int64_elt) t
type uint32_t = (int32, uint32_elt) t
type uint64_t = (int64, uint64_elt) t
type complex64_t = (Complex.t, complex32_elt) t
type complex128_t = (Complex.t, complex64_elt) t
type bit_t = (bool, bit_elt) t
val float16 : (float, float16_elt) dtype
val float32 : (float, float32_elt) dtype
val float64 : (float, float64_elt) dtype
val bfloat16 : (float, bfloat16_elt) dtype
val float8_e4m3 : (float, float8_e4m3_elt) dtype
val float8_e5m2 : (float, float8_e5m2_elt) dtype
val int4 : (int, int4_elt) dtype
val uint4 : (int, uint4_elt) dtype
val int8 : (int, int8_elt) dtype
val uint8 : (int, uint8_elt) dtype
val int16 : (int, int16_elt) dtype
val uint16 : (int, uint16_elt) dtype
val int32 : (int32, int32_elt) dtype
val uint32 : (int32, uint32_elt) dtype
val int64 : (int64, int64_elt) dtype
val uint64 : (int64, uint64_elt) dtype
val complex64 : (Complex.t, complex32_elt) dtype
val complex128 : (Complex.t, complex64_elt) dtype
val bool : (bool, bool_elt) dtype
val bit : (bool, bit_elt) dtype
type index =
  | I of int
  | L of int list
  | T of (int64, int64_elt) t
  | R of int * int
  | Rs of int * int * int
  | A
  | M of (bool, bool_elt) t
  | N
  | D of (int64, int64_elt) t * int
val shape : ('a, 'b) t -> int array
(** [shape x] is the shape of [x]'s view, which [x] shares unless [x] is
    placed split, whose shape is made fresh at each call. The caller must
    not change it. {!Nx.shape} copies it. *)

val dtype : ('a, 'b) t -> ('a, 'b) dtype
val dim : int -> ('a, 'b) t -> int
val ndim : ('a, 'b) t -> int
val numel : ('a, 'b) t -> int
val nbytes : ('a, 'b) t -> int
val is_c_contiguous : ('a, 'b) t -> bool
val array_prod : int array -> int
(** [array_prod a] is the product of [a]'s elements, [1] for [[||]]. *)

val reshape :
  Nx_array.Shape.t -> ('a, 'b) t -> ('a, 'b) t
val broadcast_to :
  Nx_array.Shape.t -> ('a, 'b) t -> ('a, 'b) t
val broadcasted :
  ?reverse:bool ->
  ('a, 'b) t ->
  ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
val expand : int array -> ('a, 'b) t -> ('a, 'b) t
val cast : ('c, 'd) dtype -> ('a, 'b) t -> ('c, 'd) t
val astype : ('a, 'b) dtype -> ('c, 'd) t -> ('a, 'b) t
(** [astype dt x] is {!cast}[ dt x]. *)

val bitcast : ('c, 'd) dtype -> ('a, 'b) t -> ('c, 'd) t
val contiguous : ('a, 'b) t -> ('a, 'b) t
val copy : ('a, 'b) t -> ('a, 'b) t
val create :
  context ->
  ('a, 'b) dtype ->
  Nx_array.Shape.t -> 'a array -> ('a, 'b) t
(** [create ctx] is {!Nx.create} at [ctx]. {!init}, {!scalar}, {!full},
    {!zeros} and {!ones} take their context the same way. *)

val init :
  context ->
  ('a, 'b) dtype ->
  Nx_array.Shape.t -> (int array -> 'a) -> ('a, 'b) t
val scalar :
  context ->
  ('a, 'b) dtype -> 'a -> ('a, 'b) t
val scalar_like : ('a, 'b) t -> 'a -> ('a, 'b) t
val full :
  context ->
  ('a, 'b) dtype -> Nx_array.Shape.t -> 'a -> ('a, 'b) t
val zeros :
  context ->
  ('a, 'b) dtype -> Nx_array.Shape.t -> ('a, 'b) t
val ones :
  context ->
  ('a, 'b) dtype -> Nx_array.Shape.t -> ('a, 'b) t

val full_like : ('a, 'b) t -> 'a -> ('a, 'b) t
(** [full_like x v] is a constant [v] of [x]'s dtype and shape at [x]'s
    context ({!Value.context}), which is a full copy on each device of a
    placed [x]. {!zeros_like} and {!ones_like} make theirs the same way. The
    {!Nx} namesakes of all three keep a placed value's split. *)

val zeros_like : ('a, 'b) t -> ('a, 'b) t
val ones_like : ('a, 'b) t -> ('a, 'b) t
val to_bigarray :
  ('a, 'b) t ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t
val of_bigarray :
  context ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t ->
  ('a, 'b) t
(** [of_bigarray ctx ba] is {!Nx.of_bigarray}[ ba] at [ctx]. *)

val to_array : ('a, 'b) t -> 'a array
val add :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val add_s : ('a, 'b) t -> 'a -> ('a, 'b) t
val sub :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val sub_s : ('a, 'b) t -> 'a -> ('a, 'b) t
val rsub_s : 'a -> ('a, 'b) t -> ('a, 'b) t
val mul :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val mul_s : ('a, 'b) t -> 'a -> ('a, 'b) t
val div :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val div_s : ('a, 'b) t -> 'a -> ('a, 'b) t
val rdiv_s : 'a -> ('a, 'b) t -> ('a, 'b) t
val pow :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val pow_s : ('a, 'b) t -> 'a -> ('a, 'b) t
val rpow_s : 'a -> ('a, 'b) t -> ('a, 'b) t
val maximum :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val maximum_s : ('a, 'b) t -> 'a -> ('a, 'b) t
val minimum :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val minimum_s : ('a, 'b) t -> 'a -> ('a, 'b) t
val mod_ :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val mod_s : ('a, 'b) t -> 'a -> ('a, 'b) t
val rmod_s : 'a -> ('a, 'b) t -> ('a, 'b) t
val bitwise_xor :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val bitwise_or :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val bitwise_and :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val logical_and : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val logical_or : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val logical_xor : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val bitwise_not : ('a, 'b) t -> ('a, 'b) t
val logical_not : ('a, 'b) t -> ('a, 'b) t

val cmpeq :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
(** {!equal}. [cmpne], [cmplt], [cmple], [cmpgt] and [cmpge] are
    {!not_equal}, {!less}, {!less_equal}, {!greater} and
    {!greater_equal}. *)

val cmpne :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val cmplt :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val cmple :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val cmpgt :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val cmpge :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val less :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val less_equal :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val greater :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val greater_equal :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val equal :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val not_equal :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val equal_s :
  ('a, 'b) t -> 'a -> (bool, bool_elt) t
val not_equal_s :
  ('a, 'b) t -> 'a -> (bool, bool_elt) t
val less_s :
  ('a, 'b) t -> 'a -> (bool, bool_elt) t
val greater_s :
  ('a, 'b) t -> 'a -> (bool, bool_elt) t
val less_equal_s :
  ('a, 'b) t -> 'a -> (bool, bool_elt) t
val greater_equal_s :
  ('a, 'b) t -> 'a -> (bool, bool_elt) t
val neg : ('a, 'b) t -> ('a, 'b) t
val sin : ('a, 'b) t -> ('a, 'b) t
val cos : ('a, 'b) t -> ('a, 'b) t
val sqrt : ('a, 'b) t -> ('a, 'b) t
val recip : ('a, 'b) t -> ('a, 'b) t
val log : ('a, 'b) t -> ('a, 'b) t
val exp : ('a, 'b) t -> ('a, 'b) t
val abs : ('a, 'b) t -> ('a, 'b) t
val narrow : ('a, 'b) dtype -> bool
(** [narrow dt] is [true] iff [dt] is a float dtype narrower than
    float32. *)

type real = { r : 'b. (float, 'b) t -> (float, 'b) t }
(** The type for functions of floats of any width. *)

val real_at_float32 : real -> (float, 'a) t -> (float, 'a) t
(** [real_at_float32 c x] is [c.r x], computed at float32 and rounded once
    when [x] is {!narrow}, as an operation of a backend rounds. *)

val log2 : ('a, 'b) t -> ('a, 'b) t
val exp2 : ('a, 'b) t -> ('a, 'b) t
val tan : ('a, 'b) t -> ('a, 'b) t
val square : ('a, 'b) t -> ('a, 'b) t
val sign : ('a, 'b) t -> ('a, 'b) t
val sigmoid : ('a, 'b) t -> ('a, 'b) t
val rsqrt : ('a, 'b) t -> ('a, 'b) t
val asin : ('a, 'b) t -> ('a, 'b) t
val acos : ('a, 'b) t -> ('a, 'b) t
val atan : ('a, 'b) t -> ('a, 'b) t
val sinh : ('a, 'b) t -> ('a, 'b) t
val cosh : ('a, 'b) t -> ('a, 'b) t
val tanh : ('a, 'b) t -> ('a, 'b) t
val trunc : ('a, 'b) t -> ('a, 'b) t
val ceil : ('a, 'b) t -> ('a, 'b) t
val floor : ('a, 'b) t -> ('a, 'b) t
val round : ('a, 'b) t -> ('a, 'b) t
val isinf :
  ('a, 'b) t -> (bool, bool_elt) t
val isnan :
  ('a, 'b) t -> (bool, bool_elt) t
val isfinite :
  ('a, 'b) t -> (bool, bool_elt) t
val lerp :
  ('a, 'b) t -> ('c, 'd) t -> ('e, 'f) t -> ('a, 'b) t
val lshift : ('a, 'b) t -> int -> ('a, 'b) t
val clamp :
  ?min:'a -> ?max:'a -> ('a, 'b) t -> ('a, 'b) t
val where :
  (bool, bool_elt) t ->
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val fma :
  ('a, 'b) t ->
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val check :
  'a Ptree.t ->
  (bool, bool_elt) t ->
  'a -> (int array -> 'a -> exn) -> unit
val rshift : ('a, 'b) t -> int -> ('a, 'b) t
val log1p : ('a, 'b) t -> ('a, 'b) t
val expm1 : ('a, 'b) t -> ('a, 'b) t
val asinh : ('a, 'b) t -> ('a, 'b) t
val acosh : ('a, 'b) t -> ('a, 'b) t
val atanh : ('a, 'b) t -> ('a, 'b) t
val atan2 :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val hypot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val sum :
  ?axes:int list ->
  ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
val max :
  ?axes:int list ->
  ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
val min :
  ?axes:int list ->
  ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
val prod :
  ?axes:int list ->
  ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
val flatten :
  ?start_dim:int ->
  ?end_dim:int -> ('a, 'b) t -> ('a, 'b) t
val cumsum : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
val cumprod : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
val cummax : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
val cummin : ?axis:int -> ('a, 'b) t -> ('a, 'b) t
val mean :
  ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
val var :
  ?axes:int list ->
  ?keepdims:bool -> ?ddof:int -> ('a, 'b) t -> ('a, 'b) t
val std :
  ?axes:int list ->
  ?keepdims:bool -> ?ddof:int -> ('a, 'b) t -> ('a, 'b) t
val all :
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t -> (bool, bool_elt) t
val any :
  ?axes:int list ->
  ?keepdims:bool ->
  ('a, 'b) t -> (bool, bool_elt) t
val array_equal :
  ('a, 'b) t ->
  ('a, 'b) t -> (bool, bool_elt) t
val pad :
  (int * int) array ->
  'a -> ('a, 'b) t -> ('a, 'b) t
val shrink :
  (int * int) array -> ('a, 'b) t -> ('a, 'b) t
(** [shrink limits x] is [x] cut to the half-open range [(lo, hi)] of
    [limits] on each axis, a view. *)

val unflatten :
  int -> int array -> ('a, 'b) t -> ('a, 'b) t
val ravel : ('a, 'b) t -> ('a, 'b) t
val squeeze :
  ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
val unsqueeze :
  ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
val transpose :
  ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
val flip :
  ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
val moveaxis :
  int -> int -> ('a, 'b) t -> ('a, 'b) t
val swapaxes :
  int -> int -> ('a, 'b) t -> ('a, 'b) t
val roll :
  ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t
val tile : int array -> ('a, 'b) t -> ('a, 'b) t
val repeat :
  ?axis:int -> int -> ('a, 'b) t -> ('a, 'b) t
val concatenate :
  axis:int -> ('a, 'b) t list -> ('a, 'b) t
val stack :
  ?axis:int -> ('a, 'b) t list -> ('a, 'b) t
val broadcast_arrays :
  ('a, 'b) t list -> ('a, 'b) t list

val eye :
  context ->
  ?m:int -> ?k:int -> ('a, 'b) dtype -> int -> ('a, 'b) t
(** [eye ctx] is {!Nx.eye} at [ctx]. {!arange}, {!arange_f}, {!linspace},
    {!logspace} and {!geomspace} take their context the same way. *)

val arange :
  context ->
  ('a, 'b) dtype -> int -> int -> int -> ('a, 'b) t
val arange_f :
  context ->
  (float, 'a) dtype ->
  float -> float -> float -> (float, 'a) t
val linspace :
  context ->
  ('a, 'b) dtype ->
  ?endpoint:bool -> float -> float -> int -> ('a, 'b) t
val logspace :
  context ->
  ('a, 'b) dtype ->
  ?endpoint:bool ->
  ?base:float -> float -> float -> int -> ('a, 'b) t
val geomspace :
  context ->
  ('a, 'b) dtype ->
  ?endpoint:bool -> float -> float -> int -> ('a, 'b) t
val meshgrid :
  ?indexing:[< `ij | `xy > `xy ] ->
  ('a, 'b) t ->
  ('c, 'd) t -> ('a, 'b) t * ('c, 'd) t
val tril : ?k:int -> ('a, 'b) t -> ('a, 'b) t
val triu : ?k:int -> ('a, 'b) t -> ('a, 'b) t
val take :
  ?axis:int ->
  indices:(int64, int64_elt) t ->
  ('a, 'b) t -> ('a, 'b) t
val take_along_axis :
  axis:int ->
  indices:(int64, int64_elt) t ->
  ('a, 'b) t -> ('a, 'b) t
val count : ?axes:int list -> ?keepdims:bool -> (bool, 'b) t -> int64_t
val scatter :
  ?mode:Nx_backend.scatter ->
  ?unique_indices:bool ->
  axis:int ->
  indices:(int64, int64_elt) t ->
  values:('a, 'b) t ->
  ('a, 'b) t -> ('a, 'b) t
val positions : ('a, 'b) t -> int64_t
val slice : index list -> ('a, 'b) t -> ('a, 'b) t
val get : int list -> ('a, 'b) t -> ('a, 'b) t
val item : int list -> ('a, 'b) t -> 'a
val extract :
  condition:(bool, bool_elt) t ->
  ('a, 'b) t -> ('a, 'b) t
val argwhere : ('a, 'b) t -> (int64, int64_elt) t
val nonzero : ('a, 'b) t -> (int64, int64_elt) t array
val set :
  index list ->
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val array_split :
  axis:int ->
  [< `Count of int | `Indices of int list ] ->
  ('a, 'b) t -> ('a, 'b) t list
val split :
  axis:int -> int -> ('a, 'b) t -> ('a, 'b) t list
val sort :
  ?descending:bool ->
  ?axis:int ->
  ('a, 'b) t -> ('a, 'b) t * (int64, int64_elt) t
val argsort :
  ?descending:bool ->
  ?axis:int -> ('a, 'b) t -> (int64, int64_elt) t
val quantile :
  ?axis:int -> float array -> (float, 'a) t -> (float, 'a) t
val argmax :
  ?axis:int ->
  ?keepdims:bool ->
  ('a, 'b) t -> (int64, int64_elt) t
val argmin :
  ?axis:int ->
  ?keepdims:bool ->
  ('a, 'b) t -> (int64, int64_elt) t
val reduce_segments :
  [< `Add | `Max | `Min ] ->
  segments:int ->
  (int64, int64_elt) t ->
  ('a, 'b) t -> ('a, 'b) t
val map_segments :
  segments:int ->
  (int64, int64_elt) t ->
  ((int64, int64_elt) t ->
  ('a, 'b) t -> ('c, 'd) t) ->
  ('a, 'b) t -> ('c, 'd) t
val associative_scan :
  ?axis:int -> 'a Ptree.t -> ('a -> 'a -> 'a) -> 'a -> 'a
val ewma :
  ?axis:int ->
  alpha:float -> ('a, 'b) t -> ('a, 'b) t
val reduce_ranges :
  [< `Add | `Max | `Min > `Add ] ->
  lo:(int64, int64_elt) t ->
  hi:(int64, int64_elt) t ->
  ('a, 'b) t -> ('a, 'b) t
val top_k :
  k:int ->
  ?axis:int ->
  ('a, 'b) t ->
  ('a, 'b) t * (int64, int64_elt) t
val order_key : ('c, 'd) dtype -> ('a, 'b) t -> ('c, 'd) t
val lexsort :
  ('a, 'b) t -> (int64, int64_elt) t
val searchsorted :
  side:[< `Left | `Right ] ->
  ('a, 'b) t -> ('a, 'b) t -> (int64, int64_elt) t
type groups = { ids : int64_t; first : int64_t; counts : int64_t }
val unique : ('a, 'b) t -> groups
val histogram :
  ?weights:('a, 'c) t ->
  ((float, 'b) t * (float, 'b) t) list -> (float, 'b) t
val matmul :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val diagonal :
  ?offset:int ->
  ?axis1:int ->
  ?axis2:int -> ('a, 'b) t -> ('a, 'b) t
val diag : ?k:int -> ('a, 'b) t -> ('a, 'b) t
val matrix_transpose : ('a, 'b) t -> ('a, 'b) t
val real : (float, 'a) dtype -> (Complex.t, 'b) t -> (float, 'a) t
val imag : (float, 'a) dtype -> (Complex.t, 'b) t -> (float, 'a) t
val magnitude : ('a, 'b) dtype -> (Complex.t, 'c) t -> ('a, 'b) t
val angle :
  (float, 'a) dtype -> (Complex.t, 'b) t -> (float, 'a) t
val complex :
  (Complex.t, 'c) dtype ->
  re:('a, 'b) t ->
  im:('a, 'b) t -> (Complex.t, 'c) t
val conjugate : ('a, 'b) t -> ('a, 'b) t
val tensordot :
  ?axes:int list * int list ->
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val dot :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val vdot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val vecdot :
  ?axis:int ->
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val inner :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val outer :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val einsum : string -> ('a, 'b) t array -> ('a, 'b) t
val kron :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val multi_dot : ('a, 'b) t array -> ('a, 'b) t
val cross :
  ?axis:int ->
  ('a, 'b) t -> ('c, 'd) t -> ('a, 'b) t
val cholesky :
  ?upper:bool -> ('a, 'b) t -> ('a, 'b) t
val qr :
  ?mode:[< `Complete | `Reduced ] ->
  ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
val lu :
  ('a, 'b) t ->
  Op.int64_t * ('a, 'b) t * ('a, 'b) t
val svd :
  ?full_matrices:bool ->
  ('a, 'b) t ->
  ('a, 'b) t * (float, float64_elt) t *
  ('a, 'b) t
val svdvals : ('a, 'b) t -> (float, float64_elt) t
val eig :
  ('a, 'b) t ->
  (Complex.t, complex64_elt) t *
  (Complex.t, complex64_elt) t
val eigh :
  ?uplo:[< `L | `U > `L ] ->
  ('a, 'b) t ->
  (float, float64_elt) t * ('a, 'b) t
val eigvals :
  ('a, 'b) t ->
  (Complex.t, complex64_elt) t
val eigvalsh :
  ?uplo:[< `L | `U > `L ] ->
  ('a, 'b) t -> (float, float64_elt) t
val norm :
  ?ord:[> `Fro
  | `Inf
  | `NegInf
  | `NegOne
  | `NegTwo
  | `Nuc
  | `One
  | `P of float
  | `Two ] ->
  ?axes:int list -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
val det : ('a, 'b) t -> ('a, 'b) t
val slogdet : ('a, 'b) t -> ('a, 'b) t * (float, float64_elt) t
val matrix_rank :
  ?tol:float ->
  ?rtol:float ->
  ?hermitian:bool ->
  ('a, 'b) t -> (int32, int32_elt) t
val trace : ?offset:int -> ('a, 'b) t -> ('a, 'b) t
val solve_triangular :
  ?upper:bool ->
  ?transpose:bool ->
  ?unit_diag:bool ->
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val solve : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val pinv :
  ?rtol:float -> ?hermitian:bool -> ('a, 'b) t -> ('a, 'b) t
val lstsq :
  ?rcond:float ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('a, 'b) t * ('a, 'b) t *
  (int32, int32_elt) t *
  (float, float64_elt) t
val inv : ('a, 'b) t -> ('a, 'b) t
val matrix_power : ('a, 'b) t -> int -> ('a, 'b) t
val cond :
  ?p:[> `Inf | `One | `Two ] -> ('a, 'b) t -> ('a, 'b) t
val tensorsolve :
  ?axes:int list -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
val tensorinv : ?ind:int -> ('a, 'b) t -> ('a, 'b) t
type fft_norm = [ `Backward | `Forward | `Ortho ]
val fft :
  ?axis:int ->
  ?n:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'a) t -> (Complex.t, 'a) t
val ifft :
  ?axis:int ->
  ?n:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'a) t -> (Complex.t, 'a) t
val rfft :
  (Complex.t, 'a) dtype ->
  ?axis:int ->
  ?n:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'b) t -> (Complex.t, 'a) t
val irfft :
  (float, 'a) dtype ->
  ?axis:int ->
  ?n:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'b) t -> (float, 'a) t
val fft2 :
  ?axes:int list ->
  ?s:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'a) t -> (Complex.t, 'a) t
val ifft2 :
  ?axes:int list ->
  ?s:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'a) t -> (Complex.t, 'a) t
val fftn :
  ?axes:int list ->
  ?s:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'a) t -> (Complex.t, 'a) t
val ifftn :
  ?axes:int list ->
  ?s:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'a) t -> (Complex.t, 'a) t
val rfft2 :
  (Complex.t, 'a) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'b) t -> (Complex.t, 'a) t
val irfft2 :
  (float, 'a) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'b) t -> (float, 'a) t
val rfftn :
  (Complex.t, 'a) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'b) t -> (Complex.t, 'a) t
val irfftn :
  (float, 'a) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'b) t -> (float, 'a) t
val hfft :
  (float, 'a) dtype ->
  ?axis:int ->
  ?n:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (Complex.t, 'b) t -> (float, 'a) t
val ihfft :
  (Complex.t, 'a) dtype ->
  ?axis:int ->
  ?n:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'b) t -> (Complex.t, 'a) t

val fftfreq :
  context ->
  (float, 'a) dtype -> ?d:float -> int -> (float, 'a) t
(** [fftfreq ctx] is {!Nx.fftfreq} at [ctx]. {!rfftfreq} and {!hann} take
    their context the same way. *)

val rfftfreq :
  context ->
  (float, 'a) dtype -> ?d:float -> int -> (float, 'a) t
val fftshift :
  ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
val ifftshift :
  ?axes:int list -> ('a, 'b) t -> ('a, 'b) t
val hann :
  context ->
  (float, 'a) dtype -> int -> (float, 'a) t
val stft :
  (Complex.t, 'c) dtype ->
  window:int ->
  ?step:int ->
  ?win:(float, 'a) t ->
  (float, 'a) t -> (Complex.t, 'c) t
val istft :
  (float, 'a) dtype ->
  window:int ->
  ?step:int ->
  ?win:(float, 'a) t ->
  ?length:int -> (Complex.t, 'b) t -> (float, 'a) t
val dct :
  ?type_:int ->
  ?axis:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'a) t -> (float, 'a) t
val idct :
  ?type_:int ->
  ?axis:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'a) t -> (float, 'a) t
val dctn :
  ?type_:int ->
  ?axes:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'a) t -> (float, 'a) t
val idctn :
  ?type_:int ->
  ?axes:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'a) t -> (float, 'a) t
val dst :
  ?type_:int ->
  ?axis:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'a) t -> (float, 'a) t
val idst :
  ?type_:int ->
  ?axis:int ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'a) t -> (float, 'a) t
val dstn :
  ?type_:int ->
  ?axes:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'a) t -> (float, 'a) t
val idstn :
  ?type_:int ->
  ?axes:int list ->
  ?norm:[< `Backward | `Forward | `Ortho > `Backward ] ->
  (float, 'a) t -> (float, 'a) t
val softmax :
  ?axes:int list -> ?scale:float -> ('a, 'b) t -> ('a, 'b) t
val log_softmax :
  ?axes:int list ->
  ?scale:float -> ('a, 'b) t -> ('a, 'b) t
val logsumexp :
  ?axes:int list ->
  ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
val logmeanexp :
  ?axes:int list ->
  ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
val standardize :
  ?axes:int list ->
  ?mean:('a, 'b) t ->
  ?variance:('c, 'd) t ->
  ?epsilon:float -> ('e, 'f) t -> ('e, 'f) t
val erf : (float, 'b) t -> (float, 'b) t
val sliding_window :
  ?axis:int ->
  window:int ->
  ?step:int -> ('a, 'b) t -> ('a, 'b) t
val extract_patches :
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('a, 'b) t -> ('a, 'b) t
val combine_patches :
  output_size:int array ->
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('a, 'b) t -> ('a, 'b) t
val correlate :
  ?padding:[< `Full | `Same | `Valid > `Valid ] ->
  ('a, 'b) t -> ('c, 'd) t -> ('a, 'b) t
val convolve :
  ?padding:[< `Full | `Same | `Valid > `Valid ] ->
  ('a, 'b) t -> ('c, 'd) t -> ('a, 'b) t
val maximum_filter :
  kernel_size:int array ->
  ?stride:int array -> ('a, 'b) t -> ('a, 'b) t
val minimum_filter :
  kernel_size:int array ->
  ?stride:int array -> ('a, 'b) t -> ('a, 'b) t
val uniform_filter :
  kernel_size:int array ->
  ?stride:int array -> ('a, 'b) t -> ('a, 'b) t
val one_hot :
  num_classes:int -> ('a, 'b) t -> (int, uint8_elt) t
val pp_shape : Format.formatter -> Nx_array.Shape.t -> unit
val pp_dtype : Format.formatter -> ('a, 'b) dtype -> unit
val pp : Format.formatter -> ('a, 'b) t -> unit
val to_string : ('a, 'b) t -> string
val print : ('a, 'b) t -> unit
val map_item :
  ('a -> 'a) -> ('a, 'b) t -> ('a, 'b) t
val iter_item : ('a -> unit) -> ('a, 'b) t -> unit
val fold_item : ('a -> 'b -> 'a) -> 'a -> ('b, 'c) t -> 'a
module Infix : sig
  val ( + ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( +$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  val ( - ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( -$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  val ( ~- ) : ('a, 'b) t -> ('a, 'b) t
  val ( * ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( *$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  val ( / ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( /$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  val ( ** ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( **$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  val ( % ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( mod ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( %$ ) : ('a, 'b) t -> 'a -> ('a, 'b) t
  val ( lxor ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( lor ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( land ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( && ) :
    ('a, 'b) t ->
    ('a, 'b) t -> ('a, 'b) t
  val ( || ) :
    ('a, 'b) t ->
    ('a, 'b) t -> ('a, 'b) t
  val ( < ) :
    ('a, 'b) t ->
    ('a, 'b) t -> (bool, bool_elt) t
  val ( <$ ) :
    ('a, 'b) t -> 'a -> (bool, bool_elt) t
  val ( <> ) :
    ('a, 'b) t ->
    ('a, 'b) t -> (bool, bool_elt) t
  val ( <>$ ) :
    ('a, 'b) t -> 'a -> (bool, bool_elt) t
  val ( = ) :
    ('a, 'b) t ->
    ('a, 'b) t -> (bool, bool_elt) t
  val ( =$ ) :
    ('a, 'b) t -> 'a -> (bool, bool_elt) t
  val ( > ) :
    ('a, 'b) t ->
    ('a, 'b) t -> (bool, bool_elt) t
  val ( >$ ) :
    ('a, 'b) t -> 'a -> (bool, bool_elt) t
  val ( <= ) :
    ('a, 'b) t ->
    ('a, 'b) t -> (bool, bool_elt) t
  val ( <=$ ) :
    ('a, 'b) t -> 'a -> (bool, bool_elt) t
  val ( >= ) :
    ('a, 'b) t ->
    ('a, 'b) t -> (bool, bool_elt) t
  val ( >=$ ) :
    ('a, 'b) t -> 'a -> (bool, bool_elt) t
  val ( *@ ) :
    ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  val ( /@ ) :
    ('a, 'b) t ->
    ('a, 'b) t -> ('a, 'b) t
  val ( **@ ) : ('a, 'b) t -> int -> ('a, 'b) t
  val ( .%{} ) : ('a, 'b) t -> int list -> ('a, 'b) t
  val ( .${} ) :
    ('a, 'b) t ->
    index list -> ('a, 'b) t
end
