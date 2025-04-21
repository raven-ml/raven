(** Rune: A JAX-like library built on Ndarray *)

(** {2 Core Rune Types} *)

type ('a, 'b, 'dev) t

(** {2 Element Types} *)

(* Layouts *)

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_elt = Bigarray.int8_signed_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type int16_elt = Bigarray.int16_signed_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt

(** GADT representing supported element data types. *)
type ('a, 'b) dtype =
  | Float16 : (float, float16_elt) dtype  (** 16-bit float. *)
  | Float32 : (float, float32_elt) dtype  (** 32-bit float. *)
  | Float64 : (float, float64_elt) dtype  (** 64-bit float. *)
  | Int8 : (int, int8_elt) dtype  (** Signed 8-bit integer. *)
  | Int16 : (int, int16_elt) dtype  (** Signed 16-bit integer. *)
  | Int32 : (int32, int32_elt) dtype  (** Signed 32-bit integer. *)
  | Int64 : (int64, int64_elt) dtype  (** Signed 64-bit integer. *)
  | UInt8 : (int, uint8_elt) dtype  (** Unsigned 8-bit integer. *)
  | UInt16 : (int, uint16_elt) dtype  (** Unsigned 16-bit integer. *)
  | Complex32 : (Complex.t, complex32_elt) dtype  (** 32-bit complex float. *)
  | Complex64 : (Complex.t, complex64_elt) dtype  (** 64-bit complex float. *)

val float16 : (float, float16_elt) dtype
val float32 : (float, float32_elt) dtype
val float64 : (float, float64_elt) dtype
val int8 : (int, int8_elt) dtype
val uint8 : (int, uint8_elt) dtype
val int16 : (int, int16_elt) dtype
val uint16 : (int, uint16_elt) dtype
val int32 : (int32, int32_elt) dtype
val int64 : (int64, int64_elt) dtype
val complex32 : (Complex.t, complex32_elt) dtype
val complex64 : (Complex.t, complex64_elt) dtype

(** {1 Creation} *)

val ndarray : ('a, 'b) Ndarray.t -> ('a, 'b, [ `cpu ]) t
val create : ('a, 'b) dtype -> int array -> 'a array -> ('a, 'b, [ `cpu ]) t
val zeros : ('a, 'b) dtype -> int array -> ('a, 'b, [ `cpu ]) t
val ones : ('a, 'b) dtype -> int array -> ('a, 'b, [ `cpu ]) t
val rand : (float, 'a) dtype -> int array -> (float, 'a, [ `cpu ]) t
val scalar : ('a, 'b) dtype -> 'a -> ('a, 'b, [ `cpu ]) t

(** {1 Properties} *)

val shape : ('a, 'b, 'dev) t -> int array
val dim : int -> ('a, 'b, 'dev) t -> int
val size : ('a, 'b, 'dev) t -> int
val dtype : ('a, 'b, 'dev) t -> ('a, 'b) dtype

(** {1 Access} *)

val get : int array -> ('a, 'b, 'dev) t -> 'a
val set : int array -> 'a -> ('a, 'b, 'dev) t -> unit

(** {1 Tensor Operations} *)

val neg : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val sin : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
val cos : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
val exp : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val log : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val add : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val add_inplace : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val sub : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val sub_inplace : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val mul : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val mul_inplace : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val div : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val div_inplace : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val maximum : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val minimum : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val sum : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val mean : (float, 'a, 'dev) t -> (float, 'a, 'dev) t
val matmul : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val transpose : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t
val reshape : ('a, 'b, 'dev) t -> int array -> ('a, 'b, 'dev) t

(** {1 Evaluation} *)

val eval : ('a -> 'b) -> 'a -> 'b

(** {1 Differentiation} *)

val grad :
  (('a, 'b, 'dev) t -> ('c, 'd, 'dev) t) -> ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t

val grads :
  (('a, 'b, 'dev) t list -> ('c, 'd, 'dev) t) ->
  ('a, 'b, 'dev) t list ->
  ('a, 'b, 'dev) t list

val value_and_grad :
  (('a, 'b, 'dev) t -> ('d, 'e, 'dev) t) ->
  ('a, 'b, 'dev) t ->
  ('d, 'e, 'dev) t * ('a, 'b, 'dev) t

val value_and_grads :
  (('a, 'b, 'dev) t list -> ('d, 'e, 'dev) t) ->
  ('a, 'b, 'dev) t list ->
  ('d, 'e, 'dev) t * ('a, 'b, 'dev) t list

(** {1 Printing and Debugging} *)

val pp : Format.formatter -> ('a, 'b, 'dev) t -> unit
val to_string : ('a, 'b, 'dev) t -> string
val print : ('a, 'b, 'dev) t -> unit
