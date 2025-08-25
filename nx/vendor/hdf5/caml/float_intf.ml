open Bigarray

module type S = sig
  type t
  type float_elt

  (** Writes the given float array to the data set. *)
  val write_float_array : t -> string -> ?deflate:int -> float array
    -> unit

  (** Reads the data set into a float array.

      @param data If provided, the storage for the data. *)
  val read_float_array : t -> ?data:float array -> string -> float array

  val write_float_genarray : t -> string -> ?deflate:int
    -> (float, float_elt, _) Genarray.t -> unit

  (** Reads the data set into a float Genarray.t.

      @param data If provided, the storage for the data. *)
  val read_float_genarray : t -> ?data:(float, float_elt, 'a) Genarray.t -> string
    -> 'a layout -> (float, float_elt, 'a) Genarray.t

  (** Writes the given float Array1.t to the data set. *)
  val write_float_array1 : t -> string -> ?deflate:int
    -> (float, float_elt, _) Array1.t -> unit

  (** Reads the data set into a float Array1.t.

      @param data If provided, the storage for the data. *)
  val read_float_array1 : t -> ?data:(float, float_elt, 'a) Array1.t -> string
    -> 'a layout -> (float, float_elt, 'a) Array1.t

  (** Writes the given float Array1.t to the data set. *)
  val write_float_array2 : t -> string -> ?deflate:int
    -> (float, float_elt, _) Array2.t -> unit

  (** Reads the data set into a float Array2.t.

      @param data If provided, the storage for the data. *)
  val read_float_array2 : t -> ?data:(float, float_elt, 'a) Array2.t -> string
    -> 'a layout -> (float, float_elt, 'a) Array2.t

  (** Writes the given float Array1.t to the data set. *)
  val write_float_array3 : t -> string -> ?deflate:int
    -> (float, float_elt, _) Array3.t -> unit

  (** Reads the data set into a float Array3.t.

      @param data If provided, the storage for the data. *)
  val read_float_array3 : t -> ?data:(float, float_elt, 'a) Array3.t -> string
    -> 'a layout -> (float, float_elt, 'a) Array3.t

  (** [write_attribute_float t name v] writes the given [float] as an attribute with the
      given name. *)
  val write_attribute_float : t -> string -> float -> unit

  (** [read_attribute_float t name] reads the attribute with the given name as a float. *)
  val read_attribute_float : t -> string -> float

  (** Writes the given [float] as an attribute with the given name. *)
  val write_attribute_float_array : t -> string -> float array -> unit

  (** Reads the attribute with the given name as a float. *)
  val read_attribute_float_array : t -> string -> float array
end
