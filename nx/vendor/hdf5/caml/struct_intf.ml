module type S_no_ppx = sig
  val fields : Field.t list
  val nfields : int
  val type_size : int
  val field_names : string array
  val field_offset : int array
  val field_sizes : int array

  (* [t = Ptr.t] is hidden so that the users cannot access low level functions. *)
  type t

  val create : unit -> t

  val pos : t -> int
  val has_next : t -> bool
  val has_prev : t -> bool

  val to_string : t -> string

  module Accessors : sig
    module Unpacked : sig
      type nonrec 'a t = {
        field : 'a Field.Unpacked.t;
        get   : t -> 'a;
        set   : t -> 'a -> unit;
      }
    end

    type t = T : _ Unpacked.t -> t

    val all : t array
  end

  module Array : sig
    type e = t
    type t

    val make : int -> t
    val init : int -> (int -> e -> unit) -> t
    val length : t -> int
    val unsafe_get : t -> int -> e
    val get : t -> int -> e
    val iter : t -> f:(e -> unit) -> unit
    val iteri : t -> f:(int -> e -> unit) -> unit

    (** Creates and writes a table. *)
    val make_table : t -> ?title:string -> ?chunk_size:int -> ?compress:bool -> H5.t
      -> string -> unit

    (** Adds records to the end of the table. *)
    val append_records : t -> H5.t -> string -> unit

    (** Ovewrites records. *)
    val write_records : t -> H5.t -> start:int -> string -> unit

    (** Reads a table. *)
    val read_table : H5.t -> string -> t

    (** Reads records. *)
    val read_records : H5.t -> start:int -> nrecords:int -> string -> t

    (** Reads only the fields that the struct contains. *)
    val read_fields_name : H5.t -> ?field_names:string -> ?start:int -> ?nrecords:int
      -> string -> t

    (** Writes records. *)
    val write : t -> ?deflate:int -> H5.t -> string -> unit

    (** Reads records. *)
    val read : H5.t -> ?data:t -> string -> t

    (** Retrieves the pointer to the data. *)
    val data : t -> Hdf5_raw.H5tb.Data.t

    (** A string representation. *)
    val to_string : t -> string
  end

  val mem : t -> Array.t

  module Vector : sig
    type e = t
    type t

    (** Create a vector with the given initial capacity and growth factor. *)
    val create : ?capacity:int -> ?growth_factor:float -> unit -> t

    (** Returns the current capacity of the given vector. *)
    val capacity : t -> int

    (** Ensures that the vector has at least the given capacity. *)
    val ensure_capacity : t -> int -> unit

    (** Returns the growth factor of the given vector. *)
    val growth_factor : t -> float

    (** Sets the growth factor to the given value. *)
    val set_growth_factor : t -> float -> unit

    (** Returns the length of the given vector. *)
    val length : t -> int

    (** Returns the end of the given vector or raises an exception if the vector is empty
     *)
    val end_ : t -> e

    (** Sets the capacity of the vector to the given size. *)
    val realloc : t -> int -> unit

    (** Appends an element to the end of the vector and returns a pointer to the end. *)
    val append : t -> e

    (** Removes all the elements in the vector. *)
    val clear : t -> unit

    (** Unsafely returns the element with the given index in the vector. *)
    val unsafe_get : t -> int -> e

    (** Returns the element with the given index in the vector. *)
    val get : t -> int -> e

    (** Iterates through all the element of the vector. *)
    val iter : t -> f:(e -> unit) -> unit

    (** Iterates through all the element of the vector. *)
    val iteri : t -> f:(int -> e -> unit) -> unit

    (** Creates a vector of the given array. *)
    val of_array : ?growth_factor:float -> Array.t -> t

    (** Creates an array of the given vector. *)
    val to_array : t -> Array.t

    (** Sets a function to be called then whe vector is reallocated. *)
    val on_realloc : t -> (t -> unit) -> unit
  end

  module Queue : sig
    type e = t
    type t

    (** Return a new queue with the given capacity. *)
    val create : ?capacity:int -> unit -> t

    (** Return [true] if the given queue is empty, [false] otherwise. *)
    val is_empty : t -> bool

    (** Return the number of elements in a queue. *)
    val length : t -> int

    (** Add an element to the end of the queue and returns a pointer to it. *)
    val add : t -> e

    (** Removes and returns the first element of the given queue or raises [Queue.Empty]
        if empty. *)
    val take : t -> e

    (** Returns the first element of the given queue or raises [Queue.Empty] if empty. *)
    val peek : t -> e
  end
end

module type S = sig
  include S_no_ppx

  val unsafe_next : t -> unit
  val unsafe_prev : t -> unit
  val unsafe_move : t -> int -> unit
  val next : t -> unit
  val prev : t -> unit
  val move : t -> int -> unit
end
