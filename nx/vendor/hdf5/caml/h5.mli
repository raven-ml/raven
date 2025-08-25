open Bigarray
open Hdf5_raw

(** Escape slash characters in the given string with '\-'. *)
val escape : string -> string

(** Unescape slash characters in the given string with '\-'. *)
val unescape : string -> string

(** Sets the default deflate to be used when writing data. *)
val set_default_deflate : int -> unit

(** Returns the default deflate used when writing data. *)
val default_deflate : unit -> int

(** Sets whether the create HDF5 files are split by default.  See [open_]. *)
val set_default_split : bool -> unit

(** Returns whether the created HDF5 files are split by default.  See [open_]. *)
val default_split : unit -> bool

(** Represents a group inside an HDF5 file. *)
type t

(** Parameters for opening files.

    @param meta_block_size Size of the meta blocks.
    @param split           If set, all HDF5 files are split into a file which contains
                           only metadata and a file which contains only raw data.  Their
                           respective extensions are "-m.h5" and "-r.h5".  Since reading
                           metadata requires many small accesses to the file, this option
                           may provide a faster access on slow file systems. *)
type open_ = ?meta_block_size:int -> ?split:bool -> string -> t

(** Creates the named file and returns the root group. *)
val create_trunc : open_

(** Opens the named file for reading and returns the root group. *)
val open_rdonly : open_

(** Opens the named file for reading and writing and returns the root group. *)
val open_rdwr : open_

(** Opens the named subgroup.  The subgroup is created if it does not already exist. *)
val open_group : t -> string -> t

(** Opens the dataset with the given name. *)
val open_dataset : t -> string -> t

(** Closes the group handle.  If given a root group closes the file. *)
val close : t -> unit

(** Performs the given function on the given dir. *)
val with_group : t -> string -> (t -> 'a) -> 'a

(** Flushes all buffers associated with a file to disk. *)
val flush : t -> unit

(** Returns the name of file to which object belongs. *)
val get_name : t -> string

(** Returns whether the given subgroup or a data set exists. *)
val exists : t -> string -> bool

(** Deletes the given subgroup or a data set. *)
val delete : t -> string -> unit

(** Returns all subdirectories and data sets. *)
val ls : ?index:H5_raw.Index.t -> ?order:H5_raw.Iter_order.t -> t -> string list

(** Copies data. *)
val copy : src:t -> src_name:string -> dst:t -> dst_name:string -> unit

(** Merges the source into the destination.  Every group and dataset from the source is
    copied to the destination.  [on_duplicate] defines what is done in the case of a
    conflict.  The argument to [`Call] is the path of the conflicting dataset. *)
val merge : src:t -> dst:t
  -> on_duplicate:[`Skip | `Raise | `Call of (string list -> unit)] -> unit

(** Creates a hard link. *)
val create_hard_link : obj:t -> obj_name:string -> link:t -> link_name:string -> unit

(** Creates a soft link. *)
val create_soft_link : target_path:string -> link:t -> link_name:string -> unit

(** Creates an external link. *)
val create_external_link : t -> target_file_name:string -> target_obj_name:string
  -> link_name:string -> unit

(** Returns the HDF5 handle *)
val hid : t -> Hid.t

(** Construct H5.t from dataset HDF5 handle *)
val dataset : Hid.t -> t

(** Construct H5.t from file HDF5 handle *)
val file : Hid.t -> t

(** Construct H5.t from group HDF5 handle *)
val group : Hid.t -> t

(** Construct H5.t from group, file, or dataset HDF5 handle *)
val h5t : Hid.t -> t

(** Writes the given uint8 Array1.t to the data set. *)
val write_uint8_array1 : t -> string -> ?deflate:int
  -> (char, int8_unsigned_elt, _) Array1.t -> unit

(** Reads the data set into a uint8 Array1.t.

    @param data If provided, the storage for the data. *)
val read_uint8_array1 : t -> ?data:(char, int8_unsigned_elt, 'a) Array1.t -> string
  -> 'a layout -> (char, int8_unsigned_elt, 'a) Array1.t

(** Writes the given string to the data set. *)
val write_string : t -> string -> ?deflate:int -> string -> unit

(** Reads the data set into a string. *)
val read_string : t -> string -> string

(** Writes the given string array to the data set. *)
val write_string_array : t -> string -> ?deflate:int -> string array -> unit

(** Reads the data set into a string array. *)
val read_string_array : t -> string -> string array

(** Reads the data set into a bigstring array.  It is slightly faster than
    [read_string_array] because it does not do [memcpy] for each element. *)
val read_bigstring_array : t -> string
  -> (char, int8_unsigned_elt, c_layout) Array1.t array

(** [write_attribute_int64 t name v] writes the given [int64] as an attribute with the
    given name. *)
val write_attribute_int64 : t -> string -> int64 -> unit

(** [read_attribute_int64 t name] reads the attribute with the given name as a [int64]. *)
val read_attribute_int64 : t -> string -> int64

(** [write_attribute_string t name v] writes the given [string] as an attribute with the
    given name. *)
val write_attribute_string : t -> string -> string -> unit

(** [read_attribute_string t name] reads the attribute with the given name as a string. *)
val read_attribute_string : t -> string -> string

(** [write_attribute_string_array t name v] writes the given [string array] as an
    attribute with the given name. *)
val write_attribute_string_array : t -> string -> string array -> unit

(** [read_attribute_string_array t name] reads the attribute with the given name as a
    string array. *)
val read_attribute_string_array : t -> string -> string array

(** [write_attribute_c_string t name v] writes the given [C_string.t] as an attribute
    with the given name. *)
val write_attribute_c_string : t -> string -> C_string.t -> unit

(** [read_attribute_c_string t name] reads the attribute with the given name as a
    [C_string.t]. *)
val read_attribute_c_string : t -> string -> C_string.t

(** Returns whether the given attribute exists. *)
val attribute_exists : t -> string -> bool

(** Deletes the given attribute. *)
val delete_attribute : t -> string -> unit

include Float_intf.S with type t := t and type float_elt := float64_elt

module Float32 : Float_intf.S with type t := t and type float_elt := float32_elt
module Float64 : Float_intf.S with type t := t and type float_elt := float64_elt
