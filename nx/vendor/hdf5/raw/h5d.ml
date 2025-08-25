open Bigarray

module Layout = struct
  type t =
  | COMPACT
  | CONTIGUOUS
  | CHUNKED
  | NLAYOUTS
end

module Alloc_time = struct
  type t =
  | DEFAULT
  | EARLY
  | LATE
  | INCR
end

module Space_status = struct
  type t =
  | NOT_ALLOCATED
  | PART_ALLOCATED
  | ALLOCATED
end

module Fill_time = struct
  type t =
  | ALLOC
  | NEVER
  | IFSET
end

module Fill_value = struct
  type t =
  | UNDEFINED
  | DEFAULT
  | USER_DEFAULT
end

external create : Hid.t -> string -> Hid.t -> ?lcpl:Hid.t -> ?dcpl:Hid.t -> ?apl: Hid.t
  -> Hid.t -> Hid.t = "hdf5_h5d_create_bytecode" "hdf5_h5d_create"
external create_anon : Hid.t -> Hid.t -> ?dcpl:Hid.t -> ?apl: Hid.t -> Hid.t -> Hid.t
  = "hdf5_h5d_create_anon"
external open_ : Hid.t -> ?dapl:Hid.t -> string -> Hid.t = "hdf5_h5d_open"
external close : Hid.t -> unit = "hdf5_h5d_close"
external get_space : Hid.t -> Hid.t = "hdf5_h5d_get_space"
external get_space_status : Hid.t -> Space_status.t = "hdf5_h5d_get_space_status"
external get_type : Hid.t -> Hid.t = "hdf5_h5d_get_type"
external get_create_plist : Hid.t -> Hid.t = "hdf5_h5d_get_create_plist"
external read_string : Hid.t -> Hid.t -> Hid.t -> Hid.t -> ?xfer_plist:Hid.t
  -> string -> unit = "hdf5_h5d_read_bytecode" "hdf5_h5d_read"
external read_float_array : Hid.t -> Hid.t -> Hid.t -> Hid.t -> ?xfer_plist:Hid.t
  -> float array -> unit = "hdf5_h5d_read_bytecode" "hdf5_h5d_read"
external read_c_string_array : Hid.t -> Hid.t -> Hid.t -> Hid.t -> ?xfer_plist:Hid.t
  -> C_string.t array -> unit = "hdf5_h5d_read_bytecode" "hdf5_h5d_read"
external read_bigarray : Hid.t -> Hid.t -> Hid.t -> Hid.t -> ?xfer_plist:Hid.t
  -> _ Genarray.t -> unit = "hdf5_h5d_read_bigarray_bytecode" "hdf5_h5d_read_bigarray"
external write_string : Hid.t -> Hid.t -> Hid.t -> Hid.t -> ?xfer_plist:Hid.t
  -> string -> unit = "hdf5_h5d_write_bytecode" "hdf5_h5d_write"
external write_float_array : Hid.t -> Hid.t -> Hid.t -> Hid.t -> ?xfer_plist:Hid.t
  -> float array -> unit = "hdf5_h5d_write_bytecode" "hdf5_h5d_write"
external write_string_array : Hid.t -> Hid.t -> Hid.t -> Hid.t -> ?xfer_plist:Hid.t
  -> string array -> unit = "hdf5_h5d_write_bytecode" "hdf5_h5d_write"
external write_bigarray : Hid.t -> Hid.t -> Hid.t -> Hid.t -> ?xfer_plist:Hid.t
  -> _ Genarray.t -> unit = "hdf5_h5d_write_bigarray_bytecode" "hdf5_h5d_write_bigarray"
external extend : Hid.t -> Hsize.t array -> unit = "hdf5_h5d_set_extent"
external set_extent : Hid.t -> Hsize.t array -> unit = "hdf5_h5d_set_extent"
external vlen_reclaim_c_string_array : Hid.t -> Hid.t -> Hid.t -> C_string.t array -> unit
  = "hdf5_h5d_vlen_reclaim"
