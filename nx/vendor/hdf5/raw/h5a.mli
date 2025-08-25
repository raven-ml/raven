module Info : sig
  type t = {
    corder_valid : bool;
    corder       : H5o.Msg_crt_idx.t;
    cset         : H5t.Cset.t;
    data_size    : int }
end

module Iterate : sig
  type 'a t = Hid.t -> string -> Info.t -> 'a -> H5_raw.Iter.t
end

external create : Hid.t -> string -> Hid.t -> ?acpl:Hid.t -> ?aapl:Hid.t -> Hid.t -> Hid.t
  = "hdf5_h5a_create_bytecode" "hdf5_h5a_create"
external create_by_name : Hid.t -> string -> string -> Hid.t -> ?acpl:Hid.t
  -> ?aapl:Hid.t -> ?lapl:Hid.t -> Hid.t -> Hid.t
  = "hdf5_h5a_create_by_name_bytecode" "hdf5_h5a_create_by_name"
external open_ : Hid.t -> ?aapl:Hid.t -> string -> Hid.t = "hdf5_h5a_open"
external open_by_name : Hid.t -> ?aapl:Hid.t -> ?lapl:Hid.t -> string -> string -> Hid.t
  = "hdf5_h5a_open_by_name"
external open_name : Hid.t -> string -> Hid.t = "hdf5_h5a_open_name"
  [@@ocaml.deprecated "This function is deprecated in favor of the function \
    H5a.open_by_name"]
external open_by_idx : Hid.t -> ?aapl:Hid.t -> ?lapl:Hid.t -> H5_raw.Index.t
  -> H5_raw.Iter_order.t -> int -> Hid.t
  = "hdf5_h5a_open_by_idx_bytecode" "hdf5_h5a_open_by_idx"
external open_idx : Hid.t -> int -> Hid.t = "hdf5_h5a_open_idx"
  [@@ocaml.deprecated "This function is deprecated in favor of the function \
    H5a.open_by_idx"]
external exists : Hid.t -> string -> bool = "hdf5_h5a_exists"
external exists_by_name : Hid.t -> string -> ?lapl:Hid.t -> string -> bool
  = "hdf5_h5a_exists_by_name"
external rename : Hid.t -> string -> string -> unit = "hdf5_h5a_rename"
external rename_by_name : Hid.t -> string -> ?lapl:Hid.t -> string -> string -> unit
  = "hdf5_h5a_rename_by_name"
external write_bigarray : Hid.t -> Hid.t -> _ Bigarray.Genarray.t -> unit
  = "hdf5_h5a_write_bigarray"
external write_c_string : Hid.t -> Hid.t -> C_string.t -> unit = "hdf5_h5a_write_value"
external write_float : Hid.t -> Hid.t -> float-> unit = "hdf5_h5a_write"
external write_int32 : Hid.t -> Hid.t -> int64 -> unit = "hdf5_h5a_write_custom"
external write_int64 : Hid.t -> Hid.t -> int64 -> unit = "hdf5_h5a_write_custom"
external write_nativeint : Hid.t -> Hid.t -> nativeint -> unit = "hdf5_h5a_write_custom"
external write_string : Hid.t -> Hid.t -> string -> unit = "hdf5_h5a_write"
external write_float_array : Hid.t -> Hid.t -> float array -> unit = "hdf5_h5a_write"
external write_string_array : Hid.t -> Hid.t -> string array -> unit = "hdf5_h5a_write"
external read_bigarray : Hid.t -> Hid.t -> _ Bigarray.Genarray.t -> unit
  = "hdf5_h5a_read_bigarray"
external read_c_string : Hid.t -> Hid.t -> C_string.t = "hdf5_h5a_read_value"
external read_float : Hid.t -> Hid.t -> float = "hdf5_h5a_read_float"
external read_int32 : Hid.t -> Hid.t -> int64 = "hdf5_h5a_read_int32"
external read_int64 : Hid.t -> Hid.t -> int64 = "hdf5_h5a_read_int64"
external read_nativeint : Hid.t -> Hid.t -> nativeint = "hdf5_h5a_read_nativeint"
external read_string : Hid.t -> Hid.t -> bytes -> unit = "hdf5_h5a_read"
external read_float_array : Hid.t -> Hid.t -> float array -> unit = "hdf5_h5a_read"
external read_string_array : Hid.t -> Hid.t -> string array -> unit
  = "hdf5_h5a_read_string_array"
external close : Hid.t -> unit = "hdf5_h5a_close"
external iterate : Hid.t -> ?idx_type:H5_raw.Index.t -> ?iter_order:H5_raw.Iter_order.t
  -> ?n:int ref -> 'a Iterate.t -> 'a -> H5_raw.Iter.t
  = "hdf5_h5a_iterate_bytecode" "hdf5_h5a_iterate"
external iterate_by_name : Hid.t -> string -> ?idx_type:H5_raw.Index.t
  -> ?iter_order:H5_raw.Iter_order.t -> ?n:int ref -> ?lapd:Hid.t -> 'a Iterate.t -> 'a
  -> H5_raw.Iter.t = "hdf5_h5a_iterate_by_name_bytecode" "hdf5_h5a_iterate_by_name"
external delete : Hid.t -> string -> unit = "hdf5_h5a_delete"
external delete_by_name : Hid.t -> ?lapl:Hid.t -> string -> string -> unit
  = "hdf5_h5a_delete_by_name"
external delete_by_idx : Hid.t -> string -> ?idx_type:H5_raw.Index.t
  -> ?iter_order:H5_raw.Iter_order.t -> ?lapl:Hid.t -> int -> unit
  = "hdf5_h5a_delete_by_idx_bytecode" "hdf5_h5a_delete_by_idx"
external get_info : Hid.t -> Info.t = "hdf5_h5a_get_info"
external get_info_by_name : Hid.t -> string -> ?lapl:Hid.t -> string -> Info.t
  = "hdf5_h5a_get_info_by_name"
external get_info_by_idx : Hid.t -> string -> ?idx_type:H5_raw.Index.t
  -> ?order:H5_raw.Iter_order.t -> ?lapl:Hid.t -> int -> Info.t
  = "hdf5_h5a_get_info_by_idx_bytecode" "hdf5_h5a_get_info_by_idx"
external get_num_attrs : Hid.t -> int = "hdf5_h5a_get_num_attrs"
  [@@ocaml.deprecated "This function is deprecated in favor of the functions \
    H5o.get_info, H5o.get_info_by_name, and H5o.get_info_by_idx"]
external get_name : Hid.t -> string = "hdf5_h5a_get_name"
external get_create_plist : Hid.t -> Hid.t = "hdf5_h5a_get_create_plist"
external get_space : Hid.t -> Hid.t = "hdf5_h5a_get_space"
external get_type : Hid.t -> Hid.t = "hdf5_h5a_get_type"
external get_storage_size : Hid.t -> int = "hdf5_h5a_get_storage_size"
external get_name_by_idx : Hid.t -> string -> ?idx_type:H5_raw.Index.t
  -> ?order:H5_raw.Iter_order.t -> ?lapl:Hid.t -> int -> string
  = "hdf5_h5a_get_name_by_idx_bytecode" "hdf5_h5a_get_name_by_idx"
