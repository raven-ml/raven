module Copy = struct
  type t =
  | SHALLOW_HIERARCHY_FLAG
  | EXPAND_SOFT_LINK_FLAG
  | EXPAND_EXT_LINK_FLAG
  | EXPAND_REFERENCE_FLAG
  | WITHOUT_ATTR_FLAG
  | PRESERVE_NULL_FLAG
  | ALL
end

module Type = struct
  type t =
  | GROUP
  | DATASET
  | NAMED_DATATYPE
  | NTYPES
end

module Hdr_info = struct
  module Space = struct
    type t = {
      total : int;
      meta  : int;
      mesg  : int;
      free  : int }
  end

  module Mesg = struct
    type t = {
      present : int;
      shared  : int }
  end

  type t = {
    version : int;
    nmesgs  : int;
    nchunks : int;
    flags   : int;
    space   : Space.t;
    mesg    : Mesg.t }
end

module Info = struct
  module Meta_size = struct
    type t = {
      obj : H5_raw.Ih_info.t;
      attr : H5_raw.Ih_info.t }
  end

  type t = {
    fileno    : int;
    addr      : H5_raw.Addr.t;
    type_     : Type.t;
    rc        : int;
    atime     : H5_raw.Time.t;
    mtime     : H5_raw.Time.t;
    ctime     : H5_raw.Time.t;
    btime     : H5_raw.Time.t;
    num_attrs : int;
    hdr       : Hdr_info.t;
    meta_size : Meta_size.t }
end

module Msg_crt_idx = struct
  type t = int
end

external open_ : Hid.t -> ?apl:Hid.t -> string -> Hid.t = "hdf5_h5o_open"
external close : Hid.t -> unit = "hdf5_h5o_close"
external copy : Hid.t -> string -> Hid.t -> ?ocpypl:Hid.t -> ?lcpl:Hid.t -> string -> unit
  = "hdf5_h5o_copy_bytecode" "hdf5_h5o_copy"
external set_comment : Hid.t -> string -> unit = "hdf5_h5o_set_comment"
external get_info : Hid.t -> Info.t = "hdf5_h5o_get_info"
external get_info_by_name : Hid.t -> ?lapl:Hid.t -> string -> Info.t
  = "hdf5_h5o_get_info_by_name"
