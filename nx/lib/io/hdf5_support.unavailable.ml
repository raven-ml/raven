(* HDF5 support is not available *)

open Utils

let hdf5_available = false

let load_h5_dataset ~path ~dataset:_ =
  fail_msg
    "H5 Load Error (%s): HDF5 support not available. Install HDF5 and rebuild."
    path

let save_h5_dataset _ ~path ~dataset:_ =
  fail_msg
    "H5 Save Error (%s): HDF5 support not available. Install HDF5 and rebuild."
    path

type h5_archive = (string, packed_nx) Hashtbl.t

let load_h5_all path =
  fail_msg
    "H5 Load Error (%s): HDF5 support not available. Install HDF5 and rebuild."
    path

let save_h5_all _ path =
  fail_msg
    "H5 Save Error (%s): HDF5 support not available. Install HDF5 and rebuild."
    path
