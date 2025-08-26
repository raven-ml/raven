(* HDF5 support is not available *)

let hdf5_available = false

type packed_nx_t = P : ('a, 'b) Nx.t -> packed_nx_t
type h5_archive = (string, packed_nx_t) Hashtbl.t

let load_h5_dataset ~dataset:_ path =
  Error
    (Error.Other
       (Printf.sprintf
          "H5 Load Error (%s): HDF5 support not available. Install HDF5 and rebuild."
          path))

let save_h5_dataset ~dataset:_ ?overwrite:_ _path _arr =
  Error
    (Error.Other
       "HDF5 support not available. Install HDF5 and rebuild.")

let load_h5_all path =
  Error
    (Error.Other
       (Printf.sprintf
          "H5 Load Error (%s): HDF5 support not available. Install HDF5 and rebuild."
          path))

let save_h5_all ?overwrite:_ _path _items =
  Error
    (Error.Other
       "HDF5 support not available. Install HDF5 and rebuild.")