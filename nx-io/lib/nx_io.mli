(** nx-io: Nx input/output for common file formats.

    Provide functions to load and save [Nx.t] arrays in image formats, NumPy
    formats (.npy, .npz), and HDF5 archives. *)

(** Existential container for an [Nx.t] of any dtype and dimensionality.

    Wrap arrays whose element type and layout are determined at runtime, such as
    those loaded from .npy or .npz files. *)
type packed_nx = P : ('a, 'b) Nx.t -> packed_nx

(** {1 Image Loading and Saving} *)

val load_image : ?grayscale:bool -> string -> Nx.uint8_t
(** [load_image ?grayscale path]

    Load an image file into a uint8 nx.

    {2 Parameters}
    - ?grayscale: if [true], load as grayscale (2D) else color (3D RGB); default
      [false].
    - path: path to the image file.

    {2 Returns}
    - uint8 nx of shape [|height; width|] if grayscale, or [|height; width; 3|]
      for RGB color images.

    {2 Raises}
    - [Failure] if file I/O fails or format is unsupported.

    {2 Notes}
    - Supported formats depend on stb_image (PNG, JPEG, BMP, TGA, GIF).
    - Alpha channels are discarded.
    - Pixel values range [0, 255]. *)

val save_image : Nx.uint8_t -> string -> unit
(** [save_image img path]

    Save a uint8 nx as an image file.

    {2 Parameters}
    - img: uint8 nx of shape [|height; width|] (grayscale),
      [|height; width; 1|], [|height; width; 3|] (RGB), or [|height; width; 4|]
      (RGBA).
    - path: destination file path; extension determines format (e.g. .png,
      .jpg).

    {2 Raises}
    - [Failure] if nx is not uint8 kind or has unsupported dimensions.
    - [Failure] on I/O error.

    {2 Notes}
    - Supported formats depend on stb_image_write library.
    - Pixel values are written in [0, 255] range. *)

(** {1 NumPy Format (.npy)} *)

val load_npy : string -> packed_nx
(** [load_npy path]

    Load a single nx from a NumPy `.npy` file.

    {2 Parameters}
    - path: path to the `.npy` file.

    {2 Returns}
    - [packed_nx] wrapping the loaded array with its runtime-detected dtype.

    {2 Raises}
    - [Failure] if file I/O fails or format is invalid.

    {2 Examples}
    {[
      let P arr = load_npy "data.npy" in
      (* arr : ('a, 'b) Nx.t *)
    ]} *)

val save_npy : ('a, 'b) Nx.t -> string -> unit
(** [save_npy arr path]

    Save a single nx to a NumPy `.npy` file.

    {2 Parameters}
    - arr: nx to save (any supported dtype).
    - path: destination path for the `.npy` file.

    {2 Raises}
    - [Failure] on I/O error or unsupported dtype.

    {2 Notes}
    - The file encodes dtype, shape, and raw data in little-endian order. *)

(** {1 NumPy Archive Format (.npz)} *)

type npz_archive = (string, packed_nx) Hashtbl.t
(** Map from array names to [packed_nx] values for a NumPy `.npz` archive. *)

val load_npz : string -> npz_archive
(** [load_npz path]

    Load all arrays from a NumPy `.npz` archive into a hash table.

    {2 Parameters}
    - path: path to the `.npz` file.

    {2 Returns}
    - [npz_archive] mapping each array name to its [packed_nx].

    {2 Raises}
    - [Failure] on I/O error or invalid archive format. *)

val load_npz_member : path:string -> name:string -> packed_nx
(** [load_npz_member ~path ~name]

    Load a single named array from a NumPy `.npz` archive.

    {2 Parameters}
    - path: path to the `.npz` file.
    - name: name of the array entry in the archive.

    {2 Returns}
    - [packed_nx] containing the loaded array.

    {2 Raises}
    - [Failure] if entry [name] is not found or on I/O error. *)

val save_npz : (string * packed_nx) list -> string -> unit
(** [save_npz items path]

    Save multiple named nxs to a NumPy `.npz` archive.

    {2 Parameters}
    - items: list of (name, array) pairs to include in the archive.
    - path: destination path for the `.npz` file.

    {2 Raises}
    - [Failure] on I/O error or archive creation failure. *)

(** {1 Conversions from Packed Arrays} *)

val to_float16 : packed_nx -> Nx.float16_t
(** [to_float16 packed_nx] converts a packed Nx to a [Nx.float16_t]. *)

val to_float32 : packed_nx -> Nx.float32_t
(** [to_float32 packed_nx] converts a packed Nx to a [Nx.float32_t]. *)

val to_float64 : packed_nx -> Nx.float64_t
(** [to_float64 packed_nx] converts a packed Nx to a [Nx.float64_t]. *)

val to_int8 : packed_nx -> Nx.int8_t
(** [to_int8 packed_nx] converts a packed Nx to a [Nx.int8_t]. *)

val to_int16 : packed_nx -> Nx.int16_t
(** [to_int16 packed_nx] converts a packed Nx to a [Nx.int16_t]. *)

val to_int32 : packed_nx -> Nx.int32_t
(** [to_int32 packed_nx] converts a packed Nx to a [Nx.int32_t]. *)

val to_int64 : packed_nx -> Nx.int64_t
(** [to_int64 packed_nx] converts a packed Nx to a [Nx.int64_t]. *)

val to_uint8 : packed_nx -> Nx.uint8_t
(** [to_uint8 packed_nx] converts a packed Nx to a [Nx.uint8_t]. *)

val to_uint16 : packed_nx -> Nx.uint16_t
(** [to_uint16 packed_nx] converts a packed Nx to a [Nx.uint16_t]. *)

val to_complex32 : packed_nx -> Nx.complex32_t
(** [to_complex32 packed_nx] converts a packed Nx to a [Nx.complex32_t]. *)

val to_complex64 : packed_nx -> Nx.complex64_t
(** [to_complex64 packed_nx] converts a packed Nx to a [Nx.complex64_t]. *)
