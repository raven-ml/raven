(** Nx_io: Nx input/output for common file formats.

    Provides functions to load and save [Nx.t] arrays in image formats, NumPy
    formats (.npy, .npz), HDF5 archives, and SafeTensors. Emphasizes safety with
    result types for error handling and labeled arguments for clarity. *)

module Cache_dir = Cache_dir

(** {1 Types} *)

(** Existential container for an [Nx.t] of any dtype and dimensionality.

    Wraps arrays whose element type and layout are determined at runtime, such
    as those loaded from files. *)
type packed_nx = P : ('a, 'b) Nx.t -> packed_nx

type archive = (string, packed_nx) Hashtbl.t
(** Generic archive type for mappings from names/paths to packed arrays, used
    across NPZ, HDF5, and SafeTensors. *)

(** {1 Conversions from Packed Arrays} *)

val as_float16 : packed_nx -> Nx.float16_t
(** [as_float16 packed] converts a packed Nx to a [Nx.float16_t], or [Error] if
    dtype mismatch. *)

val as_bfloat16 : packed_nx -> Nx.bfloat16_t
(** [as_bfloat16 packed] converts a packed Nx to a [Nx.bfloat16_t], or [Error]
    if dtype mismatch. *)

val as_float32 : packed_nx -> Nx.float32_t
(** [as_float32 packed] converts a packed Nx to a [Nx.float32_t], or [Error] if
    dtype mismatch. *)

val as_float64 : packed_nx -> Nx.float64_t
(** [as_float64 packed] converts a packed Nx to a [Nx.float64_t], or [Error] if
    dtype mismatch. *)

val as_int8 : packed_nx -> Nx.int8_t
(** [as_int8 packed] converts a packed Nx to a [Nx.int8_t], or [Error] if dtype
    mismatch. *)

val as_int16 : packed_nx -> Nx.int16_t
(** [as_int16 packed] converts a packed Nx to a [Nx.int16_t], or [Error] if
    dtype mismatch. *)

val as_int32 : packed_nx -> Nx.int32_t
(** [as_int32 packed] converts a packed Nx to a [Nx.int32_t], or [Error] if
    dtype mismatch. *)

val as_int64 : packed_nx -> Nx.int64_t
(** [as_int64 packed] converts a packed Nx to a [Nx.int64_t], or [Error] if
    dtype mismatch. *)

val as_uint8 : packed_nx -> Nx.uint8_t
(** [as_uint8 packed] converts a packed Nx to a [Nx.uint8_t], or [Error] if
    dtype mismatch. *)

val as_uint16 : packed_nx -> Nx.uint16_t
(** [as_uint16 packed] converts a packed Nx to a [Nx.uint16_t], or [Error] if
    dtype mismatch. *)

val as_bool : packed_nx -> Nx.bool_t
(** [as_bool packed] converts a packed Nx to a [Nx.bool_t], or [Error] if dtype
    mismatch. *)

val as_complex32 : packed_nx -> Nx.complex32_t
(** [as_complex32 packed] converts a packed Nx to a [Nx.complex32_t], or [Error]
    if dtype mismatch. *)

val as_complex64 : packed_nx -> Nx.complex64_t
(** [as_complex64 packed] converts a packed Nx to a [Nx.complex64_t], or [Error]
    if dtype mismatch. *)

(** {1 Image Loading and Saving} *)

val load_image : ?grayscale:bool -> string -> (int, Nx.uint8_elt) Nx.t
(** [load_image ?grayscale path]

    Load an image file into a uint8 nx.

    {2 Parameters}
    - ?grayscale: if [true], load as grayscale (2D) else color (3D RGB); default
      [false].
    - path: path to the image file.

    {2 Returns}
    - [Ok] with uint8 nx of shape [|height; width|] if grayscale, or
      [|height; width; 3|] for RGB color images.
    - [Error] on failure (e.g., I/O or unsupported format).

    {2 Notes}
    - Supported formats depend on stb_image (PNG, JPEG, BMP, TGA, GIF).
    - Alpha channels are discarded.
    - Pixel values range [0, 255].

    {2 Examples}
    {[
      (* Load color image *)
      match load_image "photo.png" with
      | Ok img -> (* img : (int, Nx.uint8_elt) Nx.t with shape [|H; W; 3|] *)
      | Error e -> failwith (string_of_error e)

      (* Load as grayscale *)
      let gray = load_image ~grayscale:true "photo.png"
    ]} *)

val save_image : ?overwrite:bool -> string -> (int, Nx.uint8_elt) Nx.t -> unit
(** [save_image ?overwrite path img]

    Save a uint8 nx as an image file.

    {2 Parameters}
    - ?overwrite: if [false], fail if file exists; default [true].
    - path: destination file path; extension determines format (e.g. .png,
      .jpg).
    - img: uint8 nx of shape [|height; width|] (grayscale),
      [|height; width; 1|], [|height; width; 3|] (RGB), or [|height; width; 4|]
      (RGBA).

    {2 Returns}
    - [Ok ()] on success, [Error] on failure (e.g., unsupported shape or I/O).

    {2 Notes}
    - Supported formats depend on stb_image_write library.
    - Pixel values are written in [0, 255] range.

    {2 Examples}
    {[
      (* Save a grayscale image *)
      let gray = Nx.create Nx.uint8 [|100; 200|] data in
      ignore (save_image ~img:gray "output.png")

      (* Save an RGB image without overwriting *)
      let rgb = Nx.create Nx.uint8 [|100; 200; 3|] data in
      save_image ~img:rgb "output.jpg" ~overwrite:false
    ]} *)

(** {1 NumPy Format (.npy)} *)

val load_npy : string -> packed_nx
(** [load_npy path]

    Load a single nx from a NumPy `.npy` file.

    {2 Parameters}
    - path: path to the `.npy` file.

    {2 Returns}
    - [Ok] with [packed_nx] wrapping the loaded array.
    - [Error] on failure.

    {2 Examples}
    {[
      (* Load and convert to specific type *)
      match load_npy "data.npy" with
      | Ok packed -> let arr = as_float32 packed in ...
      | Error _ -> ...
    ]} *)

val save_npy : ?overwrite:bool -> string -> ('a, 'b) Nx.t -> unit
(** [save_npy ~arr path ?overwrite]

    Save a single nx to a NumPy `.npy` file.

    {2 Parameters}
    - ~arr: nx to save (any supported dtype).
    - path: destination path for the `.npy` file.
    - ?overwrite: if [false], fail if file exists; default [true].

    {2 Returns}
    - [Ok ()] on success, [Error] on failure.

    {2 Notes}
    - The file encodes dtype, shape, and raw data in little-endian order. *)

(** {1 NumPy Archive Format (.npz)} *)

val load_npz : string -> archive
(** [load_npz path]

    Load all arrays from a NumPy `.npz` archive into a hash table.

    {2 Returns}
    - [Ok] with [archive] mapping each array name to its [packed_nx]. *)

val load_npz_member : name:string -> string -> packed_nx
(** [load_npz_member ~name path]

    Load a single named array from a NumPy `.npz` archive.

    {2 Returns}
    - [Ok] with [packed_nx] or [Error] (e.g., [Missing_entry]). *)

val save_npz : ?overwrite:bool -> string -> (string * packed_nx) list -> unit
(** [save_npz ?overwrite path items]

    Save multiple named nxs to a NumPy `.npz` archive. *)

(** {1 Text Format} *)

val save_txt :
  ?sep:string ->
  ?append:bool ->
  ?newline:string ->
  ?header:string ->
  ?footer:string ->
  ?comments:string ->
  out:string ->
  ('a, 'b) Nx.t ->
  unit
(** [save_txt ?sep ?append ?newline ?header ?footer ?comments ~out t]

    Save a scalar, 1D, or 2D tensor to a text file. Each row is written on a new
    line, values separated by [sep] (default: a single space). If [append] is
    [true], data is appended to [out]; otherwise the file is truncated/created.
    Optional [header] and [footer] strings are emitted before and after the
    data, prefixed with [comments] (default: ["# "]). The [newline] argument
    controls the end-of-line separator. Only numeric and boolean dtypes are
    supported. *)

val load_txt :
  ?sep:string ->
  ?comments:string ->
  ?skiprows:int ->
  ?max_rows:int ->
  ('a, 'b) Nx.dtype ->
  string ->
  ('a, 'b) Nx.t
(** [load_txt ?sep ?comments ?skiprows ?max_rows dtype path]

    Load a tensor from a text file previously written by [save_txt] or a
    compatible format. Lines beginning with [comments] (after trimming leading
    whitespace) are ignored. Leading [skiprows] raw lines are skipped. When data
    has a single row or a single column, the result is 1D; otherwise a 2D tensor
    of shape [|rows; cols|] is returned. *)

(** {1 SafeTensors Format} *)

val load_safetensor : string -> archive
(** [load_safetensor path]

    Load all tensors from a SafeTensors file into a hash table.. *)

val save_safetensor :
  ?overwrite:bool -> string -> (string * packed_nx) list -> unit
(** [save_safetensor ?overwrite path items]

    Save multiple named nxs to a SafeTensors file. *)

module Safe : sig
  type error =
    | Io_error of string
    | Format_error of string
    | Unsupported_dtype
    | Unsupported_shape
    | Missing_entry of string
    | Other of string  (** Custom error variants for I/O operations. *)

  (** {2 Conversions from Packed Arrays} *)

  val as_float16 : packed_nx -> (Nx.float16_t, error) result
  (** Safe alias for [as_float16] *)

  val as_bfloat16 : packed_nx -> (Nx.bfloat16_t, error) result
  (** Safe alias for [as_bfloat16] *)

  val as_float32 : packed_nx -> (Nx.float32_t, error) result
  (** Safe alias for [as_float32] *)

  val as_float64 : packed_nx -> (Nx.float64_t, error) result
  (** Safe alias for [as_float64] *)

  val as_int8 : packed_nx -> (Nx.int8_t, error) result
  (** Safe alias for [as_int8] *)

  val as_int16 : packed_nx -> (Nx.int16_t, error) result
  (** Safe alias for [as_int16] *)

  val as_int32 : packed_nx -> (Nx.int32_t, error) result
  (** Safe alias for [as_int32] *)

  val as_int64 : packed_nx -> (Nx.int64_t, error) result
  (** Safe alias for [as_int64] *)

  val as_uint8 : packed_nx -> (Nx.uint8_t, error) result
  (** Safe alias for [as_uint8] *)

  val as_uint16 : packed_nx -> (Nx.uint16_t, error) result
  (** Safe alias for [as_uint16] *)

  val as_bool : packed_nx -> (Nx.bool_t, error) result
  (** Safe alias for [as_bool] *)

  val as_complex32 : packed_nx -> (Nx.complex32_t, error) result
  (** Safe alias for [as_complex32] *)

  val as_complex64 : packed_nx -> (Nx.complex64_t, error) result
  (** Safe alias for [as_complex64] *)

  (** {2 Image Loading and Saving} *)

  val load_image :
    ?grayscale:bool -> string -> ((int, Nx.uint8_elt) Nx.t, error) result
  (** Safe alias for [load_image] *)

  val save_image :
    ?overwrite:bool ->
    string ->
    (int, Nx.uint8_elt) Nx.t ->
    (unit, error) result
  (** Safe alias for [save_image] *)

  (** {2 NumPy Format (.npy)} *)

  val load_npy : string -> (packed_nx, error) result
  (** Safe alias for [load_npy] *)

  val save_npy :
    ?overwrite:bool -> string -> ('a, 'b) Nx.t -> (unit, error) result
  (** Safe alias for [save_npy] *)

  (** {2 NumPy Archive Format (.npz)} *)

  val load_npz : string -> (archive, error) result
  (** Safe alias for [load_npz] *)

  val load_npz_member : name:string -> string -> (packed_nx, error) result
  (** Safe alias for [load_npz_member] *)

  val save_npz :
    ?overwrite:bool ->
    string ->
    (string * packed_nx) list ->
    (unit, error) result
  (** Safe alias for [save_npz] *)

  (** {2 Text Format} *)

  val save_txt :
    ?sep:string ->
    ?append:bool ->
    ?newline:string ->
    ?header:string ->
    ?footer:string ->
    ?comments:string ->
    out:string ->
    ('a, 'b) Nx.t ->
    (unit, error) result
  (** Safe alias for [save_txt] *)

  val load_txt :
    ?sep:string ->
    ?comments:string ->
    ?skiprows:int ->
    ?max_rows:int ->
    ('a, 'b) Nx.dtype ->
    string ->
    (('a, 'b) Nx.t, error) result
  (** Safe alias for [load_txt] *)

  (** {2 SafeTensors Format} *)

  val load_safetensor : string -> (archive, error) result
  (** Safe alias for [load_safetensor] *)

  val save_safetensor :
    ?overwrite:bool ->
    string ->
    (string * packed_nx) list ->
    (unit, error) result
  (** Safe alias for [save_safetensor] *)
end
