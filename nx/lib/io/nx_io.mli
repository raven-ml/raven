(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Input/output helpers for [Nx] tensors.

    The top-level functions fail on errors by raising [Failure].

    The {!module-Safe} submodule exposes the same operations with explicit
    [('a, error) result] values. *)

module Cache_dir = Cache_dir
module Http = Http

(** {1:types Common types} *)

type packed_nx =
  | P : ('a, 'b) Nx.t -> packed_nx  (** Existentially packed tensor value. *)

type archive = (string, packed_nx) Hashtbl.t
(** Mapping from entry names to packed tensors. *)

(** {1:conversions Packed tensor conversions} *)

val as_float16 : packed_nx -> Nx.float16_t
(** [as_float16 p] converts [p] to [Nx.float16_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_bfloat16 : packed_nx -> Nx.bfloat16_t
(** [as_bfloat16 p] converts [p] to [Nx.bfloat16_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_float32 : packed_nx -> Nx.float32_t
(** [as_float32 p] converts [p] to [Nx.float32_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_float64 : packed_nx -> Nx.float64_t
(** [as_float64 p] converts [p] to [Nx.float64_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_int8 : packed_nx -> Nx.int8_t
(** [as_int8 p] converts [p] to [Nx.int8_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_int16 : packed_nx -> Nx.int16_t
(** [as_int16 p] converts [p] to [Nx.int16_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_int32 : packed_nx -> Nx.int32_t
(** [as_int32 p] converts [p] to [Nx.int32_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_int64 : packed_nx -> Nx.int64_t
(** [as_int64 p] converts [p] to [Nx.int64_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_uint8 : packed_nx -> Nx.uint8_t
(** [as_uint8 p] converts [p] to [Nx.uint8_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_uint16 : packed_nx -> Nx.uint16_t
(** [as_uint16 p] converts [p] to [Nx.uint16_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_bool : packed_nx -> Nx.bool_t
(** [as_bool p] converts [p] to [Nx.bool_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_complex32 : packed_nx -> Nx.complex64_t
(** [as_complex32 p] converts [p] to [Nx.complex64_t].

    Raises [Failure] if [p] has a different dtype. *)

val as_complex64 : packed_nx -> Nx.complex128_t
(** [as_complex64 p] converts [p] to [Nx.complex128_t].

    Raises [Failure] if [p] has a different dtype. *)

(** {1:image Images} *)

val load_image : ?grayscale:bool -> string -> (int, Nx.uint8_elt) Nx.t
(** [load_image ?grayscale path] loads an image file as a uint8 tensor.

    If [grayscale = true], the result shape is [|h; w|]. Otherwise, the result
    shape is [|h; w; c|] (typically [c = 3]).

    Raises [Failure] on decoding or I/O errors. *)

val save_image : ?overwrite:bool -> string -> (int, Nx.uint8_elt) Nx.t -> unit
(** [save_image ?overwrite path img] writes [img] to [path].

    Accepted shapes are [|h; w|], [|h; w; 1|], [|h; w; 3|], and [|h; w; 4|].

    [overwrite] defaults to [true].

    Raises [Failure] on unsupported shape, unsupported extension, or I/O errors.
*)

(** {1:numpy NumPy formats} *)

val load_npy : string -> packed_nx
(** [load_npy path] loads a single tensor from a [.npy] file.

    Raises [Failure] on parse or I/O errors. *)

val save_npy : ?overwrite:bool -> string -> ('a, 'b) Nx.t -> unit
(** [save_npy ?overwrite path t] writes [t] to a [.npy] file.

    [overwrite] defaults to [true].

    Raises [Failure] on serialization or I/O errors. *)

val load_npz : string -> archive
(** [load_npz path] loads all tensors from an [.npz] archive.

    Raises [Failure] on parse or I/O errors. *)

val load_npz_member : name:string -> string -> packed_nx
(** [load_npz_member ~name path] loads a single member [name] from an [.npz]
    archive.

    Raises [Failure] if [name] is missing or the archive is invalid. *)

val save_npz : ?overwrite:bool -> string -> (string * packed_nx) list -> unit
(** [save_npz ?overwrite path items] writes named tensors to an [.npz] archive.

    [overwrite] defaults to [true].

    Raises [Failure] on serialization or I/O errors. *)

(** {1:text Text format} *)

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
(** [save_txt ?sep ?append ?newline ?header ?footer ?comments ~out t] serializes
    a scalar, vector, or matrix tensor to text.

    Raises [Failure] on unsupported dtype/shape or I/O errors. *)

val load_txt :
  ?sep:string ->
  ?comments:string ->
  ?skiprows:int ->
  ?max_rows:int ->
  ('a, 'b) Nx.dtype ->
  string ->
  ('a, 'b) Nx.t
(** [load_txt ?sep ?comments ?skiprows ?max_rows dtype path] parses text into a
    tensor of [dtype].

    The result is 1D or 2D depending on parsed data.

    Raises [Failure] on parse or I/O errors. *)

(** {1:safetensors SafeTensors format} *)

val load_safetensor : string -> archive
(** [load_safetensor path] loads all tensors from a SafeTensors file.

    Raises [Failure] on parse or I/O errors. *)

val save_safetensor :
  ?overwrite:bool -> string -> (string * packed_nx) list -> unit
(** [save_safetensor ?overwrite path items] writes named tensors to a
    SafeTensors file.

    [overwrite] defaults to [true].

    Raises [Failure] on serialization or I/O errors. *)

(** {1:safe Result-based API} *)

module Safe : sig
  type error =
    | Io_error of string
    | Format_error of string
    | Unsupported_dtype
    | Unsupported_shape
    | Missing_entry of string
    | Other of string  (** The type for I/O operation errors. *)

  (** {2:conversions Packed tensor conversions} *)

  val as_float16 : packed_nx -> (Nx.float16_t, error) result
  (** Result-based variant of {!val-as_float16}. *)

  val as_bfloat16 : packed_nx -> (Nx.bfloat16_t, error) result
  (** Result-based variant of {!val-as_bfloat16}. *)

  val as_float32 : packed_nx -> (Nx.float32_t, error) result
  (** Result-based variant of {!val-as_float32}. *)

  val as_float64 : packed_nx -> (Nx.float64_t, error) result
  (** Result-based variant of {!val-as_float64}. *)

  val as_int8 : packed_nx -> (Nx.int8_t, error) result
  (** Result-based variant of {!val-as_int8}. *)

  val as_int16 : packed_nx -> (Nx.int16_t, error) result
  (** Result-based variant of {!val-as_int16}. *)

  val as_int32 : packed_nx -> (Nx.int32_t, error) result
  (** Result-based variant of {!val-as_int32}. *)

  val as_int64 : packed_nx -> (Nx.int64_t, error) result
  (** Result-based variant of {!val-as_int64}. *)

  val as_uint8 : packed_nx -> (Nx.uint8_t, error) result
  (** Result-based variant of {!val-as_uint8}. *)

  val as_uint16 : packed_nx -> (Nx.uint16_t, error) result
  (** Result-based variant of {!val-as_uint16}. *)

  val as_bool : packed_nx -> (Nx.bool_t, error) result
  (** Result-based variant of {!val-as_bool}. *)

  val as_complex32 : packed_nx -> (Nx.complex64_t, error) result
  (** Result-based variant of {!val-as_complex32}. *)

  val as_complex64 : packed_nx -> (Nx.complex128_t, error) result
  (** Result-based variant of {!val-as_complex64}. *)

  (** {2:image Images} *)

  val load_image :
    ?grayscale:bool -> string -> ((int, Nx.uint8_elt) Nx.t, error) result
  (** Result-based variant of {!val-load_image}. *)

  val save_image :
    ?overwrite:bool ->
    string ->
    (int, Nx.uint8_elt) Nx.t ->
    (unit, error) result
  (** Result-based variant of {!val-save_image}. *)

  (** {2:numpy NumPy formats} *)

  val load_npy : string -> (packed_nx, error) result
  (** Result-based variant of {!val-load_npy}. *)

  val save_npy :
    ?overwrite:bool -> string -> ('a, 'b) Nx.t -> (unit, error) result
  (** Result-based variant of {!val-save_npy}. *)

  val load_npz : string -> (archive, error) result
  (** Result-based variant of {!val-load_npz}. *)

  val load_npz_member : name:string -> string -> (packed_nx, error) result
  (** Result-based variant of {!val-load_npz_member}. *)

  val save_npz :
    ?overwrite:bool ->
    string ->
    (string * packed_nx) list ->
    (unit, error) result
  (** Result-based variant of {!val-save_npz}. *)

  (** {2:text Text format} *)

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
  (** Result-based variant of {!val-save_txt}. *)

  val load_txt :
    ?sep:string ->
    ?comments:string ->
    ?skiprows:int ->
    ?max_rows:int ->
    ('a, 'b) Nx.dtype ->
    string ->
    (('a, 'b) Nx.t, error) result
  (** Result-based variant of {!val-load_txt}. *)

  (** {2:safetensors SafeTensors format} *)

  val load_safetensor : string -> (archive, error) result
  (** Result-based variant of {!val-load_safetensor}. *)

  val save_safetensor :
    ?overwrite:bool ->
    string ->
    (string * packed_nx) list ->
    (unit, error) result
  (** Result-based variant of {!val-save_safetensor}. *)
end
