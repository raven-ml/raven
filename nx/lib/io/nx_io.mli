(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tensor I/O.

    Load and save {!Nx} tensors in common formats: images (PNG, JPEG, BMP, TGA),
    NumPy (.npy, .npz), SafeTensors, and delimited text.

    All functions raise [Failure] on errors. *)

(** {1:packed Packed tensors} *)

type packed =
  | P : ('a, 'b) Nx.t -> packed
      (** An existentially packed tensor. Use {!to_typed} to recover a typed
          tensor. *)

type archive = (string, packed) Hashtbl.t
(** Named tensors. Returned by {!load_npz} and {!load_safetensors}. *)

type packed_dtype =
  | Dtype : ('a, 'b) Nx.dtype -> packed_dtype
      (** An existentially packed dtype. *)

val to_typed : ('a, 'b) Nx.dtype -> packed -> ('a, 'b) Nx.t
(** [to_typed dtype p] is the tensor in [p] with [dtype].

    Raises [Failure] if the packed tensor has a different dtype. *)

val packed_dtype : packed -> packed_dtype
(** [packed_dtype p] is the dtype of [p]. *)

val packed_shape : packed -> int array
(** [packed_shape p] is the shape of [p]. *)

(** {1:image Images} *)

val load_image : ?grayscale:bool -> string -> (int, Nx.uint8_elt) Nx.t
(** [load_image ?grayscale path] loads an image as a uint8 tensor.

    [grayscale] defaults to [false]. Shape is [[h; w]] when [grayscale] is
    [true], [[h; w; c]] otherwise.

    Raises [Failure] on I/O or decoding errors. *)

val save_image : ?overwrite:bool -> string -> (int, Nx.uint8_elt) Nx.t -> unit
(** [save_image ?overwrite path t] writes [t] to [path].

    Format is inferred from extension (.png, .jpg, .bmp, .tga). Accepted shapes
    are [[h; w]], [[h; w; 1]], [[h; w; 3]], and [[h; w; 4]]. [overwrite]
    defaults to [true].

    Raises [Failure] on unsupported shape, extension, or I/O errors. *)

(** {1:numpy NumPy formats} *)

val load_npy : string -> packed
(** [load_npy path] loads a tensor from a [.npy] file.

    Raises [Failure] on I/O or format errors. *)

val save_npy : ?overwrite:bool -> string -> ('a, 'b) Nx.t -> unit
(** [save_npy ?overwrite path t] writes [t] to a [.npy] file.

    [overwrite] defaults to [true].

    Raises [Failure] on I/O errors. *)

val load_npz : string -> archive
(** [load_npz path] loads all tensors from an [.npz] archive.

    Raises [Failure] on I/O or format errors. *)

val load_npz_entry : name:string -> string -> packed
(** [load_npz_entry ~name path] loads a single entry from an [.npz] archive.

    Raises [Failure] if [name] is missing or the archive is invalid. *)

val save_npz : ?overwrite:bool -> string -> (string * packed) list -> unit
(** [save_npz ?overwrite path entries] writes named tensors to an [.npz]
    archive.

    [overwrite] defaults to [true].

    Raises [Failure] on I/O errors. *)

(** {1:safetensors SafeTensors} *)

val load_safetensors : string -> archive
(** [load_safetensors path] loads all tensors from a SafeTensors file.

    Raises [Failure] on I/O or format errors. *)

val save_safetensors :
  ?overwrite:bool -> string -> (string * packed) list -> unit
(** [save_safetensors ?overwrite path entries] writes named tensors to a
    SafeTensors file.

    [overwrite] defaults to [true].

    Raises [Failure] on I/O errors. *)

(** {1:text Text format} *)

val load_txt :
  ?sep:string ->
  ?comments:string ->
  ?skiprows:int ->
  ?max_rows:int ->
  string ->
  ('a, 'b) Nx.dtype ->
  ('a, 'b) Nx.t
(** [load_txt ?sep ?comments ?skiprows ?max_rows path dtype] parses delimited
    text into a tensor.

    [sep] defaults to [" "]. [comments] defaults to ["#"]. [skiprows] defaults
    to [0]. The result is 1D or 2D depending on parsed data.

    Raises [Failure] on I/O or parse errors. *)

val save_txt :
  ?sep:string ->
  ?append:bool ->
  ?newline:string ->
  ?header:string ->
  ?footer:string ->
  ?comments:string ->
  string ->
  ('a, 'b) Nx.t ->
  unit
(** [save_txt ?sep ?append ?newline ?header ?footer ?comments path t] writes a
    scalar, vector, or matrix tensor to delimited text.

    [sep] defaults to [" "]. [append] defaults to [false]. [newline] defaults to
    ["\n"]. [comments] defaults to ["# "].

    Raises [Failure] on unsupported dtype/shape or I/O errors. *)
