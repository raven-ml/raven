(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tensor I/O.

    Load and save {!Nx} tensors in common formats: images (PNG and JPEG), NumPy
    ([.npy] and [.npz]), SafeTensors, and delimited text. *)

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

    @raise Failure if the packed tensor has a different dtype. *)

val packed_dtype : packed -> packed_dtype
(** [packed_dtype p] is the dtype of [p]. *)

val packed_shape : packed -> int array
(** [packed_shape p] is the shape of [p]. *)

(** {1:image Images} *)

val load_image : ?grayscale:bool -> string -> (int, Nx.uint8_elt) Nx.t
(** [load_image ?grayscale path] loads an image as a uint8 tensor.

    The file contents, rather than the extension, determine whether the image is
    PNG or JPEG. [grayscale] defaults to [false]. The result has shape
    [[|height; width|]] when [grayscale] is [true] and [[|height; width; 3|]]
    otherwise.

    @raise Failure if the stream is malformed or is neither PNG nor JPEG.
    @raise Unix.Unix_error if [path] cannot be read. *)

val save_image : ?overwrite:bool -> string -> (int, Nx.uint8_elt) Nx.t -> unit
(** [save_image ?overwrite path t] writes [t] to [path].

    The case-insensitive extension selects PNG ([.png]) or JPEG ([.jpg] and
    [.jpeg]). Accepted shapes are [[|height; width|]], [[|height; width; 1|]],
    [[|height; width; 3|]], and, for PNG only, [[|height; width; 4|]].
    [overwrite] defaults to [true]. If [overwrite] is [false], [path] must not
    exist.

    @raise Failure if the shape or extension is unsupported or encoding fails.
    @raise Unix.Unix_error
      if [path] cannot be written or already exists when [overwrite] is [false].
*)

(** {1:numpy NumPy formats} *)

val load_npy : string -> packed
(** [load_npy path] loads a tensor from a [.npy] file.

    @raise Failure
      if [path] cannot be read, the stream is malformed, or its dtype is
      unsupported. *)

val save_npy : ?overwrite:bool -> string -> ('a, 'b) Nx.t -> unit
(** [save_npy ?overwrite path t] writes [t] to a [.npy] file.

    [overwrite] defaults to [true]. If [overwrite] is [false], [path] must not
    exist.

    @raise Failure
      if [path] cannot be written, already exists when [overwrite] is [false],
      or [t]'s dtype has no standard NPY representation. *)

val load_npz : string -> archive
(** [load_npz path] loads all tensors from an [.npz] archive.

    The keys are entry names without the [.npy] suffix.

    @raise Failure
      if [path] cannot be read or the archive or an entry is malformed. *)

val load_npz_entry : name:string -> string -> packed
(** [load_npz_entry ~name path] loads a single entry from an [.npz] archive.

    [name] is the logical entry name, without the [.npy] suffix.

    @raise Failure
      if [path] cannot be read, [name] is missing, or the archive or entry is
      malformed. *)

val save_npz : ?overwrite:bool -> string -> (string * packed) list -> unit
(** [save_npz ?overwrite path entries] writes named tensors to an [.npz]
    archive.

    Names must be unique, valid UTF-8, relative, and free of empty, [.], and
    [..] path components. Compression is selected independently for each entry.
    [overwrite] defaults to [true]. If [overwrite] is [false], [path] must not
    exist.

    @raise Failure
      if a name is invalid or duplicated, a tensor dtype has no standard NPY
      representation, or [path] cannot be written. *)

val gunzip : src:string -> dst:string -> unit
(** [gunzip ~src ~dst] decompresses a gzip file to [dst]. Existing [dst] is
    replaced only after every member and checksum has been validated.

    Concatenated gzip members are supported.

    @raise Failure if [src] is malformed or a checksum is invalid.
    @raise Unix.Unix_error if [src] cannot be read or [dst] cannot be written.
*)

(** {1:safetensors SafeTensors} *)

val load_safetensors : string -> archive
(** [load_safetensors path] loads all tensors from a SafeTensors file.

    @raise Failure if [path] cannot be read or the stream is malformed. *)

val save_safetensors :
  ?overwrite:bool -> string -> (string * packed) list -> unit
(** [save_safetensors ?overwrite path entries] writes named tensors to a
    SafeTensors file.

    [overwrite] defaults to [true].

    @raise Failure
      if [path] cannot be written or a tensor's dtype has no SafeTensors
      equivalent (complex and int4 dtypes). *)

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

    @raise Failure if [path] cannot be read or its contents cannot be parsed. *)

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

    @raise Failure
      if [t]'s dtype or shape is unsupported or [path] cannot be written. *)
