(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tensor I/O.

    Load and save {!Nx} tensors in common formats: images (PNG and JPEG), NumPy
    ([.npy] and [.npz]), SafeTensors, and delimited text. *)

(** {1:archives Archives} *)

type archive = (string, Nx.packed) Hashtbl.t
(** The type for named tensors, as {!load_npz} and {!load_safetensors} return
    them. Read an entry with {!Nx.unpack}. *)

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

val encode_png : (int, Nx.uint8_elt) Nx.t -> string
(** [encode_png t] is the contents of the PNG file {!save_image} would write for
    [t], without touching the file system. Accepted shapes are
    [[|height; width|]], [[|height; width; 1|]], [[|height; width; 3|]] and
    [[|height; width; 4|]].

    @raise Failure if the shape is unsupported or encoding fails. *)

(** {1:numpy NumPy formats} *)

val load_npy : string -> Nx.packed
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

val load_npz_entry : name:string -> string -> Nx.packed
(** [load_npz_entry ~name path] loads a single entry from an [.npz] archive.

    [name] is the logical entry name, without the [.npy] suffix.

    @raise Failure
      if [path] cannot be read, [name] is missing, or the archive or entry is
      malformed. *)

val save_npz : ?overwrite:bool -> string -> (string * Nx.packed) list -> unit
(** [save_npz ?overwrite path entries] writes named tensors to an [.npz]
    archive.

    Names must be unique, valid UTF-8, relative, and free of empty, [.], and
    [..] path components. Compression is selected independently for each entry.
    [overwrite] defaults to [true]. If [overwrite] is [false], [path] must not
    exist.

    @raise Failure
      if a name is invalid or duplicated, a tensor dtype has no standard NPY
      representation, or [path] cannot be written. *)

val deflate : string -> string
(** [deflate s] is [s] compressed as a zlib stream, the format of PDF's
    [FlateDecode] filter and of PNG image data. *)

val inflate : string -> string
(** [inflate s] is the data of the zlib stream [s], so that
    [inflate (deflate s) = s].

    @raise Failure if [s] is not a zlib stream or its checksum does not match.
*)

val gunzip : src:string -> dst:string -> unit
(** [gunzip ~src ~dst] decompresses a gzip file to [dst]. Existing [dst] is
    replaced only after every member and checksum has been validated.

    Concatenated gzip members are supported.

    @raise Failure if [src] is malformed or a checksum is invalid.
    @raise Unix.Unix_error if [src] cannot be read or [dst] cannot be written.
*)

(** {1:safetensors SafeTensors} *)

val load_safetensors : string -> archive
(** [load_safetensors path] is the tensors of the SafeTensors file [path], by
    name.

    Loading reads the header and maps the file; it reads no tensor data. An
    entry whose data sits in the file at an address that suits its dtype is a
    view of the mapping, whose pages the system reads when they are first used
    and may drop again under memory pressure. Any other entry is copied by the
    load: one whose address is not a multiple of its element size, which a
    header of odd length causes, and every entry on a big-endian host.

    {b The file must not change while a tensor loaded from it is alive.}
    Truncating or rewriting it in place changes the tensors' values or kills the
    process with a bus error, which no handler catches. Replace a file by
    writing a new one and renaming it over the old one, as {!save_safetensors}
    does; [Nx.copy] gives a tensor that no longer depends on its file. The file
    stays mapped until the last tensor over it is garbage collected, and on
    Windows it may not be deleted or replaced until then.

    An entry whose dtype nx lacks is loaded as its bytes, at [uint8]: [F8_E8M0]
    keeps the entry's shape, and [F4], [F6_E2M3] and [F6_E3M2], whose elements
    are narrower than a byte, have shape [[| n |]] with [n] the entry's size in
    bytes. A [BOOL] byte other than 0 or 1 is handed out as stored. An entry
    with no elements is an empty tensor of its shape.

    @raise Failure
      naming [path], if it is not a regular file that can be read, if its header
      is malformed, longer than 100 MB or names a tensor twice, or if the file's
      length differs from the one its header describes, as a partial download's
      does. *)

val save_safetensors :
  ?overwrite:bool -> string -> (string * Nx.packed) list -> unit
(** [save_safetensors ?overwrite path entries] writes named tensors to a
    SafeTensors file.

    The tensors are written to a temporary file in [path]'s directory, which is
    synced to disk and then renamed to [path]: a reader sees the previous file
    or the new one, never a partial one, and a failed save leaves the previous
    file as it was. If the rename is refused, which happens on platforms that
    lock a file while tensors loaded from it are alive, a major collection runs
    and the rename is retried once.

    [overwrite] defaults to [true]. If [overwrite] is [false], [path] must not
    exist.

    @raise Failure
      if [path] cannot be written or a tensor's dtype has no SafeTensors
      equivalent (complex and int4 dtypes). If the rename is refused twice, the
      message names the temporary file, which is kept and holds [entries]. *)

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
