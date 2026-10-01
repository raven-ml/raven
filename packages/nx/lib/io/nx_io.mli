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
    otherwise. An alpha channel is dropped, a grayscale image loads as three
    equal channels, and [~grayscale:true] reads a colour image as its luma
    [0.299 R + 0.587 G + 0.114 B] (ITU-R BT.601), within one level.

    @raise Failure if the stream is malformed or is neither PNG nor JPEG.
    @raise Unix.Unix_error if [path] cannot be read. *)

val save_image : ?overwrite:bool -> string -> (int, Nx.uint8_elt) Nx.t -> unit
(** [save_image ?overwrite path t] writes [t] to [path].

    The case-insensitive extension selects PNG ([.png]) or JPEG ([.jpg] and
    [.jpeg]). Accepted shapes are [[|height; width|]], [[|height; width; 1|]],
    [[|height; width; 3|]], and, for PNG only, [[|height; width; 4|]]. JPEG is
    written baseline at quality 90 (the standard IJG tables scaled to 20%),
    colour with its chroma subsampled 4:2:0. [overwrite] defaults to [true]. If
    [overwrite] is [false], [path] must not exist.

    @raise Failure
      if the shape or extension is unsupported, the image has no pixels, or
      encoding fails.
    @raise Unix.Unix_error
      if [path] cannot be written or already exists when [overwrite] is [false].
*)

val encode_png : ?dpi:float -> ?srgb:bool -> (int, Nx.uint8_elt) Nx.t -> string
(** [encode_png ?dpi ?srgb t] is the contents of a PNG file of [t], without
    touching the file system, with:
    - [dpi], the image's physical resolution in pixels per inch, written as a
      [pHYs] chunk of [Float.round (dpi /. 0.0254)] pixels per metre on both
      axes, so that viewers and printers show the image at its physical size.
      Without it the file states no physical size.
    - [srgb], whether to write an [sRGB] chunk with the perceptual rendering
      intent, which states that the samples are sRGB. Defaults to [false].

    Without [dpi] and [srgb] it is the file {!save_image} writes for [t].
    Accepted shapes are [[|height; width|]], [[|height; width; 1|]],
    [[|height; width; 3|]] and [[|height; width; 4|]].

    @raise Invalid_argument
      if [Float.round (dpi /. 0.0254)] is not in \[[1];[2{^31} - 1]\], the range
      of a PNG integer: [dpi] is below about [0.0127], above about [5.45e7], or
      not a number.
    @raise Failure
      if the shape is unsupported, the image has no pixels, or encoding fails.
*)

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

    Loading opens the file and reads its header; it reads no tensor data. Each
    entry is a value on the disk device ([Nx.Device.of_runtime Nx_device.disk]),
    over the entry's bytes in the file. The host reads an entry where it lies:
    an operation on it computes on the file's pages, mapped copy-on-write, and
    so does {!Nx.place} onto the host or onto a device whose memory is the
    host's, such as Metal's, which borrows them without a copy. {!Nx.place} onto
    a device whose memory the host does not address reads the entry's bytes into
    it. A movement ({!Nx.reshape}, {!Nx.slice}, {!Nx.transpose}, ...) of an
    entry is a value on the disk too. An entry whose bytes are not aligned to
    its elements, which a header of odd length causes, is read instead of
    mapped. On a big-endian host the entries are read and put in its byte order
    as they load.

    The file stays open, and mapped once read, until the last value over it is
    garbage collected, and its values see the file that was opened:
    {!save_safetensors} to [path], which renames a new file over it, leaves them
    their values.
    {b The file must not change in place while a value loaded from it is alive}:
    rewriting it may change their values, and truncating it kills the process
    with a bus error when a truncated page is read. On Windows a mapped file may
    not be deleted or replaced until its values are collected.

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
    file as it was. Each tensor's bytes are copied into the file from where they
    are: a device writes its memory to the file, and a tensor loaded by
    {!load_safetensors} is copied from its file. If the rename is refused, a
    major collection runs, which closes the files of the tensors no longer
    reachable, and the rename is retried once.

    [overwrite] defaults to [true]. If [overwrite] is [false], [path] must not
    exist.

    @raise Failure
      if [path] cannot be written, a name is given twice, or a tensor's dtype
      has no SafeTensors equivalent (complex and int4 dtypes). If the rename is
      refused twice, the message names the temporary file, which is kept and
      holds [entries]. *)

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

    The first [skiprows] lines are skipped, then each line is a row of fields
    separated by [sep], except blank lines and lines that start with [comments],
    and at most [max_rows] rows are read. A file of one row or one column loads
    as a vector, and any other as a matrix of shape [[|rows; columns|]]. [sep]
    defaults to [" "], [comments] to ["#"], [skiprows] to [0], and [max_rows] to
    every row.

    @raise Failure
      if [path] cannot be read, its contents cannot be parsed (no row, rows of
      different lengths, a field that is no literal of [dtype] or is out of its
      range), [skiprows] is negative, [max_rows] is not positive, or text does
      not hold [dtype] (see {!save_txt}). *)

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

    A vector is written as one row. Each line of [header] and [footer] is
    written before and after the rows, prefixed with [comments]. Floats are
    written in [%.18e] notation, 19 significant digits rounded from their exact
    value, so a float reads back as itself, NaN aside: every NaN is written
    [nan], and the infinities [inf] and [-inf]. Integers are written in decimal,
    unsigned ones as unsigned, and booleans as [1] and [0]. Text holds bool, the
    integer dtypes but int4 and uint4, and the float dtypes but float8.

    [sep] defaults to [" "]. [append] defaults to [false]. [newline] defaults to
    ["\n"]. [comments] defaults to ["# "].

    @raise Failure
      if [t]'s dtype or shape is unsupported or [path] cannot be written. *)
