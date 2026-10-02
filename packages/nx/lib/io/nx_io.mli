(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tensor I/O.

    Load and save {!Nx} tensors in common formats: images (PNG and JPEG), NumPy
    ([.npy] and [.npz]), SafeTensors, and delimited text, and load GGUF. A file
    of named tensors is read and written as an {!Archive.t}, and every format's
    functions are named after it.

    See doc/04-io.md for saving and loading structured values. *)

(** {1:archives Archives} *)

(** Archives of named tensors.

    An archive is an immutable collection of tensors keyed by distinct,
    non-empty names. Each format of named tensors reads and writes one:
    {!load_safetensors} and {!save_safetensors}, {!load_npz} and {!save_npz},
    and the tensors of {!load_gguf}.

    A value enters and leaves an archive through its structure ({!Nx.Ptree.t}).
    {!of_value} names each tensor by its path, and {!to_value} reads a value
    back into the shape of one the program already has:

    {[
    Nx_io.save_safetensors path (Nx_io.Archive.of_value train state);
    let state =
      Nx_io.Archive.to_value train ~like:state (Nx_io.load_safetensors path)
    ]}

    A file written elsewhere has its own names and layouts. Its importer asks
    for each entry by name, shape and dtype with {!tensor} and {!float}, and
    rearranges it with nx:

    {[
    let linear a ~inputs ~outputs name =
      (* The file stores the weight as [outputs; inputs]. *)
      Nx.matrix_transpose
        (Nx_io.Archive.float ~shape:[| outputs; inputs |] Nx.float32 name a)
    ]}

    Asking for an entry reads none of its bytes: it is returned as stored, and
    an archive a format loads holds its entries where they lie in the file.
    Bytes are never reinterpreted: only {!float} converts, and a block-quantised
    or sub-byte entry is read as [uint8] with {!tensor}.

    An archive is input data, so a mismatch between an archive and what is asked
    of it raises [Failure], naming the entry. *)
module Archive : sig
  type t
  (** The type for archives: tensors keyed by distinct, non-empty names. *)

  (** {1:constructors Constructors} *)

  val of_list : (string * Nx.packed) list -> t
  (** [of_list entries] is the archive of [entries].

      Raises [Invalid_argument] if a name is empty or given twice. *)

  val union : t list -> t
  (** [union ts] is the archive of the entries of all [ts].

      Raises [Invalid_argument] if a name is in two of [ts]. *)

  (** {1:queries Queries} *)

  val names : t -> string list
  (** [names a] is the names of [a]'s entries, sorted. *)

  val find : string -> t -> Nx.packed option
  (** [find name a] is [a]'s entry [name], if any. *)

  val tensor :
    shape:int array -> ('a, 'b) Nx.dtype -> string -> t -> ('a, 'b) Nx.t
  (** [tensor ~shape dtype name a] is [a]'s entry [name], as stored.

      Raises [Failure] naming the entry if [a] has no entry [name], or if its
      shape is not [shape] or its dtype is not [dtype], as in
      ["Nx_io.Archive.tensor: h.0.w: shape [3] in the archive, [2; 3] asked
       for"]. *)

  val float :
    shape:int array -> (float, 'b) Nx.dtype -> string -> t -> (float, 'b) Nx.t
  (** [float ~shape dtype name a] is [a]'s entry [name] at [dtype]: the entry as
      stored when its dtype is [dtype], and otherwise its cast, which allocates.
      Both dtypes are float16, bfloat16, float32 or float64. An importer that
      ties two weights binds the result once and uses it twice.

      Raises [Invalid_argument] if [dtype] is not one of the four. Raises
      [Failure] naming the entry if [a] has no entry [name], if its shape is not
      [shape], or if its dtype is not one of the four. An 8-bit float entry is
      read with {!tensor}, since its scales are other entries. *)

  (** {1:values Values} *)

  val of_value : 's Nx.Ptree.t -> 's -> t
  (** [of_value s x] has one entry for each tensor [s] walks in [x], fixed
      tensors included, named by its path ({!Nx.Ptree.Path.to_string}). The
      entries are [x]'s tensors; nothing is copied. A section of a file is a
      value under {!Nx.Ptree.field}, and {!union} puts sections together.

      Raises [Invalid_argument] if two tensors of [x] have one name, as in
      ["Nx_io.Archive.of_value: w: two leaves have this name"], or if a tensor
      is at the root, whose name is empty. *)

  val to_value : 's Nx.Ptree.t -> like:'s -> t -> 's
  (** [to_value s ~like a] is [like] with each tensor replaced by [a]'s entry of
      its name, as stored. [like] gives the structure, the dtypes and the
      shapes, and its tensors are discarded: a value is read back with [like]'s
      list lengths, option presences and cases. For values [x] and [y] with
      equal {!Nx.Ptree.visits}, dtypes and shapes,
      [to_value s ~like:y (of_value s x)] is [x].

      [s] owns the names under its {!Nx.Ptree.prefix}, every name when that is
      the root. Raises [Failure], naming the entry, before any entry is read:
      - if an entry [like] names is missing, as in
        ["Nx_io.Archive.to_value: model.l1.w: no entry in the archive, a leaf in
         the value"];
      - if its shape or dtype differs, as in
        ["Nx_io.Archive.to_value: model.l1.w: shape [3] in the archive, [2; 3]
         in the value"];
      - if an entry under the prefix is named by no tensor of [like], as in
        ["Nx_io.Archive.to_value: blocks.12.w: an entry in the archive, no leaf
         in the value"]: a model of 12 blocks refuses a file of 24.

      Entries outside the prefix are ignored, so one file holds several
      sections. Nothing is converted: a value that states another dtype than the
      archive fails instead of narrowing its state, and {!float} converts by
      name.

      Raises [Invalid_argument] as {!of_value} does if [like]'s names are not
      distinct and non-empty. *)
end

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

val load_npz : string -> Archive.t
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

val save_npz : ?overwrite:bool -> string -> Archive.t -> unit
(** [save_npz ?overwrite path a] writes [a]'s tensors to an [.npz] archive.

    Names must be valid UTF-8, relative, and free of empty, [.], and [..] path
    components. Compression is selected independently for each entry.
    [overwrite] defaults to [true]. If [overwrite] is [false], [path] must not
    exist.

    @raise Failure
      if a name is invalid, a tensor's dtype has no standard NPY representation
      (bfloat16, the 8-bit floats, int4 and uint4), naming the entry, or [path]
      cannot be written. *)

val gunzip : src:string -> dst:string -> unit
(** [gunzip ~src ~dst] decompresses a gzip file to [dst]. Existing [dst] is
    replaced only after every member and checksum has been validated.

    Concatenated gzip members are supported.

    @raise Failure if [src] is malformed or a checksum is invalid.
    @raise Unix.Unix_error if [src] cannot be read or [dst] cannot be written.
*)

(** {1:safetensors SafeTensors} *)

val load_safetensors : string -> Archive.t
(** [load_safetensors path] is the tensors of the SafeTensors file [path], by
    name.

    Loading opens the file and reads its header; it reads no tensor data. Each
    entry is a value on the disk device ({!Nx_device.disk}), over the entry's
    bytes in the file. The host reads an entry where it lies: an operation on it
    computes on the file's pages, mapped copy-on-write, and so does {!Nx.place}
    onto the host or onto a device whose memory is the host's, such as Metal's,
    which borrows them without a copy. {!Nx.place} onto a device whose memory
    the host does not address reads the entry's bytes into it. A movement
    ({!Nx.reshape}, {!Nx.slice}, {!Nx.transpose}, ...) of an entry is a value on
    the disk too. An entry whose bytes are not aligned to its elements, which a
    header of odd length causes, is read instead of mapped. On a big-endian host
    the entries are read and put in its byte order as they load.

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
      is malformed, longer than 100 MB, names a tensor twice or gives a tensor
      an empty name, or if the file's length differs from the one its header
      describes, as a partial download's does. *)

val save_safetensors : ?overwrite:bool -> string -> Archive.t -> unit
(** [save_safetensors ?overwrite path a] writes [a]'s tensors to a SafeTensors
    file.

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
      if [path] cannot be written, or if a tensor's dtype has no SafeTensors
      equivalent (the complex dtypes, int4 and uint4), naming the entry. If the
      rename is refused twice, the message names the temporary file, which is
      kept and holds [a]'s tensors. *)

(** {1:gguf GGUF} *)

(** GGUF files.

    A GGUF file holds a model's tensors and its metadata: key-values that
    describe the architecture, the tokenizer and the file itself. *)
module Gguf : sig
  (** The type for metadata values, one case per GGUF value type. An unsigned
      64-bit value is held in [int64] by its bits. An array's elements are of
      one type. *)
  type value =
    | Uint8 of int
    | Int8 of int
    | Uint16 of int
    | Int16 of int
    | Uint32 of int
    | Int32 of int
    | Uint64 of int64
    | Int64 of int64
    | Float32 of float
    | Float64 of float
    | Bool of bool
    | String of string  (** The bytes stored, UTF-8 by the specification. *)
    | Array of value array

  (** The type for tensor types. Each tensor of a file is stored as one of them:
      a scalar type, whose elements nx holds, or a block format, which stores
      the elements of a row in blocks of a fixed number of elements and bytes.
  *)
  type dtype =
    | F32
    | F16
    | BF16
    | F64
    | I8
    | I16
    | I32
    | I64
    | Q4_0
    | Q4_1
    | Q5_0
    | Q5_1
    | Q8_0
    | Q8_1
    | Q2_K
    | Q3_K
    | Q4_K
    | Q5_K
    | Q6_K
    | Q8_K
    | IQ2_XXS
    | IQ2_XS
    | IQ3_XXS
    | IQ1_S
    | IQ4_NL
    | IQ3_S
    | IQ2_S
    | IQ4_XS
    | IQ1_M
    | TQ1_0
    | TQ2_0
    | MXFP4

  type tensor_info = {
    dtype : dtype;  (** The type the tensor is stored as. *)
    shape : int array;
        (** The tensor's logical shape, in row-major order: the reverse of the
            dimensions the file lists, whose first varies fastest. *)
  }
  (** The type for a tensor's description in the file. *)

  type t
  (** The type for the contents of a GGUF file. *)

  val version : t -> int
  (** [version g] is [g]'s format version, [2] or [3]. *)

  val metadata : t -> (string * value) list
  (** [metadata g] is [g]'s key-values, in file order. *)

  val tensors : t -> Archive.t
  (** [tensors g] is [g]'s tensors, by name. *)

  val info : string -> t -> tensor_info
  (** [info name g] is the description of [g]'s tensor [name].

      Raises [Failure] if [g] has no tensor [name]. *)
end

val load_gguf : string -> Gguf.t
(** [load_gguf path] is the contents of the GGUF file [path], of version 2 or 3,
    in little-endian byte order.

    Loading opens the file and reads its header: the key-values and the tensor
    descriptions. It reads no tensor data. Each tensor is a value on the disk
    device ({!Nx_device.disk}) over its bytes in the file, read where they lie
    as {!load_safetensors} reads its entries: an operation on it computes on the
    file's pages, mapped copy-on-write, and {!Nx.place} onto a device whose
    memory is the host's borrows them without a copy. A tensor whose bytes are
    not aligned to its elements, which a [general.alignment] below its element
    size allows, is read instead of mapped. On a big-endian host the tensors are
    read and put in its byte order as they load.

    A tensor of a scalar type has its {!Gguf.tensor_info.shape} and the dtype of
    the same name: [Float32], [Float16], [BFloat16], [Float64], [Int8], [Int16],
    [Int32] or [Int64]. A tensor of a block format is loaded as its bytes, at
    [uint8], of its logical shape with the last dimension replaced by the size
    of a row in bytes: a Q8_0 row of 64 elements is two blocks of 34 bytes, so a
    Q8_0 tensor of logical shape [[|m; 64|]] loads at shape [[|m; 68|]]. A
    tensor with no elements is an empty tensor of its shape.

    The file stays open, and mapped once read, until the last value over it is
    garbage collected, and its values see the file that was opened.
    {b The file must not change in place while a value loaded from it is alive}:
    rewriting it may change their values, and truncating it kills the process
    with a bus error when a truncated page is read. On Windows a mapped file may
    not be deleted or replaced until its values are collected.

    @raise Failure
      naming [path], if it is not a regular file that can be read, if it does
      not start with the GGUF magic, if its version is not 2 or 3 or its byte
      order is big-endian, if its header is malformed or cut short, names a key
      or a tensor twice, or gives a tensor an empty name, a type this function
      does not know, a row that is not a whole number of blocks or a position
      that is not a multiple of the alignment, or if the file ends before a
      tensor's data. *)

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
