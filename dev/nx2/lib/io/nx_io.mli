(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrays to and from files.

    A file holds one array, as npy and text files do, or several arrays under
    names, as npz, SafeTensors and GGUF files do. A name and its array are an
    {e entry}. Entries are read and written as an {!Archive.t}.

    An {e image} is a [uint8] array of shape [[|h; w; c|]], first row at the
    top, with [c] 1 for grey, 2 for grey and alpha, 3 for RGB and 4 for RGBA.

    {b Loading.} A load returns host values ({!Nx.Host}). Where the system can
    map the file, a load returns each array whose elements lie in the file in
    the host's byte order and in C order (or Fortran order, for npy), at an
    offset that is a multiple of their alignment, as a value over the file's
    pages, a {e mapped} value. {!Nx.place} shares the pages with a device that
    addresses host memory and copies them to any other ({!Rig.reaches}). A
    donation of a mapped value writes its result to new memory ({!Nx.donate}).

    Any other array of a file the system maps, such as a compressed npz entry, a
    misaligned one or an image, is read at load into memory of its own, where
    its donation may write its result. The arrays of a file the system cannot
    map are read at load into memory. An entry of a dtype nx lacks loads as its
    bytes at [uint8], at the shape its format's section states, and an entry
    with no elements is an empty array of that shape.

    A mapped value keeps its file mapped while it is reachable, and reads the
    file that was opened, so a save to its path leaves it its elements.
    {b The file must not change in place while a mapped value from it is
       reachable.} A change may show in the value's elements, and a truncation
    may end the process. A later copy of such a value to a device whose memory
    the host does not address ({!Nx.place}) may then raise [Sys_error] naming
    the file, as may one after a save has replaced the file while over 64 files
    are open ({!Rig_disk.section-borrows}, {!Rig_disk.section-descriptors}).

    {b Saving.} A save takes a value of any set. An npy, npz or SafeTensors save
    writes the elements in C order and little-endian, and a text save in C
    order. A save reads the value's own memory where the host addresses it, and
    leaves that memory unshared, so a later donation of the value may write its
    result there ({!Nx.Repr.read}).

    A save copies onto the host a value off the host or out of C order, and on a
    big-endian host one whose bytes it swaps. On a little-endian host, a value
    off the host and out of C order takes up to twice its bytes. An npz or
    SafeTensors save writes each entry before it reads the next.

    A save replaces what its path names with a new file it renames over the
    path. After a crash, the path holds the old file or the whole new one,
    except on a file system with neither a write barrier nor a flush of the
    drive's cache, such as some network file systems on macOS, where the order
    is the file system's own. A save that fails before the rename leaves the
    path as it was and nothing beside it. A refused rename keeps the new file,
    which the [Failure] names. Windows refuses it while a mapped value from the
    old file is reachable ({!Rig_disk.section-borrows}). A failed sync of the
    directory after the rename raises [Failure] with the new file at the path.
    On POSIX systems the new file has the permissions [0o666] less the umask.

    {b Defaults.} A load blocks until it has read the file's header and the
    elements of each value it returns that is not mapped. A save blocks until
    the value's set has computed it and the new file is at its path. Domains may
    load one path concurrently. Concurrent saves to one path each write a whole
    file, and the last rename wins.

    This writes a vector to an npy file and reads it back, and {!Nx.unpack}
    checks the dtype the file states:
    {[
    let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    Nx_io.save_npy "x.npy" x;
    Nx.unpack Nx.float32 (Nx_io.load_npy "x.npy")
    ]} *)

(** {1:archives Archives} *)

(** Archives of entries.

    An archive is an immutable collection of entries with distinct, non-empty
    names, whose arrays lie on one set. A structure ({!Nx.Ptree}) enters an
    archive with {!Archive.of_value}, which names each array by a prefix, a dot
    and the array's path ({!Nx.Ptree.Path.to_string}), as ["model.blocks.0.w"].
    An empty prefix, the default, is left out with its dot, and so are the dot
    and the empty path of an array at the root.

    Under a prefix [p], a value {e owns} [p], every name that starts with [p]
    and a dot, and, for an empty [p], every name. {!Archive.to_value} reads the
    entries a value owns back into the shape and the placement of one the
    program already has, and ignores the others, so one file can store several
    values under distinct prefixes:

    {[
    let checkpoint = Nx_io.Archive.union [
      Nx_io.Archive.of_value ~prefix:"model" model params;
      Nx_io.Archive.of_value ~prefix:"opt" opt state ] in
    Nx_io.save_safetensors path checkpoint;
    let params =
      Nx_io.Archive.to_value ~prefix:"model" model ~like:params
        (Nx_io.load_safetensors path)
    ]}

    Code that reads a file another program wrote, an {e importer}, asks for each
    entry by name, shape and dtype with {!Archive.tensor} and {!Archive.float},
    and rearranges it with nx:

    {[
    let linear a ~inputs ~outputs name =
      (* The file stores the weight as [outputs; inputs]. *)
      Nx.matrix_transpose
        (Nx_io.Archive.float ~shape:[| outputs; inputs |] Nx.float32 name a)
    ]}

    Asking for an entry reads none of its bytes, except {!Archive.float} when it
    converts. Only {!Archive.float} converts an entry's dtype. A block-quantised
    or sub-byte entry is read as [uint8] with {!Archive.tensor}. An 8-bit float
    entry is read with {!Archive.tensor} too, since its scales are other
    entries.

    An archive is input data, so a mismatch between an archive and what is asked
    of it raises [Failure], naming the entry. *)
module Archive : sig
  type 'd t
  (** The type for archives of values on the set ['d]. *)

  val of_list : (string * 'd Nx.packed) list -> 'd t
  (** [of_list entries] is the archive of [entries].

      Raises [Invalid_argument] if a name is empty or given twice. *)

  val union : 'd t list -> 'd t
  (** [union ts] is the archive of the entries of all [ts].

      Raises [Invalid_argument] if a name is in two of [ts]. *)

  val names : 'd t -> string list
  (** [names a] is the names of [a]'s entries, sorted. *)

  val find : string -> 'd t -> 'd Nx.packed option
  (** [find name a] is [a]'s entry [name], if any. *)

  val tensor :
    shape:int array -> ('v, 's) Nx.dtype -> string -> 'd t -> ('v, 's, 'd) Nx.t
  (** [tensor ~shape dt name a] is [a]'s entry [name], as stored.

      Raises [Failure] naming the entry if [a] has no entry [name], or if its
      shape is not [shape] or its dtype is not [dt], as in
      ["Nx_io.Archive.tensor: h.0.w: shape [3] in the archive, [2; 3] asked
       for"]. *)

  val float :
    shape:int array ->
    (float, 's) Nx.dtype ->
    string ->
    'd t ->
    (float, 's, 'd) Nx.t
  (** [float ~shape dt name a] is [a]'s entry [name] at [dt]. It is the entry as
      stored when its dtype is [dt], and otherwise its cast ({!Nx.cast}), which
      allocates and reads every element. It converts between float16, bfloat16,
      float32 and float64. Each call that casts makes a new value, so an
      importer calls [float] once for two tied weights.

      Raises [Invalid_argument] if [dt] is not one of the four. Raises [Failure]
      naming the entry if [a] has no entry [name], if its shape is not [shape],
      or if its dtype is not one of the four. *)

  val of_value : ?prefix:string -> ('s, 'd) Nx.Ptree.t -> 's -> 'd t
  (** [of_value ~prefix s x] is the archive with one entry for each array [s]
      walks in [x], fixed arrays included, named under [prefix] as the module's
      preamble states. The entries are [x]'s arrays, and nothing is copied.

      Raises [Invalid_argument] if two arrays of [x] have one name, as in
      ["Nx_io.Archive.of_value: w: two arrays have this name"], or if a name is
      empty, as that of an array at the root without a prefix. *)

  val to_value : ?prefix:string -> ('s, 'd) Nx.Ptree.t -> like:'s -> 'e t -> 's
  (** [to_value ~prefix s ~like a] is [like] with each array replaced by [a]'s
      entry of its name under [prefix], placed where [like]'s array lies
      ({!Nx.place}). [like] gives the structure, with its list lengths, option
      presences and variant cases, and each array's dtype, shape and placement.
      Its arrays' elements are not read. For values [x] and [y] with equal
      {!Nx.Ptree.val-visits}, dtypes, shapes and placements,
      [to_value ~prefix s ~like:y (of_value ~prefix s x)] is [x].

      Raises [Failure], naming the entry, before any entry is placed:
      - if an entry [like] names is missing;
      - if its shape or dtype differs;
      - if an entry [like] owns is named by none of its arrays.

      Raises [Invalid_argument] as {!of_value} does for [like]'s names, and,
      naming the entry, for an array of [like] of every set, which lies on no
      device. *)
end

(** {1:numpy NumPy formats}

    An npy file holds one array, and an npz file is a zip archive of npy files,
    each named by its entry's name and [.npy]. npy stores float16, float32,
    float64, the signed and unsigned integers of 8 to 64 bits, complex64,
    complex128 and bool. A save writes the format's version 1.0. A load reads
    its versions 1.0, 2.0 and 3.0, in either byte order and either axis order,
    and an npz entry stored or compressed with DEFLATE.

    Versions 1.0 and 2.0 are NumPy's NEP 1,
    {{:https://numpy.org/neps/nep-0001-npy-format.html}A simple file format for
     NumPy arrays}, and version 3.0 is that of
    {{:https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html}
     numpy.lib.format}. The zip container is PKWARE's
    {{:https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT}APPNOTE}
    without encryption or several disks. *)

val load_npy : string -> Nx.host Nx.packed
(** [load_npy path] is the array of the npy file [path]. It is packed because
    the file states its dtype, which [Nx.unpack dt] checks ({!Nx.unpack}). *)

val save_npy : string -> ('v, 's, 'd) Nx.t -> unit
(** [save_npy path x] writes [x] to the npy file [path].

    Raises [Invalid_argument] if npy does not store [x]'s dtype. For [bit], the
    message names [Nx.cast Nx.bool], which stores a byte per element. *)

val load_npz : string -> Nx.host Archive.t
(** [load_npz path] is the entries of the npz file [path]. Each zip entry whose
    name ends in [.npy] holds one, named by the part before [.npy], which must
    not be empty. Entries of other names are not read; their records must still
    be well-formed. It reads every npy entry's bytes, since each entry's
    checksum covers all of them.

    Raises [Failure] as {!section-errors} states, and if a checksum does not
    match its entry, if an entry's name is [.npy] alone, or if an entry whose
    name ends in [.npy] holds no npy stream, as the [__MACOSX/._w.npy] that
    macOS's Finder writes beside [w.npy] does. *)

val save_npz : string -> 'd Archive.t -> unit
(** [save_npz path a] writes [a]'s entries to the npz file [path]. Each entry is
    compressed with DEFLATE if that shrinks its first 64 KiB, and stored
    otherwise.

    Raises [Invalid_argument] naming the entry:
    - if its name is not UTF-8, starts with [/], holds a backslash, a NUL byte
      or an empty, [.] or [..] component, or names a drive;
    - if npy does not store its dtype. *)

(** {1:safetensors SafeTensors}

    A SafeTensors file is a JSON header that names each array with its dtype,
    shape and byte range, then the arrays' bytes, as
    {{:https://github.com/huggingface/safetensors}the format's README}
    specifies. It stores bool, the signed and unsigned integers of 8 to 64 bits,
    float8_e4m3fn ([F8_E4M3]), float8_e5m2, float16, bfloat16, float32, float64
    and complex64 ([C64]). The format's other dtypes load as bytes: [F8_E8M0],
    [F8_E4M3FNUZ] and [F8_E5M2FNUZ] at the entry's shape, and [F4], [F6_E2M3]
    and [F6_E3M2], whose elements are narrower than a byte, at shape [[|n|]] for
    an entry of [n] bytes. *)

val load_safetensors : string -> Nx.host Archive.t
(** [load_safetensors path] is the entries of the SafeTensors file [path].

    Raises [Failure] as {!section-errors} states, and:
    - if the header is malformed or longer than 100 MB;
    - if it is not UTF-8, names an array twice or gives one an empty name;
    - if the file's length differs from the one its header states, as a partial
      download's does. *)

val save_safetensors : string -> 'd Archive.t -> unit
(** [save_safetensors path a] writes [a]'s entries to the SafeTensors file
    [path], in the order of the format's reference implementation. The entries
    come by dtype, in the order U64, I64, F64, C64, F32, U32, I32, BF16, F16,
    U16, I16, F8_E4M3, F8_E5M2, I8, U8, BOOL, then by name.

    Raises [Invalid_argument] naming the entry if SafeTensors does not store its
    dtype, or if its name is not UTF-8 or is ["__metadata__"], the key of the
    file's metadata. *)

(** {1:gguf GGUF} *)

(** GGUF files.

    A GGUF file holds a model's arrays and its metadata, key-values that
    describe the architecture, the tokenizer and the file itself, as
    {{:https://github.com/ggml-org/ggml/blob/master/docs/gguf.md}GGML's
     specification} defines them. Each array is stored as one GGUF type
    ({!Gguf.type-dtype}). A {e scalar type} stores each element whole, and a
    {e block format} stores the elements of a row in blocks of a fixed number of
    elements and bytes. An array's {e logical shape} is its shape in elements,
    in row-major order ({!Gguf.tensor_info}).

    An array of a scalar type loads at its logical shape. An array of a block
    format loads as its bytes, at [uint8], at its logical shape with the last
    extent replaced by the size of a row in bytes. A Q8_0 block is 32 elements
    in 34 bytes, so a Q8_0 array of logical shape [[|m; 64|]] loads at shape
    [[|m; 68|]]. *)
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

  (** The type for GGUF storage types. [F32], [F16], [BF16], [F64], [I8], [I16],
      [I32] and [I64] are the scalar types, which load as float32, float16,
      bfloat16, float64, int8, int16, int32 and int64. The others are block
      formats. *)
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
    dtype : dtype;  (** The type the array is stored as. *)
    shape : int array;
        (** The array's logical shape, the reverse of the dimensions the file
            lists, whose first varies fastest. *)
  }
  (** The type for an array's description in the file. *)

  type t
  (** The type for the contents of a GGUF file. *)

  val version : t -> int
  (** [version g] is [g]'s format version, [2] or [3]. *)

  val metadata : t -> (string * value) list
  (** [metadata g] is [g]'s key-values, in file order. *)

  val tensors : t -> Nx.host Archive.t
  (** [tensors g] is the archive of [g]'s arrays. *)

  val info : string -> t -> tensor_info
  (** [info name g] is the description of [g]'s array [name].

      Raises [Failure] if [g] has no array [name]. *)
end

val load_gguf : string -> Gguf.t
(** [load_gguf path] is the contents of the GGUF file [path].

    Raises [Failure] as {!section-errors} states, and:
    - if the file does not start with the GGUF magic, its version is not 2 or 3,
      or its byte order is big-endian;
    - if its header is malformed or cut short, or names a key or an array twice;
    - if it gives an array an empty name, a type this function does not know, a
      row that is not a whole number of blocks, or a position that is not a
      multiple of the alignment;
    - if the file ends before an array's data. *)

(** {1:text Text}

    A text file holds a vector or a matrix, one row per line. A separator
    divides a row into fields. Text stores every dtype but the complex ones.

    A float field is:
    - an optional sign;
    - digits with an optional point, either side of the point possibly empty but
      not both;
    - an optional exponent, [e] or [E] followed by an optional sign and digits.

    [nan], [inf] and [infinity], in any case and after an optional sign, are
    float fields too. A load reads a float field as the float64 nearest its
    value, a tie as the one whose significand is even, on every system, then
    stores it as {!Nx.Dtype.of_float} says. A save writes a float in [%.18e]
    notation, its 19 significant digits rounded from its exact value, so it
    reads back as itself. It writes every NaN as [nan], and the infinities as
    [inf] and [-inf].

    An integer field is an optional sign and decimal digits, in its dtype's
    range, and a save writes an integer in decimal, an unsigned one unsigned. A
    bool or bit field is [true] or [false] in any case, or a float field, [true]
    where it is not zero, and a save writes [1] or [0]. Only these spellings are
    literals. *)

val load_txt :
  ?sep:string ->
  ?comments:string ->
  ?skiprows:int ->
  ('v, 's) Nx.dtype ->
  string ->
  ('v, 's, Nx.host) Nx.t
(** [load_txt dt path] is the array of the text file [path] at [dt].

    It skips the first [skiprows] lines, blank lines, and lines starting with
    [comments] after blanks. Each other line is a row that [sep] divides into
    fields, each trimmed of blanks. A run of spaces is one separator for the
    default [sep], [" "]. With any other [sep] an empty field is a field, so
    with [sep] [","], ["1, 2"] has two fields and ["1,,2"] three. A single row
    or column loads as a vector, and more as a matrix. [comments] defaults to
    ["#"] and [skiprows] to [0].

    Raises [Invalid_argument] if [dt] is complex or [skiprows] is negative.
    Raises [Failure] as {!section-errors} states if the file has no row, and,
    naming the line, if rows differ in length, or a field is not a literal of
    [dt] or is out of range. *)

val save_txt : ?sep:string -> string -> ('v, 's, 'd) Nx.t -> unit
(** [save_txt path x] writes the 0-d value, vector or matrix [x] to the text
    file [path], with no header or comment line. A 0-d value or a vector is one
    row, and a matrix one row per line. Each row ends with ["\n"], and [sep],
    which defaults to [" "], separates its fields. The file loads back with
    [sep] only if [sep] is not empty, holds no newline and occurs in no field. A
    save of a value with no element writes an empty file, which {!load_txt}
    rejects. A save also allocates [x]'s elements as an OCaml array.

    Raises [Invalid_argument] if [x] is complex or has more than two axes. *)

(** {1:images Images}

    Each save writes the format it is named after, whatever the path's
    extension. A PNG save also allocates two copies of the image's filtered
    rows, about its bytes each, and three of the encoded file, and a JPEG save
    two of the encoded file.

    {!load_image} reads the PNG files of the
    {{:https://www.w3.org/TR/png-3/}PNG Specification, Third Edition}, with
    every colour type and bit depth, palettes, transparency and interlacing.

    It reads the JPEG files of {{:https://www.itu.int/rec/T-REC-T.81}ITU-T T.81}
    with 8-bit samples and Huffman coding, baseline, extended sequential or
    progressive, in grey, YCbCr, RGB, CMYK or YCCK, with any sampling and
    restart intervals.

    {!save_png}, {!save_jpeg} and {!encode_png} raise [Invalid_argument] naming
    the function for a value that does not have three axes or has no pixel, has
    a number of channels its format does not store, or has a side longer than
    the format stores, 65535 pixels for JPEG and [2{^31} - 1] for PNG. *)

val load_image : string -> Nx.host Nx.uint8_t
(** [load_image path] is the image of the PNG or JPEG file [path], recognised by
    its contents, with the channels the file holds. A palette loads as its
    colours. A 16-bit sample keeps its high byte, and a sample of 1, 2 or 4 bits
    is scaled to 8 bits, rounded to nearest. A transparency ([tRNS]) chunk adds
    an alpha channel (PNG-3, 11.3.1.1). In a palette image, a pixel's alpha is
    its entry's alpha in the chunk, or 255 for an entry past the chunk's end. In
    any other image, it is 0 where the pixel's samples equal the chunk's at the
    file's bit depth, and 255 elsewhere. A grey JPEG file loads with one
    channel, and any other with three, CMYK and YCCK converted.

    Raises [Failure] as {!section-errors} states, and if the file is neither a
    PNG nor a JPEG stream, or if a checksum does not match. *)

val save_png : ?dpi:float -> ?srgb:bool -> string -> 'd Nx.uint8_t -> unit
(** [save_png path x] writes the image [x] to [path] as a PNG file of 8 bits per
    sample, the bytes [encode_png ?dpi ?srgb x]. PNG stores 1 to 4 channels, so
    every image {!load_image} reads from a PNG file saves back as itself.
    - [dpi] is the image's physical resolution in pixels per inch, written as a
      [pHYs] chunk of [Float.round (dpi /. 0.0254)] pixels per metre on both
      axes, so that viewers and printers show the image at its physical size.
      Without it the file states no physical size.
    - [srgb] is whether to write an [sRGB] chunk with the perceptual rendering
      intent, which states that the samples are sRGB. Defaults to [false].

    Raises [Invalid_argument] if [Float.round (dpi /. 0.0254)] is not in
    \[[1];[2{^31} - 1]\], the range of a PNG integer. *)

val save_jpeg : string -> 'd Nx.uint8_t -> unit
(** [save_jpeg path x] writes the image [x] to [path] as a baseline JPEG file at
    quality 90, the standard tables of T.81's Annex K scaled to 20%, with a
    colour image's chroma subsampled 4:2:0. JPEG stores 1 or 3 channels. *)

val encode_png : ?dpi:float -> ?srgb:bool -> 'd Nx.uint8_t -> string
(** [encode_png ?dpi ?srgb x] is the file {!save_png} writes, without touching
    the file system.

    Raises [Invalid_argument] as {!save_png} does. *)

(** {1:errors Errors}

    A load or a save raises [Failure] when its file cannot be read or written or
    breaks its format, and a load also when its path names no regular file, such
    as a directory or a FIFO. The message starts with the path, as in
    ["w.npy: the file ends inside its header"].

    A function raises [Invalid_argument] naming it for a value, a name or an
    argument its format cannot store, as in
    ["Nx_io.save_npy: npy has no bfloat16"]. A save also raises it for a value
    {!Nx.Repr.read} refuses: a traced value, a dead one, or one whose memory is
    held exclusive. A failed check pending on the value raises as
    {!Nx.Repr.read} states. *)
