(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrays to and from files.

    A file holds one array, as npy and text files do, or {e named} arrays, as
    npz, SafeTensors and GGUF files do. A file of named arrays is read and
    written as an {!Archive.t}, and each format's functions are named after it.

    {b Loading.} A loader reads the file's header and returns host values
    ({!Nx.Host}). An array whose bytes lie in the file in an order nx reads (C
    order, or Fortran order for npy), little-endian, at an offset that is a
    multiple of its element's alignment, is a value over the file's own pages:
    loading reads none of its bytes, except to check an npz entry's checksum,
    and the host reads them where they lie when a computation needs them.
    {!Nx.place} of such a value onto a device that addresses host memory, such
    as Metal's, shares the pages, and onto any other device copies them
    ({!Rig.reaches}). Any other array (a compressed npz entry, a big-endian one,
    one at a misaligned offset) is read at load into host memory of its own.

    A file stays open while a value over its pages is reachable, and its values
    see the file that was opened: a save to its path, which puts a new file
    there, leaves them their values.
    {b The file must not change in place while a value loaded from it is alive}:
    rewriting it may change their values, and a value over its pages dies by
    [SIGBUS], ending the process, when it reads a page that a truncation of the
    file cut off. A value read into memory of its own is not affected. On
    Windows a file whose pages are mapped can be neither deleted nor replaced
    until its values are collected.

    {b Saving.} A save takes a value of any set and writes its elements in C
    order, little-endian. It reads the value on the host for the extent of the
    save ({!Nx.Repr.read}): in the value's own memory where the host addresses
    it, and otherwise in a copy, made once the value's set has computed it. It
    leaves the value's memory unshared, so a later donation of the value may
    write its result there ({!Nx.donate}).

    A save holds on the host, beyond the value's own memory, a copy of each
    value the host does not address, a C-order copy of each value not in C
    order, and on a big-endian host a copy with each element's bytes swapped. It
    holds one value's copies at a time: a value off the host and out of C order
    takes up to twice its bytes, an npy file's peak is its value's, and an npz
    or SafeTensors file's is its largest entry's, since each entry is written
    before the next is read. A text file also holds its value's elements as an
    OCaml array.

    A save writes a new file beside its path, orders the file's bytes before
    every later change to the file system, then renames it over the path and
    syncs the directory. After a crash, the path holds what it held before or
    the whole new file, and a save that fails leaves it as it was and nothing
    beside it, except where the rename is refused: the new file is then kept,
    and the [Failure] names it. Every save takes [?overwrite], which defaults to
    [true]. With [~overwrite:false] the new file is linked to the path, which
    fails if the path names something; on a file system without hard links, such
    as FAT, the path is checked and then the file renamed, so a file another
    process puts at the path between the two is replaced.

    {b Errors.} A file that cannot be read or written, or whose contents break
    its format, raises [Failure] whose message starts with the file's path, as
    in ["w.npy: the file ends inside its header"]. A value, a name or an
    argument a format cannot hold raises [Invalid_argument] naming the function,
    as in ["Nx_io.save_npy: npy has no bfloat16"], and so does a value traced by
    an interpretation, which has no bytes to write. *)

(** {1:archives Archives} *)

(** Archives of named arrays.

    An archive is an immutable collection of values of one set, keyed by
    distinct, non-empty names. A value enters and leaves an archive through its
    structure ({!Nx.Ptree}): {!of_value} names each array by its path under a
    prefix, and {!to_value} reads a value back into the shape and the placement
    of one the program already has. Two values with different prefixes share one
    file:

    {[
    let checkpoint = Nx_io.Archive.union [
      Nx_io.Archive.of_value ~prefix:"model" model params;
      Nx_io.Archive.of_value ~prefix:"opt" opt state ] in
    Nx_io.save_safetensors path checkpoint;
    let params =
      Nx_io.Archive.to_value ~prefix:"model" model ~like:params
        (Nx_io.load_safetensors path)
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

    Asking for an entry reads none of its bytes. Bytes are never reinterpreted:
    only {!float} converts, and a block-quantised or sub-byte entry is read as
    [uint8] with {!tensor}.

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
  (** [float ~shape dt name a] is [a]'s entry [name] at [dt]: the entry as
      stored when its dtype is [dt], and otherwise its cast ({!Nx.cast}), which
      allocates. Both dtypes are float16, bfloat16, float32 or float64. An
      importer that ties two weights binds the result once and uses it twice.

      Raises [Invalid_argument] if [dt] is not one of the four. Raises [Failure]
      naming the entry if [a] has no entry [name], if its shape is not [shape],
      or if its dtype is not one of the four. An 8-bit float entry is read with
      {!tensor}, since its scales are other entries. *)

  val of_value : ?prefix:string -> ('s, 'd) Nx.Ptree.t -> 's -> 'd t
  (** [of_value ~prefix s x] has one entry for each array [s] walks in [x],
      fixed arrays included, named by [prefix], a dot and its path
      ({!Nx.Ptree.Path.to_string}), as ["model.blocks.0.w"]. An empty [prefix]
      (the default) and its dot are left out, and so are the dot and the empty
      path of an array at the root. The entries are [x]'s arrays; nothing is
      copied.

      Raises [Invalid_argument] if two arrays of [x] have one name, as in
      ["Nx_io.Archive.of_value: w: two arrays have this name"], or if a name is
      empty, as that of an array at the root without a prefix. *)

  val to_value : ?prefix:string -> ('s, 'd) Nx.Ptree.t -> like:'s -> 'e t -> 's
  (** [to_value ~prefix s ~like a] is [like] with each array replaced by [a]'s
      entry of its name under [prefix], as {!of_value} names it, placed where
      [like]'s array lies ({!Nx.place}). [like] gives the structure, the dtypes,
      the shapes and the placements, and its arrays are discarded: a value is
      read back with [like]'s list lengths, option presences and cases. For
      values [x] and [y] with equal {!Nx.Ptree.val-visits}, dtypes, shapes and
      placements, [to_value ~prefix s ~like:y (of_value ~prefix s x)] is [x].

      [like] owns [prefix] and every name that starts with [prefix] and a dot,
      and every name with an empty [prefix], the default. Each entry it owns
      must be named by one of its arrays. Entries it does not own are ignored
      and never moved, so one file holds several values.

      Raises [Failure], naming the entry, before any entry is placed:
      - if an entry [like] names is missing, as in
        ["Nx_io.Archive.to_value: model.l1.w: no entry in the archive, an array
         in the value"];
      - if its shape or dtype differs, as in
        ["Nx_io.Archive.to_value: model.l1.w: shape [3] in the archive, [2; 3]
         in the value"];
      - if an entry [like] owns is named by none of its arrays, as in
        ["Nx_io.Archive.to_value: blocks.1.b: an entry in the archive, no array
         in the value"]. A model of one block refuses a file of 24.

      Nothing is converted: a value that states another dtype than the archive
      fails instead of narrowing its state, and {!float} converts by name.

      Raises [Invalid_argument] as {!of_value} does if [like]'s names are not
      distinct and non-empty, and, naming the entry, if an array of [like] is a
      value of every set, which has no placement. *)
end

(** {1:numpy NumPy formats}

    An npy file holds one array, and an npz file is a zip archive of npy files,
    each entry named by its array's name and [.npy]. npy holds the dtypes
    float16, float32, float64, the signed and unsigned integers of 8 to 64 bits,
    complex64, complex128 and bool. Loading reads the format's versions 1.0, 2.0
    and 3.0, in either byte order and either axis order, and an npz entry stored
    or compressed with DEFLATE.

    The format is NumPy's NEP 1,
    {{:https://numpy.org/neps/nep-0001-npy-format.html}A simple file format for
     NumPy arrays}, and the zip container is PKWARE's
    {{:https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT}APPNOTE}
    without encryption or several disks. *)

val load_npy : string -> Nx.host Nx.packed
(** [load_npy path] is the array of the npy file [path]. *)

val save_npy : ?overwrite:bool -> string -> ('v, 's, 'd) Nx.t -> unit
(** [save_npy path x] writes [x] to the npy file [path], in version 1.0, or 2.0
    where its header needs more than 65,535 bytes.

    Raises [Invalid_argument] if npy has no dtype for [x]'s; for [bit], the
    message names [Nx.cast Nx.bool], which stores a byte per element. *)

val load_npz : string -> Nx.host Archive.t
(** [load_npz path] is the arrays of the npz file [path], named without the
    [.npy] suffix. Entries of other names are ignored.

    Raises [Failure] also if a checksum does not match its entry. *)

val load_npz_entry : name:string -> string -> Nx.host Nx.packed
(** [load_npz_entry ~name path] is the array [name] of the npz file [path], and
    reads no other entry's data.

    Raises [Failure] also if [path] has no entry [name], or its checksum does
    not match. *)

val save_npz : ?overwrite:bool -> string -> 'd Archive.t -> unit
(** [save_npz path a] writes [a]'s arrays to the npz file [path]. Each entry is
    compressed with DEFLATE if that shrinks its first 64 KiB, and stored
    otherwise.

    Raises [Invalid_argument] if a name is not valid UTF-8, starts with [/],
    holds a backslash, a NUL byte or an empty, [.] or [..] component, or names a
    drive, or if npy has no dtype for an entry's, naming it. *)

(** {1:safetensors SafeTensors}

    A SafeTensors file is a JSON header that names each array with its dtype,
    shape and byte range, then the arrays' bytes, as
    {{:https://github.com/huggingface/safetensors}the format's README}
    specifies. It holds bool, the signed and unsigned integers of 8 to 64 bits,
    float8_e4m3fn, float8_e5m2, float16, bfloat16, float32 and float64. *)

val load_safetensors : string -> Nx.host Archive.t
(** [load_safetensors path] is the arrays of the SafeTensors file [path], by
    name.

    An entry whose dtype nx lacks is its bytes, at [uint8]: [F8_E8M0] keeps the
    entry's shape, and [F4], [F6_E2M3] and [F6_E3M2], whose elements are
    narrower than a byte, have shape [[| n |]] with [n] the entry's size in
    bytes. An entry with no elements is an empty array of its shape.

    Raises [Failure] if the header is malformed, longer than 100 MB, names an
    array twice or with an empty name, or if the file's length differs from the
    one its header describes, as a partial download's does. *)

val save_safetensors : ?overwrite:bool -> string -> 'd Archive.t -> unit
(** [save_safetensors path a] writes [a]'s arrays to the SafeTensors file
    [path], ordered as the format's reference implementation orders them: by
    dtype, in the order U64, I64, F64, F32, U32, I32, BF16, F16, U16, I16,
    F8_E4M3, F8_E5M2, I8, U8, BOOL, then by name.

    Raises [Invalid_argument] if SafeTensors has no dtype for an entry's, naming
    it. *)

(** {1:gguf GGUF} *)

(** GGUF files.

    A GGUF file holds a model's arrays and its metadata: key-values that
    describe the architecture, the tokenizer and the file itself, as
    {{:https://github.com/ggml-org/ggml/blob/master/docs/gguf.md}GGML's
     specification} defines them. *)
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

  (** The type for tensor types. Each array of a file is stored as one of them:
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
    dtype : dtype;  (** The type the array is stored as. *)
    shape : int array;
        (** The array's logical shape, in row-major order: the reverse of the
            dimensions the file lists, whose first varies fastest. *)
  }
  (** The type for an array's description in the file. *)

  type t
  (** The type for the contents of a GGUF file. *)

  val version : t -> int
  (** [version g] is [g]'s format version, [2] or [3]. *)

  val metadata : t -> (string * value) list
  (** [metadata g] is [g]'s key-values, in file order. *)

  val tensors : t -> Nx.host Archive.t
  (** [tensors g] is [g]'s arrays, by name. *)

  val info : string -> t -> tensor_info
  (** [info name g] is the description of [g]'s array [name].

      Raises [Failure] if [g] has no array [name]. *)
end

val load_gguf : string -> Gguf.t
(** [load_gguf path] is the contents of the GGUF file [path], of version 2 or 3,
    in little-endian byte order.

    An array of a scalar type has its {!Gguf.tensor_info.shape} and the dtype of
    the same name: float32, float16, bfloat16, float64, int8, int16, int32 or
    int64. An array of a block format is its bytes, at [uint8], of its logical
    shape with the last extent replaced by the size of a row in bytes: a Q8_0
    row of 64 elements is two blocks of 34 bytes, so a Q8_0 array of logical
    shape [[|m; 64|]] loads at shape [[|m; 68|]]. An array with no elements is
    an empty array of its shape.

    Raises [Failure] if [path] does not start with the GGUF magic, if its
    version is not 2 or 3 or its byte order is big-endian, if its header is
    malformed or cut short, names a key or an array twice, or gives an array an
    empty name, a type this function does not know, a row that is not a whole
    number of blocks or a position that is not a multiple of the alignment, or
    if the file ends before an array's data. *)

(** {1:text Text}

    A text file holds a vector or a matrix: one row per line, its fields
    separated by a separator. Text holds every dtype but the complex ones. *)

val load_txt :
  ?sep:string ->
  ?comments:string ->
  ?skiprows:int ->
  ?max_rows:int ->
  string ->
  ('v, 's) Nx.dtype ->
  ('v, 's, Nx.host) Nx.t
(** [load_txt path dt] is the array of the text file [path] at [dt].

    The first [skiprows] lines are skipped, then each line is a row of fields
    separated by [sep], except blank lines and lines that start with [comments],
    and at most [max_rows] rows are read. A file of one row or one column loads
    as a vector, and any other as a matrix of shape [[|rows; columns|]]. [sep]
    defaults to [" "], where runs of spaces separate fields, [comments] to
    ["#"], [skiprows] to [0], and [max_rows] to every row.

    A float field is a sign, decimal digits with a point, and an exponent of [e]
    or [E], a sign and digits, each part but the digits optional, as [-1.5e3],
    [5.] or [.5], or after the sign [nan], [inf] or [infinity] in any case. It
    is read as the float64 nearest its value, then stored as
    {!Nx.Dtype.of_float} says. An integer field is a sign and decimal digits, in
    [dt]'s range. A bool or bit field is [true] or [false] in any case, or a
    float field, [true] where it is not zero. Hexadecimal, underscores and other
    spellings are no literal.

    Raises [Invalid_argument] if [dt] is complex, [skiprows] is negative or
    [max_rows] is not positive. Raises [Failure] if the file has no row, rows of
    different lengths, or a field that is no literal of [dt] or lies outside its
    range. *)

val save_txt :
  ?sep:string ->
  ?newline:string ->
  ?header:string ->
  ?footer:string ->
  ?comments:string ->
  ?overwrite:bool ->
  string ->
  ('v, 's, 'd) Nx.t ->
  unit
(** [save_txt path x] writes the 0-d value, vector or matrix [x] to the text
    file [path]: a 0-d value or a vector as one row, a matrix row by row. Each
    line of [header] and [footer] is written before and after the rows, prefixed
    with [comments]. [sep] defaults to [" "], [newline] to ["\n"] and [comments]
    to ["# "].

    A float is written in [%.18e] notation, its 19 significant digits rounded
    from its exact value, so it reads back as itself, NaN aside: every NaN is
    written [nan], and the infinities [inf] and [-inf]. An integer is written in
    decimal, an unsigned one unsigned, and a bool or bit as [1] or [0]. A value
    with no element writes no row, so its file does not load.

    Raises [Invalid_argument] if [x] is complex or has more than two axes. *)
