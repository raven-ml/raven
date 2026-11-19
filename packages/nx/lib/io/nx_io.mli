(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrays to and from files.

    A file holds one array, as npy and text files do, or several arrays under
    names, as npz, SafeTensors and GGUF files do. A name and its array are an
    {e entry}. Entries are read and written as an {!Archive.t}. A PyTorch
    checkpoint holds a Python value, whose arrays {!Pytorch.tensors} names,
    a TIFF file a chain of rasters, each with its tags ({!Tiff}), an HDF5
    file a tree of arrays named by paths ({!Hdf5}), and an audio file the
    samples of a sound ({!Audio}).

    An {e image} is a [uint8] array of shape [[|h; w; c|]], first row at the
    top, with [c] 1 for grey, 2 for grey and alpha, 3 for RGB and 4 for RGBA.

    {b Loading.} A load returns host values ({!Nx.Host}). Where the system can
    map the file, a load returns each array whose elements lie in the file in
    the host's byte order and in C order (or Fortran order, for npy, or at any
    strides, for a PyTorch checkpoint), at an offset that is a multiple of their
    alignment, as a value over the file's
    pages, a {e mapped} value. {!Nx.place} shares the pages with a device that
    addresses host memory and copies them to any other ({!Rig.reaches}). A
    donation of a mapped value writes its result to new memory ({!Nx.donate}).

    Any other array of a file the system maps, such as a compressed npz entry, a
    misaligned one or an image, is read at load into memory of its own, where
    its donation may write its result. A compressed npz entry takes twice its
    bytes while it loads: its inflated bytes and their copy. The arrays of a
    file the system cannot map are read at load into memory. An entry of a dtype
    nx lacks loads as its bytes at [uint8], at the shape its format's section
    states, and an entry with no elements is an empty array of that shape.

    An npz file and a PyTorch checkpoint are zip archives. A load checks each
    zip entry it reads against the CRC-32 the archive states for it, as
    {{:https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT}APPNOTE}
    4.1.5 requires, mapped entries included, so it reads the bytes of each such
    entry once. An entry whose stated CRC-32 is 0 is read unchecked, since
    PyTorch states 0 when told to compute none.

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

    A save copies onto the host a value off the host or out of C order, which
    takes up to twice its bytes. On a big-endian host, it first swaps the bytes
    of a value whose dtype is wider than a byte into new memory, on the value's
    set. An npz or SafeTensors save writes each entry before it reads the next.

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
    elements of each value it returns that is not mapped. {!Tiff.samples}
    and {!Tiff.window} block until they have read their IFD's chunks, from
    the file {!load_tiff} opened or the partial file {!decode_tiff} read,
    {!Hdf5.values}, {!Hdf5.window} and {!Hdf5.rows} until they have read
    their dataset's B-tree nodes and chunks, from the file {!load_hdf5}
    opened or the partial file {!decode_hdf5} read, and {!Audio.samples},
    {!Audio.window} and each block of {!Audio.blocks} until they have read
    their samples. A save blocks until the value's set has computed it and
    the new file is at its path. Domains may load one path concurrently.
    Concurrent saves to one path each write a whole file, and the last
    rename wins.

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

    An importer run by {!Archive.import} refuses a file that holds an entry it
    did not read.

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

  val import : ?unread:(string -> bool) -> ('d t -> 'a) -> 'd t -> 'a
  (** [import ~unread f a] is [f a'], where [a'] holds [a]'s entries and
      notes each name read from it by {!find}, {!tensor}, {!float}, and
      {!to_value} for the entries it reads. {!names} reads none. After [f]
      returns, [import] raises if an entry of [a] was not read and [unread]
      (default: none) does not accept its name. So a file with an extra or
      misnamed layer is refused. An entry left out on purpose is named by
      [unread], or read and dropped.

      This call notes no read of [a] itself, and each call notes afresh.
      Reads of [a'] made from several domains are all noted. Within [f], an
      [import] of [a'] notes its reads in both calls. An archive made from
      [a'] by {!union} is a new archive, whose reads are not noted. Reads
      after [f] returns are not checked.

      Raises [Failure] naming the first entry in name order that was neither
      read nor accepted, and how many there are, as in
      ["Nx_io.Archive.import: dec.9.fc4.w: an entry in the archive, never
       read (1 of 3)"]. What [f] raises passes on as raised, and nothing is
      checked. *)

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

      Raises [Invalid_argument] as {!of_value} does for [like]'s names. *)
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

val load_npy_rows :
  string -> row:int array -> rows:int array -> Nx.host Nx.packed
(** [load_npy_rows path ~row ~rows] is, for each [j] in order, the rows
    [row.(j)] to [row.(j) + rows.(j) - 1] along axis 0 of the npy file
    [path]'s array, its {e window} [j], joined along axis 0: of shape [k]
    followed by the array's other extents, [k] the sum of [rows]. Its elements
    are those {!load_npy} gives, in memory of its own, so a donation may write
    its result there. Windows may overlap and come in any order, and a window
    of no row adds none. Row [i] alone is the window [row.(j) = i],
    [rows.(j) = 1].

    It reads the file's header, then the bytes of the windows' rows and no
    others, each window one read of the file, every window at once in one
    copy ({!Rig_disk.section-copies}), never through a mapping of the file.
    So it reads from storage only the pages that hold those bytes, whether or
    not the system holds the file in memory, and a program reads all the
    windows it has in one call. {!load_npy} is the faster read of rows that
    lie together, and of a file the system holds in memory that the program
    reads again: its mapped pages are read with no copy. Each call opens
    [path] and reads its header again, so it reads the file the path names at
    the call.

    Raises [Invalid_argument] if [row] and [rows] differ in length, or, naming
    [j], if [row.(j)] or [rows.(j)] is negative or window [j] ends past the
    array's first extent, and if the windows hold more than 2{^53} elements,
    or rows of an array whose row holds none, having read no row. Raises
    [Failure] as {!section-errors} states, and for an array of no axis and for
    one in Fortran order whose row holds more than one element, as such a
    row's elements lie apart in the file. *)

val save_npy : string -> ('v, 's, 'd) Nx.t -> unit
(** [save_npy path x] writes [x] to the npy file [path].

    Raises [Invalid_argument] if npy does not store [x]'s dtype. For [bit], the
    message names [Nx.cast Nx.bool], which stores a byte per element. *)

val save_npy_concat :
  string -> ('v, 's) Nx.dtype -> int array -> ('v, 's, 'd) Nx.t Seq.t -> unit
(** [save_npy_concat path dt cell blocks] writes to the npy file [path] the
    array of dtype [dt] that is [blocks] joined along axis 0: of shape [k]
    followed by [cell], [k] the sum of the blocks' extents along axis 0. It
    forces a block and writes it before it forces the next, so it holds one
    block at a time, and an array larger than memory is saved from blocks
    that fit. A block may be a mapped value, so
    [save_npy_concat path dt cell (List.to_seq shards)] joins the files of
    [shards] into one, reading each page by page.

    Raises [Invalid_argument] if npy does not store [dt], if [cell] has a
    negative extent or 32 axes or more, or if a block's shape is not some
    [k] followed by [cell]. What forcing [blocks] raises passes on as raised.
    Each leaves [path] as it was and nothing beside it. *)

val load_npz : string -> Nx.host Archive.t
(** [load_npz path] is the entries of the npz file [path]. Each zip entry whose
    name ends in [.npy] holds one, named by the part before [.npy], which must
    not be empty. Entries of other names are not read; their records must still
    be well-formed.

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
    an entry of [n] bytes. The header may hold the file's {e metadata}, string
    key-values such as a model's configuration. *)

val load_safetensors : string -> Nx.host Archive.t
(** [load_safetensors path] is the entries of the SafeTensors file [path].

    Raises [Failure] as {!section-errors} states, and:
    - if the header is malformed or longer than 100 MB;
    - if it is not UTF-8, names an array twice or gives one an empty name;
    - if the file's length differs from the one its header states, as a partial
      download's does. *)

val load_safetensors_metadata : string -> (string * string) list
(** [load_safetensors_metadata path] is the metadata of the SafeTensors file
    [path], its header's ["__metadata__"] key-values in the header's order:
    [[]] if the header has none. It reads the header and none of the arrays'
    bytes.

    Raises [Failure] as {!load_safetensors} does, and if a key is given
    twice. *)

val save_safetensors :
  ?metadata:(string * string) list -> string -> 'd Archive.t -> unit
(** [save_safetensors ~metadata path a] writes [a]'s entries to the
    SafeTensors file [path] with the metadata [metadata] (default [[]]),
    first, as the format's reference implementation writes it, sorted by key
    so that a save is deterministic, and left out if empty. The entries
    follow in the reference implementation's order: by dtype, in the order
    U64, I64, F64, C64, F32, U32, I32, BF16, F16, U16, I16, F8_E4M3, F8_E5M2,
    I8, U8, BOOL, then by name.

    Raises [Invalid_argument] naming the entry if SafeTensors does not store its
    dtype, or if its name is not UTF-8 or is ["__metadata__"], the key of the
    file's metadata, and naming the key if [metadata] gives it twice or a key
    or value is not UTF-8. *)

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
(** [load_gguf path] is the contents of the GGUF file [path]. The file's
    {e alignment} is its [general.alignment], or 32 without one. Each array's
    data may start at any multiple of the alignment within the data after the
    header, so two arrays may share bytes or leave bytes between them.

    Raises [Failure] as {!section-errors} states, and:
    - if the file does not start with the GGUF magic, its version is not 2 or 3,
      or its byte order is big-endian;
    - if its header is malformed or cut short, or names a key or an array twice;
    - if its [general.alignment] is not a [uint32] power of two;
    - if it gives an array an empty name, a type this function does not know, a
      row that is not a whole number of blocks, or a position that is not a
      multiple of the alignment;
    - if the file ends before an array's data. *)

(** {1:pytorch PyTorch checkpoints} *)

(** PyTorch checkpoints.

    A PyTorch checkpoint is the file [torch.save] writes. It holds one Python
    value, often a dict of arrays, PyTorch's {e tensors}, with data beside
    them such as hyperparameters. The value is a {e pickle}, a program for
    Python's unpickling machine that
    {{:https://github.com/python/cpython/blob/main/Lib/pickletools.py}
     pickletools} documents. The machine has no loops, tests or arithmetic.
    Its only code is its {e calls}: a {e global}, an object the pickle names
    by module and name, applied to arguments, after which the machine may set
    the result's state and add items and entries to it, as
    {{:https://peps.python.org/pep-0307/}PEP 307} defines. A tensor's
    elements lie in a {e storage}, a run of elements of one dtype that several
    tensors may view at their own offsets, shapes and strides.

    Since PyTorch 1.6 the file is a zip archive whose entries lie under the
    directory of the first: the pickle [data.pkl], storage [k] as [data/k],
    and [byteorder], [little] or [big], little-endian without it, as
    {{:https://docs.pytorch.org/docs/stable/notes/serialization.html}PyTorch's
     serialization notes} lay out. Before, and with
    [_use_new_zipfile_serialization=False], it is the {e legacy} format: a
    stream of pickles, then the storages in the order their keys list them. A
    load recognises either by its contents, and refuses a TorchScript archive,
    which [torch.jit.load] reads, and the tar archive of PyTorch before 0.1.10.

    A pickle may use pickletools' opcodes of protocols 1 to 5. It may not use
    [POP], [POP_MARK] and [DUP], which discard or copy a value, [EXT1], [EXT2]
    and [EXT4], which index the writer's registry, [NEXT_BUFFER], whose buffer
    lies outside the file, [INST], [OBJ], or protocol 0's text opcodes, which
    PyTorch cannot read a tensor with either. [INT] and [LONG], which protocol
    1 writes bools and large ints with, are read, [LONG] up to the 4300 digits
    {{:https://docs.python.org/3/library/stdtypes.html#int-max-str-digits}
     Python reads}. A value may not gain a state, items or entries once
    another holds it, so none is cyclic.

    A load runs nothing. It reads a closed set of calls, the
    {e vocabulary}, as the values they build: [collections.OrderedDict] and
    [Counter] as dicts, [builtins.set] and [frozenset] as sets, [complex],
    [_codecs.encode], [bytes] and [bytearray], and [torch.Size] as a tuple;
    and [torch._utils._rebuild_tensor], [_v2], [_v3] and [_rebuild_parameter]
    as tensors, with empty [backward_hooks] and [metadata]. A call of the
    vocabulary given anything else is refused. Any other call is an
    {e opaque} value, {!Pytorch.Object}, which keeps its callable's name.
    Globals read as Python 3 reads them, Python 2's [__builtin__] and
    [copy_reg] as [builtins] and [copyreg].

    {b What a load returns.} Each tensor is exactly what its call and storage
    describe. Running an opaque call could change other values, as code that
    replaced PyTorch's own functions would. A load assumes that such a call
    builds a new object from what it is given and changes nothing else, no
    global included, and refuses a file where it could change a value it was
    not given: a dict, list, set, object, tensor or storage given to it, or
    given as a state, must be reachable only through it.

    {b Tensors.} A tensor is a view of its storage. A storage's persistent ids
    must state one dtype and its entry's length, unless it is empty. It loads
    once, and its tensors share it: mapped when its entry is stored in the
    host's byte order at a multiple of its elements' alignment, as PyTorch
    aligns entries since 1.6, and otherwise read into memory of its own,
    which a tensor's donation may write only if no other value reads it
    ({!Nx.donate}). Storages of float64, float32, float16, bfloat16,
    float8_e4m3fn, float8_e5m2, the integers of 8 to 64 bits, complex and bool
    load at that dtype, and other dtypes as bytes at [uint8], with a last axis
    of an element's bytes when wider than one. A tensor's [requires_grad],
    [Parameter] class and device are not kept.

    This reads a checkpoint [{"model": state_dict, "args": {"dim": 512}}]:
    {[
    let ckpt = Nx_io.load_pytorch "student.pt" in
    let dim =
      match Nx_io.Pytorch.find [ "args"; "dim" ] ckpt with
      | Some (Nx_io.Pytorch.Int d) -> d
      | _ -> failwith "student.pt: no int args.dim" in
    match Nx_io.Pytorch.find [ "model" ] ckpt with
    | Some state ->
        Nx_io.Archive.float ~shape:[| dim; dim |] Nx.float32 "layers.0.weight"
          (Nx_io.Pytorch.tensors state)
    | None -> failwith "student.pt: no model"
    ]} *)
module Pytorch : sig
  (** The type for Python values. A value holds once each tuple, list, set or
      dict that the file holds in two values, or twice in one, so a walk that
      does not expand a {!Ref} takes time linear in the file's length. A
      dict's keys and a set's elements are [Null], bools, numbers, strings,
      bytes, globals and tuples of these, pairwise unequal as Python compares
      them, so [1], [1.0], [True] and [1+0j] are one key. *)
  type value =
    | Null  (** [None]. *)
    | Bool of bool
    | Int of int  (** An [int] within OCaml's [int] range. *)
    | Long of string
        (** An [int] outside OCaml's [int] range: the fewest bytes of its
            two's complement, little-endian, as the pickle's [LONG1] holds
            it. *)
    | Float of float
    | Complex of Complex.t
    | String of string
        (** A [str], in UTF-8, which may hold the encodings of surrogate code
            points, as Python writes a lone one. *)
    | Bytes of string  (** A [bytes] or a [bytearray]. *)
    | Tuple of value list
    | List of value list
    | Set of value list  (** A [set] or a [frozenset], in the file's order. *)
    | Dict of (value * value) list
        (** A dict, an [OrderedDict] or a [Counter], in insertion order. An
            [OrderedDict]'s attributes, such as a state dict's [_metadata],
            are not kept. *)
    | Tensor of Nx.host Nx.packed
    | Global of string
        (** A global no call applies, by its dotted name, as
            ["torch.float32"]. *)
    | Object of string
        (** An opaque value, by the dotted name of the global its call
            applies, as ["argparse.Namespace"], or of the call that made its
            callable. An object the pickle's [NEWOBJ] makes is named by its
            class. *)
    | Shared of int * value
        (** A tuple, list, set or dict that the file holds in two values, or
            twice in one, where a depth-first walk of the value first meets
            it, a dict's key before its value. The numbers count from 0 in
            that order. *)
    | Ref of int
        (** The {!Shared} value of this number, at each later place that
            holds it. *)

  val tensors : value -> Nx.host Archive.t
  (** [tensors v] is the archive of [v]'s tensors, each named by its path from
      [v]: a dict's string or [Int] key, or a list's or a tuple's index in
      decimal, joined by dots, as ["model.layers.0.weight"]. A tensor the file
      shares is under each of its paths. Nothing is copied. It takes time
      linear in [v]'s size, its {!Ref}s unexpanded, and in its names' total
      length, which may be quadratic in the file's.

      Raises [Failure] naming the path:
      - if [v] holds an {!Object}, whether or not its call was given a
        tensor, since an [Object] keeps only its name: apply [tensors] to the
        value that holds the weights, such as a state dict;
      - if a {!Ref} refers to a {!Shared} that holds a tensor, which would
        have two names, or to one [v] does not hold;
      - if a tensor lies under a key that is not a string or an [Int], or in a
        set;
      - if two tensors have one name, as ["a.b"] for
        [{"a.b": x, "a": {"b": y}}], or a tensor has the empty name, as [v]
        itself does. *)

  val find : string list -> value -> value option
  (** [find path v] is the value at [path] from [v], each step the value of
      a dict's [String] key, as Python's [v[k1][k2]] reads it, or [None] if
      a dict on the way has no such key. A {!Shared} value reads as the
      value it holds, and a {!Ref} as the [Shared] value of its number in
      [v], so the result is neither. [find [] v] is [v] so read. A value
      found may hold [Ref]s to [Shared] values outside it, so a path is
      read from the value the load returned, whole: [find (a @ b) v], not
      [find b] of [find a v]. Only a
      [String] key is a step: the keys [Int 1] and [Bytes "a"] are not the
      steps ["1"] and ["a"]. It takes time linear in the dicts on the way,
      and in [v]'s size, its [Ref]s unexpanded, for each [Ref] it meets.

      Raises [Failure] naming the path so far, as
      ["Nx_io.Pytorch.find: args: an object argparse.Namespace, not a dict"]:
      - if a step reaches a value that is not a dict;
      - if a [Ref] refers to a [Shared] value [v] does not hold. *)
end

val load_pytorch : string -> Pytorch.value
(** [load_pytorch path] is the value of the PyTorch checkpoint [path], as
    {!Pytorch} states.

    Raises [Failure] as {!section-errors} states, naming the entry and the
    pickle's byte where it applies, as
    ["w.pt: data.pkl: byte 1874: POP discards a value"]:
    - if the file is neither format, or is a TorchScript or tar archive;
    - if a pickle is malformed, uses an opcode it may not, adds to a value
      another holds, or holds a key that {!Pytorch.type-value} does not admit
      or that equals another;
    - if a call of the vocabulary is given what it does not take, or a value
      given to an opaque call or as a state is reachable elsewhere;
    - if a storage's ids, entry, length, dtype or byte order are not as
      {!Pytorch} states, a legacy view's key names two views or a storage,
      or a tensor's view reaches past its storage;
    - if a checksum does not match its entry. *)

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
    [comments] after blanks, [comments] itself trimmed of blanks. An empty or
    blank [comments] marks no line as a comment. Each other line is a row that
    [sep] divides into fields, each trimmed of blanks. A run of spaces is one
    separator for the default [sep], [" "]. With any other non-empty [sep] an
    empty field is a field, so with [sep] [","], ["1, 2"] has two fields and
    ["1,,2"] three. With [sep] [""] each line is one field. A single row or
    column loads as a vector, and more as a matrix. [comments] defaults to ["#"]
    and [skiprows] to [0].

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
    rejects. A save also allocates [x]'s elements as an OCaml array. The file
    holds the bytes [encode_txt ?sep x].

    Raises [Invalid_argument] if [x] is complex or has more than two axes. *)

val encode_txt : ?sep:string -> ('v, 's, 'd) Nx.t -> string
(** [encode_txt ?sep x] is the text [save_txt ?sep path x] writes to [path],
    without touching the file system. It reads [x] as a save does, allocates
    what {!save_txt} allocates, and holds the text twice, in pieces and as the
    string it returns.

    Each row ends with ["\n"], so the encodings of values written one after
    another are the text of their rows in order, which {!load_txt} reads back as
    one array when every row has the same number of fields. This appends the
    rows of [x] to the file [path], creating it if needed:
    {[
    let append_txt path x =
      let text = Nx_io.encode_txt x in
      Out_channel.with_open_gen
        [ Open_wronly; Open_append; Open_creat; Open_binary ] 0o666 path
        (fun oc ->
          Out_channel.output_string oc text;
          Out_channel.flush oc)
    ]}
    The [flush] raises a failed write, which closing the channel would ignore.
    An append writes in place and syncs nothing, outside a save's replacement
    of its file: a failure or a crash may leave part of a row at the end of
    [path], and a crash of the system may also lose rows written since.

    Raises [Invalid_argument] as {!save_txt} does. *)

(** {1:images Images}

    Each save writes the format it is named after, whatever the path's
    extension. A PNG save encodes and writes the file a piece at a time, and
    holds about 3 MiB beyond the image and a row of it. {!encode_png} also holds
    the encoded file twice, in pieces and as the string it returns. A JPEG save
    encodes the file whole, into a buffer of at most twice its length or 16 KiB,
    whichever is larger, before it writes it.

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
    channel, and any other with three, CMYK and YCCK converted. Bytes after the
    image's end, a PNG file's [IEND] chunk or a JPEG file's EOI marker, are
    ignored.

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

(** {1:tiff TIFF}

    A TIFF file holds one or more {e IFDs} (image file directories), each a
    raster and the {e tags} that describe it: numbered fields of typed
    values, such as ImageWidth (256), or the GeoTIFF tags that place a
    raster on the Earth, which this module reads as any other. A raster is
    [h] rows of [w] pixels of [c] samples. A file cuts it into {e chunks}:
    strips of rows, or tiles of one size, padded past the raster's right
    and bottom edges. Each chunk is stored as it is or compressed, after a
    {e predictor} that replaces samples by their differences from the
    pixel before.

    The format is
    {{:https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf}TIFF
     6.0}, with the 64-bit offsets of
    {{:https://libtiff.gitlab.io/libtiff/specification/bigtiff.html}
     BigTIFF}, the Deflate compression of
    {{:https://www.alternatiff.com/resources/TIFFphotoshop.pdf}Adobe's TIFF
     technical notes} and the floating point predictor of
    {{:http://chriscox.org/TIFFTN3d1.pdf}Adobe's technical note 3}.

    {!load_tiff} reads the tags of every IFD of a classic TIFF or BigTIFF
    file of either byte order, and {!Tiff.samples} the raster of one whose
    tags state:
    {ul
    {- Compression (259) 1 (none), 5 (LZW, as {!Lz.Lzw} decodes it), or 8
       or its obsolete code 32946 (Deflate, a zlib stream a chunk);}
    {- Predictor (317) 1 (none), 2 (horizontal differencing) on samples of
       8, 16 or 32 bits or, for float samples, 3 (floating point), and 1
       with Compression 1;}
    {- BitsPerSample (258) and SampleFormat (339), the same for every
       sample: 1 to 32 bits but 24 unsigned (SampleFormat 1), read as the
       narrowest of [uint8], [uint16] and [uint32] that holds them; 16 or
       32 bits signed (2), as [int16] or [int32]; or 32 or 64 bits IEEE
       floating point (3), as [float32] or [float64];}
    {- PlanarConfiguration (284) 1, a pixel's samples together, or 2 with
       one sample a pixel;}
    {- FillOrder (266) 1, and a Photometric (262) other than 6 (YCbCr),
       whose samples may be subsampled;}
    {- strips, with StripOffsets (273), StripByteCounts (279) and
       RowsPerStrip (278), or tiles, with TileWidth (322), TileLength (323),
       TileOffsets (324) and TileByteCounts (325), each chunk of a byte or
       more.}}
    Any other value of these tags is refused, naming the tag. 24-bit
    unsigned samples are refused because TIFF 6.0 stores them as bytes,
    most significant first, while libtiff swaps them as integers in the
    file's byte order, so writers disagree in either byte order. Samples of
    8, 16, 32 or 64 bits are in the file's byte order. Samples of any
    other width are {e packed}, as TIFF 6.0's Compression 1 states: most
    significant bit first whatever the file's byte order, with each row of
    a chunk padded to a byte, before any compression. A field of a type
    {!Tiff.type-value} lacks is skipped, as TIFF 6.0 tells readers to.
    Every other tag is data, Orientation (274) among them, so a raster's
    first row is its file's first.

    A file's IFDs and the values that do not fit their entries must take
    no more bytes in all than the file holds, and so must each IFD's
    chunks, so that a load's tags take memory linear in the file's length,
    and a raster at most 2560 times its bytes, or 20480 times for packed
    samples.

    {!save_tiff} writes a raster of a sample or more, at most 65535
    samples a pixel and sides below 2{^32}, with its own tags: ImageWidth,
    ImageLength, BitsPerSample, Compression, SamplesPerPixel,
    PlanarConfiguration 1, SampleFormat, Predictor with [Deflate], and the
    strips' or the tiles' tags. Unless the tags it is given hold
    Photometric (262), it adds Photometric 1 (BlackIsZero) and, for more
    than one sample a pixel and unless they hold ExtraSamples (338),
    ExtraSamples of zeros (unspecified data). Its file is BigTIFF when a
    tag is LONG8 or SLONG8, or when the largest file it could write, each
    chunk of its bytes or their {!Lz.Zlib.bound}, passes 4 GiB, and classic
    TIFF otherwise. *)

(** TIFF IFDs and their tags. *)
module Tiff : sig
  (** The type for tag values, one case per TIFF field type, each the
      field's values in order. *)
  type value =
    | Byte of int array  (** BYTE (1): unsigned 8-bit integers. *)
    | Ascii of string
        (** ASCII (2): the field's bytes without its last NUL. A field of
            several strings keeps the NULs between them. *)
    | Short of int array  (** SHORT (3): unsigned 16-bit integers. *)
    | Long of int array  (** LONG (4): unsigned 32-bit integers. *)
    | Rational of (int * int) array
        (** RATIONAL (5): fractions, each a numerator and a denominator
            LONG. *)
    | Sbyte of int array  (** SBYTE (6): signed 8-bit integers. *)
    | Undefined of string  (** UNDEFINED (7): bytes. *)
    | Sshort of int array  (** SSHORT (8): signed 16-bit integers. *)
    | Slong of int array  (** SLONG (9): signed 32-bit integers. *)
    | Srational of (int * int) array
        (** SRATIONAL (10): fractions of two SLONGs. *)
    | Float of float array  (** FLOAT (11): binary32 floats. *)
    | Double of float array  (** DOUBLE (12): binary64 floats. *)
    | Long8 of int64 array
        (** LONG8 (16), BigTIFF's: unsigned 64-bit integers, held in
            [int64] by their bits. *)
    | Slong8 of int64 array
        (** SLONG8 (17), BigTIFF's: signed 64-bit integers. *)

  type t
  (** The type for one IFD of a TIFF file. It keeps the bytes it was read
      from: the file, as a mapped value does, and {b the file must not
      change in place while it is reachable}; or a partial file
      ({!section-partial}), which never changes. *)

  val tags : t -> (int * value) list
  (** [tags d] is [d]'s tags, each by its number, in file order. *)

  val samples : t -> Nx.host Nx.packed
  (** [samples d] is [d]'s raster, of shape [[|h; w; c|]] for [h] its
      ImageLength (257), [w] its ImageWidth (256) and [c] its
      SamplesPerPixel (277), at its samples' dtype. It is a mapped value
      where the raster lies in the file as one array in C order:
      uncompressed, in the host's byte order, in chunks as wide as the
      raster that follow each other from a multiple of a sample's bytes.
      Otherwise each call reads the file again, into memory of its own.

      For an IFD decoded from a partial file, it is {!window} of the whole
      raster, in memory of its own.

      Raises [Failure] as {!section-errors} states, naming the tag, if a
      tag the section lists is missing, of another type, or of a value or
      count the section does not read, or if a chunk lies past the file's
      end, has too few bytes or does not decode to its rows, as in
      ["l.tif: Compression (259) 5: tile 3 at byte 1024: malformed at
       byte 7: invalid code"], and naming the first range a partial file
      lacks of its chunks. *)

  val window :
    t ->
    row:int ->
    col:int ->
    rows:int ->
    cols:int ->
    (Nx.host Nx.packed, (int * int) list) result
  (** [window d ~row ~col ~rows ~cols] is [Ok x], [x] the samples of [d]'s
      rows [row] to [row + rows - 1] and columns [col] to [col + cols - 1],
      of shape [[|rows; cols; c|]] at {!samples}'s dtype, in memory of its
      own; or [Error rs], [rs] the ranges ({!section-partial}) of the chunks
      the window covers that [d]'s partial file lacks. It reads each chunk
      the window covers, whole, and no other: [x] is
      [Nx.slice [R (row, row + rows); R (col, col + cols)] (samples d)]
      where {!samples} returns, and [window] may return where {!samples}
      raises for a chunk the window does not cover. A window of no row or
      no column reads no chunk. An IFD {!load_tiff} read is never [Error].

      Raises [Invalid_argument] if the window does not lie in [d]'s raster,
      and [Failure] as {!samples} does, for its tags and every chunk's
      offset and count, and for each chunk the window covers that does not
      decode. *)

  (** The type for how {!save_tiff} stores chunks. *)
  type compression =
    | Uncompressed
    | Deflate
        (** Each chunk is a zlib stream ({!Lz.Zlib.encode}) after horizontal
            differencing for integer samples (Predictor 2), and after the
            floating point predictor for float samples (3). *)
end

val load_tiff : string -> Tiff.t list
(** [load_tiff path] is the IFDs of the TIFF file [path], in the order the
    file chains them, with their tags. It reads no sample.

    Raises [Failure] as {!section-errors} states, and if the file starts
    with no TIFF or BigTIFF header or names no IFD, if an IFD or a value
    lies past its end, if an IFD holds a tag twice, or if the chain comes
    back to an IFD. *)

val save_tiff :
  ?tile:int * int ->
  ?compression:Tiff.compression ->
  ?tags:(int * Tiff.value) list ->
  string ->
  ('v, 's, 'd) Nx.t ->
  unit
(** [save_tiff path x] writes [x], of shape [[|h; w; c|]] and a dtype
    {!Tiff.samples} reads, to [path] as a little-endian TIFF file of one
    IFD, its tags first and its chunks after them.
    - [tile] is the rows and the columns of a tile. Without it, the raster
      is cut into strips of about 8 KiB and at least a row, as TIFF 6.0
      recommends.
    - [compression] defaults to [Uncompressed], whose strips
      {!Tiff.samples} reads back as a mapped value.
    - [tags] are written beside the save's own ({!section-tiff}). A tag
      whose value is an offset into its file, such as SubIFDs (330),
      points nowhere in the new one.

    Raises [Invalid_argument] if TIFF does not store [x]'s dtype or shape,
    or a tile's side is not a positive multiple of 16, and naming the tag
    if a tag is not a SHORT, is the save's own or is given twice, or holds
    no value or one its type does not, such as a [Float] of 0.1. *)

(** {1:hdf5 HDF5}

    An HDF5 file is a tree of {e groups} whose leaves are {e datasets}, each
    an n-d array of one {e datatype}, as the HDF Group's
    {{:https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html}
     File Format Specification 3.0} defines them. A dataset is named by its
    {e path}, the names of the groups above it and its own, each after a
    [/], as ["/desi/spectrum_flux"]. Its elements lie in the file in C order
    as one block ({e contiguous}), inside its header ({e compact}), or in
    {e chunks}, boxes of one shape that tile the array, each stored alone
    and passed through the dataset's {e filters}. A group or a dataset also
    holds {e attributes}, small named arrays.

    A group holds {e links}, each a name and the object it leads to: a
    {e hard link} names the object, and a {e soft link} a path to it, from
    the root when the path starts with [/] and from the link's group
    otherwise. An object has a path for each way of reaching it from the
    root ([/]) along links. Paths are ordered name by name, so a path comes
    before the paths below it, and ["/a/b"] before ["/a-"].

    {!load_hdf5} reads files of superblock version 0 or 1 at byte 0, which
    is the format libhdf5 writes by default: object headers of version 1 or
    2, the second's checksums checked, and groups whose links lie in a
    symbol table or in link messages in their header. It refuses, naming
    the structure and the path of the object it is in, a later superblock,
    a user block before the superblock, a group whose links lie in a
    fractal heap, and an external link, which leads to another file. An
    object is read once, at the least of its paths, also when a cycle of
    links reaches it again. A soft link whose path leads to no object, or
    to another soft link of a cycle, is not listed.

    {!Hdf5.values}, {!Hdf5.window} and {!Hdf5.rows} read a dataset of the
    message versions libhdf5 writes in this format (a layout of version 3,
    a fill value of version 2, a filter pipeline of version 1), contiguous,
    compact or chunked over a version-1 B-tree, whose datatype is:
    {ul
    {- fixed-point or a bitfield of 1, 2, 4 or 8 bytes whose bits all hold
       its value, as [int8] to [uint64], a bitfield unsigned;}
    {- IEEE floating point of 2, 4 or 8 bytes, or bfloat16, as [float16],
       [float32], [float64] or [bfloat16];}
    {- complex, of two such float32 or float64 parts, as [complex64] or
       [complex128];}
    {- an enumeration, as its base type, so a boolean that h5py writes
       reads as [int8];}
    {- an array, as its base type with the array's axes after the
       dataset's;}
    {- a fixed-length string, an opaque type or a compound type, as its
       bytes at [uint8] with a last axis of its size, padding included.
       {!Hdf5.strings} cuts a fixed-length string's bytes to the string.}}
    in either byte order. Its filters may be deflate (1, a zlib stream a
    chunk), shuffle (2), Fletcher-32 (3), whose checksum is checked, and
    LZF (32000, {!Lz.Lzf}), at most one of deflate and LZF. A chunk whose
    filter mask skips a filter is read without it, as libhdf5 stores a
    chunk an optional filter did not shrink. An element never written, in
    a chunk or a contiguous block never stored, reads as the dataset's fill
    value, or zero bytes without one. A dataset of any other datatype, such
    as a variable-length string or a reference, layout or filter raises
    when it is read, naming its path and the feature, and the file's other
    datasets still read. *)

(** HDF5 datasets. *)
module Hdf5 : sig
  type t
  (** The type for one dataset of an HDF5 file. It keeps the bytes it was
      read from: the file, as a mapped value does, and {b the file must not
      change in place while it is reachable}; or a partial file
      ({!section-partial}), which never changes. *)

  val path : t -> string
  (** [path d] is the least of [d]'s paths. *)

  val shape : t -> int array
  (** [shape d] is [d]'s shape, from its dataspace. A scalar dataset's is
      [[||]].

      Raises [Failure] as {!values} does for a dataspace it does not read,
      such as a null one. *)

  val chunk : t -> int array option
  (** [chunk d] is the shape of [d]'s chunks, or [None] when [d] is not
      chunked. A window whose box is a union of chunks decodes no element
      it does not return.

      Raises [Failure] as {!values} does for a dataspace, datatype or
      layout it does not read. *)

  val attributes : t -> string list
  (** [attributes d] is the names of [d]'s attributes, in the order its
      header holds them.

      Raises [Failure] naming [d]'s path for attributes stored outside its
      header, in a fractal heap. *)

  val attribute : t -> string -> Nx.host Nx.packed
  (** [attribute d name] is the value of [d]'s attribute [name], at its
      datatype's dtype as for {!values}, in memory of its own.

      Raises [Failure] naming [d]'s path and [name] if [d] has no attribute
      [name], or for a datatype or dataspace {!section-hdf5} does not
      read. *)

  val values : t -> Nx.host Nx.packed
  (** [values d] is [d]'s array, of {!shape} followed by its datatype's
      axes. It is a mapped value where [d] is contiguous, in the host's
      byte order and at a multiple of its elements' alignment. Otherwise
      each call reads the file again, into memory of its own.

      For a dataset decoded from a partial file, it is {!window} of the
      whole array, in memory of its own.

      Raises [Failure] as {!section-errors} states, naming [d]'s path, for
      a datatype, layout or filter {!section-hdf5} does not read, for a
      chunk or contiguous block past the file's end, a chunk that does not
      decode to its bytes or fails its checksum, as in
      ["f.h5: /x: chunk at [8], byte 2091: filter 3 (Fletcher-32): the
       checksum does not match the data"], and naming the first range a
      partial file lacks. *)

  val window :
    t ->
    start:int array ->
    shape:int array ->
    (Nx.host Nx.packed, (int * int) list) result
  (** [window d ~start ~shape] is [Ok x], [x] the elements of [d] from
      [start] along each axis for [shape] elements, followed by its
      datatype's axes, in memory of its own; or [Error rs], [rs] the
      ranges ({!section-partial}) a partial file lacks of the B-tree nodes
      that index the chunks the box covers, or, once it holds them, of
      those chunks. It reads each chunk the box covers, whole, and no
      other; of a contiguous dataset, the run of elements of each row of
      the box along its last axis. [x] is [Nx.slice] of {!values} over the
      box where {!values} returns, and [window] may return where {!values}
      raises for a chunk the box does not cover. A box of no element reads
      nothing. A dataset {!load_hdf5} read is never [Error].

      Raises [Invalid_argument] if [start] or [shape] is not of [d]'s rank
      or the box does not lie in [d], and [Failure] as {!values} does, for
      each chunk the box covers. *)

  val rows :
    t ->
    row:int array ->
    rows:int array ->
    (Nx.host Nx.packed, (int * int) list) result
  (** [rows d ~row ~rows] is [Ok x], [x] for each [j] in order the rows
      [row.(j)] to [row.(j) + rows.(j) - 1] along axis 0 of [d], its window
      [j], joined along axis 0: of shape [k] followed by [d]'s other extents
      and its datatype's axes, [k] the sum of [rows], in memory of its own.
      Windows may overlap and come in any order, and a window of no row adds
      none. Or it is [Error rs], as {!window}'s, for a partial file. It
      decodes each chunk that holds a row of a window once, whole, and no
      other, and of a contiguous dataset reads each window's run of bytes.
      [x] is {!window}'s rows over each window, joined.

      Raises [Invalid_argument] if [row] and [rows] differ in length, if [d]
      is scalar, or, naming [j], if [row.(j)] or [rows.(j)] is negative or
      window [j] ends past [d]'s first extent, and if the windows hold more
      than 2{^53} elements, having read nothing. Raises [Failure] as
      {!values} does, for each chunk a window covers. *)

  val strings : t -> Nx.host Nx.packed -> string array
  (** [strings d x] is the fixed-length strings of [x], bytes read from [d]
      by {!values}, {!window} or {!rows}: one string per element of [x]
      without its last axis, in C order. Each is cut as [d]'s datatype
      states its padding: at its first NUL for null-terminated strings,
      without trailing NULs for null-padded ones, and without trailing
      spaces for space-padded ones. The bytes are returned as stored,
      whether the datatype names ASCII or UTF-8.

      Raises [Invalid_argument] if [d]'s elements, or an array datatype's
      base, are not fixed-length strings, or if [x] is not [uint8] with a
      last extent of their size. Raises [Failure] as {!values} does for a
      datatype it does not read, and for a padding the format reserves. *)

  (** Groups. *)
  module Group : sig
    type t
    (** The type for one group of an HDF5 file, which keeps the bytes it was
        read from as a dataset does. *)

    val path : t -> string
    (** [path g] is the least of [g]'s paths, ["/"] for the root. *)

    val attributes : t -> string list
    (** [attributes g] is the names of [g]'s attributes, as
        {!Hdf5.attributes} states for a dataset's. *)

    val attribute : t -> string -> Nx.host Nx.packed
    (** [attribute g name] is the value of [g]'s attribute [name], as
        {!Hdf5.attribute} states for a dataset's. *)
  end

  (**/**)

  (* For the tests: the checksum of version-2 object headers, Jenkins'
     lookup3 (hashlittle) of a string from the initial value [init]
     (default 0). *)

  val lookup3 : ?init:int -> string -> int
end

val load_hdf5 : string -> Hdf5.Group.t list * Hdf5.t list
(** [load_hdf5 path] is the groups and the datasets of the HDF5 file
    [path], each in the order of their paths, the root group first. It
    reads the superblock, the groups and each object's header, and no
    element.

    Raises [Failure] as {!section-errors} states, and naming the structure
    for a file {!section-hdf5} refuses, a structure past the file's end or
    the end of data its superblock states, a header whose checksum does not
    match, a group B-tree that is malformed or comes back to a node, or
    structures that take more bytes in all than the file holds. *)

(** {1:audio Audio}

    An {e audio file} holds [n] samples of each of [c] channels, taken at a
    {e rate} of samples a second. Its samples load as an array of shape
    [[|n; c|]], a {e row} per instant, the channels of a row in the file's
    order. {!load_audio} reads WAV and FLAC files, recognised by their
    contents. A load reads the file's header and keeps the file;
    {!Audio.samples}, {!Audio.window} and {!Audio.blocks} read samples from
    it.

    {b Samples.} A sample takes [k] bytes: a WAV file's {e container}, or a
    FLAC file's bit depth rounded up to whole bytes. An integer sample loads
    at [int16] for [k] of 2, [int32] for 3 and 4 and [int64] for 5 to 8; at
    [k = 1], at [uint8] from a WAV file, which stores it unsigned and offset
    by 128, and at [int8] from a FLAC file, which stores it signed. A
    sample of [b] significant bits ({!Audio.bits}) lies in the top [b] bits
    of its element, and the bits below are zeros: a 24-bit sample [v] loads
    as the [int32] [256 * v]. So an element's full scale is its dtype's,
    whatever the file's depth: [x /. 32768.] for [int16], [x /. 2{^31}] for
    [int32] and [(x - 128) /. 128.] for [uint8]. A float sample loads at [float32] or
    [float64] as stored, unclipped, of full scale 1. An A-law or μ-law
    sample loads at [int16] as the 13- or 14-bit value
    {{:https://www.itu.int/rec/T-REC-G.711-198811-I/en}ITU-T G.711}
    decodes it to.

    {b WAV.} A WAV file is a RIFF form [WAVE] of
    {{:https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/riffmci.pdf}
     Multimedia Programming Interface and Data Specifications 1.0}:
    chunks, each an identifier, a 32-bit size and that many bytes, with a
    zero byte after an odd size. Its [fmt ] chunk states the coding and
    comes before the [data] chunk, which holds the rows in order,
    little-endian. A load reads PCM, IEEE float, A-law and μ-law data,
    plain or as
    {{:https://learn.microsoft.com/en-us/windows/win32/api/mmreg/ns-mmreg-waveformatextensible}
     WAVEFORMATEXTENSIBLE}, and the RF64 files of
    {{:https://tech.ebu.ch/docs/tech/tech3306v1_1.pdf}EBU Tech 3306},
    whose [ds64] chunk states sizes past 4 GiB. A container is
    [nBlockAlign / nChannels] bytes, and a plain PCM file's must be its
    bits rounded up to bytes, as the specification states. Below a
    sample's valid bits its container holds zeros, as the specification
    states, and a read checks each sample whose valid bits are fewer than
    its container's. A load skips other chunks and reads nothing after
    [data]. A [data] size of [0xFFFFFFFF], which no valid RIFF file states
    and a writer that cannot seek back leaves, runs to the end of the
    file. On a little-endian host, {!Audio.samples} is a mapped value for
    PCM and float samples whose valid bits fill their containers of 1, 2,
    4 or 8 bytes, at an offset that is a multiple of their size.

    {b FLAC.} A FLAC file is the stream of
    {{:https://www.rfc-editor.org/rfc/rfc9639}RFC 9639} in its own
    container: the [fLaC] marker, metadata blocks, STREAMINFO first, then
    {e frames}, each a block of rows coded on its own. A load reads every
    stream the RFC defines, of 4 to 32 bits and 1 to 8 channels, whose
    rate, channels and depth stay those of STREAMINFO. Decoding checks each
    frame's CRC-8 and CRC-16, its rate, channels, depth, block and frame
    sizes and number against STREAMINFO, and each sample against its
    depth. A read that decodes every frame from the first checks
    STREAMINFO's MD5 unless it is 0. A read from row [r] decodes from the
    last seek point at or before [r] whose frame names the point's row and
    rows, or else from the first frame, so in a file without a seek table
    it decodes every frame before [r]. Bytes after the frames of
    STREAMINFO's length are ignored, and a frame that runs past the length
    raises. A length whose samples would take more than 2{^16} times the
    bytes after the metadata, more than any frame decodes to, is refused
    before anything is allocated.

    This reads a 16 kHz mono file as [float32] amplitudes, each the exact
    quotient by 32768:
    {[
    let a = Nx_io.load_audio "speech.flac" in
    assert (Nx_io.Audio.rate a = 16000 && Nx_io.Audio.channels a = 1);
    let x = Nx.unpack Nx.int16 (Nx_io.Audio.samples a) in
    Nx.div_s (Nx.cast Nx.float32 x) 32768.
    ]} *)

(** Audio files. *)
module Audio : sig
  type t
  (** The type for audio files. A value keeps the file it was loaded from,
      as a mapped value does: {b the file must not change in place while it
      is reachable}. A value is immutable and may be shared between
      domains. *)

  val rate : t -> int
  (** [rate a] is [a]'s samples a second of each channel, as the file
      states it. It may be 0, which RFC 9639 allows for data not sampled in
      time. *)

  val length : t -> int
  (** [length a] is [n], the rows of [a]'s samples. *)

  val channels : t -> int
  (** [channels a] is [c], the columns of [a]'s samples. *)

  val bits : t -> int
  (** [bits a] is the significant bits of each of [a]'s samples, from 1 to
      the width of {!samples}'s dtype: a WAV file's valid bits, or its PCM
      bits without them, a FLAC file's depth, 13 for A-law, 14 for μ-law,
      and the dtype's width for floats. *)

  val channel_mask : t -> int
  (** [channel_mask a] is the speakers [a]'s channels feed, as
      WAVEFORMATEXTENSIBLE's channel mask states them: bit [k] for the
      speaker of position [k], 0 front left, 1 front right, 2 front centre,
      3 low frequency, 4 back left, 5 back right, 6 front left of centre,
      7 front right of centre, 8 back centre, 9 side left, 10 side right,
      11 top centre, 12 top front left, 13 top front centre, 14 top front
      right, 15 top back left, 16 top back centre, 17 top back right.
      Channel [i] feeds the speaker of the [i]-th lowest set bit of bits 0
      to 17, and a channel past them none.

      It is the mask the file states, a WAV file's [dwChannelMask] or a
      FLAC file's [WAVEFORMATEXTENSIBLE_CHANNEL_MASK] comment. Otherwise it
      is [0x4] for one channel and [0x3] for two; for a FLAC file of three
      to eight, the order of RFC 9639's section 9.1.3 with its surround
      channels at the sides, [0x7], [0x33], [0x607], [0x60F], [0x70F] and
      [0x63F]; and 0 for a WAV file of more than two, the same value as a
      stated 0. *)

  val samples : t -> Nx.host Nx.packed
  (** [samples a] is [a]'s samples, of shape [[|length a; channels a|]] at
      the dtype {!section-audio} states. It is a mapped value where the
      section says so; otherwise each call reads or decodes the file again,
      into memory of its own.

      Raises [Failure] as {!section-errors} states, naming the row, if a
      sample has a set bit below its valid bits, and naming the frame by its
      first row and its byte for a FLAC file that breaks RFC 9639 or a check
      of {!section-audio}, and if the MD5 differs. *)

  val blocks : ?row:int -> t -> rows:int -> Nx.host Nx.packed Seq.t
  (** [blocks ~row a ~rows] is [a]'s samples from row [row], cut into
      blocks of [rows] rows, the last shorter, each in memory of its own:
      [Nx.slice [R (row, min (row + rows) n)] (samples a)] first, for [n]
      [length a], and so on to [n]. [row] defaults to [0], and [row = n]
      gives no block. Forcing a WAV block reads its rows' bytes and no
      others. A FLAC block decodes from the frame the previous one ended
      in, so a pass decodes each frame once. Blocks from row 0 check the
      MD5 when they give the last block. The sequence may be forced again,
      which reads again.

      Raises [Invalid_argument] if [rows < 1] or [row] is not in
      \[[0];[n]\]. Forcing a block raises as {!samples} does, for the
      bytes it reads. *)

  val window : t -> row:int -> rows:int -> Nx.host Nx.packed
  (** [window a ~row ~rows] is rows [row] to [row + rows - 1] of [a]'s
      samples, of shape [[|rows; channels a|]], in memory of its own. It is
      the first block of [blocks ~row a ~rows], and for no row an empty
      array that reads nothing. It may return where {!samples} raises, for bytes
      it does not read.

      Raises [Invalid_argument] if [row] or [rows] is negative or the
      window ends past [length a], and [Failure] as {!blocks} does. *)
end

val load_audio : string -> Audio.t
(** [load_audio path] is the WAV or FLAC file [path], recognised by its
    contents. It reads the header and no sample, except that it decodes
    every frame of a FLAC file whose STREAMINFO states no length, to count
    its rows; such a file's frames must run to its end.

    Raises [Failure] as {!section-errors} states, and:
    - if the file is neither a RIFF or RF64 form [WAVE] nor a FLAC stream
      with its marker, such as a RIFX, Ogg FLAC or headerless FLAC file;
    - for a WAV file, if [fmt ] is missing, malformed, given twice or after
      [data], or states a coding the section does not read, no channel, a
      container its bits do not fit or valid bits of 0; if [data] is
      missing, ends past the file or holds part of a row, such as the size
      [0x80000000] that arecord leaves in a file it writes to a pipe and
      the size [0x7ffff000] that sox once left, which state more bytes than
      such a file holds; or if an RF64 file has no [ds64] chunk first, or a
      chunk ends past the file;
    - for a FLAC file, if STREAMINFO is missing, not first, given twice or
      states sizes RFC 9639 forbids, if a block ends past the file or is of
      type 127, if a seek table or a Vorbis comment is malformed or given
      twice, if the length is refused as {!section-audio} states, or, for a
      file without a length, as {!Audio.samples} raises. *)

val save_wav :
  ?bits:int ->
  ?channel_mask:int ->
  string ->
  rate:int ->
  ('v, 's, 'd) Nx.t ->
  unit
(** [save_wav path ~rate x] writes [x], of shape [[|n; c|]], to [path] as a
    WAV file of [c] channels at [rate], which {!load_audio} reads back as
    [x], [rate], [bits] and [channel_mask]. [uint8], [int16], [int32] and
    [int64] are written as PCM, [float32] and [float64] as IEEE float.
    - [bits] is the significant bits of an integer sample, the top ones of
      its element, whose others must be zero. The container is the fewest
      bytes that hold [bits] and load back at [x]'s dtype, so [~bits:24]
      writes an [int32] in three bytes. Defaults to the dtype's width.
    - [channel_mask] defaults to {!Audio.channel_mask}'s for a WAV file of
      [c] channels that states none.

    The [fmt ] chunk is WAVEFORMATEXTENSIBLE for more than two channels,
    PCM of more than 16 bits, [bits] other than its container's or a mask
    other than the default, and plain otherwise, with a [fact] chunk for
    floats. A [JUNK] chunk before [data] puts containers of 4 or 8 bytes at
    a multiple of their size, so a load maps them. The file is RF64 when a
    32-bit size would reach [0xFFFFFFFF], and RIFF otherwise.

    Raises [Invalid_argument] if [x] does not have two axes, has no channel
    or a row of more than 65535 bytes, if WAV does not store its dtype,
    [int8] among them, since WAV stores 8-bit samples unsigned, if [rate]
    is not in \[[1];[2{^32} - 1]\] or its bytes a second are not below
    [2{^32}], if [bits] is not in \[[1]; width\] or is given for floats, or
    if [channel_mask] is negative or not below [2{^32}]; and, naming the
    row, if a sample has a set bit below [bits], leaving [path] as it
    was. *)

(** {1:partial Partial files}

    A {e partial file} is what a program holds of a file it reads in
    pieces, such as a remote file read by byte ranges: the file's length,
    and its bytes at some of its offsets. The {e partial readers},
    {!decode_tiff}, {!Tiff.window}, {!decode_hdf5}, {!Hdf5.window} and
    {!Hdf5.rows}, read only the bytes a partial file holds, and return the
    bytes it lacks as {e ranges}: [(offset, length)] pairs sorted by
    offset, disjoint, merged where they touch, and never empty, the same
    list for the same bytes. The caller fetches the ranges, adds them, and
    calls again. No byte a partial file lacks is read as any value. How
    many bytes to fetch first, whether to merge ranges across gaps, and how
    to fetch them are the caller's.

    Two laws tie a partial file to the whole file it holds part of:
    {ul
    {- {e Agreement.} A partial reader on a partial file returns [Error]
       with ranges, or returns or raises exactly as on the whole file.
       Which of the last two it does never depends on a byte the partial
       file lacks.}
    {- {e Monotony.} Adding bytes never turns a result into [Error]: once
       a partial reader returns, a partial file that holds more of the same
       file returns the same.}}

    An IFD or a dataset keeps the partial file it was decoded from, and
    {b never reads bytes added after it was decoded}. After adding bytes,
    decode the new partial file again, which reads only bytes already
    held, and take the IFD or dataset at the same position, which monotony
    keeps. A window of one decoded before the bytes were added returns the
    same [Error] again.

    A partial file keeps every byte added to it. A program that reads a
    large file in many windows, such as a pass over all its rows, holds
    each window's bytes until it drops the partial file. It bounds its
    memory by starting again from {!Partial.v}, adding the ranges of the
    structures it kept from the earlier decodes, whose bytes the next
    decode reads again, and then those the next window names.

    This reads a window of the first IFD of a remote TIFF file, given
    [get at n], which returns the [n] bytes of the file from byte [at]:
    {[
    let fill p rs =
      List.fold_left
        (fun p (at, n) -> Nx_io.Partial.add p ~at (get at n))
        p rs

    let rec window p ~row ~col ~rows ~cols =
      match Nx_io.decode_tiff p with
      | Error rs -> window (fill p rs) ~row ~col ~rows ~cols
      | Ok ds -> (
          match Nx_io.Tiff.window (List.hd ds) ~row ~col ~rows ~cols with
          | Ok x -> x
          | Error rs -> window (fill p rs) ~row ~col ~rows ~cols)
    ]} *)

(** Partial files. *)
module Partial : sig
  type t
  (** The type for partial files. A partial file is immutable, and may be
      shared between domains. *)

  val v : name:string -> length:int -> t
  (** [v ~name ~length] is the partial file of a file of [length] bytes
      that holds none of them. [name] starts the messages of failures
      reading it, as a path does.

      Raises [Invalid_argument] if [length] is negative. *)

  val add : t -> at:int -> string -> t
  (** [add p ~at s] is [p] holding the bytes [s] at offsets [at] to
      [at + String.length s - 1] as well. [p] is unchanged. [add] copies
      the bytes of [s] that [p] lacks, once, and shares [p]'s.

      Raises [Invalid_argument] if [at] is negative or [s] ends past [p]'s
      length, and [Failure] starting with [p]'s name and naming the offset
      if [s] differs from a byte [p] holds. The bytes are then of two
      versions of the file, and no partial file that holds both is
      consistent: start again from {!v}. *)
end

val decode_hdf5 :
  Partial.t -> (Hdf5.Group.t list * Hdf5.t list, (int * int) list) result
(** [decode_hdf5 p] is [Ok (gs, ds)], the groups and datasets {!load_hdf5}
    reads from the whole file [p] holds part of, or [Error rs], [rs] the
    ranges ({!section-partial}) [p] lacks of the structures it reads next:
    the superblock, the groups' B-tree nodes, symbol table nodes and local
    heaps, and the object headers. It reads no element. Each group and
    dataset keeps [p], from which {!Hdf5.window}, {!Hdf5.rows} and
    {!Hdf5.values} read.

    Raises [Failure] starting with [p]'s name as {!load_hdf5} does. *)

val decode_tiff : Partial.t -> (Tiff.t list, (int * int) list) result
(** [decode_tiff p] is [Ok ds], [ds] the IFDs {!load_tiff} reads from the
    whole file [p] holds part of, or [Error rs], [rs] the ranges
    ({!section-partial}) [p] lacks of the bytes it reads next: the header,
    an IFD's entries, and the values of every IFD it reaches, since it
    follows the IFDs' chain past an IFD whose values [p] lacks. It reads
    no sample. Each IFD keeps [p], from which {!Tiff.window} and
    {!Tiff.samples} read.

    Raises [Failure] starting with [p]'s name as {!load_tiff} does. *)

(** {1:errors Errors}

    A load or a save raises [Failure] when its file cannot be read or written or
    breaks its format, and a load also when its path names no regular file, such
    as a directory or a FIFO. The message starts with the path, as in
    ["w.npy: the file ends inside its header"].

    A function raises [Invalid_argument] naming it for a value, a name or an
    argument its format cannot store, as in
    ["Nx_io.save_npy: npy has no bfloat16"]. A save also raises it for a value
    {!Nx.Repr.read} refuses: a traced value or a dead one. On a big-endian
    host the swap of a dead value's bytes raises first, naming its nx
    operation. A failed check pending on the value raises from the save as
    the check raised it. *)
