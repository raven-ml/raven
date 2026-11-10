(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrays to and from files.

    A file holds one array, as npy and text files do, or several arrays under
    names, as npz, SafeTensors and GGUF files do. A name and its array are an
    {e entry}. Entries are read and written as an {!Archive.t}. A PyTorch
    checkpoint holds a Python value, whose arrays {!Pytorch.tensors} names,
    and a TIFF file a chain of rasters, each with its tags ({!Tiff}).

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
    blocks until it has read its IFD's chunks, from the file {!load_tiff}
    opened. A save blocks until
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
    let field k = function
      | Nx_io.Pytorch.Dict kvs -> List.assoc (Nx_io.Pytorch.String k) kvs
      | _ -> failwith "student.pt: no dict" in
    let dim =
      match field "dim" (field "args" ckpt) with
      | Nx_io.Pytorch.Int d -> d
      | _ -> failwith "student.pt: args.dim" in
    Nx_io.Archive.float ~shape:[| dim; dim |] Nx.float32 "layers.0.weight"
      (Nx_io.Pytorch.tensors (field "model" ckpt))
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
    {- Predictor (317) 1 (none), 2 (horizontal differencing) or, for float
       samples, 3 (floating point), and 1 with Compression 1;}
    {- BitsPerSample (258) and SampleFormat (339), the same for every
       sample, of [uint8], [uint16] or [uint32] (SampleFormat 1, unsigned),
       [int16] or [int32] (2, signed), or [float32] or [float64] (3, IEEE
       floating point);}
    {- PlanarConfiguration (284) 1, a pixel's samples together, or 2 with
       one sample a pixel;}
    {- FillOrder (266) 1, and a Photometric (262) other than 6 (YCbCr),
       whose samples may be subsampled;}
    {- strips, with StripOffsets (273), StripByteCounts (279) and
       RowsPerStrip (278), or tiles, with TileWidth (322), TileLength (323),
       TileOffsets (324) and TileByteCounts (325), each chunk of a byte or
       more.}}
    Any other value of these tags is refused, naming the tag. A field of a
    type {!Tiff.type-value} lacks is skipped, as TIFF 6.0 tells readers to.
    Every other tag is data, Orientation (274) among them, so a raster's
    first row is its file's first.

    A file's IFDs and the values that do not fit their entries must take
    no more bytes in all than the file holds, and so must each IFD's
    chunks, so that a load's tags take memory linear in the file's length,
    and a raster at most 2560 times its bytes.

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
  (** The type for one IFD of a TIFF file. It keeps the file it was read
      from, as a mapped value does, and {b the file must not change in
      place while it is reachable}. *)

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

      Raises [Failure] as {!section-errors} states, naming the tag, if a
      tag the section lists is missing, of another type, or of a value or
      count the section does not read, or if a chunk lies past the file's
      end, has too few bytes or does not decode to its rows, as in
      ["l.tif: Compression (259) 5: tile 3 at byte 1024: malformed at
       byte 7: invalid code"]. *)

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
