(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Model checkpoints.

    A checkpoint is an immutable collection of tensors keyed by distinct,
    non-empty names, stored as a
    {{:https://huggingface.co/docs/safetensors/}safetensors} file.

    {b Files this library wrote.} Values enter and leave checkpoints through
    their structure ({!Nx.Ptree.t}): {!of_value} names each tensor by its path,
    record fields and container positions joined with ["."], and {!to_value}
    rebuilds a value from its entries, using an existing value as the template
    for structure, dtypes and shapes. A restart holds such a value already.
    Entries not named by the template are ignored, so one file holds several
    sections side by side, model parameters, optimizer state and counters:

    {[
    Checkpoint.save path
      (Checkpoint.concat
         [
           Checkpoint.of_value ~prefix:"model" model params;
           Checkpoint.of_value ~prefix:"optim" (Vega.adam_ptree model) opt;
           Checkpoint.of_int "epoch" epoch;
         ])
    ]}

    and [to_value ~prefix:"model" model ~like:params (Checkpoint.load path)]
    reads the model back. To load a file into a partially different model, say a
    new head on a pretrained backbone, extract each sub-structure with its own
    structure and prefix.

    {b Files produced elsewhere.} A pretrained checkpoint has its own names and
    layouts. Its importer is an ordinary function that builds the parameter
    record and asks for each entry by name, shape and dtype with {!to_float} and
    {!to_tensor}, reshaping with nx where the layouts differ:

    {[
    let linear ~inputs ~outputs name ckpt =
      (* The file stores the weight as [outputs; inputs]. *)
      let w =
        Checkpoint.to_float ~shape:[| outputs; inputs |] dt (name ^ ".weight")
          ckpt
      in
      { Linear.w = Nx.matrix_transpose w; b = None }
    ]}

    A wrong configuration fails at import, with the entry's name. No template is
    allocated, and at the file's own dtype nothing is copied: every leaf is a
    view of the file. Each accessor states the dtype of the leaf it fills, so a
    record may hold float leaves beside integer ones.

    Bytes are never reinterpreted silently: only {!to_float} converts, and only
    between floating-point dtypes; every other dtype mismatch raises. *)

(** {1:checkpoints Checkpoints} *)

type t
(** The type for checkpoints: immutable collections of tensors keyed by
    distinct, non-empty names. *)

val empty : t
(** [empty] is the checkpoint with no entries. *)

val of_value : ?prefix:string -> 's Nx.Ptree.t -> 's -> t
(** [of_value ?prefix s x] is a checkpoint with one entry per tensor [s] walks
    in [x], fixed tensors included, named by its path
    ({!Nx.Ptree.Path.to_string}). When [prefix] is given, each name becomes
    [prefix ^ "." ^ path] ([prefix] alone for the root). The entries hold [x]'s
    tensors; nothing is copied.

    Raises [Invalid_argument] if two tensors of [x] have one name, as in
    ["Checkpoint.of_value: w: two leaves have this name"], or if a tensor at the
    root has no [prefix], as in
    ["Checkpoint.of_value: a leaf at the root needs ~prefix"]. *)

val of_tensor : string -> ('a, 'b) Nx.t -> t
(** [of_tensor name x] is a checkpoint with the single entry [name] holding [x].
    Raises [Invalid_argument] if [name] is empty. *)

val of_int : string -> int -> t
(** [of_int name i] is a checkpoint with the single entry [name] holding [i] as
    a one-element int32 tensor. Use it for training counters; read it back with
    {!to_int}.

    Raises [Invalid_argument] if [name] is empty or [i] does not fit in 32 bits.
*)

val concat : t list -> t
(** [concat ts] is the checkpoint with the entries of all [ts].

    Raises [Invalid_argument] if a name appears in more than one checkpoint. *)

(** {1:queries Queries} *)

val names : t -> string list
(** [names t] is the names of [t]'s entries, sorted. *)

val find : string -> t -> Nx.packed option
(** [find name t] is [name]'s entry in [t], if any. *)

val get : string -> t -> Nx.packed
(** [get name t] is [name]'s entry in [t].

    Raises [Invalid_argument] if [name] has no entry. *)

(** {1:extraction Typed extraction} *)

val to_tensor :
  shape:int array -> ('a, 'b) Nx.dtype -> string -> t -> ('a, 'b) Nx.t
(** [to_tensor ~shape dtype name t] is [name]'s entry in [t], which must have
    [dtype] and [shape]. The entry is returned as stored, so an entry of a
    loaded file stays a view of the file.

    Raises [Invalid_argument], naming the entry, if [name] has no entry or the
    entry's shape or dtype differs. *)

val to_float :
  shape:int array -> (float, 'b) Nx.dtype -> string -> t -> (float, 'b) Nx.t
(** [to_float ~shape dtype name t] is [name]'s entry in [t], a [float16],
    [bfloat16], [float32] or [float64] tensor of [shape], at [dtype]: the entry
    as stored when it already has [dtype], and otherwise its cast, which
    allocates the leaf. [dtype] is one of the same four dtypes.

    An 8-bit float entry is refused, since its scales live in other entries;
    read it with {!to_tensor}. An importer that ties two weights binds the
    tensor once and uses it twice.

    Raises [Invalid_argument], naming the entry, if [name] has no entry, if the
    entry's shape differs, if the entry is not one of the four dtypes, or if
    [dtype] is an 8-bit float. *)

val to_value : ?prefix:string -> 's Nx.Ptree.t -> like:'s -> t -> 's
(** [to_value ?prefix s ~like t] is [like] with every tensor replaced by [t]'s
    entry of the same name, the tensor's path prefixed as in {!of_value}. [like]
    supplies the structure, names, dtypes and shapes; its values are discarded.
    The result holds [t]'s entries; nothing is copied, so a call that consumes
    the result also ends those entries. Entries of [t] not named by [like] are
    ignored. A checkpoint stores no reports: a value loads into [like]'s shape,
    its list lengths, option presences and cases.

    A template states the dtype it expects, so nothing is converted: a restart
    that names the wrong dtype fails instead of narrowing its state. To convert,
    ask by name with {!to_float}.

    Raises [Invalid_argument] naming the entry and what the checkpoint and the
    template hold there if an entry named by [like] is missing or its shape or
    dtype differs, as in
    ["Checkpoint.to_value: model.l1.w: shape [3] in the checkpoint, [2; 3] in
     the template"], and as {!of_value} if [like]'s names are not distinct and
    non-empty. *)

val to_int : string -> t -> int
(** [to_int name t] is the integer stored at [name] by {!of_int}.

    Raises [Invalid_argument] if [name] has no entry, or its entry is not a
    one-element int32 tensor. *)

(** {1:files Files} *)

val save : string -> t -> unit
(** [save path t] writes [t] to a safetensors file at [path], replacing any
    existing file atomically: the entries are written to a temporary file that
    is then renamed to [path], so saving over a checkpoint whose tensors are
    still in use is safe.

    Raises [Failure] on I/O errors, or if an entry's dtype is not supported by
    safetensors (see {!Nx_io.save_safetensors}). *)

val load : string -> t
(** [load path] is the checkpoint stored in the safetensors file at [path],
    whether written by {!save} or produced elsewhere. The file is mapped and its
    entries are views of it, read when first used: the file must not be modified
    in place while an entry is alive, and [Nx.copy] gives a tensor that no
    longer depends on it. An entry whose dtype nx lacks is loaded as its bytes,
    at [uint8]. See {!Nx_io.load_safetensors}.

    Raises [Failure] on I/O or format errors. *)
