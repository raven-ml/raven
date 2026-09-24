(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Model checkpoints.

    A checkpoint is an immutable collection of tensors keyed by distinct,
    non-empty names, stored as a
    {{:https://huggingface.co/docs/safetensors/}safetensors} file.

    {b Files this library wrote.} Parameter structures enter and leave
    checkpoints through their {!Nx.Ptree.Uniform} instance: {!of_params} names
    each leaf by its path, record fields and container positions joined with
    ["."], and {!to_params} rebuilds a structure from its entries, using an
    existing value as the template for structure, dtypes and shapes. A restart
    holds such a value already. Entries not named by the template are ignored,
    so one file holds several sections side by side, model parameters,
    parameter-shaped optimizer state and counters:

    {[
    Checkpoint.save path
      (Checkpoint.concat
         [
           Checkpoint.of_params (module Model) ~prefix:"model" params;
           Checkpoint.of_params (module Model) ~prefix:"optim.mu" st.mu;
           Checkpoint.of_params (module Model) ~prefix:"optim.nu" st.nu;
           Checkpoint.of_tensor "optim.step" st.step;
         ])
    ]}

    and
    [to_params (module Model) ~prefix:"model" ~like:params (Checkpoint.load
     path)] reads the model back. To load a file into a partially different
    model, say a new head on a pretrained backbone, extract each sub-structure
    with its own module and prefix. Structures with mixed leaf dtypes, the stock
    dynamic tree {!Rune.Ptree.t} among them, hold packed leaves and use
    {!of_packed} and {!to_packed}.

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

val of_params :
  (module U : Nx.Ptree.Uniform) -> ?prefix:string -> ('a, 'b) Nx.t U.t -> t
(** [of_params (module U) ?prefix params] is a checkpoint with one entry per
    leaf of [params], named by its path ([U.fold]'s path convention). When
    [prefix] is given, each name becomes [prefix ^ "." ^ path] ([prefix] alone
    for the empty path).

    Raises [Invalid_argument] if the resulting names are not distinct and
    non-empty. *)

val of_packed :
  (module U : Nx.Ptree.Uniform) -> ?prefix:string -> Rune.Ptree.tensor U.t -> t
(** [of_packed (module U) ?prefix params] is like {!of_params} for a structure
    with packed leaves, whose dtypes may differ. For the stock dynamic tree,
    pass [(module Rune.Ptree.Tree)]: leaves are named by dict keys and
    zero-based list positions joined with ["."] (e.g. ["layers.0.w"]), and a
    bare root tensor has the empty path and is named by [prefix] alone.

    Raises [Invalid_argument] if the resulting names are not distinct and
    non-empty. *)

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

val find : string -> t -> Rune.Ptree.tensor option
(** [find name t] is [name]'s entry in [t], if any. *)

val get : string -> t -> Rune.Ptree.tensor
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

val to_params :
  (module U : Nx.Ptree.Uniform) ->
  ?prefix:string -> like:('a, 'b) Nx.t U.t -> t -> ('a, 'b) Nx.t U.t
(** [to_params (module U) ?prefix ~like t] is [like] with every leaf replaced by
    [t]'s entry of the same name, the leaf's path prefixed as in {!of_params}:
    {!to_tensor} at each leaf's name, shape and dtype. [like] supplies the
    structure, names, dtypes and shapes; its values are discarded. Entries of
    [t] not named by [like] are ignored.

    A template states the dtype it expects, so nothing is converted: a restart
    that names the wrong dtype fails instead of narrowing its state. To convert,
    ask by name with {!to_float}.

    Raises [Invalid_argument] if an entry named by [like] is missing, on a shape
    or dtype mismatch, or if [like]'s names are not distinct and non-empty. *)

val to_packed :
  (module U : Nx.Ptree.Uniform) ->
  ?prefix:string -> like:Rune.Ptree.tensor U.t -> t -> Rune.Ptree.tensor U.t
(** [to_packed (module U) ?prefix ~like t] is like {!to_params} for a structure
    with packed leaves, with names as in {!of_packed}. Each template leaf's
    runtime dtype and shape check the corresponding entry. *)

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
