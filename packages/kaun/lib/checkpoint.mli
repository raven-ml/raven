(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Model checkpoints.

    A checkpoint is an immutable collection of tensors keyed by distinct,
    non-empty names, stored as a
    {{:https://huggingface.co/docs/safetensors/}safetensors} file. Parameter
    structures enter and leave checkpoints through their {!Nx.Ptree.Uniform}
    instance: {!of_params} names each leaf by its path — record fields and
    container positions joined with ["."] — and {!to_params} rebuilds a
    structure from its entries, using an existing value as the template for
    structure, dtypes, and shapes. Unlike the transformations, which take an
    instantiated walker ({!Nx.Ptree.instantiate}), checkpointing takes the
    structure's module itself: leaf names come from the structure's shape.

    Entries not named by the template are ignored on extraction, so one file
    holds several sections side by side — model parameters, parameter-shaped
    optimizer state, counters:

    {[
    Checkpoint.save path
      (Checkpoint.concat
         [
           Checkpoint.of_params (module Model) ~prefix:"model" params;
           Checkpoint.of_params (module Model) ~prefix:"optim.mu" st.mu;
           Checkpoint.of_params (module Model) ~prefix:"optim.nu" st.nu;
           Checkpoint.of_tensor "optim.c1" st.c1;
           Checkpoint.of_tensor "optim.c2" st.c2;
           Checkpoint.of_int "optim.step"
             (Int32.to_int (Nx.item [] st.step));
         ])
    ]}

    Loading is template-based: construct the model first, then replace its
    values with
    [to_params (module Model) ~prefix:"model" ~like:model (Checkpoint.load
     path)]. To load a file into a partially different model (say, a new head on
    a pretrained backbone), extract each sub-structure with its own module and
    prefix.

    Structures with mixed leaf dtypes — the stock dynamic tree
    {!Rune.Ptree.t} among them — hold packed leaves, and enter and leave
    checkpoints through {!of_packed} and {!to_packed}. *)

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
  (module U : Nx.Ptree.Uniform) ->
  ?prefix:string ->
  Rune.Ptree.tensor U.t ->
  t
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

val to_params :
  (module U : Nx.Ptree.Uniform) ->
  ?prefix:string ->
  ?cast:bool ->
  like:('a, 'b) Nx.t U.t ->
  t ->
  ('a, 'b) Nx.t U.t
(** [to_params (module U) ?prefix ?cast ~like t] is [like] with every leaf
    replaced by [t]'s entry of the same name — the leaf's path, prefixed as in
    {!of_params}. [like] supplies the structure, names, dtypes, and shapes; its
    values are discarded. Entries of [t] not named by [like] are ignored.

    Each entry must have its leaf's shape, and its dtype: when [cast] is
    [false] (default) a dtype mismatch raises, when [true] mismatched entries
    are cast to the leaf's dtype.

    Raises [Invalid_argument] if an entry named by [like] is missing, on shape
    mismatch, on dtype mismatch when [cast] is [false], or if [like]'s names
    are not distinct and non-empty. *)

val to_packed :
  (module U : Nx.Ptree.Uniform) ->
  ?prefix:string ->
  ?cast:bool ->
  like:Rune.Ptree.tensor U.t ->
  t ->
  Rune.Ptree.tensor U.t
(** [to_packed (module U) ?prefix ?cast ~like t] is like {!to_params} for a
    structure with packed leaves, with names as in {!of_packed}. Each template
    leaf's runtime dtype and shape check the corresponding entry. *)

val to_int : string -> t -> int
(** [to_int name t] is the integer stored at [name] by {!of_int}.

    Raises [Invalid_argument] if [name] has no entry, or its entry is not a
    one-element int32 tensor. *)

(** {1:files Files} *)

val save : string -> t -> unit
(** [save path t] writes [t] to a safetensors file at [path], replacing any
    existing file.

    Raises [Failure] on I/O errors, or if an entry's dtype is not supported by
    safetensors (see {!Nx_io.save_safetensors}). *)

val load : string -> t
(** [load path] is the checkpoint stored in the safetensors file at [path],
    whether written by {!save} or produced elsewhere. Entries whose dtype
    {!Nx_io} cannot represent are skipped with a warning on stderr.

    Raises [Failure] on I/O or format errors. *)
