(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Model checkpoints.

    A checkpoint is an immutable collection of tensors keyed by distinct,
    non-empty names, stored as a
    {{:https://huggingface.co/docs/safetensors/}safetensors} file. Typed
    parameter structures enter and leave checkpoints through {!Named}, which
    extends {!Nx.Ptree.S} with stable leaf names: {!of_params} turns a structure
    into named entries, and {!to_params} rebuilds one from them, using an
    existing value as the template for structure, dtypes, and shapes.

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
           Checkpoint.of_int "optim.step" st.step;
         ])
    ]}

    Loading is template-based: construct the model first, then replace its
    values with
    [to_params (module Model) ~prefix:"model" ~like:model (Checkpoint.load
     path)]. To load a file into a partially different model (say, a new head on
    a pretrained backbone), extract each sub-structure with its own module and
    prefix. *)

(** {1:named Named structures} *)

(** Parameter trees with stable leaf names.

    Implementations must ensure that [names x] has exactly one name per tensor
    leaf of [x], paired with leaves in traversal order (the order [iter] and
    [map] visit them), and that the names are distinct and non-empty. By
    convention leaves are named after record fields, with nested structures
    joined by ["."] (e.g. ["encoder.w"]). *)
module type Named = sig
  include Nx.Ptree.S

  val names : t -> string list
  (** [names x] is the name of each tensor leaf of [x], in traversal order. *)
end

module Ptree : Named with type t = Rune_next.Ptree.t
(** Dynamic parameter trees as a named structure. Leaves are named by their path
    from the root: dict keys and zero-based list positions joined with ["."]
    (e.g. ["layers.0.w"]). A bare root tensor has the empty path and is named by
    the [prefix] argument of {!of_params} and {!to_params} alone. *)

(** {1:checkpoints Checkpoints} *)

type t
(** The type for checkpoints: immutable collections of tensors keyed by
    distinct, non-empty names. *)

val empty : t
(** [empty] is the checkpoint with no entries. *)

val of_params : (module P : Named) -> ?prefix:string -> P.t -> t
(** [of_params (module P) ?prefix params] is a checkpoint with one entry per
    tensor leaf of [params], named by [P.names params]. When [prefix] is given,
    each name becomes [prefix ^ "." ^ name] ([prefix] alone for an empty name).

    Raises [Invalid_argument] if [P.names params] does not have exactly one name
    per leaf, or if the resulting names are not distinct and non-empty. *)

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

val find : string -> t -> Rune_next.Ptree.tensor option
(** [find name t] is [name]'s entry in [t], if any. *)

val get : string -> t -> Rune_next.Ptree.tensor
(** [get name t] is [name]'s entry in [t].

    Raises [Invalid_argument] if [name] has no entry. *)

(** {1:extraction Typed extraction} *)

val to_params :
  (module P : Named) -> ?prefix:string -> ?cast:bool -> like:P.t -> t -> P.t
(** [to_params (module P) ?prefix ?cast ~like t] is [like] with every tensor
    leaf replaced by [t]'s entry of the same name — [P.names like], prefixed as
    in {!of_params}. [like] supplies the structure, names, dtypes, and shapes;
    its values are discarded. Entries of [t] not named by [like] are ignored.

    Each entry must have its leaf's shape, and its dtype: when [cast] is [false]
    (default) a dtype mismatch raises, when [true] mismatched entries are cast
    to the leaf's dtype.

    Raises [Invalid_argument] if an entry named by [like] is missing, on shape
    mismatch, on dtype mismatch when [cast] is [false], or if [P.names like] is
    invalid (see {!of_params}). *)

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
