(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Model checkpointing.

    {!Checkpoint} serializes {!Ptree.t} parameter trees to and from
    {{:https://huggingface.co/docs/safetensors/}SafeTensors} files. Tensor paths
    from {!Ptree.flatten_with_paths} become file keys (e.g.
    ["layers.0.weight"]). *)

val save : string -> Ptree.t -> unit
(** [save path t] writes [t]'s tensors to a safetensors file at [path].

    Raises [Failure] on I/O errors. *)

val load : string -> like:Ptree.t -> Ptree.t
(** [load path ~like] loads tensors from a safetensors file and reconstructs a
    tree with the same structure as [like].

    Each tensor is cast to [like]'s dtype if needed. Extra keys in the file are
    silently ignored.

    Raises [Invalid_argument] if a key required by [like] is missing from the
    file, or if a tensor's shape does not match [like]. Raises [Failure] on I/O
    errors. *)
