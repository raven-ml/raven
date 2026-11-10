(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PyTorch checkpoints. {!Nx_io.Pytorch} and {!Nx_io.load_pytorch} document
    them. *)

type value =
  | Null
  | Bool of bool
  | Int of int
  | Long of string
  | Float of float
  | Complex of Complex.t
  | String of string
  | Bytes of string
  | Tuple of value list
  | List of value list
  | Set of value list
  | Dict of (value * value) list
  | Tensor of Nx.host Nx.packed
  | Global of string
  | Object of string
  | Shared of int * value
  | Ref of int

val tensors : value -> Nx.host Archive.t

val load : Rig.Buffer.t -> value
(** [load file] is the value of the checkpoint in the disk buffer [file].

    Raises [Failure] as {!Nx_io.load_pytorch} states. *)
