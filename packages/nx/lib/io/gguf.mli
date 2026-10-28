(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** GGUF files. {!Nx_io.Gguf} documents them. *)

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
  | String of string
  | Array of value array

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

type tensor_info = { dtype : dtype; shape : int array }
type t

val read : Rig.Buffer.t -> t
(** [read file] is the contents of the disk buffer [file].

    Raises [Failure] as {!Nx_io.load_gguf} states. *)

val version : t -> int
val metadata : t -> (string * value) list
val tensors : t -> Nx.host Archive.t
val info : string -> t -> tensor_info
