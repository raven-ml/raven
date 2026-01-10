(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kind =
  | Params
  | Optimizer
  | Rng
  | Payload of string
  | Custom of string
  | Unknown of string

type t = { kind : kind; label : string; snapshot : Snapshot.t }

val create : ?label:string -> kind -> Snapshot.t -> t
val slug : t -> string
val kind_to_string : kind -> string
val kind_of_string : string -> kind option
