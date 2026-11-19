(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Refusals: the exceptions aion raises. Each message names the function that
    refuses, as in ["Aion.Roar.sample: no input"]: [fn] is that name after
    [Aion.], as ["Roar.sample"]. Messages format as {!Format} does, so that a
    shape prints by [%a] and {!Nx.pp_shape}. *)

val invalid : string -> ('a, Format.formatter, unit, 'b) format4 -> 'a
(** [invalid fn fmt] raises [Invalid_argument] with the message of [fmt]. *)

val failure : string -> ('a, Format.formatter, unit, 'b) format4 -> 'a
(** [failure fn fmt] raises [Failure] with the message of [fmt]. *)

val exn : string -> ('a, Format.formatter, unit, exn) format4 -> 'a
(** [exn fn fmt] is the [Invalid_argument] {!invalid} raises, for the refusal
    of {!Nx.check}. *)
