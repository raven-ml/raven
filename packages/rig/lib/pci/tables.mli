(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Hash tables keyed by addresses.

    An address lies on a large alignment, so a key's hash mixes every bit into
    the low ones a table indexes by, and integer keys compare as integers. *)

module Address : Hashtbl.S with type key = int
(** Tables keyed by an address. *)

module Range : Hashtbl.S with type key = int * int
(** Tables keyed by a range: its address and its bytes. *)

module Window : Hashtbl.S with type key = Window.t
(** Tables keyed by a window, hashed by its address. *)
