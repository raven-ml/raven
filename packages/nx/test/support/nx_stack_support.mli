(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The C stack's depth at the calls OCaml makes to C. *)

val deeper : int -> (unit -> 'a) -> 'a
(** [deeper n f] is [f ()], run under a C frame that holds [n + 1] bytes:
    the C functions [f] calls start that much lower on the stack, rounded
    to the stack's alignment. On a stack kept 16-byte aligned, [n] of 0, 8,
    16 and 24 starts them at both of its residues modulo 32.

    Raises [Invalid_argument] if [n] is negative or over 4096. *)
