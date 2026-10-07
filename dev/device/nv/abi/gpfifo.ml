(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let max_words = (1 lsl D.nvc56f_gp_entry1_length.bits) - 1

(* An entry's segment address is its low 40 bits: word 0's GET holds the
   address's bits 2 to 31 in place, a 4-byte aligned address's low bits being
   zero, and word 1's GET_HI its bits 32 to 39. *)
let address_bits = 32 + D.nvc56f_gp_entry1_get_hi.bits

let entry addr ~offset ~words =
  if words < 0 || words > max_words then
    invalid_arg
      (Printf.sprintf "Gpfifo.entry: %d words, expected 0 to %d" words max_words);
  if offset < 0 || offset >= 1 lsl address_bits then
    invalid_arg
      (Printf.sprintf "Gpfifo.entry: offset 0x%x, expected 0 to 2^%d-1" offset
         address_bits);
  let flags =
    (D.nvc56f_gp_entry1_level_subroutine lsl D.nvc56f_gp_entry1_level.lo)
    lor (words lsl D.nvc56f_gp_entry1_length.lo)
  in
  let n = Int64.(logor (of_int offset) (shift_left (of_int flags) 32)) in
  [ Packet.W64 (Add (Value addr, n)) ]
