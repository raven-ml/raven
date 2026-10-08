(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = Defs.register = {
  name : string;
  offset : int;
  segment : int;
  fields : (string * (int * int)) list;
}

let gc_name (g : Gpu.t) =
  let a, b, c = g.gc in
  strf "GC %d.%d.%d" a b c

let registers (g : Gpu.t) = Defs.registers (Defs.gc g.gc)
let find (g : Gpu.t) name = Defs.find (Defs.gc g.gc) name

let address (g : Gpu.t) r =
  let major, _, _ = g.gc in
  let b = Defs.gc_base major r.segment in
  if b < 0 then
    invalid_argf "Register.address: %s's segment %d has no base on %s" r.name
      r.segment (gc_name g);
  b + r.offset

(* The lowest and highest bits of [r]'s field [f]. *)
let rec bits_of r f = function
  | (name, bits) :: fields ->
      if String.equal name f then bits else bits_of r f fields
  | [] -> invalid_argf "Register.encode: %s has no field %s" r.name f

let rec encode_fields r w = function
  | (f, v) :: fs ->
      let lo, hi = bits_of r f r.fields in
      encode_fields r (w lor ((v land ((1 lsl (hi - lo + 1)) - 1)) lsl lo)) fs
  | [] -> w

let encode r fs = encode_fields r 0 fs
