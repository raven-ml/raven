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

(* The bases of the latest generation at or before [major], from [latest]. *)
let rec bases major latest = function
  | (m, b) :: rest when m <= major -> bases major b rest
  | _ -> latest

(* The base of segment [s] of [bases], or [-1] if it has none. *)
let rec base s = function
  | b :: _ when s = 0 -> b
  | _ :: rest when s > 0 -> base (s - 1) rest
  | _ -> -1

let address (g : Gpu.t) r =
  let major, _, _ = g.gc in
  let b = base r.segment (bases major [] Defs.gc_bases) in
  if b < 0 then
    invalid_argf "Register.address: %s's segment %d has no base on %s" r.name
      r.segment (gc_name g);
  b + r.offset

let encode r fs =
  let set w (f, v) =
    match List.assoc_opt f r.fields with
    | Some (lo, hi) -> w lor ((v land ((1 lsl (hi - lo + 1)) - 1)) lsl lo)
    | None -> invalid_argf "Register.encode: %s has no field %s" r.name f
  in
  List.fold_left set 0 fs
