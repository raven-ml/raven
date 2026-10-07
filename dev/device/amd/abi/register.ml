(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = Defs.register = {
  name : string;
  offset : int;
  segment : int;
  fields : (string * (int * int)) list;
}

let major ((m, _, _) : Gpu.version) = m

(* The registers of the latest version of [g]'s GC major at or before its GC: GC
   11.0.2 takes 11.0.0's. *)
let registers (g : Gpu.t) =
  let latest best (v, rs) =
    if major v <> major g.gc || compare v g.gc > 0 then best
    else
      match best with
      | Some (b, _) when compare b v >= 0 -> best
      | _ -> Some (v, rs)
  in
  match List.fold_left latest None Defs.gc_registers with
  | Some (_, rs) -> rs
  | None -> []

let find g name =
  List.find_opt (fun r -> String.equal r.name name) (registers g)

(* The bases of the latest generation at or before [g]'s GC major. *)
let address (g : Gpu.t) r =
  let bases =
    List.fold_left
      (fun acc (m, b) -> if m <= major g.gc then b else acc)
      [] Defs.gc_bases
  in
  match List.nth_opt bases r.segment with
  | Some base -> base + r.offset
  | None ->
      let a, b, c = g.gc in
      invalid_argf
        "Register.address: %s's segment %d has no base on GC %d.%d.%d" r.name
        r.segment a b c

let encode r fs =
  let set w (f, v) =
    match List.assoc_opt f r.fields with
    | Some (lo, hi) -> w lor ((v land ((1 lsl (hi - lo + 1)) - 1)) lsl lo)
    | None -> invalid_argf "Register.encode: %s has no field %s" r.name f
  in
  List.fold_left set 0 fs
