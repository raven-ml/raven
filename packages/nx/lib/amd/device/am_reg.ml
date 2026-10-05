(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type version = int * int * int

type t = {
  name : string;
  offset : int;
  segment : int;
  fields : (string * (int * int)) list; (* bit ranges, lowest bit first *)
  addr : (int * int) list; (* by instance *)
}

let pp_version (a, b, c) = Printf.sprintf "%d.%d.%d" a b c

let no_family prefix target =
  failwith
    (Printf.sprintf "no %s definitions for version %s" prefix
       (pp_version target))

(* The family closest below [target] with its major version: the header set a
   block of version [target] is programmed with. A few versions are named after
   another. GC's registers are the packets' ({!Nx_amd_packet.Gc}). *)
let family prefix target =
  let target =
    match (prefix, target) with
    | "smu", (13, 0, 7 | 13, 0, 10) -> (13, 0, 0)
    | _ -> target
  in
  let major, _, _ = target in
  let candidates =
    List.filter
      (fun (p, ((m, _, _) as v)) ->
        p = prefix && m = major && compare v target <= 0)
      Amd_defs.families
  in
  let best = List.sort (fun (_, a) (_, b) -> compare b a) candidates in
  match (prefix, best) with
  | "gc", _ -> (
      match Nx_amd_packet.Gc.family target with
      | Some v -> v
      | None -> no_family prefix target)
  | _, (_, v) :: _ -> v
  | _, [] -> no_family prefix target

(* The registers of [prefix] at [version], each as (name, offset, segment,
   fields). *)
let table prefix version =
  if prefix = "gc" then
    List.map
      (fun (r : Nx_amd_packet.Gc.register) ->
        (r.name, r.offset, r.segment, r.fields))
      (Nx_amd_packet.Gc.registers (family prefix version))
  else
    let v = family prefix version in
    match
      List.find_opt (fun (p, v', _) -> p = prefix && v' = v) Amd_defs.registers
    with
    | Some (_, _, regs) ->
        List.map
          (fun (name, offset, segment, fields) ->
            ( name,
              offset,
              segment,
              List.map (fun (f, lo, hi) -> (f, (lo, hi))) fields ))
          regs
    | None -> []

(* The registers of [prefix] at [version], at the instances whose segment bases
   are [bases]. *)
let registers prefix version ~bases =
  List.map
    (fun (name, offset, segment, fields) ->
      let addr =
        List.filter_map
          (fun (inst, segs) ->
            if segment < Array.length segs then
              Some (inst, segs.(segment) + offset)
            else None)
          bases
      in
      (name, { name; offset; segment; fields; addr }))
    (table prefix version)

let field r name =
  match List.assoc_opt name r.fields with
  | Some f -> f
  | None -> invalid_arg (Printf.sprintf "%s has no field %s" r.name name)

let encode r kvs =
  List.fold_left (fun acc (name, v) -> acc lor (v lsl fst (field r name))) 0 kvs

let mask r names =
  List.fold_left
    (fun acc name ->
      let lo, hi = field r name in
      acc lor (((1 lsl (hi - lo + 1)) - 1) lsl lo))
    0 names

let decode r v =
  List.map
    (fun (name, (lo, hi)) ->
      (name, (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)))
    r.fields

let addr ?(inst = 0) r =
  match List.assoc_opt inst r.addr with
  | Some a -> a
  | None -> invalid_arg (Printf.sprintf "%s has no instance %d" r.name inst)
