(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

(* Load valid simplification *)

(* A load in the index runs whatever [valid], so its own index cannot assume it:
   a gather through a pad would read its indices outside the pad. Each load
   stands as a variable of its bounds while [valid] simplifies the arithmetic
   around it. *)
let simplify_valid_load buf start_idx valid =
  let loads =
    List.filter
      (fun u -> op u = Op.Load)
      (Nodes.to_list (backward_slice ~calls:Skip start_idx))
  in
  let held =
    List.mapi
      (fun i l ->
        let name = "load" ^ string_of_int i in
        (l, variable ~dtype:(dtype l) name (vmin l) (vmax l)))
      loads
  in
  let hold u = substitute ~calls:Skip ~pass:Fixed_point u held in
  let idx = Symbolic.uop_given_valid (hold valid) (hold start_idx) in
  let idx =
    substitute ~calls:Skip ~pass:Fixed_point idx
      (List.map (fun (l, v) -> (v, l)) held)
  in
  if idx == start_idx || idx == simplify start_idx then None
  else Some (index buf [ Shape.valid idx valid ])

let indexing_simplify =
  Pattern_matcher.v
    (fun () -> [
      Pattern_matcher.rule
        (Upat.op Op.Index ~src:[ Upat.var "buf"; Shape.invalid_gate ])
        (fun m -> simplify_valid_load (m "buf") (m "x") (m "cond"));
    ])

(* Memory coalescing *)

(* The accesses that may merge: of one operation, buffer, base index, gate and
   argument. The base is [None] for constant indices. *)
type key = { op : Op.t; buf : t; base : t option; valid : t; arg : arg }

let same k0 k1 =
  k0.op = k1.op && k0.buf == k1.buf
  && Option.equal ( == ) k0.base k1.base
  && k0.valid == k1.valid && equal_arg k0.arg k1.arg

let int_value u =
  match value u with
  | #Dtype.value as v -> Dtype.Value.to_z v
  | `Invalid -> invalid_arg "an invalid index"

(* The runs of consecutive integers of the sorted [l]. *)
let runs l =
  let rec go acc run = function
    | [] -> List.rev (List.rev run :: acc)
    | x :: rest -> (
        match run with
        | prev :: _ when Bigint.equal x (Bigint.succ prev) -> go acc (x :: run) rest
        | [] -> go acc [ x ] rest
        | _ -> go (List.rev run :: acc) [ x ] rest)
  in
  match l with [] -> [] | _ -> go [] [] l

let memory_coalescing sink (r : Renderer.t) =
  if Setting.value Setting.dmc then sink
  else begin
    (* Collect, each group's offsets in the order first seen. *)
    let groups = ref [] in
    let add key offset u =
      match List.find_opt (fun (k, _) -> same k key) !groups with
      | Some (_, offsets) -> (
          match List.find_opt (fun (o, _) -> Bigint.equal o offset) !offsets with
          | Some (_, us) -> us := u :: !us
          | None -> offsets := (offset, ref [ u ]) :: !offsets)
      | None -> groups := (key, ref [ (offset, ref [ u ]) ]) :: !groups
    in
    List.iter
      (fun u ->
        if op u = Op.Load || op u = Op.Store then begin
          if List.length (src u) <> if op u = Op.Store then 2 else 1 then
            invalid_arg "memory coalescing does not support gated loads/stores";
          let target = nth u 0 in
          if op target <> Op.Index then
            invalid_arg
              (Format.asprintf "memory coalescing should be on INDEX, not %a"
                 Op.pp (op target));
          match src target with
          | [ buf; idx_u ] ->
              let volatile =
                let b = buf_uop buf in
                op b = Op.Param
                && match arg b with Param p -> p.volatile | _ -> false
              in
              let idx = get_idx idx_u and valid = get_valid idx_u in
              (* Volatile accesses never merge. *)
              if
                addrspace buf <> Some Dtype.Reg
                && (not volatile)
                && not (is_invalid idx)
              then begin
                let base, offset =
                  match (op idx, src idx) with
                  | Op.Add, [ b; c ] when op c = Op.Const ->
                      (Some b, int_value c)
                  | Op.Add, [ c; b ] when op c = Op.Const ->
                      (Some b, int_value c)
                  | Op.Const, _ -> (None, int_value idx)
                  | _ -> (Some idx, Bigint.zero)
                in
                add { op = op u; buf; base; valid; arg = arg u } offset u
              end
          | _ -> invalid_arg "an index of memory needs a buffer and one index"
        end)
      (toposort ~calls:Enter sink);
    (* Build the replacements. *)
    let replacements = ref [] in
    List.iter
      (fun (key, offsets) ->
        let offsets =
          List.rev_map (fun (o, us) -> (o, List.rev !us)) !offsets
        in
        let accesses o = List.assoc o offsets in
        let lengths =
          (if
             List.mem (dtype key.buf)
               Dtype.([ Float32; Float16; Int32; Uint32 ] @ fp8s)
             && r.supports_float4
           then
             if
               dtype key.buf = Dtype.Float16
               && Setting.value Setting.allow_half8
             then [ 8; 4; 2 ]
             else [ 4; 2 ]
           else [])
          @ [ 1 ]
        in
        (* Elements before the buffer's first from the boundary behind it, a
           multiple of its alignment: an access of [l] elements is aligned
           where the element count from that boundary is a multiple of [l], and
           is no wider than the alignment. *)
        let size = Dtype.itemsize (dtype key.buf) in
        let phase, align =
          match arg (buf_uop key.buf) with
          | Param p -> (p.phase, p.align)
          | _ -> (0, 16)
        in
        let lead = Bigint.of_int (phase / size) in
        (* An access of another width than the storage's elements, through a
           bitcast, may start between its own elements: it merges nothing. *)
        let lengths =
          if phase mod size = 0 then
            List.filter (fun l -> l = 1 || l * size <= align) lengths
          else [ 1 ]
        in
        let at first =
          match key.base with
          | Some b -> O.(b + const (`Int first))
          | None -> const (`Int first)
        in
        let sorted = List.sort Bigint.compare (List.map fst offsets) in
        List.iter
          (fun run ->
            let rec take grp =
              match grp with
              | [] -> ()
              | first :: _ ->
                  let offset = at first in
                  let length =
                    List.find
                      (fun l ->
                        l <= List.length grp
                        && Option.is_some
                             (divides (at (Bigint.add first lead)) (Bigint.of_int l)))
                      lengths
                  in
                  let now = List.filteri (fun i _ -> i < length) grp
                  and rest = List.filteri (fun i _ -> i >= length) grp in
                  (* The gate applies again once the length is known. *)
                  let offset = valid offset key.valid in
                  let idx =
                    if length > 1 then
                      v Op.Shrink ~src:[ key.buf; offset; int length ]
                    else index key.buf [ offset ]
                  in
                  (if key.op = Op.Store then begin
                     let stores = List.map accesses now in
                     if List.exists (fun us -> List.length us <> 1) stores then
                       invalid_arg "attempting multiple stores";
                     let datas =
                       List.map (fun us -> nth (List.hd us) 1) stores
                     in
                     let st =
                       store idx
                         (if length > 1 then stack datas else List.hd datas)
                     in
                     List.iter
                       (fun us ->
                         replacements := (List.hd us, st) :: !replacements)
                       stores
                   end
                   else
                     let ld = v Op.Load ~src:[ idx ] ~arg:key.arg in
                     List.iteri
                       (fun i g ->
                         let value =
                           if length > 1 then index ld [ int i ] else ld
                         in
                         List.iter
                           (fun oo ->
                             replacements := (oo, value) :: !replacements)
                           (accesses g))
                       now);
                  take rest
            in
            take run)
          (runs sorted))
      (List.rev !groups);
    substitute ~calls:Skip ~pass:Fixed_point sink !replacements
  end
