(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

type rule =
  | Elementwise
  | Reduce of int array
  | Along of int array
  | Gather of int
  | Into of int
  | Replicated

type 'd t = {
  operands : 'd Devices.placement array;
  result : 'd Devices.placement;
}

let is_constant p = p == Devices.anywhere

(* The axes operand [i] of rank [rank] reads whole on each device. *)
let whole rule i rank =
  match rule with
  | Elementwise -> [||]
  | Reduce axes | Along axes -> axes
  | Gather axis -> if i = 1 then [| axis |] else [||]
  | Into axis -> [| axis |]
  | Replicated -> Array.init rank Fun.id

(* The result axis that operand axis [a] becomes, if any. *)
let to_result rule a =
  match rule with
  | Reduce axes ->
      if Array.mem a axes then None
      else
        Some (a - Array.fold_left (fun n r -> if r < a then n + 1 else n) 0 axes)
  | Elementwise | Along _ | Gather _ | Into _ | Replicated -> Some a

let uncut_all axes g = Array.fold_left (fun g axis -> Grid.uncut g ~axis) g axes

(* Operand [i]'s grid on the result's axes: the axes it reads whole, and those
   the result drops, hold copies. *)
let forward rule i rank g =
  let g = uncut_all (whole rule i rank) g in
  let dropped =
    List.filter (fun a -> to_result rule a = None) (List.init rank Fun.id)
  in
  let g = uncut_all (Array.of_list dropped) g in
  Grid.map_axes (fun a -> Option.get (to_result rule a)) g

(* The target moved back to operand [i]'s axes: a result axis no operand axis
   becomes holds copies, then the axes it reads whole do too. *)
let backward rule i rank target =
  let source = Hashtbl.create 8 in
  for a = 0 to rank - 1 do
    Option.iter (fun r -> Hashtbl.replace source r a) (to_result rule a)
  done;
  let orphans =
    Array.of_list
      (List.filter_map
         (fun (axis, _) -> if Hashtbl.mem source axis then None else Some axis)
         (Array.to_list (Grid.cuts target)))
  in
  let g = Grid.map_axes (Hashtbl.find source) (uncut_all orphans target) in
  uncut_all (whole rule i rank) g

(* Whether an operand at [g] gives every device of [want] the part [want] gives
   it: the same windows, or the whole on each of those devices. *)
let covers g want =
  Grid.equal g want
  || Grid.cuts g = [||]
     && Array.for_all
          (fun k -> Array.mem k (Grid.devices g))
          (Grid.devices want)

let route ~by rule ps shapes =
  let n = Array.length ps in
  if n = 0 || n <> Array.length shapes then
    invalid_argf "%s: a route of %d placements and %d shapes" by n
      (Array.length shapes);
  let rank i = Array.length shapes.(i) in
  let check axes i =
    Array.iter
      (fun a ->
        if a < 0 || a >= rank i then
          invalid_argf "%s: axis %d is not an axis of an operand of rank %d" by
            a (rank i))
      axes
  in
  (match rule with
  | Reduce axes | Along axes -> Array.iteri (fun i _ -> check axes i) ps
  | Gather axis ->
      if n <> 2 then invalid_argf "%s: a gather reads 2 operands, not %d" by n;
      check [| axis |] 1
  | Into axis -> Array.iteri (fun i _ -> check [| axis |] i) ps
  | Elementwise | Replicated -> ());
  let concrete =
    List.filter (fun i -> not (is_constant ps.(i))) (List.init n Fun.id)
  in
  match concrete with
  | [] -> { operands = Array.copy ps; result = Devices.anywhere }
  | first :: _ ->
      let set = Devices.set ps.(first) in
      List.iter
        (fun i ->
          let s = Devices.set ps.(i) in
          if Devices.number s <> Devices.number set then
            invalid_argf "%s: operands on %a and %a" by Devices.pp set
              Devices.pp s)
        concrete;
      let fwd =
        List.map
          (fun i -> forward rule i (rank i) (Devices.grid ps.(i)))
          concrete
      in
      let target =
        match rule with
        | Replicated -> Devices.grid (Devices.on set)
        | Into _ when not (is_constant ps.(n - 1)) ->
            forward rule (n - 1) (rank (n - 1)) (Devices.grid ps.(n - 1))
        | _ -> (
            match List.rev (List.filter (fun g -> Grid.cuts g <> [||]) fwd) with
            | g :: _ -> g
            | [] -> (
                match
                  List.sort_uniq Int.compare (List.filter_map Grid.one fwd)
                with
                | [ k ] -> Grid.device k
                | k :: k' :: _ ->
                    invalid_argf
                      "%s: operands on %s and %s alone; place one beside the \
                       other"
                      by
                      (Rig.name (Devices.rig set k))
                      (Rig.name (Devices.rig set k'))
                | [] -> List.hd fwd))
      in
      let operands =
        Array.mapi
          (fun i p ->
            let want = backward rule i (rank i) target in
            let p =
              if (not (is_constant p)) && covers (Devices.grid p) want then p
              else Devices.v ~by set want
            in
            (match Grid.window (Devices.grid p) shapes.(i) 0 with
            | Ok _ -> ()
            | Error e -> invalid_argf "%s: %s" by e);
            p)
          ps
      in
      { operands; result = Devices.v ~by set target }
