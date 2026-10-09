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
  | Move of Nx_array.Move.t
  | Replicated

type 'd t = {
  operands : 'd Devices.placement array;
  result : 'd Devices.placement;
}

let is_constant p = p == Devices.anywhere

(* The axes operand [i] of shape [shape] reads whole on each device. *)
let whole rule i shape =
  match rule with
  | Elementwise | Move _ -> [||]
  | Reduce axes | Along axes -> axes
  | Gather axis -> if i = 1 then [| axis |] else [||]
  | Into axis -> [| axis |]
  | Replicated -> Array.init (Array.length shape) Fun.id

let product s lo hi =
  let p = ref 1 in
  for a = lo to hi - 1 do
    p := !p * s.(a)
  done;
  !p

(* The result axis that axis [a] of a value of shape [s] becomes under [m],
   where it keeps its elements in order: a reshape keeps an axis whose extent,
   and the product of the extents before it, the result has too. *)
let moved (m : Nx_array.Move.t) s a =
  match m with
  | Permute p -> Array.find_index (( = ) a) p
  | Broadcast s' ->
      let a' = a + Array.length s' - Array.length s in
      if s.(a) = s'.(a') then Some a' else None
  | Reshape s' ->
      let before = product s 0 a in
      let rec find a' =
        if a' >= Array.length s' then None
        else if product s' 0 a' = before && s'.(a') = s.(a) then Some a'
        else find (a' + 1)
      in
      find 0
  | Slice rs ->
      let r = rs.(a) in
      if r.start = 0 && r.step = 1 && r.count = s.(a) then Some a else None
  | Window ws ->
      if Array.exists (fun (w : Nx_array.Move.window) -> w.axis = a) ws then
        None
      else Some a

(* The result axis that axis [a] of an operand of shape [shape] becomes, if
   any. *)
let to_result rule shape a =
  match rule with
  | Reduce axes ->
      if Array.mem a axes then None
      else
        Some (a - Array.fold_left (fun n r -> if r < a then n + 1 else n) 0 axes)
  | Move m -> moved m shape a
  | Elementwise | Along _ | Gather _ | Into _ | Replicated -> Some a

let uncut_all axes g = Array.fold_left (fun g axis -> Grid.uncut g ~axis) g axes

(* Operand [i]'s grid on the result's axes: the axes it reads whole, and those
   the result drops, hold copies. *)
let forward rule i shape g =
  let g = uncut_all (whole rule i shape) g in
  let dropped =
    List.filter
      (fun a -> to_result rule shape a = None)
      (List.init (Array.length shape) Fun.id)
  in
  let g = uncut_all (Array.of_list dropped) g in
  Grid.map_axes (fun a -> Option.get (to_result rule shape a)) g

(* The target moved back to operand [i]'s axes: a result axis no operand axis
   becomes holds copies, then the axes it reads whole do too. *)
let backward rule i shape target =
  let source = Hashtbl.create 8 in
  for a = 0 to Array.length shape - 1 do
    Option.iter (fun r -> Hashtbl.replace source r a) (to_result rule shape a)
  done;
  let orphans =
    Array.of_list
      (List.filter_map
         (fun (axis, _) -> if Hashtbl.mem source axis then None else Some axis)
         (Array.to_list (Grid.cuts target)))
  in
  let g = Grid.map_axes (Hashtbl.find source) (uncut_all orphans target) in
  uncut_all (whole rule i shape) g

(* Whether an operand at [g] gives every device of [want] the part [want] gives
   it: the same windows, or the whole on each of those devices. *)
let covers g want =
  Grid.equal g want
  || Grid.cuts g = [||]
     && Array.for_all
          (fun k -> Array.mem k (Grid.devices g))
          (Grid.devices want)

(* Placements an elementwise operation or a movement reads its operands at with
   nothing to work out: every operand a constant, or every one at one placement
   [p], constants apart, that cuts no axis. *)
type 'd common = Constants | Uncut of 'd Devices.placement | Other

let common (type d) (ps : d Devices.placement array) : d common =
  let n = Array.length ps in
  let rec go i (p : d Devices.placement) =
    if i = n then p
    else if is_constant ps.(i) then go (i + 1) p
    else if is_constant p || ps.(i) == p then go (i + 1) ps.(i)
    else raise_notrace Exit
  in
  match go 0 Devices.anywhere with
  | p when is_constant p -> Constants
  | p -> if Grid.cuts (Devices.grid p) = [||] then Uncut p else Other
  | exception Exit -> Other

(* Any route, worked out from the operands' grids. *)
let general ~by rule ps shapes =
  let n = Array.length ps in
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
  | Move m -> (
      if n <> 1 then invalid_argf "%s: a movement reads 1 operand, not %d" by n;
      try ignore (Nx_array.Move.shape m shapes.(0))
      with Invalid_argument e -> invalid_argf "%s: %s" by e)
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
          (fun i -> forward rule i shapes.(i) (Devices.grid ps.(i)))
          concrete
      in
      let target =
        match rule with
        | Replicated -> Devices.grid (Devices.on set)
        | Into _ when not (is_constant ps.(n - 1)) ->
            forward rule (n - 1) shapes.(n - 1) (Devices.grid ps.(n - 1))
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
            let want = backward rule i shapes.(i) target in
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

let route ~by rule ps shapes =
  let n = Array.length ps in
  if n = 0 || n <> Array.length shapes then
    invalid_argf "%s: a route of %d placements and %d shapes" by n
      (Array.length shapes);
  let simple p =
    match rule with
    | Elementwise -> Some { operands = Array.make n p; result = p }
    | Move m when n = 1 -> (
        match Nx_array.Move.shape m shapes.(0) with
        | _ -> Some { operands = [| p |]; result = p }
        | exception Invalid_argument e -> invalid_argf "%s: %s" by e)
    | Move _ | Reduce _ | Along _ | Gather _ | Into _ | Replicated -> None
  in
  let fast =
    match common ps with
    | Constants -> (
        match simple Devices.anywhere with
        | Some r -> Some { r with operands = Array.copy ps }
        | None -> None)
    | Uncut p -> simple p
    | Other -> None
  in
  match fast with Some r -> r | None -> general ~by rule ps shapes
