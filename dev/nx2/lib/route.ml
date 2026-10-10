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
(* CR: Match reshape cuts using their tile count. Reshaping [8] split over
   two devices to [2;4] currently gathers a full [8] on each device;
   each existing [4] shard can instead reshape to [1;4]. For positive
   shapes, match axes by equal prefix products and extents divisible by
   that count. Walk actual cuts and use this one correspondence in
   forward, backward and localize; extent equality needlessly loses it. *)
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
      (* CR: Preserve this axis's cut for a window of size 1 and step 1.
         It keeps every element in order and only appends a unit axis.
         sliding_window ~window:1 of [8] split over two devices currently
         gathers two full [8] copies; each [4] shard can give a [4;1] view.
         Recognize the identity axis here: localize and Layout.move already
         handle it, preserving the Window primitive's other semantics. *)
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
  || (not (Grid.is_cut g))
     && Array.for_all
          (fun k -> Array.mem k (Grid.devices g))
          (Grid.devices want)

(* Placements an elementwise operation or a movement reads its operands at with
   nothing to work out: every operand of every set, or every one at one
   placement [p], those of every set apart, that cuts no axis. *)
type 'd common = Every_set | Uncut of 'd Devices.placement | Other

let common (type d) (ps : d Devices.placement option array) : d common =
  let n = Array.length ps in
  let rec go i (p : d Devices.placement option) =
    if i = n then p
    else
      match (ps.(i), p) with
      | None, _ -> go (i + 1) p
      | Some q, None -> go (i + 1) (Some q)
      | Some q, Some p' -> if q == p' then go (i + 1) p else raise_notrace Exit
  in
  match go 0 None with
  | None -> Every_set
  | Some p -> if Grid.is_cut (Devices.grid p) then Other else Uncut p
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
  let placed i = Option.get ps.(i) in
  let concrete = List.filter (fun i -> ps.(i) <> None) (List.init n Fun.id) in
  match concrete with
  | [] -> None
  | first :: _ ->
      let set = Devices.set (placed first) in
      List.iter
        (fun i ->
          let s = Devices.set (placed i) in
          if Devices.number s <> Devices.number set then
            invalid_argf "%s: operands on %a and %a" by Devices.pp set
              Devices.pp s)
        concrete;
      let fwd =
        List.map
          (fun i -> forward rule i shapes.(i) (Devices.grid (placed i)))
          concrete
      in
      let target =
        match rule with
        | Replicated -> Devices.grid (Devices.on set)
        | Into _ when ps.(n - 1) <> None ->
            forward rule (n - 1) shapes.(n - 1) (Devices.grid (placed (n - 1)))
        | _ -> (
            match List.rev (List.filter Grid.is_cut fwd) with
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
              match p with
              | Some p when covers (Devices.grid p) want -> p
              | Some _ | None -> Devices.v ~by set want
            in
            (match Grid.window (Devices.grid p) shapes.(i) 0 with
            | Ok _ -> ()
            | Error e -> invalid_argf "%s: %s" by e);
            p)
          ps
      in
      Some { operands; result = Devices.v ~by set target }

let route ~by rule ps shapes =
  let n = Array.length ps in
  if n = 0 || n <> Array.length shapes then
    invalid_argf "%s: a route of %d placements and %d shapes" by n
      (Array.length shapes);
  let simple p =
    match rule with
    | Elementwise -> Some (Some { operands = Array.make n p; result = p })
    | Move m when n = 1 -> (
        match Nx_array.Move.shape m shapes.(0) with
        | _ -> Some (Some { operands = [| p |]; result = p })
        | exception Invalid_argument e -> invalid_argf "%s: %s" by e)
    | Move _ | Reduce _ | Along _ | Gather _ | Into _ | Replicated -> None
  in
  let fast =
    match (common ps, rule) with
    | Every_set, (Elementwise | Replicated) -> Some None
    | Every_set, Move m -> (
        match Nx_array.Move.shape m shapes.(0) with
        | _ -> Some None
        | exception Invalid_argument e -> invalid_argf "%s: %s" by e)
    | Uncut p, _ -> simple p
    | (Every_set | Other), _ -> None
  in
  match fast with Some r -> r | None -> general ~by rule ps shapes
