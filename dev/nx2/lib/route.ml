(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

type axis = Becomes of int | Whole | Reduced of int | Scanned of int
type rule = Elementwise | Axes of axis array array | Into of axis array array

type 'd t = {
  operands : 'd Devices.placement array;
  result : 'd Devices.placement;
}

(* The result axis that axis [a] of operand [i] becomes, if any. *)
let becomes rule i a =
  match rule with
  | Elementwise -> Some a
  | Axes t | Into t -> (
      match t.(i).(a) with
      | Becomes a' | Scanned a' -> Some a'
      | Whole | Reduced _ -> None)

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

let full d = { Nx_array.Move.start = 0; count = d; step = 1 }

let window rule ~operand:i shape (w : Nx_array.Move.range array) =
  match rule with
  | Elementwise -> w
  | Axes t | Into t ->
      Array.mapi
        (fun a d ->
          match t.(i).(a) with
          | Becomes a' | Scanned a' -> w.(a')
          | Whole | Reduced _ -> full d)
        shape

let keeps rule a' =
  match rule with
  | Elementwise -> false
  | Axes t | Into t ->
      not
        (Array.exists
           (fun row -> Array.mem (Becomes a') row || Array.mem (Scanned a') row)
           t)

let uncut_all axes g = Array.fold_left (fun g axis -> Grid.uncut g ~axis) g axes

(* Operand [i]'s grid, of rank [rank], on the result's axes: an axis it becomes
   none of holds copies. [Grid.map_axes] renumbers the cut axes alone, each one
   that becomes a result axis. *)
let forward rule i rank g =
  let g = ref g in
  for a = 0 to rank - 1 do
    if becomes rule i a = None then g := Grid.uncut !g ~axis:a
  done;
  Grid.map_axes (fun a -> Option.get (becomes rule i a)) !g

(* The target moved back to operand [i]'s axes, of rank [rank]: a result axis no
   axis of it becomes holds copies. *)
let backward rule i rank target =
  let source a' =
    let rec find a =
      if a = rank then None
      else if becomes rule i a = Some a' then Some a
      else find (a + 1)
    in
    find 0
  in
  let orphans =
    Array.of_list
      (List.filter_map
         (fun (axis, _) -> if source axis = None then Some axis else None)
         (Array.to_list (Grid.cuts target)))
  in
  Grid.map_axes (fun a' -> Option.get (source a')) (uncut_all orphans target)

(* The factor of operand [i]'s axis [a], if [rule] reduces it. *)
let factor rule i a =
  match rule with
  | Elementwise -> None
  | Axes t | Into t -> (
      match t.(i).(a) with
      | Reduced f -> Some f
      | Becomes _ | Whole | Scanned _ -> None)

let reduced rule i rank =
  Array.of_list
    (List.filter (fun a -> factor rule i a <> None) (List.init rank Fun.id))

(* Operand [i]'s grid [g], of rank [rank], along its reduced axes alone, each
   numbered by its factor: operands whose reduced axes are cut alike have equal
   ones. *)
let factors rule i rank g =
  let g = ref g in
  for a = 0 to rank - 1 do
    if factor rule i a = None then g := Grid.uncut !g ~axis:a
  done;
  Grid.map_axes (fun a -> Option.get (factor rule i a)) !g

let scanned rule i rank =
  match rule with
  | Elementwise -> [||]
  | Axes t | Into t ->
      Array.of_list
        (List.filter
           (fun a -> match t.(i).(a) with Scanned _ -> true | _ -> false)
           (List.init rank Fun.id))

(* Whether an operand at [g] gives every device of [want] the part [want] gives
   it, and lies on no other device: the same windows, or the whole on each of
   exactly those devices. An operand whole on more devices is read at [want],
   each device of [want] taking its own array. *)
let covers g want =
  let same_devices () =
    let have = Grid.devices g and need = Grid.devices want in
    Array.length have = Array.length need
    && Array.for_all (fun k -> Array.mem k have) need
  in
  Grid.equal g want || ((not (Grid.is_cut g)) && same_devices ())

type 'd common = Every_set | One of 'd Devices.placement | Other

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
  | Some p -> One p
  | exception Exit -> Other

(* Raises naming [by] where [rule] does not name each axis of operands of
   [shapes]. *)
let check ~by rule shapes =
  match rule with
  | Elementwise -> ()
  | Axes t | Into t ->
      let fits axes s = Array.length axes = Array.length s in
      if
        Array.length t <> Array.length shapes
        || not (Array.for_all2 fits t shapes)
      then invalid_argf "%s: a rule that does not name each operand's axes" by

(* Any route, worked out from the operands' grids. *)
let general ~by rule ps shapes =
  let n = Array.length ps in
  let placed i = Option.get ps.(i) in
  let rank i = Array.length shapes.(i) in
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
          (fun i -> forward rule i (rank i) (Devices.grid (placed i)))
          concrete
      in
      let target =
        match rule with
        | Into _ when ps.(n - 1) <> None ->
            forward rule (n - 1) (rank (n - 1)) (Devices.grid (placed (n - 1)))
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
      let wants = Array.init n (fun i -> backward rule i (rank i) target) in
      (* Where an operand lies cut along a reduced axis, the grid each operand
         with reduced axes is read at: where it lies, or the first such
         operand's grid where it reads its axes as that one does and lies
         whole on each of its devices or is of every set. Each must cut its
         reduced axes as the first does each factor, give what the target
         wants of it with them uncut, and give the target, less the axes it
         lacks, forward, one of them all of it: each device then holds part
         of each output's terms, and the devices that hold one window of the
         results hold them all. An operand with no reduced axis is read as
         any other. *)
      let partial =
        let cuts_reduced i =
          let axes = reduced rule i (rank i) in
          Array.exists
            (fun (a, _) -> Array.mem a axes)
            (Grid.cuts (Devices.grid (placed i)))
        in
        let row i =
          match rule with Elementwise -> [||] | Axes t | Into t -> t.(i)
        in
        let read c i =
          let g = Devices.grid (placed c) in
          let fits q =
            let axes = reduced rule i (rank i) in
            let lies =
              match ps.(i) with
              | Some p ->
                  let h = Devices.grid p in
                  Grid.equal h q
                  || (covers h q && Result.is_ok (Grid.window q shapes.(i) 0))
              | None -> Result.is_ok (Grid.window q shapes.(i) 0)
            in
            lies
            && covers (uncut_all axes q) wants.(i)
            && Grid.equal
                 (forward rule i (rank i) q)
                 (forward rule i (rank i) wants.(i))
            && Grid.equal
                 (factors rule i (rank i) q)
                 (factors rule c (rank c) g)
          in
          let q =
            match ps.(i) with
            | Some p when cuts_reduced i -> Some (Devices.grid p)
            | Some _ | None -> if row i = row c then Some g else None
          in
          match q with Some q when fits q -> Some q | Some _ | None -> None
        in
        match List.find_opt cuts_reduced concrete with
        | None -> None
        | Some c ->
            let reads = Array.init n (read c) in
            let all = List.init n Fun.id in
            let read_cut i =
              reduced rule i (rank i) = [||] || reads.(i) <> None
            in
            (* One operand gives the whole target forward: the grid axes that
               cut the reduced axes then hold copies in the results. *)
            let gives i =
              match reads.(i) with
              | Some q -> Grid.equal (forward rule i (rank i) q) target
              | None -> false
            in
            if List.for_all read_cut all && List.exists gives all then
              Some reads
            else None
      in
      let operands =
        Array.mapi
          (fun i p ->
            let want = wants.(i) in
            let at =
              match partial with
              | Some reads -> reads.(i)
              | None -> None
            in
            let p =
              match (at, p) with
              | Some q, Some p when Grid.equal (Devices.grid p) q -> p
              | Some q, (Some _ | None) -> Devices.v ~by set q
              | None, Some p when covers (Devices.grid p) want -> p
              | None, (Some _ | None) -> Devices.v ~by set want
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
  check ~by rule shapes;
  (* Operands at one placement that cuts no axis are read there, and the
     results lie there: each device reads them whole and computes all of the
     results. *)
  match common ps with
  | Every_set -> None
  | One p when not (Grid.is_cut (Devices.grid p)) ->
      Some { operands = Array.make n p; result = p }
  | One p when rule = Elementwise -> (
      (* Operands of one shape: one that has [p]'s cut axes is all of them. *)
      match Grid.window (Devices.grid p) shapes.(0) 0 with
      | Ok _ -> Some { operands = Array.make n p; result = p }
      | Error e -> invalid_argf "%s: %s" by e)
  | One _ | Other -> general ~by rule ps shapes

let at ~by rule shapes p =
  let kept =
    Array.of_list
      (List.filter_map
         (fun (a', _) -> if keeps rule a' then Some a' else None)
         (Array.to_list (Grid.cuts (Devices.grid p))))
  in
  let set = Devices.set p in
  let target = uncut_all kept (Devices.grid p) in
  let operand i s =
    let g = backward rule i (Array.length s) target in
    (match Grid.window g s 0 with
    | Ok _ -> ()
    | Error e -> invalid_argf "%s: %s" by e);
    Devices.v ~by set g
  in
  { operands = Array.mapi operand shapes; result = Devices.v ~by set target }
