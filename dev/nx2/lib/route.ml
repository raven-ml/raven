(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

type axis =
  | Becomes of int
  | Whole
  | Reduced of int
  | Scanned of int
  | Regroups of { axis : int; extent : int }

type rule = Elementwise | Axes of axis array array | Into of axis array array

type compute = Windows | Partials

type 'd t = {
  operands : 'd Devices.placement array;
  result : 'd Devices.placement;
  compute : compute;
}

(* The result axis that a cut of axis [a] of operand [i], of shape [s], into
   [t] tiles lands on: one it becomes, or one it regroups into where [t]
   divides both extents. *)
let lands rule i s a t =
  match rule with
  | Elementwise -> Some a
  | Axes r | Into r -> (
      match r.(i).(a) with
      | Becomes a' | Scanned a' -> Some a'
      | Regroups { axis; extent } ->
          if s.(a) mod t = 0 && extent mod t = 0 then Some axis else None
      | Whole | Reduced _ -> None)

let product s lo hi =
  let p = ref 1 in
  for a = lo to hi - 1 do
    p := !p * s.(a)
  done;
  !p

(* What [m] makes of axis [a] of a value of shape [s]. A reshape pairs the
   axes of one product of the extents before them in order, each with the
   result's next axis of that product and of its extent: [1; 8] to [1; 1; 8]
   pairs 0 with 0 and 1 with 2, and [1; 1] to [1] pairs 0 alone. An unpaired
   axis regroups into the result's axis of more than one element of that
   product: [8] into [2; 4]'s axis 0, [2; 4]'s axis 0 into [8], and [8] into
   [1; 2; 4]'s axis 1. Each window of a cut into tiles that divide both
   extents holds whole rows on both sides, so each shard reshapes alone: [8]
   over two devices holds [4] on each, which becomes [1; 4]. *)
let moved (m : Nx_array.Move.t) s a : axis =
  match m with
  | Permute p -> (
      match Iarray.find_index (( = ) a) p with
      | Some a' -> Becomes a'
      | None -> Whole)
  | Broadcast s' ->
      let a' = a + Iarray.length s' - Array.length s in
      if s.(a) = Iarray.get s' a' then Becomes a' else Whole
  | Reshape s' -> (
      let s' = Iarray.to_array s' in
      let before = product s 0 a in
      let run sh =
        List.filter
          (fun x -> product sh 0 x = before)
          (List.init (Array.length sh) Fun.id)
      in
      let rec next e = function
        | [] -> None
        | y :: ys -> if s'.(y) = e then Some (y, ys) else next e ys
      in
      let rec pair xs ys =
        match xs with
        | [] -> None
        | x :: xs -> (
            match next s.(x) ys with
            | Some (y, ys') -> if x = a then Some y else pair xs ys'
            | None -> if x = a then None else pair xs ys)
      in
      match pair (run s) (run s') with
      | Some a' -> Becomes a'
      | None when before > 0 && s.(a) > 1 -> (
          match List.find_opt (fun y -> s'.(y) > 1) (run s') with
          | Some y -> Regroups { axis = y; extent = s'.(y) }
          | None -> Whole)
      | None -> Whole)
  | Slice rs ->
      let r = Iarray.get rs a in
      if r.start = 0 && r.step = 1 && r.count = s.(a) then Becomes a else Whole
  | Window ws ->
      (* CR: Preserve this axis's cut for a window of size 1 and step 1.
         It keeps every element in order and only appends a unit axis.
         sliding_window ~window:1 of [8] split over two devices currently
         gathers two full [8] copies; each [4] shard can give a [4;1] view.
         Recognize the identity axis here: localize and Layout.move already
         handle it, preserving the Window primitive's other semantics. *)
      if Iarray.exists (fun (w : Nx_array.Move.window) -> w.axis = a) ws then
        Whole
      else Becomes a

let full d = { Nx_array.Move.start = 0; count = d; step = 1 }

let window rule ~operand:i shape (w : Nx_array.Move.range array) =
  match rule with
  | Elementwise -> w
  | Axes t | Into t ->
      Array.mapi
        (fun a d ->
          match t.(i).(a) with
          | Becomes a' | Scanned a' -> w.(a')
          | Regroups { axis; extent } ->
              let r = w.(axis) in
              let scale n = n * d / extent in
              { r with start = scale r.start; count = scale r.count }
          | Whole | Reduced _ -> full d)
        shape

let keeps rule a' =
  match rule with
  | Elementwise -> false
  | Axes t | Into t ->
      let into : axis -> bool = function
        | Becomes b | Scanned b | Regroups { axis = b; _ } -> b = a'
        | Whole | Reduced _ -> false
      in
      not (Array.exists (Array.exists into) t)

let uncut_all axes g = Array.fold_left (fun g axis -> Grid.uncut g ~axis) g axes

(* The tiles of [g]'s cut axis [a]. *)
let tiles g a =
  Option.get
    (Array.find_map (fun (c, t) -> if c = a then Some t else None) (Grid.cuts g))

(* Operand [i]'s grid [g], of shape [s], on the result's axes: a cut that lands
   on none of them holds copies. [Grid.map_axes] renumbers the cut axes alone,
   each one that lands on a result axis. *)
let forward rule i s g =
  let lost =
    Array.of_list
      (List.filter_map
         (fun (a, t) -> if lands rule i s a t = None then Some a else None)
         (Array.to_list (Grid.cuts g)))
  in
  let g = uncut_all lost g in
  Grid.map_axes (fun a -> Option.get (lands rule i s a (tiles g a))) g

(* The target moved back to operand [i]'s axes, of shape [s]: a cut of a result
   axis that no cut of it lands on holds copies. *)
let backward rule i s target =
  let source a' t =
    let rec find a =
      if a = Array.length s then None
      else if lands rule i s a t = Some a' then Some a
      else find (a + 1)
    in
    find 0
  in
  let orphans =
    Array.of_list
      (List.filter_map
         (fun (a', t) -> if source a' t = None then Some a' else None)
         (Array.to_list (Grid.cuts target)))
  in
  let target = uncut_all orphans target in
  Grid.map_axes (fun a' -> Option.get (source a' (tiles target a'))) target

(* [target] with each cut undone that an operand of [shapes] regroups into
   across rows: its tiles do not divide both extents. *)
let aligned rule shapes target =
  match rule with
  | Elementwise -> target
  | Axes r | Into r ->
      let across (a', t) =
        let straddles i s =
          Array.exists
            (fun a ->
              match r.(i).(a) with
              | Regroups { axis; extent } ->
                  axis = a' && (s.(a) mod t <> 0 || extent mod t <> 0)
              | Becomes _ | Scanned _ | Whole | Reduced _ -> false)
            (Array.init (Array.length s) Fun.id)
        in
        Array.exists Fun.id (Array.mapi straddles shapes)
      in
      let undone =
        Array.of_list
          (List.filter_map
             (fun ((a', _) as c) -> if across c then Some a' else None)
             (Array.to_list (Grid.cuts target)))
      in
      uncut_all undone target

(* The factor of operand [i]'s axis [a], if [rule] reduces it. *)
let factor rule i a =
  match rule with
  | Elementwise -> None
  | Axes t | Into t -> (
      match t.(i).(a) with
      | Reduced f -> Some f
      | Becomes _ | Whole | Scanned _ | Regroups _ -> None)

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

(* The axes of operand [i], of shape [s], that [rule] folds: those it reduces
   or scans. *)
let folds rule i s =
  let r = Array.length s in
  Array.append (reduced rule i r) (scanned rule i r)

(* Whether an axis [rule] folds of an operand of [shapes] has no element. *)
let folds_empty rule shapes =
  Array.exists Fun.id
    (Array.mapi
       (fun i s -> Array.exists (fun a -> s.(a) = 0) (folds rule i s))
       shapes)

(* How the devices compute over operands of [shapes] read at [operands]: by
   partials where one is read cut along an axis [rule] folds, each such axis
   holding elements, since the window law fails there; by windows
   otherwise. *)
let computed rule shapes operands =
  let cut i p =
    Array.exists
      (fun (a, _) -> Array.mem a (folds rule i shapes.(i)))
      (Grid.cuts (Devices.grid p))
  in
  if
    Array.exists Fun.id (Array.mapi cut operands)
    && not (folds_empty rule shapes)
  then Partials
  else Windows

(* The number of windows [g] cuts a value into. *)
let windows g = Array.fold_left (fun n (_, t) -> n * t) 1 (Grid.cuts g)

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
          (fun i -> forward rule i shapes.(i) (Devices.grid (placed i)))
          concrete
      in
      (* The target is a fixpoint: placed where the route reads it, each
         operand lies, forward, at the target with the cuts of the axes it
         lacks undone, which makes fewer windows, and the operand the
         target came from lies at it. Routing them there again picks the
         same target. *)
      let most =
        List.fold_left
          (fun best g ->
            match best with
            | Some b when windows g < windows b -> best
            | Some _ | None -> if Grid.is_cut g then Some g else best)
          None fwd
      in
      let target =
        match rule with
        | Into _ when ps.(n - 1) <> None ->
            forward rule (n - 1) shapes.(n - 1) (Devices.grid (placed (n - 1)))
        | _ -> (
            match most with
            | Some g -> g
            | None -> (
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
      (* An [Into]'s results lie where its last operand is read: of every
         set, it is read at the target with the axes it lacks whole. *)
      let target =
        match rule with
        | Into _ when ps.(n - 1) = None ->
            let last = n - 1 in
            forward rule last shapes.(last)
              (backward rule last shapes.(last) target)
        | Elementwise | Axes _ | Into _ -> target
      in
      let target = aligned rule shapes target in
      let wants = Array.init n (fun i -> backward rule i shapes.(i) target) in
      (* An axis the operation folds of no element: each reads it whole. *)
      let empty = folds_empty rule shapes in
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
                 (forward rule i shapes.(i) q)
                 (forward rule i shapes.(i) wants.(i))
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
        | Some _ when empty -> None
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
              | Some q -> Grid.equal (forward rule i shapes.(i) q) target
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
      let compute = computed rule shapes operands in
      Some { operands; result = Devices.v ~by set target; compute }

let route ~by rule ps shapes =
  let n = Array.length ps in
  (* Operands at one placement that cuts no axis are read there, and the
     results lie there: each device reads them whole and computes all of the
     results. *)
  match common ps with
  | Every_set -> None
  | One p when not (Grid.is_cut (Devices.grid p)) ->
      Some { operands = Array.make n p; result = p; compute = Windows }
  | One p when rule = Elementwise -> (
      (* Operands of one shape: one that has [p]'s cut axes is all of them. *)
      match Grid.window (Devices.grid p) shapes.(0) 0 with
      | Ok _ -> Some { operands = Array.make n p; result = p; compute = Windows }
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
  let target = aligned rule shapes (uncut_all kept (Devices.grid p)) in
  let operand i s =
    let g = backward rule i s target in
    (match Grid.window g s 0 with
    | Ok _ -> ()
    | Error e -> invalid_argf "%s: %s" by e);
    Devices.v ~by set g
  in
  let operands = Array.mapi operand shapes in
  {
    operands;
    result = Devices.v ~by set target;
    compute = computed rule shapes operands;
  }
