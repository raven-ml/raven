(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

type rule =
  | Elementwise
  | Reduce of { axes : int array; along : int array }
  | Along of int array
  | Gather of int
  | Into of int
  | Contract of { batch : (int * int) array; contracting : (int * int) array }
  | Move of Nx_array.Move.t
  | Replicated

type 'd t = {
  operands : 'd Devices.placement array;
  result : 'd Devices.placement;
}

(* A contraction operand's side of its pairs: [a]'s first, [b]'s second. *)
let side i (p : int * int) = if i = 0 then fst p else snd p

(* The axes of [axes] that a value of [shape] has. *)
let within shape axes =
  Array.of_list
    (List.filter (fun a -> a < Array.length shape) (Array.to_list axes))

(* The result axis that axis [a] becomes where the axes [axes] drop. *)
let dropped axes a =
  a - Array.fold_left (fun n r -> if r < a then n + 1 else n) 0 axes

(* The axes operand [i] of shape [shape] reads whole on each device. *)
let whole rule i shape =
  match rule with
  | Elementwise | Move _ -> [||]
  | Reduce { axes; along } -> within shape (Array.append axes along)
  | Along axes -> within shape axes
  | Gather axis -> if i = 1 then [| axis |] else [||]
  | Into axis -> [| axis |]
  | Contract { contracting; _ } ->
      if i = 2 then [||] else Array.map (side i) contracting
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

(* The number of [a]'s axes a contraction of [shapes] keeps; [0] for any other
   rule. *)
let kept_free rule shapes =
  match rule with
  | Contract { batch; contracting } ->
      Array.length shapes.(0) - Array.length batch - Array.length contracting
  | Elementwise | Reduce _ | Along _ | Gather _ | Into _ | Move _ | Replicated
    ->
      0

(* The result axis that axis [a] of operand [i] of shape [shape] becomes, if
   any; [free] is {!kept_free}. A contraction's result has the batch pairs'
   axes, then [a]'s other axes, then [b]'s; [init] has its shape. *)
let to_result rule ~free i shape a =
  match rule with
  | Reduce { axes; _ } ->
      if Array.mem a axes then None else Some (dropped axes a)
  | Move m -> moved m shape a
  | Contract { batch; contracting } ->
      let paired a = Array.exists (fun p -> side i p = a) in
      if i = 2 then Some a
      else if paired a contracting then None
      else
        begin match Array.find_index (fun p -> side i p = a) batch with
        | Some j -> Some j
        | None ->
            let before = ref 0 in
            for a' = 0 to a - 1 do
              if not (paired a' batch || paired a' contracting) then incr before
            done;
            Some (Array.length batch + (if i = 0 then 0 else free) + !before)
        end
  | Elementwise | Along _ | Gather _ | Into _ | Replicated -> Some a

let full d = { Nx_array.Move.start = 0; count = d; step = 1 }

let window rule i shape (w : Nx_array.Move.range array) =
  match rule with
  | Elementwise -> w
  | Reduce _ | Along _ | Gather _ | Into _ | Contract _ | Move _ | Replicated ->
      (* [b]'s free axes follow [a]'s, whose number the result's rank gives. *)
      let free =
        match rule with
        | Contract { contracting; _ } ->
            Array.length w - Array.length shape + Array.length contracting
        | Elementwise | Reduce _ | Along _ | Gather _ | Into _ | Move _
        | Replicated ->
            0
      in
      let whole = whole rule i shape in
      Array.mapi
        (fun a d ->
          if Array.mem a whole then full d
          else
            match to_result rule ~free i shape a with
            | Some a' -> w.(a')
            | None -> full d)
        shape

let uncut_all axes g = Array.fold_left (fun g axis -> Grid.uncut g ~axis) g axes

(* Operand [i]'s grid on the result's axes: the axes it reads whole, and those
   the result drops, hold copies. *)
let forward rule ~free i shape g =
  let g = uncut_all (whole rule i shape) g in
  let dropped =
    List.filter
      (fun a -> to_result rule ~free i shape a = None)
      (List.init (Array.length shape) Fun.id)
  in
  let g = uncut_all (Array.of_list dropped) g in
  Grid.map_axes (fun a -> Option.get (to_result rule ~free i shape a)) g

(* The target moved back to operand [i]'s axes: a result axis no operand axis
   becomes holds copies, then the axes it reads whole do too. *)
let backward rule ~free i shape target =
  let source = Hashtbl.create 8 in
  for a = 0 to Array.length shape - 1 do
    Option.iter
      (fun r -> Hashtbl.replace source r a)
      (to_result rule ~free i shape a)
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

(* Raises naming [by] where [rule] does not fit operands of [shapes]. *)
let check ~by rule shapes =
  let n = Array.length shapes in
  let fits axis i =
    let rank = Array.length shapes.(i) in
    if axis < 0 || axis >= rank then
      invalid_argf "%s: axis %d is not an axis of an operand of rank %d" by axis
        rank
  in
  (* Axes the results may have beyond every operand's. *)
  let natural axes =
    for j = 0 to Array.length axes - 1 do
      if axes.(j) < 0 then invalid_argf "%s: a negative axis %d" by axes.(j)
    done
  in
  match rule with
  | Reduce { axes; along } ->
      natural axes;
      natural along
  | Along axes -> natural axes
  | Contract _ ->
      if n <> 2 && n <> 3 then
        invalid_argf "%s: a contraction reads 2 or 3 operands, not %d" by n
  | Gather axis ->
      if n <> 2 then invalid_argf "%s: a gather reads 2 operands, not %d" by n;
      fits axis 1
  | Into axis ->
      for i = 0 to n - 1 do
        fits axis i
      done
  | Move m -> (
      if n <> 1 then invalid_argf "%s: a movement reads 1 operand, not %d" by n;
      try ignore (Nx_array.Move.shape m shapes.(0))
      with Invalid_argument e -> invalid_argf "%s: %s" by e)
  | Elementwise | Replicated -> ()

(* Any route, worked out from the operands' grids. *)
let general ~by rule ps shapes =
  let n = Array.length ps in
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
      let free = kept_free rule shapes in
      let fwd =
        List.map
          (fun i -> forward rule ~free i shapes.(i) (Devices.grid (placed i)))
          concrete
      in
      let target =
        match rule with
        | Into _ when ps.(n - 1) <> None ->
            forward rule ~free (n - 1)
              shapes.(n - 1)
              (Devices.grid (placed (n - 1)))
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
            let want = backward rule ~free i shapes.(i) target in
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
  check ~by rule shapes;
  (* Operands at one placement that cuts no axis are read there, and the
     results lie there: each device reads them whole and computes all of the
     results. *)
  match common ps with
  | Every_set -> None
  | One p when not (Grid.is_cut (Devices.grid p)) ->
      Some { operands = Array.make n p; result = p }
  | One p when rule = Elementwise -> (
      (* Operands of one shape: one that [p] divides is all of them. *)
      match Grid.window (Devices.grid p) shapes.(0) 0 with
      | Ok _ -> Some { operands = Array.make n p; result = p }
      | Error e -> invalid_argf "%s: %s" by e)
  | One _ | Other -> general ~by rule ps shapes

(* Whether the results keep their axis [a'] whole by [rule], over operands of
   [shapes]. *)
let keeps rule shapes a' =
  match rule with
  | Elementwise | Gather _ | Contract _ -> false
  | Reduce { axes; along } ->
      Array.exists
        (fun a -> (not (Array.mem a axes)) && dropped axes a = a')
        along
  | Along axes -> Array.mem a' axes
  | Into axis -> a' = axis
  | Move m ->
      let s = shapes.(0) in
      let image = ref false in
      for a = 0 to Array.length s - 1 do
        if moved m s a = Some a' then image := true
      done;
      not !image
  | Replicated -> true

let at ~by rule shapes p =
  let kept =
    Array.of_list
      (List.filter_map
         (fun (a', _) -> if keeps rule shapes a' then Some a' else None)
         (Array.to_list (Grid.cuts (Devices.grid p))))
  in
  let set = Devices.set p in
  let target = uncut_all kept (Devices.grid p) in
  let free = kept_free rule shapes in
  let operand i s =
    let g = backward rule ~free i s target in
    (match Grid.window g s 0 with
    | Ok _ -> ()
    | Error e -> invalid_argf "%s: %s" by e);
    Devices.v ~by set g
  in
  { operands = Array.mapi operand shapes; result = Devices.v ~by set target }
