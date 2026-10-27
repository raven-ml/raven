(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Routes, through the engine's private Route, copied here, against the part of
   each operand a result window depends on. *)

open Windtrap
module R = Nx_array.Move

type b

let m = Nx_support.memory
let mint ds : b Devices.t = Devices.mint ~by:"Nx.devices" ds
let s4 = mint [ m 0; m 1; m 2; m 3 ]
let other = mint [ m 0; m 1 ]
let mesh = Devices.mesh_v ~by:"t" s4 [ ("a", 2); ("b", 2) ]
let placement = Testable.make ~pp:Devices.pp_placement ~equal:Devices.equal
let invalid f = raises_match (Exn.invalid_arg ~substring:"Nx.f: ") f
let route r ps shapes = Route.route ~by:"Nx.f" r ps shapes

(* The route of operands that all have a placement. *)
let placed r ps shapes = Option.get (route r (Array.map Option.some ps) shapes)

(* A matrix product's rule: [a]'s columns and [b]'s rows are its contracting
   pair. *)
let matrix_product =
  Route.Axes [| [| Becomes 0; Reduced 0 |]; [| Reduced 0; Becomes 1 |] |]

let pp_at ppf = function
  | Some p -> Devices.pp_placement ppf p
  | None -> Format.pp_print_string ppf "every set"

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_axis ppf : Route.axis -> unit = function
  | Becomes a -> Format.pp_print_int ppf a
  | Whole -> Format.pp_print_string ppf "whole"
  | Reduced f -> Format.fprintf ppf "reduced %d" f
  | Scanned a -> Format.fprintf ppf "scanned %d" a
  | Regroups { axis; extent } ->
      Format.fprintf ppf "regroups %d of %d" axis extent

let pp_axes ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; "
       (Array.to_list (Array.map (Format.asprintf "%a" pp_axis) a)))

let pp_table ppf t =
  Format.fprintf ppf "[%s]"
    (String.concat "; "
       (Array.to_list (Array.map (Format.asprintf "%a" pp_axes) t)))

let pp_rule ppf = function
  | Route.Elementwise -> Format.pp_print_string ppf "Elementwise"
  | Axes t -> Format.fprintf ppf "Axes %a" pp_table t
  | Into t -> Format.fprintf ppf "Into %a" pp_table t

(* An operation: its rule, its operands' placements and shapes, and its results'
   shape. *)
type case = {
  rule : Route.rule;
  ps : b Devices.placement option array;
  shapes : int array array;
  result : int array;
}

let pp_case ppf c =
  Format.fprintf ppf "%a, results %a, over" pp_rule c.rule pp_ints c.result;
  Array.iteri
    (fun i p -> Format.fprintf ppf " %a at %a," pp_ints c.shapes.(i) pp_at p)
    c.ps

(* A placement of a value of [rank]; of a scalar, uncut. *)
let placement_of rank =
  let uncut =
    [
      Gen.constant None;
      Gen.map (fun k -> Some (Devices.one s4 k)) (Gen.int_range 0 3);
      Gen.constant (Some (Devices.on s4));
    ]
  in
  if rank = 0 then Gen.one_of uncut
  else
    let axis = Gen.int_range 0 (rank - 1) in
    Gen.one_of
      (uncut
      @ [
          Gen.map (fun axis -> Some (Devices.split ~by:"t" ~axis s4)) axis;
          Gen.map
            (fun (x, y) ->
              Some
                (if x = y then Devices.mesh ~by:"t" mesh [ (x, [ "a"; "b" ]) ]
                 else Devices.mesh ~by:"t" mesh [ (x, [ "a" ]); (y, [ "b" ]) ]))
            (Gen.pair axis axis);
        ])

let extent = Gen.of_list [ 0; 1; 3; 4; 5; 8 ]

(* An operand of results of shape [out]: each axis becomes a distinct axis of
   the results, of its extent, plainly or scanned, or regroups into one, both
   extents above one and its own even, or none, whole of any extent or
   reduced. Its reduced axes, in order, have the factors 0, 1 and 2, of
   [factors]' extents. *)
let operand_of factors out =
  let open Gen in
  let rank = Array.length out in
  let* r = int_range 1 3 in
  let+ order = permutation (List.init rank Fun.id @ List.init r (fun _ -> -1))
  and+ along = array ~size:(constant r) extent
  and+ reduced = array ~size:(constant r) bool
  and+ scanned = array ~size:(constant r) bool
  and+ regroups = array ~size:(constant r) bool
  and+ rows = array ~size:(constant r) (of_list [ 2; 4; 6; 8; 12; 16 ]) in
  let axes = Array.of_list (List.filteri (fun a _ -> a < r) order) in
  let factor a =
    let f = ref 0 in
    for a' = 0 to a - 1 do
      if axes.(a') < 0 && reduced.(a') then incr f
    done;
    !f
  in
  let regroup a t =
    t >= 0 && regroups.(a) && out.(t) > 1 && rows.(a) <> out.(t)
  in
  let axis a t : Route.axis =
    if regroup a t then Regroups { axis = t; extent = out.(t) }
    else if t >= 0 then if scanned.(a) then Scanned t else Becomes t
    else if reduced.(a) then Reduced (factor a)
    else Whole
  in
  let extent a t =
    if regroup a t then rows.(a)
    else if t >= 0 then out.(t)
    else if reduced.(a) then factors.(factor a)
    else along.(a)
  in
  (Array.mapi axis axes, Array.mapi extent axes)

(* The operand a scatter writes into: the results' shape, each axis becoming
   itself or read whole. *)
let into_of out =
  Gen.map
    (fun whole ->
      ( Array.mapi
          (fun a w : Route.axis -> if w then Whole else Becomes a)
          whole,
        Array.copy out ))
    (Gen.array ~size:(Gen.constant (Array.length out)) Gen.bool)

let case =
  let open Gen in
  let* rank = int_range 1 3 in
  let* out = array ~size:(constant rank) extent in
  let* n = int_range 1 3 in
  let* kind = int_range 0 2 in
  let* factors = array ~size:(constant 3) extent in
  let* operands =
    match kind with
    | 0 -> constant (Array.make n ([||], out))
    | 1 -> array ~size:(constant n) (operand_of factors out)
    | _ ->
        let+ others = array ~size:(constant (n - 1)) (operand_of factors out)
        and+ into = into_of out in
        Array.append others [| into |]
  in
  let rule =
    match kind with
    | 0 -> Route.Elementwise
    | 1 -> Axes (Array.map fst operands)
    | _ -> Into (Array.map fst operands)
  in
  let shapes = Array.map snd operands in
  (* Often each operand's axis of factor 0 split over the four devices: the
     operands' terms pair up there. *)
  let* paired = bool in
  let factor0 i =
    match rule with
    | Route.Elementwise -> None
    | Axes t | Into t -> Array.find_index (( = ) (Route.Reduced 0)) t.(i)
  in
  (* Often an operand's axis that regroups split over the four devices. *)
  let* split_regroup = bool in
  let regroup i =
    match rule with
    | Route.Elementwise -> None
    | Axes t | Into t ->
        Array.find_index
          (function Route.Regroups _ -> true | _ -> false)
          t.(i)
  in
  let rec placements i =
    if i = n then constant []
    else
      let* p =
        match (factor0 i, regroup i) with
        | Some axis, _ when paired ->
            constant (Some (Devices.split ~by:"t" ~axis s4))
        | _, Some axis when split_regroup ->
            constant (Some (Devices.split ~by:"t" ~axis s4))
        | _ -> placement_of (Array.length shapes.(i))
      in
      let+ rest = placements (i + 1) in
      p :: rest
  in
  let+ ps = placements 0 in
  { rule; ps = Array.of_list ps; shapes; result = out }

let case = Gen.with_pp pp_case case

(* The part of operand [i] that result window [w] depends on. Along an axis of
   [d] that regroups into one of [e], the rows of [d / e] the window's range
   touches. *)
let depends c i (w : R.range array) =
  match c.rule with
  | Route.Elementwise -> w
  | Axes t | Into t ->
      Array.mapi
        (fun a d ->
          match t.(i).(a) with
          | Route.Becomes a' | Scanned a' -> w.(a')
          | Regroups { axis; extent = e } ->
              let r = w.(axis) in
              let lo = r.start * d / e in
              let hi = (((r.start + r.count) * d) + e - 1) / e in
              { R.start = lo; count = hi - lo; step = 1 }
          | Whole | Reduced _ -> { R.start = 0; count = d; step = 1 })
        c.shapes.(i)

let pp_ranges ppf (w : R.range array) =
  Format.fprintf ppf "[%s]"
    (String.concat "; "
       (Array.to_list
          (Array.map
             (fun (r : R.range) -> Printf.sprintf "%d+%d" r.start r.count)
             w)))

let ranges = Testable.make ~pp:pp_ranges ~equal:( = )

let contains (outer : R.range array) (inner : R.range array) =
  Array.for_all2
    (fun (o : R.range) (i : R.range) ->
      i.start >= o.start && i.start + i.count <= o.start + o.count)
    outer inner

let position p k = Array.find_index (( = ) k) (Grid.devices (Devices.grid p))

(* The factor of operand [i]'s axis [a], if the rule reduces it. *)
let factor_of c i a =
  match c.rule with
  | Route.Elementwise -> None
  | Axes t | Into t -> (
      match t.(i).(a) with
      | Route.Reduced f -> Some f
      | Becomes _ | Whole | Scanned _ | Regroups _ -> None)

let is_reduced c i a = factor_of c i a <> None

(* [box] along the axes [keep] holds, whole along the others. *)
let along keep (box : R.range array) (whole : R.range array) =
  Array.mapi (fun a r -> if keep a then r else whole.(a)) box

(* Whether the windows [ws] together hold every index of [box]. *)
let together (ws : R.range array list) (box : R.range array) =
  let r = Array.length box in
  let idx = Array.make r 0 in
  let rec walk a =
    if a = r then
      List.exists
        (fun (w : R.range array) ->
          Array.for_all2
            (fun (h : R.range) i -> i >= h.start && i < h.start + h.count)
            w idx)
        ws
    else
      let ok = ref true in
      for i = box.(a).start to box.(a).start + box.(a).count - 1 do
        idx.(a) <- i;
        if !ok && not (walk (a + 1)) then ok := false
      done;
      !ok
  in
  walk 0

let law =
  prop ~count:2000 "each result window reads operands on its own device" case
    (fun c ->
      let all_constant = Array.for_all (fun p -> p = None) c.ps in
      cover "constants only" all_constant;
      match route c.rule c.ps c.shapes with
      | exception Invalid_argument msg ->
          (* Two operands alone on two devices is the one refusal of valid
             operands. *)
          let alone =
            List.sort_uniq compare
              (List.filter_map
                 (fun p -> Option.bind p Devices.device)
                 (Array.to_list c.ps))
          in
          cover "two devices alone" true;
          greater int ~msg ~than:1 (List.length alone)
      | None -> equal bool ~msg:"no route of values of every set" true all_constant
      | Some r ->
          if all_constant then fail "a route of values of every set"
          else begin
            let rs = c.result in
            let target = Devices.grid r.result in
            cover "a split result" (Grid.cuts target <> [||]);
            let read_whole =
              match c.rule with
              | Route.Elementwise -> false
              | Axes t | Into t ->
                  Array.exists
                    (Array.exists (function
                      | Route.Becomes _ | Scanned _ | Regroups _ -> false
                      | Whole | Reduced _ -> true))
                    t
            in
            cover "a split result beside an axis read whole"
              (read_whole && Grid.cuts target <> [||]);
            (* Cuts of an axis that regroups, at [ps], whose tiles divide both
               extents or not. *)
            let regrouped ps aligned =
              match c.rule with
              | Route.Elementwise -> false
              | Axes t | Into t ->
                  Array.exists Fun.id
                    (Array.mapi
                       (fun i p ->
                         match p with
                         | None -> false
                         | Some p ->
                             Array.exists
                               (fun (a, n) ->
                                 match t.(i).(a) with
                                 | Route.Regroups { extent; _ } ->
                                     aligned
                                     = (c.shapes.(i).(a) mod n = 0
                                       && extent mod n = 0)
                                 | Becomes _ | Scanned _ | Whole | Reduced _ ->
                                     false)
                               (Grid.cuts (Devices.grid p)))
                       ps)
            in
            cover "a cut read along an axis that regroups"
              (regrouped (Array.map Option.some r.operands) true);
            cover "a cut across rows of an axis that regroups"
              (regrouped c.ps false);
            let windows = Grid.devices target in
            Array.iteri
              (fun j k ->
                let w = Devices.window ~by:"t" r.result rs j in
                (* The devices that hold [w]: together they hold the axes the
                   rule reduces. *)
                let group =
                  List.filter
                    (fun j' -> Devices.window ~by:"t" r.result rs j' = w)
                    (List.init (Array.length windows) Fun.id)
                in
                Array.iteri
                  (fun i p ->
                    let window k =
                      Devices.window ~by:"t" p c.shapes.(i)
                        (require_some
                           ~msg:"the operand is on the result's device"
                           (position p k))
                    in
                    let have = window k and need = depends c i w in
                    let reduced = is_reduced c i in
                    let cut_reduced =
                      Array.exists (fun (a, _) -> reduced a)
                        (Grid.cuts (Devices.grid p))
                    in
                    cover "an operand read cut along an axis the rule reduces"
                      cut_reduced;
                    equal bool
                      ~msg:"the window holds what the result reads, but along \
                            the axes the rule reduces"
                      true
                      (contains
                         (along (fun a -> not (reduced a)) have need)
                         need);
                    equal bool
                      ~msg:"the devices of the result's window hold the axes \
                            the rule reduces"
                      true
                      (together
                         (List.map (fun j' -> window windows.(j')) group)
                         need);
                    equal ranges ~msg:"the rule's window" (depends c i w)
                      (Route.window c.rule ~operand:i c.shapes.(i) w))
                  r.operands;
                (* Axes of one factor hold one range on each device, so
                   each device's terms pair up across operands. *)
                let held = Hashtbl.create 3 in
                Array.iteri
                  (fun i p ->
                    let pos = Option.get (position p k) in
                    let wk = Devices.window ~by:"t" p c.shapes.(i) pos in
                    Array.iteri
                      (fun a (rg : R.range) ->
                        match factor_of c i a with
                        | Some f ->
                            let cut = rg.count < c.shapes.(i).(a) in
                            Hashtbl.add held f (i, rg, cut)
                        | None -> ())
                      wk)
                  r.operands;
                List.iter
                  (fun f ->
                    match Hashtbl.find_all held f with
                    | [] -> ()
                    | (i0, first, _) :: rest ->
                        cover "operands of one factor read cut alike"
                          (List.exists
                             (fun (i, _, cut) -> i <> i0 && cut)
                             rest);
                        List.iter
                          (fun (_, rg, _) ->
                            equal ranges ~msg:"one factor's range" [| first |]
                              [| rg |])
                          rest)
                  [ 0; 1; 2 ])
              windows
          end)

(* The devices of [p], in increasing order. *)
let devices_of p =
  List.sort Int.compare (Array.to_list (Grid.devices (Devices.grid p)))

let exact =
  prop ~count:500 "every operand is read on the target's devices alone" case
    (fun c ->
      match route c.rule c.ps c.shapes with
      | exception Invalid_argument _ -> ()
      | None -> ()
      | Some r ->
          let target = devices_of r.result in
          cover "an operand moved off a device the target lacks"
            (Array.exists
               (fun p ->
                 match p with
                 | Some p -> devices_of p <> target
                 | None -> false)
               c.ps);
          Array.iter
            (fun p -> equal (list int) target (devices_of p))
            r.operands)

(* The engine places operands where their route reads them, then routes the
   operation again, as a deferred value's form does over unplaced operands:
   that route must be the first. *)
(* Whether [c]'s rule folds axis [a] of operand [i]: reduces or scans it. *)
let folded c i a =
  match c.rule with
  | Route.Elementwise -> false
  | Axes t | Into t -> (
      match t.(i).(a) with
      | Route.Reduced _ | Scanned _ -> true
      | Becomes _ | Whole | Regroups _ -> false)

(* Whether an axis [c]'s rule folds has no element. *)
let empty_fold c =
  Array.exists Fun.id
    (Array.mapi
       (fun i s ->
         Array.exists Fun.id (Array.mapi (fun a d -> folded c i a && d = 0) s))
       c.shapes)

(* Whether operands read at [operands] are read cut along an axis [c]'s rule
   folds. *)
let cut_fold c (operands : b Devices.placement array) =
  Array.exists Fun.id
    (Array.mapi
       (fun i p ->
         Array.exists (fun (a, _) -> folded c i a) (Grid.cuts (Devices.grid p)))
       operands)

(* Where a route reads an operand cut along an axis the rule reduces or
   scans, every axis it so folds holding elements, the window law does not
   hold and the devices compute by partials: there alone. *)
let computes =
  prop ~count:500
    "a route computes by partials where it reads an operand cut along an \
     axis the rule folds"
    case (fun c ->
      match route c.rule c.ps c.shapes with
      | exception Invalid_argument _ -> ()
      | None -> ()
      | Some r ->
          let empty = empty_fold c and cut = cut_fold c r.operands in
          cover "by partials" (cut && not empty);
          cover "a folded axis of no element" empty;
          equal bool ~msg:"by partials" (cut && not empty)
            (r.compute = Route.Partials);
          (* A reduced axis of no element is read whole. *)
          if empty then
            Array.iteri
              (fun i p ->
                Array.iter
                  (fun (a, _) ->
                    equal bool ~msg:"a reduced axis read cut" false
                      (is_reduced c i a))
                  (Grid.cuts (Devices.grid p)))
              r.operands)

let keeps =
  prop "operands already at the result's placement stay there" case (fun c ->
      assume (c.ps.(0) <> None);
      let p = Option.get c.ps.(0) in
      let ps = Array.make (Array.length c.ps) p in
      let shapes = Array.make (Array.length c.ps) c.shapes.(0) in
      match c.rule with
      | Route.Elementwise ->
          let r = placed Elementwise ps shapes in
          Array.iter (fun q -> equal bool true (q == p)) r.operands;
          equal placement p r.result
      | _ -> ())

(* A constant computed for results at [p]: its route keeps [p]'s devices, keeps
   every cut of [p] its rule allows, and reads each operand where each device's
   window of the results depends on it. It computes by partials where it reads
   an operand cut along an axis the rule scans, as {!computes} states for any
   route: a scan read so would otherwise scan each window from its start. *)
let at_law =
  let wanted =
    let open Gen in
    with_pp
      (fun ppf (c, p) ->
        Format.fprintf ppf "%a wanted at %a" pp_case c Devices.pp_placement p)
      (let* c = case in
       let+ p = placement_of (Array.length c.result) in
       (c, Option.value ~default:(Devices.on s4) p))
  in
  prop ~count:500 "a constant's results are computed where it reads operands"
    wanted (fun (c, p) ->
      let r = Route.at ~by:"Nx.f" c.rule c.shapes p in
      let rs = c.result in
      equal (list int) ~msg:"the devices wanted" (devices_of p)
        (devices_of r.result);
      cover "a cut kept" (Grid.cuts (Devices.grid r.result) <> [||]);
      let partials = cut_fold c r.operands && not (empty_fold c) in
      cover "by partials" partials;
      equal bool ~msg:"by partials" partials (r.compute = Route.Partials);
      Array.iteri
        (fun j k ->
          let w = Devices.window ~by:"t" r.result rs j in
          let wp = Devices.window ~by:"t" p rs (require_some (position p k)) in
          equal bool ~msg:"the window holds the one wanted" true
            (contains w wp);
          Array.iteri
            (fun i q ->
              let pos = require_some (position q k) in
              let have = Devices.window ~by:"t" q c.shapes.(i) pos in
              equal bool ~msg:"the operand's window holds what it reads" true
                (contains have (depends c i w)))
            r.operands)
        (Grid.devices (Devices.grid r.result)))

let cases_ =
  group "routes"
    [
      test "operands of every set alone have no route" (fun () ->
          equal bool true
            (route Elementwise [| None; None |] [| [| 4 |]; [| 4 |] |] = None));
      test "a value of every set beside a host value is read on the host"
        (fun () ->
          let h : b Devices.placement =
            Devices.rebrand (Devices.one Devices.host 0)
          in
          let r =
            Option.get
              (route Elementwise [| None; Some h |] [| [| 4 |]; [| 4 |] |])
          in
          equal bool true (r.result == h));
      test "a value of every set beside a split operand is read split"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          let r =
            Option.get
              (route Elementwise [| Some x; None |] [| [| 8 |]; [| 8 |] |])
          in
          equal placement x r.result;
          equal placement x r.operands.(1));
      test "a reduction of a split axis reads it where it lies" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r =
            placed
              (Axes [| [| Becomes 0; Reduced 1 |] |])
              [| x |]
              [| [| 4; 8 |] |]
          in
          equal placement (Devices.on s4) r.result;
          equal bool true (r.operands.(0) == x));
      test
        "a reduction reads an operand whole on each device at another's cut \
         of a reduced axis"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let rule =
            Route.Axes
              [| [| Becomes 0; Reduced 1 |]; [| Becomes 0; Reduced 1 |] |]
          in
          let r =
            placed rule [| Devices.on s4; x |] [| [| 4; 8 |]; [| 4; 8 |] |]
          in
          equal placement (Devices.on s4) r.result;
          equal placement x r.operands.(0);
          equal bool true (r.operands.(1) == x));
      test "a reduction over operands cut at two grids reads them whole"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let y = Devices.split ~by:"t" ~axis:0 s4 in
          let rule =
            Route.Axes
              [| [| Becomes 0; Reduced 1 |]; [| Becomes 0; Reduced 1 |] |]
          in
          let r = placed rule [| x; y |] [| [| 4; 8 |]; [| 4; 8 |] |] in
          equal placement y r.result;
          equal placement y r.operands.(0);
          equal placement y r.operands.(1));
      test "an operand read whole along an axis it acts along is moved"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r =
            placed (Axes [| [| Becomes 0; Whole |] |]) [| x |] [| [| 4; 8 |] |]
          in
          equal placement (Devices.on s4) r.operands.(0));
      test "a scan of a split axis reads it where it lies and keeps it split"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r =
            placed
              (Axes [| [| Becomes 0; Scanned 1 |] |])
              [| x |]
              [| [| 4; 8 |] |]
          in
          equal placement x r.result;
          equal bool true (r.operands.(0) == x));
      test "a reduction keeps the cut of an axis it does not reduce" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r =
            placed
              (Axes [| [| Reduced 0; Becomes 0 |] |])
              [| x |]
              [| [| 4; 8 |] |]
          in
          equal placement (Devices.split ~by:"t" ~axis:0 s4) r.result;
          equal bool true (r.operands.(0) == x));
      test "a scatter writes where its target lies" (fun () ->
          let into = Devices.split ~by:"t" ~axis:1 s4 and u = Devices.on s4 in
          let rule =
            Route.Into [| [| Whole; Becomes 1 |]; [| Whole; Becomes 1 |] |]
          in
          let r = placed rule [| u; into |] [| [| 4; 8 |]; [| 8; 8 |] |] in
          equal placement into r.result);
      test "an operation reading all whole lies on its operands' devices"
        (fun () ->
          let whole = Route.Axes [| [| Whole |] |] in
          let r = placed whole [| Devices.one s4 2 |] [| [| 4 |] |] in
          equal placement (Devices.one s4 2) r.result;
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          let r = placed whole [| x |] [| [| 8 |] |] in
          equal placement (Devices.on s4) r.result);
      test "operands read whole alone on two devices raise" (fun () ->
          invalid (fun () ->
              placed
                (Axes [| [| Whole |]; [| Whole |] |])
                [| Devices.one s4 0; Devices.one s4 1 |]
                [| [| 4 |]; [| 4 |] |]));
      test "a contraction keeps its batch and free cuts" (fun () ->
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          let r =
            placed matrix_product
              [| x; Devices.on s4 |]
              [| [| 8; 4 |]; [| 4; 4 |] |]
          in
          equal placement x r.result;
          equal bool true (r.operands.(0) == x);
          equal placement (Devices.on s4) r.operands.(1));
      test "a contraction reads operands cut alike along a pair where they lie"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let y = Devices.split ~by:"t" ~axis:0 s4 in
          let shapes = [| [| 8; 5 |]; [| 5; 6 |] |] in
          let r = placed matrix_product [| x; y |] shapes in
          equal placement (Devices.on s4) r.result;
          equal bool true (r.operands.(0) == x);
          equal bool true (r.operands.(1) == y));
      test "a contraction keeps a free cut beside a pair cut alike" (fun () ->
          let x = Devices.mesh ~by:"t" mesh [ (0, [ "a" ]); (1, [ "b" ]) ] in
          let y = Devices.mesh ~by:"t" mesh [ (0, [ "b" ]) ] in
          let shapes = [| [| 8; 5 |]; [| 5; 6 |] |] in
          let r = placed matrix_product [| x; y |] shapes in
          equal placement (Devices.mesh ~by:"t" mesh [ (0, [ "a" ]) ]) r.result;
          equal bool true (r.operands.(0) == x);
          equal bool true (r.operands.(1) == y));
      test "a batched contraction over an operand cut on fewer axes routes \
            alike once placed"
        (fun () ->
          let s8 =
            mint
              (List.init 8 (fun k ->
                   Result.get_ok
                     (Rig.memory_device (Printf.sprintf "q%d" k))))
          in
          let mesh =
            Devices.mesh_v ~by:"t" s8 [ ("a", 2); ("b", 2); ("c", 2) ]
          in
          let rule =
            Route.Axes
              [|
                [| Becomes 0; Becomes 1; Reduced 0 |];
                [| Becomes 0; Reduced 0; Becomes 2 |];
              |]
          in
          let shapes = [| [| 4; 4; 4 |]; [| 4; 4; 3 |] |] in
          let a =
            Devices.mesh ~by:"t" mesh
              [ (0, [ "c" ]); (1, [ "a" ]); (2, [ "b" ]) ]
          and b = Devices.mesh ~by:"t" mesh [ (1, [ "b" ]) ] in
          let r = placed rule [| a; b |] shapes in
          let r' = placed rule r.operands shapes in
          equal placement r.result r'.result;
          equal placement r.operands.(0) r'.operands.(0);
          equal placement r.operands.(1) r'.operands.(1));
      test "a contraction reads a pair cut unlike whole" (fun () ->
          let x = Devices.mesh ~by:"t" mesh [ (1, [ "a" ]) ] in
          let y = Devices.mesh ~by:"t" mesh [ (0, [ "b" ]) ] in
          let shapes = [| [| 8; 5 |]; [| 5; 6 |] |] in
          let r = placed matrix_product [| x; y |] shapes in
          equal placement (Devices.on s4) r.result;
          equal placement (Devices.on s4) r.operands.(0);
          equal placement (Devices.on s4) r.operands.(1));
      test "operands on two sets raise" (fun () ->
          invalid (fun () ->
              placed Elementwise
                [| Devices.on s4; Devices.rebrand (Devices.on other) |]
                [| [| 4 |]; [| 4 |] |]));
      test "operands alone on two devices raise" (fun () ->
          invalid (fun () ->
              placed Elementwise
                [| Devices.one s4 0; Devices.one s4 1 |]
                [| [| 4 |]; [| 4 |] |]));
      test "a split of an axis the operand lacks raises" (fun () ->
          invalid (fun () ->
              placed Elementwise
                [| Devices.split ~by:"t" ~axis:1 s4 |]
                [| [| 6 |] |]));
    ]

(* Movements: the operand index each result index reads, as Move states it. *)
let source mv s idx =
  let r = Array.length s in
  match mv with
  | R.Reshape s' ->
      let s' = Iarray.to_array s' in
      let k = ref 0 in
      Array.iteri (fun i j -> k := (!k * s'.(i)) + j) idx;
      let src = Array.make r 0 in
      for i = r - 1 downto 0 do
        src.(i) <- !k mod s.(i);
        k := !k / s.(i)
      done;
      src
  | Broadcast s' ->
      let s' = Iarray.to_array s' in
      let off = Array.length s' - r in
      Array.init r (fun i -> if s.(i) = 1 then 0 else idx.(off + i))
  | Permute p ->
      let p = Iarray.to_array p in
      let src = Array.make r 0 in
      Array.iteri (fun i a -> src.(a) <- idx.(i)) p;
      src
  | Slice rs ->
      let rs = Iarray.to_array rs in
      Array.init r (fun i -> rs.(i).start + (idx.(i) * rs.(i).step))
  | Window ws ->
      let ws = Iarray.to_array ws in
      let src = Array.sub idx 0 r in
      Array.iteri
        (fun j (w : R.window) ->
          src.(w.axis) <- (idx.(w.axis) * w.step) + (idx.(r + j) * w.dilation))
        ws;
      src

let move_of s =
  let open Gen in
  let r = Array.length s in
  one_of
    ([
       map
         (fun p -> R.Permute (Iarray.of_list p))
         (permutation (List.init r Fun.id));
       constant (R.Broadcast (Iarray.of_array (Array.append [| 3 |] s)));
       constant
         (R.Reshape
            (Iarray.of_array
               (Array.append [| 2; s.(0) / 2 |] (Array.sub s 1 (r - 1)))));
       map
         (fun choice ->
           R.Slice
             (Iarray.of_array
                (Array.mapi
                   (fun i d ->
                     match (choice + i) mod 3 with
                     | 0 -> { R.start = 0; count = d; step = 1 }
                     | 1 -> { R.start = 0; count = d / 2; step = 1 }
                     | _ -> { R.start = d - 1; count = d; step = -1 })
                   s)))
         (int_range 0 2);
       map
         (fun a ->
           R.Window [| { axis = a; size = 2; step = 2; dilation = 1 } |])
         (int_range 0 (r - 1));
     ]
    @
    if r >= 2 then
      [
        constant
          (R.Reshape
             (Iarray.of_array
                (Array.append [| s.(0) * s.(1) |] (Array.sub s 2 (r - 2)))));
      ]
    else [])

let move_case =
  let open Gen in
  with_pp
    (fun ppf (s, p, _) ->
      Format.fprintf ppf "%a at %a" pp_ints s Devices.pp_placement p)
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 4; 8 ]) in
     let* p =
       map
         (function Some p -> p | None -> Devices.on s4)
         (placement_of rank)
     in
     let+ mv = move_of s in
     (s, p, mv))

let move_law =
  prop ~count:300 "each moved result window reads the operand on its own device"
    move_case (fun (s, p, mv) ->
      let x =
        Value.Array
          {
            at = Devices.one s4 0;
            a =
              Nx_array.create (m 0) Nx_array.Dtype.Float32 (Iarray.of_array s);
            dead = Value.Live;
          }
      in
      let rule = Prim.rule (Value.Move (mv, x)) in
      let r = placed rule [| p |] [| s |] in
      let s' = Iarray.to_array (R.shape mv (Iarray.of_array s)) in
      cover "a split operand" (Grid.cuts (Devices.grid p) <> [||]);
      cover "a split result" (Grid.cuts (Devices.grid r.result) <> [||]);
      Array.iteri
          (fun j k ->
            let w = Devices.window ~by:"t" r.result s' j in
            let pos = require_some (position r.operands.(0) k) in
            let have = Devices.window ~by:"t" r.operands.(0) s pos in
            let rec walk a idx =
              if a = Array.length s' then
                let src = source mv s (Array.of_list (List.rev idx)) in
                equal bool ~msg:"the source index lies in the device's window"
                  true
                  (Array.for_all2
                     (fun (h : R.range) i ->
                       i >= h.start && i < h.start + h.count)
                     have src)
              else
                for i = w.(a).start to w.(a).start + w.(a).count - 1 do
                  walk (a + 1) (i :: idx)
                done
            in
            walk 0 [])
          (Grid.devices (Devices.grid r.result)))

let () =
  exit
    (run "nx route"
       [
         group "laws"
           [ law; exact; computes; keeps; at_law; move_law ];
         cases_;
       ])
