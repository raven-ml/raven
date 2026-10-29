(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Routes, through the engine's private Route, copied here, against the part of
   each operand a result window depends on. *)

open Windtrap
open Nx_support
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

(* [f ()] raises the route's refusal [msg], whole. *)
let refused msg f = raises (Invalid_argument msg) f

(* A matrix product's rule: [a]'s columns and [b]'s rows are its contracting
   pair. *)
let matrix_product =
  Route.Axes [| [| Becomes 0; Reduced 0 |]; [| Reduced 0; Becomes 1 |] |]

let pp_axis ppf : Route.axis -> unit = function
  | Becomes a -> Format.pp_print_int ppf a
  | Whole -> Format.pp_print_string ppf "whole"
  | Reduced f -> Format.fprintf ppf "reduced %d" f
  | Blocks { factor; extent } ->
      Format.fprintf ppf "blocks of %d of %d" factor extent
  | Scanned a -> Format.fprintf ppf "scanned %d" a
  | Indexed a -> Format.fprintf ppf "indexed %d" a
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

(* An operation: its rule, its operands' placements and shapes, and its results'
   shape. *)
type case = {
  rule : Route.rule;
  ps : b Devices.placement array;
  shapes : int array array;
  result : int array;
}

let pp_case ppf c =
  Format.fprintf ppf "%a, results %a, over" pp_rule c.rule pp_ints c.result;
  Array.iteri
    (fun i p ->
      Format.fprintf ppf " %a at %a," pp_ints c.shapes.(i) Devices.pp_placement
        p)
    c.ps

(* A placement of a value of [rank]; of a scalar, uncut. *)
let placement_of rank =
  let uncut =
    [
      Gen.map (fun k -> Devices.one s4 k) (Gen.int_range 0 3);
      Gen.constant (Devices.on s4);
    ]
  in
  if rank = 0 then Gen.one_of uncut
  else
    let axis = Gen.int_range 0 (rank - 1) in
    Gen.one_of
      (uncut
      @ [
          Gen.map (fun axis -> Devices.split ~by:"t" ~axis s4) axis;
          Gen.map
            (fun (x, y) ->
              if x = y then Devices.mesh ~by:"t" mesh [ (x, [ "a"; "b" ]) ]
              else Devices.mesh ~by:"t" mesh [ (x, [ "a" ]); (y, [ "b" ]) ])
            (Gen.pair axis axis);
        ])

let extent = Gen.of_list [ 0; 1; 3; 4; 5; 8 ]

(* An operand of results of shape [out]: each axis becomes a distinct axis of
   the results, of its extent, plainly or scanned, or regroups into one, both
   extents above one and its own even, or none, whole of any extent or
   reduced, plainly or in blocks of 2 or 4 where more than one block fills
   the factor. Its reduced axes, in order, have the factors 0, 1 and 2, of
   [factors]' extents. *)
let operand_of factors out =
  let open Gen in
  let rank = Array.length out in
  let* r = int_range 1 3 in
  let+ order = permutation (List.init rank Fun.id @ List.init r (fun _ -> -1))
  and+ along = array ~size:(constant r) extent
  and+ reduced = array ~size:(constant r) bool
  and+ scanned = array ~size:(constant r) bool
  and+ indexed = array ~size:(constant r) bool
  and+ regroups = array ~size:(constant r) bool
  and+ rows = array ~size:(constant r) (of_list [ 2; 4; 6; 8; 12; 16 ])
  and+ blocks = array ~size:(constant r) (of_list [ 0; 2; 4 ]) in
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
  let in_blocks a =
    let e = factors.(factor a) and b = blocks.(a) in
    b > 0 && e mod b = 0 && e / b > 1
  in
  let axis a t : Route.axis =
    if regroup a t then Regroups { axis = t; extent = out.(t) }
    else if t >= 0 then
      if scanned.(a) then Scanned t
      else if indexed.(a) then Indexed t
      else Becomes t
    else if reduced.(a) && in_blocks a then
      Blocks { factor = factor a; extent = factors.(factor a) }
    else if reduced.(a) then Reduced (factor a)
    else Whole
  in
  let extent a t =
    if regroup a t then rows.(a)
    else if t >= 0 then out.(t)
    else if reduced.(a) && in_blocks a then factors.(factor a) / blocks.(a)
    else if reduced.(a) then factors.(factor a)
    else along.(a)
  in
  (Array.mapi axis axes, Array.mapi extent axes)

let case =
  let open Gen in
  let* rank = int_range 1 3 in
  let* out = array ~size:(constant rank) extent in
  let* n = int_range 1 3 in
  (* An elementwise rule a third of the time. *)
  let* elementwise = frequency [ (1, constant true); (2, constant false) ] in
  let* factors = array ~size:(constant 3) extent in
  let* operands =
    if elementwise then constant (Array.make n ([||], out))
    else array ~size:(constant n) (operand_of factors out)
  in
  let rule =
    if elementwise then Route.Elementwise else Axes (Array.map fst operands)
  in
  let shapes = Array.map snd operands in
  (* Often each operand's axis of factor 0 split over the four devices: the
     operands' terms pair up there. *)
  let* paired = bool in
  let factor0 i =
    match rule with
    | Route.Elementwise -> None
    | Axes t ->
        Array.find_index
          (function
            | Route.Reduced 0 | Blocks { factor = 0; _ } -> true | _ -> false)
          t.(i)
  in
  (* Often an operand's axis that regroups split over the four devices. *)
  let* split_regroup = bool in
  let regroup i =
    match rule with
    | Route.Elementwise -> None
    | Axes t ->
        Array.find_index
          (function Route.Regroups _ -> true | _ -> false)
          t.(i)
  in
  let rec placements i =
    if i = n then constant []
    else
      let* p =
        match (factor0 i, regroup i) with
        | Some axis, _ when paired -> constant (Devices.split ~by:"t" ~axis s4)
        | _, Some axis when split_regroup ->
            constant (Devices.split ~by:"t" ~axis s4)
        | _ -> placement_of (Array.length shapes.(i))
      in
      let+ rest = placements (i + 1) in
      p :: rest
  in
  let+ ps = placements 0 in
  { rule; ps = Array.of_list ps; shapes; result = out }

(* A matrix product beside its second operand's scale, a value per block of
   [k / d] along its pair: [a] [[m; k]], [b] [[k; n]] and the scale
   [[n; d]]. Half the time each is split along its pair over the four
   devices; else each lies anywhere. *)
let scaled =
  let open Gen in
  let* m = of_list [ 1; 3; 4 ]
  and+ n = of_list [ 1; 2; 4 ]
  and+ k = of_list [ 8; 16; 32 ]
  and+ per = of_list [ 2; 4 ]
  and+ paired = bool in
  let rule =
    Route.Axes
      [|
        [| Becomes 0; Reduced 0 |];
        [| Reduced 0; Becomes 1 |];
        [| Becomes 1; Blocks { factor = 0; extent = k } |];
      |]
  in
  let shapes = [| [| m; k |]; [| k; n |]; [| n; k / per |] |] in
  let split axis = Devices.split ~by:"t" ~axis s4 in
  let+ ps =
    if paired then constant [| split 1; split 0; split 1 |]
    else
      let+ pa = placement_of 2
      and+ pb = placement_of 2
      and+ pc = placement_of 2 in
      [| pa; pb; pc |]
  in
  { rule; ps; shapes; result = [| m; n |] }

(* A product of rows by groups: [a] [[m; k]], each row by the matrix
   [[n; k]] of [b] [[g; n; k]] its group names, the [g + 1] offsets the
   groups' fences. Half the time [b] is split along its groups over the four
   devices beside [a] and the offsets whole on each; else each lies
   anywhere. *)
let segmented =
  let open Gen in
  let* m = of_list [ 1; 3; 4 ]
  and+ n = of_list [ 1; 2; 4 ]
  and+ k = of_list [ 2; 8 ]
  and+ g = of_list [ 0; 1; 3; 4; 8 ]
  and+ grouped = bool in
  let rule =
    Route.Axes
      [|
        [| Whole; Whole |]; [| Reduced 1; Becomes 1; Whole |]; [| Whole |];
      |]
  in
  let shapes = [| [| m; k |]; [| g; n; k |]; [| g + 1 |] |] in
  let+ ps =
    if grouped then
      let on = Devices.on s4 in
      constant [| on; Devices.split ~by:"t" ~axis:0 s4; on |]
    else
      let+ pa = placement_of 2
      and+ pb = placement_of 3
      and+ po = placement_of 1 in
      [| pa; pb; po |]
  in
  { rule; ps; shapes; result = [| m; n |] }

let case =
  Gen.with_pp pp_case
    (Gen.frequency [ (3, case); (1, scaled); (1, segmented) ])

(* The part of operand [i] that result window [w] depends on. Along an axis of
   [d] that regroups into one of [e], the rows of [d / e] the window's range
   touches. *)
let depends c i (w : R.range array) =
  match c.rule with
  | Route.Elementwise -> w
  | Axes t ->
      Array.mapi
        (fun a d ->
          match t.(i).(a) with
          | Route.Becomes a' | Scanned a' | Indexed a' -> w.(a')
          | Regroups { axis; extent = e } ->
              let r = w.(axis) in
              let lo = r.start * d / e in
              let hi = (((r.start + r.count) * d) + e - 1) / e in
              { R.start = lo; count = hi - lo; step = 1 }
          | Whole | Reduced _ | Blocks _ ->
              { R.start = 0; count = d; step = 1 })
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
  | Axes t -> (
      match t.(i).(a) with
      | Route.Reduced f -> Some f
      | Becomes _ | Whole | Scanned _ | Indexed _ | Regroups _ | Blocks _ ->
          None)

(* The factor of operand [i]'s axis [a] and its extent, if it holds the
   factor's blocks. *)
let blocks_of c i a =
  match c.rule with
  | Route.Elementwise -> None
  | Axes t -> (
      match t.(i).(a) with
      | Route.Blocks { factor; extent } -> Some (factor, extent)
      | Becomes _ | Whole | Scanned _ | Indexed _ | Regroups _ | Reduced _ ->
          None)

let is_reduced c i a = factor_of c i a <> None || blocks_of c i a <> None

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

(* The window [p] gives device [k] of a value of shape [s], if [k] is one of
   [p]'s devices. *)
let held p s k = Option.map (Devices.window ~by:"t" p s) (position p k)

let is_empty (w : R.range array) =
  Array.exists (fun (r : R.range) -> r.count = 0) w

(* Whether a value of shape [s] at [p] holds, on each device of [q], the
   window [q] gives that device. An empty window reads nothing. *)
let holds p s q =
  Array.for_all
    (fun k ->
      let w = Option.get (held q s k) in
      is_empty w
      || match held p s k with Some h -> contains h w | None -> false)
    (Grid.devices (Devices.grid q))

(* The cuts of operand [i] at [p] that land on a result axis, each with that
   axis: one it becomes or scans, or regroups into in tiles that divide both
   extents. *)
let landing c i p =
  List.filter_map
    (fun (a, t) ->
      match c.rule with
      | Route.Elementwise -> Some (a, a)
      | Axes r -> (
          match r.(i).(a) with
          | Route.Becomes a' | Scanned a' | Indexed a' -> Some (a, a')
          | Regroups { axis; extent }
            when c.shapes.(i).(a) mod t = 0 && extent mod t = 0 ->
              Some (a, axis)
          | Regroups _ | Whole | Reduced _ | Blocks _ -> None))
    (Array.to_list (Grid.cuts (Devices.grid p)))


(* The devices of [p], in increasing order. *)
let devices_of p =
  List.sort Int.compare (Array.to_list (Grid.devices (Devices.grid p)))

let law =
  prop ~count:2000
    "each operand is read where it lies, on the result's devices, and each \
     result window on its own device"
    case (fun c ->
      match route c.rule c.ps c.shapes with
      | exception Invalid_argument _ -> cover "a refusal" true
      | r ->
          let rs = c.result in
          let target = Devices.grid r.result in
          cover "a split result" (Grid.cuts target <> [||]);
          let read_whole =
            match c.rule with
            | Route.Elementwise -> false
            | Axes t ->
                Array.exists
                  (Array.exists (function
                    | Route.Becomes _ | Scanned _ | Indexed _ | Regroups _ ->
                        false
                    | Whole | Reduced _ | Blocks _ -> true))
                  t
          in
          cover "a split result beside an axis read whole"
            (read_whole && Grid.cuts target <> [||]);
          (* Cuts of an axis that regroups, at [ps], whose tiles divide both
             extents or not. *)
          let regrouped ps aligned =
            match c.rule with
            | Route.Elementwise -> false
            | Axes t ->
                Array.exists Fun.id
                  (Array.mapi
                     (fun i p ->
                       Array.exists
                         (fun (a, n) ->
                           match t.(i).(a) with
                           | Route.Regroups { extent; _ } ->
                               aligned
                               = (c.shapes.(i).(a) mod n = 0
                                 && extent mod n = 0)
                           | Becomes _ | Scanned _ | Indexed _ | Whole
                           | Reduced _ | Blocks _ ->
                               false)
                         (Grid.cuts (Devices.grid p)))
                     ps)
          in
          cover "a cut read along an axis that regroups"
            (regrouped r.operands true);
          cover "a cut across rows of an axis that regroups"
            (regrouped c.ps false);
          (* Each operand is read where it lies: at its own placement, or
             at one whose window on each device its own holds there, over
             the result's devices alone. *)
          cover "an operand moved off a device the result lacks"
            (Array.exists (fun p -> devices_of p <> devices_of r.result) c.ps);
          Array.iteri
            (fun i q ->
              let p = c.ps.(i) in
              cover "an operand read by views of its own windows"
                (not (Devices.equal p q));
              equal bool ~msg:"the operand holds each window it is read at"
                true
                (holds p c.shapes.(i) q);
              equal (list int)
                ~msg:"the operand is read on the result's devices"
                (devices_of r.result) (devices_of q))
            r.operands;
          (* The forward union: each cut of the results is a cut of an
             operand that lands there, tile for tile, and the first
             operand's cuts all stay where it has elements: without, it
             holds every window it is read at. *)
          let matches i p (a, _) =
            Array.for_all
              (fun k ->
                match held p c.shapes.(i) k with
                | None -> false
                | Some h ->
                    let w = Option.get (held r.result rs k) in
                    (depends c i w).(a) = h.(a))
              (Grid.devices target)
          in
          Array.iter
            (fun (a', _) ->
              let from i p =
                List.exists
                  (fun ((_, b) as l) -> b = a' && matches i p l)
                  (landing c i p)
              in
              let msg =
                Printf.sprintf "result axis %d is cut as an operand cuts it" a'
              in
              equal bool ~msg true
                (Array.exists Fun.id (Array.mapi from c.ps)))
            (Grid.cuts target);
          let p = c.ps.(0) in
          if not (Array.mem 0 c.shapes.(0)) then
            List.iter
              (fun l ->
                cover "a cut of the first operand" true;
                equal bool ~msg:"its cut stays" true (matches 0 p l))
              (landing c 0 p);
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
                 each device's terms pair up across operands, and an axis
                 of its blocks holds the blocks that range spans. *)
              let held = Hashtbl.create 3 and spans = Hashtbl.create 3 in
              Array.iteri
                (fun i p ->
                  let pos = Option.get (position p k) in
                  let wk = Devices.window ~by:"t" p c.shapes.(i) pos in
                  Array.iteri
                    (fun a (rg : R.range) ->
                      let cut = rg.count < c.shapes.(i).(a) in
                      match (factor_of c i a, blocks_of c i a) with
                      | Some f, _ -> Hashtbl.add held f (i, rg, cut)
                      | None, Some (f, e) ->
                          let per = e / c.shapes.(i).(a) in
                          Hashtbl.add spans f (rg, per, e, cut)
                      | None, None -> ())
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
                        rest;
                      List.iter
                        (fun ((rg : R.range), per, e, cut) ->
                          cover "a factor's blocks read cut alike" cut;
                          let lo = rg.start * per
                          and hi = min e ((rg.start + rg.count) * per) in
                          let spanned =
                            first.count = 0
                            || (lo <= first.start
                               && first.start + first.count <= hi)
                          in
                          equal bool
                            ~msg:"the blocks a factor's range spans" true
                            spanned)
                        (Hashtbl.find_all spans f))
                [ 0; 1; 2 ])
            windows)

(* Refusals, as the route writes them: "<by>: operand <i> lies at <p> and is
   read at <q>; Nx.place (<call>) <name>". *)

(* [s] around the first [sep], or the last under [~last]: the text before it
   and the text after. *)
let cut_at ?(last = false) sep s =
  let n = String.length sep and m = String.length s in
  let rec find i =
    if i < 0 || i + n > m then None
    else if String.sub s i n = sep then Some i
    else find (if last then i - 1 else i + 1)
  in
  Option.map
    (fun i -> (String.sub s 0 i, String.sub s (i + n) (m - i - n)))
    (find (if last then m - n else 0))

type refusal = {
  operand : int;
  lies : string;
  read : string;
  call : string;
  named : string;
}

let refusal msg =
  let ( let* ) = Option.bind in
  let* by, rest = cut_at ": operand " msg in
  let* () = if by = "Nx.f" then Some () else None in
  let* i, rest = cut_at " lies at " rest in
  let* lies, rest = cut_at " and is read at " rest in
  let* read, rest = cut_at "; Nx.place (" rest in
  let* call, named = cut_at ~last:true ") " rest in
  let* operand = int_of_string_opt i in
  Some { operand; lies; read; call; named }

let name = function 0 -> "x" | 1 -> "y" | 2 -> "z" | i -> "x" ^ string_of_int i

(* Each placement over [s4] of a value of rank [rank] that a route reads at,
   with the expression a refusal writes for it: each device alone, the whole
   on every device, each split, and each other cut by the mesh's axes [a]
   and [b], which the expression names ["g0"] and ["g1"]. *)
let readable rank =
  let axes = List.init rank Fun.id in
  let quoted gs = String.concat "; " (List.map (Printf.sprintf "%S") gs) in
  let call cuts =
    Printf.sprintf
      "Placement.mesh (Mesh.v s [ (\"g0\", 2); (\"g1\", 2) ]) [ %s ]"
      (String.concat "; "
         (List.map
            (fun (a, gs) -> Printf.sprintf "(%d, [ %s ])" a (quoted gs))
            cuts))
  in
  let g = function "a" -> "g0" | _ -> "g1" in
  let named cuts = call (List.map (fun (a, ns) -> (a, List.map g ns)) cuts) in
  let ones =
    List.init 4 (fun k ->
        (Devices.one s4 k, Printf.sprintf "Placement.device s m%d" k))
  in
  let splits =
    List.map
      (fun a ->
        ( Devices.split ~by:"t" ~axis:a s4,
          Printf.sprintf "Placement.split ~axis:%d s" a ))
      axes
  in
  (* The mesh's [a] cutting axis [x] and [b] axis [y], [-1] for none; [a]
     then [b] over one axis is a split. *)
  let owners =
    List.concat_map
      (fun x -> List.map (fun y -> (x, y)) (-1 :: axes))
      (-1 :: axes)
  in
  let meshes =
    List.concat_map
      (fun (x, y) ->
        if x = y && x >= 0 then
          let cuts = [ (x, [ "b"; "a" ]) ] in
          [ (Devices.mesh ~by:"t" mesh cuts, named cuts) ]
        else if x < 0 && y < 0 then []
        else
          let cuts =
            List.sort compare
              (List.filter
                 (fun (a, _) -> a >= 0)
                 [ (x, [ "a" ]); (y, [ "b" ]) ])
          in
          [ (Devices.mesh ~by:"t" mesh cuts, named cuts) ])
      owners
  in
  ((Devices.on s4, "Placement.on s") :: ones) @ splits @ meshes

let refusals =
  prop ~count:2000
    "a refusal names an operand that lacks a window it is read at, and the \
     Nx.place after which the call routes"
    case (fun c ->
      (* The operand a refusal names and where it is read, its text checked
         against them. *)
      let parse ps msg =
        let r = require_some ~msg (refusal msg) in
        let i = r.operand in
        let p = ps.(i) in
        equal string ~msg:"where it lies"
          (Format.asprintf "%a" Devices.pp_placement p)
          r.lies;
        equal string ~msg:"its name" (name i) r.named;
        let q, call =
          require_some ~msg:("a placement printed " ^ r.read)
            (List.find_opt
               (fun (q, _) ->
                 Format.asprintf "%a" Devices.pp_placement q = r.read)
               (readable (Array.length c.shapes.(i))))
        in
        equal string ~msg:"the call" call r.call;
        equal bool ~msg:"it lacks a window it is read at" false
          (holds p c.shapes.(i) q);
        (i, q, call)
      in
      match route c.rule c.ps c.shapes with
      | _ -> ()
      | exception Invalid_argument msg ->
          let i, q, call = parse c.ps msg in
          cover "read whole on every device" (Devices.equal q (Devices.on s4));
          cover "read on one device" (Devices.device q <> None);
          cover "read split"
            (String.starts_with ~prefix:"Placement.split" call);
          cover "read at a mesh's cut"
            (String.starts_with ~prefix:"Placement.mesh" call);
          (* The remedies, written one after another, let the call route. *)
          let ps = Array.copy c.ps in
          ps.(i) <- q;
          let rec settle n =
            match route c.rule ps c.shapes with
            | _ -> ()
            | exception Invalid_argument msg ->
                if n = 0 then fail ("the remedies do not settle: " ^ msg)
                else begin
                  let i, q, _ = parse ps msg in
                  ps.(i) <- q;
                  settle (n - 1)
                end
          in
          settle (2 * Array.length ps))

(* Whether [c]'s rule folds axis [a] of operand [i]: reduces, scans or
   indexes it. *)
let folded c i a =
  match c.rule with
  | Route.Elementwise -> false
  | Axes t -> (
      match t.(i).(a) with
      | Route.Reduced _ | Blocks _ | Scanned _ | Indexed _ -> true
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
      | r ->
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
      let p = c.ps.(0) in
      let ps = Array.make (Array.length c.ps) p in
      let shapes = Array.make (Array.length c.ps) c.shapes.(0) in
      match c.rule with
      | Route.Elementwise ->
          let r = route Elementwise ps shapes in
          Array.iter (fun q -> equal bool true (q == p)) r.operands;
          equal placement p r.result
      | _ -> ())

let cases_ =
  group "routes"
    [
      test "a reduction of a split axis reads it where it lies" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r =
            route
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
            route rule [| Devices.on s4; x |] [| [| 4; 8 |]; [| 4; 8 |] |]
          in
          equal placement (Devices.on s4) r.result;
          equal placement x r.operands.(0);
          equal bool true (r.operands.(1) == x));
      test
        "a reduction over operands cut at two grids refuses the second, read \
         cut alike"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let y = Devices.split ~by:"t" ~axis:0 s4 in
          let rule =
            Route.Axes
              [| [| Becomes 0; Reduced 1 |]; [| Becomes 0; Reduced 1 |] |]
          in
          refused
            "Nx.f: operand 1 lies at split ~axis:0 [m0; m1; m2; m3] and is \
             read at split ~axis:1 [m0; m1; m2; m3]; Nx.place \
             (Placement.split ~axis:1 s) y"
            (fun () -> route rule [| x; y |] [| [| 4; 8 |]; [| 4; 8 |] |]));
      test "an operand cut along an axis it acts along is refused, read whole"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          refused
            "Nx.f: operand 0 lies at split ~axis:1 [m0; m1; m2; m3] and is \
             read at on [m0; m1; m2; m3]; Nx.place (Placement.on s) x"
            (fun () ->
              route
                (Axes [| [| Becomes 0; Whole |] |])
                [| x |]
                [| [| 4; 8 |] |]));
      test "a scan of a split axis reads it where it lies and keeps it split"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r =
            route
              (Axes [| [| Becomes 0; Scanned 1 |] |])
              [| x |]
              [| [| 4; 8 |] |]
          in
          equal placement x r.result;
          equal bool true (r.operands.(0) == x));
      test "a reduction keeps the cut of an axis it does not reduce" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r =
            route
              (Axes [| [| Reduced 0; Becomes 0 |] |])
              [| x |]
              [| [| 4; 8 |] |]
          in
          equal placement (Devices.split ~by:"t" ~axis:0 s4) r.result;
          equal bool true (r.operands.(0) == x));
      test "a scatter's target cut beside updates whole on each device keeps \
            its cut"
        (fun () ->
          let into = Devices.split ~by:"t" ~axis:1 s4 and u = Devices.on s4 in
          let rule =
            Route.Axes [| [| Whole; Becomes 1 |]; [| Whole; Becomes 1 |] |]
          in
          let r = route rule [| u; into |] [| [| 4; 8 |]; [| 8; 8 |] |] in
          equal placement into r.result);
      test "an operation reading all whole lies on its operand's device"
        (fun () ->
          let whole = Route.Axes [| [| Whole |] |] in
          let r = route whole [| Devices.one s4 2 |] [| [| 4 |] |] in
          equal placement (Devices.one s4 2) r.result);
      test "an operation reading a split axis whole refuses it" (fun () ->
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          refused
            "Nx.f: operand 0 lies at split ~axis:0 [m0; m1; m2; m3] and is \
             read at on [m0; m1; m2; m3]; Nx.place (Placement.on s) x"
            (fun () -> route (Axes [| [| Whole |] |]) [| x |] [| [| 8 |] |]));
      test "operands read whole alone on two devices refuse the second"
        (fun () ->
          refused
            "Nx.f: operand 1 lies at m1 and is read at m0; Nx.place \
             (Placement.device s m0) y"
            (fun () ->
              route
                (Axes [| [| Whole |]; [| Whole |] |])
                [| Devices.one s4 0; Devices.one s4 1 |]
                [| [| 4 |]; [| 4 |] |]));
      test "a contraction keeps its batch and free cuts" (fun () ->
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          let r =
            route matrix_product
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
          let r = route matrix_product [| x; y |] shapes in
          equal placement (Devices.on s4) r.result;
          equal bool true (r.operands.(0) == x);
          equal bool true (r.operands.(1) == y));
      test "a contraction keeps a free cut beside a pair cut alike" (fun () ->
          let x = Devices.mesh ~by:"t" mesh [ (0, [ "a" ]); (1, [ "b" ]) ] in
          let y = Devices.mesh ~by:"t" mesh [ (0, [ "b" ]) ] in
          let shapes = [| [| 8; 5 |]; [| 5; 6 |] |] in
          let r = route matrix_product [| x; y |] shapes in
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
          let r = route rule [| a; b |] shapes in
          let r' = route rule r.operands shapes in
          equal placement r.result r'.result;
          equal placement r.operands.(0) r'.operands.(0);
          equal placement r.operands.(1) r'.operands.(1));
      test "a contraction over a pair cut unlike refuses the second" (fun () ->
          let x = Devices.mesh ~by:"t" mesh [ (1, [ "a" ]) ] in
          let y = Devices.mesh ~by:"t" mesh [ (0, [ "b" ]) ] in
          let shapes = [| [| 8; 5 |]; [| 5; 6 |] |] in
          refused
            "Nx.f: operand 1 lies at mesh 2x2 [m0; m1; m2; m3] ~axis:0/1 and \
             is read at mesh 2x2 [m0; m1; m2; m3] ~axis:0/0; Nx.place \
             (Placement.mesh (Mesh.v s [ (\"g0\", 2); (\"g1\", 2) ]) [ (0, \
             [ \"g0\" ]) ]) y"
            (fun () -> route matrix_product [| x; y |] shapes));
      test
        "a contraction refuses a free axis cut over the devices that cut its \
         pair"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let shapes = [| [| 8; 5 |]; [| 5; 6 |] |] in
          refused
            "Nx.f: operand 1 lies at split ~axis:1 [m0; m1; m2; m3] and is \
             read at split ~axis:0 [m0; m1; m2; m3]; Nx.place \
             (Placement.split ~axis:0 s) y"
            (fun () -> route matrix_product [| x; x |] shapes));
      (* Rows of [a] [[6; 4]] by the groups of [b] [[5; 3; 4]], the offsets
         [[6]] their fences, as a segmented contraction's rule reads them. *)
      test
        "a product by groups cut along them computes by partials, its fences \
         read whole on each device"
        (fun () ->
          let rule =
            Route.Axes
              [|
                [| Whole; Whole |];
                [| Reduced 1; Becomes 1; Whole |];
                [| Whole |];
              |]
          in
          let shapes = [| [| 6; 4 |]; [| 5; 3; 4 |]; [| 6 |] |] in
          let on = Devices.on s4 and b = Devices.split ~by:"t" ~axis:0 s4 in
          let r = route rule [| on; b; on |] shapes in
          equal (array placement) [| on; b; on |] r.operands;
          equal placement on r.result;
          equal bool ~msg:"by partials" true (r.compute = Route.Partials);
          refused
            "Nx.f: operand 2 lies at split ~axis:0 [m0; m1; m2; m3] and is \
             read at on [m0; m1; m2; m3]; Nx.place (Placement.on s) z"
            (fun () -> route rule [| on; b; b |] shapes));
      test "operands cut by two mesh axes join, each read by views" (fun () ->
          let x = Devices.mesh ~by:"t" mesh [ (0, [ "a" ]) ] in
          let y = Devices.mesh ~by:"t" mesh [ (1, [ "b" ]) ] in
          let both = Devices.mesh ~by:"t" mesh [ (0, [ "a" ]); (1, [ "b" ]) ] in
          let r = route Elementwise [| x; y |] [| [| 4; 4 |]; [| 4; 4 |] |] in
          equal placement both r.result;
          equal placement both r.operands.(0);
          equal placement both r.operands.(1));
      test "a coarser cut of an axis the results cut finer is read by views"
        (fun () ->
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          let y = Devices.mesh ~by:"t" mesh [ (0, [ "a" ]) ] in
          let r = route Elementwise [| x; y |] [| [| 8 |]; [| 8 |] |] in
          equal bool true (r.result == x);
          equal placement x r.operands.(1));
      test "a device alone beside the whole on every device takes the results"
        (fun () ->
          let one = Devices.one s4 2 in
          let r =
            route Elementwise [| Devices.on s4; one |] [| [| 4 |]; [| 4 |] |]
          in
          equal bool true (r.result == one);
          equal bool true (r.operands.(0) == one));
      test "a device alone beside a split operand is refused" (fun () ->
          refused
            "Nx.f: operand 1 lies at m2 and is read at split ~axis:0 [m0; m1; \
             m2; m3]; Nx.place (Placement.split ~axis:0 s) y"
            (fun () ->
              route Elementwise
                [| Devices.split ~by:"t" ~axis:0 s4; Devices.one s4 2 |]
                [| [| 4 |]; [| 4 |] |]));
      test "operands on two sets refuse the second, read on the first's"
        (fun () ->
          refused
            "Nx.f: operand 1 lies at on [m0; m1] and is read at on [m0; m1; \
             m2; m3]; Nx.place (Placement.on s) y"
            (fun () ->
              route Elementwise
                [| Devices.on s4; Devices.on other |]
                [| [| 4 |]; [| 4 |] |]));
      test "operands alone on two devices refuse the second" (fun () ->
          refused
            "Nx.f: operand 1 lies at m1 and is read at m0; Nx.place \
             (Placement.device s m0) y"
            (fun () ->
              route Elementwise
                [| Devices.one s4 0; Devices.one s4 1 |]
                [| [| 4 |]; [| 4 |] |]));
      test "a split of an axis the operand lacks raises" (fun () ->
          invalid (fun () ->
              route Elementwise
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
     let* p = placement_of rank in
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
      cover "a split operand" (Grid.cuts (Devices.grid p) <> [||]);
      match route rule [| p |] [| s |] with
      | exception Invalid_argument _ ->
          (* A move refuses an operand cut along an axis it reads whole: one
             it does not keep in order, or regroups across rows. *)
          let whole (a, t) =
            match rule with
            | Route.Axes r -> (
                match r.(0).(a) with
                | Route.Whole -> true
                | Regroups { extent; _ } ->
                    s.(a) mod t <> 0 || extent mod t <> 0
                | Becomes _ | Scanned _ | Indexed _ | Reduced _ | Blocks _ ->
                    false)
            | Elementwise -> false
          in
          cover "a move reading a cut axis whole" true;
          equal bool ~msg:"the operand is cut along an axis read whole" true
            (Array.exists whole (Grid.cuts (Devices.grid p)))
      | r ->
      let s' = Iarray.to_array (R.shape mv (Iarray.of_array s)) in
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

(* [s]'s elements on [m 0], each its own index in C order. *)
let ramp s =
  let n = Array.fold_left ( * ) 1 s in
  Nx_array.to_device (m 0)
    (Nx_array.of_array Nx_array.Dtype.Float32 (Iarray.of_array s)
       (Array.init n Float.of_int))

(* [a]'s elements in its window [w], in C order. *)
let elements_at w a =
  let v = Option.get (Nx_array.move (Slice (Iarray.of_array w)) a) in
  Nx_array.to_array (Nx_array.copy v)

let array_of x = Iarray.get (Prim.arrays x) 0

let movement_kind : R.t -> string = function
  | Reshape _ -> "reshape"
  | Broadcast _ -> "broadcast"
  | Slice _ -> "slice"
  | Permute _ -> "permutation"
  | Window _ -> "sliding window"

(* Each device's window of a moved result, computed as {!Prim.local} gives
   the movement over the operand's window there, holds the elements that
   window holds of the movement of the whole operand. *)
let local_move_law =
  prop ~count:500 "each moved result window is its operand window moved there"
    move_case (fun (s, p, mv) ->
      let a = ramp s in
      let x = Value.Array { at = Devices.one s4 0; a; dead = Value.Live } in
      let rule = Prim.rule (Value.Move (mv, x)) in
      let routed =
        match route rule [| p |] [| s |] with
        | r -> Some r
        | exception Invalid_argument _ -> None
      in
      let cut kind =
        match routed with
        | Some r ->
            Grid.is_cut (Devices.grid r.result) && movement_kind mv = kind
        | None -> false
      in
      cover "a reshape of a cut result" (cut "reshape");
      cover "a broadcast of a cut result" (cut "broadcast");
      cover "a slice of a cut result" (cut "slice");
      match routed with
      | None -> ()
      | Some r ->
          let whole = array_of (Eval.run ~by:"t" (Value.Move (mv, x))) in
          let s' = Iarray.to_array (R.shape mv (Iarray.of_array s)) in
          Array.iteri
            (fun j _ ->
              let w = Devices.window ~by:"t" r.result s' j in
              let w' = Route.window rule ~operand:0 s w in
              let window =
                Option.get (Nx_array.move (Slice (Iarray.of_array w')) a)
              in
              let x' =
                Value.Array
                  { at = Devices.one s4 0; a = window; dead = Value.Live }
              in
              let local =
                Eval.run ~by:"t" (Prim.local (Value.Move (mv, x')) w)
              in
              equal (array float_exact)
                ~msg:(Printf.sprintf "device %d's window" j)
                (elements_at w whole)
                (Nx_array.to_array (Nx_array.copy (array_of local))))
            (Grid.devices (Devices.grid r.result)))

let () =
  exit
    (run "nx route"
       [
         group "laws"
           [ law; refusals; computes; keeps; move_law; local_move_law ];
         cases_;
       ])
