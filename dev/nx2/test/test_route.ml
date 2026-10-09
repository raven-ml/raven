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

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_rule ppf = function
  | Route.Elementwise -> Format.pp_print_string ppf "Elementwise"
  | Reduce a -> Format.fprintf ppf "Reduce %a" pp_ints a
  | Along a -> Format.fprintf ppf "Along %a" pp_ints a
  | Gather a -> Format.fprintf ppf "Gather %d" a
  | Into a -> Format.fprintf ppf "Into %d" a
  | Replicated -> Format.pp_print_string ppf "Replicated"

(* An operation: its rule, and its operands' placements and shapes, of one shape
   each but along the axes the rule reads whole. *)
type case = {
  rule : Route.rule;
  ps : b Devices.placement array;
  shapes : int array array;
}

let pp_case ppf c =
  Format.fprintf ppf "%a over" pp_rule c.rule;
  Array.iteri
    (fun i p ->
      Format.fprintf ppf " %a at %a," pp_ints c.shapes.(i) Devices.pp_placement
        p)
    c.ps

let placement_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.one_of
    [
      Gen.constant Devices.anywhere;
      Gen.map (Devices.one s4) (Gen.int_range 0 3);
      Gen.constant (Devices.on s4);
      Gen.map (fun axis -> Devices.split ~by:"t" ~axis s4) axis;
      Gen.map
        (fun (x, y) ->
          if x = y then Devices.mesh ~by:"t" mesh [ (x, [ "a"; "b" ]) ]
          else Devices.mesh ~by:"t" mesh [ (x, [ "a" ]); (y, [ "b" ]) ])
        (Gen.pair axis axis);
    ]

let axes_of rank =
  Gen.map
    (fun l -> Array.of_list (List.sort_uniq Int.compare l))
    (Gen.list ~size:(Gen.int_range 1 rank) (Gen.int_range 0 (rank - 1)))

let case =
  let open Gen in
  with_pp pp_case
    (let* rank = int_range 1 3 in
     let* shape = array ~size:(constant rank) (of_list [ 4; 8 ]) in
     let* rule =
       one_of
         [
           constant Route.Elementwise;
           map (fun a -> Route.Reduce a) (axes_of rank);
           map (fun a -> Route.Along a) (axes_of rank);
           map (fun a -> Route.Gather a) (int_range 0 (rank - 1));
           map (fun a -> Route.Into a) (int_range 0 (rank - 1));
           constant Route.Replicated;
         ]
     in
     let* n =
       match rule with Route.Gather _ -> constant 2 | _ -> int_range 1 3
     in
     (* Operands differ along an axis the rule reads whole, as a gather's source
        or a scatter's updates do. *)
     let* along = array ~size:(constant n) (of_list [ 4; 8 ]) in
     let shapes =
       Array.init n (fun i ->
           let s = Array.copy shape in
           (match rule with
           | Route.Gather a when i = 1 -> s.(a) <- along.(i)
           | Route.Into a -> s.(a) <- along.(i)
           | _ -> ());
           s)
     in
     let+ ps = array ~size:(constant n) (placement_of rank) in
     { rule; ps; shapes })

(* The shape of the results. *)
let result_shape c =
  let s = c.shapes.(0) in
  match c.rule with
  | Route.Reduce axes ->
      Array.of_list
        (List.filteri (fun a _ -> not (Array.mem a axes)) (Array.to_list s))
  | Into _ -> c.shapes.(Array.length c.shapes - 1)
  | Elementwise | Along _ | Gather _ | Replicated -> s

(* The part of operand [i] that result window [w] depends on. *)
let depends c i (w : R.range array) =
  let s = c.shapes.(i) in
  let full a = { R.start = 0; count = s.(a); step = 1 } in
  match c.rule with
  | Route.Elementwise -> w
  | Reduce axes ->
      let k = ref 0 in
      Array.init (Array.length s) (fun a ->
          if Array.mem a axes then full a
          else
            let r = w.(!k) in
            incr k;
            r)
  | Along axes ->
      Array.mapi (fun a r -> if Array.mem a axes then full a else r) w
  | Gather axis ->
      if i = 1 then Array.mapi (fun a r -> if a = axis then full a else r) w
      else w
  | Into axis -> Array.mapi (fun a r -> if a = axis then full a else r) w
  | Replicated -> Array.init (Array.length s) full

let contains (outer : R.range array) (inner : R.range array) =
  Array.for_all2
    (fun (o : R.range) (i : R.range) ->
      i.start >= o.start && i.start + i.count <= o.start + o.count)
    outer inner

let position p k = Array.find_index (( = ) k) (Grid.devices (Devices.grid p))

let law =
  prop ~count:500 "each result window reads operands on its own device" case
    (fun c ->
      let all_constant = Array.for_all (fun p -> p == Devices.anywhere) c.ps in
      cover "constants only" all_constant;
      match route c.rule c.ps c.shapes with
      | exception Invalid_argument msg ->
          (* Two operands alone on two devices is the one refusal of valid
             operands. *)
          let alone =
            List.sort_uniq compare
              (List.filter_map
                 (fun p ->
                   if p == Devices.anywhere then None else Devices.device p)
                 (Array.to_list c.ps))
          in
          cover "two devices alone" true;
          greater int ~msg ~than:1 (List.length alone)
      | r ->
          if all_constant then equal placement Devices.anywhere r.result
          else begin
            let rs = result_shape c in
            let target = Devices.grid r.result in
            cover "a split result" (Grid.cuts target <> [||]);
            Array.iteri
              (fun j k ->
                let w = Devices.window ~by:"t" r.result rs j in
                Array.iteri
                  (fun i p ->
                    let pos =
                      require_some ~msg:"the operand is on the result's device"
                        (position p k)
                    in
                    let have = Devices.window ~by:"t" p c.shapes.(i) pos in
                    equal bool ~msg:"the window holds what the result reads"
                      true
                      (contains have (depends c i w)))
                  r.operands)
              (Grid.devices target)
          end)

let keeps =
  prop "operands already at the result's placement stay there" case (fun c ->
      let p = c.ps.(0) in
      assume (p != Devices.anywhere);
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
      test "a constant's placement is the host's device, apart from it"
        (fun () ->
          let a : b Devices.placement = Devices.anywhere in
          equal int 0 (Devices.number (Devices.set a));
          equal (option int) (Some 0) (Devices.device a);
          equal bool false (a == Devices.rebrand (Devices.one Devices.host 0));
          equal string "anywhere" (Format.asprintf "%a" Devices.pp_placement a));
      test "a constant beside a host value is read on the host" (fun () ->
          let h : b Devices.placement =
            Devices.rebrand (Devices.one Devices.host 0)
          in
          let r =
            route Elementwise [| Devices.anywhere; h |] [| [| 4 |]; [| 4 |] |]
          in
          equal bool true (r.result == h));
      test "a constant beside a split operand is read split" (fun () ->
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          let r =
            route Elementwise [| x; Devices.anywhere |] [| [| 8 |]; [| 8 |] |]
          in
          equal placement x r.result;
          equal placement x r.operands.(1));
      test "a reduction of a split axis is whole on each device" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r = route (Reduce [| 1 |]) [| x |] [| [| 4; 8 |] |] in
          equal placement (Devices.on s4) r.result;
          equal placement (Devices.on s4) r.operands.(0));
      test "a reduction keeps the cut of an axis it does not reduce" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r = route (Reduce [| 0 |]) [| x |] [| [| 4; 8 |] |] in
          equal placement (Devices.split ~by:"t" ~axis:0 s4) r.result;
          equal bool true (r.operands.(0) == x));
      test "a scatter writes where its target lies" (fun () ->
          let into = Devices.split ~by:"t" ~axis:1 s4 and u = Devices.on s4 in
          let r = route (Into 0) [| u; into |] [| [| 4; 8 |]; [| 8; 8 |] |] in
          equal placement into r.result);
      test "the result of a replicated operation is on every device" (fun () ->
          let r = route Replicated [| Devices.one s4 2 |] [| [| 4 |] |] in
          equal placement (Devices.on s4) r.result);
      test "operands on two sets raise" (fun () ->
          invalid (fun () ->
              route Elementwise
                [| Devices.on s4; Devices.rebrand (Devices.on other) |]
                [| [| 4 |]; [| 4 |] |]));
      test "operands alone on two devices raise" (fun () ->
          invalid (fun () ->
              route Elementwise
                [| Devices.one s4 0; Devices.one s4 1 |]
                [| [| 4 |]; [| 4 |] |]));
      test "a split that does not divide its axis raises" (fun () ->
          invalid (fun () ->
              route Elementwise
                [| Devices.split ~by:"t" ~axis:0 s4 |]
                [| [| 6 |] |]));
      test "an axis the operand lacks raises" (fun () ->
          invalid (fun () ->
              route (Along [| 2 |]) [| Devices.on s4 |] [| [| 4; 4 |] |]));
      test "placements and shapes of two lengths raise" (fun () ->
          invalid (fun () -> route Elementwise [| Devices.on s4 |] [||]));
    ]

let () = exit (run "nx route" [ group "laws" [ law; keeps ]; cases_ ])
