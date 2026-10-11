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

let pp_at ppf = function
  | Some p -> Devices.pp_placement ppf p
  | None -> Format.pp_print_string ppf "every set"

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
  | Move _ -> Format.pp_print_string ppf "Move"

(* An operation: its rule, and its operands' placements and shapes, of one shape
   each but along the axes the rule reads whole. *)
type case = {
  rule : Route.rule;
  ps : b Devices.placement option array;
  shapes : int array array;
}

let pp_case ppf c =
  Format.fprintf ppf "%a over" pp_rule c.rule;
  Array.iteri
    (fun i p ->
      Format.fprintf ppf " %a at %a," pp_ints c.shapes.(i) pp_at p)
    c.ps

let placement_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.one_of
    [
      Gen.constant None;
      Gen.map (fun k -> Some (Devices.one s4 k)) (Gen.int_range 0 3);
      Gen.constant (Some (Devices.on s4));
      Gen.map (fun axis -> Some (Devices.split ~by:"t" ~axis s4)) axis;
      Gen.map
        (fun (x, y) ->
          Some
            (if x = y then Devices.mesh ~by:"t" mesh [ (x, [ "a"; "b" ]) ]
             else Devices.mesh ~by:"t" mesh [ (x, [ "a" ]); (y, [ "b" ]) ]))
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
  | Move mv -> R.shape mv s

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
  | Replicated | Move _ -> Array.init (Array.length s) full

let contains (outer : R.range array) (inner : R.range array) =
  Array.for_all2
    (fun (o : R.range) (i : R.range) ->
      i.start >= o.start && i.start + i.count <= o.start + o.count)
    outer inner

let position p k = Array.find_index (( = ) k) (Grid.devices (Devices.grid p))

let law =
  prop ~count:500 "each result window reads operands on its own device" case
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
      test "a reduction of a split axis is whole on each device" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r = placed (Reduce [| 1 |]) [| x |] [| [| 4; 8 |] |] in
          equal placement (Devices.on s4) r.result;
          equal placement (Devices.on s4) r.operands.(0));
      test "a reduction keeps the cut of an axis it does not reduce" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r = placed (Reduce [| 0 |]) [| x |] [| [| 4; 8 |] |] in
          equal placement (Devices.split ~by:"t" ~axis:0 s4) r.result;
          equal bool true (r.operands.(0) == x));
      test "a scatter writes where its target lies" (fun () ->
          let into = Devices.split ~by:"t" ~axis:1 s4 and u = Devices.on s4 in
          let r = placed (Into 0) [| u; into |] [| [| 4; 8 |]; [| 8; 8 |] |] in
          equal placement into r.result);
      test "the result of a replicated operation is on every device" (fun () ->
          let r = placed Replicated [| Devices.one s4 2 |] [| [| 4 |] |] in
          equal placement (Devices.on s4) r.result);
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
      test "a split that does not divide its axis raises" (fun () ->
          invalid (fun () ->
              placed Elementwise
                [| Devices.split ~by:"t" ~axis:0 s4 |]
                [| [| 6 |] |]));
      test "an axis the operand lacks raises" (fun () ->
          invalid (fun () ->
              placed (Along [| 2 |]) [| Devices.on s4 |] [| [| 4; 4 |] |]));
      test "placements and shapes of two lengths raise" (fun () ->
          invalid (fun () -> placed Elementwise [| Devices.on s4 |] [||]));
    ]

(* Movements: the operand index each result index reads, as Move states it. *)
let source mv s idx =
  let r = Array.length s in
  match mv with
  | R.Reshape s' ->
      let k = ref 0 in
      Array.iteri (fun i j -> k := (!k * s'.(i)) + j) idx;
      let src = Array.make r 0 in
      for i = r - 1 downto 0 do
        src.(i) <- !k mod s.(i);
        k := !k / s.(i)
      done;
      src
  | Broadcast s' ->
      let off = Array.length s' - r in
      Array.init r (fun i -> if s.(i) = 1 then 0 else idx.(off + i))
  | Permute p ->
      let src = Array.make r 0 in
      Array.iteri (fun i a -> src.(a) <- idx.(i)) p;
      src
  | Slice rs -> Array.init r (fun i -> rs.(i).start + (idx.(i) * rs.(i).step))
  | Window ws ->
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
         (fun p -> R.Permute (Array.of_list p))
         (permutation (List.init r Fun.id));
       constant (R.Broadcast (Array.append [| 3 |] s));
       constant
         (R.Reshape (Array.append [| 2; s.(0) / 2 |] (Array.sub s 1 (r - 1))));
       map
         (fun choice ->
           R.Slice
             (Array.mapi
                (fun i d ->
                  match (choice + i) mod 3 with
                  | 0 -> { R.start = 0; count = d; step = 1 }
                  | 1 -> { R.start = 0; count = d / 2; step = 1 }
                  | _ -> { R.start = d - 1; count = d; step = -1 })
                s))
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
          (R.Reshape (Array.append [| s.(0) * s.(1) |] (Array.sub s 2 (r - 2))));
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
      let r = placed (Move mv) [| p |] [| s |] in
      let s' = R.shape mv s in
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
    (run "nx route" [ group "laws" [ law; exact; keeps; move_law ]; cases_ ])
