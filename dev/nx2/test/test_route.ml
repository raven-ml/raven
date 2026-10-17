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

let pp_table ppf t =
  Format.fprintf ppf "[%s]"
    (String.concat "; "
       (Array.to_list (Array.map (Format.asprintf "%a" pp_ints) t)))

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
   the results, of its extent, or none, of any extent. *)
let operand_of out =
  let open Gen in
  let rank = Array.length out in
  let* r = int_range 1 3 in
  let+ order = permutation (List.init rank Fun.id @ List.init r (fun _ -> -1))
  and+ along = array ~size:(constant r) extent in
  let axes = Array.of_list (List.filteri (fun a _ -> a < r) order) in
  (axes, Array.mapi (fun a t -> if t < 0 then along.(a) else out.(t)) axes)

(* The operand a scatter writes into: the results' shape, each axis becoming
   itself or read whole. *)
let into_of out =
  Gen.map
    (fun whole ->
      (Array.mapi (fun a w -> if w then -1 else a) whole, Array.copy out))
    (Gen.array ~size:(Gen.constant (Array.length out)) Gen.bool)

let case =
  let open Gen in
  let* rank = int_range 1 3 in
  let* out = array ~size:(constant rank) extent in
  let* n = int_range 1 3 in
  let* kind = int_range 0 2 in
  let* operands =
    match kind with
    | 0 -> constant (Array.make n ([||], out))
    | 1 -> array ~size:(constant n) (operand_of out)
    | _ ->
        let+ others = array ~size:(constant (n - 1)) (operand_of out)
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
  let rec placements i =
    if i = n then constant []
    else
      let* p = placement_of (Array.length shapes.(i)) in
      let+ rest = placements (i + 1) in
      p :: rest
  in
  let+ ps = placements 0 in
  { rule; ps = Array.of_list ps; shapes; result = out }

let case = Gen.with_pp pp_case case

(* The part of operand [i] that result window [w] depends on. *)
let depends c i (w : R.range array) =
  match c.rule with
  | Route.Elementwise -> w
  | Axes t | Into t ->
      Array.mapi
        (fun a d ->
          if t.(i).(a) < 0 then { R.start = 0; count = d; step = 1 }
          else w.(t.(i).(a)))
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
            let rs = c.result in
            let target = Devices.grid r.result in
            cover "a split result" (Grid.cuts target <> [||]);
            let read_whole =
              match c.rule with
              | Route.Elementwise -> false
              | Axes t | Into t -> Array.exists (Array.mem (-1)) t
            in
            cover "a split result beside an axis read whole"
              (read_whole && Grid.cuts target <> [||]);
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
                      (contains have (depends c i w));
                    equal ranges ~msg:"the rule's window" (depends c i w)
                      (Route.window c.rule ~operand:i c.shapes.(i) w))
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

(* A constant computed for results at [p]: its route keeps [p]'s devices, keeps
   every cut of [p] its rule allows, and reads each operand where each device's
   window of the results depends on it. *)
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
      test "a reduction of a split axis is whole on each device" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r = placed (Axes [| [| 0; -1 |] |]) [| x |] [| [| 4; 8 |] |] in
          equal placement (Devices.on s4) r.result;
          equal placement (Devices.on s4) r.operands.(0));
      test "a reduction keeps the cut of an axis it does not reduce" (fun () ->
          let x = Devices.split ~by:"t" ~axis:1 s4 in
          let r = placed (Axes [| [| -1; 0 |] |]) [| x |] [| [| 4; 8 |] |] in
          equal placement (Devices.split ~by:"t" ~axis:0 s4) r.result;
          equal bool true (r.operands.(0) == x));
      test "a scatter writes where its target lies" (fun () ->
          let into = Devices.split ~by:"t" ~axis:1 s4 and u = Devices.on s4 in
          let rule = Route.Into [| [| -1; 1 |]; [| -1; 1 |] |] in
          let r = placed rule [| u; into |] [| [| 4; 8 |]; [| 8; 8 |] |] in
          equal placement into r.result);
      test "an operation reading all whole lies on its operands' devices"
        (fun () ->
          let whole = Route.Axes [| [| -1 |] |] in
          let r = placed whole [| Devices.one s4 2 |] [| [| 4 |] |] in
          equal placement (Devices.one s4 2) r.result;
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          let r = placed whole [| x |] [| [| 8 |] |] in
          equal placement (Devices.on s4) r.result);
      test "operands read whole alone on two devices raise" (fun () ->
          invalid (fun () ->
              placed
                (Axes [| [| -1 |]; [| -1 |] |])
                [| Devices.one s4 0; Devices.one s4 1 |]
                [| [| 4 |]; [| 4 |] |]));
      test "a contraction keeps its batch and free cuts" (fun () ->
          let x = Devices.split ~by:"t" ~axis:0 s4 in
          let rule = Route.Axes [| [| 0; -1 |]; [| -1; 1 |] |] in
          let r =
            placed rule [| x; Devices.on s4 |] [| [| 8; 4 |]; [| 4; 4 |] |]
          in
          equal placement x r.result;
          equal bool true (r.operands.(0) == x);
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
      test "a rule that does not name each axis raises" (fun () ->
          invalid (fun () ->
              placed
                (Axes [| [| 0; 1 |]; [| 0 |] |])
                [| Devices.on s4; Devices.on s4 |]
                [| [| 4; 4 |]; [| 4; 4 |] |]));
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
      let x =
        Value.Array
          {
            at = Devices.one s4 0;
            a = Nx_array.create (m 0) Nx_array.Dtype.Float32 s;
            dead = Value.Live;
          }
      in
      let rule = Prim.rule (Value.Move (mv, x)) in
      let r = placed rule [| p |] [| s |] in
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
    (run "nx route"
       [ group "laws" [ law; exact; keeps; at_law; move_law ]; cases_ ])
