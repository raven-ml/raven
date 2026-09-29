(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Devices and placement, on test devices that hold their storage in host memory
   of their own and count what moves. A placed value equals its host value under
   every movement, or refuses exactly the movements that would move elements
   between devices; results live with their operands; reads copy; views share
   their cell. *)

open Windtrap
open Nx_test
open Devices

let four = [ d1; d2; d3; d4 ]
let values = array float_exact
let on1 = Nx.Placement.device d1

let iota shape =
  Nx.create Nx.float32 shape
    (Array.init (Ref.numel shape) (fun i -> float_of_int (i + 1)))

let refuses = List.iter raises_invalid_arg

(* Where a value lives, as nx.mli states it. *)
type where =
  | One of Nx.Device.t
  | Copies of Nx.Device.t list
  | Split of int * Nx.Device.t list

let to_placement = function
  | One d -> Nx.Placement.device d
  | Copies ds -> Nx.Placement.replicated ds
  | Split (axis, ds) -> Nx.Placement.sharded ~axis ds

let pp_where ppf w = Nx.Placement.pp ppf (to_placement w)
let devices_of = function One d -> [ d ] | Copies ds | Split (_, ds) -> ds
let sorted ds = List.sort Nx.Device.compare ds
let same_set ds es = List.equal Nx.Device.equal (sorted ds) (sorted es)

(* Placements *)

let windows = Testable.(array (pair int int))

(* The window of a value of [shape] that [d] holds under the placement of
   devices [ds] split along [axis], if any. *)
let window_of shape (ds, axis) d =
  let whole = Array.map (fun n -> (0, n)) shape in
  match (List.find_index (( == ) d) ds, axis) with
  | None, _ -> None
  | Some _, None -> Some whole
  | Some i, Some a ->
      let k = shape.(a) / List.length ds in
      whole.(a) <- (i * k, (i + 1) * k);
      Some whole

let make (ds, axis) =
  match axis with
  | None -> Nx.Placement.replicated ds
  | Some axis -> Nx.Placement.sharded ~axis ds

let placement_args =
  let open Gen in
  let+ ds = permutation ~pp:Nx.Device.pp four
  and+ n = int_range 1 4
  and+ axis = option (int_range 0 2) in
  (List.filteri (fun i _ -> i < n) ds, axis)

let placements =
  group "placements"
    [
      test "are in normal form" (fun () ->
          List.iter
            (fun (a, b) -> equal placement a b)
            [
              (on1, Nx.Placement.replicated [ d1 ]);
              (on1, Nx.Placement.sharded ~axis:3 [ d1 ]);
              (Nx.Placement.host, Nx.Placement.device Nx.Device.host);
              ( Nx.Placement.replicated [ d1; d2 ],
                Nx.Placement.replicated [ d2; d1 ] );
            ];
          not_equal placement
            (Nx.Placement.sharded ~axis:0 [ d1; d2 ])
            (Nx.Placement.sharded ~axis:0 [ d2; d1 ]));
      prop "window gives each device its slice, and equal compares the windows"
        (Gen.pair placement_args placement_args) (fun (a, b) ->
          let shape = [| 12; 12; 12 |] in
          let p = make a in
          List.iter
            (fun d ->
              equal (option windows) (window_of shape a d)
                (Some (Nx.Placement.window p shape d)))
            (Nx.Placement.devices p);
          let same =
            List.for_all
              (fun d -> window_of shape a d = window_of shape b d)
              four
          in
          cover "equal placements" same;
          equal bool same (Nx.Placement.equal p (make b)));
      test
        "refuse no devices, a repeated device, mixed engines, a negative axis, \
         and windows of a device they lack or an axis that does not divide"
        (fun () ->
          let s = Nx.Placement.sharded ~axis:1 [ d1; d2; d3 ] in
          refuses
            [
              (fun () -> ignore (Nx.Placement.replicated []));
              (fun () -> ignore (Nx.Placement.replicated [ d1; d1 ]));
              (fun () -> ignore (Nx.Placement.sharded ~axis:0 [ d1; other ]));
              (fun () -> ignore (Nx.Placement.sharded ~axis:(-1) [ d1; d2 ]));
              (fun () -> ignore (Nx.Placement.window s [| 2; 6 |] d4));
              (fun () -> ignore (Nx.Placement.window s [| 2; 5 |] d1));
              (fun () -> ignore (Nx.Placement.window s [| 6 |] d1));
            ]);
      (* Grids are built inside nx.effect only. *)
      test "a grid is kept in normal form and compared by its windows"
        (fun () ->
          let grid = Nx_effect.Grid.v four in
          List.iter
            (fun (a, b) -> equal placement a b)
            [
              ( Nx.Placement.sharded ~axis:0 four,
                grid [ 2; 2 ] [ (0, [ 0; 1 ]) ] );
              (Nx.Placement.replicated four, grid [ 2; 2 ] []);
              ( Nx.Placement.sharded ~axis:1 four,
                grid [ 1; 4; 1 ] [ (1, [ 1 ]) ] );
              ( Nx.Placement.sharded ~axis:0 [ d1; d3; d2; d4 ],
                grid [ 2; 2 ] [ (0, [ 1; 0 ]) ] );
            ];
          equal windows
            [| (2, 4); (0, 3) |]
            (Nx.Placement.window
               (grid [ 2; 2 ] [ (0, [ 0 ]); (1, [ 1 ]) ])
               [| 4; 6 |] d3);
          equal windows
            [| (0, 2); (0, 6) |]
            (Nx.Placement.window (grid [ 2; 2 ] [ (0, [ 0 ]) ]) [| 4; 6 |] d2);
          refuses
            [
              (fun () -> ignore (grid [ 2; 3 ] []));
              (fun () -> ignore (grid [ 2; 2 ] [ (0, [ 0 ]); (1, [ 0 ]) ]));
            ]);
    ]

(* Values at a placement *)

(* A shape and a placement of it: on one device, copies, or a split over two or
   four devices along an axis they divide. *)
let placed_shape =
  let open Gen in
  let* shape =
    array ~size:(int_range 1 3)
      (of_list ~pp:Format.pp_print_int [ 1; 2; 3; 4; 6; 8 ])
  in
  let+ kind = int_range 0 3 and+ pick = int_range 0 100 in
  let split ds =
    match
      List.filter
        (fun a -> shape.(a) mod List.length ds = 0)
        (List.init (Array.length shape) Fun.id)
    with
    | [] -> Copies ds
    | axes -> Split (List.nth axes (pick mod List.length axes), ds)
  in
  ( shape,
    match kind with
    | 0 -> One d1
    | 1 -> Copies [ d1; d2 ]
    | 2 -> split [ d1; d2 ]
    | _ -> split four )

let pp_placed ppf (shape, where) =
  Format.fprintf ppf "%a %a" pp_shape shape pp_where where

let place_tests =
  group "place"
    [
      prop "holds the value at the placement, the source staying where it was"
        (Gen.with_pp pp_placed placed_shape) (fun (shape, where) ->
          let x = iota shape and p = to_placement where in
          let y = Nx.place p x in
          equal (pair placement placement) (p, Nx.Placement.host)
            (Nx.placement y, Nx.placement x);
          is_true ~msg:"placing where it is gives the value" (Nx.place p y == y);
          Law.round_trip (tensor float_exact) pass (Nx.place p)
            (Nx.place Nx.Placement.host)
            x);
      test "refuses a split of an axis the value lacks or does not divide"
        (fun () ->
          let x = iota [| 2; 3 |] in
          refuses
            [
              (fun () ->
                ignore (Nx.place (Nx.Placement.sharded ~axis:1 [ d1; d2 ]) x));
              (fun () ->
                ignore (Nx.place (Nx.Placement.sharded ~axis:2 [ d1; d2 ]) x));
            ]);
      test "a consumed value is neither placed nor read, and keeps its shape"
        (fun () ->
          let p = Nx.place on1 (iota [| 2; 3 |]) in
          (cell_of p).state <- Consumed { path = "2.keys" };
          raises_match (Exn.invalid_arg ~substring:"consumed at 2.keys")
            (fun () -> Nx.place on1 p);
          raises_invalid_arg (fun () -> Nx.to_array p);
          equal (array int) [| 2; 3 |] (Nx.shape p));
    ]

(* Movements *)

type movement =
  | Transpose of int list
  | Reshape of int array
  | Index of int * int
  | Range of int * int * int
  | Flip of int option
  | Broadcast of int array
  | Window of int * int * int

let move : type a b. movement -> (a, b) Nx.t -> (a, b) Nx.t =
 fun m t ->
  let at d s = List.init d (fun _ -> Nx.A) @ [ s ] in
  match m with
  | Transpose axes -> Nx.transpose ~axes t
  | Reshape s -> Nx.reshape s t
  | Index (d, i) -> Nx.slice (at d (I i)) t
  | Range (d, lo, hi) -> Nx.slice (at d (R (lo, hi))) t
  | Flip None -> Nx.flip t
  | Flip (Some d) -> Nx.flip ~axes:[ d ] t
  | Broadcast s -> Nx.broadcast_to s t
  | Window (axis, window, step) -> Nx.sliding_window ~axis ~window ~step t

let pp_movement ppf = function
  | Transpose axes ->
      Format.fprintf ppf "transpose %a" pp_shape (Array.of_list axes)
  | Reshape s -> Format.fprintf ppf "reshape %a" pp_shape s
  | Index (d, i) -> Format.fprintf ppf "index %d of axis %d" i d
  | Range (d, lo, hi) -> Format.fprintf ppf "range %d-%d of axis %d" lo hi d
  | Flip None -> Format.fprintf ppf "flip"
  | Flip (Some d) -> Format.fprintf ppf "flip axis %d" d
  | Broadcast s -> Format.fprintf ppf "broadcast to %a" pp_shape s
  | Window (d, w, s) ->
      Format.fprintf ppf "windows of %d every %d along %d" w s d

(* Where a value at [where] of [shape] lands after [m], [None] where [m] would
   move elements between devices: a split axis follows the movement, a cut
   inside one shard lands on that shard's device, and a flip, windows or a cut
   across shards of the split axis, or a reshape that does not keep the extents
   before it or divide its new extent, would move elements. *)
let fate shape m where =
  match where with
  | One _ | Copies _ -> Some where
  | Split (a, ds) -> (
      let k = List.length ds in
      let c = shape.(a) / k and split a = Some (Split (a, ds)) in
      match m with
      | Transpose axes -> split (Option.get (List.find_index (( = ) a) axes))
      | Reshape target ->
          let lead = Ref.numel (Array.sub shape 0 a) in
          let b = ref None and acc = ref 1 in
          Array.iteri
            (fun i d ->
              if !acc = lead then b := Some i;
              acc := !acc * d)
            target;
          Option.bind !b (fun b ->
              if target.(b) mod k = 0 then split b else None)
      | Index (d, i) ->
          if d = a then Some (One (List.nth ds (i / c)))
          else split (if d < a then a - 1 else a)
      | Range (d, lo, hi) ->
          if d <> a || (lo = 0 && hi = shape.(a)) then split a
          else if lo / c = (hi - 1) / c then Some (One (List.nth ds (lo / c)))
          else None
      | Flip (Some d) when d <> a -> split a
      | Flip _ -> None
      | Broadcast s -> split (a + Array.length s - Array.length shape)
      | Window (d, _, _) -> if d = a then None else split a)

let rec permutations = function
  | [] -> [ [] ]
  | l ->
      List.concat_map
        (fun x ->
          List.map (List.cons x) (permutations (List.filter (( <> ) x) l)))
        l

(* The movements of [host] of each kind: views only, so reshapes that keep
   [host]'s elements where they are, and cuts that are non-empty. *)
let kinds host =
  let s = Nx.shape host and n = Nx.numel host in
  let axes = List.init (Array.length s) Fun.id in
  let divisors = List.filter (fun d -> n mod d = 0) (List.init n succ) in
  let shapes =
    [ [| n |] ]
    @ List.map (fun a -> [| a; n / a |]) divisors
    @ List.concat_map
        (fun a ->
          List.map
            (fun b -> [| a; b; n / a / b |])
            (List.filter (fun b -> n / a mod b = 0) divisors))
        divisors
  in
  let per_axis f = List.concat_map f axes in
  let wider = Array.map (fun d -> if d = 1 then 3 else d) s in
  [
    List.map (fun p -> Transpose p) (permutations axes);
    List.filter_map
      (fun t -> if viewable host t then Some (Reshape t) else None)
      shapes;
    per_axis (fun d ->
        List.map
          (fun i -> Index (d, i))
          (List.sort_uniq compare [ 0; s.(d) / 2; s.(d) - 1 ])
        @ List.concat_map
            (fun lo ->
              List.init (s.(d) - lo) (fun k -> Range (d, lo, lo + k + 1)))
            (List.init s.(d) Fun.id));
    Flip None :: per_axis (fun d -> [ Flip (Some d) ]);
    Broadcast (Array.append [| 2 |] s)
    :: (if wider = s then [] else [ Broadcast wider ]);
    per_axis (fun d ->
        List.concat_map
          (fun w -> [ Window (d, w, 1); Window (d, w, 2) ])
          (List.init s.(d) succ));
  ]

let move_both ((shape, where), steps) =
  let host = iota shape in
  let placed = Nx.place (to_placement where) host in
  let cell = cell_of placed in
  List.fold_left
    (fun state (kind, pick) ->
      match state with
      | None -> None
      | Some (host, placed, where) -> (
          (* A kind with no movement of [host] falls back to transposes. *)
          let cs =
            match List.nth (kinds host) kind with
            | [] -> List.hd (kinds host)
            | cs -> cs
          in
          let m = List.nth cs (pick mod List.length cs) in
          let msg = Format.asprintf "%a" pp_movement m in
          let expected = fate (Nx.shape host) m where in
          let uploaded = !uploads and read = !elements_read in
          match (move m placed, expected) with
          | exception Invalid_argument e ->
              cover "a refusal" true;
              is_true
                ~msg:(msg ^ " refused, but moves no element")
                (expected = None);
              contains ~msg:"the remedy"
                ~sub:"place the value replicated or on one device first" e;
              None
          | _, None -> failf "%s moved elements between devices" msg
          | moved, Some where ->
              cover "a cut inside one shard"
                (match (state, where) with
                | Some (_, _, Split _), One _ -> true
                | _ -> false);
              let host = move m host in
              equal ~msg (pair int int) (uploaded, read)
                (!uploads, !elements_read);
              equal ~msg placement (to_placement where) (Nx.placement moved);
              is_true ~msg:(msg ^ ": the source's cell") (cell_of moved == cell);
              equal ~msg (tensor float_exact) host moved;
              Some (host, moved, where)))
    (Some (host, placed, where))
    steps
  |> Option.iter (fun (_, moved, _) ->
         cell.state <- Consumed { path = "0" };
         raises_invalid_arg (fun () -> Nx.to_array moved))

let movements =
  group "movements"
    [
      prop
        "of a placed value equal those of its host value, or raise exactly \
         when they would move elements between devices; a view of a consumed \
         value is not read"
        (Gen.pair
           (Gen.with_pp pp_placed placed_shape)
           (Gen.list ~size:(Gen.int_range 1 3)
              (Gen.pair (Gen.int_range 0 5) (Gen.int_range 0 10_000))))
        move_both;
      test "a value cut along two axes moves by the whole shapes" (fun () ->
          let p = Nx_effect.Grid.v four [ 2; 2 ] [ (0, [ 0 ]); (1, [ 1 ]) ] in
          let x = Nx.reshape [| 2; 4 |] (iota [| 8 |]) in
          let s = Nx.place p x in
          let r = Nx.reshape [| 2; 2; 2 |] s and row = Nx.slice [ I 1 ] s in
          equal (pair placement placement)
            (p, Nx.Placement.sharded ~axis:0 [ d3; d4 ])
            (Nx.placement r, Nx.placement row);
          equal
            (pair (tensor float_exact) (tensor float_exact))
            (Nx.reshape [| 2; 2; 2 |] x, Nx.slice [ I 1 ] x)
            (r, row);
          raises_invalid_arg (fun () -> Nx.reshape [| 8 |] s));
      test "a view off its storage's devices is refused" (fun () ->
          match
            Nx.place (Nx.Placement.sharded ~axis:0 four) (iota [| 8; 6 |])
          with
          | Nx_effect.Placed r ->
              raises_invalid_arg (fun () ->
                  Nx_effect.placed
                    (Nx.Placement.device other)
                    r.r_dtype r.r_view r.r_cell);
              ignore
                (Nx_effect.placed (Nx.Placement.device d2) r.r_dtype r.r_view
                   r.r_cell)
          | _ -> fail "expected a placed value");
    ]

(* Results *)

(* Where an elementwise result of operands at [a] and [b] lives, [None] for the
   host: with the placed operand, copies taking a split; [Error] where the
   operands' devices or splits differ. *)
let join a b =
  let same ok = if ok then Ok b else Error () in
  match (a, b) with
  | None, w | w, None -> Ok w
  | Some (Copies ds), Some (Split (_, es)) -> same (same_set ds es)
  | Some (Split (_, ds)), Some (Copies es) ->
      if same_set ds es then Ok a else Error ()
  | Some (Split (x, ds)), Some (Split (y, es)) ->
      same (x = y && List.equal ( == ) ds es)
  | Some w, Some v -> same (same_set (devices_of w) (devices_of v))

let at where x =
  match where with None -> x | Some w -> Nx.place (to_placement w) x

let results =
  let ds = [ d1; d2 ] in
  let rows = Nx.Placement.sharded ~axis:0 ds
  and cols = Nx.Placement.sharded ~axis:1 ds
  and copies = Nx.Placement.replicated ds in
  let x = iota [| 8; 6 |] and w = iota [| 6; 4 |] in
  let s () = Nx.place rows x
  and t () = Nx.place cols x
  and r () = Nx.place copies x in
  let batch = Nx.reshape [| 2; 4; 4 |] (iota [| 32 |]) in
  let spd =
    Nx.add
      (Nx.matmul batch (Nx.transpose ~axes:[ 0; 2; 1 ] batch))
      (Nx.mul_s (Nx.eye Nx.float32 4) 1000.)
  in
  let operand =
    Gen.of_list
      ~pp:
        (Format.pp_print_option
           ~none:(fun ppf () -> Format.pp_print_string ppf "host")
           pp_where)
      [
        None;
        Some (One d1);
        Some (One d2);
        Some (Copies ds);
        Some (Copies [ d2; d1 ]);
        Some (Split (0, ds));
        Some (Split (1, ds));
        Some (Split (0, [ d2; d1 ]));
      ]
  in
  group "results"
    [
      prop
        "of an elementwise operation live with their operands, or raise when \
         the operands' devices or splits differ"
        (Gen.pair operand operand) (fun (wa, wb) ->
          let y = Nx.flip (iota [| 4; 6 |]) and x = iota [| 4; 6 |] in
          let expected = join wa wb in
          cover "a refusal" (Result.is_error expected);
          cover "copies taking a split"
            (match (wa, wb, expected) with
            | Some (Copies _), Some (Split _), Ok _ -> true
            | _ -> false);
          elements_read := 0;
          match (Nx.add (at wa x) (at wb y), expected) with
          | exception Invalid_argument _ ->
              equal ~msg:"refused, having read nothing" (pair bool int) (true, 0)
                (Result.is_error expected, !elements_read)
          | _, Error () -> fail "the operands' devices or splits differ"
          | z, Ok w ->
              equal placement
                (Option.fold ~none:Nx.Placement.host ~some:to_placement w)
                (Nx.placement z);
              equal (tensor float_exact) (Nx.add x y) z);
      prop
        "of a reduction over a split axis are copies, and others keep the split"
        (Gen.triple
           (Gen.of_list ~pp:pp_where
              [ Split (0, ds); Split (1, ds); Split (2, four) ])
           (Gen.subsequence ~pp:Format.pp_print_int [ 0; 1; 2 ])
           Gen.bool)
        (fun (where, axes, keepdims) ->
          let x = iota [| 2; 4; 4 |] in
          let expected =
            match where with
            | Split (a, ds) when axes = [] || List.mem a axes -> Copies ds
            | Split (a, ds) ->
                Split
                  ( (if keepdims then a
                     else a - List.length (List.filter (fun b -> b < a) axes)),
                    ds )
            | w -> w
          in
          let axes = if axes = [] then None else Some axes in
          let y = Nx.sum ?axes ~keepdims (Nx.place (to_placement where) x) in
          equal placement (to_placement expected) (Nx.placement y);
          equal (tensor float_exact) (Nx.sum ?axes ~keepdims x) y);
      cases "an operation along the split axis raises, naming the remedy"
        ~name:fst
        [
          ("cumsum", fun () -> ignore (Nx.cumsum ~axis:0 (s ())));
          ("sort", fun () -> ignore (Nx.sort ~axis:0 (s ())));
          ("pad", fun () -> ignore (Nx.pad [| (1, 1); (0, 0) |] 0. (s ())));
          ( "concatenate",
            fun () -> ignore (Nx.concatenate ~axis:0 [ s (); s () ]) );
          ( "fft",
            fun () -> ignore (Nx.fft ~axis:0 (Nx.cast Nx.complex64 (s ()))) );
          ( "cholesky of split columns",
            fun () ->
              ignore
                (Nx.cholesky (Nx.place (Nx.Placement.sharded ~axis:2 ds) spd))
          );
        ]
        (fun (_, f) ->
          raises_match
            (Exn.invalid_arg
               ~substring:"place the value replicated or on one device first")
            f);
      cases "of an operation along the other axis keep the split" ~name:fst
        [
          ("a scalar operand", fun v -> Nx.add_s v 1.);
          ("exp", Nx.exp);
          ("a comparison", fun v -> Nx.cast Nx.float32 (Nx.less v (Nx.flip x)));
          ("cumsum", Nx.cumsum ~axis:1);
          ("sort", fun v -> fst (Nx.sort ~axis:1 v));
          ("pad", Nx.pad [| (0, 0); (1, 1) |] 0.);
          ("concatenate", fun v -> Nx.concatenate ~axis:1 [ v; v ]);
          ("copy", Nx.copy);
          ("repeat along the split axis", Nx.repeat ~axis:0 2);
        ]
        (fun (_, f) ->
          let y = f (s ()) in
          equal placement rows (Nx.placement y);
          equal (tensor float_exact) (f x) y);
      cases "of products, gathers and whole shards live where nx.mli says"
        ~name:(fun (n, _, _) -> n)
        [
          ( "rows times copies",
            rows,
            fun () -> (Nx.matmul x w, Nx.matmul (s ()) (Nx.place copies w)) );
          ( "copies times columns",
            cols,
            fun () -> (Nx.matmul x w, Nx.matmul (r ()) (Nx.place cols w)) );
          ( "columns times rows",
            copies,
            fun () -> (Nx.matmul x w, Nx.matmul (t ()) (Nx.place rows w)) );
          ( "rows gathered from columns",
            cols,
            fun () -> (Nx.slice [ L [ 1; 0 ] ] x, Nx.slice [ L [ 1; 0 ] ] (t ()))
          );
          ( "rows gathered along the split axis",
            copies,
            fun () ->
              (Nx.slice [ L [ 7; 0; 3 ] ] x, Nx.slice [ L [ 7; 0; 3 ] ] (s ()))
          );
          ( "rows taken by split positions",
            rows,
            fun () ->
              let ids =
                Nx.create Nx.int32 [| 8 |] [| 5l; 0l; 3l; 3l; 1l; 2l; 4l; 0l |]
              in
              ( Nx.take ~axis:0 ~indices:ids w,
                Nx.take ~axis:0 ~indices:(Nx.place rows ids) (Nx.place copies w)
              ) );
          ( "linear algebra over a split batch",
            rows,
            fun () -> (Nx.cholesky spd, Nx.cholesky (Nx.place rows spd)) );
          ( "constants over copies",
            copies,
            fun () -> (Nx.tril x, Nx.tril (r ())) );
          ( "a roll by one whole shard",
            copies,
            fun () -> (Nx.roll ~axis:0 4 x, Nx.roll ~axis:0 4 (s ())) );
          ( "a sum of two whole shards of four",
            Nx.Placement.replicated four,
            fun () ->
              let s4 = Nx.place (Nx.Placement.sharded ~axis:0 four) x in
              ( Nx.add (Nx.slice [ R (0, 2) ] x) (Nx.slice [ R (2, 4) ] x),
                Nx.add (Nx.slice [ R (0, 2) ] s4) (Nx.slice [ R (2, 4) ] s4) )
          );
        ]
        (fun (_, p, f) ->
          let expected, y = f () in
          equal placement p (Nx.placement y);
          equal (tensor float_exact) expected y);
      test "parts of shards that are not whole shards of one storage raise"
        (fun () ->
          let s = s () in
          let bottom = Nx.slice [ R (4, 8) ] s in
          refuses
            [
              (fun () -> ignore (Nx.roll ~axis:0 1 s));
              (fun () -> ignore (Nx.add (Nx.slice [ R (0, 2) ] s) bottom));
              (fun () ->
                ignore
                  (Nx.add
                     (Nx.broadcast_to [| 4; 6 |] (Nx.slice [ R (0, 1) ] s))
                     bottom));
              (fun () ->
                ignore
                  (Nx.add
                     (Nx.slice [ R (0, 4) ] s)
                     (Nx.place (Nx.Placement.device d2)
                        (Nx.slice [ R (4, 8) ] x))));
            ]);
      test
        "a scalar made on a device is held, a filled value is uploaded split \
         like its model" (fun () ->
          let p = Nx.place on1 (iota [| 2; 3 |])
          and split = Nx.place (Nx.Placement.sharded ~axis:0 four) x in
          let sp = Nx.sum p and ss = Nx.sum split in
          let uploads_of f =
            let before = !uploads in
            let y = f () in
            (!uploads - before, y)
          in
          let made =
            [
              uploads_of (fun () -> Nx.mul_s p 2.);
              uploads_of (fun () -> Nx.zeros_like p);
              uploads_of (fun () -> Nx.zeros_like sp);
              uploads_of (fun () -> Nx.full_like ss 2.);
              uploads_of (fun () -> Nx.ones_like (Nx.transpose split));
            ]
          in
          equal
            (list (pair int placement))
            [
              (1, on1);
              (1, on1);
              (0, on1);
              (0, Nx.Placement.replicated four);
              (1, Nx.Placement.sharded ~axis:1 four);
            ]
            (List.map (fun (n, y) -> (n, Nx.placement y)) made);
          let z = snd (List.nth made 1) and filled = snd (List.nth made 4) in
          is_true ~msg:"a filled value's view covers its storage"
            (match z with
            | Nx_effect.Placed r -> Nx_effect.covers r
            | _ -> false);
          (match (cell_of filled).state with
          | Live (Mem (_, shards)) ->
              equal ~msg:"a slice on each device" (list int) [ 12; 12; 12; 12 ]
                (List.map Nx_buffer.length shards)
          | _ -> fail "expected storage of the test engine");
          equal (tensor float_exact)
            (Nx.mul_s (iota [| 2; 3 |]) 2.)
            (snd (List.hd made));
          equal (tensor float_exact)
            (Nx.full Nx.float32 [| 6; 8 |] 3.)
            (Nx.fill 3. (Nx.transpose split)));
    ]

(* Reads and views *)

let reads =
  group "reads and views"
    [
      test "a read copies what it reads and leaves the value where it is"
        (fun () ->
          let p = Nx.place on1 (iota [| 2; 3 |])
          and s =
            Nx.place (Nx.Placement.sharded ~axis:0 four) (iota [| 8; 6 |])
          in
          let counted f =
            elements_read := 0;
            ignore (f ());
            !elements_read
          in
          equal (list int) [ 1; 6; 1; 6 ]
            [
              counted (fun () -> Nx.item [ 1; 2 ] p);
              counted (fun () -> Nx.to_array p);
              counted (fun () -> Nx.item [ 5; 2 ] s);
              counted (fun () ->
                  Nx_effect.to_host (Nx.slice [ R (4, 6); R (1, 4) ] s));
            ];
          equal placement on1 (Nx.placement p);
          equal string
            (Nx.to_string (Nx.transpose (iota [| 8; 6 |])))
            (Nx.to_string (Nx.transpose s));
          raises_invalid_arg (fun () -> Nx.data p));
      test "views share their cell; a window's contiguous copy does not"
        (fun () ->
          let p = Nx.place on1 (iota [| 2; 3 |]) in
          let v = Nx.transpose (Nx.slice [ R (0, 1) ] p) in
          let c = Nx.contiguous v in
          equal (list bool) [ true; true; false ]
            [
              cell_of v == cell_of p;
              cell_of (Nx.contiguous p) == cell_of p;
              cell_of c == cell_of p;
            ];
          (cell_of p).state <- Consumed { path = "0" };
          refuses
            [
              (fun () -> ignore (Nx.to_array v));
              (fun () -> ignore (Nx.add v v));
            ];
          equal values [| 1.; 2.; 3. |] (Nx.to_array c));
    ]

(* Identities *)

type Nx_effect.node += Identity_probe

let traced d =
  Nx_effect.traced (Nx_effect.On [ d ]) Nx.float32 [| 1 |] Identity_probe

let identities =
  group "identities"
    [
      test "constructors on several domains give distinct identities" (fun () ->
          let start = Atomic.make false in
          let ids () =
            while not (Atomic.get start) do
              Domain.cpu_relax ()
            done;
            List.concat_map
              (fun _ ->
                let d = Nx_effect.Device.make "IDENTITY" engine in
                let v =
                  Nx_effect.placed (Nx.Placement.device d) Nx.float32
                    (Nx_core.View.create [| 1 |])
                    (Nx_effect.cell ~placement:(Nx.Placement.device d) ~length:1
                       (Nx_effect.Held (Nx.float32, 1.)))
                in
                d.d_id
                :: List.map Nx_effect.identity_hash
                     [ v; traced d; Nx_effect.reshape v [| 1; 1 |] ])
              (List.init 4096 Fun.id)
          in
          let workers = List.init 4 (fun _ -> Domain.spawn ids) in
          Atomic.set start true;
          let ids = List.concat_map Domain.join workers in
          equal int (List.length ids)
            (List.length (List.sort_uniq Int.compare ids)));
      test "the trace frontier separates existing and new traces" (fun () ->
          let before = Nx_effect.identity_hash (traced d1) in
          let horizon = Nx_effect.next_traced_id () in
          equal ~msg:"observing the frontier allocates no identity" int horizon
            (Nx_effect.next_traced_id ());
          let after = Nx_effect.identity_hash (traced d1) in
          let device = Nx_effect.Device.make "FRONTIER" engine in
          let later = Nx_effect.identity_hash (traced d1) in
          equal ~msg:"the next trace starts at the frontier" int horizon after;
          less int ~than:horizon before;
          equal (list bool) [ true; true ]
            [ after < device.d_id; device.d_id < later ]);
    ]

let () =
  exit
    (run "nx placement"
       [ placements; place_tests; movements; results; reads; identities ])
