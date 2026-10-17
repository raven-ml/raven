(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Device grids, through the engine's private Grid, copied here. *)

open Windtrap
module M = Nx_array.Move

let grid = Testable.make ~pp:(Grid.pp Format.pp_print_int) ~equal:Grid.equal

let range =
  Testable.make
    ~pp:(fun ppf (r : M.range) ->
      Format.fprintf ppf "%d+%d/%d" r.start r.count r.step)
    ~equal:( = )

let ok r = require_ok ~pp:Format.pp_print_string r
let v devices extents cuts = ok (Grid.v ~devices ~extents ~cuts)
let all = [| 0; 1; 2; 3 |]
let on = v all [| 4 |] [||]
let split axis = v all [| 4 |] [| (axis, [| 0 |]) |]
let mesh cuts = v all [| 2; 2 |] cuts
let pp g = Format.asprintf "%a" (Grid.pp Format.pp_print_int) g

(* Grids over four devices, of a value of rank [rank]. *)
let grid_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.with_pp
    (Grid.pp Format.pp_print_int)
    (Gen.one_of
       [
         Gen.map Grid.device (Gen.int_range 0 3);
         Gen.constant on;
         Gen.map split axis;
         Gen.map
           (fun (x, y) ->
             if x = y then mesh [| (x, [| 0; 1 |]) |]
             else mesh [| (x, [| 0 |]); (y, [| 1 |]) |])
           (Gen.pair axis axis);
         Gen.map (fun x -> mesh [| (x, [| 1 |]) |]) axis;
       ])

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let gridded =
  Gen.with_pp
    (fun ppf (g, s) ->
      Format.fprintf ppf "%a over %a" (Grid.pp Format.pp_print_int) g pp_shape s)
    (Gen.bind (Gen.int_range 1 3) (fun rank ->
         Gen.pair (grid_of rank)
           (Gen.array ~size:(Gen.constant rank) (Gen.of_list [ 4; 8 ]))))

let rec indices = function
  | [] -> [ [] ]
  | n :: rest ->
      List.concat_map
        (fun i -> List.map (fun t -> i :: t) (indices rest))
        (List.init n Fun.id)

let inside (w : M.range array) idx =
  List.for_all2
    (fun (r : M.range) i -> i >= r.start && i < r.start + r.count)
    (Array.to_list w) idx

let window g s i = ok (Grid.window g s i)

let laws =
  group "laws"
    [
      prop "every index lies in exactly one distinct window" gridded
        (fun (g, s) ->
          cover "a cut" (Grid.cuts g <> [||]);
          cover "two cut axes" (Array.length (Grid.cuts g) = 2);
          let ws =
            List.sort_uniq compare
              (List.init (Grid.count g) (fun i -> Array.to_list (window g s i)))
          in
          let ws = List.map Array.of_list ws in
          List.iter
            (fun idx ->
              equal int ~msg:"windows holding the index" 1
                (List.length (List.filter (fun w -> inside w idx) ws)))
            (indices (Array.to_list s)));
      prop "a leading axis added and removed gives the grid back" gridded
        (fun (g, _) ->
          equal grid g
            (Grid.map_axes pred (Grid.uncut (Grid.map_axes succ g) ~axis:0)));
      prop "a new leading axis is whole and moves the windows up" gridded
        (fun (g, s) ->
          let g' = Grid.map_axes succ g and s' = Array.append [| 2 |] s in
          for i = 0 to Grid.count g - 1 do
            let w = window g s i and w' = window g' s' i in
            equal range { start = 0; count = 2; step = 1 } w'.(0);
            equal (array range) w (Array.sub w' 1 (Array.length s))
          done);
    ]

let forms =
  group "normal form"
    [
      test "a mesh with every grid axis cutting one axis in order is a split"
        (fun () -> equal grid (split 0) (mesh [| (0, [| 0; 1 |]) |]));
      test "a mesh cut minor axis first is not a split" (fun () ->
          not_equal grid (split 0) (mesh [| (0, [| 1; 0 |]) |]));
      test "a mesh without cuts is the whole on every device" (fun () ->
          equal grid on (mesh [||]));
      test "a grid of one device is that device" (fun () ->
          equal (option int) (Some 3)
            (Grid.one (v [| 3 |] [| 1; 1 |] [| (0, [| 1 |]) |])));
      cases "Grid.v refuses" ~name:fst
        [
          ( "extents that miss the devices",
            Grid.v ~devices:[| 0; 1 |] ~extents:[| 3 |] ~cuts:[||] );
          ( "a repeated device",
            Grid.v ~devices:[| 0; 0 |] ~extents:[| 2 |] ~cuts:[||] );
          ( "a negative device",
            Grid.v ~devices:[| -1; 0 |] ~extents:[| 2 |] ~cuts:[||] );
          ("a zero extent", Grid.v ~devices:[||] ~extents:[| 0 |] ~cuts:[||]);
          ( "an axis cut twice",
            Grid.v ~devices:all ~extents:[| 2; 2 |]
              ~cuts:[| (0, [| 0 |]); (0, [| 1 |]) |] );
          ( "a grid axis named twice",
            Grid.v ~devices:all ~extents:[| 2; 2 |]
              ~cuts:[| (0, [| 0 |]); (1, [| 0 |]) |] );
          ( "a negative axis",
            Grid.v ~devices:[| 0; 1 |] ~extents:[| 2 |]
              ~cuts:[| (-1, [| 0 |]) |] );
          ( "a grid axis out of range",
            Grid.v ~devices:[| 0; 1 |] ~extents:[| 2 |] ~cuts:[| (0, [| 1 |]) |]
          );
        ]
        (fun (_, r) ->
          ignore
            (require_error
               ~pp:(fun ppf _ -> Format.pp_print_string ppf "a grid")
               r));
    ]

let operations =
  group "operations"
    [
      test "a split's windows are the axis in order of devices" (fun () ->
          equal (array range)
            [|
              { start = 0; count = 3; step = 1 };
              { start = 4; count = 2; step = 1 };
            |]
            (window (split 1) [| 3; 8 |] 2));
      test "a split that does not divide its axis is an error" (fun () ->
          ignore
            (require_error
               ~pp:(fun ppf _ -> Format.pp_print_string ppf "a window")
               (Grid.window (split 0) [| 6 |] 0)));
      test "selecting a tile keeps the devices that hold it" (fun () ->
          equal (option int) (Some 2)
            (Grid.one (Grid.select (split 0) ~axis:0 2));
          raises_match (Exn.invalid_arg ~substring:"") (fun () ->
              Grid.select (split 0) ~axis:0 4));
      test "uncutting an axis leaves copies on the same devices" (fun () ->
          equal grid on (Grid.uncut (split 1) ~axis:1));
      test "a grid prints as the placement that makes it" (fun () ->
          equal string "on [0; 1; 2; 3]" (pp on);
          equal string "split ~axis:1 [0; 1; 2; 3]" (pp (split 1));
          equal string "mesh 2x2 [0; 1; 2; 3] ~axis:0/1 ~axis:1/0"
            (pp (mesh [| (0, [| 1 |]); (1, [| 0 |]) |])));
    ]

let () = exit (run "nx grid" [ laws; forms; operations ])
