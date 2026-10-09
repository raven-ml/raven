(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Device sets, placements and grids, through the engine's private Devices and
   Grid, copied here. *)

open Windtrap

type b

let m = Nx_support.memory
let mint ?kernels ds : b Devices.t = Devices.mint ~by:"Nx.devices" ?kernels ds
let s4 = mint [ m 0; m 1; m 2; m 3 ]
let s2 = mint [ m 0; m 1 ]
let s1 = mint [ m 2 ]
let placement = Testable.make ~pp:Devices.pp_placement ~equal:Devices.equal

let range =
  Testable.make
    ~pp:(fun ppf (r : Nx_array.Move.range) ->
      Format.fprintf ppf "%d+%d/%d" r.start r.count r.step)
    ~equal:( = )

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let kernels_name s =
  match Devices.kernels s with
  | Some (module K : Nx_kernel.S) -> Some K.name
  | None -> None

module Nowhere = struct
  include Nx_cpu

  let computes_on _ = false
end

let pp s = Format.asprintf "%a" Devices.pp s
let pp_at p = Format.asprintf "%a" Devices.pp_placement p

let sets =
  group "sets"
    [
      test "host is set 0, the host alone, computed by nx.cpu" (fun () ->
          equal int 0 (Devices.number Devices.host);
          equal int 1 (Devices.count Devices.host);
          equal bool true (Rig.equal Rig.host (Devices.rig Devices.host 0));
          equal (option string) (Some "nx.cpu") (kernels_name Devices.host));
      test "a mint is numbered after every earlier one" (fun () ->
          let a = mint [ m 0 ] in
          let b = mint [ m 0 ] in
          greater int ~than:(Devices.number s1) (Devices.number a);
          greater int ~than:(Devices.number a) (Devices.number b));
      test "two mints of one list are two sets" (fun () ->
          let a = mint [ m 0; m 1 ] and b = mint [ m 0; m 1 ] in
          not_equal int (Devices.number a) (Devices.number b);
          equal bool false
            (Devices.equal (Devices.on a) (Devices.rebrand (Devices.on b))));
      test "a set of host-run devices is computed by nx.cpu by default"
        (fun () -> equal (option string) (Some "nx.cpu") (kernels_name s4));
      test "a set keeps its devices in order" (fun () ->
          equal (list bool) [ true; true; true; true ]
            (List.init 4 (fun k -> Rig.equal (m k) (Devices.rig s4 k)));
          equal (option int) (Some 2) (Devices.position s4 (m 2));
          equal (option int) None (Devices.position s2 (m 3)));
      test "a set prints its number and devices; the host prints host"
        (fun () ->
          equal string
            (Printf.sprintf "set %d [m0; m1]" (Devices.number s2))
            (pp s2);
          equal string "host" (pp Devices.host));
      cases "a mint refuses" ~name:fst
        [
          ("no device", fun () -> mint []);
          ("a repeated device", fun () -> mint [ m 0; m 1; m 0 ]);
          ( "kernels that do not compute on a device",
            fun () -> mint ~kernels:(module Nowhere) [ m 0 ] );
        ]
        (fun (_, f) -> invalid ~by:"Nx.devices" f);
      test "a position past the set raises" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"") (fun () ->
              Devices.rig s2 2));
    ]

let placements =
  group "placements"
    [
      test "a device's placement is the one the mint made" (fun () ->
          equal bool true (Devices.one s4 1 == Devices.one s4 1);
          equal bool true (Devices.on s1 == Devices.one s1 0);
          equal bool true (Devices.split ~by:"t" ~axis:0 s1 == Devices.one s1 0);
          equal bool true
            (Devices.v ~by:"t" s4 (Grid.device 3) == Devices.one s4 3));
      test "on and split over a set print as their placements" (fun () ->
          equal string "m0" (pp_at (Devices.one s2 0));
          equal string "on [m0; m1]" (pp_at (Devices.on s2));
          equal string "split ~axis:1 [m0; m1]"
            (pp_at (Devices.split ~by:"t" ~axis:1 s2)));
      test "a constant's placement is the host's device, apart from it"
        (fun () ->
          let a : b Devices.placement = Devices.anywhere in
          equal int 0 (Devices.number (Devices.set a));
          equal (option int) (Some 0) (Devices.device a);
          equal bool false (a == Devices.rebrand (Devices.one Devices.host 0));
          equal string "anywhere" (pp_at a));
      cases "split refuses the axis" ~name:string_of_int
        [ -1; Nx_array.Layout.max_rank ] (fun axis ->
          invalid ~by:"Nx.split" (fun () ->
              Devices.split ~by:"Nx.split" ~axis s2));
      test "a grid naming a device past the set raises" (fun () ->
          invalid ~by:"t" (fun () -> Devices.v ~by:"t" s2 (Grid.device 2)));
      test "a mesh of every axis cut in order is the split over the set"
        (fun () ->
          let mesh = Devices.mesh_v ~by:"t" s4 [ ("a", 2); ("b", 2) ] in
          equal placement
            (Devices.split ~by:"t" ~axis:0 s4)
            (Devices.mesh ~by:"t" mesh [ (0, [ "a"; "b" ]) ]);
          equal placement (Devices.on s4) (Devices.mesh ~by:"t" mesh []));
      test "a mesh cut over its minor axis is not the split" (fun () ->
          let mesh = Devices.mesh_v ~by:"t" s4 [ ("a", 2); ("b", 2) ] in
          not_equal placement
            (Devices.split ~by:"t" ~axis:0 s4)
            (Devices.mesh ~by:"t" mesh [ (0, [ "b"; "a" ]) ]));
      cases "a mesh refuses" ~name:fst
        [
          ( "extents that miss the set's devices",
            fun () -> ignore (Devices.mesh_v ~by:"t" s4 [ ("a", 3) ]) );
          ( "a name twice",
            fun () -> ignore (Devices.mesh_v ~by:"t" s4 [ ("a", 2); ("a", 2) ])
          );
          ( "an extent of 0",
            fun () -> ignore (Devices.mesh_v ~by:"t" s4 [ ("a", 0); ("b", 4) ])
          );
          ( "a name it lacks",
            fun () ->
              ignore
                (Devices.mesh ~by:"t"
                   (Devices.mesh_v ~by:"t" s4 [ ("a", 4) ])
                   [ (0, [ "z" ]) ]) );
          ( "an axis cut twice",
            fun () ->
              ignore
                (Devices.mesh ~by:"t"
                   (Devices.mesh_v ~by:"t" s4 [ ("a", 2); ("b", 2) ])
                   [ (0, [ "a" ]); (0, [ "b" ]) ]) );
        ]
        (fun (_, f) -> invalid ~by:"t" f);
      test "a split's windows are the axis in order of devices" (fun () ->
          let p = Devices.split ~by:"t" ~axis:1 s4 in
          equal (array range)
            [|
              { start = 0; count = 3; step = 1 };
              { start = 4; count = 2; step = 1 };
            |]
            (Devices.window ~by:"t" p [| 3; 8 |] 2));
      test "a split that does not divide its axis raises" (fun () ->
          invalid ~by:"t" (fun () ->
              Devices.window ~by:"t"
                (Devices.split ~by:"t" ~axis:0 s4)
                [| 6 |] 0));
    ]

(* Placements over a set of four devices, of a value of rank [rank]. *)
let placement_of rank =
  let mesh = Devices.mesh_v ~by:"t" s4 [ ("a", 2); ("b", 2) ] in
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.with_pp Devices.pp_placement
    (Gen.one_of
       [
         Gen.map (Devices.one s4) (Gen.int_range 0 3);
         Gen.constant (Devices.on s4);
         Gen.map (fun axis -> Devices.split ~by:"t" ~axis s4) axis;
         Gen.map
           (fun (x, y) ->
             if x = y then Devices.mesh ~by:"t" mesh [ (x, [ "a"; "b" ]) ]
             else Devices.mesh ~by:"t" mesh [ (x, [ "a" ]); (y, [ "b" ]) ])
           (Gen.pair axis axis);
         Gen.map (fun x -> Devices.mesh ~by:"t" mesh [ (x, [ "b" ]) ]) axis;
       ])

let shape_of rank = Gen.array ~size:(Gen.constant rank) (Gen.of_list [ 4; 8 ])

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let placed =
  Gen.with_pp
    (fun ppf (p, s) ->
      Format.fprintf ppf "%a over %a" Devices.pp_placement p pp_shape s)
    (Gen.bind (Gen.int_range 1 3) (fun rank ->
         Gen.pair (placement_of rank) (shape_of rank)))

let rec indices = function
  | [] -> [ [] ]
  | n :: rest ->
      List.concat_map
        (fun i -> List.map (fun t -> i :: t) (indices rest))
        (List.init n Fun.id)

let inside (w : Nx_array.Move.range array) idx =
  List.for_all2
    (fun (r : Nx_array.Move.range) i -> i >= r.start && i < r.start + r.count)
    (Array.to_list w) idx

let windows =
  group "windows"
    [
      prop "every index lies in exactly one distinct window" placed
        (fun (p, s) ->
          cover "split" (Grid.cuts (Devices.grid p) <> [||]);
          cover "two cut axes" (Array.length (Grid.cuts (Devices.grid p)) = 2);
          let ws =
            List.sort_uniq compare
              (List.init
                 (Grid.count (Devices.grid p))
                 (fun i -> Array.to_list (Devices.window ~by:"t" p s i)))
          in
          let ws = List.map Array.of_list ws in
          List.iter
            (fun idx ->
              equal int ~msg:"windows holding the index" 1
                (List.length (List.filter (fun w -> inside w idx) ws)))
            (indices (Array.to_list s)));
      prop "a placement keeps its windows across a leading axis and back" placed
        (fun (p, _) ->
          equal placement p
            (Devices.without_leading_axis (Devices.with_leading_axis p)));
      prop "a new leading axis is whole and moves the windows up" placed
        (fun (p, s) ->
          let q = Devices.with_leading_axis p in
          let s' = Array.append [| 2 |] s in
          for i = 0 to Grid.count (Devices.grid p) - 1 do
            let w = Devices.window ~by:"t" p s i
            and w' = Devices.window ~by:"t" q s' i in
            equal range { start = 0; count = 2; step = 1 } w'.(0);
            equal (array range) w (Array.sub w' 1 (Array.length s))
          done);
    ]

let grids =
  group "grids"
    [
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
            Grid.v ~devices:[| 0; 1; 2; 3 |] ~extents:[| 2; 2 |]
              ~cuts:[| (0, [| 0 |]); (0, [| 1 |]) |] );
          ( "a grid axis named twice",
            Grid.v ~devices:[| 0; 1; 2; 3 |] ~extents:[| 2; 2 |]
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
      test "a grid of one device is that device" (fun () ->
          let g =
            require_ok ~pp:Format.pp_print_string
              (Grid.v ~devices:[| 3 |] ~extents:[| 1; 1 |]
                 ~cuts:[| (0, [| 1 |]) |])
          in
          equal (option int) (Some 3) (Grid.one g));
      test "selecting a tile keeps the devices that hold it" (fun () ->
          let g = Devices.grid (Devices.split ~by:"t" ~axis:0 s4) in
          equal (option int) (Some 2) (Grid.one (Grid.select g ~axis:0 2));
          raises_match (Exn.invalid_arg ~substring:"") (fun () ->
              Grid.select g ~axis:0 4));
      test "uncutting an axis leaves copies on the same devices" (fun () ->
          let g = Devices.grid (Devices.split ~by:"t" ~axis:1 s4) in
          equal bool true
            (Grid.equal (Devices.grid (Devices.on s4)) (Grid.uncut g ~axis:1)));
      test "a mesh prints its extents, devices and cuts" (fun () ->
          let mesh = Devices.mesh_v ~by:"t" s4 [ ("a", 2); ("b", 2) ] in
          equal string "mesh 2x2 [m0; m1; m2; m3] ~axis:0/1 ~axis:1/0"
            (pp_at (Devices.mesh ~by:"t" mesh [ (0, [ "b" ]); (1, [ "a" ]) ])));
    ]

let () = exit (run "nx devices" [ sets; placements; windows; grids ])
