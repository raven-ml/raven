(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Device sets, meshes and placements, through Nx. *)

open Windtrap

let m = Nx_support.memory

module S4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])
module S2 = (val Nx.devices [ m 0; m 1 ])

let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let pp p = Format.asprintf "%a" Nx.Placement.pp p
let names s = List.map Rig.name (Nx.rigs s)

module Nowhere = struct
  include Nx_cpu

  let computes_on _ = false
end

let sets =
  group "sets"
    [
      test "the host set is the host alone" (fun () ->
          equal (list string) [ Rig.name Rig.host ] (names Nx.Host.v));
      test "a set keeps its devices in order" (fun () ->
          equal (list string) [ "m0"; "m1"; "m2"; "m3" ] (names S4.v));
      cases "a mint refuses" ~name:fst
        [
          ("no device", fun () -> ignore (Nx.devices []));
          ("a repeated device", fun () -> ignore (Nx.devices [ m 0; m 1; m 0 ]));
          ( "kernels that do not compute on a device",
            fun () -> ignore (Nx.devices ~kernels:(module Nowhere) [ m 0 ]) );
        ]
        (fun (_, f) -> invalid ~by:"Nx.devices" f);
    ]

let placements =
  group "placements"
    [
      test "a set's placements are its own" (fun () ->
          equal bool true (Nx.Placement.devices S4.on == S4.v);
          equal placement (Nx.Placement.on S4.v) S4.on;
          equal placement (Nx.Placement.split ~axis:1 S4.v) (S4.split ~axis:1));
      test "placements print as the placement that makes them" (fun () ->
          equal string (Rig.name Rig.host) (pp Nx.Host.on);
          equal string "on [m0; m1]" (pp S2.on);
          equal string "split ~axis:1 [m0; m1]" (pp (S2.split ~axis:1)));
      test "a placement over one device is that device" (fun () ->
          let module S1 = (val Nx.devices [ m 2 ]) in
          equal string "m2" (pp S1.on);
          equal string "m2" (pp (S1.split ~axis:0)));
      test "a placement on one device of a set is that device alone" (fun () ->
          let p = Nx.Placement.device S4.v (m 2) in
          equal string "m2" (pp p);
          equal bool true (Nx.Placement.devices p == S4.v);
          equal bool true (p == Nx.Placement.device S4.v (m 2));
          not_equal placement S4.on p;
          let x = Nx.place p (Nx.zeros Nx.float32 [| 3 |]) in
          equal (list string) [ "m2" ]
            (List.map
               (fun a -> Rig.name (Nx_array.device a))
               (Iarray.to_list (Option.get (Nx.Repr.shards x)))));
      test "a placement on the one device of a set is the set's whole"
        (fun () ->
          let module S1 = (val Nx.devices [ m 2 ]) in
          equal bool true (S1.on == Nx.Placement.device S1.v (m 2)));
      test "device refuses a device outside the set" (fun () ->
          invalid ~by:"Nx.Placement.device" (fun () ->
              Nx.Placement.device S2.v (m 2)));
      test "a split is not the whole on every device" (fun () ->
          not_equal placement S4.on (S4.split ~axis:0));
      cases "split refuses the axis" ~name:string_of_int
        [ -1; Nx_array.Layout.max_rank ] (fun axis ->
          invalid ~by:"Nx.Placement.split" (fun () -> S2.split ~axis));
    ]

let meshes =
  group "meshes"
    [
      test "a mesh of every axis cut in order is the split over the set"
        (fun () ->
          let mesh = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ] in
          equal placement (S4.split ~axis:0)
            (Nx.Placement.mesh mesh [ (0, [ "a"; "b" ]) ]);
          equal placement S4.on (Nx.Placement.mesh mesh []));
      test "a mesh cut over its minor axis first is not the split" (fun () ->
          let mesh = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ] in
          not_equal placement (S4.split ~axis:0)
            (Nx.Placement.mesh mesh [ (0, [ "b"; "a" ]) ]));
      test "a mesh of two cuts prints its extents and cuts" (fun () ->
          let mesh = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ] in
          equal string "mesh 2x2 [m0; m1; m2; m3] ~axis:0/1 ~axis:1/0"
            (pp (Nx.Placement.mesh mesh [ (0, [ "b" ]); (1, [ "a" ]) ])));
      cases "Mesh.v refuses" ~name:fst
        [
          ("extents that miss the set's devices", [ ("a", 3) ]);
          ("a name twice", [ ("a", 2); ("a", 2) ]);
          ("an extent of 0", [ ("a", 0); ("b", 4) ]);
        ]
        (fun (_, axes) ->
          invalid ~by:"Nx.Mesh.v" (fun () -> Nx.Mesh.v S4.v axes));
      cases "Placement.mesh refuses" ~name:fst
        [
          ("a name the mesh lacks", [ (0, [ "z" ]) ]);
          ("an axis cut twice", [ (0, [ "a" ]); (0, [ "b" ]) ]);
          ("a name in two cuts", [ (0, [ "a" ]); (1, [ "a" ]) ]);
          ("a negative axis", [ (-1, [ "a" ]) ]);
        ]
        (fun (_, cuts) ->
          let mesh = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ] in
          invalid ~by:"Nx.Placement.mesh" (fun () ->
              Nx.Placement.mesh mesh cuts));
    ]

let () = exit (run "nx devices" [ sets; placements; meshes ])
