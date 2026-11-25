(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Device sets, meshes and placements, through Nx. *)

open Windtrap
open Nx_support
module A = Nx_array

let m = Nx_support.memory

module S1 = (val Nx.devices [ m 2 ])
module S2 = (val Nx.devices [ m 0; m 1 ])
module S3 = (val Nx.devices [ m 0; m 1; m 2 ])
module S4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])
module Twin = (val Nx.devices [ m 0; m 1 ])

(* One set unpacked twice. *)
let named_twice = Nx.devices [ m 3 ]

module Named = (val named_twice)
module Renamed = (val named_twice)

(* A misuse raises [Invalid_argument] whose message is the function the user
   called, a colon and the reason (nx.mli's Errors). *)
let refuses ~by f =
  let named = function
    | Invalid_argument msg ->
        String.starts_with ~prefix:(by ^ ": ") msg
        && String.length msg > String.length by + 2
    | _ -> false
  in
  raises_match named f

let pp p = Format.asprintf "%a" Nx.Placement.pp p
let names s = List.map Rig.name (Nx.Devices.to_list s)
let verdict b = if b then "holds" else "does not hold"

(* [x] placed at [p], as each device's name and the elements it holds. *)
let windows p x =
  Nx.Repr.shards ~by:"test" (Nx.place p x)
  |> Iarray.to_list
  |> List.map (fun a -> (Rig.name (A.device a), Array.to_list (A.to_array a)))

module Nowhere = struct
  include Nx_cpu

  let computes_on _ = false
end

(* A kernel library that claims every device: a mint over a device the host does
   not run takes it only when asked. Nothing here computes with it. *)
module Everywhere = struct
  include Nx_cpu

  let computes_on _ = true
end

module Host2 = (val Nx.devices [ Rig.host ])

(* Devices *)

let devices_group =
  group "Devices"
    [
      test "the host set is the host alone" (fun () ->
          equal (list string) [ Rig.name Rig.host ] (names Nx.Host.v));
      test "a set keeps its devices in order" (fun () ->
          equal (list string) [ "m0"; "m1"; "m2"; "m3" ] (names S4.v));
      cases "has_kernels holds iff a kernel library computes on the set"
        ~id:(fun (n, _, _) -> n)
        [
          ("the host", true, fun () -> Nx.Devices.has_kernels Nx.Host.v);
          ("host-run devices", true, fun () -> Nx.Devices.has_kernels S2.v);
          ( "a set over the host's device",
            true,
            fun () -> Nx.Devices.has_kernels Host2.v );
          ( "a device the host does not run",
            false,
            fun () ->
              let module Off = (val Nx.devices [ Nx_array_support.off_host () ])
              in
              Nx.Devices.has_kernels Off.v );
          ( "that device with the mint's kernels",
            true,
            fun () ->
              let module Given =
                (val Nx.devices
                       ~kernels:(module Everywhere)
                       [ Nx_array_support.off_host () ])
              in
              Nx.Devices.has_kernels Given.v );
        ]
        (fun (_, expected, has) ->
          equal string (verdict expected) (verdict (has ())));
      test "a set over the host's device computes there" (fun () ->
          let x = Nx.place Host2.on (Nx.ones Nx.float32 [| 2 |]) in
          let z = Nx.add x x in
          equal (placement ()) Host2.on (Nx.placement z);
          equal (array float_exact) [| 2.; 2. |] (Nx.to_array z));
    ]

(* Mesh.v *)

let mesh_group =
  group "Mesh.v"
    [
      cases
        "Mesh.v raises unless its extents are positive, multiply to the \
         devices, and its names are distinct"
        ~id:fst
        [
          ("extents that miss the set's devices", [ ("a", 3) ]);
          ("a name twice", [ ("a", 2); ("a", 2) ]);
          ("an extent of 0", [ ("a", 0); ("b", 4) ]);
          ("negative extents of the right product", [ ("a", -2); ("b", -2) ]);
        ]
        (fun (_, axes) ->
          refuses ~by:"Nx.Mesh.v" (fun () -> Nx.Mesh.v S4.v axes));
    ]

(* Placement *)

let grid = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ]
let column = Nx.Mesh.v S4.v [ ("a", 4); ("b", 1) ]

(* A split of [arange n] over the first [k] software devices. *)
type split = Split : int * 'd Nx.Placement.t -> split

let splits =
  Gen.of_list
    ~pp:(fun ppf (Split (k, _)) -> Format.fprintf ppf "over %d devices" k)
    [
      Split (1, S1.split ~axis:0);
      Split (2, S2.split ~axis:0);
      Split (3, S3.split ~axis:0);
      Split (4, S4.split ~axis:0);
    ]

(* nx.mli: of n elements over k devices, the first n mod k windows hold n / k +
   1 and the others n / k, in order. *)
let law_split (n, Split (k, p)) =
  let size i = (n / k) + if i < n mod k then 1 else 0 in
  let start i = List.fold_left ( + ) 0 (List.init i size) in
  let expected =
    List.init k (fun i ->
        List.init (size i) (fun j -> Int32.of_int (start i + j)))
  in
  equal
    (list (list int32))
    expected
    (List.map snd (windows p (Nx.arange Nx.int32 0 n 1)))

(* Over the 2x2 mesh, the device at (i1, i2) holds window i1 × 2 + i2 of the
   axis cut over both, major first; an axis no cut names holds copies. *)
let mesh_windows =
  [
    ( "cut over a then b",
      [ (0, [ "a"; "b" ]) ],
      [ [ 0; 1 ]; [ 2; 3 ]; [ 4; 5 ]; [ 6; 7 ] ] );
    ( "cut over b then a",
      [ (0, [ "b"; "a" ]) ],
      [ [ 0; 1 ]; [ 4; 5 ]; [ 2; 3 ]; [ 6; 7 ] ] );
    ( "cut over a, copies over b",
      [ (0, [ "a" ]) ],
      [ [ 0; 1; 2; 3 ]; [ 0; 1; 2; 3 ]; [ 4; 5; 6; 7 ]; [ 4; 5; 6; 7 ] ] );
    ( "cut over b, copies over a",
      [ (0, [ "b" ]) ],
      [ [ 0; 1; 2; 3 ]; [ 4; 5; 6; 7 ]; [ 0; 1; 2; 3 ]; [ 4; 5; 6; 7 ] ] );
  ]

(* Placements over S4 for [equal]: some built two ways, as a split and a mesh,
   or over a mesh axis of extent 1. *)
let built =
  [
    ("on", S4.on);
    ("device m2", Nx.Placement.device S4.v (m 2));
    ("split 0", S4.split ~axis:0);
    ("split 1", S4.split ~axis:1);
    ("2x2 cut 0 over a, b", Nx.Placement.mesh grid [ (0, [ "a"; "b" ]) ]);
    ("2x2 cut 0 over b, a", Nx.Placement.mesh grid [ (0, [ "b"; "a" ]) ]);
    ( "2x2 cuts 0 over a, 1 over b",
      Nx.Placement.mesh grid [ (0, [ "a" ]); (1, [ "b" ]) ] );
    ("2x2 cut 0 over b", Nx.Placement.mesh grid [ (0, [ "b" ]) ]);
    ("2x2 uncut", Nx.Placement.mesh grid []);
    ("4x1 cut 0 over b", Nx.Placement.mesh column [ (0, [ "b" ]) ]);
    ("4x1 cut 0 over a, b", Nx.Placement.mesh column [ (0, [ "a"; "b" ]) ]);
    ("4x1 cut 1 over a", Nx.Placement.mesh column [ (1, [ "a" ]) ]);
    ("4x1 cut 2 over a", Nx.Placement.mesh column [ (2, [ "a" ]) ]);
  ]

let pairs =
  let one =
    Gen.of_list ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n) built
  in
  Gen.pair one one

let cube = lazy (Nx.reshape [| 4; 4; 4 |] (Nx.arange Nx.float32 0 64 1))

let law_equal ((n, p), (n', q)) =
  let same_windows =
    windows p (Lazy.force cube) = windows q (Lazy.force cube)
  in
  cover "equal, built apart" (same_windows && n <> n');
  cover "unequal" (not same_windows);
  equal string (verdict same_windows) (verdict (Nx.Placement.equal p q))

(* Placements over several sets, a set unpacked twice and twin mints among them,
   as values of no brand, each with the row of its set. *)
type any = Any : 'd Nx.Placement.t -> any

let over_sets =
  [
    (0, Any Nx.Host.on);
    (0, Any (Nx.Host.split ~axis:0));
    (1, Any S2.on);
    (1, Any (S2.split ~axis:0));
    (1, Any (Nx.Placement.device S2.v (m 1)));
    (2, Any S4.on);
    (2, Any (S4.split ~axis:1));
    (3, Any Twin.on);
    (3, Any (Twin.split ~axis:1));
    (4, Any Named.on);
    (4, Any (Renamed.split ~axis:0));
  ]

let same (_, Any p) (_, Any q) = Option.is_some (Nx.Placement.same p q)

let brands =
  Testable.make
    ~pp:(fun ppf (r, Any p) ->
      Format.fprintf ppf "set %d: %a" r Nx.Placement.pp p)
    ~equal:same

let law_same (((r, _) as a), ((r', _) as b)) =
  Law.equivalence brands (a, b);
  equal string (verdict (r = r')) (verdict (same a b))

(* The axes a placement cuts into two windows or more. In the law, each axis of
   a mesh over [S4]'s devices takes a role: it holds copies, the value is a sum
   over it, or it cuts one of the value's axes [0] to [2]. A value's axis is cut
   into as many windows as the extents of the mesh axes that cut it multiply
   to. *)
let summed = Nx.Placement.partial grid [ "a" ] [ (1, [ "b" ]) ]
let row name expected p = (name, expected, fun () -> Nx.Placement.cut_axes p)
let layouts = [| [ 4 ]; [ 2; 2 ]; [ 1; 4 ]; [ 4; 1 ]; [ 2; 1; 2 ] |]
let copies = -2
let sums = -1

let meshes =
  Gen.(
    pair
      (int_range 0 (Array.length layouts - 1))
      (list ~size:(constant 3) (int_range copies 2)))

let law_cut_axes (layout, roles) =
  let axis g extent = (Printf.sprintf "g%d" g, extent, List.nth roles g) in
  let laid = List.mapi axis layouts.(layout) in
  let mesh = Nx.Mesh.v S4.v (List.map (fun (n, e, _) -> (n, e)) laid) in
  let over r = List.filter (fun (_, _, role) -> role = r) laid in
  let names over = List.map (fun (name, _, _) -> name) over in
  let tiles r = List.fold_left (fun t (_, e, _) -> t * e) 1 (over r) in
  (* The value's axes that a mesh axis cuts, given last first. *)
  let named = List.filter (fun a -> over a <> []) [ 2; 1; 0 ] in
  let expected = List.rev (List.filter (fun a -> tiles a >= 2) named) in
  cover "no axis cut" (expected = []);
  cover "two axes cut" (List.length expected = 2);
  cover "an axis cut over one device alone"
    (List.exists (fun a -> tiles a = 1) named);
  cover "an axis cut over two mesh axes"
    (List.exists (fun a -> List.length (over a) = 2) named);
  cover "a sum over two devices or more" (tiles sums >= 2);
  let cut = List.map (fun a -> (a, names (over a))) named in
  let p = Nx.Placement.partial mesh (names (over sums)) cut in
  let closed = Nx.Placement.closed p and uncut = Nx.Placement.uncut p in
  let axes p = Nx.Placement.cut_axes p in
  equal ~msg:"the sum" (list int) expected (axes p);
  equal ~msg:"the mesh" (list int) expected (axes (Nx.Placement.mesh mesh cut));
  equal ~msg:"closed" (list int) expected (axes closed);
  equal ~msg:"uncut" (list int) [] (axes uncut);
  equal ~msg:"every axis whole on each device" string
    (verdict (expected = []))
    (verdict (Nx.Placement.equal uncut closed))

let placement_group =
  group "Placement"
    [
      test "device holds the whole value on that device alone" (fun () ->
          let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
          equal
            (list (pair string (list float_exact)))
            [ ("m2", [ 1.; 2.; 3. ]) ]
            (windows (Nx.Placement.device S4.v (m 2)) x));
      test "device of a set's one device is the set's whole" (fun () ->
          equal (placement ()) S1.on (Nx.Placement.device S1.v (m 2)));
      test "device raises for a device outside the set" (fun () ->
          refuses ~by:"Nx.Placement.device" (fun () ->
              Nx.Placement.device S2.v (m 2)));
      prop
        "split cuts n elements in order, the first n mod k of k windows one \
         longer"
        Gen.(pair (int_range 0 9) splits)
        law_split;
      cases "split raises for an axis below 0 or past a layout's last"
        ~id:string_of_int [ -1; Nx_array.Layout.max_rank ] (fun axis ->
          refuses ~by:"Nx.Placement.split" (fun () -> S2.split ~axis));
      cases
        "mesh gives the device at (i1, i2) window i1 × e2 + i2, and copies \
         along a mesh axis no cut names"
        ~id:(fun (n, _, _) -> n)
        mesh_windows
        (fun (_, cuts, expected) ->
          let x = Nx.arange Nx.int32 0 8 1 in
          equal
            (list (list int32))
            (List.map (List.map Int32.of_int) expected)
            (List.map snd (windows (Nx.Placement.mesh grid cuts) x)));
      cases "mesh raises" ~id:fst
        [
          ("a name the mesh lacks", [ (0, [ "z" ]) ]);
          ("an axis cut twice", [ (0, [ "a" ]); (0, [ "b" ]) ]);
          ("a name in two cuts", [ (0, [ "a" ]); (1, [ "a" ]) ]);
          ("a negative axis", [ (-1, [ "a" ]) ]);
          ( "an axis past a layout's last",
            [ (Nx_array.Layout.max_rank, [ "a" ]) ] );
        ]
        (fun (_, cuts) ->
          refuses ~by:"Nx.Placement.mesh" (fun () ->
              Nx.Placement.mesh grid cuts));
      cases "partial raises for a name summed twice or summed and cut" ~id:fst
        [
          ("a name summed twice", ([ "a"; "a" ], []));
          ("a name summed and cut", ([ "a" ], [ (0, [ "a" ]) ]));
        ]
        (fun (_, (sums, cuts)) ->
          refuses ~by:"Nx.Placement.partial" (fun () ->
              Nx.Placement.partial grid sums cuts));
      cases "closed and uncut give back a placement they leave as it is"
        ~id:(fun (n, _, _) -> n)
        [
          ("closed of on", Nx.Placement.closed, S4.on);
          ("closed of a split", Nx.Placement.closed, S4.split ~axis:1);
          ( "closed of a mesh",
            Nx.Placement.closed,
            Nx.Placement.mesh grid [ (0, [ "b" ]) ] );
          ( "closed of a device",
            Nx.Placement.closed,
            Nx.Placement.device S4.v (m 2) );
          ("uncut of on", Nx.Placement.uncut, S4.on);
          ( "uncut of a device",
            Nx.Placement.uncut,
            Nx.Placement.device S4.v (m 2) );
        ]
        (fun (_, f, p) ->
          satisfies ~claim:"is the placement given" (placement ()) (( == ) p)
            (f p));
      test "devices is a placement's set" (fun () ->
          equal (placement ()) S4.on
            (Nx.Placement.on (Nx.Placement.devices (S4.split ~axis:1))));
      prop ~count:169 "equal holds iff each device holds the same window" pairs
        law_equal;
      cases "equal holds over one set alone, whatever its name"
        ~id:(fun (n, _, _) -> n)
        [
          ("twin mints", false, fun () -> Nx.Placement.equal S2.on Twin.on);
          ( "one set unpacked twice",
            true,
            fun () -> Nx.Placement.equal Named.on Renamed.on );
        ]
        (fun (_, expected, eq) ->
          equal string (verdict expected) (verdict (eq ())));
      prop ~count:121 "same is an equivalence whose classes are the sets"
        Gen.(
          let one =
            of_list
              ~pp:(fun ppf (r, _) -> Format.fprintf ppf "set %d" r)
              over_sets
          in
          pair one one)
        law_same;
      test "one set unpacked twice is of one brand" (fun () ->
          match Nx.Placement.same Named.on Renamed.on with
          | None -> fail "Named and Renamed name one set"
          | Some Type.Equal ->
              let x : (float, Nx.float32_elt, Named.d) Nx.t =
                Nx.place Renamed.on (Nx.ones Nx.float32 [| 2 |])
              in
              equal (array float_exact) [| 1.; 1. |] (Nx.to_array x));
      cases "cut_axes is the axes cut into two windows or more"
        ~id:(fun (name, _, _) -> name)
        [
          row "the host" [] Nx.Host.on;
          row "on" [] S4.on;
          row "one device of a set" [] (Nx.Placement.device S4.v (m 2));
          row "a split over one device" [] (S1.split ~axis:1);
          row "a split over two devices" [ 1 ] (S2.split ~axis:1);
          row "a split along an axis counted from the front" [ 5 ]
            (S4.split ~axis:5);
          row "a split along a layout's last axis"
            [ Nx_array.Layout.max_rank - 1 ]
            (S4.split ~axis:(Nx_array.Layout.max_rank - 1));
          row "a mesh that cuts two axes" [ 0; 2 ]
            (Nx.Placement.mesh grid [ (2, [ "a" ]); (0, [ "b" ]) ]);
          row "a mesh that cuts one axis over both of its own" [ 3 ]
            (Nx.Placement.mesh grid [ (3, [ "a"; "b" ]) ]);
          row "a mesh that cuts no axis" [] (Nx.Placement.mesh grid []);
          row "a cut over a mesh axis of extent 1 alone" []
            (Nx.Placement.mesh column [ (0, [ "b" ]) ]);
          row "a sum" [] (Nx.Placement.partial grid [ "a"; "b" ] []);
          row "a sum of terms cut along an axis" [ 1 ] summed;
          row "that sum closed" [ 1 ] (Nx.Placement.closed summed);
          row "that sum uncut" [] (Nx.Placement.uncut summed);
          row "a split uncut" [] (Nx.Placement.uncut (S4.split ~axis:0));
        ]
        (fun (_, expected, axes) -> equal (list int) expected (axes ()));
      test "placements of equal cut_axes need not be equal" (fun () ->
          let over name = Nx.Placement.mesh grid [ (0, [ name ]) ] in
          equal (list int)
            (Nx.Placement.cut_axes (over "a"))
            (Nx.Placement.cut_axes (over "b"));
          not_equal (placement ()) (over "a") (over "b"));
      prop ~count:400 "a mesh cuts each axis it cuts over two devices or more"
        meshes law_cut_axes;
      cases "pp formats a placement as nx.mli states"
        ~id:(fun (n, _, _) -> n)
        [
          ("the host", Rig.name Rig.host, fun () -> pp Nx.Host.on);
          ("on", "on [m0; m1]", fun () -> pp S2.on);
          ("a split", "split ~axis:1 [m0; m1]", fun () -> pp (S2.split ~axis:1));
          ("a set of one device", "m2", fun () -> pp S1.on);
          ("a split over one device", "m2", fun () -> pp (S1.split ~axis:0));
          ( "one device of a set",
            "m2",
            fun () -> pp (Nx.Placement.device S4.v (m 2)) );
        ]
        (fun (_, expected, printed) -> equal string expected (printed ()));
      test "pp formats a mesh's extents and cuts" (fun () ->
          expect (pp (Nx.Placement.mesh grid [ (0, [ "b" ]); (1, [ "a" ]) ]))
          @@ __POS_OF__ {|mesh 2x2 [m0; m1; m2; m3] ~axis:0/1 ~axis:1/0|});
    ]

(* The Devices module type and devices *)

let module_group =
  group "the Devices module type"
    [
      test "on and split are Placement's over v" (fun () ->
          equal (placement ()) (Nx.Placement.on S4.v) S4.on;
          equal (placement ())
            (Nx.Placement.split ~axis:1 S4.v)
            (S4.split ~axis:1));
    ]

let mint_group =
  group "devices"
    [
      cases
        "devices raises for no device, a repeated one, or kernels that do not \
         compute on one"
        ~id:fst
        [
          ("no device", fun () -> ignore (Nx.devices []));
          ("a repeated device", fun () -> ignore (Nx.devices [ m 0; m 1; m 0 ]));
          ( "kernels that do not compute on a device",
            fun () -> ignore (Nx.devices ~kernels:(module Nowhere) [ m 0 ]) );
        ]
        (fun (_, f) -> refuses ~by:"Nx.devices" f);
    ]

(* Prim.coerce: a dtype and a set are each checked. Two mints of one device list
   are two sets. *)
let coercions =
  let on2 = Nx.place S2.on (Nx.zeros Nx.float32 [| 2 |]) in
  let coerce dt s x =
    Option.map (fun y -> Nx.Dtype.name (Nx.dtype y)) (Nx.Prim.coerce dt s x)
  in
  let some ~msg dt s x =
    equal ~msg (option string) (Some (Nx.Dtype.name dt)) (coerce dt s x)
  in
  let none ~msg dt s x = equal ~msg (option string) None (coerce dt s x) in
  group "Prim.coerce"
    [
      test "a value coerces to its own dtype and set" (fun () ->
          some ~msg:"placed" Nx.float32 S2.v on2;
          some ~msg:"a host creation" Nx.float32 Nx.Host.v
            (Nx.zeros Nx.float32 [| 2 |]));
      test "any placement of the set, at either name of its brand" (fun () ->
          some ~msg:"split" Nx.float32 S2.v
            (Nx.place
               (Nx.Placement.split ~axis:0 S2.v)
               (Nx.zeros Nx.float32 [| 2 |]));
          some ~msg:"one set unpacked twice" Nx.float32 Named.v
            (Nx.place Renamed.on (Nx.zeros Nx.float32 [| 2 |])));
      test "another set or another dtype fails" (fun () ->
          let twin = Nx.place Twin.on (Nx.zeros Nx.float32 [| 2 |]) in
          none ~msg:"a twin set of the same devices" Nx.float32 S2.v twin;
          none ~msg:"another dtype" Nx.float32 S2.v
            (Nx.place S2.on (Nx.zeros Nx.float64 [| 2 |]));
          none ~msg:"a host creation where a set is asked" Nx.float32 S2.v
            (Nx.zeros Nx.float32 [| 2 |]);
          none ~msg:"a placed value where the host is asked" Nx.float32
            Nx.Host.v on2);
    ]

let () =
  exit
    (run "nx devices"
       [
         devices_group;
         mesh_group;
         placement_group;
         module_group;
         mint_group;
         coercions;
       ])
