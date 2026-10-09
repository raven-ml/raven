(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values from arrays and arrays from values, through Nx and Nx.Repr. *)

open Windtrap
module A = Nx_array

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])
module Other = (val Nx.devices [ m 2 ])

let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let floats shape = Array.init (Array.fold_left ( * ) 1 shape) float_of_int
let on d shape = A.to_device d (A.of_array A.Dtype.Float32 shape (floats shape))

let same a b =
  A.buffer a == A.buffer b && A.Layout.equal (A.layout a) (A.layout b)

let shapes =
  Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4)
  |> Gen.with_pp (fun ppf s ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_int s))))

let values =
  group "values"
    [
      prop "a value from an array has the array's shape and dtype" shapes
        (fun s ->
          cover "no element" (Array.exists (( = ) 0) s);
          cover "rank 0" (s = [||]);
          let x =
            Nx.Repr.of_array Nx.Host.v (A.of_array A.Dtype.Float32 s (floats s))
          in
          equal (array int) s (Nx.shape x);
          equal bool true (A.Dtype.equal A.Dtype.Float32 (Nx.dtype x)));
      test "shape is a fresh array" (fun () ->
          let x = Nx.Repr.of_array Nx.Host.v (on Rig.host [| 2; 3 |]) in
          (Nx.shape x).(0) <- 7;
          equal (array int) [| 2; 3 |] (Nx.shape x));
      test "a value lies on its array's device" (fun () ->
          let x = Nx.Repr.of_array S2.v (on (m 1) [| 4 |]) in
          equal string "m1"
            (Format.asprintf "%a" Nx.Placement.pp (Option.get (Nx.placement x))));
      test "two values on one device share its placement" (fun () ->
          let x = Nx.Repr.of_array S2.v (on (m 1) [| 4 |])
          and y = Nx.Repr.of_array S2.v (on (m 1) [| 2 |]) in
          equal bool true
            (Option.get (Nx.placement x) == Option.get (Nx.placement y)));
      test "a sharded value's shape is the whole's" (fun () ->
          let x =
            Nx.Repr.of_shards (S2.split ~axis:1)
              [| on (m 0) [| 3; 2 |]; on (m 1) [| 3; 2 |] |]
          in
          equal (array int) [| 3; 4 |] (Nx.shape x));
    ]

let repr =
  group "repr"
    [
      test "array of a value from an array is that array" (fun () ->
          let a = on (m 0) [| 2; 2 |] in
          let b = require_some (Nx.Repr.array (Nx.Repr.of_array S2.v a)) in
          equal bool true (same a b));
      test "shards of a value from shards are those arrays" (fun () ->
          let a = on (m 0) [| 2 |] and b = on (m 1) [| 2 |] in
          let s =
            require_some (Nx.Repr.shards (Nx.Repr.of_shards S2.on [| a; b |]))
          in
          equal bool true (same a s.(0) && same b s.(1)));
      test "a value on two devices has no single array" (fun () ->
          let x =
            Nx.Repr.of_shards S2.on [| on (m 0) [| 2 |]; on (m 1) [| 2 |] |]
          in
          is_none (Nx.Repr.array x));
      test "shards of a value on one device is its array" (fun () ->
          let a = on (m 0) [| 2 |] in
          let s = require_some (Nx.Repr.shards (Nx.Repr.of_array S2.v a)) in
          equal int 1 (Array.length s);
          equal bool true (same a s.(0)));
      test "shards on one device of a placement make an array value" (fun () ->
          let module S1 = (val Nx.devices [ m 3 ]) in
          let a = on (m 3) [| 2 |] in
          let x = Nx.Repr.of_shards S1.on [| a |] in
          equal bool true (same a (require_some (Nx.Repr.array x))));
      test "an array on another set's device is refused" (fun () ->
          invalid ~by:"Nx.Repr.of_array" (fun () ->
              Nx.Repr.of_array Other.v (on (m 0) [| 2 |])));
      cases "of_shards refuses" ~name:fst
        [
          ( "one array for two devices",
            fun () -> Nx.Repr.of_shards S2.on [| on (m 0) [| 2 |] |] );
          ( "an array on another device than its place",
            fun () ->
              Nx.Repr.of_shards S2.on [| on (m 1) [| 2 |]; on (m 1) [| 2 |] |]
          );
          ( "arrays of two shapes",
            fun () ->
              Nx.Repr.of_shards S2.on [| on (m 0) [| 2 |]; on (m 1) [| 3 |] |]
          );
          ( "a cut axis the arrays lack",
            fun () ->
              Nx.Repr.of_shards (S2.split ~axis:1)
                [| on (m 0) [| 2 |]; on (m 1) [| 2 |] |] );
        ]
        (fun (_, f) -> invalid ~by:"Nx.Repr.of_shards" f);
      test "placement of a value from shards is the placement given" (fun () ->
          let x =
            Nx.Repr.of_shards (S2.split ~axis:0)
              [| on (m 0) [| 2 |]; on (m 1) [| 2 |] |]
          in
          equal placement (S2.split ~axis:0) (Option.get (Nx.placement x)));
    ]

(* The one message format: every refusal of the interface, called wrongly. *)
let errors =
  let mesh = Nx.Mesh.v S2.v [ ("a", 2) ] in
  cases "every refusal names the function called, then a reason" ~name:fst
    [
      ("Nx.devices", fun () -> ignore (Nx.devices []));
      ("Nx.Mesh.v", fun () -> ignore (Nx.Mesh.v S2.v [ ("a", 3) ]));
      ("Nx.Placement.split", fun () -> ignore (S2.split ~axis:(-1)));
      ( "Nx.Placement.mesh",
        fun () -> ignore (Nx.Placement.mesh mesh [ (0, [ "z" ]) ]) );
      ( "Nx.place",
        fun () ->
          ignore
            (Nx.place (S2.split ~axis:0)
               (Nx.Repr.of_array Nx.Host.v (on Rig.host [| 3 |]))) );
      ( "Nx.Repr.of_array",
        fun () -> ignore (Nx.Repr.of_array Other.v (on (m 0) [| 2 |])) );
      ( "Nx.Repr.of_shards",
        fun () -> ignore (Nx.Repr.of_shards S2.on [| on (m 0) [| 2 |] |]) );
    ]
    (fun (name, f) ->
      match f () with
      | () -> fail (name ^ " raised nothing")
      | exception Invalid_argument msg ->
          starts_with ~affix:(name ^ ": ") msg;
          greater int ~than:(String.length name + 2) (String.length msg))

(* Values of every set have no placement, and a value placed on a set lies
   there: a formula never names a set (the external review's program, which no
   longer type-checks, read a host set at another brand from one). *)

module Host2 = (val Nx.devices [ Rig.host ])

let f32 = Nx.float32

let test_every_set_unplaced () =
  let c = Nx.zeros f32 [| 2 |] in
  let none ~msg x = equal ~msg bool true (Nx.placement x = None) in
  none ~msg:"zeros" c;
  none ~msg:"scalar" (Nx.scalar f32 1.);
  none ~msg:"an operation over formulas" (Nx.add c c);
  none ~msg:"a movement of one" (Nx.reshape [| 1; 2 |] c)

let test_two_values_of_a_host_set () =
  let x = Nx.place Host2.on (Nx.zeros f32 [| 2 |]) in
  let y = Nx.place Host2.on (Nx.zeros f32 [| 2 |]) in
  let z = Nx.add x y in
  equal ~msg:"lies on its set" (list string)
    [ Rig.name Rig.host ]
    (List.map Rig.name
       (Nx.rigs (Nx.Placement.devices (Option.get (Nx.placement z)))))

(* A placement over a set of two memory devices, drawn. *)
let placements =
  Gen.of_list
    ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
    [
      ("on", S2.on);
      ("split 0", S2.split ~axis:0);
      ("first", Nx.Placement.mesh (Nx.Mesh.v S2.v [ ("x", 2) ]) []);
    ]

let law_placed ((_, p), shape) =
  let shape = Array.map (fun n -> 2 * n) shape in
  assume (Array.length shape > 0);
  let x = Nx.place p (Nx.zeros f32 shape) in
  let q = Option.get (Nx.placement x) in
  equal placement p q;
  equal (list string)
    (List.map Rig.name (Nx.rigs (Nx.Placement.devices p)))
    (List.map Rig.name (Nx.rigs (Nx.Placement.devices q)))

(* A formula has no bytes for Repr to hand out. *)
let test_repr_of_formula () =
  let c = Nx.add (Nx.zeros f32 [| 2 |]) (Nx.scalar f32 1.) in
  equal ~msg:"array" bool true (Nx.Repr.array c = None);
  equal ~msg:"shards" bool true (Nx.Repr.shards c = None)

let every_set =
  group "every set"
    [
      test "Repr reads no array of a value of every set" test_repr_of_formula;
      test "a value of every set has no placement" test_every_set_unplaced;
      test "values placed on a set over the host meet"
        test_two_values_of_a_host_set;
      prop "a placed value lies where it was placed"
        (Gen.pair placements shapes)
        law_placed;
    ]

let () = exit (run "nx values" [ values; repr; errors; every_set ])
