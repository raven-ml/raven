(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values from arrays and arrays from values, through Nx and Nx.Repr. *)

open Windtrap
open Nx_support
module A = Nx_array

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])
module Other = (val Nx.devices [ m 2 ])

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let floats shape = Array.init (Array.fold_left ( * ) 1 shape) float_of_int

let on d shape =
  A.to_device d
    (A.of_array A.Dtype.Float32 (Iarray.of_array shape) (floats shape))

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
            Nx.Repr.of_array Nx.Host.v
              (A.of_array A.Dtype.Float32 (Iarray.of_array s) (floats s))
          in
          equal (array int) s (Nx.shape x);
          equal bool true (A.Dtype.equal A.Dtype.Float32 (Nx.dtype x)));
      test "Nx's element types are Nx.Dtype's" (fun () ->
          (* Compiling is the check: each function's annotation names one
             module's element type, its argument the other's. *)
          let f32 (x : (float, Nx.float32_elt, Nx.host) Nx.t) = Nx.shape x in
          let x : (float, Nx.Dtype.float32_elt, Nx.host) Nx.t =
            Nx.zeros Nx.float32 [| 2 |]
          in
          equal ~msg:"float32" (array int) [| 2 |] (f32 x);
          let bf16 (x : (float, Nx.Dtype.bfloat16_elt, Nx.host) Nx.t) =
            Nx.shape x
          in
          let x : (float, Nx.bfloat16_elt, Nx.host) Nx.t =
            Nx.zeros Nx.bfloat16 [| 3 |]
          in
          equal ~msg:"bfloat16" (array int) [| 3 |] (bf16 x);
          let i32 (x : (int32, Nx.int32_elt, Nx.host) Nx.t) = Nx.shape x in
          let x : Nx.host Nx.int32_t = Nx.zeros Nx.int32 [| 4 |] in
          equal ~msg:"int32" (array int) [| 4 |] (i32 x);
          let u8 (dt : (int, Nx.int8_unsigned_elt) Nx.dtype) =
            Nx.Dtype.name dt
          in
          equal ~msg:"uint8" string "uint8" (u8 Nx.Dtype.Uint8);
          let b (dt : (bool, Nx.Dtype.bool_elt) Nx.dtype) = Nx.Dtype.name dt in
          equal ~msg:"bool" string "bool"
            (b (Nx.bool : (bool, Nx.bool_elt) Nx.dtype)));
      test "shape is a fresh array" (fun () ->
          let x = Nx.Repr.of_array Nx.Host.v (on Rig.host [| 2; 3 |]) in
          (Nx.shape x).(0) <- 7;
          equal (array int) [| 2; 3 |] (Nx.shape x));
      test "a value lies on its array's device" (fun () ->
          let x = Nx.Repr.of_array S2.v (on (m 1) [| 4 |]) in
          equal string "m1"
            (Format.asprintf "%a" Nx.Placement.pp (Nx.placement x)));
      test "two values on one device share its placement" (fun () ->
          let x = Nx.Repr.of_array S2.v (on (m 1) [| 4 |])
          and y = Nx.Repr.of_array S2.v (on (m 1) [| 2 |]) in
          equal bool true
            (Nx.placement x == Nx.placement y));
      test "a sharded value's shape is the whole's" (fun () ->
          let x =
            Nx.Repr.of_shards (S2.split ~axis:1)
              (Iarray.of_list [ on (m 0) [| 3; 2 |]; on (m 1) [| 3; 2 |] ])
          in
          equal (array int) [| 3; 4 |] (Nx.shape x));
      test "a sharded value's cut extent is the sum of its windows'" (fun () ->
          let shape a b =
            Nx.shape
              (Nx.Repr.of_shards (S2.split ~axis:1)
                 (Iarray.of_list [ on (m 0) [| 3; a |]; on (m 1) [| 3; b |] ]))
          in
          equal (array int) [| 3; 3 |] (shape 2 1);
          equal (array int) [| 3; 1 |] (shape 1 0);
          equal (array int) [| 3; 0 |] (shape 0 0));
    ]

module S4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh4 = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ]

(* Placements over four memory devices of a value of rank [rank]: a split, a
   mesh cutting one or two axes, and a cut over the minor mesh axis, whose
   major axis holds copies. *)
let placement_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  let open Gen in
  one_of
    [
      map (fun a -> (Printf.sprintf "split %d" a, S4.split ~axis:a)) axis;
      map
        (fun (x, y) ->
          if x = y then
            ( Printf.sprintf "cut %d/a,b" x,
              Nx.Placement.mesh mesh4 [ (x, [ "a"; "b" ]) ] )
          else
            ( Printf.sprintf "cuts %d/a %d/b" x y,
              Nx.Placement.mesh mesh4 [ (x, [ "a" ]); (y, [ "b" ]) ] ))
        (pair axis axis);
      map
        (fun x ->
          ( Printf.sprintf "cut %d/b" x,
            Nx.Placement.mesh mesh4 [ (x, [ "b" ]) ] ))
        axis;
    ]

let shards_case =
  let open Gen in
  with_pp
    (fun ppf (s, (name, _)) ->
      Format.fprintf ppf "[%s] at %s"
        (String.concat "; " (Array.to_list (Array.map string_of_int s)))
        name)
    (let* rank = int_range 1 3 in
     pair (array ~size:(constant rank) (int_range 0 6)) (placement_of rank))

(* A host value placed at [p], its shards made a value again at [p]: the same
   shape and elements. *)
let law_shards (s, (_, p)) =
  cover "an odd extent" (Array.exists (fun n -> n mod 2 = 1) s);
  cover "an empty window" (Array.exists (fun n -> n > 0 && n < 4) s);
  let x = Nx.Repr.of_array Nx.Host.v (on Rig.host s) in
  let y = Nx.Repr.of_shards p ((Nx.Repr.shards (Nx.place p x))) in
  equal (array int) ~msg:"shape" s (Nx.shape y);
  equal (array float_exact) ~msg:"elements" (floats s)
    (A.to_array (require_some (Nx.Repr.array (Nx.place Nx.Host.on y))))

let repr =
  group "repr"
    [
      test "array of a value from an array is that array" (fun () ->
          let a = on (m 0) [| 2; 2 |] in
          let b = require_some (Nx.Repr.array (Nx.Repr.of_array S2.v a)) in
          equal bool true (same a b));
      test "shards of a value from shards are those arrays" (fun () ->
          let a = on (m 0) [| 2 |] and b = on (m 1) [| 2 |] in
          let x = Nx.Repr.of_shards S2.on (Iarray.of_list [ a; b ]) in
          let s = (Nx.Repr.shards x) in
          equal bool true (same a (Iarray.get s 0) && same b (Iarray.get s 1)));
      test "a value on two devices has no single array" (fun () ->
          let x =
            Nx.Repr.of_shards S2.on
              (Iarray.of_list [ on (m 0) [| 2 |]; on (m 1) [| 2 |] ])
          in
          is_none (Nx.Repr.array x));
      test "shards of a value on one device is its array" (fun () ->
          let a = on (m 0) [| 2 |] in
          let s = (Nx.Repr.shards (Nx.Repr.of_array S2.v a)) in
          equal int 1 (Iarray.length s);
          equal bool true (same a (Iarray.get s 0)));
      test "shards on one device of a placement make an array value" (fun () ->
          let module S1 = (val Nx.devices [ m 3 ]) in
          let a = on (m 3) [| 2 |] in
          let x = Nx.Repr.of_shards S1.on (Iarray.of_list [ a ]) in
          equal bool true (same a (require_some (Nx.Repr.array x))));
      test "an array on another set's device is refused" (fun () ->
          invalid ~by:"Nx.Repr.of_array" (fun () ->
              Nx.Repr.of_array Other.v (on (m 0) [| 2 |])));
      cases "of_shards refuses" ~name:fst
        [
          ( "one array for two devices",
            fun () ->
              Nx.Repr.of_shards S2.on (Iarray.of_list [ on (m 0) [| 2 |] ]) );
          ( "an array on another device than its place",
            fun () ->
              Nx.Repr.of_shards S2.on
                (Iarray.of_list [ on (m 1) [| 2 |]; on (m 1) [| 2 |] ])
          );
          ( "arrays of two shapes",
            fun () ->
              Nx.Repr.of_shards S2.on
                (Iarray.of_list [ on (m 0) [| 2 |]; on (m 1) [| 3 |] ])
          );
          ( "a cut axis the arrays lack",
            fun () ->
              Nx.Repr.of_shards (S2.split ~axis:1)
                (Iarray.of_list [ on (m 0) [| 2 |]; on (m 1) [| 2 |] ]) );
          ( "windows of no one shape along the cut axis",
            fun () ->
              Nx.Repr.of_shards (S2.split ~axis:0)
                (Iarray.of_list [ on (m 0) [| 1 |]; on (m 1) [| 2 |] ]) );
          ( "an empty window before a full one",
            fun () ->
              Nx.Repr.of_shards (S2.split ~axis:0)
                (Iarray.of_list [ on (m 0) [| 0 |]; on (m 1) [| 1 |] ]) );
          ( "windows of two extents along an axis no cut names",
            fun () ->
              Nx.Repr.of_shards (S2.split ~axis:0)
                (Iarray.of_list [ on (m 0) [| 2; 3 |]; on (m 1) [| 2; 4 |] ])
          );
          ( "arrays of two ranks",
            fun () ->
              Nx.Repr.of_shards (S2.split ~axis:0)
                (Iarray.of_list [ on (m 0) [| 2 |]; on (m 1) [| 2; 1 |] ]) );
        ]
        (fun (_, f) -> invalid ~by:"Nx.Repr.of_shards" f);
      test "of_shards refuses copies of two extents" (fun () ->
          invalid ~by:"Nx.Repr.of_shards" (fun () ->
              Nx.Repr.of_shards
                (Nx.Placement.mesh mesh4 [ (0, [ "b" ]) ])
                (Iarray.of_list
                   [
                     on (m 0) [| 2 |];
                     on (m 1) [| 1 |];
                     on (m 2) [| 1 |];
                     on (m 3) [| 1 |];
                   ])));
      prop "shards made a value again give it back" shards_case law_shards;
      test "placement of a value from shards is the placement given" (fun () ->
          let x =
            Nx.Repr.of_shards (S2.split ~axis:0)
              (Iarray.of_list [ on (m 0) [| 2 |]; on (m 1) [| 2 |] ])
          in
          equal (placement ()) (S2.split ~axis:0) (Nx.placement x));
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
            (Nx.place (S2.split ~axis:1)
               (Nx.Repr.of_array Nx.Host.v (on Rig.host [| 3 |]))) );
      ( "Nx.Repr.of_array",
        fun () -> ignore (Nx.Repr.of_array Other.v (on (m 0) [| 2 |])) );
      ( "Nx.Repr.of_shards",
        fun () ->
          ignore (Nx.Repr.of_shards S2.on (Iarray.of_list [ on (m 0) [| 2 |] ]))
      );
    ]
    (fun (name, f) ->
      match f () with
      | () -> fail (name ^ " raised nothing")
      | exception Invalid_argument msg ->
          starts_with ~affix:(name ^ ": ") msg;
          greater int ~than:(String.length name + 2) (String.length msg))

(* Values on sets: two over the host meet, and Repr reads a creation's bytes. *)

module Host2 = (val Nx.devices [ Rig.host ])

let f32 = Nx.float32

let test_two_values_of_a_host_set () =
  let x = Nx.place Host2.on (Nx.zeros f32 [| 2 |]) in
  let y = Nx.place Host2.on (Nx.zeros f32 [| 2 |]) in
  let z = Nx.add x y in
  equal ~msg:"lies on its set" (list string)
    [ Rig.name Rig.host ]
    (List.map Rig.name
       (Nx.rigs (Nx.Placement.devices (Nx.placement z))))

(* A creation holds bytes from the call on, which Repr hands out. *)
let test_repr_of_creation () =
  let c = Nx.add (Nx.zeros f32 [| 2 |]) (Nx.full f32 [| 2 |] 1.) in
  equal ~msg:"array" (option (array float_exact)) (Some [| 1.; 1. |])
    (Option.map A.to_array (Nx.Repr.array c));
  equal ~msg:"shards" int 1 (Iarray.length (Nx.Repr.shards c))

let every_set =
  group "placements"
    [
      test "Repr reads a creation's bytes" test_repr_of_creation;
      test "values placed on a set over the host meet"
        test_two_values_of_a_host_set;
    ]

let () = exit (run "nx values" [ values; repr; errors; every_set ])
