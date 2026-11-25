(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values: a value's shape and dtype and the bounds of a shape, reads of a value
   past its interpretation, nx's message for a misuse, and the crossing between
   values and arrays (Nx.Repr). *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])
module S4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])
module Other = (val Nx.devices [ m 2 ])

let f32 = Nx.float32

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

let floats shape = Array.init (Array.fold_left ( * ) 1 shape) float_of_int

(* A float32 array of [shape] on device [d], its elements 0, 1, 2, ... *)
let on d shape =
  A.to_device d (A.of_array D.Float32 (Iarray.of_array shape) (floats shape))

(* Arrays equal where they are one buffer at one layout. *)
let memory () =
  Testable.make
    ~pp:(fun ppf a ->
      Format.fprintf ppf "%s %a" (Rig.name (A.device a)) L.pp (A.layout a))
    ~equal:(fun a b ->
      A.buffer a == A.buffer b && L.equal (A.layout a) (A.layout b))

let layout = Testable.make ~pp:L.pp ~equal:L.equal

let shapes =
  Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4) |> Gen.with_pp pp_ints

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let mark i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

(* A value of the interpretation [test.values], kept past its return. *)
let escaped () =
  let x = Nx.create f32 [| 2 |] [| 1.; 2. |] in
  Nx.Prim.interpret ~name:"test.values" Values mark (fun i ->
      Nx.Prim.traced i (Nx.Prim.form x) Mark)

(* Values *)

let law_from_array (D.Any dt, s) =
  cover "no element" (Array.exists (( = ) 0) s);
  cover "rank 0" (s = [||]);
  cover "a sub-byte dtype" (D.bits dt < 8);
  let x =
    Nx.Repr.of_array Nx.Host.v (A.create Rig.host dt (Iarray.of_array s))
  in
  equal (array int) ~msg:"shape" s (Nx.shape x);
  equal string ~msg:"dtype" (D.name dt) (D.name (Nx.dtype x))

let bounds =
  let r = L.max_rank in
  [
    ( "more than max_rank axes",
      "Nx.zeros",
      fun () -> Nx.zeros f32 (Array.make (r + 1) 1) );
    ("a negative extent", "Nx.zeros", fun () -> Nx.zeros f32 [| -1 |]);
    ( "extents that multiply past max_numel",
      "Nx.zeros",
      fun () -> Nx.zeros f32 [| 1 lsl 27; 1 lsl 27 |] );
    ( "an extent past max_numel beside a 0",
      "Nx.zeros",
      fun () -> Nx.zeros f32 [| 0; L.max_numel + 1 |] );
    ( "an axis added to a value of max_rank axes",
      "Nx.unsqueeze",
      fun () -> Nx.unsqueeze ~axes:[ 0 ] (Nx.zeros f32 (Array.make r 1)) );
  ]

let test_at_bounds () =
  let most = Array.make L.max_rank 1 in
  equal (array int) ~msg:"max_rank axes" most (Nx.shape (Nx.zeros f32 most));
  equal (array int) ~msg:"max_numel beside a 0" [| 0; L.max_numel |]
    (Nx.shape (Nx.zeros f32 [| 0; L.max_numel |]))

let values =
  group "values"
    [
      prop ~count:200 "a value from an array has the array's shape and dtype"
        Gen.(
          pair (of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all) shapes)
        law_from_array;
      test "shape is a fresh array" (fun () ->
          let x = Nx.Repr.of_array Nx.Host.v (on Rig.host [| 2; 3 |]) in
          (Nx.shape x).(0) <- 7;
          equal (array int) [| 2; 3 |] (Nx.shape x));
      cases "a shape past the bounds raises naming the function"
        ~id:(fun (n, _, _) -> n)
        bounds
        (fun (_, by, f) -> refuses ~by f);
      test "a shape at the bounds is made" test_at_bounds;
    ]

(* Reads *)

let reads =
  group "reads"
    [
      cases "a read of a value whose interpretation returned raises naming it"
        ~id:fst
        [
          ("Nx.to_array", fun x -> ignore (Nx.to_array x));
          ("Nx.item", fun x -> ignore (Nx.item [ 0 ] x));
          ("Nx.to_bigarray", fun x -> ignore (Nx.to_bigarray Bigarray.float32 x));
          ( "test.reader",
            fun x -> Nx.Repr.read ~by:"test.reader" x (fun _ -> ()) );
        ]
        (fun (by, read) ->
          let t = escaped () in
          raises
            (Invalid_argument
               (by ^ ": a value of test.values, used after test.values returned"))
            (fun () -> read t));
      test "to_bigarray raises past 16 axes" (fun () ->
          let b =
            Nx.to_bigarray Bigarray.float32 (Nx.ones f32 (Array.make 16 1))
          in
          equal int ~msg:"16 axes" 16 (Bigarray.Genarray.num_dims b);
          refuses ~by:"Nx.to_bigarray" (fun () ->
              Nx.to_bigarray Bigarray.float32 (Nx.ones f32 (Array.make 17 1))));
      test "Host.read passes on what f raises" (fun () ->
          raises Exit (fun () ->
              Nx.Host.read (Nx.ones f32 [| 2 |]) (fun _ -> raise Exit)));
    ]

(* Errors *)

let errors =
  group "errors"
    [
      test "a misuse raises naming the function called, then a reason"
        (fun () ->
          refuses ~by:"Nx.place" (fun () ->
              Nx.place (S2.split ~axis:1)
                (Nx.Repr.of_array Nx.Host.v (on Rig.host [| 3 |]))));
    ]

(* Repr *)

let test_of_array () =
  let a = on (m 1) [| 4 |] in
  let x = Nx.Repr.of_array S2.v a in
  equal (placement ()) ~msg:"placement"
    (Nx.Placement.device S2.v (m 1))
    (Nx.placement x);
  equal (memory ()) ~msg:"bytes" a (require_some (Nx.Repr.array ~by:"test" x))

let test_array () =
  let module S1 = (val Nx.devices [ m 3 ]) in
  let a = on (m 0) [| 2 |] and b = on (m 1) [| 2 |] and c = on (m 3) [| 2 |] in
  let single x = Nx.Repr.array ~by:"test" x in
  equal ~msg:"from an array"
    (option (memory ()))
    (Some a)
    (single (Nx.Repr.of_array S2.v a));
  equal ~msg:"from shards on one device"
    (option (memory ()))
    (Some c)
    (single (Nx.Repr.of_shards S1.on (Iarray.of_list [ c ])));
  equal ~msg:"a value computed on the host"
    (option (array float_exact))
    (Some [| 1.; 1. |])
    (Option.map A.to_array
       (single (Nx.add (Nx.zeros f32 [| 2 |]) (Nx.full f32 [| 2 |] 1.))));
  is_none ~msg:"on two devices"
    (single (Nx.Repr.of_shards S2.on (Iarray.of_list [ a; b ])))

let test_read () =
  let a = on Rig.host [| 3 |] in
  Nx.Repr.read ~by:"test" (Nx.Repr.of_array Nx.Host.v a) (fun r ->
      equal (memory ()) ~msg:"over the host value's memory" a r);
  let off = Nx_array_support.off_host () in
  let module Off = (val Nx.devices [ off ]) in
  let y = Nx.place Off.on (Nx.create f32 [| 3 |] [| 0.; 1.; 2. |]) in
  Nx.Repr.read ~by:"test" y (fun r ->
      equal string ~msg:"a copy on the host" (Rig.name Rig.host)
        (Rig.name (A.device r));
      equal (array float_exact) ~msg:"its elements" [| 0.; 1.; 2. |]
        (A.to_array r))

let mesh4 = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ]

let not_windows =
  [
    ( "one array for two devices",
      "Nx.Repr.of_shards",
      fun () ->
        ignore @@ Nx.Repr.of_shards S2.on (Iarray.of_list [ on (m 0) [| 2 |] ])
    );
    ( "an array on another device than its place",
      "Nx.Repr.of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.of_shards S2.on
             (Iarray.of_list [ on (m 1) [| 2 |]; on (m 1) [| 2 |] ]) );
    ( "arrays of two shapes",
      "Nx.Repr.of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.of_shards S2.on
             (Iarray.of_list [ on (m 0) [| 2 |]; on (m 1) [| 3 |] ]) );
    ( "a cut axis the arrays lack",
      "Nx.Repr.of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.of_shards (S2.split ~axis:1)
             (Iarray.of_list [ on (m 0) [| 2 |]; on (m 1) [| 2 |] ]) );
    ( "windows of no one shape along the cut axis",
      "Nx.Repr.of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.of_shards (S2.split ~axis:0)
             (Iarray.of_list [ on (m 0) [| 1 |]; on (m 1) [| 2 |] ]) );
    ( "an empty window before a full one",
      "Nx.Repr.of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.of_shards (S2.split ~axis:0)
             (Iarray.of_list [ on (m 0) [| 0 |]; on (m 1) [| 1 |] ]) );
    ( "windows of two extents along an axis no cut names",
      "Nx.Repr.of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.of_shards (S2.split ~axis:0)
             (Iarray.of_list [ on (m 0) [| 2; 3 |]; on (m 1) [| 2; 4 |] ]) );
    ( "arrays of two ranks",
      "Nx.Repr.of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.of_shards (S2.split ~axis:0)
             (Iarray.of_list [ on (m 0) [| 2 |]; on (m 1) [| 2; 1 |] ]) );
    ( "copies of two extents",
      "Nx.Repr.of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.of_shards
             (Nx.Placement.mesh mesh4 [ (0, [ "b" ]) ])
             (Iarray.of_list
                [
                  on (m 0) [| 2 |];
                  on (m 1) [| 1 |];
                  on (m 2) [| 1 |];
                  on (m 3) [| 1 |];
                ]) );
    ( "unsafe_of_shards, one array for two devices",
      "Nx.Repr.unsafe_of_shards",
      fun () ->
        ignore
        @@ Nx.Repr.unsafe_of_shards S2.on (Iarray.of_list [ on (m 0) [| 2 |] ])
    );
  ]

(* Placements over four software devices of a value of rank [rank]: a split, a
   mesh cutting one or two axes, and a cut over the minor mesh axis, whose major
   axis holds copies. *)
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
          (Printf.sprintf "cut %d/b" x, Nx.Placement.mesh mesh4 [ (x, [ "b" ]) ]))
        axis;
    ]

let shards_case =
  let open Gen in
  with_pp
    (fun ppf (s, (name, _)) -> Format.fprintf ppf "%a at %s" pp_ints s name)
    (let* rank = int_range 1 3 in
     pair (array ~size:(constant rank) (int_range 0 6)) (placement_of rank))

(* A host value placed at [p], its shards made a value again at [p]: the same
   shape, placement, layouts and elements. *)
let law_shards (s, (_, p)) =
  let x = Nx.Repr.of_array Nx.Host.v (on Rig.host s) in
  let shards = Nx.Repr.shards ~by:"test" (Nx.place p x) in
  let empty a = Iarray.exists (( = ) 0) (L.shape (A.layout a)) in
  cover "an odd extent" (Array.exists (fun n -> n mod 2 = 1) s);
  cover "an empty window of a value with elements"
    (Array.for_all (( <> ) 0) s && Iarray.exists empty shards);
  let y = Nx.Repr.of_shards p shards in
  equal (array int) ~msg:"shape" s (Nx.shape y);
  equal (placement ()) ~msg:"placement" p (Nx.placement y);
  equal (list layout) ~msg:"layouts"
    (Iarray.to_list (Iarray.map A.layout shards))
    (Iarray.to_list (Nx.Repr.layouts y));
  equal (array float_exact) ~msg:"elements" (floats s)
    (Nx.Repr.read ~by:"test" y A.to_array)

let test_shards () =
  let a = on (m 0) [| 2 |] and b = on (m 1) [| 2 |] in
  let shards x = Iarray.to_list (Nx.Repr.shards ~by:"test" x) in
  equal ~msg:"from shards"
    (list (memory ()))
    [ a; b ]
    (shards (Nx.Repr.of_shards S2.on (Iarray.of_list [ a; b ])));
  equal ~msg:"on one device"
    (list (memory ()))
    [ a ]
    (shards (Nx.Repr.of_array S2.v a))

let repr =
  group "repr"
    [
      test "a value from an array lies on its device, over its bytes"
        test_of_array;
      cases "of_array and unsafe_of_array raise for an array off the set"
        ~id:fst
        [
          ( "Nx.Repr.of_array",
            fun () -> Nx.Repr.of_array Other.v (on (m 0) [| 2 |]) );
          ( "Nx.Repr.unsafe_of_array",
            fun () -> Nx.Repr.unsafe_of_array Other.v (on (m 0) [| 2 |]) );
        ]
        (fun (by, f) -> refuses ~by f);
      test "array is Some iff the value lies on one device" test_array;
      test
        "read is over a value's own memory where the host addresses it, else a \
         copy on the host"
        test_read;
      cases "of_shards raises unless each array is its device's window"
        ~id:(fun (n, _, _) -> n)
        not_windows
        (fun (_, by, f) -> refuses ~by f);
      prop
        "a value placed, its shards made a value again at its placement, is \
         the value"
        shards_case law_shards;
      test "shards are a value's arrays in device order" test_shards;
      test "layouts of a value on one device are its form's" (fun () ->
          let x = Nx.matrix_transpose (Nx.ones f32 [| 2; 3 |]) in
          equal (list layout)
            [ (Nx.Prim.form x).layout ]
            (Iarray.to_list (Nx.Repr.layouts x)));
      test "a dead value keeps its layouts" (fun () ->
          let x = Nx.ones f32 [| 2; 3 |] in
          let before = Iarray.to_list (Nx.Repr.layouts x) in
          ignore (Nx.neg (Nx.donate x));
          equal (list layout) before (Iarray.to_list (Nx.Repr.layouts x)));
      test "layouts raises for a traced value" (fun () ->
          let x = Nx.ones f32 [| 2 |] in
          Nx.Prim.interpret ~name:"test" Values
            (fun _ ~by:_ _ -> invalid_arg "test: no operation")
            (fun i ->
              let t = Nx.Prim.traced i (Nx.Prim.form x) Mark in
              refuses ~by:"Nx.Repr.layouts" (fun () -> Nx.Repr.layouts t)));
    ]

let () = exit (run "nx values" [ values; reads; errors; repr ])
