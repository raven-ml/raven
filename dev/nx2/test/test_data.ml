(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Data in and out: values made from OCaml data, and the reads that give a
   value's elements back, at every dtype and brand. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

(* Elements *)

(* Two elements of [dt] are one value: floats by their bits, any two NaNs
   alike. *)
let same_float a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let same (type v s) (dt : (v, s) D.t) (a : v) (b : v) =
  match D.kind dt with
  | D.Float -> same_float a b
  | D.Complex -> same_float a.Complex.re b.Complex.re && same_float a.im b.im
  | D.Signed | D.Unsigned | D.Boolean -> a = b

let elements dt =
  Testable.make ~pp:(Format.pp_print_list (D.pp_value dt)) ~equal:(fun a b ->
      List.length a = List.length b && List.for_all2 (same dt) a b)

let equal_elements dt a b =
  equal (elements dt) (Array.to_list a) (Array.to_list b)

(* Values of [dt] at shape [s], from drawn bytes: every value of the format,
   NaNs, signed zeros and infinities among them. *)
let drawn (type v s) (dt : (v, s) D.t) s : v array Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  A.to_array
    (A.v dt (L.contiguous s)
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 4) ])

type case = Case : ('v, 's) D.t * int array * 'v array -> case

let cases_gen =
  let open Gen in
  let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
  let* s = shape in
  let+ vs = drawn dt s in
  Case (dt, s, vs)

let case =
  Gen.with_pp
    (fun ppf (Case (dt, s, _)) -> Format.fprintf ppf "%a %a" D.pp dt pp_ints s)
    cases_gen

let floats s = Array.init (Array.fold_left ( * ) 1 s) float_of_int
let f32 = Nx.float32

(* From OCaml values *)

(* A float is stored as [Dtype.of_float] says: float8_e5m2's infinities,
   which no store makes, saturate. *)
let stored (type v s) (dt : (v, s) D.t) (v : v) : v =
  match D.kind dt with D.Float -> D.of_float dt v | _ -> v

let law_round_trip (Case (dt, s, vs)) =
  cover "no element" (Array.exists (( = ) 0) s);
  cover "rank 0" (s = [||]);
  let x = Nx.create dt s vs in
  equal (array int) s (Nx.shape x);
  equal_elements dt (Array.map (stored dt) vs) (Nx.to_array x)

(* [item i x] is the element of [to_array x] at [i]'s position in C order,
   written with positive or negative positions. *)
let law_item (Case (dt, s, vs), (seed, negative)) =
  assume (Array.length vs > 0);
  let flat = seed mod Array.length vs in
  let i = Array.make (Array.length s) 0 and j = ref flat in
  for a = Array.length s - 1 downto 0 do
    i.(a) <- !j mod s.(a);
    j := !j / s.(a)
  done;
  let p = Array.mapi (fun a q -> if negative then q - s.(a) else q) i in
  cover "negative" negative;
  equal (elements dt) [ vs.(flat) ]
    [ Nx.item (Array.to_list p) (Nx.create dt s vs) ]

let test_init_order () =
  let calls = ref [] in
  let x =
    Nx.init Nx.int32 [| 2; 3 |] (fun i ->
        calls := Array.copy i :: !calls;
        Int32.of_int ((10 * i.(0)) + i.(1)))
  in
  equal (list (array int))
    [ [| 0; 0 |]; [| 0; 1 |]; [| 0; 2 |]; [| 1; 0 |]; [| 1; 1 |]; [| 1; 2 |] ]
    (List.rev !calls);
  equal (array int32) [| 0l; 1l; 2l; 10l; 11l; 12l |] (Nx.to_array x)

let test_init_fresh_index () =
  let kept = ref [] in
  let _ = Nx.init Nx.int8 [| 3 |] (fun i -> kept := i :: !kept; 0) in
  equal (list (array int)) [ [| 2 |]; [| 1 |]; [| 0 |] ] !kept

let test_unpack () =
  let x = Nx.create f32 [| 2 |] [| 1.; 2. |] in
  equal (array float_exact) [| 1.; 2. |] (Nx.to_array (Nx.unpack f32 (Nx.P x)));
  invalid ~by:"Nx.unpack" (fun () -> Nx.unpack Nx.int32 (Nx.P x))

let refusals =
  [
    ( "create: too few values",
      fun () -> ignore (Nx.create f32 [| 2; 2 |] [| 1.; 2.; 3. |]) );
    ( "create: a negative extent",
      fun () -> ignore (Nx.create f32 [| -1 |] [||]) );
    ( "create: an int outside the dtype",
      fun () -> ignore (Nx.create Nx.uint8 [| 1 |] [| 256 |]) );
    ( "create: an int below the dtype",
      fun () -> ignore (Nx.create Nx.int4 [| 1 |] [| -9 |]) );
    ("init: a negative extent", fun () -> ignore (Nx.init f32 [| -2 |] (fun _ -> 0.)));
    ( "item: too few positions",
      fun () -> ignore (Nx.item [ 0 ] (Nx.zeros f32 [| 2; 2 |])) );
    ( "item: a position past the axis",
      fun () -> ignore (Nx.item [ 2 ] (Nx.zeros f32 [| 2 |])) );
    ( "item: a position before the axis",
      fun () -> ignore (Nx.item [ -3 ] (Nx.zeros f32 [| 2 |])) );
    ( "item: an empty axis",
      fun () -> ignore (Nx.item [ 0 ] (Nx.zeros f32 [| 0 |])) );
    ( "of_bigarray: a char bigarray",
      fun () ->
        ignore
          (Nx.of_bigarray (Bigarray.Genarray.create Bigarray.char Bigarray.c_layout [| 2 |]))
    );
  ]

let name_of label = "Nx." ^ String.sub label 0 (String.index label ':')

let ocaml =
  group "from OCaml values"
    [
      prop "to_array (create dt s vs) is vs as dt stores them" case law_round_trip;
      prop "item reads to_array's element at its index"
        (Gen.pair case (Gen.pair Gen.nat Gen.bool))
        law_item;
      test "init calls f once per index, in C order" test_init_order;
      test "init gives f a fresh index each call" test_init_fresh_index;
      test "unpack fixes the dtype or names both" test_unpack;
      cases "refusals name the function" ~name:fst refusals (fun (label, f) ->
          invalid ~by:(name_of label) f);
    ]

(* Bigarrays *)

let test_bigarray_round_trip () =
  let b = Bigarray.Genarray.create Bigarray.float64 Bigarray.c_layout [| 2; 3 |] in
  for i = 0 to 1 do
    for j = 0 to 2 do
      Bigarray.Genarray.set b [| i; j |] (float_of_int ((3 * i) + j))
    done
  done;
  let x = Nx.of_bigarray b in
  equal (array int) [| 2; 3 |] (Nx.shape x);
  Bigarray.Genarray.set b [| 0; 0 |] 99.;
  equal (array float_exact) [| 0.; 1.; 2.; 3.; 4.; 5. |] (Nx.to_array x);
  let c = Nx.to_bigarray Bigarray.float64 x in
  equal (array int) [| 2; 3 |] (Bigarray.Genarray.dims c);
  Bigarray.Genarray.set c [| 1; 2 |] 77.;
  equal float_exact 5. (Nx.item [ 1; 2 ] x)

let test_bigarray_kinds () =
  let ints = Nx.to_bigarray Bigarray.int8_unsigned (Nx.create Nx.uint8 [| 2 |] [| 0; 255 |]) in
  equal int 255 (Bigarray.Genarray.get ints [| 1 |]);
  let z = Nx.create Nx.complex64 [| 1 |] [| { Complex.re = 1.5; im = -0. } |] in
  let back = Nx.of_bigarray (Nx.to_bigarray Bigarray.complex32 z) in
  equal_elements Nx.complex64 (Nx.to_array z) (Nx.to_array back);
  let h = Nx.to_bigarray Bigarray.float16 (Nx.create Nx.float16 [||] [| 0.5 |]) in
  equal float_exact 0.5 (Bigarray.Genarray.get h [||])

let bigarrays =
  group "bigarrays"
    [
      test "of_bigarray copies; to_bigarray is fresh" test_bigarray_round_trip;
      test "kinds carry their dtypes" test_bigarray_kinds;
    ]

(* Reads at every brand *)

let test_read_split () =
  let s = [| 4; 3 |] in
  let x = Nx.place (S2.split ~axis:0) (Nx.create f32 s (floats s)) in
  equal (array float_exact) (floats s) (Nx.to_array x);
  equal float_exact 10. (Nx.item [ 3; 1 ] x);
  equal (array float_exact) (floats s)
    (let b = Nx.to_bigarray Bigarray.float32 x in
     Array.init 12 (fun j -> Bigarray.Genarray.get b [| j / 3; j mod 3 |]));
  equal bool true
    (Nx.Placement.equal (S2.split ~axis:0) (Option.get (Nx.placement x)))

let test_read_every_set () =
  let x = Nx.add (Nx.zeros f32 [| 3 |]) (Nx.scalar f32 2.) in
  equal (array float_exact) [| 2.; 2.; 2. |] (Nx.to_array x);
  equal float_exact 2. (Nx.item [ -1 ] x);
  equal bool true (Nx.placement x = None)

let test_read_view () =
  let x = Nx.transpose (Nx.create Nx.int16 [| 2; 3 |] [| 1; 2; 3; 4; 5; 6 |]) in
  equal (array int) [| 1; 4; 2; 5; 3; 6 |] (Nx.to_array x);
  equal int 6 (Nx.item [ 2; 1 ] x)

let test_read_dead () =
  let x = Nx.create f32 [| 2 |] [| 1.; 2. |] in
  let _ = Nx.add (Nx.donate x) (Nx.create f32 [| 2 |] [| 0.; 0. |]) in
  invalid ~by:"Nx.to_array" (fun () -> Nx.to_array x);
  invalid ~by:"Nx.item" (fun () -> Nx.item [ 0 ] x)

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let mark i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

let test_read_traced () =
  let x = Nx.create f32 [| 2 |] [| 1.; 2. |] in
  Nx.Prim.interpret ~name:"test.reads" Values mark (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form x) Mark in
      raises_match
        (Exn.invalid_arg
           ~substring:"Nx.to_array: the value is traced by test.reads")
        (fun () -> Nx.to_array t);
      raises_match
        (Exn.invalid_arg ~substring:"Nx.item: the value is traced by test.reads")
        (fun () -> Nx.item [ 0 ] t))

(* An extent reaches every operation on its fiber; a read is none. *)
let test_read_under_extent () =
  let x = Nx.create f32 [| 2 |] [| 1.; 2. |] and c = Nx.zeros f32 [| 1 |] in
  let got =
    Nx.Prim.interpret ~name:"test.extent" Extent mark (fun _ ->
        (Nx.to_array x, Nx.item [ 1 ] x, Nx.to_array c))
  in
  let a, v, z = got in
  equal (array float_exact) [| 1.; 2. |] a;
  equal float_exact 2. v;
  equal (array float_exact) [| 0. |] z

let reads =
  group "reads"
    [
      test "a split value reads whole and stays split" test_read_split;
      test "a value of every set reads computed on the host" test_read_every_set;
      test "a view reads in its own C order" test_read_view;
      test "a dead value raises naming the read" test_read_dead;
      test "a traced value raises naming the read" test_read_traced;
      test "an extent receives no read" test_read_under_extent;
    ]

let () = exit (run "nx data" [ ocaml; bigarrays; reads ])
