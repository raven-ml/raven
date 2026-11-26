(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.mli's values from OCaml data and the reads that give a value's elements
   back, at every dtype and brand, with unpack, Host and Repr's reads, which
   take and give the same data. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let f32 = Nx.float32
let floats s = Array.init (Array.fold_left ( * ) 1 s) float_of_int

(* [by]'s raises, each row a reason and a call. *)
let raises_naming by rows =
  cases "raises naming itself" ~id:fst rows (fun (_, f) -> invalid ~by f)

let r reason f = (reason, fun () -> ignore (f ()))

(* Elements *)

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
    (A.v dt
       (L.contiguous (Iarray.of_array s))
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

let shape ~min =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency
       [ (1, Gen.constant min); (1, Gen.constant 1); (4, Gen.int_range 2 4) ])

type case = Case : ('v, 's) D.t * int array * 'v array -> case

let pp_dtype ppf (D.Any dt) = D.pp ppf dt

(* Values of [dtypes], of shapes whose extents are at least [min]. *)
let case ?(min = 0) dtypes =
  Gen.with_pp
    (fun ppf (Case (dt, s, _)) -> Format.fprintf ppf "%a %a" D.pp dt pp_ints s)
    (let open Gen in
     let* (D.Any dt) = of_list ~pp:pp_dtype dtypes in
     let* s = shape ~min in
     let+ vs = drawn dt s in
     Case (dt, s, vs))

(* A float is stored as [Dtype.of_float] says. *)
let stored (type v s) (dt : (v, s) D.t) (v : v) : v =
  match D.kind dt with D.Float -> D.of_float dt v | _ -> v

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let mark i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

(* [f] of a value traced by the interpretation [test.reads]. *)
let traced f =
  let x = Nx.create f32 [| 2 |] [| 1.; 2. |] in
  Nx.Prim.interpret ~name:"test.reads" Values mark (fun i ->
      f (Nx.Prim.traced i (Nx.Prim.form x) Mark))

(* A value another operation consumed: dead. *)
let dead () =
  let x = Nx.create f32 [| 2 |] [| 1.; 2. |] in
  ignore (Nx.neg (Nx.donate x));
  x

(* unpack *)

let unpack =
  group "unpack"
    [
      test "is the value at its own dtype, and names both dtypes otherwise"
        (fun () ->
          let x = Nx.create f32 [| 2 |] [| 1.; 2. |] in
          equal (array float_exact) [| 1.; 2. |]
            (Nx.to_array (Nx.unpack f32 (Nx.P x)));
          match Nx.unpack Nx.int32 (Nx.P x) with
          | _ -> fail "Nx.unpack raised nothing"
          | exception Invalid_argument m ->
              starts_with ~prefix:"Nx.unpack: " m;
              contains ~sub:"float32" m;
              contains ~sub:"int32" m);
    ]

(* Host *)

(* A fresh flat bigarray of [n] elements of [k]. *)
let vector k n = Bigarray.Array1.create k Bigarray.c_layout n

(* Two float32 elements mapped from a file at byte 1, so their data is not
   aligned for them. The file is the test's temporary file: Windows refuses to
   delete a mapped file, and the runner leaves one it cannot delete. *)
let misaligned () =
  let path = temp_file ~suffix:".bin" () in
  Out_channel.with_open_bin path (fun oc ->
      output_string oc (String.make 9 'a'));
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd)
    (fun () ->
      Bigarray.array1_of_genarray
        (Unix.map_file fd ~pos:1L Bigarray.float32 Bigarray.c_layout false
           [| 2 |]))

(* The layouts Host.read reads through, of a value of [Case]'s shape: in place
   where C-contiguous, at an offset among them, else a copy. *)
let law_host_read (Case (dt, s, vs), l) =
  cover "no element" (Array.exists (( = ) 0) s);
  cover "rank 0" (s = [||]);
  cover "C-contiguous at an offset"
    (l = At_offset && not (Array.exists (( = ) 0) s));
  cover "a view" (l <> C_order && l <> At_offset);
  let x, want = laid l dt s (Array.map (stored dt) vs) in
  let got =
    Nx.Host.read x (fun b ->
        Array.init (Bigarray.Array1.dim b) (Bigarray.Array1.get b))
  in
  equal_elements dt want got

let held_by_bigarray =
  List.filter (fun (D.Any dt) -> D.to_bigarray_kind dt <> None) D.all

let host_read =
  group "Host.read"
    [
      prop "gives a host value's elements in C order, through any layout"
        ~examples:
          [
            ( Case (D.Float32, [| 2; 3 |], [| 1.; 2.; 3.; 4.; 5.; 6. |]),
              At_offset );
          ]
        (Gen.pair (case held_by_bigarray) layouts)
        law_host_read;
      test "of wrap's value gives the bigarray's elements" (fun () ->
          let b =
            Bigarray.Array1.init Bigarray.int32 Bigarray.c_layout 4 Int32.of_int
          in
          Nx.Host.read
            (Nx.Host.wrap [| 2; 2 |] b)
            (fun b' ->
              equal (array int32) [| 0l; 1l; 2l; 3l |]
                (Array.init (Bigarray.Array1.dim b') (Bigarray.Array1.get b'))));
      raises_naming "Nx.Host.read"
        [
          r "a dtype with no Bigarray kind" (fun () ->
              Nx.Host.read (Nx.zeros Nx.bfloat16 [| 2 |]) ignore);
          r "a dead value" (fun () -> Nx.Host.read (dead ()) ignore);
        ];
    ]

let host_wrap =
  group "Host.wrap"
    [
      test "is the value of a shape over a bigarray's elements" (fun () ->
          let c = Bigarray.c_layout in
          let b = Bigarray.Array1.init Bigarray.float32 c 6 float_of_int in
          let x = Nx.Host.wrap [| 2; 3 |] b in
          equal (array int) [| 2; 3 |] (Nx.shape x);
          equal (array float_exact) [| 0.; 1.; 2.; 3.; 4.; 5. |] (Nx.to_array x);
          equal (array float_exact) [| 3.; 4.; 5. |]
            (Nx.to_array (Nx.get [ 1 ] x));
          let one = Bigarray.Array1.init Bigarray.int64 c 1 Int64.of_int in
          equal (array int) [||] (Nx.shape (Nx.Host.wrap [||] one));
          let none = vector Bigarray.int8_unsigned 0 in
          equal (array int) [| 0; 3 |] (Nx.shape (Nx.Host.wrap [| 0; 3 |] none)));
      test "a donation of its value leaves the bigarray" (fun () ->
          let c = Bigarray.c_layout in
          let b = Bigarray.Array1.init Bigarray.float32 c 3 Float.of_int in
          ignore (Nx.to_array (Nx.neg (Nx.donate (Nx.Host.wrap [| 3 |] b))));
          equal (array float_exact) [| 0.; 1.; 2. |]
            (Array.init 3 (Bigarray.Array1.get b)));
      raises_naming "Nx.Host.wrap"
        [
          r "a shape of other elements" (fun () ->
              Nx.Host.wrap [| 2; 2 |] (vector Bigarray.float32 3));
          r "a negative extent" (fun () ->
              Nx.Host.wrap [| -1; -2 |] (vector Bigarray.float32 2));
          r "a char bigarray" (fun () ->
              Nx.Host.wrap [| 2 |] (vector Bigarray.char 2));
          r "data not aligned for its elements" (fun () ->
              Nx.Host.wrap [| 2 |] (misaligned ()));
        ];
    ]

(* create *)

let law_round_trip (Case (dt, s, vs)) =
  cover "no element" (Array.exists (( = ) 0) s);
  cover "rank 0" (s = [||]);
  let x = Nx.create dt s vs in
  equal (array int) s (Nx.shape x);
  equal_elements dt (Array.map (stored dt) vs) (Nx.to_array x)

let create =
  group "create"
    [
      prop "holds vs in C order as dt stores them" (case D.all) law_round_trip;
      raises_naming "Nx.create"
        [
          r "too few values" (fun () ->
              Nx.create f32 [| 2; 2 |] [| 1.; 2.; 3. |]);
          r "a negative extent" (fun () -> Nx.create f32 [| -1 |] [||]);
          r "an int above the dtype" (fun () ->
              Nx.create Nx.uint8 [| 1 |] [| 256 |]);
          r "an int below the dtype" (fun () ->
              Nx.create Nx.int4 [| 1 |] [| -9 |]);
        ];
    ]

(* init *)

let init =
  group "init"
    [
      test "calls f once per index, in C order" (fun () ->
          let calls = ref [] in
          let x =
            Nx.init Nx.int32 [| 2; 3 |] (fun i ->
                calls := Array.copy i :: !calls;
                Int32.of_int ((10 * i.(0)) + i.(1)))
          in
          equal
            (list (array int))
            [
              [| 0; 0 |];
              [| 0; 1 |];
              [| 0; 2 |];
              [| 1; 0 |];
              [| 1; 1 |];
              [| 1; 2 |];
            ]
            (List.rev !calls);
          equal (array int32) [| 0l; 1l; 2l; 10l; 11l; 12l |] (Nx.to_array x));
      test "gives f a fresh index each call" (fun () ->
          let kept = ref [] in
          let _ =
            Nx.init Nx.int8 [| 3 |] (fun i ->
                kept := i :: !kept;
                0)
          in
          equal (list (array int)) [ [| 2 |]; [| 1 |]; [| 0 |] ] !kept);
      test "passes on what f raises" (fun () ->
          raises Exit (fun () -> Nx.init f32 [| 2 |] (fun _ -> raise Exit)));
      raises_naming "Nx.init"
        [
          r "a negative extent" (fun () -> Nx.init f32 [| -2 |] (fun _ -> 0.));
          r "an int outside the dtype" (fun () ->
              Nx.init Nx.uint8 [| 1 |] (fun _ -> 300));
        ];
    ]

(* map_item *)

(* An element's image under [f], at each dtype a function of that dtype: [f] is
   applied to the element, and the store rounds a float once. *)
let test_map_values () =
  let x = Nx.create f32 [| 2; 2 |] [| 1.; -0.; nan; 0.1 |] in
  equal_elements Nx.float32
    (Nx.to_array (Nx.create f32 [| 4 |] [| -1.; 0.; nan; -0.1 |]))
    (Nx.to_array (Nx.map_item Float.neg x));
  let i = Nx.create Nx.int8 [| 3 |] [| -128; 0; 127 |] in
  equal (array int) [| 127; -1; -128 |]
    (Nx.to_array (Nx.map_item (fun v -> -v - 1) i));
  let b = Nx.create Nx.bool [| 2 |] [| true; false |] in
  equal (array bool) [| false; true |] (Nx.to_array (Nx.map_item not b))

let map_item =
  group "map_item"
    [
      prop "of the identity gives the value back" (case D.all)
        (fun (Case (dt, s, vs)) ->
          let x = Nx.create dt s vs in
          let y = Nx.map_item Fun.id x in
          equal (array int) s (Nx.shape y);
          equal_elements dt (Nx.to_array x) (Nx.to_array y));
      test "is f of each element, stored in the dtype" test_map_values;
      test "runs f once per element in C order" (fun () ->
          let seen = ref [] in
          let x =
            Nx.transpose
              (Nx.create Nx.int32 [| 2; 3 |] [| 0l; 1l; 2l; 3l; 4l; 5l |])
          in
          let _ =
            Nx.map_item
              (fun v ->
                seen := v :: !seen;
                v)
              x
          in
          equal (list int32) [ 0l; 3l; 1l; 4l; 2l; 5l ] (List.rev !seen));
      test "of a value on a device set gives a host value" (fun () ->
          let s = [| 4; 3 |] in
          let x = Nx.place (S2.split ~axis:0) (Nx.create f32 s (floats s)) in
          let y : Nx.host Nx.float32_t = Nx.map_item Float.neg x in
          equal (array float_exact)
            (Array.map Float.neg (floats s))
            (Nx.to_array y);
          equal (array float_exact) (floats s) (Nx.to_array x));
      test "passes on what f raises" (fun () ->
          raises Exit (fun () ->
              Nx.map_item (fun _ -> raise Exit) (Nx.zeros f32 [| 2 |])));
      raises_naming "Nx.map_item"
        [
          r "an int outside the dtype" (fun () ->
              Nx.map_item (fun _ -> 300) (Nx.zeros Nx.uint8 [| 2 |]));
        ];
    ]

(* of_bigarray and to_bigarray *)

let bigarrays =
  group "of_bigarray and to_bigarray"
    [
      test "of_bigarray copies, to_bigarray is fresh" (fun () ->
          let b =
            Bigarray.Genarray.create Bigarray.float64 Bigarray.c_layout
              [| 2; 3 |]
          in
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
          equal float_exact 5. (Nx.item [ 1; 2 ] x));
      test "kinds carry their dtypes" (fun () ->
          let ints =
            Nx.to_bigarray Bigarray.int8_unsigned
              (Nx.create Nx.uint8 [| 2 |] [| 0; 255 |])
          in
          equal int 255 (Bigarray.Genarray.get ints [| 1 |]);
          let z =
            Nx.create Nx.complex64 [| 1 |] [| { Complex.re = 1.5; im = -0. } |]
          in
          let back = Nx.of_bigarray (Nx.to_bigarray Bigarray.complex32 z) in
          equal_elements Nx.complex64 (Nx.to_array z) (Nx.to_array back);
          let h =
            Nx.to_bigarray Bigarray.float16
              (Nx.create Nx.float16 [||] [| 0.5 |])
          in
          equal float_exact 0.5 (Bigarray.Genarray.get h [||]));
      raises_naming "Nx.of_bigarray"
        [
          r "a char bigarray" (fun () ->
              Nx.of_bigarray
                (Bigarray.Genarray.create Bigarray.char Bigarray.c_layout
                   [| 2 |]));
        ];
    ]

(* Reads *)

(* [a b] over two devices, the contracted axis cut: asked for its sum, a partial
   value whose terms each device holds. *)
let partial_product () =
  let a = Nx.create f32 [| 2; 4 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
  let b = Nx.create f32 [| 4; 1 |] [| 1.; -1.; 2.; 3. |] in
  let a' = Nx.place (S2.split ~axis:1) a
  and b' = Nx.place (S2.split ~axis:0) b in
  (Nx.matmul a b, Nx.matmul ~sum:true a' b')

(* Each read of a dead value and of a traced one, by the name it raises with. *)
let read_raises =
  let reads =
    [
      ("Nx.to_array", fun x -> ignore (Nx.to_array x));
      ("Nx.item", fun x -> ignore (Nx.item [ 0 ] x));
      ("Nx.to_bigarray", fun x -> ignore (Nx.to_bigarray Bigarray.float32 x));
      ("Nx.map_item", fun x -> ignore (Nx.map_item Fun.id x));
      ("Nx.Host.read", fun x -> Nx.Host.read x ignore);
    ]
  in
  List.concat_map
    (fun (by, f) ->
      [
        (by ^ ", dead", fun () -> invalid ~by (fun () -> f (dead ())));
        ( by ^ ", traced",
          fun () ->
            traced (fun t ->
                raises_match
                  (Exn.invalid_arg
                     ~sub:(by ^ ": the value is traced by test.reads"))
                  (fun () -> f t)) );
      ])
    reads

let reads =
  group "reads"
    [
      test "read a split value whole and leave it split" (fun () ->
          let s = [| 4; 3 |] in
          let x = Nx.place (S2.split ~axis:0) (Nx.create f32 s (floats s)) in
          equal ~msg:"to_array" (array float_exact) (floats s) (Nx.to_array x);
          equal ~msg:"item" float_exact 10. (Nx.item [ 3; 1 ] x);
          equal ~msg:"to_bigarray" (array float_exact) (floats s)
            (let b = Nx.to_bigarray Bigarray.float32 x in
             Array.init 12 (fun j ->
                 Bigarray.Genarray.get b [| j / 3; j mod 3 |]));
          equal ~msg:"placement" (placement ()) (S2.split ~axis:0)
            (Nx.placement x));
      test "read a computed host value's elements" (fun () ->
          let x = Nx.add (Nx.zeros f32 [| 3 |]) (Nx.scalar f32 2.) in
          equal (array float_exact) [| 2.; 2.; 2. |] (Nx.to_array x);
          equal float_exact 2. (Nx.item [ -1 ] x);
          equal ~msg:"placement" (placement ()) Nx.Host.on (Nx.placement x));
      test "read a view in its own C order" (fun () ->
          let x =
            Nx.transpose (Nx.create Nx.int16 [| 2; 3 |] [| 1; 2; 3; 4; 5; 6 |])
          in
          equal (array int) [| 1; 4; 2; 5; 3; 6 |] (Nx.to_array x);
          equal int 6 (Nx.item [ 2; 1 ] x));
      test "read a partial value's terms' sums and leave it partial" (fun () ->
          let closed, partial = partial_product () in
          let at = Nx.placement partial in
          equal ~msg:"to_array" (array float_exact) (Nx.to_array closed)
            (Nx.to_array partial);
          equal ~msg:"item" float_exact
            (Nx.item [ 1; 0 ] closed)
            (Nx.item [ 1; 0 ] partial);
          equal ~msg:"placement" (placement ()) at (Nx.placement partial));
      test "reach no interpretation" (fun () ->
          let x = Nx.create f32 [| 2 |] [| 1.; 2. |]
          and c = Nx.zeros f32 [| 1 |] in
          let a, v, z =
            Nx.Prim.interpret ~name:"test.extent" Extent mark (fun _ ->
                (Nx.to_array x, Nx.item [ 1 ] x, Nx.to_array c))
          in
          equal (array float_exact) [| 1.; 2. |] a;
          equal float_exact 2. v;
          equal (array float_exact) [| 0. |] z);
      cases "raise naming themselves for a dead and a traced value" ~id:fst
        read_raises (fun (_, f) -> f ());
    ]

(* item *)

(* [item i x] is the element of [to_array x] at [i]'s position in C order,
   written with positive or negative positions. *)
let law_item (Case (dt, s, vs), (seed, negative)) =
  let flat = seed mod Array.length vs in
  cover "0-d" (s = [||]);
  cover "negative" negative;
  let i = index s flat in
  let p = Array.mapi (fun a q -> if negative then q - s.(a) else q) i in
  equal (elements dt)
    [ vs.(flat) ]
    [ Nx.item (Array.to_list p) (Nx.create dt s vs) ]

let item =
  group "item"
    [
      prop "reads to_array's element at its index, from the start or the end"
        (Gen.pair (case ~min:1 D.all) (Gen.pair Gen.nat Gen.bool))
        law_item;
      raises_naming "Nx.item"
        [
          r "too few positions" (fun () ->
              Nx.item [ 0 ] (Nx.zeros f32 [| 2; 2 |]));
          r "a position past the axis" (fun () ->
              Nx.item [ 2 ] (Nx.zeros f32 [| 2 |]));
          r "a position before the axis" (fun () ->
              Nx.item [ -3 ] (Nx.zeros f32 [| 2 |]));
          r "an empty axis" (fun () -> Nx.item [ 0 ] (Nx.zeros f32 [| 0 |]));
        ];
    ]

(* Repr *)

(* [Nx.Repr.read ~by] names [by] where it refuses. *)
let test_repr_read_by () =
  invalid ~by:"test.reader" (fun () ->
      Nx.Repr.read ~by:"test.reader" (dead ()) A.to_array);
  traced (fun t ->
      raises_match
        (Exn.invalid_arg ~sub:"test.reader: the value is traced by test.reads")
        (fun () -> Nx.Repr.read ~by:"test.reader" t A.to_array))

(* What a read raises that it did not refuse, a pending check's failure or its
   function's own error, passes on as raised, whatever [~by] names. *)
let test_repr_read_passes () =
  let module S = (val Nx.devices [ m 0 ]) in
  let failed = Invalid_argument "test.check: a flag is false" in
  let ok = Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; false |]) in
  Nx.check Nx.Ptree.unit ok () (fun _ () -> failed);
  raises failed (fun () -> Nx.Repr.read ~by:"test.reader" ok A.to_array);
  let x = Nx.create f32 [| 2 |] [| 1.; 2. |] in
  let own = Invalid_argument "test.f: its own error" in
  raises own (fun () -> Nx.Repr.read ~by:"test.reader" x (fun _ -> raise own))

(* [Nx.Repr.array ~by] and [Nx.Repr.shards ~by] name [by] where they refuse: a
   dead value, on one device or several, and a traced one. *)
let test_repr_crossing_by () =
  let by = "test.crosser" in
  let refused why f = raises_match (Exn.invalid_arg ~sub:(by ^ ": " ^ why)) f in
  let gone x =
    ignore (Nx.donate x);
    x
  in
  let one = gone (Nx.create f32 [| 2 |] [| 1.; 2. |]) in
  let four = Nx.create f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let cut = gone (Nx.place (S2.split ~axis:0) four) in
  refused "float32 [2] was donated" (fun () -> Nx.Repr.array ~by one);
  refused "float32 [2] was donated" (fun () -> Nx.Repr.shards ~by one);
  refused "float32 [4] was donated" (fun () -> Nx.Repr.array ~by cut);
  refused "float32 [4] was donated" (fun () -> Nx.Repr.shards ~by cut);
  traced (fun t ->
      let traced = "a value traced by test.reads has no bytes" in
      refused traced (fun () -> Nx.Repr.array ~by t);
      refused traced (fun () -> Nx.Repr.shards ~by t))

let repr =
  group "Repr"
    [
      test "read raises naming its ~by" test_repr_read_by;
      test "read passes on what it did not refuse" test_repr_read_passes;
      test "array and shards raise naming their ~by" test_repr_crossing_by;
    ]

let () =
  exit
    (run "nx data"
       [
         unpack;
         host_read;
         host_wrap;
         create;
         init;
         map_item;
         bigarrays;
         reads;
         item;
         repr;
       ])
