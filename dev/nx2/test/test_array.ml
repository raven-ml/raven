(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module S = Nx_array_support

let layout = Testable.make ~pp:L.pp ~equal:L.equal
let ints = array int
let f32 = D.Float32

(* A witness of [dt]'s values that compares floats bit for bit. *)
let value : type v s. (v, s) D.t -> v testable =
 fun dt ->
  let float = Testable.equal float_exact in
  match D.kind dt with
  | D.Float -> Testable.make ~pp:(D.pp_value dt) ~equal:float
  | D.Complex ->
      Testable.make ~pp:(D.pp_value dt) ~equal:(fun (a : Complex.t) b ->
          float a.re b.re && float a.im b.im)
  | D.Boolean -> bool
  | D.Signed | D.Unsigned -> Testable.make ~pp:(D.pp_value dt) ~equal:( = )

let values dt = array (value dt)
let floats32 s xs = A.of_array f32 s xs

(* Every index's element by [get], in C order. *)
let gets a = Array.of_list (List.map (A.get a) (indices (L.shape (A.layout a))))

(* Kills [b]: a donation consumes its memory. *)
let kill b =
  Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"consumed by the test" b))

(* Whether no claim holds [b]'s memory: a donation of it is exclusive. *)
let unclaimed b =
  Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c -> Rig.Claim.exclusive c b)

(* Arrays of every dtype *)

(* A host array of [dt] of shape [s], its values stores of drawn floats. *)
let filled dt s =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ xs = array ~size:(const n) (float_range (-300.) 300.) in
  A.of_array dt s (Array.map (D.of_float dt) xs)

type case = Case : ('v, 's) A.t * M.t option -> case

let pp_case ppf (Case (a, m)) =
  Format.fprintf ppf "%a %a%a" D.pp (A.dtype a) L.pp (A.layout a)
    (fun ppf -> function
      | None -> () | Some m -> Format.fprintf ppf ", %a" pp_move m)
    m

(* An array of any dtype and shape, and a movement of it. *)
let case =
  let open Gen in
  let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all in
  let* s = shape in
  let* a = filled dt s in
  let+ m = option (movement ~apart:false s) in
  Case (a, m)

let case = Gen.with_pp pp_case case

(* Construction *)

let test_bounds () =
  let b = B.create Rig.host 16 in
  let fails l = raises_match Exn.invalid_arg (fun () -> A.v f32 l b) in
  ignore (A.v f32 (L.contiguous [| 4 |]) b);
  ignore (A.v f32 (L.v ~offset:3 ~strides:[| -1 |] [| 4 |]) b);
  fails (L.contiguous [| 5 |]);
  fails (L.v ~offset:1 ~strides:[| 1 |] [| 4 |]);
  ignore (A.v D.Int4 (L.contiguous [| 32 |]) b);
  ignore (A.v D.Bit (L.contiguous [| 128 |]) b);
  raises_match Exn.invalid_arg (fun () -> A.v D.Int4 (L.contiguous [| 33 |]) b);
  raises_match Exn.invalid_arg (fun () -> A.v D.Bit (L.contiguous [| 129 |]) b);
  raises_match Exn.invalid_arg (fun () ->
      A.v D.Complex128 (L.contiguous [| 2 |]) b);
  ignore (A.v f32 (L.contiguous [| 0 |]) (B.create Rig.host 0))

let test_alignment () =
  let b = B.create Rig.host 64 in
  let at first = B.view b ~first ~length:32 in
  let fails dt l b = raises_match Exn.invalid_arg (fun () -> A.v dt l b) in
  fails f32 (L.contiguous [| 1 |]) (at 1);
  fails f32 (L.contiguous [| 1 |]) (at 2);
  fails D.Float64 (L.contiguous [| 1 |]) (at 4);
  fails D.Int16 (L.contiguous [| 1 |]) (at 1);
  ignore (A.v D.Complex64 (L.contiguous [| 1 |]) (at 4));
  ignore (A.v D.Uint8 (L.contiguous [| 1 |]) (at 1));
  ignore (A.v D.Int4 (L.contiguous [| 1 |]) (at 1));
  ignore (A.v f32 (L.v ~offset:1 ~strides:[| 1 |] [| 1 |]) (at 4));
  fails f32 (L.v ~offset:1 ~strides:[| 1 |] [| 1 |]) (at 2);
  ignore (A.v f32 (L.contiguous [| 0 |]) (at 1))

let test_dead () =
  let b = B.create Rig.host 16 in
  kill b;
  raises_match (Exn.invalid_arg ~substring:"dead") (fun () ->
      A.v f32 (L.contiguous [| 4 |]) b)

(* The last byte of a fresh sub-byte array: its bits past the last element are
   zero. *)
let test_tail () =
  let tail dt n mask =
    let a = A.create Rig.host dt [| n |] in
    let b = A.buffer a in
    let byte =
      A.get (A.v D.Uint8 (L.contiguous [| B.length b |]) b) [| B.length b - 1 |]
    in
    equal ~msg:(D.name dt) int 0 (byte land mask)
  in
  tail D.Int4 3 0xF0;
  tail D.Uint4 1 0xF0;
  tail D.Float4_e2m1fn 5 0xF0;
  tail D.Bit 5 0xE0;
  tail D.Bit 1 0xFE

let test_create () =
  let a = A.create Rig.host D.Complex64 [| 2; 3 |] in
  equal layout (L.contiguous [| 2; 3 |]) (A.layout a);
  equal int 48 (B.length (A.buffer a));
  equal int 2 (B.length (A.buffer (A.create Rig.host D.Int4 [| 3 |])));
  equal bool true (Rig.equal Rig.host (A.device a))

(* Every movement keeps the array inside its buffer: [v] takes its parts
   back. *)
let law_bounds (Case (a, m)) =
  match m with
  | None -> ()
  | Some m -> (
      match A.move m a with
      | None -> cover "a reshape copies" true
      | Some b ->
          cover "a view" true;
          ignore (A.v (A.dtype b) (A.layout b) (A.buffer b)))

(* The reshape hole: the shape a movement takes is not kept. *)
let test_reshape_ownership () =
  let a = floats32 [| 6 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let s = [| 2; 3 |] in
  let b = Option.get (A.move (M.Reshape s) a) in
  s.(0) <- 1_000_000;
  equal ints [| 2; 3 |] (L.shape (A.layout b));
  raises_match Exn.invalid_arg (fun () -> A.get b [| 1_000; 0 |]);
  let shape = L.shape (A.layout b) in
  shape.(0) <- 1_000_000;
  equal ints [| 2; 3 |] (L.shape (A.layout b))

(* Bitcasts *)

let test_bitcast_bytes () =
  let a = floats32 [| 2 |] [| 1.; -2. |] in
  let u = Option.get (A.bitcast D.Uint8 a) in
  equal ints [| 2; 4 |] (L.shape (A.layout u));
  equal (array int) [| 0; 0; 0x80; 0x3f; 0; 0; 0; 0xc0 |] (A.to_array u);
  let back = Option.get (A.bitcast f32 u) in
  equal layout (A.layout a) (A.layout back);
  equal (values f32) [| 1.; -2. |] (A.to_array back)

let test_bitcast_sub_byte () =
  let a = A.of_array D.Uint8 [| 2 |] [| 0x21; 0xF7 |] in
  let n = Option.get (A.bitcast D.Int4 a) in
  equal ints [| 2; 2 |] (L.shape (A.layout n));
  equal (array int) [| 1; 2; 7; -1 |] (A.to_array n);
  let bits = Option.get (A.bitcast D.Bit a) in
  equal ints [| 2; 8 |] (L.shape (A.layout bits));
  equal (array bool)
    [| true; false; false; false; false; true; false; false |]
    (Array.sub (A.to_array bits) 0 8)

let test_bitcast_refuses () =
  let a = A.of_array D.Uint8 [| 2; 4 |] (Array.init 8 Fun.id) in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) a) in
  is_none (A.bitcast D.Uint16 t);
  let s =
    Option.get
      (A.move
         (M.Slice
            [|
              { start = 0; count = 2; step = 1 };
              { start = 1; count = 2; step = 1 };
            |])
         a)
  in
  is_none (A.bitcast D.Uint16 s);
  is_none (A.bitcast f32 (A.of_array D.Uint8 [| 2; 3 |] (Array.make 6 0)));
  let e = A.create Rig.host D.Uint8 [| 0; 4 |] in
  equal (option ints) (Some [| 0 |])
    (Option.map (fun a -> L.shape (A.layout a)) (A.bitcast f32 e));
  is_none (A.bitcast f32 (A.create Rig.host D.Uint8 [| 0; 3 |]));
  let b = B.view (B.create Rig.host 32) ~first:4 ~length:16 in
  let c = A.v D.Complex64 (L.contiguous [| 2 |]) b in
  is_none (A.bitcast D.Float64 c);
  let deep = A.create Rig.host D.Uint16 (Array.make 32 1) in
  raises_match Exn.invalid_arg (fun () -> A.bitcast D.Uint8 deep)

(* A narrowing bitcast and its widening are inverse, over any layout of a
   byte-wide array. *)
let law_bitcast_round_trip (Case (a, m)) =
  let a =
    match m with Some m -> Option.value ~default:a (A.move m a) | None -> a
  in
  if D.bits (A.dtype a) >= 8 && L.rank (A.layout a) < L.max_rank then
    match A.bitcast D.Uint8 a with
    | None -> fail "a narrowing bitcast answered None"
    | Some u -> (
        match A.bitcast (A.dtype a) u with
        | None -> fail "widening a narrowing answered None"
        | Some back ->
            cover "a strided array" (not (L.is_contiguous (A.layout a)));
            equal layout (A.layout a) (A.layout back))

let test_expect () =
  let a = floats32 [| 1 |] [| 3. |] in
  equal (values f32) [| 3. |] (A.to_array (A.expect f32 (A.Any a)));
  raises_match (Exn.invalid_arg ~substring:"float32") (fun () ->
      A.expect D.Float64 (A.Any a))

let test_settle () =
  let a = floats32 [| 2; 3 |] (Array.make 6 0.) in
  let b = A.create Rig.host D.Int8 [| 4 |] in
  (match A.settle "add" 1 [ A.Any a; A.Any b ] with
  | () -> fail "settle returned on a refusal"
  | exception Invalid_argument m ->
      starts_with ~affix:"add: " m;
      contains ~sub:"dtype" m;
      contains ~sub:"float32 [2; 3]" m;
      contains ~sub:"int8 [4]" m);
  A.settle "add" 4 [ A.Any a; A.Any b ];
  raises_match (Exn.invalid_arg ~substring:"shapes") (fun () ->
      A.settle "add" 10 [ A.Any a ])

(* Elements *)

let law_set_get (Case (a, _)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let dt = A.dtype a in
    List.iteri
      (fun k idx ->
        let x = D.of_float dt (Float.of_int ((k * 37) - 100) /. 4.) in
        A.set a idx x;
        equal (value dt) x (A.get a idx))
      (indices (L.shape (A.layout a)))
  in
  check a

let test_index () =
  let a = floats32 [| 2; 3 |] (Array.make 6 0.) in
  let fails f = raises_match Exn.invalid_arg f in
  fails (fun () -> A.get a [| 2; 0 |]);
  fails (fun () -> A.get a [| 0; -1 |]);
  fails (fun () -> A.get a [| 0 |]);
  fails (fun () -> A.set a [| 0; 3 |] 1.);
  fails (fun () -> A.get a [| 0; 0; 0 |])

let test_set_refuses () =
  let a = floats32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Option.get (A.move (M.Broadcast [| 2; 3 |]) a) in
  raises_match (Exn.invalid_arg ~substring:"twice") (fun () ->
      A.set b [| 0; 0 |] 5.);
  let u = A.create Rig.host D.Uint8 [| 1 |] in
  raises_match (Exn.invalid_arg ~substring:"256") (fun () ->
      A.set u [| 0 |] 256);
  raises_match Exn.invalid_arg (fun () -> A.set u [| 0 |] (-1));
  let i = A.create Rig.host D.Int4 [| 1 |] in
  raises_match Exn.invalid_arg (fun () -> A.set i [| 0 |] 8);
  raises_match Exn.invalid_arg (fun () -> A.set i [| 0 |] (-9));
  A.set i [| 0 |] (-8);
  equal int (-8) (A.get i [| 0 |]);
  raises_match Exn.invalid_arg (fun () ->
      A.of_array D.Uint16 [| 2 |] [| 1; 65536 |])

let test_io_refuses () =
  let d = S.io_device () in
  let a = A.to_device d (floats32 [| 2 |] [| 1.; 2. |]) in
  let fails f = raises_match (Exn.invalid_arg ~substring:"host") f in
  fails (fun () -> A.get a [| 0 |]);
  fails (fun () -> A.set a [| 0 |] 0.);
  fails (fun () -> A.to_array a);
  fails (fun () -> A.copy a);
  is_none (A.bigarray Bigarray.float32 a)

let test_dead_access () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  kill (A.buffer a);
  let fails f = raises_match (Exn.invalid_arg ~substring:"dead") f in
  fails (fun () -> A.get a [| 0 |]);
  fails (fun () -> A.to_array a)

(* Writes to the elements of one byte from two domains keep each other. *)
let int4s = abstract "a"
let element = Gen.int_range 0 3
let nibble = Gen.int_range (-8) 7

let int4_commands =
  [
    command "create"
      (Gen.unit @-> makes int4s)
      (fun () -> Array.make 4 0)
      (fun () -> A.of_array D.Int4 [| 4 |] (Array.make 4 0));
    command "set"
      (element @-> nibble @-> int4s ^-> returns unit)
      (fun i x m -> m.(i) <- x)
      (fun i x a -> A.set a [| i |] x);
    command "get"
      (element @-> int4s ^-> returns int)
      (fun i m -> m.(i))
      (fun i a -> A.get a [| i |]);
  ]

(* Bulk access *)

let law_round_trip (Case (a, _)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let dt = A.dtype a in
    let xs = A.to_array a in
    equal (values dt) xs (A.to_array (A.of_array dt (L.shape (A.layout a)) xs))
  in
  check a

(* [to_array] reads any layout in C order of indices, as [get] does. *)
let law_to_array (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a -> equal (values (A.dtype a)) (gets a) (A.to_array a)
  in
  match Option.bind m (fun m -> A.move m a) with
  | Some b ->
      cover "a strided view" (not (L.is_contiguous (A.layout b)));
      check b
  | None -> check a

(* Every pattern of a narrow float format decodes alike in the run form
   ([to_array]) and the scalar one ([get]). *)
let test_decode_patterns () =
  let all dt patterns =
    let a = Option.get (A.bitcast dt patterns) in
    equal ~msg:(D.name dt) (values dt) (gets a) (A.to_array a)
  in
  let u16 = A.of_array D.Uint16 [| 65536 |] (Array.init 65536 Fun.id) in
  let u8 = A.of_array D.Uint8 [| 256 |] (Array.init 256 Fun.id) in
  all D.Float16 u16;
  all D.Bfloat16 u16;
  all D.Float8_e4m3fn u8;
  all D.Float8_e5m2 u8;
  all D.Float4_e2m1fn u8

(* And every store of a run ([of_array]) is the scalar store ([of_float]). *)
let law_encode (D.Any dt) =
  let open Gen in
  let x =
    frequency
      [
        (4, float_range (-70000.) 70000.);
        (2, float_range (-1e-4) 1e-4);
        (1, any_float);
        ( 1,
          map
            (fun k -> Float.ldexp (Float.of_int ((2 * k) + 1)) (-11))
            (int_range 0 4096) );
      ]
  in
  match D.kind dt with
  | D.Float ->
      [
        prop (D.name dt)
          (array ~size:(int_range 1 40) x)
          (fun xs ->
            equal (values dt)
              (Array.map (D.of_float dt) xs)
              (A.to_array (A.of_array dt [| Array.length xs |] xs)));
      ]
  | _ -> []

let test_copy () =
  let a = floats32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) a) in
  let c = A.copy t in
  equal layout (L.contiguous [| 3; 2 |]) (A.layout c);
  equal (values f32) [| 0.; 3.; 1.; 4.; 2.; 5. |] (A.to_array c);
  equal bool false (B.overlaps (A.buffer c) (A.buffer a));
  A.set c [| 0; 0 |] 9.;
  equal (values f32) [| 0. |] [| A.get a [| 0; 0 |] |]

let test_copy_bits () =
  let bits = [| 0x7fc00001l; 0xffa00002l; 0x80000000l; 0x7f800001l |] in
  let u = A.of_array D.Uint32 [| 4 |] bits in
  let f = Option.get (A.bitcast f32 u) in
  let back = Option.get (A.bitcast D.Uint32 (A.copy f)) in
  equal (array int32) bits (A.to_array back)

let law_copy (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let c = A.copy a in
    equal layout (L.contiguous (L.shape (A.layout a))) (A.layout c);
    equal (values (A.dtype a)) (A.to_array a) (A.to_array c)
  in
  match Option.bind m (fun m -> A.move m a) with
  | Some b ->
      cover "a strided view" (not (L.is_contiguous (A.layout b)));
      check b
  | None -> check a

let law_to_device (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let c = A.to_device Rig.host a in
    equal ints (L.strides (A.layout a)) (L.strides (A.layout c));
    equal (values (A.dtype a)) (A.to_array a) (A.to_array c);
    if B.length (A.buffer c) > 0 then
      equal bool false (B.overlaps (A.buffer c) (A.buffer a))
  in
  match Option.bind m (fun m -> A.move m a) with
  | Some b ->
      cover "a view" true;
      check b
  | None -> check a

let test_to_device_io () =
  let d = S.io_device () in
  let a = A.of_array D.Int4 [| 7 |] [| 1; -2; 3; -4; 5; -6; 7 |] in
  let s =
    Option.get (A.move (M.Slice [| { start = 5; count = 3; step = -2 } |]) a)
  in
  let back = A.to_device Rig.host (A.to_device d s) in
  equal bool true (Rig.equal Rig.host (A.device back));
  equal (array int) [| -6; -4; -2 |] (A.to_array back);
  equal int 1 (L.offset (A.layout back) - 4)

(* Bigarrays *)

let test_bigarray () =
  let a = floats32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let g = Option.get (A.bigarray Bigarray.float32 a) in
  equal ints [| 2; 3 |] (Bigarray.Genarray.dims g);
  Bigarray.Genarray.set g [| 1; 2 |] 9.;
  equal (values f32) [| 9. |] [| A.get a [| 1; 2 |] |];
  let t = Option.get (A.move (M.Permute [| 1; 0 |]) a) in
  is_none (A.bigarray Bigarray.float32 t);
  let row =
    Option.get
      (A.move
         (M.Slice
            [|
              { start = 1; count = 1; step = 1 };
              { start = 0; count = 3; step = 1 };
            |])
         a)
  in
  let g = Option.get (A.bigarray Bigarray.float32 row) in
  equal float_exact 9. (Bigarray.Genarray.get g [| 0; 2 |]);
  equal bool false (unclaimed (A.buffer a))

let test_of_bigarray () =
  let g =
    Bigarray.Genarray.create Bigarray.int16_signed Bigarray.c_layout [| 2; 2 |]
  in
  Bigarray.Genarray.fill g 3;
  let a = A.of_bigarray D.Int16 g in
  equal bool true (D.equal D.Int16 (A.dtype a));
  Bigarray.Genarray.set g [| 1; 0 |] (-7);
  equal (array int) [| 3; 3; -7; 3 |] (A.to_array a);
  A.set a [| 0; 1 |] 11;
  equal int 11 (Bigarray.Genarray.get g [| 0; 1 |])

(* The door *)

let zeros s = floats32 s (Array.make (Array.fold_left ( * ) 1 s) 0.)

let ok = 0
and dtype_code = 1
and dead = 2
and not_host = 3
and exclusive = 5

let not_distinct = 7
and overlap = 8
and shape_code = 10

let test_door_add () =
  let x = floats32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let y =
    Option.get
      (A.move
         (M.Permute [| 1; 0 |])
         (floats32 [| 3; 2 |] [| 10.; 20.; 30.; 40.; 50.; 60. |]))
  in
  let z = zeros [| 2; 3 |] in
  equal int ok (S.add z x y);
  equal (values f32) [| 10.; 31.; 52.; 23.; 44.; 65. |] (A.to_array z);
  equal int ok (S.add (zeros [| 0; 3 |]) (zeros [| 0; 3 |]) (zeros [| 0; 3 |]))

(* Each operand position given another dtype, then another shape: the call is
   refused and touches no byte. *)
let test_door_positions () =
  let s = [| 2; 3 |] in
  let operands () =
    [| zeros s; floats32 s (Array.make 6 1.); floats32 s (Array.make 6 2.) |]
  in
  let call (ops : 'a array) = S.add ops.(0) ops.(1) ops.(2) in
  for k = 0 to 2 do
    let ops = operands () in
    let wrong = A.of_array D.Float64 s (Array.make 6 5.) in
    let args = Array.map (fun a -> A.Any a) ops in
    args.(k) <- A.Any wrong;
    let e =
      match args with
      | [| A.Any z; A.Any x; A.Any y |] -> S.add z x y
      | _ -> assert false
    in
    equal ~msg:(Printf.sprintf "dtype at %d" k) int dtype_code e;
    equal (values f32) (Array.make 6 0.) (A.to_array ops.(0));
    let ops = operands () in
    ops.(k) <- zeros [| 3; 2 |];
    equal ~msg:(Printf.sprintf "shape at %d" k) int shape_code (call ops);
    if k > 0 then equal (values f32) (Array.make 6 0.) (A.to_array ops.(0))
  done

let test_door_written () =
  let x = floats32 [| 3 |] [| 1.; 2.; 3. |] in
  equal ~msg:"z is x" int overlap (S.add x x (zeros [| 3 |]));
  let b = A.buffer (zeros [| 8 |]) in
  let z = A.v f32 (L.contiguous [| 3 |]) b in
  let y = A.v f32 (L.v ~offset:2 ~strides:[| 1 |] [| 3 |]) b in
  equal ~msg:"z overlaps y" int overlap (S.add z x y);
  let y = A.v f32 (L.v ~offset:3 ~strides:[| 1 |] [| 3 |]) b in
  equal ~msg:"z beside y" int ok (S.add z x y);
  let r = Option.get (A.move (M.Broadcast [| 3 |]) (zeros [| 1 |])) in
  equal ~msg:"z broadcast" int not_distinct (S.add r x x)

let test_door_buffers () =
  let x = floats32 [| 2 |] [| 1.; 2. |] in
  let d = zeros [| 2 |] in
  kill (A.buffer d);
  equal ~msg:"dead" int dead (S.add (zeros [| 2 |]) x d);
  let io = A.to_device (S.io_device ()) x in
  equal ~msg:"io" int not_host (S.add (zeros [| 2 |]) x io);
  let held = zeros [| 2 |] in
  Rig.Claim.with_ ~read:[]
    ~donate:[ [ A.buffer held ] ]
    (fun c ->
      equal bool true (Rig.Claim.exclusive c (A.buffer held));
      equal ~msg:"exclusive" int exclusive (S.add (zeros [| 2 |]) x held))

let test_door_releases () =
  let z = zeros [| 2 |] and x = floats32 [| 2 |] [| 1.; 2. |] in
  equal int ok (S.add z x x);
  equal int shape_code (S.add z x (zeros [| 3 |]));
  equal bool true (unclaimed (A.buffer z));
  equal bool true (unclaimed (A.buffer x))

let test_door_moving_gc () =
  let x = floats32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  equal int ok (S.collect x);
  equal bool true (unclaimed (A.buffer x));
  equal (values f32) [| 1.; 2.; 3.; 4. |] (A.to_array x)

let float_dtypes = List.filter (fun (D.Any dt) -> D.is D.Float dt) D.all

let tests =
  [
    group "construction"
      [
        test "v keeps the layout inside the buffer" test_bounds;
        test "v refuses a misaligned first element" test_alignment;
        test "v refuses a dead buffer" test_dead;
        test "create is C-contiguous at offset 0" test_create;
        test "a fresh sub-byte array's tail bits are zero" test_tail;
        prop "every movement keeps an array in its buffer" case law_bounds;
        test "no shape a movement takes or returns is held"
          test_reshape_ownership;
      ];
    group "bitcast"
      [
        test "a narrowing reads the bytes, its widening returns"
          test_bitcast_bytes;
        test "sub-byte elements lie LSB first" test_bitcast_sub_byte;
        test "a widening refuses strides it cannot divide" test_bitcast_refuses;
        prop "a narrowing then its widening is the identity" case
          law_bitcast_round_trip;
        test "expect recovers the dtype or names both" test_expect;
        test "settle returns on pending work and names every refusal"
          test_settle;
      ];
    group "elements"
      [
        prop "get reads what set stores" case law_set_get;
        test "an index outside the shape is refused" test_index;
        test "set refuses repeated elements and ints out of range"
          test_set_refuses;
        test "memory the host does not address is refused" test_io_refuses;
        test "a dead buffer is refused" test_dead_access;
        stateful ~domains:2
          "writes to one byte from two domains keep each other" int4_commands;
      ];
    group "bulk"
      [
        prop "of_array then to_array is the identity" case law_round_trip;
        prop "to_array reads any layout in C order" case law_to_array;
        test "run and scalar decodes agree over every pattern"
          test_decode_patterns;
        group "run and scalar stores agree"
          (List.concat_map law_encode float_dtypes);
        test "copy gathers into a fresh buffer" test_copy;
        test "copy keeps NaN payloads" test_copy_bits;
        prop "copy is contiguous with the same elements" case law_copy;
        prop "to_device copies and keeps the layout" case law_to_device;
        test "to_device moves sub-byte views through an io device"
          test_to_device_io;
      ];
    group "bigarray"
      [
        test "bigarray shares a contiguous host array's bytes" test_bigarray;
        test "of_bigarray shares the bigarray's bytes" test_of_bigarray;
      ];
    group "door"
      [
        test "a kernel reads its operands through the door" test_door_add;
        test "another dtype or shape at any position is refused untouched"
          test_door_positions;
        test "a written operand must be distinct and alone" test_door_written;
        test "dead, foreign and exclusive buffers are refused" test_door_buffers;
        test "a read releases its claims" test_door_releases;
        test "a read survives a moving collection" test_door_moving_gc;
      ];
  ]

let () = exit (run "nx_array" tests)
