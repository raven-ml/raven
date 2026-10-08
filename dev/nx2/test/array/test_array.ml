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

(* The alignment [v] asks of a dtype's first element, in bytes: its width, one
   component's for complex dtypes, none below a byte. *)
let alignment (D.Any dt) =
  match D.kind dt with D.Complex -> D.bits dt / 16 | _ -> max 1 (D.bits dt / 8)

(* Bytes [n] elements of [b] bits reach, rounded up. *)
let reach_bytes b n = ((n * b) + 7) / 8
let any_dtype = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) D.all

(* A dtype, a layout and a view of a host buffer at any byte. *)
let parts =
  let open Gen in
  with_pp
    (fun ppf (D.Any dt, l, first, length) ->
      Format.fprintf ppf "%a %a, bytes %d to %d" D.pp dt L.pp l first
        (first + length))
    (let* dt = any_dtype in
     let* l = any_layout in
     let* first = int_range 0 17 in
     let (D.Any d) = dt in
     let need = reach_bytes (D.bits d) (snd (L.span l)) - first in
     let+ length = int_range (max 0 (need - 2)) (max 0 need + 2) in
     (dt, l, first, length))

let law_v (D.Any dt, l, first, length) =
  let b = B.view (B.create Rig.host (first + length)) ~first ~length in
  let within = reach_bytes (D.bits dt) (snd (L.span l)) <= length in
  let aligned = B.address b mod alignment (D.Any dt) = 0 in
  let empty = L.numel l = 0 in
  cover "no element" empty;
  cover "past the bytes" ((not empty) && not within);
  cover "misaligned" ((not empty) && within && not aligned);
  cover "accepted" ((not empty) && within && aligned);
  if empty || (within && aligned) then equal layout l (A.layout (A.v dt l b))
  else raises_match Exn.invalid_arg (fun () -> A.v dt l b)

(* [create] is contiguous at offset 0 over the bytes its elements fill, and a
   sub-byte array's last byte is zero past its last element. *)
let law_create (D.Any dt, s) =
  let a = A.create Rig.host dt s in
  let n = L.numel (L.contiguous s) in
  let b = A.buffer a in
  equal layout (L.contiguous s) (A.layout a);
  equal int (D.bytes dt n) (B.length b);
  equal bool true (Rig.equal Rig.host (A.device a));
  let used = n * D.bits dt mod 8 in
  if used > 0 then begin
    cover "a partial last byte" true;
    let bytes = A.v D.Uint8 (L.contiguous [| B.length b |]) b in
    let last = A.get bytes [| B.length b - 1 |] in
    equal ~msg:"bits past the last element" int 0 (last lsr used)
  end

let dtype_and_shape =
  Gen.with_pp
    (fun ppf (D.Any dt, s) -> Format.fprintf ppf "%a %a" D.pp dt pp_ints s)
    Gen.(pair any_dtype (array ~size:(int_range 0 3) (int_range 0 9)))

let test_create_refuses () =
  raises_match Exn.invalid_arg (fun () -> A.create Rig.host f32 [| 2; -1 |]);
  raises_match Exn.invalid_arg (fun () ->
      A.create Rig.host f32 (Array.make 33 1));
  raises_match Exn.invalid_arg (fun () ->
      A.create Rig.host f32 [| max_int; 2 |])

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

let law_move (Case (a, m)) =
  match m with
  | None -> ()
  | Some m -> (
      match (A.move m a, L.move m (A.layout a)) with
      | None, None -> cover "no view" true
      | Some b, Some l ->
          cover "a view" true;
          equal layout l (A.layout b);
          equal bool true (A.buffer a == A.buffer b)
      | Some _, None -> fail "move answered a view Layout.move refuses"
      | None, Some _ -> fail "move answered None where Layout.move has a view")

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

(* The layout [bitcast] gives a layout [l] of [src] read as [dst] over [b], by
   the rule its interface states; [None] where widening's conditions fail or the
   first element is misaligned for [dst]. *)
let bitcast_reference (D.Any src) (D.Any dst) l b =
  let bs = D.bits src and bd = D.bits dst in
  let s = L.shape l and st = L.strides l and o = L.offset l in
  let r = Array.length s in
  let result =
    if bs = bd then Some l
    else if bs > bd then
      let k = bs / bd in
      Some
        (L.v ~offset:(o * k)
           ~strides:(Array.append (Array.map (( * ) k) st) [| 1 |])
           (Array.append s [| k |]))
    else
      let k = bd / bs in
      let outer = Array.sub s 0 (max 0 (r - 1)) in
      if r = 0 || s.(r - 1) <> k then None
      else if L.numel l = 0 then
        Some (L.v ~strides:(Array.make (r - 1) 0) outer)
      else
        let st' = Array.sub st 0 (r - 1) in
        if
          st.(r - 1) <> 1
          || o mod k <> 0
          || Array.exists (fun t -> t mod k <> 0) st'
        then None
        else
          Some
            (L.v ~offset:(o / k)
               ~strides:(Array.map (fun t -> t / k) st')
               outer)
  in
  Option.bind result (fun l' ->
      if L.numel l' = 0 || B.address b mod alignment (D.Any dst) = 0 then
        Some l'
      else None)

(* Two dtypes and a layout of the first over a host buffer at any byte where its
   elements are aligned: often the narrowing of a layout of the second, so that
   widening has its trailing axis. *)
let bitcast_case =
  let open Gen in
  with_pp
    (fun ppf (D.Any src, D.Any dst, l, first) ->
      Format.fprintf ppf "%a to %a, %a at byte %d" D.pp src D.pp dst L.pp l
        first)
    (let* (D.Any src as s) = any_dtype in
     let* (D.Any dst as d) = any_dtype in
     let* l0 = any_layout in
     let k = D.bits dst / D.bits src in
     let* narrowed = bool in
     let l =
       if narrowed && k >= 2 && L.rank l0 < L.max_rank then
         L.v
           ~offset:(L.offset l0 * k)
           ~strides:(Array.append (Array.map (( * ) k) (L.strides l0)) [| 1 |])
           (Array.append (L.shape l0) [| k |])
       else l0
     in
     let+ slot = int_range 0 4 in
     (s, d, l, slot * alignment s))

let law_bitcast (D.Any src, (D.Any dst as d), l, first) =
  let length = reach_bytes (D.bits src) (snd (L.span l)) in
  let b = B.view (B.create Rig.host (length + 64)) ~first ~length in
  let a = A.v src l b in
  let bs = D.bits src and bd = D.bits dst in
  cover "equal widths" (bs = bd);
  cover "narrowing" (bs > bd);
  match (bitcast_reference (D.Any src) d l b, A.bitcast dst a) with
  | Some l', Some c ->
      cover "widening" (bs < bd);
      equal layout l' (A.layout c);
      equal bool true (A.buffer c == b)
  | None, None -> cover "widening refused" true
  | Some l', None -> failf "None where %a was expected" L.pp l'
  | None, Some c -> failf "%a where None was expected" L.pp (A.layout c)

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

(* Every refusal code raises, naming the function and each operand, with a
   reason of its own. *)
let test_settle_codes () =
  let a = floats32 [| 2; 3 |] (Array.make 6 0.) in
  let refusals = [ 1; 2; 3; 5; 6; 7; 8; 9; 10; 11 ] in
  let reason code =
    match A.settle "Nx.f" code [ A.Any a ] with
    | () -> failf "settle returned on code %d" code
    | exception Invalid_argument m ->
        starts_with ~msg:(string_of_int code) ~affix:"Nx.f: " m;
        contains ~msg:(string_of_int code) ~sub:"float32 [2; 3]" m;
        m
  in
  let reasons = List.map reason refusals in
  equal int (List.length refusals)
    (List.length (List.sort_uniq compare reasons))

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

(* An [int] outside its dtype's range is refused by [set] and [of_array], and
   one inside stores as itself. *)
let law_range (type s) (dt : (int, s) D.t) =
  let lo = D.min_value dt and hi = D.max_value dt in
  let x =
    Gen.frequency
      [
        (4, Gen.int_range (lo - 3) (hi + 3));
        ( 2,
          Gen.of_list ~pp:Format.pp_print_int
            [ min_int; lo - 1; lo; hi; hi + 1; max_int ] );
      ]
  in
  prop (D.name dt) x (fun x ->
      let a = A.create Rig.host dt [| 1 |] in
      if x < lo || x > hi then begin
        cover "out of range" true;
        raises_match Exn.invalid_arg (fun () -> A.set a [| 0 |] x);
        raises_match Exn.invalid_arg (fun () -> A.of_array dt [| 1 |] [| x |])
      end
      else begin
        cover "a bound" (x = lo || x = hi);
        A.set a [| 0 |] x;
        equal int x (A.get a [| 0 |]);
        equal (array int) [| x |] (A.to_array (A.of_array dt [| 1 |] [| x |]))
      end)

let ranges =
  [
    law_range D.Int4;
    law_range D.Uint4;
    law_range D.Int8;
    law_range D.Uint8;
    law_range D.Int16;
    law_range D.Uint16;
  ]

let test_of_array_refuses () =
  raises_match Exn.invalid_arg (fun () -> A.of_array f32 [| 2; 2 |] [| 1. |]);
  raises_match Exn.invalid_arg (fun () -> A.of_array f32 [| 1 |] [| 1.; 2. |]);
  raises_match Exn.invalid_arg (fun () -> A.of_array f32 [| -1 |] [||]);
  let a = A.of_array f32 [| 0; 3 |] [||] in
  equal bool true (Rig.equal Rig.host (A.device a));
  equal layout (L.contiguous [| 0; 3 |]) (A.layout a)

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
  fails (fun () -> A.to_array a);
  fails (fun () -> A.bitcast D.Uint32 a);
  fails (fun () -> A.bitcast D.Uint8 a)

(* Memory held exclusive is refused by every function that reads it on the
   host. *)
let test_exclusive () =
  let a = floats32 [| 2 |] [| 1.; 2. |] in
  Rig.Claim.with_ ~read:[]
    ~donate:[ [ A.buffer a ] ]
    (fun c ->
      equal bool true (Rig.Claim.exclusive c (A.buffer a));
      let fails what f =
        raises_match ~msg:what (Exn.invalid_arg ~substring:"exclusive") f
      in
      fails "get" (fun () -> A.get a [| 0 |]);
      fails "set" (fun () -> A.set a [| 0 |] 0.);
      fails "to_array" (fun () -> A.to_array a);
      fails "copy" (fun () -> A.copy a);
      fails "to_device" (fun () -> A.to_device Rig.host a);
      fails "bigarray" (fun () -> A.bigarray Bigarray.float32 a))

let test_dead_bigarray () =
  let a = A.to_device (S.io_device ()) (floats32 [| 2 |] [| 1.; 2. |]) in
  kill (A.buffer a);
  raises_match (Exn.invalid_arg ~substring:"dead") (fun () ->
      A.bigarray Bigarray.float32 a)

let test_dead_empty () =
  let b = B.create Rig.host 16 in
  let a = A.v f32 (L.contiguous [| 0 |]) b in
  kill b;
  raises_match (Exn.invalid_arg ~substring:"dead") (fun () ->
      A.to_device Rig.host a)

(* A copy the host cannot make is refused before it allocates. *)
let test_refused_copy () =
  let d = S.io_device () in
  let a = A.to_device d (floats32 [| 2 |] [| 1.; 2. |]) in
  Rig.free_cache d;
  let before = S.io_allocations () in
  raises_match (Exn.invalid_arg ~substring:"host") (fun () -> A.copy a);
  equal int before (S.io_allocations ())

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

(* Arrays of every byte-wide dtype over drawn bytes, NaN payloads, non-0/1 bools
   and every other pattern included, and a movement of them. *)
let raw =
  let open Gen in
  let byte_wide = List.filter (fun (D.Any dt) -> D.bits dt >= 8) D.all in
  let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) byte_wide in
  let* s = shape in
  let w = D.bits dt / 8 in
  let n = Array.fold_left ( * ) 1 s * w in
  let* bytes = array ~size:(const n) (int_range 0 255) in
  let u = A.of_array D.Uint8 (Array.append s [| w |]) bytes in
  let a = Option.get (A.bitcast dt u) in
  let+ m = option (movement ~apart:false (L.shape (A.layout a))) in
  Case (a, m)

let raw = Gen.with_pp pp_case raw

(* The bytes of [a]'s elements in C order of indices. *)
let bytes_of a = A.to_array (Option.get (A.bitcast D.Uint8 a))

let law_copy_bits (Case (a, m)) =
  let a = Option.value ~default:a (Option.bind m (fun m -> A.move m a)) in
  cover "a strided view" (not (L.is_contiguous (A.layout a)));
  equal (array int) (bytes_of a) (bytes_of (A.copy a))

let law_to_device (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    let c = A.to_device Rig.host a in
    let bits = D.bits (A.dtype a) in
    let lo, hi = L.span (A.layout a) in
    let first = lo * bits / 8 in
    equal ~msg:"bytes" int (reach_bytes bits hi - first) (B.length (A.buffer c));
    equal ~msg:"offset" int
      (L.offset (A.layout a) - (8 * first / bits))
      (L.offset (A.layout c));
    equal ints (L.shape (A.layout a)) (L.shape (A.layout c));
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

(* Bigarray's kind for a dtype, where Bigarray has the format. *)
let kind : type v s. (v, s) D.t -> (v, s) Bigarray.kind option = function
  | D.Float64 -> Some Bigarray.float64
  | D.Float32 -> Some Bigarray.float32
  | D.Float16 -> Some Bigarray.float16
  | D.Int64 -> Some Bigarray.int64
  | D.Int32 -> Some Bigarray.int32
  | D.Int16 -> Some Bigarray.int16_signed
  | D.Uint16 -> Some Bigarray.int16_unsigned
  | D.Int8 -> Some Bigarray.int8_signed
  | D.Uint8 -> Some Bigarray.int8_unsigned
  | D.Complex128 -> Some Bigarray.complex64
  | D.Complex64 -> Some Bigarray.complex32
  | _ -> None

(* [bigarray] is [Some] iff the array is C-contiguous, of rank at most 16, and
   its elements are the array's. *)
let law_bigarray (Case (a, m)) =
  let check : type v s. (v, s) A.t -> unit =
   fun a ->
    match kind (A.dtype a) with
    | None -> ()
    | Some k -> (
        let l = A.layout a in
        match A.bigarray k a with
        | None ->
            cover "None" true;
            equal bool false (L.is_contiguous l && L.rank l <= 16)
        | Some g ->
            cover "Some" true;
            equal bool true (L.is_contiguous l);
            equal ints (L.shape l) (Bigarray.Genarray.dims g);
            equal
              (values (A.dtype a))
              (A.to_array a)
              (Array.of_list
                 (List.map (Bigarray.Genarray.get g) (indices (L.shape l)))))
  in
  match Option.bind m (fun m -> A.move m a) with
  | Some b -> check b
  | None -> check a

let test_bigarray_rank () =
  let a = A.create Rig.host D.Uint8 (Array.make 16 1) in
  is_some (A.bigarray Bigarray.int8_unsigned a);
  let a = A.create Rig.host D.Uint8 (Array.make 17 1) in
  is_none (A.bigarray Bigarray.int8_unsigned a)

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

(* An operand with no element passes the door wherever its memory lies. *)
let test_door_empty () =
  let host = zeros [| 0; 2 |] in
  let off = Option.get (A.move (M.Permute [| 1; 0 |]) (zeros [| 2; 0 |])) in
  let io = A.to_device (S.io_device ()) host in
  equal ~msg:"host" int ok (S.add (zeros [| 0; 2 |]) host host);
  equal ~msg:"a view" int ok (S.add (zeros [| 0; 2 |]) off host);
  equal ~msg:"off the host" int ok (S.add (zeros [| 0; 2 |]) io io)

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
        prop ~count:500
          "v takes a layout within its buffer's bytes on an aligned first \
           element, and only those"
          parts law_v;
        prop
          "create is C-contiguous at offset 0 over its bytes, its tail bits \
           zero"
          dtype_and_shape law_create;
        test "create refuses what contiguous refuses" test_create_refuses;
        prop "move is Layout.move over the same buffer" case law_move;
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
        prop ~count:1000 "bitcast follows its rule over any layout" bitcast_case
          law_bitcast;
        test "expect recovers the dtype or names both" test_expect;
        test "settle returns on pending work and names every refusal"
          test_settle;
        test "settle gives each refusal its own reason" test_settle_codes;
      ];
    group "elements"
      [
        prop "get reads what set stores" case law_set_get;
        test "an index outside the shape is refused" test_index;
        test "set refuses repeated elements and ints out of range"
          test_set_refuses;
        group "an int outside its dtype's range is refused" ranges;
        test "of_array refuses a count other than the shape's"
          test_of_array_refuses;
        test "memory the host does not address is refused" test_io_refuses;
        test "a dead buffer is refused" test_dead_access;
        test "memory held exclusive is refused" test_exclusive;
        test "bigarray refuses a dead buffer off the host" test_dead_bigarray;
        test "to_device refuses a dead buffer under no element" test_dead_empty;
        test "a refused copy allocates nothing" test_refused_copy;
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
        prop "copy keeps every byte of every element" raw law_copy_bits;
        prop "to_device copies and keeps the layout" case law_to_device;
        test "to_device moves sub-byte views through an io device"
          test_to_device_io;
      ];
    group "bigarray"
      [
        test "bigarray shares a contiguous host array's bytes" test_bigarray;
        test "of_bigarray shares the bigarray's bytes" test_of_bigarray;
        prop "bigarray views a C-contiguous host array" case law_bigarray;
        test "bigarray takes up to 16 axes" test_bigarray_rank;
      ];
    group "door"
      [
        test "a kernel reads its operands through the door" test_door_add;
        test "another dtype or shape at any position is refused untouched"
          test_door_positions;
        test "a written operand must be distinct and alone" test_door_written;
        test "dead, foreign and exclusive buffers are refused" test_door_buffers;
        test "an operand with no element passes the door, on the host or off it"
          test_door_empty;
        test "a read releases its claims" test_door_releases;
        test "a read survives a moving collection" test_door_moving_gc;
      ];
  ]

let () = exit (run "nx_array" tests)
