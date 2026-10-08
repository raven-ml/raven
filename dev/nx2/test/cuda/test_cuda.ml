(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Nx_cuda_support
module D = Nx_array.Dtype

let strf = Printf.sprintf

(* The first [n] differences [show i] names, of the [len] indices [same]
   rejects: [] when there is none. *)
let differences ?(n = 8) len same show =
  let rec go i acc k =
    if i = len || k = n then List.rev acc
    else if same i then go (i + 1) acc k
    else go (i + 1) (show i :: acc) (k + 1)
  in
  go 0 [] 0

let no_difference d = equal (list string) [] d

(* Floats from bytes *)

let f32 s i = Int32.float_of_bits (String.get_int32_le s (4 * i))
let f64 s i = Int64.float_of_bits (String.get_int64_le s (8 * i))
let bits32 x = Int32.bits_of_float x
let to32 x = Int32.float_of_bits (Int32.bits_of_float x)

let put32 xs =
  let b = Bytes.create (4 * List.length xs) in
  List.iteri (fun i x -> Bytes.set_int32_le b (4 * i) (bits32 x)) xs;
  Bytes.to_string b

let put64 xs =
  let b = Bytes.create (8 * List.length xs) in
  List.iteri
    (fun i x -> Bytes.set_int64_le b (8 * i) (Int64.bits_of_float x))
    xs;
  Bytes.to_string b

(* Every pair of these, as the operands' first elements. *)
let edges = [ 0.; -0.; infinity; neg_infinity; nan; 1.; -1.; 3.; 0.1 ]

let edges32 =
  edges
  @ List.map Int32.float_of_bits
      [ 0x00000001l; 0x007FFFFFl; 0x00800000l; 0x7F7FFFFFl; 0x80000001l ]

let edges64 =
  edges
  @ [ 4.9e-324; 2.2250738585072009e-308; Float.min_float; Float.max_float ]

let pairs xs = List.concat_map (fun a -> List.map (fun b -> (a, b)) xs) xs

(* The kernels' run over buffers *)

let launch g name ~n ps =
  S.run g
    (S.record (S.harness g)
       [ S.launch name ~grid:((n + 255) / 256, 1, 1) ~block:256 ps ])

(* Records *)

let xor a b =
  String.mapi (fun i c -> Char.chr (Char.code c lxor Char.code b.[i])) a

let records_in_order () =
  let g = S.gpu () in
  let n = 1 lsl 16 in
  let buf () = S.buffer g (4 * n) in
  let a = buf () and b = buf () and c = buf () and d = buf () in
  S.generate g a D.Uint32 Uniform ~seed:1;
  S.generate g b D.Uint32 Uniform ~seed:2;
  (* c = a ^ b, d = c ^ b: d is a only if both ran, in order. *)
  let copy x y out = S.floor_copy ~ins:[ x; y ] ~out in
  S.run g (S.record (S.harness g) [ copy a b c; copy c b d ]);
  equal string (xor (S.read a) (S.read b)) (S.read c);
  equal string (S.read a) (S.read d)

let floor_copy_xors () =
  let g = S.gpu () in
  let a = S.buffer g 4096 and b = S.buffer g 2048 and c = S.buffer g 4096 in
  S.generate g a D.Uint8 Uniform ~seed:3;
  S.generate g b D.Uint8 Uniform ~seed:4;
  S.run g (S.record (S.harness g) [ S.floor_copy ~ins:[ a; b ] ~out:c ]);
  let a = S.read a and b = S.read b in
  equal string (xor (String.sub a 0 2048) b ^ String.sub a 2048 2048) (S.read c)

(* Operands *)

let draws_are_functions () =
  let g = S.gpu () in
  let x = S.buffer g 4096 and y = S.buffer g 4096 in
  S.generate g x D.Float32 (Wide 20) ~seed:7;
  S.generate g y D.Float32 (Wide 20) ~seed:7;
  let x0 = S.read x in
  equal string x0 (S.read y);
  S.generate g y D.Float32 (Wide 20) ~seed:8;
  not_equal string x0 (S.read y)

let draws_in_range () =
  let g = S.gpu () in
  let n = 1 lsl 16 in
  let b = S.buffer g (4 * n) in
  S.generate g b D.Float32 Uniform ~seed:3;
  let s = S.read b in
  no_difference
    (differences n
       (fun i -> f32 s i >= -1. && f32 s i < 1.)
       (fun i -> strf "%d: %h" i (f32 s i)));
  S.generate g b D.Float32 (Wide 20) ~seed:3;
  let s = S.read b in
  no_difference
    (differences n
       (fun i ->
         let _, e = Float.frexp (f32 s i) in
         e - 1 >= -20 && e - 1 <= 20)
       (fun i -> strf "%d: %h" i (f32 s i)));
  S.generate g b D.Int32 Small ~seed:3;
  let s = S.read b in
  no_difference
    (differences n
       (fun i ->
         let x = Int32.to_int (String.get_int32_le s (4 * i)) in
         x >= -8 && x < 8)
       (fun i -> strf "%d: %ld" i (String.get_int32_le s (4 * i))))

(* Floats: the build's division and square root *)

(* Float32 results computed in float64, then rounded: a float64 quotient or root
   rounded to float32 is the correctly rounded float32 one, since 53 ≥ 2 × 24 +
   2 bits. NaNs compare as NaNs. *)
let same32 got want =
  (Float.is_nan got && Float.is_nan want) || bits32 got = bits32 want

let same64 got want =
  (Float.is_nan got && Float.is_nan want)
  || Int64.bits_of_float got = Int64.bits_of_float want

let div_sqrt ~n ~bytes ~dt ~spread ~edges ~put ~get ~round ~same name =
  let g = S.gpu () in
  let a = S.buffer g (bytes * n) and b = S.buffer g (bytes * n) in
  let q = S.buffer g (bytes * n) and r = S.buffer g (bytes * n) in
  S.generate g a dt (Wide spread) ~seed:11;
  S.generate g b dt (Wide spread) ~seed:12;
  let es = pairs edges in
  S.write
    (Rig.Buffer.view a ~first:0 ~length:(bytes * List.length es))
    (put (List.map fst es));
  S.write
    (Rig.Buffer.view b ~first:0 ~length:(bytes * List.length es))
    (put (List.map snd es));
  launch g name ~n [ A a; A b; A q; A r; W n ];
  let a = S.read a and b = S.read b and q = S.read q and r = S.read r in
  no_difference
    (differences n
       (fun i -> same (get q i) (round (get a i /. get b i)))
       (fun i ->
         strf "%d: %h / %h = %h, not %h" i (get a i) (get b i) (get q i)
           (round (get a i /. get b i))));
  no_difference
    (differences n
       (fun i -> same (get r i) (round (Float.sqrt (get a i))))
       (fun i ->
         strf "%d: sqrt %h = %h, not %h" i (get a i) (get r i)
           (round (Float.sqrt (get a i)))))

let div_sqrt_f32 () =
  div_sqrt ~n:(1 lsl 24) ~bytes:4 ~dt:D.Float32 ~spread:150 ~edges:edges32
    ~put:put32 ~get:f32 ~round:to32 ~same:same32 "div_sqrt_f32"

let div_sqrt_f64 () =
  div_sqrt ~n:(1 lsl 22) ~bytes:8 ~dt:D.Float64 ~spread:1100 ~edges:edges64
    ~put:put64 ~get:f64 ~round:Fun.id ~same:same64 "div_sqrt_f64"

(* The device's codecs *)

(* Every code of [dt] decodes, and drawn doubles encode, to the bits nx.array's
   host codecs give. The draws: every edge below, then doubles of a random sign
   and an exponent in [-spread, spread]. *)
let codecs (type s) (dt : (float, s) D.t) ~code ~spread =
  let g = S.gpu () in
  let codes = 1 lsl D.bits dt and n = 1 lsl 20 in
  let x = S.buffer g (8 * n) in
  S.generate g x D.Float64 (Wide spread) ~seed:13;
  let ties =
    List.concat_map
      (fun k ->
        List.concat_map
          (fun j ->
            let t = Float.ldexp 1. k *. (1. +. Float.ldexp 1. (-j)) in
            [ t; Float.succ t; Float.pred t ])
          [ 2; 3; 4; 8; 11; 12 ])
      (List.init 301 (fun k -> k - 160))
  in
  let edges =
    [ 0.; infinity; nan; Float.max_float ] @ ties
    |> List.concat_map (fun x -> [ x; -.x ])
  in
  let xs =
    put64 edges ^ String.sub (S.read x) 0 (8 * (n - List.length edges))
  in
  S.write x xs;
  let dec = S.buffer g (8 * codes) and enc = S.buffer g (D.bits dt / 8 * n) in
  launch g "codecs" ~n [ A x; A dec; A enc; W n; D (D.code dt, 0) ];
  let host_dec =
    let cs = Nx_array.of_array code [| codes |] (Array.init codes Fun.id) in
    Nx_array.to_array (Option.get (Nx_array.bitcast dt cs))
  in
  let dec = S.read dec in
  no_difference
    (differences codes
       (fun c ->
         Int64.bits_of_float (f64 dec c) = Int64.bits_of_float host_dec.(c))
       (fun c -> strf "%#x decodes to %h, not %h" c (f64 dec c) host_dec.(c)));
  let host_enc =
    let v = Nx_array.of_array dt [| n |] (Array.init n (f64 xs)) in
    Nx_array.to_array (Option.get (Nx_array.bitcast code v))
  in
  let enc = S.read enc in
  let get i =
    if codes > 256 then String.get_uint16_le enc (2 * i) else Char.code enc.[i]
  in
  no_difference
    (differences n
       (fun i -> get i = host_enc.(i))
       (fun i ->
         strf "%h encodes to %#x, not %#x" (f64 xs i) (get i) host_enc.(i)))

let codecs_f16 () = codecs D.Float16 ~code:D.Uint16 ~spread:30
let codecs_bf16 () = codecs D.Bfloat16 ~code:D.Uint16 ~spread:140
let codecs_e4m3 () = codecs D.Float8_e4m3fn ~code:D.Uint8 ~spread:20
let codecs_e5m2 () = codecs D.Float8_e5m2 ~code:D.Uint8 ~spread:30

(* Determinism *)

let sms_of s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)))

(* Beside the hog, work runs only on the SMs the hog left free. *)
let hog_holds_its_sms () =
  let g = S.gpu () in
  let blocks = 4 * S.sms g in
  let sm = S.buffer g (4 * blocks) in
  S.write sm (String.make (4 * blocks) '\xff');
  let where =
    S.record (S.harness g)
      [ S.launch "where" ~grid:(blocks, 1, 1) ~block:256 [ A sm ] ]
  in
  let hog = S.hog g ~ns:2_000_000 in
  S.run g ~beside:hog where;
  let held = S.held_sms hog in
  equal int
    (Int.max 1 (S.sms g / 2))
    (List.length (List.sort_uniq compare held));
  let ran = sms_of (S.read sm) in
  equal (list int) [] (List.filter (fun s -> s < 0 || s >= S.sms g) ran);
  equal (list int) [] (List.filter (fun s -> List.mem s held) ran)

let tests =
  [
    group "records"
      [
        test "run in order" records_in_order;
        test "the copy floor XORs its inputs" floor_copy_xors;
      ];
    group "operands"
      [
        test "draws are functions of the seed" draws_are_functions;
        test "draws stay in range" draws_in_range;
        test "draws refuse sub-byte dtypes" (fun () ->
            let g = S.gpu () in
            raises
              (Invalid_argument "Nx_cuda_support.generate: int4 is not drawn")
              (fun () -> S.generate g (S.buffer g 16) D.Int4 Uniform ~seed:0));
      ];
    group "build"
      [
        test "float32 division and square root round correctly" div_sqrt_f32;
        test "float64 division and square root round correctly" div_sqrt_f64;
      ];
    group "codecs"
      [
        test "float16 as the host's" codecs_f16;
        test "bfloat16 as the host's" codecs_bf16;
        test "float8 e4m3fn as the host's" codecs_e4m3;
        test "float8 e5m2 as the host's" codecs_e5m2;
      ];
    group "determinism" [ test "the hog holds its SMs" hog_holds_its_sms ];
  ]

let () =
  S.hold_gpu ();
  exit (run "nx.cuda" tests)
