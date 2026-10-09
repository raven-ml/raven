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

(* Contract *)

(* How a case draws its operands: floats of exponents in [-e, e] (integers over
   their range); those with the second half of each row of a negating its first,
   so that sums cancel; or those with NaN, infinities, -0 and subnormals
   (integer extremes) at every seventh element. *)
type draw = Spread of int | Cancel | Edges

(* A contraction y[z, i, j] = init[z, i, j] + Σ_k a[z, i, k] · b[z, j, k] of
   [batch], [m], [n], [k], a and b of [dt] in the layout whose [`K] or [`Free]
   axis is contiguous, [pad] elements between batch elements, accumulated in
   [acc], y of [out]. *)
type case = {
  dt : any;
  acc : any;
  out : any;
  batch : int;
  m : int;
  n : int;
  k : int;
  la : [ `K | `Free ];
  lb : [ `K | `Free ];
  init : [ `None | `Bias | `Full ];
  draw : draw;
  pad : int;
}

and any = D.any

let code (D.Any dt) = D.code dt
let width (D.Any dt) = D.bits dt / 8

let case_name c =
  let name (D.Any dt) = D.name dt in
  let l = function `K -> "k" | `Free -> "f" in
  let draw =
    match c.draw with
    | Spread e -> strf "spread %d" e
    | Cancel -> "cancel"
    | Edges -> "edges"
  in
  strf "%s->%s acc %s %dx%dx%dx%d %s%s%s %s pad %d" (name c.dt) (name c.out)
    (name c.acc) c.batch c.m c.n c.k (l c.la) (l c.lb)
    (match c.init with `None -> "" | `Bias -> " bias" | `Full -> " init")
    draw c.pad

(* An operand of [rows] x [k] per batch element holding [values] (their bytes
   with k contiguous), laid out with its k or its rows contiguous, [pad]
   elements between batch elements, with [skip] elements before its first. *)
let operand g dt values ~batch ~rows ~k ~layout ~skip ~pad =
  let w = width dt in
  let n = batch * ((rows * k) + pad) in
  let buffer = S.buffer g (w * (skip + Int.max 1 n)) in
  let strides =
    match layout with
    | `K -> [| (rows * k) + pad; k; 1 |]
    | `Free -> [| (rows * k) + pad; 1; rows |]
  in
  let laid = Bytes.make (w * n) '\000' in
  for z = 0 to batch - 1 do
    for r = 0 to rows - 1 do
      for q = 0 to k - 1 do
        let src = (z * rows * k) + (r * k) + q in
        let dst = (z * strides.(0)) + (r * strides.(1)) + (q * strides.(2)) in
        Bytes.blit_string values (w * src) laid (w * dst) w
      done
    done
  done;
  if n > 0 then
    S.write
      (Rig.Buffer.view buffer ~first:(w * skip) ~length:(w * n))
      (Bytes.to_string laid);
  {
    S.buffer;
    dtype = code dt;
    shape = [| batch; rows; k |];
    strides;
    first = w * skip;
  }

let view (o : S.operand) =
  let all = S.read o.buffer in
  {
    Nx_gpu_ref.bytes = String.sub all o.first (String.length all - o.first);
    dtype = o.dtype;
    strides = o.strides;
  }

(* The codes of a float dtype's NaN, infinities, -0 and two subnormals, and an
   integer dtype's extremes and -1, little-endian. float8 e4m3fn has no
   infinity: its largest finite values stand in. *)
let edges (D.Any d) =
  let codes =
    match d with
    | D.Float64 ->
        [
          0x7ff8000000000000;
          0x7ff0000000000000;
          -0x10000000000000;
          min_int;
          1;
          0x8000000000000;
        ]
    | D.Float32 ->
        [ 0x7fc00000; 0x7f800000; 0xff800000; 0x80000000; 1; 0x400000 ]
    | D.Float16 -> [ 0x7e00; 0x7c00; 0xfc00; 0x8000; 1; 0x200 ]
    | D.Bfloat16 -> [ 0x7fc0; 0x7f80; 0xff80; 0x8000; 1; 0x40 ]
    | D.Float8_e4m3fn -> [ 0x7f; 0x7e; 0xfe; 0x80; 1; 4 ]
    | D.Float8_e5m2 -> [ 0x7e; 0x7c; 0xfc; 0x80; 1; 2 ]
    | _ ->
        let bits = D.bits d in
        let signed = D.is D.Signed d in
        if signed then [ 1 lsl (bits - 1); (1 lsl (bits - 1)) - 1; -1 ]
        else [ (1 lsl bits) - 1; 0; 1 lsl (bits - 1) ]
  in
  let w = D.bits d / 8 in
  List.map
    (fun c -> String.init w (fun i -> Char.chr ((c lsr (8 * i)) land 255)))
    codes

(* [rows] x [k] elements of [dt] drawn on the GPU from [seed] by [draw], for
   [batch] batch elements; [cancel] makes a's rows cancel. Integers draw over
   their range, so that their sums wrap. *)
let values g dt ~batch ~rows ~k ~draw ~seed ~cancel =
  let (D.Any d) = dt in
  let n = batch * rows * k and w = width dt in
  let gen =
    match draw with
    | _ when not (D.is D.Float d) -> S.Uniform
    | Spread e -> S.Wide e
    | Cancel | Edges -> S.Wide 6
  in
  let tmp = S.buffer g (w * Int.max 1 n) in
  S.generate g tmp d gen ~seed;
  let v = Bytes.of_string (String.sub (S.read tmp) 0 (w * n)) in
  (match draw with
  | Cancel when cancel && D.is D.Float d ->
      (* The sign bit is the top byte's top bit. *)
      for r = 0 to (batch * rows) - 1 do
        for q = 0 to (k / 2) - 1 do
          let src = w * ((r * k) + q) and dst = w * ((r * k) + (k / 2) + q) in
          Bytes.blit v src v dst w;
          let top = dst + w - 1 in
          Bytes.set_uint8 v top (Bytes.get_uint8 v top lxor 0x80)
        done
      done
  | Edges ->
      let es = edges dt in
      for i = 0 to n - 1 do
        if i mod 7 = seed mod 7 then
          Bytes.blit_string
            (List.nth es (i / 7 mod List.length es))
            0 v (w * i) w
      done
  | _ -> ());
  Bytes.to_string v

(* Plans and runs the case on operands in its layouts, [skip] elements into
   their buffers; a, b, init, y and the plan. *)
let contract g c ?(skip = 0) ?(seed = 1) () =
  let draw = c.draw and pad = c.pad and batch = c.batch and k = c.k in
  let a =
    let v = values g c.dt ~batch ~rows:c.m ~k ~draw ~seed ~cancel:true in
    operand g c.dt v ~batch ~rows:c.m ~k ~layout:c.la ~skip ~pad
  in
  let b =
    let v =
      values g c.dt ~batch ~rows:c.n ~k ~draw ~seed:(seed + 1) ~cancel:false
    in
    operand g c.dt v ~batch ~rows:c.n ~k ~layout:c.lb ~skip ~pad
  in
  let init_values rows =
    values g c.out ~batch:1 ~rows ~k:1 ~draw ~seed:(seed + 2) ~cancel:false
  in
  let wy = width c.out and ny = c.batch * c.m * c.n in
  let y =
    {
      S.buffer = S.buffer g (wy * Int.max 1 ny);
      dtype = code c.out;
      shape = [| c.batch; c.m; c.n |];
      strides = [| c.m * c.n; c.n; 1 |];
      first = 0;
    }
  in
  let init =
    match c.init with
    | `None -> None
    | `Bias ->
        let buffer = S.buffer g (wy * Int.max 1 c.n) in
        S.write buffer (init_values (Int.max 1 c.n));
        Some { y with buffer; strides = [| 0; 0; 1 |] }
    | `Full ->
        let buffer = S.buffer g (wy * Int.max 1 ny) in
        S.write buffer (init_values (Int.max 1 ny));
        Some { y with buffer }
  in
  match
    S.contract g ~a ~b ?init ~y
      ~batch:[ (0, 0) ]
      ~contracting:[ (2, 2) ]
      ~acc:(code c.acc) ()
  with
  | None -> failf "the plan declines %s" (case_name c)
  | Some p ->
      S.run g p;
      (a, b, init, y, p)

let within_bound c =
  let g = S.gpu () in
  let a, b, init, y, _ = contract g c () in
  let r =
    Nx_gpu_ref.contract ~a:(view a) ~b:(view b) ?init:(Option.map view init)
      ~y:(view y) ~batch:c.batch ~m:c.m ~n:c.n ~k:c.k ~acc:(code c.acc)
      ~samples:4096 ()
  in
  equal ~msg:(strf "wrong outputs, the first %d" r.at) int 0 r.wrong;
  at_most ~msg:(strf "worst at output %d" r.at) float_exact ~than:1. r.worst

let shapes =
  [
    (1, 1, 1, 1);
    (1, 1, 64, 300);
    (1, 16, 128, 64);
    (1, 17, 129, 65);
    (1, 127, 128, 33);
    (1, 128, 127, 32);
    (1, 129, 1, 0);
    (1, 200, 257, 1000);
    (3, 33, 70, 129);
    (1, 64, 64, 8192);
    (1, 1, 64, 20000);
    (1, 5, 300, 4100);
  ]

let configs =
  let open D in
  [
    (Any Bfloat16, Any Float32, Any Bfloat16);
    (Any Bfloat16, Any Float32, Any Float32);
    (Any Float16, Any Float32, Any Float16);
    (Any Float8_e4m3fn, Any Float32, Any Bfloat16);
    (Any Float32, Any Float32, Any Float32);
    (Any Float64, Any Float64, Any Float64);
    (Any Int8, Any Int32, Any Int32);
    (Any Int8, Any Int32, Any Int8);
    (Any Int8, Any Int32, Any Int64);
    (Any Int16, Any Int32, Any Int64);
    (Any Uint8, Any Int32, Any Int32);
    (Any Int16, Any Int64, Any Int64);
    (Any Int32, Any Int32, Any Int32);
  ]

let cases_of ?(layouts = [ (`K, `K); (`K, `Free); (`Free, `K); (`Free, `Free) ])
    () =
  List.concat_map
    (fun (dt, acc, out) ->
      List.concat_map
        (fun (batch, m, n, k) ->
          List.mapi
            (fun i (la, lb) ->
              let init =
                match i mod 3 with 0 -> `None | 1 -> `Bias | _ -> `Full
              in
              let draw = Spread (if dt = D.Any D.Float16 then 2 else 6) in
              { dt; acc; out; batch; m; n; k; la; lb; init; draw; pad = 0 })
            layouts)
        shapes)
    configs

(* The bits depend on the shape alone. The same values in the other layouts,
   behind an element that breaks the 16-byte vectors, and twice beside the hog,
   give the same bytes. *)
let same_bits c =
  let g = S.gpu () in
  let _, _, _, y, p = contract g c () in
  let want = S.read y.buffer in
  List.iter
    (fun (la, lb, skip) ->
      let _, _, _, y, _ = contract g { c with la; lb } ~skip () in
      equal
        ~msg:(strf "layouts with %d skipped" skip)
        string want (S.read y.buffer))
    [ (`Free, `K, 0); (`K, `Free, 0); (`Free, `Free, 0); (`K, `K, 1) ];
  S.run g ~beside:(S.hog g ~ns:2_000_000) p;
  equal ~msg:"beside the hog" string want (S.read y.buffer)

let shape_cases =
  List.concat_map
    (fun (dt, acc, out) ->
      List.map
        (fun (batch, m, n, k) ->
          let draw = Spread 6 and pad = 0 in
          {
            dt;
            acc;
            out;
            batch;
            m;
            n;
            k;
            la = `K;
            lb = `K;
            init = `Bias;
            draw;
            pad;
          })
        [
          (1, 17, 129, 65);
          (2, 130, 200, 300);
          (1, 64, 64, 8192);
          (1, 3, 100, 5000);
        ])
    configs

(* Edge inputs: wide and no spread, cancellation, special values and integer
   extremes, batch elements apart by 0, 1 and 4 elements. *)
let edge_cases =
  let draws = [ Spread 0; Spread 8; Spread 40; Cancel; Edges ] in
  List.concat_map
    (fun (dt, acc, out) ->
      List.concat_map
        (fun (batch, m, n, k) ->
          List.mapi
            (fun i draw ->
              let pad = List.nth [ 0; 1; 4 ] (i mod 3) in
              let la, lb = if i mod 2 = 0 then (`K, `K) else (`Free, `K) in
              { dt; acc; out; batch; m; n; k; la; lb; init = `Bias; draw; pad })
            draws)
        [
          (2, 17, 129, 65); (2, 64, 64, 64); (1, 1, 64, 300); (1, 64, 64, 8192);
        ])
    configs

(* An int32 sum reaches a wider output as a cast from int32 does, by its sign:
   -1 times 1 over 64 terms is -64. *)
let sign_extends () =
  let g = S.gpu () in
  List.iter
    (fun (D.Any d as dt) ->
      let w = D.bits d / 8 in
      let ones c = String.concat "" (List.init 64 (fun _ -> String.make w c)) in
      let one = String.init w (fun i -> if i = 0 then '\001' else '\000') in
      let a =
        operand g dt (ones '\xff') ~batch:1 ~rows:1 ~k:64 ~layout:`K ~skip:0
          ~pad:0
      in
      let b =
        operand g dt
          (String.concat "" (List.init 64 (fun _ -> one)))
          ~batch:1 ~rows:1 ~k:64 ~layout:`K ~skip:0 ~pad:0
      in
      let y : S.operand =
        {
          buffer = S.buffer g 8;
          dtype = D.code D.Int64;
          shape = [| 1; 1; 1 |];
          strides = [| 1; 1; 1 |];
          first = 0;
        }
      in
      match
        S.contract g ~a ~b ~y
          ~batch:[ (0, 0) ]
          ~contracting:[ (2, 2) ]
          ~acc:(D.code D.Int32) ()
      with
      | None -> fail "the plan declines"
      | Some p ->
          S.run g p;
          equal ~msg:(D.name d) int64 (-64L)
            (String.get_int64_le (S.read y.buffer) 0))
    [ D.Any D.Int8; D.Any D.Int16 ]

(* Every dtype quadruple *)

let pp_dtype ppf (D.Any d) = Format.pp_print_string ppf (D.name d)
let named = List.map (fun (D.Any d as x) -> (D.name d, x)) D.all
let dt name = List.assoc name named

(* [rows] x [k] elements of [dt] per batch element, one batch element shared by
   all when [broadcast], drawn on the GPU where it draws [dt] and as random
   bytes otherwise. *)
let any_operand g (D.Any d as dt) ~batch ~rows ~k ~broadcast ~seed : S.operand =
  let w = Int.max 1 (D.bits d / 8)
  and stored = if broadcast then 1 else batch in
  let n = stored * rows * k in
  let bytes =
    match d with
    | Float4_e2m1fn | Int4 | Uint4 | Complex128 | Complex64 | Bit ->
        let st = Random.State.make [| seed |] in
        String.init (w * n) (fun _ -> Char.chr (Random.State.int st 256))
    | _ ->
        let draw = Spread (if dt = D.Any D.Float16 then 2 else 6) in
        values g dt ~batch:stored ~rows ~k ~draw ~seed ~cancel:false
  in
  let buffer = S.buffer g (Int.max 1 (w * n)) in
  if n > 0 then S.write buffer bytes;
  let strides = [| (if broadcast then 0 else rows * k); k; 1 |] in
  { buffer; dtype = D.code d; shape = [| batch; rows; k |]; strides; first = 0 }

(* For every operand, accumulator and output dtypes, the plan declines, or its
   result is within the error bound (float sums) or exact (integer sums), the
   output written as a cast from the accumulator writes it. *)
let every_quadruple ((a, b, acc, out), init, (batch, m, n, k), broadcast) =
  let g = S.gpu () in
  let x = any_operand g a ~batch ~rows:m ~k ~broadcast:false ~seed:1 in
  let w = any_operand g b ~batch ~rows:n ~k ~broadcast ~seed:2 in
  let operand d =
    any_operand g d ~batch ~rows:m ~k:n ~broadcast:false ~seed:3
  in
  let init = Option.map operand init in
  let (D.Any o) = out in
  let wy = Int.max 1 (D.bits o / 8) in
  let buffer = S.buffer g (wy * batch * m * n) in
  S.write buffer (String.make (wy * batch * m * n) '\xee');
  let y : S.operand =
    {
      buffer;
      dtype = D.code o;
      shape = [| batch; m; n |];
      strides = [| m * n; n; 1 |];
      first = 0;
    }
  in
  let float_dt (D.Any d) = D.is D.Float d in
  match
    S.contract g ~a:x ~b:w ?init ~y
      ~batch:[ (0, 0) ]
      ~contracting:[ (2, 2) ]
      ~acc:(code acc) ()
  with
  | None -> collect "declines"
  | Some p ->
      cover "an init" (Option.is_some init);
      cover "a float sum" (float_dt acc);
      cover "an integer sum" (not (float_dt acc));
      cover "an output of the other kind" (float_dt acc <> float_dt out);
      cover "an unsigned accumulator" (acc = dt "uint32" || acc = dt "uint64");
      cover "a broadcast operand" broadcast;
      S.run g p;
      let r =
        Nx_gpu_ref.contract ~a:(view x) ~b:(view w) ?init:(Option.map view init)
          ~y:(view y) ~batch ~m ~n ~k ~acc:(code acc) ~samples:512 ()
      in
      equal ~msg:(strf "wrong outputs, the first %d" r.at) int 0 r.wrong;
      at_most ~msg:(strf "worst at output %d" r.at) float_exact ~than:1. r.worst

let quadruples =
  let open Gen in
  let any = of_list ~pp:pp_dtype D.all in
  let some names = of_list ~pp:pp_dtype (List.map dt names) in
  let floats =
    some
      [
        "float64";
        "float32";
        "float16";
        "bfloat16";
        "float8_e4m3fn";
        "float8_e5m2";
      ]
  in
  let ints =
    some
      [
        "int64";
        "uint64";
        "int32";
        "uint32";
        "int16";
        "uint16";
        "int8";
        "uint8";
        "bool";
      ]
  in
  let operand = frequency [ (2, floats); (2, ints); (1, any) ] in
  let acc =
    frequency
      [
        (3, some [ "float32"; "float64"; "int32"; "uint32"; "int64"; "uint64" ]);
        (1, any);
      ]
  in
  let shape = of_list [ (2, 40, 70, 300); (2, 3, 70, 300) ] in
  let pp ppf ((a, b, acc, out), init, (batch, m, n, k), broadcast) =
    Format.fprintf ppf "%a x %a acc %a -> %a%s, %dx%dx%dx%d%s" pp_dtype a
      pp_dtype b pp_dtype acc pp_dtype out
      (match init with None -> "" | Some (D.Any d) -> ", init " ^ D.name d)
      batch m n k
      (if broadcast then ", b broadcast" else "")
  in
  with_pp pp (quad (quad operand operand acc any) (option operand) shape bool)

(* A float operand or init wider than a float accumulator would be rounded
   before it is summed, and the kernels read no complex element: the plan
   declines. *)
let declines (a, b, acc, out, init) =
  let g = S.gpu () in
  let x = any_operand g (dt a) ~batch:1 ~rows:3 ~k:5 ~broadcast:false ~seed:1 in
  let w = any_operand g (dt b) ~batch:1 ~rows:4 ~k:5 ~broadcast:false ~seed:2 in
  let y =
    any_operand g (dt out) ~batch:1 ~rows:3 ~k:4 ~broadcast:false ~seed:3
  in
  let init =
    Option.map
      (fun d ->
        any_operand g (dt d) ~batch:1 ~rows:3 ~k:4 ~broadcast:false ~seed:4)
      init
  in
  let p =
    S.contract g ~a:x ~b:w ?init ~y
      ~batch:[ (0, 0) ]
      ~contracting:[ (2, 2) ]
      ~acc:(code (dt acc))
      ()
  in
  equal ~msg:"the plan" (option pass) None p

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
    group "contract"
      [
        cases ~name:case_name "within the error bound" (cases_of ())
          within_bound;
        cases ~name:case_name "bits fixed by the shape" shape_cases same_bits;
        cases ~name:case_name "edge inputs within the error bound" edge_cases
          within_bound;
        test "an int32 sum sign-extends into int64" sign_extends;
        prop ~count:400 "every dtype quadruple: declines or within the bound"
          quadruples every_quadruple;
        cases
          ~name:(fun (a, b, acc, out, init) ->
            strf "%s x %s acc %s -> %s%s" a b acc out
              (Option.fold ~none:"" ~some:(( ^ ) ", init ") init))
          "declines what it would round or cannot read"
          [
            ("float64", "float64", "float32", "float32", None);
            ("float64", "float32", "float32", "float64", None);
            ("complex64", "complex64", "float32", "float32", None);
            ("float32", "complex64", "float64", "float64", None);
            ("float32", "float32", "complex64", "complex64", None);
            ("float32", "float32", "float32", "complex64", None);
            ("float32", "float32", "float32", "float64", Some "float64");
            ("bfloat16", "bfloat16", "float32", "float32", Some "float64");
            ("float32", "float32", "float32", "float32", Some "int32");
          ]
          declines;
      ];
  ]

let () =
  S.hold_gpu ();
  exit (run "nx.cuda" tests)
