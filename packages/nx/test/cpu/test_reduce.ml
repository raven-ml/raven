(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's reductions and scans, as nx_cpu.mli states them: their order,
   bit for bit, against a reference built here, and the same bits on one
   thread and under every layout; Arg, Logsumexp and Moments within their
   bounds, their two passes' bits and their network's bits; programs'
   outputs, over plain and padded loads; chunks' carries; refusals; and
   guard pages. *)

open Windtrap
open Nx_array_gen
open Reduce_cases
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

(* Views *)

(* Floats, whose order and NaNs the laws check, come twice as often. *)
let computed = Gen.with_pp pp_case (case_of (floats @ floats @ base))

(* Operands of a few hundred thousand elements, which the job gives several
   threads. *)
let large =
  Gen.with_pp pp_case
    (case_of
       ~shapes:
         (Gen.of_list
            [
              [| 400_000 |];
              [| 640; 640 |];
              [| 3; 150_000 |];
              [| 150_000; 3 |];
              [| 64; 8_000 |];
            ])
       (base @ D.[ Any Bfloat16; Any Int4 ]))

let computed_examples =
  [
    example Sum f32 [| 0; 3 |] [| 1 |];
    example Sum f32 [| 3; 0 |] [| 1 |];
    example Sum f32 [| 5 |] [||];
    example ~specials:20 Sum f32 [| 5000 |] [| 0 |];
    example Max f32 [| 100; 3 |] [| 1 |];
    (* Short rows of few terms, which units take in bands of rows. *)
    example ~specials:20 Sum f32 [| 46; 13; 2 |] [| 2 |];
    example ~specials:20 Max f64 [| 46; 13; 2 |] [| 2 |];
    example
      ~view:
        {
          perm = [| 2; 0; 1 |];
          stepped = true;
          reversed = Some 0;
          broadcast = true;
        }
      Sum f64 [| 3; 4; 5 |] [| 1 |];
  ]

(* NaNs made from numbers, before a NaN term and with none: opposite
   infinities summed, zero times an infinity, along one block, past one,
   past a chunk, in a few-term reduction's rows and through a stepped view. *)
let made_examples =
  let inf = Float.infinity and ninf = Float.neg_infinity in
  let nan = Int64.float_of_bits 0x7FF0_0000_0000_0042L in
  let case ?(views = []) ?(axes = [| 0 |]) scan monoid x =
    { scan; monoid; axes; x; views }
  in
  let f32 = vector_of D.Float32 and f64 = vector_of D.Float64 in
  let stepped = [ "stepped" ] in
  let at n pairs =
    Array.init n (fun i -> Option.value ~default:1. (List.assoc_opt i pairs))
  in
  let rows = at 128 [ (10, inf); (11, ninf) ] in
  let rows = A.Any (A.of_array D.Float32 [| 64; 2 |] rows) in
  [
    case false Sum (f32 [| inf; ninf |]);
    case false Prod (f32 [| 0.; inf |]);
    case true Sum (f32 [| inf; ninf; 1. |]);
    case ~views:stepped false Sum
      (f64 ~stepped:true [| 1.; inf; 2.; ninf; 3. |]);
    case ~views:stepped true Prod
      (f64 ~stepped:true [| 2.; 0.; inf; 1.; nan |]);
    case false Sum (f32 (at 1025 [ (0, inf); (1024, ninf) ]));
    case true Sum (f32 (at 4097 [ (3, inf); (4096, ninf) ]));
    case ~axes:[| 1 |] false Sum rows;
  ]

let large_examples =
  [
    example ~scan:true Sum f32 [| 400_000 |] [| 0 |];
    example Sum f32 [| 400_000 |] [| 0 |];
    example Sum f32 [| 4000; 13; 2 |] [| 2 |];
    example ~scan:true Sum (D.Any D.Bfloat16) [| 400_000 |] [| 0 |];
    (* Rows of more than 512 bytes of outputs and more than a chunk of
       terms, which one thread totals in pieces. *)
    example ~scan:true Sum f32 [| 4100; 300 |] [| 0 |];
    example ~scan:true Sum (D.Any D.Bfloat16) [| 4100; 300 |] [| 0 |];
  ]

(* A sum and a maximum, a running sum and product, of 9000 float32 terms
   with opposite infinities at 100 and 101 and NaNs at 5000 and 6000, past
   the first block and chunk: a NaN made from numbers before the NaN terms,
   a NaN where one reaches. *)
let far_nan_examples =
  let n = 9000 in
  let r = Random.State.make [| 7 |] in
  let xs = Array.init n (fun _ -> Int32.bits_of_float (full r 23)) in
  xs.(100) <- Int32.bits_of_float infinity;
  xs.(101) <- Int32.bits_of_float neg_infinity;
  xs.(5000) <- 0x7FC12345l;
  xs.(6000) <- 0xFFC00042l;
  let bytes = Bytes.create (4 * n) in
  Array.iteri (fun i v -> Bytes.set_int32_le bytes (4 * i) v) xs;
  let x =
    A.Any
      (A.v D.Float32 (L.contiguous [| n |])
         (Rig.Buffer.of_string (Bytes.to_string bytes)))
  in
  List.map
    (fun (scan, monoid) -> { scan; monoid; axes = [| 0 |]; x; views = [] })
    [ (false, Sum); (false, Max); (true, Sum); (true, Prod) ]

(* Products and sums of 8- and 16-bit integers at their extremes, which
   wrap in their width: 0xFFFF times 0xFFFF times 3 is 3 at uint16. *)
let wrap_examples =
  let vector (type s) (dt : (int, s) D.t) monoid xs =
    {
      scan = false;
      monoid;
      axes = [| 0 |];
      x = A.Any (A.of_array dt [| Array.length xs |] xs);
      views = [];
    }
  in
  [
    vector D.Uint16 Prod [| 0xFFFF; 0xFFFF; 3 |];
    vector D.Int16 Prod [| -32768; -1 |];
    vector D.Int16 Prod [| 32767; 32767 |];
    vector D.Uint8 Prod [| 255; 255 |];
    vector D.Int8 Prod [| -128; -1 |];
    vector D.Uint16 Sum [| 0xFFFF; 0xFFFF |];
    vector D.Int16 Sum [| 32767; 1 |];
  ]

(* Rows of -0 alone, summed from +0: +0. *)
let zero_example =
  {
    scan = false;
    monoid = Sum;
    axes = [| 1 |];
    x = A.Any (A.of_array D.Float32 [| 64; 9 |] (Array.make (64 * 9) (-0.)));
    views = [];
  }

(* The reference *)

(* [ts] reduced in nx.cpu's order: blocks of 1024 terms in 16 lanes from
   the identity, or from their first term where [first], term t into lane
   t mod 16, the lanes' tree, in which a lane holding no term takes no
   part, then the blocks' tree whose left part holds the largest power of
   two of blocks below their count. *)
let ordered ?(first = false) (f, id) ts =
  let k = Array.length ts in
  let block x =
    let l = Array.make 16 id in
    let hi = min k ((x + 1) * 1024) in
    for t = x * 1024 to hi - 1 do
      l.(t mod 16) <-
        (if first && t - (x * 1024) < 16 then ts.(t) else f l.(t mod 16) ts.(t))
    done;
    let used = min 16 (hi - (x * 1024)) in
    List.iter
      (fun w ->
        for i = 0 to w - 1 do
          if i + w < used then l.(i) <- f l.(i) l.(i + w)
        done)
      [ 8; 4; 2; 1 ];
    l.(0)
  in
  let rec tree s n =
    if n = 1 then s.(0)
    else
      let h = ref 1 in
      while 2 * !h < n do
        h := 2 * !h
      done;
      f (tree s !h) (tree (Array.sub s !h (n - !h)) (n - !h))
  in
  let n = (k + 1023) / 1024 in
  if n = 0 then id else tree (Array.init n block) n

(* [ts] scanned in nx.cpu's order: chunks of 4096 terms from the start, each
   result the carry combined with the chunk's terms up to it, left to right,
   the carry into a chunk the carry into the one before combined with that
   chunk's total in [ordered]'s order. *)
let scanned ((f, id) as m) ts =
  let n = Array.length ts in
  let out = Array.make n id in
  let carry = ref id in
  for c = 0 to ((n + 4095) / 4096) - 1 do
    let lo = 4096 * c and hi = min n (4096 * (c + 1)) in
    let acc = ref !carry in
    for i = lo to hi - 1 do
      acc := f !acc ts.(i);
      out.(i) <- !acc
    done;
    carry := f !carry (ordered m (Array.sub ts lo (hi - lo)))
  done;
  Array.map want_of out

(* The carry into each chunk of [ts] scanned as [scanned] does: the
   identity into the first. *)
let carries ((f, id) as m) ts =
  let n = Array.length ts in
  let k = (n + 4095) / 4096 in
  let out = Array.make (max 1 k) id in
  for c = 1 to k - 1 do
    out.(c) <- f out.(c - 1) (ordered m (Array.sub ts (4096 * (c - 1)) 4096))
  done;
  out

(* The expected results of [c] in C order of the result's indices. *)
let expected (c : case) =
  let (A.Any x) = c.x in
  let m = op (D.Any (A.dtype x)) c.monoid in
  results c (terms c.x) ~fill:Some_nan
    ~reduce:(fun o -> want_of (ordered m o))
    ~scan:(scanned m)

(* Complex numbers *)

(* A complex term: its parts' values, which a double holds exactly, and
   their bits as stored, a float32's sign-extended. A number with a NaN part
   is a NaN. *)
type z = { re : float; im : float; bits : int64 * int64 }

let znan z = Float.is_nan z.re || Float.is_nan z.im

(* The number of parts [re] and [im], of float32 where [narrow]. *)
let zof narrow re im =
  let b v =
    if narrow then Int64.of_int32 (Int32.bits_of_float v)
    else Int64.bits_of_float v
  in
  { re; im; bits = (b re, b im) }

let narrow (A.Any x) = D.equal (A.dtype x) D.Complex64

(* [x]'s complex elements in C order of indices. *)
let zterms (A.Any x as a) =
  let c = A.copy x in
  let bits =
    match A.dtype c with
    | D.Complex64 ->
        let p = Option.get (A.bitcast D.Float32 c) in
        Array.map Int64.of_int32 (A.to_array (Option.get (A.bitcast D.Int32 p)))
    | D.Complex128 ->
        let p = Option.get (A.bitcast D.Float64 c) in
        A.to_array (Option.get (A.bitcast D.Int64 p))
    | _ -> invalid_arg "zterms: not a complex array"
  in
  let v b =
    if narrow a then Int32.float_of_bits (Int64.to_int32 b)
    else Int64.float_of_bits b
  in
  Array.init
    (Array.length bits / 2)
    (fun i ->
      let rb = bits.(2 * i) and ib = bits.((2 * i) + 1) in
      { re = v rb; im = v ib; bits = (rb, ib) })

(* The float32 next to [x] toward [y]. *)
let step32 x y =
  if x = 0. then Float.copy_sign 0x1p-149 y
  else
    let b = Int32.bits_of_float x in
    Int32.float_of_bits
      (if y > x = (x > 0.) then Int32.succ b else Int32.pred b)

(* [a b + c] of float32s rounded once to float32. The exact product is a
   double; their sum [s] rounds as the exact value does, since every
   float32 midpoint is a double, but at a midpoint, where the sum's error
   [e] (Knuth's TwoSum) decides. *)
let fma32 a b c =
  let p = Sys.opaque_identity (a *. b) in
  let s = p +. c in
  let r = round32 s in
  if r = s || not (Float.is_finite s) then r
  else
    let t = s -. p in
    let e = (p -. (s -. t)) +. (c -. t) in
    let o = step32 r s in
    if (r +. o) /. 2. <> s || e = 0. then r
    else if e > 0. = (o > r) then o
    else r

(* nx_kinds.h's complex Mul: each part one fma over the other product. *)
let zmul narrow a b =
  let rd = if narrow then round32 else Fun.id in
  let fma = if narrow then fma32 else Float.fma in
  let ii = rd (a.im *. b.im) and ir = rd (a.im *. b.re) in
  zof narrow (fma a.re b.re (-.ii)) (fma a.re b.im ir)

(* The order of real parts, then imaginary parts, [-0] below [+0]. *)
let zorder a b =
  let part x y =
    if x = y then Bool.compare (Float.sign_bit y) (Float.sign_bit x)
    else Float.compare x y
  in
  let c = part a.re b.re in
  if c <> 0 then c else part a.im b.im

(* [m] at the complex dtype, float32's where [narrow], and its identity: a
   NaN operand of an extreme gives a NaN. *)
let zop narrow m =
  let extreme keep a b =
    if znan a then a
    else if znan b then b
    else if keep (zorder a b) then a
    else b
  in
  match m with
  | Prod -> (zmul narrow, zof narrow 1. 0.)
  | Max -> (extreme (fun c -> c >= 0), zof narrow neg_infinity neg_infinity)
  | Min -> (extreme (fun c -> c <= 0), zof narrow infinity infinity)
  | Sum -> invalid_arg "zop: Sum"

(* An expected complex result: each part's want, or [Z_nan], any number
   with a NaN part, as every reduction but a sum gives a NaN. *)
type zwant = Z of want * want | Z_nan

let zwant_of z =
  if znan z then Z_nan else Z (Bits (fst z.bits), Bits (snd z.bits))

(* [ts] scanned as [scanned] does, but from the first term where [first]:
   the first chunk then has no carry. *)
let zscanned ~first ((f, id) as m) ts =
  let n = Array.length ts in
  let out = Array.make n id in
  let step a t = match a with None -> t | Some a -> f a t in
  let carry = ref (if first then None else Some id) in
  for c = 0 to ((n + 4095) / 4096) - 1 do
    let lo = 4096 * c and hi = min n (4096 * (c + 1)) in
    let acc = ref !carry in
    for i = lo to hi - 1 do
      acc := Some (step !acc ts.(i));
      out.(i) <- Option.get !acc
    done;
    carry := Some (step !carry (ordered ~first m (Array.sub ts lo (hi - lo))))
  done;
  Array.map zwant_of out

(* A complex sum's results: each part's, a float sum of the terms' parts in
   nx.cpu's order. *)
let zsums (c : case) ts =
  let part = if narrow c.x then D.Any D.Float32 else D.Any D.Float64 in
  let m = op part Sum in
  let sums f =
    results c (Array.map f ts) ~fill:Some_nan
      ~reduce:(fun o -> want_of (ordered m o))
      ~scan:(scanned m)
  in
  Array.map2
    (fun r i -> Z (r, i))
    (sums (fun t -> F (t.re, fst t.bits)))
    (sums (fun t -> F (t.im, snd t.bits)))

(* A product starts from its first term: 1 + 0i times a term would change
   its signed zeros and make NaN parts from its infinite ones. *)
let zexpected (c : case) =
  let narrow = narrow c.x and first = c.monoid = Prod in
  let ts = zterms c.x in
  if c.monoid = Sum then zsums c ts
  else
    let m = zop narrow c.monoid in
    results c ts ~fill:Z_nan
      ~reduce:(fun o -> zwant_of (ordered ~first m o))
      ~scan:(zscanned ~first m)

(* The first complex results [got] that differ from [want]: a NaN part
   wanted matches any NaN, [Z_nan] any number with a NaN part. *)
let zdiffer want got =
  let part w x b = match w with Some_nan -> Float.is_nan x | Bits w -> w = b in
  let bad = ref [] in
  Array.iteri
    (fun i w ->
      let g = got.(i) in
      let gr, gi = g.bits in
      let ok =
        match w with
        | Z_nan -> znan g
        | Z (wr, wi) -> part wr g.re gr && part wi g.im gi
      in
      let shown =
        match w with Z_nan -> "nan" | Z (wr, wi) -> show wr ^ " " ^ show wi
      in
      if (not ok) && List.length !bad < 8 then
        bad := Printf.sprintf "result %d: %Lx %Lx, expected %s" i gr gi shown :: !bad)
    want;
  List.rev !bad

(* Complex sums, products and extremes, reduced and scanned, in nx.cpu's
   order, bit for bit, a NaN as any NaN: a sum's NaN part, else a number
   with a NaN part. *)
let law_complex (b : Support.backend) (c : case) =
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  cover "no term" (per = 0 && total (result_shape c) > 0);
  cover "several blocks" (per > 1024);
  let ts = zterms c.x in
  cover "a NaN term" (Array.exists znan ts);
  (* A product whose last block leaves lanes without a term, which its tree
     leaves out, of a zero part, whose sign 1 + 0i times would change. *)
  let zero t = t.re = 0. || t.im = 0. in
  cover "a product of a lane of no term and a zero part"
    (c.monoid = Prod && per mod 1024 > 0 && per mod 1024 < 16
    && Array.exists zero ts);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  match run b c with
  | None -> failf "%s declined a complex %s" b.name (monoid_name c.monoid)
  | Some y -> equal (list string) [] (zdiffer (zexpected c) (zterms y))

let complex_case =
  Gen.with_pp pp_case
    (case_of D.[ Any Complex64; Any Complex128 ])

(* [n] terms, -0 - i then 1 - 0i, whose products in every order are
   -0 - i; 1 + 0i times that would be +0 - i. *)
let units n =
  let z re im = { Complex.re; im } in
  A.Any
    (A.of_array D.Complex64 [| n |]
       (Array.init n (fun i -> if i = 0 then z (-0.) (-1.) else z 1. (-0.))))

(* Products of one term, which are that term: -0 - 0i, which 1 + 0i times
   would make (+0, -0), 3 - 0i, (3, +0), and inf + 0i, (inf, NaN); the
   scans of two, whose first results are so; and scans past a chunk, of
   one slice, of slices side by side and of rows. *)
let complex_examples =
  let one ~scan dt ts =
    {
      scan;
      monoid = Prod;
      axes = [| 0 |];
      x = A.Any (A.of_array dt [| Array.length ts |] ts);
      views = [];
    }
  in
  let z re im = Complex.{ re; im } in
  [
    one ~scan:false D.Complex128 [| z (-0.) (-0.) |];
    one ~scan:false D.Complex64 [| z 3. (-0.) |];
    one ~scan:false D.Complex128 [| z infinity 0. |];
    one ~scan:true D.Complex64 [| z infinity 0.; z 2. 0. |];
    one ~scan:true D.Complex128 [| z (-0.) (-0.); z 1. 1. |];
    (* Outputs of no term, which the generator draws now and then. *)
    example Sum (D.Any D.Complex64) [| 3; 0 |] [| 1 |];
  ]
  @ List.map
      (fun (s, axis) ->
        let plain =
          {
            perm = Array.init (Array.length s) Fun.id;
            stepped = false;
            reversed = None;
            broadcast = false;
          }
        in
        {
          scan = true;
          monoid = Prod;
          axes = [| axis |];
          x = operand_with units s plain;
          views = [];
        })
      [ ([| 9000 |], 0); ([| 3; 5000 |], 1); ([| 4100; 3 |], 0) ]

(* Covers *)

let covers (c : case) =
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  let outs = total (result_shape c) in
  cover "no output" (outs = 0);
  cover "no term" (per = 0 && outs > 0);
  cover "one term" (per = 1);
  cover "two terms" (per = 2);
  let pow2 k = k > 0 && k land (k - 1) = 0 in
  cover "2^k ± 1 terms" (per > 2 && (pow2 (per - 1) || pow2 (per + 1)));
  cover "several blocks" (per > 1024);
  cover "many outputs" (outs >= 64);
  let nans =
    List.filter_map (Array.find_index is_nan) (outputs c (terms c.x))
  in
  cover "a NaN term" (nans <> []);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ]

(* The laws of S *)

(* nx.cpu's cases and order *)

(* Whether a float result of [c] is a NaN made from numbers: one whose terms,
   a scan's up to it, hold no NaN. *)
let makes_nan (c : case) =
  let (A.Any x) = c.x in
  let dt = D.Any (A.dtype x) in
  match A.dtype x with
  | D.Float32 | D.Float64 ->
      let m = op dt c.monoid in
      let one o =
        let n = Array.length o in
        let k = Option.value ~default:n (Array.find_index is_nan o) in
        if c.scan then Array.mem Some_nan (scanned m (Array.sub o 0 k))
        else k = n && is_nan (ordered m o)
      in
      List.exists one (outputs c (terms c.x))
  | _ -> false

let staged d = not (same (Nx_kernel.carrier d) d)

(* As many rows as nx.cpu aims its calls at, each a few more full blocks
   than a call of float32 terms holds. *)
let wide_rows =
  let open Gen in
  let* r = int_range 64 66 in
  let+ k = int_range 9217 9300 in
  [| r; k |]

let plain2 =
  { perm = [| 0; 1 |]; stepped = false; reversed = None; broadcast = false }

let law_order (b : Support.backend) c =
  covers c;
  cover "a NaN made from numbers" (makes_nan c);
  match run b c with
  | None -> failf "%s declined a case nx_cpu.mli says it computes" b.name
  | Some y -> equal (list string) [] (differ (expected c) y)

let law_computes b (c : case) =
  cover "narrow" (staged (dtype_of c.x));
  equal ~msg:"computed" bool true (Option.is_some (run b c))

(* Narrow dtypes and destinations of another dtype *)

(* [x] cast into [d] by nx.cpu, C-contiguous. *)
let cast_into (D.Any d) (A.Any x) =
  let dst = A.create Rig.host d (L.shape (A.layout x)) in
  match Support.map1 (module Nx_cpu) Cast ~dst x with
  | A.Done -> A.Any dst
  | r -> failf "a cast answered %a" Nx_array_support.pp_answer r

(* [c] by [k] into a fresh destination of [d]. *)
let run_into k (D.Any d) c x =
  let dst = A.Any (A.create Rig.host d (Iarray.of_array (result_shape c))) in
  match compute k ~scan:c.scan (identity (dtype_of x)) c.monoid c.axes dst
          [| plain_of x |] with
  | A.Done -> dst
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let narrow_dtypes =
  D.
    [
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
      Any Int4;
      Any Uint4;
      Any Bit;
    ]

(* A case of a narrow dtype, or of a byte-wide one, and a destination's
   dtype: mostly the operand's, else another. A vector past a scan's chunk
   is drawn often, so that a scan runs its chunks side by side, and so are
   [wide_rows], so that every run reaches their calls. *)
let into_case =
  let open Gen in
  let any =
    let shapes =
      frequency
        [ (3, shape_gen); (1, map (fun n -> [| n |]) (int_range 4097 20000)) ]
    in
    let* c =
      case_of ~shapes (narrow_dtypes @ narrow_dtypes @ floats @ D.[ Any Int32 ])
    in
    let+ d =
      frequency
        [
          (3, constant (dtype_of c.x));
          ( 1,
            of_list ~pp:pp_dtype
              D.[ Any Float32; Any Float64; Any Float16; Any Int64; Any Int8 ] );
        ]
    in
    (c, d)
  in
  (* A narrow float's reduction along [wide_rows], into its own dtype:
     some of each row's calls stage several blocks, and some reach the
     cap. *)
  let rows =
    let* d =
      of_list ~pp:pp_dtype D.[ Any Float16; Any Bfloat16; Any Float8_e5m2 ]
    in
    let* monoid = of_list (List.filter (fun m -> accepts m d) monoids) in
    let* s = wide_rows in
    let+ seed = int in
    let x = operand d s plain2 ~specials:20 seed in
    ({ scan = false; monoid; axes = [| 1 |]; x; views = [] }, d)
  in
  with_pp
    (fun ppf (c, d) -> Format.fprintf ppf "%a into %a" pp_case c pp_dtype d)
    (frequency [ (5, any); (1, rows) ])

(* For each result of [c] in C order, whether a NaN term reaches it, a
   scan's up to it. *)
let reached (c : case) =
  let nan = Array.map is_nan (terms (cast_into f32 c.x)) in
  let scan o =
    let seen = ref false in
    Array.map
      (fun k ->
        seen := !seen || nan.(k);
        !seen)
      o
  in
  let all = Array.init (Array.length nan) Fun.id in
  results c all ~reduce:(Array.exists (Array.get nan)) ~scan ~fill:false

(* NaNs made from numbers at narrow floats, reduced and scanned, into their
   own dtype and float32. *)
let into_examples =
  let inf = Float.infinity and ninf = Float.neg_infinity in
  let case scan monoid x = { scan; monoid; axes = [| 0 |]; x; views = [] } in
  let bf16 = vector_of D.Bfloat16 [| inf; ninf; 1. |] in
  [
    (case false Sum bf16, D.Any D.Bfloat16);
    (case true Sum bf16, D.Any D.Bfloat16);
    (case false Prod (vector_of D.Float8_e5m2 [| 0.; inf |]), f32);
    (* A scan down columns, a unit a whole row of 4096 staged outputs. *)
    ( {
        scan = true;
        monoid = Sum;
        axes = [| 0 |];
        x = operand (D.Any D.Bfloat16) [| 5; 4100 |] plain2 ~specials:0 3;
        views = [];
      },
      D.Any D.Bfloat16 );
  ]

(* Narrow floats' NaNs, one signalling, by their codes. *)
let nan_codes =
  D.
    [
      (Any Float16, 0x7C01, 0x7E00);
      (Any Bfloat16, 0x7F81, 0x7FC0);
      (Any Float8_e5m2, 0x7D, 0x7F);
      (Any Float8_e5m2, 0x7E, 0x7F);
      (Any Float8_e5m2, 0xFD, 0x7E);
      (Any Float8_e4m3fn, 0xFF, 0x7F);
    ]

(* [n] zeros of [d] with the codes [first] at [i] and [other] at [j]. *)
let nan_vector (D.Any dt as d) first other (n, i, j) =
  let w = D.bits dt / 8 in
  let code c = String.init w (fun k -> Char.chr ((c lsr (8 * k)) land 0xFF)) in
  Elements.of_elements d [| n |]
    (Array.init n (fun k ->
         code (if k = i then first else if k = j then other else 0)))

(* Two NaNs of a narrow float among zeros, in few terms and past a block
   and a chunk, by each monoid, reduced and scanned into their own dtype:
   a NaN from the first NaN on, signalling or quiet. *)
let nan_code_examples =
  List.concat_map
    (fun (d, first, other) ->
      List.concat_map
        (fun at ->
          let x = nan_vector d first other at in
          List.concat_map
            (fun monoid ->
              List.map
                (fun scan -> ({ scan; monoid; axes = [| 0 |]; x; views = [] }, d))
                [ false; true ])
            monoids)
        [ (4, 1, 2); (5000, 4500, 4700) ])
    nan_codes

(* A narrow operand's terms reduce in its carrier, and each result rounds
   once into its destination: the bits of the carrier's reduction of the
   operand cast into it, cast into the destination, a NaN as any NaN. A
   float result a NaN term reaches is a NaN. *)
let law_into (b : Support.backend) ((c : case), d) =
  let x = dtype_of c.x in
  cover "narrow" (staged x);
  cover "into another dtype" (not (same d x));
  cover "scan" c.scan;
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  cover "several blocks" (per > 1024);
  cover "a scan of several chunks" (c.scan && per > 4096);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  let got = run_into b.kernels d c c.x in
  let wide = run_into b.kernels (Nx_kernel.carrier x) c (cast_into (Nx_kernel.carrier x) c.x) in
  let reached = reached c in
  let made =
    Array.exists2 (fun t f -> is_nan t && not f) (terms wide) reached
  in
  cover "a NaN made from numbers" made;
  let got = Elements.elements got in
  equal (array (Elements.same d)) (Elements.elements (cast_into d wide)) got;
  let (D.Any dt) = d in
  if D.is D.Float dt then
    equal ~msg:"a NaN where a NaN term reaches" (array bool) reached
      (Array.map2 (fun r e -> r && Elements.nan_of d e) reached got)

(* nx.cpu with its reductions and scans on one thread. *)
module One = struct
  include Nx_cpu

  let reduce p ~shape ~axes rs ~dsts loads =
    Nx_cpu.one_thread (fun () -> Nx_cpu.reduce p ~shape ~axes rs ~dsts loads)

  let scan ?carry p ~axis rk ~dsts loads =
    Nx_cpu.one_thread (fun () -> Nx_cpu.scan ?carry p ~axis rk ~dsts loads)
end

(* Padded loads *)

(* The fills a law draws: zero, one, the extremes, a NaN and -0 of floats,
   the least integer; a sub-byte dtype's codes 0, 7 and 8 in both halves
   of their byte. *)
let fills (D.Any dt) =
  let le w v =
    String.init w (fun i ->
        Char.chr (Int64.to_int (Int64.logand (Int64.shift_right_logical v (8 * i)) 0xFFL)))
  in
  let w = D.bits dt / 8 in
  match dt with
  | D.Float32 ->
      List.map (fun v -> le 4 (Int64.of_int32 v))
        [ 0l; 0x3F800000l; 0xFF800000l; 0x7FC01234l; 0x80000000l ]
  | D.Float64 ->
      List.map (le 8)
        [ 0L; Int64.bits_of_float 1.; Int64.bits_of_float neg_infinity;
          0x7FF8000000000123L; Int64.min_int ]
  | D.Bool -> [ "\000"; "\001" ]
  | D.Bit -> [ "\000"; "\255" ]
  | _ when w = 0 -> [ "\000"; "\119"; "\136" ]
  | _ -> [ le w 0L; le w 7L; le w (Int64.shift_left 1L ((8 * w) - 1)) ]

(* A padding of an operand of shape [s]: low and high padding from -2 to 3
   (negative crops), interior padding up to 2 and at most one window of size
   up to 3, as drawn, so that some crop an axis below zero or hold a window
   wider than its axis; one in eight is of rank one more than [s]'s. *)
let pad_of s =
  let open Gen in
  let* extra = frequency [ (7, constant 0); (1, constant 1) ] in
  let r = Array.length s + extra in
  let ints lo hi = array ~size:(constant r) (int_range lo hi) in
  let* lo = ints (-2) 3 in
  let* hi = ints (-2) 3 in
  let* interior = ints 0 2 in
  let* axis = if r = 0 then constant None else option (int_range 0 (r - 1)) in
  let* size = int_range 1 3 in
  let* step = int_range 1 2 in
  let+ dilation = int_range 1 2 in
  let windows =
    match axis with
    | Some axis -> [| { M.axis; size; step; dilation } |]
    | None -> [||]
  in
  let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
  let interior = Iarray.of_array interior in
  { Nx_kernel.lo; hi; interior; windows = Iarray.of_array windows }

(* [x] loaded padded with the element of bits [fill] as [pad] says. *)
let padded_load (A.Any x) fill pad =
  let dt = A.dtype x in
  let fill = A.get (A.v dt (L.contiguous [||]) (Rig.Buffer.of_string fill)) [||] in
  Nx_kernel.Padded { x; fill; pad }

(* Refusals *)

(* A reduction or scan by [rs] along [axes] of a float32 loop over [x],
   padded by [pad] where given, into destinations of shapes [dsts]: near a
   fit, missed by one change at a time. *)
type misfit = {
  scan : bool;
  x : int array;
  pad : Nx_kernel.pad option;
  axes : int array;
  rs : Nx_kernel.reduction array;
  dsts : int array array;
}

let pp_misfit ppf m =
  Format.fprintf ppf "%s %s along %a of %a%s into %a"
    (if m.scan then "scan" else "reduce")
    (String.concat ", " (Array.to_list (Array.map reduction_name m.rs)))
    pp_ints m.axes pp_ints m.x
    (if m.pad <> None then ", padded" else "")
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_ints)
    (Array.to_list m.dsts)

(* The destinations [r] takes. *)
let width : Nx_kernel.reduction -> int = function
  | Monoid _ | Logsumexp -> 1
  | Arg _ | Moments -> 2

(* The shape the destinations of [m] fit, if its loop and axes do. *)
let fit_of m =
  let module Shape = Nx_kernel.Shape in
  let loop = match m.pad with None -> Ok m.x | Some p -> Shape.padded p m.x in
  Result.bind loop (fun y ->
      if m.scan then Shape.scan ~axis:m.axes.(0) y
      else Shape.reduce ~axes:(Iarray.of_array m.axes) m.rs y)

let misfits =
  Gen.with_pp pp_misfit
    (let open Gen in
     let* x = Nx_array_gen.shape in
     let* pad = frequency [ (3, constant None); (1, map Option.some (pad_of x)) ] in
     let loop =
       match pad with
       | None -> x
       | Some p -> Result.value (Nx_kernel.Shape.padded p x) ~default:x
     in
     let r = Array.length loop in
     let* scan = bool in
     let* axes =
       if scan then map (fun a -> [| a |]) (Elements.axis_near loop)
       else
         let* keep = array ~size:(constant r) bool in
         let axes = List.filter (fun i -> keep.(i)) (List.init r Fun.id) in
         frequency
           [
             (8, constant (Array.of_list axes));
             (1, constant (Array.of_list (List.rev axes)));
             (1, constant (Array.of_list (axes @ [ r ])));
           ]
     in
     let* rs =
       array ~size:(int_range 1 (if scan then 1 else 2))
         (of_list Nx_kernel.[ Monoid Sum; Monoid Max; Arg Max; Logsumexp; Moments ])
     in
     let m = { scan; x; pad; axes; rs; dsts = [||] } in
     let y = Result.value (fit_of m) ~default:loop in
     let n = Array.fold_left (fun n r -> n + width r) 0 rs in
     let* count =
       frequency [ (8, constant n); (1, constant (n - 1)); (1, constant (n + 1)) ]
     in
     let+ dsts = array ~size:(constant (max 0 count)) (Elements.near y) in
     { m with dsts })

(* nx.cpu computes what Nx_kernel.Shape fits and refuses the rest with
   Shape_mismatch, writing nothing: a padding Shape refuses, an axis past
   the rank or out of order, an extreme of no term, a destination of
   another shape, one too few or one too many. *)
let law_refusals (b : Support.backend) m =
  let module K = (val b.kernels) in
  let fit =
    match fit_of m with
    | Ok z ->
        Array.length m.dsts = Array.fold_left (fun n r -> n + width r) 0 m.rs
        && Array.for_all (( = ) z) m.dsts
    | Error _ -> false
  in
  cover "fits" fit;
  cover "misfits" (not fit);
  cover "a padding that misfits"
    (match m.pad with Some p -> Result.is_error (Nx_kernel.Shape.padded p m.x) | None -> false);
  cover "a scan" m.scan;
  cover "two reductions" (Array.length m.rs = 2);
  let x = A.of_array D.Float32 (Iarray.of_array m.x) (Array.make (total m.x) 1.) in
  let load : Nx_kernel.load =
    match m.pad with None -> Plain x | Some pad -> Padded { x; fill = 0.; pad }
  in
  (* Destination [i]'s dtype: an Arg's second its positions'. *)
  let kinds =
    Array.concat
      (Array.to_list
         (Array.map
            (function
              | Nx_kernel.Arg _ -> [| f32; D.Any D.Int64 |]
              | Moments -> [| f32; f32 |]
              | Monoid _ | Logsumexp -> [| f32 |])
            m.rs))
  in
  let dsts =
    Array.mapi
      (fun i s ->
        let (D.Any dt as d) = if i < Array.length kinds then kinds.(i) else f32 in
        let seven = Elements.int_bits d (D.bits dt / 8) 7L in
        Elements.of_elements d s (Array.make (total s) seven))
      m.dsts
  in
  let before = Array.map Elements.elements dsts in
  let p = identity f32 in
  let answer =
    if m.scan then K.scan p ~axis:m.axes.(0) (m.rs.(0), 0) ~dsts [| load |]
    else
      K.reduce p ~shape:(loop_shape [| load |]) ~axes:(Iarray.of_array m.axes)
        (Array.map (fun r -> (r, 0)) m.rs)
        ~dsts [| load |]
  in
  equal Elements.answer (if fit then A.Done else A.Shape_mismatch) answer;
  if not fit then
    equal ~msg:"untouched" (array (array string)) before
      (Array.map Elements.elements dsts)

(* What nx.cpu answers for a call its contract names: Wrong_dtype for a
   dtype a reduction does not take or positions other than int64, and
   Shape_mismatch for a load of another shape than the loop's. *)
let answered (b : Support.backend) =
  let module K = (val b.kernels) in
  let make dt s = A.Any (A.create Rig.host dt s) in
  let reduce (A.Any x as a) r dsts =
    K.reduce (identity (dtype_of a)) ~shape:(shape_of a) ~axes:[| 1 |]
      [| (r, 0) |] ~dsts [| Nx_kernel.Plain x |]
  in
  let scan (A.Any x as a) r dsts =
    K.scan (identity (dtype_of a)) ~axis:1 (r, 0) ~dsts [| Nx_kernel.Plain x |]
  in
  let wrong = A.Wrong_dtype in
  [
    ( "a Sum of bool",
      wrong,
      fun () -> reduce (make D.Bool [| 2; 3 |]) (Monoid Sum) [| make D.Bool [| 2 |] |] );
    ( "a Prod of bit",
      wrong,
      fun () -> reduce (make D.Bit [| 2; 3 |]) (Monoid Prod) [| make D.Bit [| 2 |] |] );
    ( "a Moments scan of int32",
      wrong,
      fun () ->
        scan (make D.Int32 [| 2; 3 |]) Moments
          [| make D.Int32 [| 2; 3 |]; make D.Int32 [| 2; 3 |] |] );
    ( "a Logsumexp of complex64",
      wrong,
      fun () ->
        reduce (make D.Complex64 [| 2; 3 |]) Logsumexp [| make D.Complex64 [| 2 |] |] );
    ( "an Arg's positions in int32",
      wrong,
      fun () ->
        reduce (make D.Float32 [| 2; 3 |]) (Arg Max)
          [| make D.Float32 [| 2 |]; make D.Int32 [| 2 |] |] );
    ( "a carry of another dtype",
      wrong,
      fun () ->
        let (A.Any x) = make D.Float32 [| 2; 3 |] in
        K.scan ~carry:(make D.Float64 [| 2; 1 |]) (identity f32) ~axis:1
          (Monoid Sum, 0) ~dsts:[| make D.Float32 [| 2; 3 |] |]
          [| Nx_kernel.Plain x |] );
    ( "a load of another shape than the loop's",
      A.Shape_mismatch,
      fun () ->
        let (A.Any x) = make D.Float32 [| 2; 3 |] in
        K.reduce (identity f32) ~shape:[| 2; 4 |] ~axes:[| 1 |] [| (Monoid Sum, 0) |]
          ~dsts:[| make D.Float32 [| 2 |] |] [| Nx_kernel.Plain x |] );
  ]

(* A padded load of 2^50 elements, more than any host's memory, or past
   max_numel, whose loop does not fit its destinations, is refused before
   the kernel copies it: Shape_mismatch, and the destinations as they
   were. *)
let test_huge_padding (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2 |] [| 1.; 2. |] in
  let padded lo : Nx_kernel.load =
    Padded
      { x; fill = 0.; pad = { lo = [| lo |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] } }
  in
  let big = 1 lsl 50 in
  let huge = padded big and past = padded (L.max_numel - 1) in
  let twice = P.v ~ins:[| f32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |] in
  let dst s = A.of_array D.Float32 s (Array.make (Iarray.fold_left ( * ) 1 s) 9.) in
  let refused msg call dsts =
    equal ~msg Elements.answer A.Shape_mismatch (call (Array.map (fun d -> A.Any d) dsts));
    Array.iter
      (fun d ->
        equal ~msg:(msg ^ ": untouched") (array float_exact)
          (Array.make (Iarray.fold_left ( * ) 1 (L.shape (A.layout d))) 9.)
          (A.to_array d))
      dsts
  in
  let sum = [| (Nx_kernel.Monoid Sum, 0) |] in
  List.iter
    (fun (name, p) ->
      refused (name ^ ": a reduction into another shape")
        (fun dsts -> K.reduce p ~shape:[| big + 2 |] ~axes:[| 0 |] sum ~dsts [| huge |])
        [| dst [| 2 |] |];
      refused (name ^ ": a scan into another shape")
        (fun dsts -> K.scan p ~axis:0 (Monoid Sum, 0) ~dsts [| huge |])
        [| dst [| 3 |] |];
      refused (name ^ ": a scan past max_numel")
        (fun dsts -> K.scan p ~axis:0 (Monoid Sum, 0) ~dsts [| past |])
        [| dst [| 3 |] |];
      refused (name ^ ": a Moments scan into another shape")
        (fun dsts -> K.scan p ~axis:0 (Moments, 0) ~dsts [| huge |])
        [| dst [| 3 |]; dst [| 3 |] |])
    [ ("identity", identity f32); ("a program", twice) ]

(* Carries *)

(* A scan of a monoid along rows past a chunk, and a chunk after the
   first: of a float, an integer, a complex number or a narrow float. *)
type chunked = { c : case; chunk : int }

let chunk_case =
  Gen.with_pp
    (fun ppf k -> Format.fprintf ppf "%a from chunk %d" pp_case k.c k.chunk)
    (let open Gen in
     let* k = int_range 1 3 in
     let* n = frequency [ (3, int_range 4097 9000); (1, int_range 12289 13000) ] in
     let* c =
       case_of ~scan:(constant true) ~shapes:(constant [| k; n |])
         (floats @ floats
         @ D.[ Any Int32; Any Int64; Any Int16; Any Uint8; Any Complex64;
               Any Complex128; Any Float16; Any Bfloat16; Any Float8_e5m2 ])
     in
     let+ chunk = int_range 1 ((n - 1) / 4096) in
     { c = { c with axes = [| 1 |] }; chunk })

(* The carry into each chunk of [ts] scanned as [zscanned ~first] does,
   from chunk 1: the chunk before's total, from its first term where
   [first], combined with the carry into it. *)
let zcarries ~first ((f, _) as m) ts =
  let n = Array.length ts in
  let k = (n + 4095) / 4096 in
  let out = Array.make (max 1 k) ts.(0) in
  for c = 1 to k - 1 do
    let total = ordered ~first m (Array.sub ts (4096 * (c - 1)) 4096) in
    out.(c) <- (if c = 1 then total else f out.(c - 1) total)
  done;
  out

(* The scan of each row from a chunk's start, with the carry into that
   chunk in the carrier, has the bits of the scan of the whole row from
   that chunk on. A complex sum carries each part's sum; a narrow float
   scans in its carrier, from which each result rounds once. *)
let law_chunk_carry (b : Support.backend) { c; chunk } =
  let module K = (val b.kernels) in
  let (A.Any x) = c.x in
  let (D.Any dt as d) = dtype_of c.x in
  let s = shape_of c.x in
  let k = s.(0) and n = s.(1) in
  let p = 4096 * chunk in
  let (D.Any cdt as carrier) = Nx_kernel.carrier d in
  let w = D.bits cdt / 8 in
  let complex = D.is D.Complex dt in
  cover "the last chunk" (p + 4096 >= n);
  cover "a float" (is_float d && not (staged d));
  cover "a narrow float" (staged d);
  cover "an integer" (not (is_float d || complex));
  cover "a complex number" complex;
  cover "a product" (c.monoid = Prod);
  cover "a NaN term" (Array.exists (Elements.nan_of d) (Elements.elements c.x));
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  let whole = { M.start = 0; count = k; step = 1 } in
  let tail =
    Option.get (A.move (M.Slice [| whole; { M.start = p; count = n - p; step = 1 } |]) x)
  in
  let dst = A.Any (A.create Rig.host dt [| k; n - p |]) in
  let scan carry =
    equal ~msg:"answer" Elements.answer A.Done
      (K.scan ~carry (identity d) ~axis:1 (Monoid c.monoid, 0) ~dsts:[| dst |]
         [| Nx_kernel.Plain tail |])
  in
  let carry_of es = Elements.of_elements carrier [| k; 1 |] es in
  let rows_of ts = Array.init k (fun i -> Array.sub ts (i * n) n) in
  let tails = Array.map (fun (_, want) -> want) in
  if complex then begin
    let narrow = narrow c.x in
    let rows = rows_of (zterms c.x) in
    let part = Elements.int_bits (if narrow then f32 else f64) (w / 2) in
    let each =
      if c.monoid = Sum then
        let m = op (if narrow then f32 else f64) Sum in
        let one f row =
          let ts = Array.map f row in
          ((carries m ts).(chunk), Array.sub (scanned m ts) p (n - p))
        in
        Array.map
          (fun row ->
            let cr, wr = one (fun z -> F (z.re, fst z.bits)) row in
            let ci, wi = one (fun z -> F (z.im, snd z.bits)) row in
            (part (bits_of cr) ^ part (bits_of ci), Array.map2 (fun r i -> Z (r, i)) wr wi))
          rows
      else
        let first = c.monoid = Prod in
        let m = zop narrow c.monoid in
        Array.map
          (fun row ->
            let z = (zcarries ~first m row).(chunk) in
            (part (fst z.bits) ^ part (snd z.bits), Array.sub (zscanned ~first m row) p (n - p)))
          rows
    in
    scan (carry_of (Array.map fst each));
    equal (list string) [] (zdiffer (Array.concat (Array.to_list (tails each))) (zterms dst))
  end
  else begin
    let wide = if staged d then cast_into carrier c.x else c.x in
    let m = op carrier c.monoid in
    let rows = rows_of (terms wide) in
    let le b = Elements.int_bits carrier w b in
    scan (carry_of (Array.map (fun row -> le (bits_of (carries m row).(chunk))) rows));
    let want =
      Array.concat (Array.to_list (Array.map (fun row -> Array.sub (scanned m row) p (n - p)) rows))
    in
    if not (staged d) then equal (list string) [] (differ want dst)
    else
      let quiet = if same carrier f32 then 0x7FC00000L else 0x7FF8000000000000L in
      let bits = Array.map (function Bits b -> le b | Some_nan -> le quiet) want in
      equal
        (array (Elements.same d))
        (Elements.elements (cast_into d (Elements.of_elements carrier [| k; n - p |] bits)))
        (Elements.elements dst)
  end

(* The first results that miss [want]. *)
let misses want got =
  let bad = ref [] in
  Array.iteri
    (fun i w ->
      let t = got.(i) in
      let ok =
        match (w, t) with
        | Is Some_nan, t -> is_nan t
        | Is (Bits b), t -> bits_of t = b
        | Near (v, _), F (g, _) when not (Float.is_finite v) -> g = v
        | Near (v, tol), F (g, _) -> Float.abs (g -. v) <= tol
        | Near _, I _ -> false
      in
      if (not ok) && List.length !bad < 8 then
        let w =
          match w with
          | Is w -> show w
          | Near (v, tol) -> Printf.sprintf "%h within %g" v tol
        in
        bad :=
          Printf.sprintf "result %d: %Lx, expected %s" i (bits_of t) w :: !bad)
    want;
  List.rev !bad

(* Stateful reductions *)

(* Values that break Arg, Logsumexp and Moments: about [center] within
   [scale]; on [ties], multiples of half [scale], so that extremes repeat
   and -0 meets +0. *)
type values = { name : string; center : float; scale : float; ties : bool }

let value_kinds =
  [
    { name = "small"; center = 0.; scale = 1.; ties = false };
    (* A mean that dwarfs the spread. *)
    { name = "offset"; center = 1e4; scale = 1e-2; ties = false };
    (* exp of the terms less their maximum spans past both ends of float32. *)
    { name = "wide"; center = 0.; scale = 40.; ties = false };
    { name = "ties"; center = 0.; scale = 1.; ties = true };
  ]

(* [n] elements of the float [dt] of the kind [v]: one in [outliers] (none
   for 0) is a power of two up to 2^20 times larger, one in [specials] a
   NaN, an infinity or -0, as [elements] draws them. *)
let float_elements (type a b) (dt : (a, b) D.t) n v ~outliers ~specials seed
    =
  let r = Random.State.make [| seed |] in
  let one_in k = k > 0 && Random.State.int r k = 0 in
  let pick a = a.(Random.State.int r (Array.length a)) in
  let b = Bytes.create (max 1 (D.bytes dt n)) in
  for i = 0 to n - 1 do
    let x = v.center +. (v.scale *. full r 52) in
    let x =
      if v.ties then Float.round (2. *. x /. v.scale) *. v.scale /. 2. else x
    in
    let x =
      if one_in outliers then x *. Float.ldexp 1. (Random.State.int r 21)
      else x
    in
    let special = one_in specials in
    match dt with
    | D.Float32 ->
        Bytes.set_int32_le b (4 * i)
          (if special then
             pick
               (Array.append nans32
                  [| 0x80000000l; 0x7F800000l; 0xFF800000l |])
           else Int32.bits_of_float x)
    | D.Float64 ->
        Bytes.set_int64_le b (8 * i)
          (if special then
             pick
               (Array.append nans64
                  [|
                    Int64.bits_of_float (-0.);
                    Int64.bits_of_float infinity;
                    Int64.bits_of_float neg_infinity;
                  |])
           else Int64.bits_of_float x)
    | _ -> invalid_arg "float_elements: not float32 or float64"
  done;
  A.Any (A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string (Bytes.to_string b)))

(* Many outputs of several blocks each, beside a short axis or none. *)
let many_long =
  let open Gen in
  let* outer = int_range 64 66 in
  let* n = int_range 1025 1100 in
  let+ inner = array ~size:(int_range 0 1) (int_range 2 3) in
  Array.append [| outer; n |] inner

let stat_case =
  Gen.with_pp pp_stat
    (let open Gen in
     let ints =
       List.filter (fun (D.Any dt) -> not (D.is D.Float dt || D.equal dt D.Bool)) base
     in
     let* d = of_list ~pp:pp_dtype (floats @ floats @ ints) in
     let* r =
       of_list
         (if is_float d then Nx_kernel.[ Arg Max; Arg Min; Logsumexp; Moments ]
          else Nx_kernel.[ Arg Max; Arg Min ])
     in
     let* s = frequency [ (6, shape_gen); (1, many_long) ] in
     let rank = Array.length s in
     let* keep = array ~size:(constant rank) bool in
     let axes =
       Array.of_list (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let* v = view_of rank in
     let* values = of_list value_kinds in
     let* outliers = frequency [ (3, constant 0); (1, constant 50) ] in
     let* specials =
       frequency [ (3, constant 0); (1, constant 300); (1, constant 20) ]
     in
     let+ seed = int in
     let (D.Any dt) = d in
     let make n =
       if is_float d then float_elements dt n values ~outliers ~specials seed
       else A.Any (elements dt n ~specials seed)
     in
     stat_of r s axes v values.name make)

(* The destinations of [c]: the extreme and its int64 position for Arg, the
   mean and the variance for Moments, one for Logsumexp. *)
let stat_dsts (c : stat) =
  let (A.Any x) = c.x in
  let s = Iarray.of_array (stat_shape c) in
  let like () = A.Any (A.create Rig.host (A.dtype x) s) in
  match c.r with
  | Arg _ -> [| like (); A.Any (A.create Rig.host D.Int64 s) |]
  | Moments -> [| like (); like () |]
  | Monoid _ | Logsumexp -> [| like () |]

(* [c] by [k] into fresh destinations, their terms; [None] if declined. *)
let run_stat k c =
  let module K = (val k : Nx_kernel.S) in
  let dsts = stat_dsts c in
  match
    K.reduce (identity (dtype_of c.x)) ~shape:(shape_of c.x)
      ~axes:(Iarray.of_array c.axes) [| (c.r, 0) |] ~dsts [| plain_of c.x |]
  with
  | A.Done -> Some (Array.map terms dsts)
  | A.Declined -> None
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* [xs] summed by Neumaier's compensated sum, within 2 ε |Σ| of the exact
   sum, ε the unit roundoff of float64, barring terms of n² ε. *)
let neumaier xs =
  let s = ref 0. and c = ref 0. in
  Array.iter
    (fun x ->
      let t = !s +. x in
      (c := !c +. if Float.abs !s >= Float.abs x then !s -. t +. x else x -. t +. !s);
      s := t)
    xs;
  !s +. !c

(* nx_cpu.mli's terms of the bounds for [n] terms: [l] the most roundings
   of a block's sum in lane order and [d] the depth of the blocks' tree. *)
let depths n =
  let blocks = (n + 1023) / 1024 in
  let rec log2 k p = if p >= k then 0 else 1 + log2 k (2 * p) in
  (Float.of_int (((min n 1024 + 15) / 16) + 3), Float.of_int (log2 blocks 1))

let eps = 0x1p-53

(* Logsumexp of [ts] within nx_cpu.mli's bound at the unit roundoff [u],
   from a float64 reference within ε (ln n + 4 + |r|). *)
let lse_expect u ts =
  if Array.exists is_nan ts then [| Is Some_nan |]
  else
    let xs = Array.map value ts in
    let n = Array.length xs in
    let m = Array.fold_left Float.max neg_infinity xs in
    if not (Float.is_finite m) then [| Near (m, 0.) |]
    else
      let r = m +. log (neumaier (Array.map (fun x -> exp (x -. m)) xs)) in
      let l, d = depths n and ln = log (Float.of_int n) in
      let bound = ((l +. d +. 5. +. (6. *. ln)) *. u) +. (u *. Float.abs r) in
      [| Near (r, bound +. (eps *. (ln +. 4. +. Float.abs r))) |]

(* Moments of [ts] within nx_cpu.mli's bounds, [u] the dtype's roundoff,
   from float64 references of the terms less the first, by compensated
   sums: within ε (4 R + 2 |μ|) and ε (5 V + 8 R √V). *)
let moments_expect u ts =
  let nan = Is Some_nan in
  if ts = [||] || Array.exists is_nan ts then [| nan; nan |]
  else (
    let xs = Array.map value ts in
    let pos = Array.mem infinity xs and neg = Array.mem neg_infinity xs in
    match (pos, neg) with
    | true, true -> [| nan; nan |]
    | true, false -> [| Near (infinity, 0.); nan |]
    | false, true -> [| Near (neg_infinity, 0.); nan |]
    | false, false ->
        let n = Array.length xs and c = xs.(0) in
        let nf = Float.of_int n in
        let ys = Array.map (fun x -> x -. c) xs in
        let my = neumaier ys /. nf in
        let mean = c +. my in
        let v = neumaier (Array.map (fun y -> (y -. my) *. (y -. my)) ys) /. nf in
        let hi = Array.fold_left Float.max neg_infinity xs in
        let r = hi -. Array.fold_left Float.min infinity xs in
        let l, d = depths n and sv = Float.sqrt v in
        let m = Array.fold_left (fun a x -> a +. Float.abs x) 0. xs /. nf in
        let tm = ((l +. d) *. u *. m) +. (u *. Float.abs mean) in
        let k = l +. d in
        let tv =
          ((k +. 5.) *. u *. v)
          +. (2. *. (k +. 1.) *. u *. tm *. sv)
          +. (((3. *. k) +. 7.) *. u *. tm *. tm)
        in
        [|
          Near (mean, tm +. (eps *. ((4. *. r) +. (2. *. Float.abs mean))));
          Near (v, tv +. (eps *. ((5. *. v) +. (8. *. r *. sv))));
        |])

(* The expected results of each destination of [c], in C order. *)
let stat_expected (c : stat) =
  let (A.Any x) = c.x in
  let d = D.Any (A.dtype x) in
  let u = if D.bits (A.dtype x) = 32 then 0x1p-24 else 0x1p-53 in
  let per = outputs (as_case c) (terms c.x) in
  let one ts =
    match c.r with
    | Arg e -> arg_expect d e ts
    | Logsumexp -> lse_expect u ts
    | Moments -> moments_expect u ts
    | Monoid _ -> invalid_arg "stat_expected: a monoid"
  in
  let all = List.map one per in
  Array.init (Array.length (stat_dsts c)) (fun k ->
      Array.of_list (List.map (fun e -> e.(k)) all))

(* Each result exact (Arg) or within nx_cpu.mli's bound of the exact one,
   NaNs and infinities as Nx_kernel.reduction states them, and the same
   bits on one thread and from a C-contiguous copy of the operand, a NaN as
   any NaN. *)
let law_stateful (b : Support.backend) (c : stat) =
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  let outs = total (stat_shape c) in
  cover "no output" (outs = 0);
  cover "no term" (per = 0 && outs > 0);
  cover "one term" (per = 1);
  cover "several blocks" (per > 1024);
  cover "many outputs" (outs >= 64);
  cover "many outputs of several blocks" (outs >= 64 && per > 1024);
  cover "many outputs of several blocks, through a view"
    (outs >= 64 && per > 1024 && c.views <> []);
  List.iter (fun (k : values) -> cover k.name (c.values = k.name)) value_kinds;
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  match run_stat b.kernels c with
  | None -> failf "nx.cpu declined %s" (reduction_name c.r)
  | Some got ->
      let want = stat_expected c in
      let wanted p = Array.exists (Array.exists p) want in
      cover "a NaN" (wanted (function Is Some_nan -> true | _ -> false));
      cover "an infinity"
        (wanted (function Near (v, _) -> not (Float.is_finite v) | Is _ -> false));
      Array.iteri
        (fun k w -> equal ~msg:(Printf.sprintf "destination %d" k) (list string) [] (misses w got.(k)))
        want

(* A Logsumexp's terms: shifts from the maximum about the least integers whose
   exp is a normal float32 (-87) and float64 (-708), infinities, signed zeros
   and NaNs with payloads. *)
type lse =
  | Lse :
      (float, 's) D.t * int array * int array * float array * int list
      -> lse

let pp_lse ppf (Lse (dt, shape, axes, xs, _)) =
  Format.fprintf ppf "%s %a along %a: %a" (D.name dt)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int)
    (Array.to_list shape)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int)
    (Array.to_list axes)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_float)
    (Array.to_list xs)

let lse_terms =
  [
    0.; -0.; 1.; 50.; -37.; -86.5; -87.; -87.25; -88.; -100.; -660.; -707.5;
    -708.; -708.25; -750.; Float.infinity; Float.neg_infinity; Float.nan;
    Int64.float_of_bits 0x7ff8_0000_2000_0000L;
    Int64.float_of_bits 0xfff8_0000_0000_0001L;
  ]

let lse_case =
  let open Gen in
  with_pp pp_lse
  @@ let* rank = int_range 1 2 in
     let* shape = array ~size:(constant rank) (int_range 0 9) in
     let* axes =
       such_that
         (fun a -> a <> [||])
         (map
            (fun keep ->
              Array.of_list
                (List.filteri (fun a _ -> keep.(a)) (List.init rank Fun.id)))
            (array ~size:(constant rank) bool))
     in
     let n = Array.fold_left ( * ) 1 shape in
     let* xs = array ~size:(constant n) (of_list lse_terms) in
     let* signalling =
       array ~size:(constant n)
         (frequency [ (11, constant false); (1, constant true) ])
     in
     let sn = List.filter (fun i -> signalling.(i)) (List.init n Fun.id) in
     let+ dt = int_range 0 3 in
     match dt with
     | 0 -> Lse (D.Float32, shape, axes, xs, sn)
     | 1 -> Lse (D.Float64, shape, axes, xs, sn)
     | 2 -> Lse (D.Float16, shape, axes, xs, sn)
     | _ -> Lse (D.Bfloat16, shape, axes, xs, sn)

(* [xs] of [shape] at [dt], a signalling NaN with a payload at each flat
   position of [sn]. *)
let signalling (type s) (dt : (float, s) D.t) shape xs sn : (float, s) A.t =
  let a = A.of_array dt (Iarray.of_array shape) xs in
  let index i =
    let idx = Array.make (Array.length shape) 0 and r = ref i in
    for k = Array.length shape - 1 downto 0 do
      idx.(k) <- !r mod shape.(k);
      r := !r / shape.(k)
    done;
    Iarray.of_array idx
  in
  let put (type u r) (u : (u, r) D.t) (code : u) =
    let raw = Option.get (A.bitcast u a) in
    List.iter (fun i -> A.set raw (index i) code) sn
  in
  (match dt with
  | D.Float16 -> put D.Uint16 0x7d01
  | D.Bfloat16 -> put D.Uint16 0x7f81
  | D.Float32 -> put D.Uint32 0x7f800101l
  | D.Float64 -> put D.Uint64 0x7ff0_0000_0000_0101L
  | _ -> ());
  a

(* [L + D + 5 + 6 ln n] of nx.cpu's Logsumexp bound for [n] terms. *)
let lse_rounds n =
  let l, d = depths n in
  l +. d +. 5. +. (6. *. Float.log (Float.of_int n))

(* [m + log Σ exp (t - m)] of [ts] in float64, [m] their maximum: a NaN
   where a term is, [m] where it is infinite. *)
let lse_ref ts =
  if List.exists Float.is_nan ts then Float.nan
  else
    let m = List.fold_left Float.max Float.neg_infinity ts in
    if not (Float.is_finite m) then m
    else
      let s = List.fold_left (fun s t -> s +. Float.exp (t -. m)) 0. ts in
      m +. Float.log s

(* nx.cpu's Logsumexp is [m + log Σ exp (x - m)] within the bound nx_cpu.mli
   states, in the float32 carrier for a narrow float, which adds its own
   rounding: a NaN where a term is, [-∞] for no term or terms all [-∞], [+∞]
   for a [+∞] term. *)
let law_logsumexp (b : Support.backend) (Lse (dt, shape, axes, xs, sn)) =
  let module K = (val b.kernels : Nx_kernel.S) in
  let cut = match dt with D.Float64 -> -708. | _ -> -87. in
  let x = signalling dt shape xs sn in
  cover "no term" (Array.exists (fun a -> shape.(a) = 0) axes);
  cover "a NaN term" (Array.exists Float.is_nan xs);
  cover "a signalling NaN term" (sn <> []);
  cover "a signalling NaN of a narrow float" (sn <> [] && D.bits dt = 16);
  (* Each output's terms, as [x] holds them, in C order of the outputs. *)
  let kept =
    List.filter
      (fun a -> not (Array.mem a axes))
      (List.init (Array.length shape) Fun.id)
  in
  let outs = List.fold_left (fun n a -> n * shape.(a)) 1 kept in
  let rows = Array.make outs [] in
  Array.iteri
    (fun i t ->
      let idx = Array.make (Array.length shape) 0 and r = ref i in
      for a = Array.length shape - 1 downto 0 do
        idx.(a) <- !r mod shape.(a);
        r := !r / shape.(a)
      done;
      let o = List.fold_left (fun o a -> (o * shape.(a)) + idx.(a)) 0 kept in
      rows.(o) <- t :: rows.(o))
    (A.to_array x);
  let below =
    Array.exists
      (fun row ->
        let m = List.fold_left Float.max Float.neg_infinity row in
        Float.is_finite m
        && List.exists (fun t -> t -. m < cut && t -. m > -1000.) row)
      rows
  in
  cover "a term past the cutoff below a finite maximum" below;
  let dst =
    A.create Rig.host dt
      (Iarray.of_list (List.map (fun a -> shape.(a)) kept))
  in
  (match
     K.reduce (identity (D.Any dt)) ~shape ~axes:(Iarray.of_array axes)
       [| (Nx_kernel.Logsumexp, 0) |]
       ~dsts:[| A.Any dst |] [| Nx_kernel.Plain x |]
   with
  | A.Done -> ()
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a);
  let got = A.to_array dst in
  let u = match dt with D.Float64 -> Float.epsilon /. 2. | _ -> 0x1p-24 in
  let narrow =
    match dt with D.Float16 -> 0x1p-11 | D.Bfloat16 -> 0x1p-8 | _ -> 0.
  in
  equal ~msg:"one result per output" int outs (Array.length got);
  Array.iteri
    (fun o row ->
      let want = lse_ref row and y = got.(o) in
      let msg = Printf.sprintf "output %d" o in
      if not (Float.is_finite want) then
        equal ~msg (option float_exact)
          (if Float.is_nan want then None else Some want)
          (if Float.is_nan y then None else Some y)
      else
        let n = List.length row in
        (* The float64 reference errs by about [n + 4] roundings. *)
        let slack =
          match dt with
          | D.Float64 -> Float.of_int (n + 4) *. u *. (1. +. Float.abs want)
          | _ -> 0.
        in
        let bound =
          (lse_rounds n *. u) +. ((u +. narrow) *. Float.abs want) +. slack
        in
        at_most ~msg float_exact ~than:bound (Float.abs (y -. want)))
    rows

(* Arg, Logsumexp and Moments scans *)

(* Lengths around a multiple of nx.cpu's network tiles of 256 terms, from
   [lo] to [hi] tiles. *)
let tiled lo hi =
  let open Gen in
  let* k = int_range lo hi in
  let+ d = int_range (-2) 2 in
  (k * 256) + d

(* A scan by Arg of a float, an integer or a boolean, or by Logsumexp or
   Moments of a float, along one axis. Now and then: a vector of 1 to 40
   network tiles, past an Arg scan's chunk of 4096 from 16; 65 to 72
   slices of 1 to 8 tiles, in units of 16 slices, the last partial; two or
   three slices of 12 to 24 tiles of finite terms, which a few threads run
   as one unit; slices of one term; and rows of 32-term tiles, 2 to 130
   slices of 32, 33 or more terms, scanned along the last axis or along
   the first through a transposed operand. *)
let stat_scan_case =
  Gen.with_pp pp_stat
    (let open Gen in
     let ints = List.filter (fun (D.Any dt) -> not (D.is D.Float dt)) base in
     let* d = of_list ~pp:pp_dtype (floats @ floats @ ints) in
     let* r =
       of_list
         (if is_float d then Nx_kernel.[ Arg Max; Arg Min; Logsumexp; Moments ]
          else Nx_kernel.[ Arg Max; Arg Min ])
     in
     let any =
       let+ s = shape_gen in
       (s, None, false, false)
     in
     let long = map (fun n -> ([| n |], Some 0, false, false)) (tiled 1 40) in
     let wide =
       let* n = tiled 1 8 in
       let+ k = int_range 65 72 in
       ([| n; k |], Some 0, false, false)
     in
     let few =
       let* n = tiled 12 24 in
       let+ k = int_range 2 3 in
       ([| n; k |], Some 0, false, true)
     in
     let one = map (fun n -> ([| n; 1 |], Some 1, false, false)) (int_range 1 40) in
     let tiles =
       let* len =
         frequency
           [
             (1, constant 32);
             (1, constant 33);
             ( 3,
               let* k = int_range 1 9 in
               let+ r = int_range 1 31 in
               (32 * k) + r );
           ]
       in
       let* slices = int_range 2 130 in
       let+ last = bool in
       if last then ([| slices; len |], Some 1, false, false)
       else ([| len; slices |], Some 0, true, false)
     in
     let* s, axis, transposed, finite =
       frequency
         [ (6, any); (1, long); (1, wide); (1, few); (1, one); (2, tiles) ]
     in
     let s = if Array.length s = 0 then [| 3 |] else s in
     let rank = Array.length s in
     let* a =
       match axis with Some a -> constant a | None -> int_range 0 (rank - 1)
     in
     let* v = view_of rank in
     let v = if transposed then { v with perm = [| 1; 0 |] } else v in
     let* values = of_list value_kinds in
     let* outliers = frequency [ (3, constant 0); (1, constant 50) ] in
     let* specials =
       if finite then constant 0
       else frequency [ (3, constant 0); (1, constant 300); (1, constant 20) ]
     in
     let+ seed = int in
     let (D.Any dt) = d in
     let make n =
       if is_float d then float_elements dt n values ~outliers ~specials seed
       else A.Any (elements dt n ~specials seed)
     in
     stat_of r s [| a |] v values.name make)

(* [c]'s scan by [k] into fresh destinations of [c.x]'s shape, their
   terms; [None] if declined. *)
let run_stat_scan k (c : stat) =
  let module K = (val k : Nx_kernel.S) in
  let (A.Any x) = c.x in
  let like dt = A.Any (A.create Rig.host dt (Iarray.of_array (shape_of c.x))) in
  let dsts =
    match c.r with
    | Arg _ -> [| like (A.dtype x); like D.Int64 |]
    | Moments -> [| like (A.dtype x); like (A.dtype x) |]
    | Monoid _ | Logsumexp -> [| like (A.dtype x) |]
  in
  match
    K.scan (identity (dtype_of c.x)) ~axis:c.axes.(0) (c.r, 0) ~dsts
      [| plain_of c.x |]
  with
  | A.Done -> Some (Array.map terms dsts)
  | A.Declined -> None
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* The most combinations a network scan's result [i] passes through:
   [max 0 (2 ⌈log₂ (i + 1)⌉ - 1)]. *)
let depth i =
  let rec log2 k p = if p >= k then 0 else 1 + log2 k (2 * p) in
  Float.of_int (max 0 ((2 * log2 (i + 1) 1) - 1))

(* nx_cpu.mli's bound of a Logsumexp scan's result [r] after [n] terms,
   at the unit roundoff [u]. *)
let lse_scan_bound ~n u r =
  (((7. *. depth (n - 1)) +. (6. *. log (Float.of_int n))) *. u)
  +. (u *. Float.abs r)

(* The running Logsumexp of a slice [ts] within nx_cpu.mli's bound at
   [u]: past a NaN, a NaN; past +∞, +∞; -∞ while every term is; else
   from a float64 running maximum and a compensated sum rescaled as it
   rises, within the bound at float64's roundoff besides. *)
let lse_scan u ts =
  let nan = ref false and m = ref Float.neg_infinity in
  let s = ref 0. and comp = ref 0. in
  let add x =
    let t = !s +. x in
    let lost =
      if Float.abs !s >= Float.abs x then !s -. t +. x else x -. t +. !s
    in
    comp := !comp +. lost;
    s := t
  in
  Array.mapi
    (fun i t ->
      (if is_nan t then nan := true
       else if not !nan then
         let x = value t in
         if x > !m then begin
           let f = if Float.is_finite !m then exp (!m -. x) else 0. in
           s := !s *. f;
           comp := !comp *. f;
           m := x
         end;
         if Float.is_finite !m then add (exp (x -. !m)));
      if !nan then [| Is Some_nan |]
      else if not (Float.is_finite !m) then [| Near (!m, 0.) |]
      else
        let r = !m +. log (!s +. !comp) in
        let b u = lse_scan_bound ~n:(i + 1) u r in
        [| Near (r, b u +. b eps) |])
    ts

(* nx_kinds.h's kind [name] over the vector [xs], float32 where [f32]
   else float64, in one call. *)
let kinded ~f32 name xs =
  let n = Array.length xs in
  if f32 then begin
    let ba k = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout k in
    let x = ba n and y = ba n in
    Array.iteri (fun i v -> x.{i} <- v) xs;
    Nx_kinds_support.run name x x y;
    Array.init n (fun i -> y.{i})
  end
  else begin
    let ba k = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout k in
    let x = ba n and y = ba n in
    Array.iteri (fun i v -> x.{i} <- v) xs;
    Nx_kinds_support.run name x x y;
    Array.init n (fun i -> y.{i})
  end

(* The bits of a Logsumexp scan of a slice [ts], float32 where [f32] else
   float64, by nx_cpu.mli's network replayed term by term as a stream,
   as replayed_moments does, each operation as stated: a term [x] is the
   state [(x, e (x - c))], and states [a] then [b] join into
   [m = Maximum (ma, mb)] and [sa e (ma - c) + sb e (mb - c)], [c] being [m]
   where finite and [0] elsewhere and [e] nx_kinds.h's exp, [0] below
   nx_cpu.mli's cut; the result is [m + log s], or [m] where [m] is not
   finite. Each operation rounds to the dtype. From the first NaN term on,
   the result is a NaN. *)
let lse_replay ~f32 ts =
  let rnd x = if f32 then round32 x else x in
  let kind = kinded ~f32 in
  let cut = if f32 then -87. else -708. in
  let first = Array.find_index is_nan ts in
  let k = Option.value first ~default:(Array.length ts) in
  (* The network, its exps from [ex] in the order it takes them: no exp's
     argument depends on a sum, so a first run records the arguments,
     which run through exp at once, and a second takes them. *)
  let replay ex =
    let shift m = if Float.is_finite m then m else 0. in
    let e z = if z < cut then 0. else ex z in
    let leaf x = (x, e (rnd (x -. shift x))) in
    let join (ma, sa) (mb, sb) =
      let m = Float.max ma mb in
      let c = shift m in
      let a = rnd (sa *. e (rnd (ma -. c))) in
      let b = rnd (sb *. e (rnd (mb -. c))) in
      (m, rnd (a +. b))
    in
    let held = Array.make 64 (0., 0.) and got = Array.make 65 (0., 0.) in
    Array.init k (fun t ->
        let st = ref (leaf (value ts.(t))) and h = ref 0 and j = ref t in
        while !j land 1 = 1 do
          st := join held.(!h) !st;
          incr h;
          j := !j lsr 1
        done;
        let res = if !j = 0 then !st else join got.(!h + 1) !st in
        held.(!h) <- !st;
        for g = 0 to !h do
          got.(g) <- res
        done;
        res)
  in
  let args = ref [] in
  ignore (replay (fun y -> args := y :: !args; 0.));
  let es = kind "exp" (Array.of_list (List.rev !args)) in
  let next = ref 0 in
  let states =
    replay (fun _ ->
        let e = es.(!next) in
        incr next;
        e)
  in
  let logs = kind "log" (Array.map snd states) in
  let bits r =
    if f32 then Int64.of_int32 (Int32.bits_of_float r) else Int64.bits_of_float r
  in
  Array.init (Array.length ts) (fun i ->
      if i >= k then Is Some_nan
      else
        let m, _ = states.(i) in
        let r = if Float.is_finite m then rnd (m +. logs.(i)) else m in
        Is (Bits (bits r)))

(* The running mean and variance of a slice [ts] within nx.cpu's bound at
   the unit roundoff [u]: past a NaN term, NaNs; past an infinity, the
   infinities' sum and a NaN, the mean a NaN too past infinities of both
   signs; else from float64 sums of the terms less the first and of their
   squares, within the bound at the depth [D] of the index and the range
   [R] and variance [V] of the terms so far, plus the sums' own bound in
   float64. *)
let moments_scan u ts =
  let gamma k = Float.of_int k *. eps /. (1. -. (Float.of_int k *. eps)) in
  let nan = ref false and pos = ref false and neg = ref false in
  let s1 = ref 0. and s2 = ref 0. and c = ref None in
  let lo = ref infinity and hi = ref neg_infinity in
  Array.mapi
    (fun j t ->
      let v = value t in
      if Float.is_nan v then nan := true
      else if v = infinity then pos := true
      else if v = neg_infinity then neg := true
      else begin
        let c =
          match !c with
          | Some c -> c
          | None ->
              c := Some v;
              v
        in
        s1 := !s1 +. (v -. c);
        s2 := !s2 +. ((v -. c) *. (v -. c));
        lo := Float.min !lo v;
        hi := Float.max !hi v
      end;
      let n = Float.of_int (j + 1) in
      if !nan || (!pos && !neg) then [| Is Some_nan; Is Some_nan |]
      else if !pos || !neg then
        [| Near ((if !pos then infinity else neg_infinity), 0.); Is Some_nan |]
      else
        let c = Option.get !c and d = depth j and k = j + 1 in
        let m = !s1 /. n and r = !hi -. !lo in
        let v = Float.max 0. ((!s2 /. n) -. (m *. m)) in
        let slack = gamma ((3 * k) + 4) *. ((r *. r) +. r +. Float.abs c) in
        let tm =
          (((4. *. d) +. 1.) *. u *. r) +. (u *. Float.abs (c +. m)) +. slack
        in
        let tv =
          (((7. *. d) +. 1.) *. u *. v)
          +. (((4. *. d *. d) -. (2. *. d)) *. u *. r *. Float.sqrt v)
          +. slack
        in
        [| Near (c +. m, tm); Near (v, tv) |])
    ts

(* The bits of a slice's running mean and variance by nx_cpu.mli's
   network, replayed term by term as a stream: term t's state joins its
   level's waiting state while t's low bits are 1, and the result at the
   level it reaches is that state, or the next level's last result joined
   with it. Each operation rounds to the dtype, float32 where [f32]. A
   slice with a term not finite takes nx_kinds.h's NaN rules, which this
   does not replay: [None]. *)
let moments_replay ~f32 ts =
  let rnd v = if f32 then round32 v else v in
  let bits v =
    if f32 then Int64.of_int32 (Int32.bits_of_float v) else Int64.bits_of_float v
  in
  (* A state: its mean less the slice's first term, S and count. *)
  let join (ma, sa, na) (mb, sb, nb) =
    let k = rnd (na +. nb) in
    let f = rnd (nb /. k) in
    let g = rnd (na *. f) and d = rnd (mb -. ma) in
    (rnd (ma +. rnd (d *. f)), rnd (rnd (sa +. sb) +. rnd (rnd (d *. d) *. g)), k)
  in
  let xs = Array.map value ts in
  if xs = [||] || not (Array.for_all Float.is_finite xs) then None
  else
    let c = xs.(0) in
    let held = Array.make 64 (0., 0., 0.) and got = Array.make 65 (0., 0., 0.) in
    Some
      (Array.mapi
         (fun t v ->
           let st = ref (rnd (v -. c), 0., 1.) and k = ref 0 and j = ref t in
           while !j land 1 = 1 do
             st := join held.(!k) !st;
             incr k;
             j := !j lsr 1
           done;
           let res = if !j = 0 then !st else join got.(!k + 1) !st in
           held.(!k) <- !st;
           for h = 0 to !k do
             got.(h) <- res
           done;
           let mu, sv, _ = res in
           let n = Float.of_int (t + 1) in
           [| Is (Bits (bits (rnd (c +. mu)))); Is (Bits (bits (rnd (sv /. n)))) |])
         xs)

(* The expected results of [c]'s scan in C order, by destination. *)
let stat_scan_expected (c : stat) =
  let (D.Any dt as d) = dtype_of c.x in
  let u = if D.bits dt = 32 then 0x1p-24 else 0x1p-53 in
  let slice ts =
    match c.r with
    | Arg e -> arg_scan d e ts
    | Logsumexp -> lse_scan u ts
    | Moments -> moments_scan u ts
    | Monoid _ -> invalid_arg "stat_scan_expected: a monoid"
  in
  let scanned = { (as_case c) with scan = true } in
  Array.init (width c.r) (fun j ->
      results scanned (terms c.x) ~fill:(Is Some_nan)
        ~reduce:(fun _ -> assert false)
        ~scan:(fun ts -> Array.map (fun e -> e.(j)) (slice ts)))

(* Arg scans are exact, Logsumexp and Moments scans within nx_cpu.mli's
   bounds and the bits of its network replayed, NaNs and infinities as
   Nx_kernel.reduction states them. *)
let law_stat_scan (b : Support.backend) (c : stat) =
  let s = shape_of c.x in
  let a = c.axes.(0) in
  let n = s.(a) in
  cover "an Arg scan's several chunks" (n > 4096);
  cover "no result" (total s = 0);
  cover "one term" (n = 1);
  cover "more than 64 slices" (n > 0 && total s / n > 64);
  cover "one slice past 256 terms" (n > 256 && total s = n);
  cover "more than 64 slices side by side, past 256 terms"
    (n > 256 && total s / n > 64);
  let last = Array.length s - 1 in
  cover "slices side by side" (a < last && s.(last) > 1);
  cover "a Moments scan" (c.r = Moments);
  List.iter (fun (k : values) -> cover k.name (c.values = k.name)) value_kinds;
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  let replayed = ref false in
  (match run_stat_scan b.kernels c with
  | None -> failf "nx.cpu declined %s" (reduction_name c.r)
  | Some got ->
      let want = stat_scan_expected c in
      let wanted p = Array.exists (Array.exists p) want in
      let stated = c.r = Logsumexp || c.r = Moments in
      cover "a NaN" (stated && wanted (function Is Some_nan -> true | _ -> false));
      cover "a -∞ term"
        (c.r = Logsumexp
        && Array.exists
             (fun t -> (not (is_nan t)) && value t = Float.neg_infinity)
             (terms c.x));
      cover "an infinity"
        (wanted (function Near (v, _) -> not (Float.is_finite v) | Is _ -> false));
      Array.iteri
        (fun k w ->
          equal ~msg:(Printf.sprintf "destination %d" k) (list string) []
            (misses w got.(k)))
        want;
      let (D.Any dt) = dtype_of c.x in
      let f32 = D.bits dt = 32 in
      let replay =
        match c.r with
        | Logsumexp ->
            Some (fun ts -> Some (Array.map (fun e -> [| e |]) (lse_replay ~f32 ts)))
        | Moments -> Some (moments_replay ~f32)
        | Arg _ | Monoid _ -> None
      in
      Option.iter
        (fun replay ->
          let net j =
            results { (as_case c) with scan = true } (terms c.x) ~fill:None
              ~reduce:(fun _ -> assert false)
              ~scan:(fun ts ->
                match replay ts with
                | None -> Array.map (fun _ -> None) ts
                | Some es -> Array.map (fun e -> Some e.(j)) es)
          in
          Array.iteri
            (fun j g ->
              let w =
                Array.mapi
                  (fun i w ->
                    match w with
                    | Some w ->
                        replayed := true;
                        w
                    | None -> Is (Bits (bits_of g.(i))))
                  (net j)
              in
              equal ~msg:"the replayed network" (list string) [] (misses w g))
            got)
        replay);
  cover "replayed" !replayed

let stat_example r d s axes ?view ?(outliers = 0) ?(specials = 0) values =
  let rank = Array.length s in
  let view =
    Option.value view
      ~default:
        { perm = Array.init rank Fun.id; stepped = false; reversed = None; broadcast = false }
  in
  let v = List.find (fun v -> v.name = values) value_kinds in
  let (D.Any dt) = d in
  let make n =
    if is_float d then float_elements dt n v ~outliers ~specials 3
    else A.Any (elements dt n ~specials 3)
  in
  stat_of r s axes view values make

(* The float32 or float64 [dt], of [bits], array of shape [s] of [xs]. *)
let array_bits (type a b) (dt : (a, b) D.t) bits s xs =
  let b = Bytes.create (D.bytes dt (Array.length xs)) in
  Array.iteri
    (fun i x ->
      if bits = 32 then Bytes.set_int32_le b (4 * i) (Int32.bits_of_float x)
      else Bytes.set_int64_le b (8 * i) (Int64.bits_of_float x))
    xs;
  let l = L.contiguous (Iarray.of_array s) in
  A.Any (A.v dt l (Rig.Buffer.of_string (Bytes.to_string b)))

(* Logsumexp of 70,000 terms along a ramp from 0 to -60 with noise, whose
   blocks' maxima differ: a sum rescaled block by block loses their bits. *)
let ramp_examples =
  List.concat_map
    (fun (D.Any dt) ->
      List.map
        (fun seed ->
          let rs = Random.State.make [| seed |] in
          let n = 70_000 in
          let xs =
            Array.init n (fun i ->
                (-60. *. Float.of_int i /. Float.of_int n)
                +. (1e-3 *. (Random.State.float rs 2. -. 1.)))
          in
          {
            r = Nx_kernel.Logsumexp;
            axes = [| 0 |];
            x = array_bits dt (D.bits dt) [| n |] xs;
            views = [];
            values = "ramp";
          })
        [ 0; 1; 2; 3 ])
    floats

(* Moments scans of slices of 15 to 25 tiles, one alone or three side by
   side, about 2^-20 to 2^20 in magnitude, of both signs, whose joins of
   the tiles' states round differently in each order. *)
let mixed_examples =
  let mixed (D.Any dt) s seed =
    let n = Array.fold_left ( * ) 1 s in
    let r = Random.State.make [| seed |] in
    let term _ =
      let sign = if Random.State.bool r then 1. else -1. in
      let e = Random.State.int r 41 - 20 in
      sign *. Float.ldexp (1. +. Random.State.float r 1.) e
    in
    array_bits dt (D.bits dt) s (Array.init n term)
  in
  List.map
    (fun (d, s, seed) ->
      { r = Nx_kernel.Moments; axes = [| 0 |]; x = mixed d s seed; views = []; values = "mixed" })
    [ (f32, [| 6400 |], 1); (f64, [| 5000 |], 2); (f32, [| 4000; 3 |], 3) ]

let stat_examples =
  Nx_kernel.
    [
      (* Groups of blocks on several threads, an outlier among offset terms. *)
      stat_example Moments f32 [| 70_000 |] [| 0 |] ~outliers:5000 "offset";
      (* Many outputs of several blocks: along stepped rows with NaNs and
         infinities, and strided down a reversed axis. *)
      stat_example Moments f64 [| 65; 1100 |] [| 1 |] ~specials:2000 "small"
        ~view:{ perm = [| 0; 1 |]; stepped = true; reversed = None; broadcast = false };
      stat_example Moments f32 [| 64; 1025; 2 |] [| 1 |] "offset"
        ~view:{ perm = [| 0; 1; 2 |]; stepped = false; reversed = Some 1; broadcast = false };
      stat_example Logsumexp f32 [| 64; 1030 |] [| 1 |] ~specials:2000 "wide";
      stat_example Logsumexp f32 [| 3000 |] [| 0 |] ~specials:20 "wide";
      stat_example Logsumexp f64 [| 300; 9 |] [| 1 |] ~specials:20 "small";
      stat_example Moments f64 [| 300; 9 |] [| 1 |] ~specials:20 "small";
      stat_example (Arg Max) f32 [| 46; 13; 2 |] [| 2 |] ~specials:300 "ties";
      stat_example (Arg Min) (D.Any D.Int64) [| 5; 2000 |] [| 0 |] "small";
      stat_example (Arg Max) f64 [| 3; 4; 5 |] [| 1 |] "ties"
        ~view:{ perm = [| 2; 0; 1 |]; stepped = true; reversed = Some 0; broadcast = true };
      stat_example (Arg Min) f32 [| 3 |] [||] "small";
      stat_example Moments f32 [| 3; 0 |] [| 1 |] "small";
      stat_example Moments f32 [| 0; 3 |] [| 0 |] "small";
      stat_example Logsumexp f64 [| 2; 0 |] [| 1 |] "small";
      stat_example (Arg Max) (D.Any D.Uint8) [| 200; 64 |] [| 1 |] "small";
      (* Every block's extreme ties with the others'. *)
      stat_example (Arg Max) (D.Any D.Int16) [| 70_000 |] [| 0 |] "small";
      stat_example (Arg Min) f32 [| 2; 5000 |] [| 1 |] "ties";
    ]

(* The two passes' arithmetic *)

(* A Logsumexp or Moments of a float, of finite terms: now and then a
   vector at a block's or a chunk's edge or of 70,000 terms, or many
   outputs of several blocks. *)
let two_pass_case =
  Gen.with_pp pp_stat
    (let open Gen in
     let* d = of_list ~pp:pp_dtype floats in
     let* r = of_list Nx_kernel.[ Logsumexp; Moments ] in
     let* s =
       frequency
         [
           (4, shape_gen);
           (1, many_long);
           ( 1,
             map (fun n -> [| n |])
               (of_list ~pp:Format.pp_print_int [ 1; 2; 17; 1023; 1025; 4097; 70_000 ]) );
         ]
     in
     let rank = Array.length s in
     let* keep = array ~size:(constant rank) bool in
     let axes = Array.of_list (List.filter (fun i -> keep.(i)) (List.init rank Fun.id)) in
     let* v = view_of rank in
     let* values = of_list value_kinds in
     let* outliers = frequency [ (3, constant 0); (1, constant 50) ] in
     let+ seed = int in
     let (D.Any dt) = d in
     stat_of r s axes v values.name (fun n ->
         float_elements dt n values ~outliers ~specials:0 seed))

(* Many outputs of several blocks, along rows, side by side and with their
   terms two apart beside no output's. *)
let two_pass_examples =
  let stepped = { perm = [| 0; 1 |]; stepped = true; reversed = None; broadcast = false } in
  List.concat_map
    (fun r ->
      [
        stat_example r f32 [| 64; 1025 |] [| 1 |] "offset";
        stat_example r f32 [| 64; 1025; 2 |] [| 1 |] "offset";
        stat_example r f32 [| 64; 1025 |] [| 1 |] ~view:stepped "offset";
      ])
    Nx_kernel.[ Logsumexp; Moments ]

(* The float [v] as a term of [d]. *)
let fterm d v =
  F (v, if same d f32 then Int64.of_int32 (Int32.bits_of_float v) else Int64.bits_of_float v)

(* Logsumexp: the maximum [m] in lane order, [c] where it is finite or
   else [0], then [c + log] of the sum in lane order of [exp (x - c)], an
   exp of an argument below the cut taken as [0]. Moments: the sum over
   the count, the mean, then the sums [s1] of the deviations [d] from it
   and [s2] of their squares, and [(s2 - s1 (s1 / n)) / n], +0 where that
   is negative, or [s2] where that is +inf. Each operation rounds to the
   dtype, each sum is in lane order from +0 and each exp and log is
   nx_kinds.h's. *)
let law_two_passes (b : Support.backend) (c : stat) =
  let d = dtype_of c.x in
  let f32 = same d f32 in
  let round x = if f32 then round32 x else x in
  let fold m xs = value (ordered (op d m) (Array.map (fterm d) xs)) in
  let kind = kinded ~f32 in
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  cover "several blocks" (per > 1024);
  cover "groups of blocks" (per >= 70_000);
  cover "many outputs of several blocks" (per > 1024 && total (stat_shape c) >= 64);
  cover "no term" (per = 0 && total (stat_shape c) > 0);
  List.iter (fun (k : values) -> cover k.name (c.values = k.name)) value_kinds;
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  let outs = Array.of_list (List.map (Array.map value) (outputs (as_case c) (terms c.x))) in
  let expect v = if Float.is_nan v then Is Some_nan else Is (Bits (bits_of (fterm d v))) in
  match run_stat b.kernels c with
  | None -> failf "nx.cpu declined %s" (reduction_name c.r)
  | Some got -> (
      match c.r with
      | Logsumexp ->
          let cut = if f32 then -87. else -708. in
          let one xs =
            let m = fold Max xs in
            let c0 = if Float.is_finite m then m else 0. in
            let ys = Array.map (fun x -> round (x -. c0)) xs in
            let es = kind "exp" (Array.map (Float.max cut) ys) in
            let es = Array.mapi (fun i e -> if ys.(i) < cut then 0. else e) es in
            expect (round (c0 +. (kind "log" [| fold Sum es |]).(0)))
          in
          equal ~msg:"logsumexp" (list string) [] (misses (Array.map one outs) got.(0))
      | _ ->
          let one xs =
            let n = round (Float.of_int (Array.length xs)) in
            let mean = round (fold Sum xs /. n) in
            let dev x = round (x -. mean) in
            let s1 = fold Sum (Array.map dev xs) in
            let s2 = fold Sum (Array.map (fun x -> round (dev x *. dev x)) xs) in
            let q = round (round (s2 -. round (s1 *. round (s1 /. n))) /. n) in
            (expect mean, expect (if s2 = infinity then s2 else if q < 0. then 0. else q))
          in
          let want = Array.map one outs in
          equal ~msg:"mean" (list string) [] (misses (Array.map fst want) got.(0));
          equal ~msg:"variance" (list string) [] (misses (Array.map snd want) got.(1)))

(* A reduction or a scan into its own operand gives the bits of one into
   fresh destinations, as nx_cpu.mli states, though it overwrites the terms
   it reads: a Moments scan whose terms take its exact pass, its mean or its
   variance in place; running sums, Logsumexps and Arg's extremes of slices
   past a chunk, more units than a job splits into chunks; and a float16
   sum along no axis of signalling NaNs, which stay NaNs. A NaN result is
   any NaN. *)
let test_in_place (b : Support.backend) () =
  let module K = (val b.kernels) in
  let copy (A.Any x) = A.Any (A.copy x) in
  let fresh (A.Any x) =
    A.Any (A.create Rig.host (A.dtype x) (L.shape (A.layout x)))
  in
  let scan r axis x dsts =
    match
      K.scan (identity (dtype_of x)) ~axis (r, 0) ~dsts [| plain_of x |]
    with
    | A.Done ->
        Array.map
          (fun d ->
            Array.map
              (fun e -> if Elements.nan_of (dtype_of d) e then "nan" else e)
              (Elements.elements d))
          dsts
    | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a
  in
  (* Destination [k] the operand; an Arg's positions int64. *)
  let check what (r : Nx_kernel.reduction) axis x k =
    let dst j =
      match r with
      | Arg _ when j = 1 ->
          A.Any (A.create Rig.host D.Int64 (Iarray.of_array (shape_of x)))
      | _ -> fresh x
    in
    let n = match r with Moments | Arg _ -> 2 | Monoid _ | Logsumexp -> 1 in
    let want = scan r axis x (Array.init n dst) in
    let x' = copy x in
    let dsts = Array.init n (fun j -> if j = k then x' else dst j) in
    equal ~msg:what (array (array string)) want (scan r axis x' dsts)
  in
  let terms = A.Any (A.of_array D.Float32 [| 5 |] [| 1.; 2.; infinity; 4.; 5. |]) in
  let long =
    let l = elements D.Float32 (300 * 4100) ~specials:0 7 in
    A.Any (Option.get (A.move (M.Reshape [| 300; 4100 |]) l))
  in
  check "a Moments scan's mean in place" Moments 0 terms 0;
  check "a Moments scan's variance in place" Moments 0 terms 1;
  check "running sums in place" (Monoid Sum) 1 long 0;
  check "a Logsumexp scan in place" Logsumexp 1 long 0;
  check "an Arg scan's extreme in place" (Arg Max) 1 long 0;
  let codes = [| "\001\124"; "\000\060"; "\001\252" |] in
  let x = Elements.of_elements (D.Any D.Float16) [| 3 |] codes in
  let x' = copy x in
  let answer =
    K.reduce (identity (D.Any D.Float16)) ~shape:[| 3 |] ~axes:[||]
      [| (Monoid Sum, 0) |] ~dsts:[| x' |] [| plain_of x' |]
  in
  equal ~msg:"its answer" bool true (answer = A.Done);
  equal ~msg:"a float16 sum along no axis in place"
    (array (Elements.same (D.Any D.Float16)))
    codes (Elements.elements x')

(* A scan by Arg, Logsumexp or Moments along the rows of [k; n] terms: a
   few rows of up to a chunk's terms, past one now and then. *)
let lanes_case =
  Gen.with_pp pp_stat
    (let open Gen in
     let* r =
       of_list Nx_kernel.[ Arg Max; Arg Min; Logsumexp; Moments ]
     in
     let* d =
       of_list ~pp:pp_dtype
         (match r with Arg _ -> D.Any D.Int32 :: floats | _ -> floats)
     in
     let* k = int_range 1 20 in
     let* n = frequency [ (4, int_range 1 600); (1, int_range 4097 9000) ] in
     let* values = of_list value_kinds in
     let* specials = frequency [ (2, constant 0); (1, constant 100) ] in
     let+ seed = int in
     let (D.Any dt) = d in
     let make m =
       if is_float d then float_elements dt m values ~outliers:0 ~specials seed
       else A.Any (elements dt m ~specials seed)
     in
     let plain = { perm = [| 0; 1 |]; stepped = false; reversed = None; broadcast = false } in
     stat_of r [| k; n |] [| 1 |] plain values.name make)

(* Each row of a scan by Arg, Logsumexp or Moments has the bits of the row
   scanned alone, a NaN as any NaN, whichever way its lanes run: the rows
   side by side, staged into lanes; their transpose scanned down its
   columns, whose lanes lie side by side; and one row alone, whose chunks
   run as lanes past a chunk, or a Logsumexp or Moments scan's tiles. *)
let law_lanes (b : Support.backend) (c : stat) =
  let s = shape_of c.x in
  let k = s.(0) and n = s.(1) in
  cover "past a chunk" (n > 4096);
  cover "sixteen lanes" (k >= 16);
  let (A.Any x) = c.x in
  let move m a = Option.get (A.move m a) in
  let scan a axis =
    let c = { c with x = A.Any a; axes = [| axis |] } in
    match run_stat_scan b.kernels c with
    | Some g -> Array.map (Array.map (fun t -> show (want_of t))) g
    | None -> failf "nx.cpu declined %s" (reduction_name c.r)
  in
  let rows = scan x 1 in
  let cols = scan (A.copy (move (M.Permute [| 1; 0 |]) x)) 0 in
  for i = 0 to k - 1 do
    let whole = { M.start = 0; count = n; step = 1 } in
    let row = move (M.Slice [| { M.start = i; count = 1; step = 1 }; whole |]) x in
    let alone = scan (move (M.Reshape [| n |]) (A.copy row)) 0 in
    Array.iteri
      (fun d a ->
        let msg = Printf.sprintf "row %d, destination %d" i d in
        equal ~msg:(msg ^ ", side by side") (array string) a
          (Array.sub rows.(d) (i * n) n);
        equal ~msg:(msg ^ ", transposed") (array string) a
          (Array.init n (fun j -> cols.(d).((j * k) + i))))
      alone
  done

let lanes_examples =
  Nx_kernel.
    [
      stat_example Logsumexp f32 [| 17; 5000 |] [| 1 |] "wide";
      stat_example (Arg Max) f32 [| 17; 5000 |] [| 1 |] ~specials:100 "ties";
      stat_example Moments f32 [| 17; 5000 |] [| 1 |] "offset";
      stat_example Moments f64 [| 3; 1000 |] [| 1 |] ~specials:100 "small";
    ]

(* Scans past several tiles and an Arg's chunks, one slice's several at a
   time and slices side by side, and a NaN, an infinity and -∞ among finite
   terms in a Logsumexp's terms. *)
let stat_scan_examples =
  let vector xs =
    {
      r = Logsumexp;
      axes = [| 0 |];
      x = A.Any (A.of_array D.Float32 [| Array.length xs |] xs);
      views = [];
      values = "small";
    }
  in
  Nx_kernel.
    [
      stat_example Logsumexp f32 [| 20_000 |] [| 0 |] ~specials:3000
        "wide";
      stat_example Logsumexp f64 [| 5000; 3 |] [| 0 |] "small";
      (* One slice of 274 tiles, the last partial; 40 slices of two tiles,
         the last partial, contiguous and staged; and 70 slices, past a
         unit's. *)
      stat_example Logsumexp f32 [| 70_000 |] [| 0 |] ~specials:3000
        "small";
      stat_example Logsumexp f32 [| 300; 40 |] [| 0 |] "wide";
      stat_example Logsumexp f64 [| 40; 300 |] [| 1 |] "offset";
      stat_example Logsumexp f32 [| 514; 70 |] [| 0 |] ~specials:300 "wide";
      stat_example (Arg Min) f64 [| 70_000 |] [| 0 |] ~specials:3000 "ties";
      stat_example (Arg Max) f32 [| 9000 |] [| 0 |] ~specials:300 "ties";
      stat_example (Arg Min) (D.Any D.Int32) [| 300; 5 |] [| 0 |] "small";
      (* Moments along rows of a tile and a partial one, and down their
         transpose. *)
      stat_example Moments f32 [| 40; 45 |] [| 1 |] "small";
      stat_example Moments f32 [| 45; 40 |] [| 0 |] "small"
        ~view:{ perm = [| 1; 0 |]; stepped = false; reversed = None; broadcast = false };
      vector [| 1.; Float.infinity; 2. |];
      vector [| 1.; Float.nan; 2. |];
      vector [| Float.neg_infinity; 1.; Float.neg_infinity; 3. |];
    ]

(* A complex Arg scan orders real parts, then imaginary ones; a number with
   a NaN part holds from there on. *)
let test_complex_arg_scan (b : Support.backend) () =
  let module K = (val b.kernels) in
  let z re im = { Complex.re; im } in
  let xs = [| z 1. 0.; z 1. 1.; z 0. 5.; z Float.nan 0.; z 3. 0. |] in
  let x = A.of_array D.Complex64 [| 5 |] xs in
  let scan e =
    let v = A.create Rig.host D.Complex64 [| 5 |]
    and at = A.create Rig.host D.Int64 [| 5 |] in
    let answer =
      K.scan (identity (D.Any D.Complex64)) ~axis:0 (Arg e, 0)
        ~dsts:[| A.Any v; A.Any at |] [| Nx_kernel.Plain x |]
    in
    equal ~msg:"its answer" bool true (answer = A.Done);
    (Elements.elements (A.Any v), A.to_array at)
  in
  let pick ks = Array.map (fun k -> (Elements.elements (A.Any x)).(k)) ks in
  let zs = pair (array (Elements.same (D.Any D.Complex64))) (array int64) in
  equal ~msg:"Arg Max" zs
    (pick [| 0; 1; 1; 3; 3 |], [| 0L; 1L; 1L; 3L; 3L |])
    (scan Max);
  equal ~msg:"Arg Min" zs
    (pick [| 0; 0; 2; 3; 3 |], [| 0L; 0L; 2L; 3L; 3L |])
    (scan Min)

(* Guard pages *)

(* A reduction by [r] along [axes], or a scan along [axes.(0)], of [x]
   through the views [views]: each that nx.cpu computes. *)
type guard = {
  scan : bool;
  r : Nx_kernel.reduction;
  axes : int array;
  x : A.any;
  views : string list;
}

let pp_guard ppf (g : guard) =
  let (A.Any x) = g.x in
  Format.fprintf ppf "%s %s along %a of %a %a (%s)"
    (if g.scan then "scan" else "reduce")
    (reduction_name g.r) pp_ints g.axes D.pp (A.dtype x) L.pp (A.layout x)
    (String.concat ", " g.views)

(* Innermost extents of one element to a few vectors of the widest dtype
   past the widest vector, and at the edges of lane blocks and scan
   chunks. *)
let inner =
  let open Gen in
  frequency
    [
      (6, int_range 1 72);
      (2, of_list ~pp:Format.pp_print_int [ 127; 128; 129; 255; 256; 257 ]);
      (1, of_list ~pp:Format.pp_print_int [ 1023; 1025; 4095; 4097 ]);
    ]

let guard_case =
  Gen.with_pp pp_guard
    (let open Gen in
     let* d =
       of_list ~pp:pp_dtype
         (floats @ floats @ floats @ floats @ base @ narrow_dtypes)
     in
     let* scan = bool in
     let float = is_float d
     and boolean = same d (D.Any D.Bool) || same d (D.Any D.Bit) in
     (* Moments and Logsumexp, which floats alone take, come twice as
        often. *)
     let* r =
       of_list
         (Nx_kernel.[ Monoid Max; Monoid Min ]
         @ (if boolean then [] else Nx_kernel.[ Monoid Sum; Monoid Prod ])
         @ (if float then Nx_kernel.[ Moments; Moments ] else [])
         @ Nx_kernel.[ Arg Max; Arg Min ]
         @ if float then Nx_kernel.[ Logsumexp; Logsumexp ]
           else [])
     in
     let* n = inner in
     let* outer = array ~size:(int_range 0 2) (int_range 1 5) in
     let s = Array.append outer [| n |] in
     let rank = Array.length s in
     let* axes =
       if scan then map (fun a -> [| a |]) (int_range 0 (rank - 1))
       else
         let+ keep = array ~size:(constant rank) bool in
         Array.of_list (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let* v = view_of rank in
     let* specials = frequency [ (3, constant 0); (1, constant 20) ] in
     let+ seed = int in
     { scan; r; axes; x = operand d s v ~specials seed; views = view_names v })

(* A running sum of bfloat16 [+inf; -inf] at the page after: its last
   result is a NaN made from numbers, and no term is a NaN for the results
   to take. *)
let guard_examples =
  let x = A.Any (A.of_array D.Bfloat16 [| 2 |] [| infinity; neg_infinity |]) in
  [
    ( ({ scan = true; r = Monoid Sum; axes = [| 0 |]; x; views = [] } : guard),
      Support.After );
  ]

(* The same bits with the operand, through its view, and each destination
   against an inaccessible page. *)
let law_guards (b : Support.backend) ((g : guard), side) =
  let (module K : Nx_kernel.S) = b.kernels in
  let after = side = Support.After in
  let rank = Array.length (shape_of g.x) in
  let n = (shape_of g.x).(rank - 1) in
  collect (Printf.sprintf "%s %s" (if g.scan then "scan" else "reduce")
       (reduction_name g.r));
  let is r = g.r = r in
  cover "a monoid reduced" ((not g.scan) && is (Monoid Sum));
  cover "a monoid scanned" (g.scan && is (Monoid Max));
  cover "Logsumexp" (is Logsumexp);
  cover "an Arg" (is (Arg Max) || is (Arg Min));
  cover "an Arg or Logsumexp scanned"
    (g.scan && (is (Arg Max) || is (Arg Min) || is Logsumexp));
  cover "Moments reduced" ((not g.scan) && is Moments);
  cover "Moments scanned" (g.scan && is Moments);
  cover "the innermost axis reduced" (Array.mem (rank - 1) g.axes);
  cover "outer axes alone reduced"
    (g.axes <> [||] && not (Array.mem (rank - 1) g.axes));
  cover "an innermost extent past a vector, not a multiple of one"
    (n > 32 && n mod 16 <> 0);
  cover "a stepped operand at the page before"
    ((not after) && List.mem "stepped" g.views);
  cover "a reversed operand at the page after"
    (after && List.mem "reversed" g.views);
  cover "a broadcast operand" (List.mem "broadcast" g.views);
  let (A.Any x) = g.x in
  let dt = D.Any (A.dtype x) in
  let s =
    let s = shape_of g.x in
    if g.scan then s
    else
      Array.of_list
        (List.filteri (fun i _ -> not (Array.mem i g.axes)) (Array.to_list s))
  in
  let dsts =
    match g.r with
    | Arg _ -> [| (dt, s); (D.Any D.Int64, s) |]
    | Moments -> [| (dt, s); (dt, s) |]
    | Monoid _ | Logsumexp -> [| (dt, s) |]
  in
  let call xs ds =
    let p = identity dt and loads = [| plain_of xs.(0) |] in
    if g.scan then K.scan p ~axis:g.axes.(0) (g.r, 0) ~dsts:ds loads
    else
      K.reduce p ~shape:(loop_shape loads) ~axes:(Iarray.of_array g.axes)
        [| (g.r, 0) |] ~dsts:ds loads
  in
  equal Elements.answer A.Done
    (Elements.same_guarded ~any_nan:true side ~call [| g.x |] dsts)

(* Fused programs *)

(* How a program's second load spreads over the loop: an operand of the
   loop's shape, a row of lanes broadcast along the reduced axes, or a
   kernel broadcast along the kept ones, as a convolution's is. *)
type spread = Full | Lanes | Kernel

(* A reduction of output [k] of [prog], a program over loads of one loop
   shape, by [red] along [axes], or with [scan] its scan along [axes.(0)].
   Load [i] reads [operands.(i)], padded with the fill and padding of
   [pads.(i)] where given; [guard] puts each operand against an
   inaccessible page on its side. *)
type fused = {
  name : string;
  shape : int array;
  scan : bool;
  prog : P.t;
  k : int;
  red : Nx_kernel.reduction;
  axes : int array;
  operands : A.any array;
  pads : (string * Nx_kernel.pad) option array;
  second : spread;
  guard : Support.side option;
  fviews : string list;
}

let pp_fused ppf c =
  Format.fprintf ppf "%s, %s %s of output %d along %a of %a%s%s (%s)" c.name
    (if c.scan then "scan" else "reduction")
    (reduction_name c.red) c.k pp_ints c.axes pp_ints c.shape
    (if Array.exists Option.is_some c.pads then ", padded" else "")
    (match c.guard with
    | None -> ""
    | Some Support.Before -> ", guarded before"
    | Some Support.After -> ", guarded after")
    (String.concat ", " c.fviews)

(* The loads over [operands], padded as [pads] say. *)
let loads_with pads operands =
  Array.mapi
    (fun i (A.Any x as a) ->
      match pads.(i) with
      | Some (fill, pad) -> padded_load a fill pad
      | None -> Nx_kernel.Plain x)
    operands

let loads_of c = loads_with c.pads c.operands

(* The arrays [c]'s loads read, a padded one built here. *)
let reads_of c =
  Array.mapi
    (fun i x ->
      match c.pads.(i) with
      | Some (fill, pad) -> Elements.pad_array x fill pad
      | None -> x)
    c.operands

(* The product of two factors of [fdt] at [d], or of one by itself, cast
   where they differ: the nodes and the output. *)
let factors ~same (D.Any l) (D.Any p as d) =
  let nodes =
    match (same, D.equal_witness l p) with
    | true, Some _ -> P.[| In 0; Op2 (Binary Mul, 0, 0) |]
    | true, None -> P.[| In 0; Op1 (Cast, d, 0); Op2 (Binary Mul, 1, 1) |]
    | false, Some _ -> P.[| In 0; In 1; Op2 (Binary Mul, 0, 1) |]
    | false, None ->
        P.[| In 0; In 1; Op1 (Cast, d, 0); Op1 (Cast, d, 1); Op2 (Binary Mul, 2, 3) |]
  in
  (nodes, [| Array.length nodes - 1 |])

(* The programs over a loop of the float or complex dtype [d], with
   factors of [fdt]: each one's name, its loads' dtypes, whether its second
   load is a row of lanes, its nodes, its outputs, and the output the
   reduction reads. [rank] bounds its coordinates. *)
let templates (D.Any dt as d) fdt rank =
  let num x =
    match dt with
    | D.Float32 -> P.Const (d, P.bits D.Float32 x)
    | D.Float64 -> P.Const (d, P.bits D.Float64 x)
    | D.Complex64 -> P.Const (d, P.bits D.Complex64 { Complex.re = x; im = 0. })
    | D.Complex128 ->
        P.Const (d, P.bits D.Complex128 { Complex.re = x; im = 0. })
    | _ -> invalid_arg "templates: a float or complex dtype"
  in
  let i64 = D.Any D.Int64 in
  let two, out2 = factors ~same:false fdt d in
  let one, out1 = factors ~same:true fdt d in
  if D.is D.Complex dt then
    [
      ("a difference with lanes", [| d; d |], true,
       P.[| In 0; In 1; Op2 (Binary Sub, 0, 1) |], [| 2 |], 0);
      ("a scaled operand", [| d |], false,
       P.[| In 0; num 0.5; Op2 (Binary Mul, 0, 1) |], [| 2 |], 0);
      ("a product of two operands", [| d; d |], false,
       P.[| In 0; In 1; Op2 (Binary Mul, 0, 1) |], [| 2 |], 0);
    ]
  else
  [
    ("exp of a difference with lanes", [| d; d |], true,
     P.[| In 0; In 1; Op2 (Binary Sub, 0, 1); Op1 (Unary Exp, d, 2) |],
     [| 3 |], 0);
    ("a scaled operand", [| d |], false,
     P.[| In 0; num 0.5; Op2 (Binary Mul, 0, 1) |], [| 2 |], 0);
    ("a product of two operands", [| d; d |], false,
     P.[| In 0; In 1; Op2 (Binary Mul, 0, 1) |], [| 2 |], 0);
    ("a product of two factors", [| fdt; fdt |], false, two, out2, 0);
    ("a squared factor", [| fdt |], false, one, out1, 0);
    ("a log, NaN below 0", [| d |], false,
     P.[| In 0; Op1 (Unary Log, d, 0) |], [| 1 |], 0);
    ("the second of two outputs", [| d |], false,
     P.[| In 0; Op1 (Unary Neg, d, 0); Op1 (Unary Abs, d, 0) |], [| 1; 2 |], 1);
    ("a non-zero test", [| d |], false,
     P.[| In 0; num 0.; Op2 (Compare Not_equal, 0, 1) |], [| 2 |], 0);
    ("a count of positives", [| d |], false,
     P.[| In 0; num 0.; Op2 (Compare Less, 1, 0); Op1 (Cast, i64, 2) |],
     [| 3 |], 0);
    ("a constant alone, of no load", [||], false, [| num 0.5 |], [| 0 |], 0);
    ("an operand, beside its negation", [| d |], false,
     P.[| In 0; Op1 (Unary Neg, d, 0) |], [| 0; 1 |], 0);
    ("an operand cast into bfloat16", [| d |], false,
     P.[| In 0; Op1 (Cast, D.Any D.Bfloat16, 0) |], [| 1 |], 0);
    ("an operand cast into int4", [| d |], false,
     P.[| In 0; Op1 (Cast, D.Any D.Int4, 0) |], [| 1 |], 0);
    ("an int4 operand cast up", [| D.Any D.Int4 |], false,
     P.[| In 0; Op1 (Cast, d, 0) |], [| 1 |], 0);
    ("a bit operand cast up", [| D.Any D.Bit |], false,
     P.[| In 0; Op1 (Cast, d, 0) |], [| 1 |], 0);
  ]
  @
  if rank = 0 then []
  else
    [
      ("an operand plus its last coordinate", [| d |], false,
       P.[| In 0; Coord (rank - 1); Op1 (Cast, d, 1); Op2 (Binary Add, 0, 2) |],
       [| 3 |], 0);
      ("its last coordinate alone, of no load", [||], false,
       P.[| Coord (rank - 1); Op1 (Cast, d, 0) |], [| 1 |], 0);
    ]

let template d fdt rank name =
  List.find (fun (n, _, _, _, _, _) -> n = name) (templates d fdt rank)

(* The reductions an output of [dt] takes, or with [scan] its scans. *)
let reductions_of ~scan (D.Any dt) : Nx_kernel.reduction list =
  match D.kind dt with
  | D.Float ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Logsumexp;
        Arg Max; Arg Min; Moments ]
  | D.Complex ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Arg Max; Arg Min ]
  | D.Boolean -> [ Monoid Max; Monoid Min ]
  | _ when scan ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Arg Max ]
  | _ -> [ Monoid Sum; Monoid Max; Monoid Min; Arg Max ]

(* The factors' dtypes a loop of [d] takes: [d] mostly, the narrow floats
   and float64 for float32, float32 for float64. *)
let factor_dtypes d =
  let open Gen in
  if same d f32 then
    frequency
      [
        (4, constant f32);
        ( 4,
          of_list ~pp:pp_dtype
            D.[ Any Bfloat16; Any Float16; Any Float8_e4m3fn; Any Float8_e5m2;
                Any Float4_e2m1fn; Any Float64 ] );
      ]
  else if same d f64 then frequency [ (3, constant f64); (1, constant f32) ]
  else constant d

(* [n] elements of [d]: float32's values and specials cast into a narrow
   float, else [elements]. *)
let product_elements (D.Any dt as d) n ~specials seed =
  match dt with
  | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
  | D.Float4_e2m1fn ->
      cast_into d (A.Any (elements D.Float32 n ~specials seed))
  | _ -> A.Any (elements dt n ~specials seed)

let product_operand d s v ~specials seed =
  operand_with (fun n -> product_elements d n ~specials seed) s v

let plain_view r =
  { perm = Array.init r Fun.id; stepped = false; reversed = None; broadcast = false }

(* A load of [d] over the loop [s] reduced along [axes], spread as
   [second] says, its elements through the view [v]. *)
let spread_operand d s axes second v ~specials seed =
  match second with
  | Full -> product_operand d s v ~specials seed
  | Lanes | Kernel ->
      let row =
        Array.mapi (fun i e -> if Array.mem i axes = (second = Lanes) then 1 else e) s
      in
      let (A.Any x) = product_operand d row v ~specials seed in
      A.Any (Option.get (A.move (M.Broadcast (Iarray.of_array s)) x))

let fused_shapes =
  let open Gen in
  let small = array ~size:(int_range 1 3) (int_range 1 5) in
  let rows =
    let* r = int_range 1 40 in
    let+ k = of_list ~pp:Format.pp_print_int [ 9; 17; 64; 1100; 5000 ] in
    [| r; k |]
  in
  (* Rows of few terms, which units take in bands of rows. *)
  let few =
    let* r = frequency [ (3, int_range 1 20); (1, of_list ~pp:Format.pp_print_int [ 64; 130 ]) ] in
    let+ n = frequency [ (8, int_range 1 16); (1, constant 17); (1, int_range 18 40) ] in
    [| r; n |]
  in
  let columns =
    let* k = of_list ~pp:Format.pp_print_int [ 9; 17; 300; 1100 ] in
    let+ r = int_range 1 200 in
    [| k; r |]
  in
  let apart =
    let+ a = int_range 2 20
    and+ b = int_range 2 5
    and+ c = int_range 2 20
    and+ d = int_range 2 9 in
    [| a; b; c; d |]
  in
  let empty =
    of_list ~pp:pp_ints [ [| 0 |]; [| 0; 5 |]; [| 3; 0 |]; [| 2; 0; 4 |] ]
  in
  (* Rank 0, no element and two reduced axes split by a kept one are each
     drawn often enough that every run of 100 cases reaches them. *)
  frequency
    [ (3, small); (3, rows); (3, few); (2, columns); (1, long); (5, apart);
      (4, constant ~pp:pp_ints [||]); (4, empty) ]

(* The padding of an operand of shape [s] that Nx_kernel.Shape takes. *)
let fitting_pad s =
  Gen.such_that
    (fun p -> Result.is_ok (Nx_kernel.Shape.padded p s))
    (pad_of s)

let fused_any =
  let open Gen in
  with_pp pp_fused
  @@ let* d =
       frequency
         [
           (3, constant ~pp:pp_dtype (D.Any D.Float32));
           (3, constant ~pp:pp_dtype (D.Any D.Float64));
           (1, of_list ~pp:pp_dtype D.[ Any Complex64; Any Complex128 ]);
         ]
     in
  let* fdt = factor_dtypes d in
  let* s = fused_shapes in
  (* One float loop in four reads its first load padded. *)
  let* pad =
    if not (is_float d) then constant None
    else frequency [ (3, constant None); (1, map Option.some (fitting_pad s)) ]
  in
  let loop =
    match pad with
    | None -> s
    | Some p -> Result.get_ok (Nx_kernel.Shape.padded p s)
  in
  let r = Array.length loop in
  (* A scan runs along an axis: a shape of rank 0 has none. *)
  let* scan =
    if r = 0 then constant false
    else frequency [ (2, constant false); (1, constant true) ]
  in
  (* Any subset of the axes, or for a shape of four, the first and third
     at times: two reduced axes split by a kept one, the last kept. *)
  let* axes =
    let any =
      let+ keep = array ~size:(constant r) bool in
      let axes = List.filter (fun i -> keep.(i)) (List.init r Fun.id) in
      Array.of_list (if axes = [] && r > 0 then [ r - 1 ] else axes)
    in
    if scan then map (fun a -> [| a |]) (int_range 0 (r - 1))
    else if r = 4 then frequency [ (1, any); (9, constant [| 0; 2 |]) ]
    else any
  in
  let ts =
    List.filter
      (fun (_, ins, _, _, _, _) -> pad = None || ins <> [||])
      (templates d fdt r)
  in
  let* t = int_range 0 (List.length ts - 1) in
  let name, ins, lanes, nodes, outs, k = List.nth ts t in
  let prog = P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
      ~outs:(Iarray.of_array outs) in
  let o = P.dtype prog outs.(k) in
  let* red = of_list (reductions_of ~scan o) in
  let* second =
    if lanes then constant Lanes
    else if Array.length ins = 2 then frequency [ (2, constant Full); (1, constant Kernel) ]
    else constant Full
  in
  let* v = view_of (Array.length s) in
  let* lv = view_of r in
  let* specials = frequency [ (3, constant 0); (1, constant 20) ] in
  let* guard =
    frequency
      [ (3, constant None); (1, constant (Some Support.Before));
        (1, constant (Some Support.After)) ]
  in
  let* fill =
    match pad with
    | Some _ -> of_list (fills ins.(0))
    | None -> constant ""
  in
  let+ seed = int in
  let n = Array.length ins in
  let operand i =
    if i = 0 then
      product_operand ins.(0) (if pad = None then loop else s) v ~specials seed
    else if i = 1 then spread_operand ins.(1) loop axes second lv ~specials (seed + 1)
    else product_operand ins.(i) loop lv ~specials (seed + i)
  in
  {
    name; shape = loop; scan; prog; k; red; axes; second; guard;
    operands = Array.init n operand;
    pads = Array.init n (fun i -> if i = 0 then Option.map (fun p -> (fill, p)) pad else None);
    fviews = view_names v;
  }

(* The case of the template [name] over [d] with factors of [fdt] of the
   loop [loop], reduced by [red] along [axes], or with [scan] scanned, over
   [operands] padded as [pads] say. *)
let fused_of ?(d = f32) ?fdt ?(scan = false) ?(second = Full) ?pads
    ?(fviews = []) name red loop axes operands =
  let fdt = Option.value fdt ~default:d in
  let _, ins, lanes, nodes, outs, k = template d fdt (Array.length loop) name in
  let pads = Option.value pads ~default:(Array.map (fun _ -> None) operands) in
  {
    name; shape = loop; scan; k; red; axes; guard = None; fviews; operands; pads;
    second = (if lanes then Lanes else second);
    prog =
      P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
        ~outs:(Iarray.of_array outs);
  }

(* The case of the template [name] over [d] of shape [s], its first load
   through [view], by [red] along [axes], or with [scan] scanned. *)
let fused_example ?(d = D.Any D.Float32) ?fdt ?scan ?(second = Full) ?view
    ?(specials = 0) name red s axes =
  let r = Array.length s in
  let fdt = Option.value fdt ~default:d in
  let _, ins, lanes, _, _, _ = template d fdt r name in
  let plain = plain_view r in
  let second = if lanes then Lanes else second in
  let load j =
    if j = 1 then spread_operand ins.(1) s axes second plain ~specials (7 + j)
    else product_operand ins.(j) s (if j = 0 then Option.value view ~default:plain else plain) ~specials (7 + j)
  in
  let c = fused_of ~d ~fdt ?scan ~second name red s axes (Array.init (Array.length ins) load) in
  { c with fviews = (match view with Some v -> view_names v | None -> []) }

(* A padding of [lo] and [hi] along each axis, with [windows]. *)
let pad_v ?(windows = [||]) lo hi =
  let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
  let interior = Iarray.map (fun _ -> 0) lo in
  { Nx_kernel.lo; hi; interior; windows = Iarray.of_array windows }

let window ?(step = 1) ?(dilation = 1) axis size = { M.axis; size; step; dilation }

(* A sum of the template [name] over [operands] of float32, each padded
   with one as its padding says, where given. *)
let padded_sum ?(fill = "\000\000\128\063") ?(scan = false) name axes operands =
  let pads = Array.map (fun (_, p) -> Option.map (fun p -> (fill, p)) p) operands in
  let operands = Array.map fst operands in
  fused_of ~scan ~pads name (Monoid Sum) (loop_shape (loads_with pads operands)) axes operands

(* A depthwise convolution's loop: an operand [[2; 5; 6; c]] of [fdt],
   padded by [lo] and [hi] along its spatial axes with [fill], windows of
   [k1] by [k2] taps appending the taps, times a kernel broadcast over the
   batch and the windows, summed over the taps at float32. *)
let depthwise ?(fdt = f32) ?(c = 64) ?(lo = 1) ?(hi = 1) ?fill k1 k2 =
  let (D.Any ft) = fdt in
  let fill = Option.value fill ~default:(String.make (max 1 (D.bits ft / 8)) '\000') in
  let s = [| 2; 5; 6; c |] in
  let x = product_operand fdt s (plain_view 4) ~specials:0 13 in
  let sp v = [| 0; v; v; 0 |] in
  let pad = pad_v (sp lo) (sp hi) ~windows:[| window 1 k1; window 2 k2 |] in
  let loop = Result.get_ok (Nx_kernel.Shape.padded pad s) in
  let axes = [| 4; 5 |] in
  let y = spread_operand fdt loop axes Kernel (plain_view 6) ~specials:0 14 in
  fused_of ~fdt ~second:Kernel ~pads:[| Some (fill, pad); None |]
    "a product of two factors" (Monoid Sum) loop axes [| x; y |]

(* Sums along axis 1 of [160; 2; 64], more outputs than a thread's unit
   of a sum of products, of a kernel by a factor with a NaN at [0; 0; 5]:
   the NaN's output is a NaN while the units after it on the thread read
   the kernel's terms. *)
let nan_before_units =
  let s = [| 160; 2; 64 |] in
  let n = total s in
  let r = Random.State.make [| 17 |] in
  let bits = Bytes.create (4 * n) in
  for i = 0 to n - 1 do
    Bytes.set_int32_le bits (4 * i) (Int32.bits_of_float (full r 23))
  done;
  Bytes.set_int32_le bits (4 * 5) (Int32.bits_of_float Float.nan);
  let b =
    A.Any
      (A.v D.Float32 (L.contiguous (Iarray.of_array s))
         (Rig.Buffer.of_string (Bytes.to_string bits)))
  in
  let a = spread_operand f32 s [| 1 |] Kernel (plain_view 3) ~specials:0 19 in
  fused_of ~second:Kernel "a product of two factors" (Monoid Sum) s [| 1 |] [| a; b |]

(* A constant's sum over a long axis and a broadcast load's own sum, whose
   terms map.c holds once for a whole row; complex products, whose lanes
   and scans start from their first terms: one output's block, rows of
   outputs side by side, two reduced axes split by a kept one, several
   blocks, and scans past a chunk; float32 scans past a chunk: of one
   slice, in quads of chunks, as Moments, and of a row of slices side by
   side; a reduction of several blocks; Logsumexp down columns, several
   outputs' blocks computed at once; outputs narrower than their carrier:
   bfloat16's Arg, Moments scan and scan in quads of chunks, and int4's
   scan so; Logsumexp scans by tiles, of one slice and of three side by
   side, an Arg scan past a chunk, and bfloat16's Arg scan, its extreme
   the program's term; Moments of many outputs of several blocks, each
   staged once for both passes, plain and stepped. *)
let fused_examples =
  let c = D.Any D.Complex64 in
  let stepped = { (plain_view 2) with stepped = true } in
  [
    fused_example ~d:c "a product of two operands" (Monoid Prod) [| 3; 40 |] [| 1 |];
    fused_example ~d:c "a product of two operands" (Monoid Prod) [| 40; 3 |] [| 0 |];
    fused_example ~d:c "a product of two operands" (Monoid Prod) [| 3; 4; 5 |] [| 0; 2 |];
    fused_example ~d:c "a difference with lanes" (Monoid Prod) [| 2; 2000 |] [| 1 |];
    fused_example ~d:c ~scan:true "a product of two operands" (Monoid Prod) [| 9000 |] [| 0 |];
    fused_example ~d:c ~scan:true "a product of two operands" (Monoid Prod) [| 3; 5000 |] [| 1 |];
    fused_example ~d:c ~scan:true "a product of two operands" (Monoid Prod) [| 5000; 3 |] [| 0 |];
    fused_example "a constant alone, of no load" (Monoid Sum) [| 100000 |] [| 0 |];
    fused_example ~view:{ (plain_view 1) with broadcast = true }
      "an operand, beside its negation" (Monoid Sum) [| 100000 |] [| 0 |];
    fused_example ~scan:true "exp of a difference with lanes" (Monoid Sum) [| 9000 |] [| 0 |];
    fused_example ~scan:true "a scaled operand" Moments [| 3; 9000 |] [| 1 |];
    fused_example ~scan:true "a scaled operand" Moments [| 9000 |] [| 0 |];
    fused_example ~scan:true "a log, NaN below 0" (Monoid Max) [| 9000; 3 |] [| 0 |];
    fused_example ~scan:true "a product of two operands" (Monoid Prod) [| 20000 |] [| 0 |];
    fused_example "a product of two operands" (Monoid Sum) [| 5000 |] [| 0 |];
    fused_example "exp of a difference with lanes" Logsumexp [| 300; 20 |] [| 0 |];
    fused_example "an operand cast into bfloat16" (Arg Max) [| 3; 40 |] [| 1 |];
    fused_example ~scan:true "an operand cast into bfloat16" Moments [| 3; 100 |] [| 1 |];
    fused_example ~scan:true "an operand cast into bfloat16" (Monoid Sum) [| 20000 |] [| 0 |];
    fused_example ~scan:true "an operand cast into int4" (Monoid Max) [| 20000 |] [| 0 |];
    fused_example ~scan:true "exp of a difference with lanes" Logsumexp [| 20000 |] [| 0 |];
    fused_example ~scan:true "a scaled operand" Logsumexp [| 9000; 3 |] [| 0 |];
    fused_example ~scan:true "a log, NaN below 0" (Arg Min) [| 20000 |] [| 0 |];
    fused_example ~scan:true "an operand cast into bfloat16" (Arg Max) [| 3; 40 |] [| 1 |];
    fused_example "a scaled operand" Moments [| 64; 1100 |] [| 1 |];
    fused_example ~view:stepped "a scaled operand" Moments [| 64; 1100 |] [| 1 |];
  ]

(* Sums of products: each count of terms from 1 to 17, rows of 64 outputs
   with the second factor a kernel and of 3, factors of each narrow float,
   of float64 at float32 and float32 at float64, a squared factor, two
   reduced axes, each view, a NaN before a thread's units, and depthwise
   convolutions: of float64, bfloat16 and float4_e2m1fn factors, with a
   NaN fill, a -0 fill, padded unevenly and on one side alone. *)
let product_examples =
  let flat ?fdt ?(second = Full) ?view ?(specials = 0) s axes =
    fused_example ?fdt ~second ?view ~specials "a product of two factors" (Monoid Sum) s axes
  in
  let narrow = D.[ Any Bfloat16; Any Float16; Any Float8_e4m3fn; Any Float8_e5m2; Any Float4_e2m1fn; Any Float64 ] in
  List.init 17 (fun i -> flat ~second:Kernel [| 64; i + 1 |] [| 1 |])
  @ List.init 17 (fun i -> flat [| 3; i + 1 |] [| 1 |])
  @ List.map (fun fdt -> flat ~fdt ~specials:3 [| 70; 5 |] [| 1 |]) narrow
  @ List.map
      (fun view -> flat ~view [| 33; 6 |] [| 1 |])
      [ { (plain_view 2) with stepped = true }; { (plain_view 2) with reversed = Some 0 };
        { (plain_view 2) with perm = [| 1; 0 |] }; { (plain_view 2) with broadcast = true } ]
  @ [
      nan_before_units;
      fused_example ~d:f64 ~fdt:f32 "a product of two factors" (Monoid Sum) [| 9; 3 |] [| 1 |];
      fused_example ~d:f64 ~fdt:f32 "a product of two factors" (Monoid Sum) [| 64; 9 |] [| 1 |];
      fused_example "a squared factor" (Monoid Sum) [| 20; 7 |] [| 1 |];
      flat [| 3; 4; 5 |] [| 0; 2 |];
      flat ~specials:3 [| 64; 9 |] [| 1 |];
      depthwise 3 3;
      depthwise ~fdt:f64 3 3;
      depthwise ~fdt:(D.Any D.Bfloat16) 3 3;
      depthwise ~c:5 2 3;
      depthwise ~fill:"\052\018\192\127" 4 4;
      depthwise ~fill:"\000\000\000\128" ~lo:2 ~hi:0 3 3;
      depthwise ~fdt:(D.Any D.Float4_e2m1fn) ~c:5 ~lo:0 ~hi:2 2 2;
      depthwise 4 5;
    ]
  (* Products of -0 by a positive kernel, whose sums from +0 are +0. *)
  @ List.map
      (fun (D.Any dt as d) ->
        let s = [| 64; 9 |] in
        let x = array_bits dt (D.bits dt) s (Array.make (total s) (-0.)) in
        let (A.Any row) =
          array_bits dt (D.bits dt) [| 1; 9 |] (Array.init 9 (fun i -> 0.5 +. Float.of_int i))
        in
        let y = A.Any (Option.get (A.move (M.Broadcast (Iarray.of_array s)) row)) in
        fused_of ~d ~second:Kernel "a product of two factors" (Monoid Sum) s [| 1 |] [| x; y |])
      floats

(* Sums over padded loads: a coordinate read, two loads padded at
   different ends, a windowed load whose taps lie along an axis another
   load fills along, and an empty loop, of a window of 5 over 3 elements,
   whose sum along its taps has no output and along its windows five of no
   term; then one padded load of each regime: cropped, interior, a window
   wider than its axis, a fill along a kept and along a reduced axis, a
   padding and a NaN or -0 fill through views, an int4 and a bit operand,
   and a scan over a padded load. *)
let padded_examples =
  let operand ?(d = f32) s seed = product_operand d s (plain_view (Array.length s)) ~specials:20 seed in
  let one ?fill ?scan ?(d = f32) name axes s pad =
    padded_sum ?fill ?scan name axes [| (operand ~d s 7, Some pad) |]
  in
  [
    padded_sum "an operand plus its last coordinate" [| 0 |]
      [| (operand [| 5; 2 |] 2, Some (pad_v [| 1; 0 |] [| 1; 0 |])) |];
    padded_sum "a product of two factors" [| 1; 2 |]
      [| (operand [| 5; 3 |] 5, Some (pad_v [| 1; 0 |] [| 0; 0 |] ~windows:[| window 0 2 |]));
         (operand [| 3; 3 |] 6, Some (pad_v [| 0; 0 |] [| 3; 0 |] ~windows:[| window 0 2 |])) |];
    padded_sum "a product of two factors" [| 0 |]
      [| (operand [| 5 |] 3, Some (pad_v [| 0 |] [| 0 |] ~windows:[| window 0 3 |]));
         (operand [| 3; 2 |] 4, Some (pad_v [| 0; 0 |] [| 0; 1 |])) |];
    one "an operand, beside its negation" [| 1 |] [| 3 |] (pad_v [| 0 |] [| 0 |] ~windows:[| window 0 5 |]);
    one "an operand, beside its negation" [| 0 |] [| 3 |] (pad_v [| 0 |] [| 0 |] ~windows:[| window 0 5 |]);
    one "a scaled operand" [| 1 |] [| 4 |]
      { (pad_v [| -1 |] [| 2 |] ~windows:[| window 0 2 |]) with interior = [| 1 |] };
    one "a scaled operand" [| 1 |] [| 3 |] (pad_v [| 0 |] [| 0 |] ~windows:[| window 0 4 ~step:2 |]);
    one "a scaled operand" [| 0 |] [| 4 |] (pad_v [| 1 |] [| 1 |]);
    one "a scaled operand" [| 2 |] [| 5; 6 |] (pad_v [| 1; -1 |] [| 0; 2 |] ~windows:[| window 1 3 ~step:2 |]);
    one "a scaled operand" [| 1 |] [| 4 |] (pad_v [| 3 |] [| 0 |] ~windows:[| window 0 1 ~step:2 |]);
    one "a scaled operand" [| 1 |] [| 2 |] (pad_v [| 1 |] [| 1 |] ~windows:[| window 0 3 |]);
    one ~fill:"\052\018\192\127" "a scaled operand" [||] [| 3; 5 |] (pad_v [| 1; 0 |] [| 1; 0 |]);
    one ~fill:"\000\000\000\128" "a scaled operand" [| 1 |] [| 6 |] (pad_v [| 2 |] [| 1 |] ~windows:[| window 0 2 |]);
    one ~d:(D.Any D.Int4) ~fill:"\136" "an int4 operand cast up" [| 2 |] [| 3; 5 |]
      (pad_v [| 0; 2 |] [| 0; 1 |] ~windows:[| window 1 2 |]);
    one ~d:(D.Any D.Bit) ~fill:"\001" "a bit operand cast up" [| 0 |] [| 9 |] (pad_v [| 3 |] [| 2 |]);
    one ~scan:true "a scaled operand" [| 1 |] [| 4; 5 |] (pad_v [| 1; 0 |] [| 2; 0 |]);
    padded_sum "a scaled operand" [| 1 |]
      [| (product_operand f32 [| 4; 5 |] { (plain_view 2) with reversed = Some 1 } ~specials:20 7,
          Some (pad_v [| 1; 0 |] [| 2; 0 |])) |];
    padded_sum "a scaled operand" [| 1 |]
      [| (product_operand f32 [| 4; 5 |] { (plain_view 2) with stepped = true } ~specials:20 7,
          Some (pad_v [| 1; 0 |] [| 2; 0 |])) |];
  ]

(* A case of [fused_any], or a float32 program's monoid along
   [wide_rows] often enough that every run reaches their calls. *)
let fused_case =
  let open Gen in
  let rows =
    let* name, ins, _, nodes, outs, k =
      of_list
        ~pp:(fun ppf (n, _, _, _, _, _) -> Format.pp_print_string ppf n)
        (templates f32 f32 2)
    in
    let prog =
      P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
        ~outs:(Iarray.of_array outs)
    in
    let monoids =
      List.filter
        (function Nx_kernel.Monoid _ -> true | _ -> false)
        (reductions_of ~scan:false (P.dtype prog outs.(k)))
    in
    let* red = of_list monoids in
    let+ s = wide_rows in
    fused_example name red s [| 1 |]
  in
  with_pp pp_fused (frequency [ (6, fused_any); (1, rows) ])

(* The destinations of [c]'s reduction, fresh, of output [c.k]'s dtype. *)
let fused_dsts (b : Support.backend) c =
  let s = c.shape in
  let y =
    if c.scan then Iarray.of_array s
    else
      Iarray.of_list
        (List.filteri (fun i _ -> not (Array.mem i c.axes)) (Array.to_list s))
  in
  let (D.Any dt) = P.dtype c.prog (Iarray.get c.prog.outs c.k) in
  let make () = A.Any (A.create b.device dt y) in
  match c.red with
  | Monoid _ | Logsumexp -> [| make () |]
  | Moments -> [| make (); make () |]
  | Arg _ -> [| make (); A.Any (A.create b.device D.Int64 y) |]

(* Whether Nx_kernel.Shape fits [c]: an extreme of no term is refused. *)
let fused_fits c =
  Result.is_ok
    (if c.scan then Nx_kernel.Shape.scan ~axis:c.axes.(0) c.shape
     else Nx_kernel.Shape.reduce ~axes:(Iarray.of_array c.axes) [| c.red |] c.shape)

(* [c] by [M] into [dsts] over [loads]. *)
let fused_compute (module M : Nx_kernel.S) c ~dsts loads =
  if c.scan then M.scan c.prog ~axis:c.axes.(0) (c.red, c.k) ~dsts loads
  else
    M.reduce c.prog ~shape:c.shape ~axes:(Iarray.of_array c.axes) [| (c.red, c.k) |]
      ~dsts loads

(* [c] fused, its loads guarded where it says, against [c]'s program mapped
   over the arrays its loads read into fresh arrays and reduced as their
   outputs' identity, a NaN as any NaN. *)
let law_fused (b : Support.backend) c =
  let module K = (val b.kernels) in
  let s = c.shape in
  let terms = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  classify c.name true;
  cover "a loop of no load" (c.operands = [||]);
  let (D.Any odt as o) = P.dtype c.prog (Iarray.get c.prog.outs c.k) in
  let complex = D.is D.Complex odt in
  cover "a complex output" complex;
  cover "a complex product" (complex && c.red = Monoid Prod);
  cover "a narrow output" (staged o);
  let fdt = if c.operands = [||] then o else dtype_of c.operands.(0) in
  cover "narrow factors" (staged fdt);
  cover "float32 factors at float64" (same fdt f32 && same o f64);
  cover "float64 factors at float32" (same fdt f64 && same o f32);
  cover (reduction_name c.red) true;
  cover "a scan" c.scan;
  cover "a Moments scan" (c.scan && c.red = Moments);
  cover "a stateful scan"
    (c.scan && (c.red = Logsumexp || c.red = Arg Max || c.red = Arg Min));
  cover "a scan past a chunk" (c.scan && s.(c.axes.(0)) > 4096);
  cover "a row of lanes" (c.second = Lanes);
  cover "a kernel" (c.second = Kernel);
  cover "a lane or fewer terms an output" (terms <= 16);
  cover "17 terms an output" (terms = 17);
  cover "several blocks of terms" (terms > 1024);
  let padded = Array.exists Option.is_some c.pads in
  cover "a padded load" padded;
  cover "a padded load of few products" (padded && terms <= 16 && Array.length c.operands = 2);
  cover "a padded float4_e2m1fn load" (padded && same fdt (D.Any D.Float4_e2m1fn));
  let (D.Any fd) = fdt in
  cover "a padded sub-byte load" (padded && D.bits fd < 8);
  (match if c.pads = [||] then None else c.pads.(0) with
  | Some (fill, pad) ->
      cover "windows" (pad.windows <> [||]);
      cover "cropped" (Iarray.exists (fun l -> l < 0) pad.lo);
      cover "interior" (Iarray.exists (fun i -> i > 0) pad.interior);
      cover "a NaN fill" (Elements.nan_of fdt fill);
      cover "a -0 fill"
        (is_float fdt && String.length fill >= 4
        && String.for_all (fun ch -> ch = '\000') (String.sub fill 0 (String.length fill - 1))
        && fill.[String.length fill - 1] = '\128');
      cover "a scan over a padded load" c.scan
  | None -> ());
  let r = Array.length s in
  let side = r > 1 && not (Array.mem (r - 1) c.axes) in
  let split =
    Array.exists (fun a -> a + 1 < r && not (Array.mem (a + 1) c.axes)
                           && Array.exists (fun b -> b > a + 1) c.axes) c.axes
  in
  cover "outputs side by side" side;
  cover "reduced axes split by a kept one" split;
  cover "reduced axes split, outputs side by side, more than a lane of terms"
    (split && side && terms > 16);
  cover "guarded" (c.guard <> None);
  cover "guarded, padded" (c.guard <> None && padded);
  cover "rank 0" (r = 0);
  cover "no element" (total s = 0);
  let plain = Array.map (fun (A.Any x) -> Nx_kernel.Plain x) in
  let outs =
    Array.map
      (fun o ->
        let (D.Any dt) = P.dtype c.prog o in
        A.Any (A.create b.device dt (Iarray.of_array s)))
      (Iarray.to_array c.prog.outs)
  in
  (match K.map c.prog ~dsts:outs (plain (reads_of c)) with
  | A.Done -> ()
  | r -> failf "the map answered %a" Nx_array_support.pp_answer r);
  let (A.Any y) = outs.(c.k) in
  let want = fused_dsts b c in
  (* An extreme of no term refuses alike. *)
  let expected =
    fused_compute b.kernels { c with prog = identity (D.Any (A.dtype y)); k = 0 }
      ~dsts:want [| Nx_kernel.Plain y |]
  in
  let guarded =
    Array.map
      (fun (A.Any x as a) ->
        match c.guard with
        | None -> (a, Fun.id)
        | Some side ->
            let g, free = Support.guarded side (A.dtype x) (A.layout x) in
            (* Each element once: one index along an axis stepped 0. *)
            let once y =
              let l = A.layout y in
              let st = L.strides l in
              Option.get
                (A.move
                   (M.Slice
                      (Iarray.mapi
                         (fun i e ->
                           { M.start = 0; count = (if Iarray.get st i = 0 then min e 1 else e);
                             step = 1 })
                         (L.shape l)))
                   y)
            in
            A.transfer ~src:(once x) ~dst:(once g);
            (A.Any g, free))
      c.operands
  in
  let got = fused_dsts b c in
  Fun.protect
    ~finally:(fun () -> Array.iter (fun (_, free) -> free ()) guarded)
    (fun () ->
      let answer =
        fused_compute b.kernels c ~dsts:got (loads_with c.pads (Array.map fst guarded))
      in
      equal ~msg:"answer" Elements.answer expected answer;
      if answer = A.Done then
        Array.iteri
          (fun i w ->
            equal ~msg:(Printf.sprintf "destination %d" i)
              (array (Elements.same (dtype_of w)))
              (Elements.elements (host w))
              (Elements.elements (host got.(i))))
          want)

(* A program computed into its own load gives the bits of one computed
   into fresh destinations, as nx_cpu.mli states, a NaN as any NaN: a
   float16 negation of signalling NaNs summed along no axis; scanned;
   summed beside its maximum, each written after the other reads; and its
   Arg, reduced along no axis and scanned over numbers, each extreme
   computed from the load at its position after earlier results are
   written. *)
let test_program_in_place (b : Support.backend) () =
  let module K = (val b.kernels) in
  let h = D.Any D.Float16 in
  let p =
    P.v ~ins:(Iarray.of_array [| h |])
      (Iarray.of_array P.[| In 0; Op1 (Unary Neg, h, 0) |])
      ~outs:(Iarray.of_array [| 1 |])
  in
  let nans = [| "\001\124"; "\000\060"; "\001\252" |] in
  (* 3, 1 and 2: their negations' running maximum is at 0, 1, then 1. *)
  let numbers = [| "\000\066"; "\000\060"; "\000\064" |] in
  let check what codes dts call =
    let load () = Elements.of_elements h [| 3 |] codes in
    let fresh j =
      let (D.Any dt) = dts.(j) in
      A.Any (A.create Rig.host dt [| 3 |])
    in
    let want = Array.init (Array.length dts) fresh in
    equal ~msg:(what ^ ", fresh") Elements.answer A.Done
      (call (load ()) want);
    let x = load () in
    let got = Array.init (Array.length dts) (fun j -> if j = 0 then x else fresh j) in
    equal ~msg:(what ^ ", into its load") Elements.answer A.Done (call x got);
    Array.iteri
      (fun j w ->
        equal ~msg:what (array (Elements.same dts.(j)))
          (Elements.elements w) (Elements.elements got.(j)))
      want
  in
  let reduce rs x dsts =
    K.reduce p ~shape:[| 3 |] ~axes:[||] rs ~dsts [| plain_of x |]
  in
  let scan r x dsts = K.scan p ~axis:0 (r, 0) ~dsts [| plain_of x |] in
  let i64 = D.Any D.Int64 in
  check "a sum along no axis" nans [| h |] (reduce [| (Monoid Sum, 0) |]);
  check "a running sum" nans [| h |] (scan (Monoid Sum));
  check "a sum and a maximum" nans [| h; h |]
    (reduce [| (Monoid Sum, 0); (Monoid Max, 0) |]);
  check "an Arg along no axis" numbers [| h; i64 |]
    (reduce [| (Arg Max, 0) |]);
  check "an Arg scan" numbers [| h; i64 |] (scan (Arg Max));
  (* A padded load, read from a copy: windows of 3 over the load padded by
     one, whose sums along the taps have the load's shape. *)
  let pad = pad_v [| 1 |] [| 1 |] ~windows:[| window 0 3 |] in
  let one = "\000\060" in
  check "a sum of a padded load" numbers [| h |] (fun x dsts ->
      K.reduce p ~shape:[| 3; 3 |] ~axes:[| 1 |] [| (Monoid Sum, 0) |] ~dsts
        [| padded_load x one pad |])

(* Several reductions of one loop: [c]'s and [red] of output [k], with
   [c]'s axes, one call. *)
type several = { first : fused; red2 : Nx_kernel.reduction; k2 : int }

let pp_several ppf s =
  Format.fprintf ppf "%a, then %s of output %d" pp_fused s.first
    (reduction_name s.red2) s.k2

let several_case =
  let open Gen in
  with_pp pp_several
  @@ let* c = fused_case in
     let c = { c with scan = false; guard = None } in
     let outs = Iarray.length c.prog.outs in
     (* The other output where there are two, mostly. *)
     let* k2 =
       if outs = 1 then constant 0
       else frequency [ (3, constant (1 - c.k)); (1, constant c.k) ]
     in
     let o = P.dtype c.prog (Iarray.get c.prog.outs k2) in
     let+ red2 = of_list (reductions_of ~scan:false o) in
     { first = c; red2; k2 }

(* The destinations of [red] of output [k] of [c], fresh. *)
let dsts_of (b : Support.backend) c red k =
  fused_dsts b { c with red; k }

(* Several reductions in one call give each one's bits alone, a NaN as any
   NaN. *)
let law_several (b : Support.backend) s =
  let module K = (val b.kernels) in
  let c = s.first in
  cover "two outputs" (s.k2 <> c.k);
  cover "a state and a monoid"
    ((match c.red with Monoid _ -> false | _ -> true)
    <> (match s.red2 with Monoid _ -> false | _ -> true));
  let loads = loads_of c in
  let alone red k =
    let d = dsts_of b c red k in
    let axes = Iarray.of_array c.axes in
    (K.reduce c.prog ~shape:c.shape ~axes [| (red, k) |] ~dsts:d loads, d)
  in
  let a1, w1 = alone c.red c.k and a2, w2 = alone s.red2 s.k2 in
  let d1 = dsts_of b c c.red c.k and d2 = dsts_of b c s.red2 s.k2 in
  let answer =
    K.reduce c.prog ~shape:c.shape ~axes:(Iarray.of_array c.axes)
      [| (c.red, c.k); (s.red2, s.k2) |]
      ~dsts:(Array.append d1 d2) loads
  in
  (* The pair refuses as the first of them that refuses alone. *)
  let expected = if a1 <> A.Done then a1 else a2 in
  equal ~msg:"answer" Elements.answer expected answer;
  if answer = A.Done then begin
    let same w g =
      Array.iteri
        (fun i x ->
          equal (array (Elements.same (dtype_of x)))
            (Elements.elements (host x)) (Elements.elements (host g.(i))))
        w
    in
    same w1 d1;
    same w2 d2
  end

(* The suite *)

(* A plain sum and scan whose program is Prog.of_node's [In 0], which keeps
   its operand as two nodes, compute as [In 0]'s. *)
let test_of_node (b : Support.backend) () =
  let x = A.Any (A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]) in
  let dt = D.Any D.Float32 in
  let of_node = P.of_node ~ins:[| dt |] (P.In 0) in
  let run scan prog shape =
    let dst = A.Any (A.create b.device D.Float32 shape) in
    let answer = compute b.kernels ~scan prog Sum [| 1 |] dst [| plain_of (on b x) |] in
    equal ~msg:"its answer" bool true (answer = A.Done);
    let (A.Any h) = host dst in
    Array.map Int32.bits_of_float (A.to_array (A.expect D.Float32 (A.Any h)))
  in
  equal (array int32)
    (run false (identity dt) [| 2 |])
    (run false of_node [| 2 |]);
  equal (array int32)
    (run true (identity dt) [| 2; 3 |])
    (run true of_node [| 2; 3 |])

(* Bools stored as bytes other than 0 and 1, through a Uint8 bitcast: a
   bool is true where its byte is not 0 (dtype.mli), so a Max or a Min of
   them, or an Arg's extreme, stores 0 or 1, and nx.cpu computes each. Rows
   of [0; 2], [255; 0], [3; 7] and [0; 0]. *)
let test_bool_bytes (b : Support.backend) () =
  let bytes = [| 0; 2; 255; 0; 3; 7; 0; 0 |] in
  let x =
    A.Any
      (Option.get (A.bitcast D.Bool (A.of_array D.Uint8 [| 4; 2 |] bytes)))
  in
  let p = identity (D.Any D.Bool) in
  let answered what = function
    | A.Done -> ()
    | a -> failf "%s: %s answered %a" what b.name Nx_array_support.pp_answer a
  in
  let bits d =
    let (A.Any h) = host d in
    A.to_array (Option.get (A.bitcast D.Uint8 (A.expect D.Bool (A.Any h))))
  in
  let check what scan m shape want =
    let dst = A.create b.device D.Bool shape in
    let answer =
      compute b.kernels ~scan p m [| 1 |] (A.Any dst) [| plain_of (on b x) |]
    in
    answered what answer;
    equal ~msg:what (array int) want (bits (A.Any dst))
  in
  check "max" false Max [| 4 |] [| 1; 1; 1; 0 |];
  check "min" false Min [| 4 |] [| 0; 0; 1; 0 |];
  check "max scan" true Max [| 4; 2 |] [| 0; 1; 1; 1; 1; 1; 0; 0 |];
  let arg what e want =
    let v = A.create b.device D.Bool [| 4 |]
    and i = A.create b.device D.Int64 [| 4 |] in
    let module K = (val b.kernels) in
    let answer =
      K.reduce p ~shape:[| 4; 2 |] ~axes:[| 1 |] [| (Arg e, 0) |]
        ~dsts:[| A.Any v; A.Any i |]
        [| plain_of (on b x) |]
    in
    answered what answer;
    equal ~msg:what (pair (array int) (array int64)) want
      (bits (A.Any v), A.to_array (A.expect D.Int64 (host (A.Any i))))
  in
  arg "arg max: the first true" Max ([| 1; 1; 1; 0 |], [| 1L; 0L; 0L; 0L |]);
  arg "arg min: the first false" Min ([| 0; 0; 1; 0 |], [| 0L; 1L; 0L; 0L |])

(* Arg, Logsumexp and Moments of narrow dtypes and booleans, and into
   destinations of other dtypes *)

(* The dtype a reference reads terms of [d] through: their carrier, a byte
   integer for booleans, which keeps their order and tells their bits
   apart. *)
let arg_carrier = function
  | D.Any (D.Bool | D.Bit) -> D.Any D.Uint8
  | d -> Nx_kernel.carrier d

(* A stateful reduction, a Moments scan where [scan], and its destinations'
   dtype, an Arg's positions aside. *)
type stat_into = { st : stat; scan : bool; into : D.any }

let pp_stat_into ppf { st; scan; into } =
  Format.fprintf ppf "%s%a into %a"
    (if scan then "scan " else "")
    pp_stat st pp_dtype into

let narrow_floats =
  D.
    [
      Any Float16; Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2;
      Any Float4_e2m1fn;
    ]

(* An Arg of a narrow dtype or a boolean, or a Logsumexp or Moments of a
   float, reduced or scanned, into its dtype or another float's. Narrow
   floats are float32's values of [value_kinds] rounded into them. *)
let stat_into_case =
  Gen.with_pp pp_stat_into
    (let open Gen in
     let* r =
       of_list Nx_kernel.[ Arg Max; Arg Min; Logsumexp; Moments ]
     in
     let arg = match r with Nx_kernel.Arg _ -> true | _ -> false in
     let* (D.Any dt as d) =
       of_list ~pp:pp_dtype
         (if arg then D.Any D.Bool :: narrow_dtypes
          else narrow_floats @ narrow_floats @ floats)
     in
     let* scan = bool in
     let* s =
       frequency
         [
           (3, shape_gen);
           (1, map (fun n -> [| n |]) (int_range 1025 5000));
           (1, many_long);
         ]
     in
     let s = if scan && Array.length s = 0 then [| 3 |] else s in
     let rank = Array.length s in
     let* axes =
       if scan then map (fun a -> [| a |]) (int_range 0 (rank - 1))
       else
         let+ keep = array ~size:(constant rank) bool in
         Array.of_list
           (List.filter (fun i -> keep.(i)) (List.init rank Fun.id))
     in
     let* v = view_of rank in
     let* values = of_list value_kinds in
     let* specials = frequency [ (1, constant 0); (1, constant 20) ] in
     let* into =
       frequency
         [
           (2, constant d);
           ( 1,
             of_list ~pp:pp_dtype D.[ Any Float32; Any Float64; Any Float16 ] );
         ]
     in
     let+ seed = int in
     let make n =
       if not (is_float d) then A.Any (elements dt n ~specials seed)
       else if D.bits dt >= 32 then
         float_elements dt n values ~outliers:0 ~specials seed
       else
         cast_into d
           (float_elements D.Float32 n values ~outliers:0 ~specials seed)
     in
     { st = stat_of r s axes v values.name make; scan; into })

(* A boolean's Arg, which the generator draws once in 18 cases, and the
   Moments of many bfloat16 outputs of several blocks, stepped. *)
let stat_into_examples =
  let v = { perm = [| 0 |]; stepped = false; reversed = None; broadcast = false } in
  let make n = A.Any (elements D.Bool n ~specials:0 1) in
  let offset = List.find (fun (k : values) -> k.name = "offset") value_kinds in
  let narrow n =
    cast_into (D.Any D.Bfloat16)
      (float_elements D.Float32 n offset ~outliers:0 ~specials:2000 3)
  in
  let stepped = { perm = [| 0; 1 |]; stepped = true; reversed = None; broadcast = false } in
  [
    {
      st = stat_of (Arg Max) [| 9 |] [| 0 |] v "small" make;
      scan = false;
      into = D.Any D.Bool;
    };
    {
      st = stat_of Moments [| 64; 1025 |] [| 1 |] stepped "offset" narrow;
      scan = false;
      into = D.Any D.Bfloat16;
    };
  ]

(* Two NaNs of a narrow float among zeros, one signalling, past a block
   and a chunk: Logsumexp, Moments and Arg, reduced and scanned, give a
   NaN from the first NaN on, as their carrier does. *)
let nan_code_stat_examples =
  List.concat_map
    (fun (d, first, other) ->
      let x = nan_vector d first other (5000, 4500, 4700) in
      List.map
        (fun (r, scan) ->
          { st = { r; axes = [| 0 |]; x; views = []; values = "nan" }; scan; into = d })
        Nx_kernel.
          [ (Logsumexp, false); (Logsumexp, true); (Moments, false); (Moments, true);
            (Arg Max, false); (Arg Min, true) ])
    nan_codes

(* [c]'s reduction, or its scan where [scan], of [x] by [k], into fresh
   destinations of [d], an Arg's positions of int64. *)
let stat_answer k ~scan (c : stat) (D.Any d) x =
  let module K = (val k : Nx_kernel.S) in
  let s = if scan then shape_of x else stat_shape c in
  let make : type a b. (a, b) D.t -> A.any =
   fun dt -> A.Any (A.create Rig.host dt (Iarray.of_array s))
  in
  let dsts =
    match c.r with
    | Arg _ -> [| make d; make D.Int64 |]
    | Moments -> [| make d; make d |]
    | Monoid _ | Logsumexp -> [| make d |]
  in
  let p = identity (dtype_of x) and loads = [| plain_of x |] in
  let answer =
    if scan then K.scan p ~axis:c.axes.(0) (c.r, 0) ~dsts loads
    else
      K.reduce p ~shape:(shape_of x) ~axes:(Iarray.of_array c.axes)
        [| (c.r, 0) |] ~dsts loads
  in
  (answer, dsts)

let stat_run k ~scan c d x =
  match stat_answer k ~scan c d x with
  | A.Done, dsts -> dsts
  | a, _ -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* A narrow operand's terms reduce in their carrier, and each result rounds
   once into its destination: the bits of the reduction of the terms cast
   into [arg_carrier], cast into the destination, a NaN as any NaN. *)
let law_stat_into (b : Support.backend) { st = c; scan; into } =
  let x = dtype_of c.x in
  let s = shape_of c.x in
  cover "narrow" (staged x);
  cover "bool" (same x (D.Any D.Bool));
  cover "into another dtype" (not (same into x));
  cover "a scan" scan;
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  cover "several blocks" (per > 1024);
  cover "many outputs of several blocks"
    ((not scan) && per > 1024 && total (stat_shape c) >= 64);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  let wide = arg_carrier x in
  let got = stat_run b.kernels ~scan c into c.x in
  let want = stat_run b.kernels ~scan c wide (cast_into wide c.x) in
  (* Into the operand's own dtype, a staged Arg's extreme is the term at
     its position, as the operand holds it. *)
  let own = staged x && same into x in
  let case = { (as_case c) with scan } in
  (* Each result's terms, a scan's its slice's. *)
  let terms () =
    results case (Elements.elements c.x) ~fill:[||] ~reduce:Fun.id
      ~scan:(fun ts -> Array.map (fun _ -> ts) ts)
  in
  Array.iteri
    (fun k w ->
      let want =
        match (c.r, k) with
        | Arg _, 1 -> Elements.elements w
        | Arg _, _ when own ->
            let at = A.to_array (A.expect D.Int64 want.(1)) in
            Array.mapi (fun o ts -> ts.(Int64.to_int at.(o))) (terms ())
        | _ -> Elements.elements (cast_into into w)
      in
      equal
        ~msg:(Printf.sprintf "destination %d" k)
        (array (Elements.same (dtype_of got.(k)))) want
        (Elements.elements got.(k)))
    want

(* A reduction and a scan of a program of [nl] loads, for [nl] one below a
   block's operands to thousands past them: the sum of the loads, [x], a
   transposed [y] and a broadcast [z] in turn, summed along the last axis
   and run along it; a sum of the products of load 0 and the last; and a
   reduction of the last of twenty outputs. Every value is an integer a
   float32 holds, so each result is exact. *)
let test_wide_loads (b : Support.backend) () =
  let module K = (val b.kernels) in
  let f32 = D.Any D.Float32 and shape = [| 4; 3 |] in
  let x = A.of_array D.Float32 [| 4; 3 |] (Array.init 12 Float.of_int) in
  let y =
    Option.get
      (A.move (A.Move.Permute [| 1; 0 |])
         (A.of_array D.Float32 [| 3; 4 |]
            (Array.init 12 (fun q -> Float.of_int (100 + q)))))
  in
  let z =
    Option.get
      (A.move (A.Move.Broadcast [| 4; 3 |])
         (A.of_array D.Float32 [| 3 |] [| 10.; 20.; 30. |]))
  in
  let arrays = [| x; y; z |] in
  let at k i j = A.get arrays.(k mod 3) [| i; j |] in
  let wide nl =
    let loads = Array.init nl (fun k -> Nx_kernel.Plain arrays.(k mod 3)) in
    let ins = Iarray.init nl (fun _ -> f32) in
    let total ~outs =
      let nodes =
        List.init nl (fun k -> P.In k)
        @ List.init (nl - 1) (fun i ->
              P.Op2 (Binary Add, (if i = 0 then 0 else nl + i - 1), i + 1))
      in
      let last = List.length nodes - 1 in
      P.v ~ins (Iarray.of_list nodes) ~outs:(Iarray.init outs (fun _ -> last))
    in
    let product =
      P.v ~ins
        (Iarray.init (nl + 1) (fun k ->
             if k < nl then P.In k else P.Op2 (Binary Mul, 0, nl - 1)))
        ~outs:[| nl |]
    in
    let element i j =
      let s = ref 0. in
      for k = 0 to nl - 1 do
        s := !s +. at k i j
      done;
      !s
    in
    let msg = Printf.sprintf "%d loads" nl in
    let rows f = Array.init 4 (fun i -> f i) in
    let sum_of f i = f i 0 +. f i 1 +. f i 2 in
    let reduced p k want =
      let d = A.create Rig.host D.Float32 [| 4 |] in
      equal ~msg Elements.answer A.Done
        (K.reduce p ~shape ~axes:[| 1 |] [| (Monoid Sum, k) |]
           ~dsts:[| A.Any d |] loads);
      equal ~msg (array float_exact) want (A.to_array d)
    in
    reduced (total ~outs:1) 0 (rows (sum_of element));
    reduced (total ~outs:20) 19 (rows (sum_of element));
    reduced product 0 (rows (sum_of (fun i j -> at 0 i j *. at (nl - 1) i j)));
    let running = A.create Rig.host D.Float32 [| 4; 3 |] in
    equal ~msg Elements.answer A.Done
      (K.scan (total ~outs:1) ~axis:1 (Monoid Sum, 0)
         ~dsts:[| A.Any running |] loads);
    equal ~msg (array float_exact)
      (Array.init 12 (fun q ->
           let i = q / 3 and c = q mod 3 in
           let s = ref 0. in
           for j = 0 to c do
             s := !s +. element i j
           done;
           !s))
      (A.to_array running)
  in
  List.iter wide [ A.max_operands - 3; 16; 17; 18; 19; 32; 100; 4000 ]

(* Nine Moments of one loop take 18 destinations, twenty Args 40, and many
   sums as many tables of reductions, more than one loop holds beside its
   load: each result has the bits of a call of few reductions. *)
let test_many_reductions (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p = identity (D.Any D.Float32) and shape = [| 4; 3 |] in
  let x =
    A.of_array D.Float32 (Iarray.of_array shape)
      [| 1.; 5.; 2.; 0.5; 7.; 3.; 9.; 4.; 4.; 2.; 8.; 6. |]
  in
  let width : Nx_kernel.reduction -> int = function
    | Monoid _ | Logsumexp -> 1
    | Arg _ | Moments -> 2
  in
  (* An Arg's second destination is its int64 positions. *)
  let dst r i =
    match r with
    | Nx_kernel.Arg _ when i mod 2 = 1 -> A.Any (A.create Rig.host D.Int64 [| 4 |])
    | _ -> A.Any (A.create Rig.host D.Float32 [| 4 |])
  in
  let bits (A.Any d) =
    match A.dtype d with
    | D.Int64 -> A.to_array (A.expect D.Int64 (A.Any d))
    | _ ->
        Array.map
          (fun f -> Int64.of_int32 (Int32.bits_of_float f))
          (A.to_array (A.expect D.Float32 (A.Any d)))
  in
  let reduce r n =
    let dsts = Array.init (n * width r) (dst r) in
    equal
      ~msg:(Printf.sprintf "%d reductions" n)
      Elements.answer A.Done
      (K.reduce p ~shape ~axes:[| 1 |] (Array.make n (r, 0)) ~dsts
         [| plain_of (A.Any x) |]);
    Array.map bits dsts
  in
  let agree r few many =
    let want = reduce r few and got = reduce r many in
    Array.iteri
      (fun i g ->
        equal
          ~msg:(Printf.sprintf "destination %d" i)
          (array int64)
          want.(i mod Array.length want)
          g)
      got
  in
  agree Nx_kernel.Moments 1 9;
  agree (Nx_kernel.Arg Max) 1 20;
  (* Sums of 18, 19, 36 and 37 operands, a run of the door's reads and its
     neighbours, and of 64 and 65 items with the program's node, past
     which nx.cpu checks and prices them with the lock released. *)
  List.iter (agree (Nx_kernel.Monoid Sum) 1) [ 17; 18; 35; 36; 63; 64 ]

(* Nine Moments over no axis, the mean each element and the variance 0, take
   every destination and the load in one claim: a broadcast destination or two
   identical ones are refused before any is written, and a destination
   identical to the load reads it as it was. *)
let test_many_claims (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p = identity (D.Any D.Float32) in
  let x = A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let fresh () = A.of_array D.Float32 [| 4 |] [| 7.; 7.; 7.; 7. |] in
  let dsts () = Array.init 18 (fun _ -> A.Any (fresh ())) in
  let call dsts =
    K.reduce p ~shape:[| 4 |] ~axes:[||]
      (Array.make 9 (Nx_kernel.Moments, 0))
      ~dsts [| plain_of (A.Any x) |]
  in
  let values (A.Any d) = A.to_array (A.expect D.Float32 (A.Any d)) in
  let untouched msg ds =
    equal ~msg (array float_exact) [| 7.; 7.; 7.; 7. |] (values ds.(0))
  in
  let ds = dsts () in
  let one = A.of_array D.Float32 [| 1 |] [| 7. |] in
  ds.(17) <- A.Any (Option.get (A.move (Nx_array.Move.Broadcast [| 4 |]) one));
  equal ~msg:"a broadcast last destination" Elements.answer A.Repeated_elements
    (call ds);
  untouched "first destination after the broadcast" ds;
  let ds = dsts () in
  ds.(17) <- ds.(0);
  equal ~msg:"the first destination as the last" Elements.answer A.Overlapping
    (call ds);
  untouched "first destination after the overlap" ds;
  let ds = dsts () in
  let x' = A.copy x in
  ds.(0) <- A.Any x';
  equal ~msg:"the first destination as the load" Elements.answer A.Done
    (K.reduce p ~shape:[| 4 |] ~axes:[||]
       (Array.make 9 (Nx_kernel.Moments, 0))
       ~dsts:ds [| plain_of (A.Any x') |]);
  Array.iteri
    (fun i d ->
      equal ~msg:(Printf.sprintf "destination %d" i) (array float_exact)
        (if i mod 2 = 0 then [| 1.; 2.; 3.; 4. |] else [| 0.; 0.; 0.; 0. |])
        (values d))
    ds

(* One thread and every layout *)

(* A reduction or scan: of a monoid; of Arg, Logsumexp or Moments, reduced
   or with [true] scanned; or of a program's output. *)
type job = Fold of case | State of bool * stat | Program of fused

let pp_job ppf = function
  | Fold c -> pp_case ppf c
  | State (scan, c) ->
      Format.fprintf ppf "%s%a" (if scan then "scan " else "") pp_stat c
  | Program c -> pp_fused ppf c

let views_of = function
  | Fold c -> c.views
  | State (_, c) -> c.views
  | Program c -> c.fviews

(* [j] by [k] into fresh destinations: each one's elements, a NaN as any
   NaN; [None] where [k] declines. *)
let run_job (b : Support.backend) k j =
  let blind d =
    Array.map
      (fun e -> if Elements.nan_of (dtype_of d) e then "nan" else e)
      (Elements.elements (host d))
  in
  let answer, dsts =
    match j with
    | Fold c ->
        let (A.Any x) = c.x in
        let dst =
          A.Any (A.create Rig.host (A.dtype x) (Iarray.of_array (result_shape c)))
        in
        ( compute k ~scan:c.scan (identity (dtype_of c.x)) c.monoid c.axes dst
            [| plain_of c.x |],
          [| dst |] )
    | State (scan, c) -> stat_answer k ~scan c (dtype_of c.x) c.x
    | Program c ->
        let dsts = fused_dsts b c in
        (fused_compute k c ~dsts (loads_of c), dsts)
  in
  match answer with
  | A.Done -> Some (Array.map blind dsts)
  | A.Declined -> None
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

(* [j] over C-contiguous copies of its operands. *)
let copied = function
  | Fold c ->
      let (A.Any x) = c.x in
      Fold { c with x = A.Any (A.copy x); views = [] }
  | State (scan, c) ->
      let (A.Any x) = c.x in
      State (scan, { c with x = A.Any (A.copy x); views = [] })
  | Program c ->
      Program
        {
          c with
          operands = Array.map (fun (A.Any x) -> A.Any (A.copy x)) c.operands;
          fviews = [];
        }

let covers_job = function
  | Fold c ->
      cover "a monoid" true;
      cover "a monoid scanned" c.scan;
      cover "narrow" (staged (dtype_of c.x));
      cover "many outputs of several blocks"
        (let s = shape_of c.x in
         let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
         per > 1024 && total (result_shape c) >= 64)
  | State (scan, c) ->
      cover (reduction_name c.r) true;
      cover "a state scanned" scan
  | Program _ -> cover "a program" true

(* One thread gives the bits of the job's threads, a NaN as any NaN. *)
let law_threads (b : Support.backend) j =
  covers_job j;
  let many = run_job b b.kernels j in
  equal (option (array (array string))) many (run_job b (module One) j)

(* The same bits with each operand as a C-contiguous copy of its view. *)
let law_layouts (b : Support.backend) j =
  covers_job j;
  cover "computed from views" (views_of j <> []);
  let viewed = run_job b b.kernels j in
  equal (option (array (array string))) viewed (run_job b b.kernels (copied j))

let job_case =
  let open Gen in
  with_pp pp_job
    (frequency
       [
         (3, map (fun c -> Fold c) computed);
         (1, map (fun c -> Fold c) large);
         (2, map (fun c -> State (false, c)) stat_case);
         (2, map (fun c -> State (true, c)) stat_scan_case);
         ( 2,
           map
             (fun c -> Program { c with guard = None })
             (such_that fused_fits fused_case) );
       ])

(* Operands of a few hundred thousand elements, which the job gives several
   threads; long ramps whose blocks' maxima differ; and the stateful and
   fused examples. *)
let job_examples =
  List.map (fun c -> Fold c) (computed_examples @ large_examples)
  @ List.map (fun c -> State (false, c)) (ramp_examples @ stat_examples)
  @ List.map (fun c -> State (true, c)) (stat_scan_examples @ mixed_examples)
  @ List.map (fun c -> Program c) fused_examples

(* The suite *)

let order (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("order, " ^ b.name)
    [
      prop
        ~examples:
          (computed_examples @ made_examples @ far_nan_examples @ wrap_examples
         @ [ zero_example ])
        "each result folds its terms in nx.cpu's order" computed
        (run (law_order b));
      prop ~examples:any_examples "computes every Sum, Prod, Max and Min"
        any_case (run (law_computes b));
      prop ~examples:complex_examples
        "complex sums, products and extremes fold in nx.cpu's order"
        complex_case (run (law_complex b));
      prop
        ~examples:(into_examples @ nan_code_examples)
        "narrow terms reduce in their carrier, rounded once" into_case
        (run (law_into b));
      prop
        ~examples:(stat_into_examples @ nan_code_stat_examples)
        "narrow Arg, Logsumexp and Moments reduce in their carrier"
        stat_into_case (run (law_stat_into b));
      test "a bool of any non-zero byte is true" (fun () ->
          b.around (test_bool_bytes b));
      test "a program of Prog.of_node's operand computes" (fun () ->
          b.around (test_of_node b));
    ]

let values_alone (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("values alone, " ^ b.name)
    [
      prop ~count:40 ~examples:job_examples "one thread gives the job's bits"
        job_case (run (law_threads b));
      prop ~count:40 ~examples:job_examples
        "bits do not depend on layouts, as nx_cpu.mli states" job_case
        (run (law_layouts b));
    ]

let stateful (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("stateful, " ^ b.name)
    [
      prop
        ~examples:(stat_examples @ ramp_examples)
        "Arg is exact, Logsumexp and Moments within their bounds" stat_case
        (run (law_stateful b));
      prop
        ~examples:(ramp_examples @ two_pass_examples)
        "Logsumexp and Moments give the bits of their two passes"
        two_pass_case (run (law_two_passes b));
      prop
        ~examples:(stat_scan_examples @ mixed_examples)
        "Arg scans are exact, Logsumexp and Moments scans within their \
         bounds, with their network's bits"
        stat_scan_case (run (law_stat_scan b));
      prop ~examples:lanes_examples
        "a scan by Arg, Logsumexp or Moments gives each slice its bits alone"
        lanes_case (run (law_lanes b));
      prop ~count:500
        "a Logsumexp of every float dtype is within its bound, NaNs and \
         infinities as stated"
        lse_case (run (law_logsumexp b));
      test "a complex Arg scan holds its first extreme and first NaN"
        (fun () -> b.around (test_complex_arg_scan b));
      test "a reduction or scan into its own operand gives fresh bits"
        (fun () -> b.around (test_in_place b));
    ]

let programs (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("programs, " ^ b.name)
    [
      prop
        ~examples:(fused_examples @ product_examples @ padded_examples)
        "a reduction or scan of a program's output gives the bits of its \
         map's, over plain and padded loads"
        fused_case (run (law_fused b));
      prop
        ~examples:
          [
            { first = fused_example "the second of two outputs" Moments [| 5; 40 |] [| 1 |];
              red2 = Monoid Max; k2 = 0 };
          ]
        "several reductions in one call give each one's bits" several_case
        (run (law_several b));
      test "a destination aliasing a load computes as a fresh one" (fun () ->
          b.around (test_program_in_place b));
      test "reduces many reductions in one claim of their operands" (fun () ->
          b.around (test_many_claims b));
      test "reduces more reductions than one loop holds" (fun () ->
          b.around (test_many_reductions b));
      test "reduces and scans a program of any number of loads" (fun () ->
          b.around (test_wide_loads b));
    ]

let carries (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("carries, " ^ b.name)
    [
      prop
        "a scan from a chunk's carry has the whole scan's bits from that \
         chunk on"
        chunk_case (run (law_chunk_carry b));
    ]

let refusals (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("refusals, " ^ b.name)
    [
      prop
        "computes what Nx_kernel.Shape fits and refuses the rest with \
         Shape_mismatch, writing nothing"
        misfits (run (law_refusals b));
      cases ~name:(fun (n, _, _) -> n) "answers as its contract names"
        (answered b) (fun (_, want, f) ->
          b.around (fun () -> equal Elements.answer want (f ())));
      test
        "refuses a padded load past max_numel or its destinations before \
         copying it"
        (fun () -> b.around (test_huge_padding b));
    ]

let guards (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("guards, " ^ b.name)
    [
      prop ~count:500 ~examples:guard_examples
        "reads and writes no byte past the operand and destinations, \
         whatever pages lie beside them"
        (Gen.pair guard_case Elements.sides)
        (run (law_guards b));
    ]

let () =
  let b = Support.cpu in
  exit
    (Windtrap.run "nx_cpu.reduce"
       [ order b; values_alone b; stateful b; programs b; carries b;
         refusals b; guards b ])
