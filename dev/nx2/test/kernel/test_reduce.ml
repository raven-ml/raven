(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reductions and scans through every kernel library the host runs, with
   operands and results on its device: a float sum within its bound of the
   exact sum, the same bits under every layout of the same values, a scan of
   a prefix the prefix of the scan, and nothing written where declined.
   Every library states nx.cpu's cases and order, which a group checks for
   each: the cases it computes, and its order, bit for bit, against a
   reference built here. nx.cpu's own group checks the same bits on one
   thread as on the job's, and a sum as a contraction with ones. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module Support = Nx_kernels_support

type monoid = Nx_kernel.monoid = Sum | Prod | Max | Min

let pp_dtype ppf (D.Any dt) = D.pp ppf dt
let total s = Array.fold_left ( * ) 1 s
let shape_of (A.Any x) = L.shape (A.layout x)

let monoid_name = function
  | Sum -> "Sum"
  | Prod -> "Prod"
  | Max -> "Max"
  | Min -> "Min"

(* Cases *)

(* A reduction or scan of [x] by [monoid] along [axes], a scan's one, [x]
   drawn through the views [views]. *)
type case = {
  scan : bool;
  monoid : monoid;
  axes : int array;
  x : A.any;
  views : string list;
}

let pp_case ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "%s %s along %a of %a %a (%s)"
    (if c.scan then "scan" else "reduce")
    (monoid_name c.monoid) pp_ints c.axes D.pp (A.dtype x) L.pp (A.layout x)
    (String.concat ", " c.views)

(* [In 0] of one operand of [dt]. *)
let identity dt = P.v ~ins:[| dt |] [| P.In 0 |] ~outs:[| 0 |]

let plain_of (A.Any x) = Nx_kernel.Plain x

(* The loop's shape: the loads' once loaded, else, for a padding that fits
   none, the first load's operand's, which the kernels refuse with the
   padding. *)
let loop_shape (loads : Nx_kernel.load array) =
  match Nx_kernel.Shape.loaded loads with
  | Ok s -> s
  | Error _ -> (
      match loads.(0) with
      | Plain x -> L.shape (A.layout x)
      | Padded { x; _ } -> L.shape (A.layout x))
let dtype_of (A.Any x) = D.Any (A.dtype x)
let accepts m (D.Any dt) = Nx_kernel.accepts (Monoid m) dt

(* By [k], the reduction by [m] of [p]'s output along [axes], or with [scan]
   its scan along [axes.(0)], of [loads] into [dst]. *)
let compute (module K : Nx_kernel.S) ~scan p m axes dst loads =
  if scan then K.scan p ~axis:axes.(0) (Monoid m, 0) ~dsts:[| dst |] loads
  else
    K.reduce p ~shape:(loop_shape loads) ~axes:(Iarray.of_array axes)
      [| (Monoid m, 0) |] ~dsts:[| dst |] loads

(* The result's shape. *)
let result_shape c =
  let s = shape_of c.x in
  if c.scan then s
  else
    Array.of_list
      (List.filteri (fun i _ -> not (Array.mem i c.axes)) (Array.to_list s))

(* [x] where the host reads it: [x] itself on the host, a copy elsewhere. *)
let host (A.Any x as a) =
  if Rig.equal (A.device x) Rig.host then a
  else A.Any (A.to_device Rig.host x)

(* [x] on [b]'s device, its layout kept. *)
let on (b : Support.backend) (A.Any x as a) =
  if Rig.equal (A.device x) b.device then a
  else A.Any (A.to_device b.device x)

(* [c] by [b]'s kernels into [dst], on [b]'s device. *)
let call (b : Support.backend) c dst =
  let x = on b c.x in
  compute b.kernels ~scan:c.scan (identity (dtype_of x)) c.monoid c.axes dst
    [| plain_of x |]

(* The kernels' result into a fresh destination, read where the host reads
   it; [None] if they declined. *)
let run (b : Support.backend) c =
  let (A.Any x) = c.x in
  let dst = A.Any (A.create b.device (A.dtype x) (result_shape c)) in
  match call b c dst with
  | A.Done -> Some (host dst)
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Values *)

(* Bits of float32 and float64 NaNs, quiet and signalling, of both signs. *)
let nans32 = [| 0x7FC00000l; 0xFFC12345l; 0x7F800001l; 0xFFA00002l |]

let nans64 =
  [| 0x7FF8000000000000L; 0x7FF0000000000001L; 0xFFF4000000000002L |]

(* A float of [p] significant bits in (-4, 4). *)
let full r p =
  let m = Random.State.int64 r (Int64.shift_left 1L (p + 1)) in
  Float.ldexp (Int64.to_float (Int64.sub m (Int64.shift_left 1L p))) (2 - p)

(* Integers at and about the edges of every width. *)
let edges =
  [|
    0L; 1L; -1L; 0x7FL; 0x80L; 0xFFL; 0x7FFFL; 0x8000L; 0x7FFF_FFFFL;
    0x8000_0000L; Int64.max_int; Int64.min_int;
  |]

(* [n] elements of [dt] as the bytes of a C-contiguous array, from [seed]:
   floats, and complex numbers' parts, of full precision in (-4, 4), with
   [-0.], infinities and NaNs of drawn payloads at the rate [specials] (one
   in that many, none for 0); integers small, or at the edges of the widths
   at that rate. *)
let elements (type v s) (dt : (v, s) D.t) n ~specials seed =
  let r = Random.State.make [| seed |] in
  let special () = specials > 0 && Random.State.int r specials = 0 in
  let pick a = a.(Random.State.int r (Array.length a)) in
  let b = Bytes.create (max 1 (D.bytes dt n)) in
  let f32 at =
    let bits =
      if special () then
        pick (Array.append nans32 [| 0x80000000l; 0x7F800000l; 0xFF800000l |])
      else Int32.bits_of_float (full r 23)
    in
    Bytes.set_int32_le b at bits
  in
  let f64 at =
    let bits =
      if special () then
        pick
          (Array.append nans64
             [|
               Int64.bits_of_float (-0.);
               Int64.bits_of_float infinity;
               Int64.bits_of_float neg_infinity;
             |])
      else Int64.bits_of_float (full r 52)
    in
    Bytes.set_int64_le b at bits
  in
  for i = 0 to n - 1 do
    match dt with
    | D.Float32 -> f32 (4 * i)
    | D.Float64 -> f64 (8 * i)
    | D.Complex64 ->
        f32 (8 * i);
        f32 ((8 * i) + 4)
    | D.Complex128 ->
        f64 (16 * i);
        f64 ((16 * i) + 8)
    | _ when D.bits dt >= 8 ->
        let v =
          if special () then pick edges
          else Int64.of_int (Random.State.int r 16 - 8)
        in
        let w = D.bits dt / 8 in
        for j = 0 to w - 1 do
          Bytes.set b ((w * i) + j)
            (Char.chr
               (Int64.to_int
                  (Int64.logand (Int64.shift_right_logical v (8 * j)) 0xFFL)))
        done;
        if D.equal dt D.Bool then Bytes.set b i (Char.chr (Random.State.int r 2))
    | _ ->
        (* Sub-byte: random codes. *)
        Bytes.set b (i * D.bits dt / 8) (Char.chr (Random.State.int r 256))
  done;
  A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string (Bytes.to_string b))

(* Views *)

(* How an array of shape [s] is viewed: made with its axes in another order,
   stepped by two along its last made axis, reversed along a made axis, or
   broadcast along its first axis from one element. *)
type view = {
  perm : int array;
  stepped : bool;
  reversed : int option;
  broadcast : bool;
}

let view_names v =
  (if v.perm <> Array.init (Array.length v.perm) Fun.id then [ "permuted" ]
   else [])
  @ (if v.stepped then [ "stepped" ] else [])
  @ (if v.reversed <> None then [ "reversed" ] else [])
  @ if v.broadcast then [ "broadcast" ] else []

let view_of r =
  let open Gen in
  let* p = permutation ~pp:Format.pp_print_int (List.init r Fun.id) in
  let* stepped = bool in
  let* reversed = option (int_range 0 (max 0 (r - 1))) in
  let+ broadcast = frequency [ (4, constant false); (1, constant true) ] in
  {
    perm = Array.of_list p;
    stepped = stepped && r > 0;
    reversed = (if r > 0 then reversed else None);
    broadcast = broadcast && r > 0;
  }

(* An array of shape [s] through the view [v]: its elements, [make n] of
   them, are made C-contiguous in the shape before the view, then moved into
   it. *)
let operand_with make s v =
  let r = Array.length s in
  let s0 = Array.mapi (fun i e -> if v.broadcast && i = 0 then 1 else e) s in
  let inv = Array.make r 0 in
  Array.iteri (fun i m -> inv.(m) <- i) v.perm;
  (* Made: [s0] with its axes in made order, the last doubled if stepped. *)
  let made = Array.init r (fun m -> s0.(inv.(m))) in
  let last = r - 1 in
  let made' = Array.mapi (fun i e -> if v.stepped && i = last then 2 * e else e) made in
  let whole e = { M.start = 0; count = e; step = 1 } in
  let move m x = Option.get (A.move m x) in
  let (A.Any x) = make (total made') in
  let x = move (M.Reshape (Iarray.of_array made')) x in
  let x =
    if not v.stepped then x
    else
      move
        (M.Slice
           (Iarray.of_array
              (Array.mapi
              (fun i e ->
                if i = last then { M.start = 1; count = made.(i); step = 2 }
                else whole e)
              made')))
        x
  in
  let x =
    match v.reversed with
    | Some a when made.(a) > 0 ->
        move
          (M.Slice
             (Iarray.of_array
                (Array.mapi
                (fun i e ->
                  if i = a then { M.start = e - 1; count = e; step = -1 }
                  else whole e)
                made)))
          x
    | _ -> x
  in
  let x = move (M.Permute (Iarray.of_array v.perm)) x in
  A.Any (if v.broadcast then move (M.Broadcast (Iarray.of_array s)) x else x)

(* An array of [dt] of shape [s] through the view [v], of [elements]. *)
let operand (D.Any dt) s v ~specials seed =
  operand_with (fun n -> A.Any (elements dt n ~specials seed)) s v

(* One long axis past several blocks or chunks, or at a block's or a
   chunk's edge, between up to two short ones. *)
let long =
  let open Gen in
  let edge =
    of_list ~pp:Format.pp_print_int
      [ 15; 16; 17; 1023; 1024; 1025; 4095; 4096; 4097 ]
  in
  let* n = frequency [ (2, edge); (1, int_range 5000 70000) ] in
  let* before = array ~size:(int_range 0 1) (int_range 1 3) in
  let+ after = array ~size:(int_range 0 1) (int_range 1 3) in
  Array.concat [ before; [| n |]; after ]

(* Shapes: small extents of any rank; a long axis; rows of a few to a few
   hundred terms; an empty axis beside others. *)
let shape_gen =
  let open Gen in
  let small = array ~size:(int_range 0 4) (int_range 0 5) in
  let rows =
    let* r = int_range 1 300 in
    let+ k = of_list ~pp:Format.pp_print_int [ 2; 4; 9; 64; 300 ] in
    [| r; k |]
  in
  let empty =
    of_list [ [| 0 |]; [| 3; 0 |]; [| 0; 4 |]; [| 2; 0; 3 |] ]
  in
  frequency [ (5, small); (3, long); (2, rows); (4, empty) ]

let monoids = [ Sum; Prod; Max; Min ]

(* A case of a dtype of [dts]: a reduction over any subset of the axes, or a
   scan along one. A scan of rank 0 takes a shape of one axis; a maximum or
   minimum takes an element along each empty reduced axis, since one of no
   term is refused. *)
let case_of ?(scan = Gen.bool) ?(monoids = monoids) ?(shapes = shape_gen) dts =
  let open Gen in
  let* scan = scan in
  let* d = of_list ~pp:pp_dtype dts in
  let* monoid =
    of_list (List.filter (fun m -> accepts m d) monoids)
  in
  let* s = shapes in
  let s = if scan && Array.length s = 0 then [| 3 |] else s in
  let r = Array.length s in
  let* axes =
    if scan then map (fun a -> [| a |]) (int_range 0 (r - 1))
    else
      let+ keep = array ~size:(constant r) bool in
      Array.of_list (List.filter (fun i -> keep.(i)) (List.init r Fun.id))
  in
  let s =
    if scan || not (monoid = Max || monoid = Min) then s
    else Array.mapi (fun i e -> if Array.mem i axes then max 1 e else e) s
  in
  let* v = view_of r in
  let* specials =
    frequency
      [ (2, constant 0); (2, constant 500); (1, constant 3000); (3, constant 20) ]
  in
  let+ seed = int in
  { scan; monoid; axes; x = operand d s v ~specials seed; views = view_names v }

let base =
  D.
    [
      Any Float32;
      Any Float64;
      Any Int8;
      Any Uint8;
      Any Int16;
      Any Uint16;
      Any Int32;
      Any Uint32;
      Any Int64;
      Any Uint64;
      Any Bool;
    ]

let floats = D.[ Any Float32; Any Float64 ]
(* Floats, whose order and NaNs the laws check, come twice as often. *)
let computed = Gen.with_pp pp_case (case_of (floats @ floats @ base))
let any_case = Gen.with_pp pp_case (case_of D.all)
(* Scans draw long axes as often as the others, so that prefixes pass a
   chunk. *)
let scans =
  Gen.with_pp pp_case
    (case_of ~scan:(Gen.constant true)
       ~shapes:(Gen.frequency [ (1, shape_gen); (1, long) ])
       base)

let sums =
  Gen.with_pp pp_case
    (case_of ~scan:(Gen.constant false) ~monoids:[ Sum ] floats)

(* Sums of one axis, of any length. *)
let dots =
  Gen.with_pp pp_case
    (case_of ~scan:(Gen.constant false) ~monoids:[ Sum ]
       ~shapes:(Gen.map (fun n -> [| n |]) (Gen.int_range 0 6000))
       floats)

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

(* Examples: one case for each regime a law covers, so that every seed
   reaches every label whatever the generator draws. *)
let example ?(scan = false) ?view ?(specials = 0) monoid dt s axes =
  let r = Array.length s in
  let view =
    Option.value view
      ~default:
        {
          perm = Array.init r Fun.id;
          stepped = false;
          reversed = None;
          broadcast = false;
        }
  in
  {
    scan;
    monoid;
    axes;
    x = operand dt s view ~specials 1;
    views = view_names view;
  }

let f32 = D.Any D.Float32
let f64 = D.Any D.Float64

let computed_examples =
  [
    example Sum f32 [| 0; 3 |] [| 1 |];
    example Sum f32 [| 3; 0 |] [| 1 |];
    example Sum f32 [| 5 |] [||];
    example ~specials:20 Sum f32 [| 5000 |] [| 0 |];
    example Max f32 [| 100; 3 |] [| 1 |];
    (* Short rows of few terms, which units take in bands of rows. *)
    (* CR: These two examples and the matching large_examples case have
       contiguous kept axes, which coalesce into one: no multirow bands.
       Use the existing view with perm=[|0;2;1|], no step, reversal or
       broadcast. Its strides [26;1;13] retain row width 13 and separate
       outer rows, covering full and partial bands with the same laws. *)
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

(* A float vector of [xs] at [dt], read with a step of two where
   [stepped]. *)
let vector_of (type s) (dt : (float, s) D.t) ?(stepped = false) xs =
  let n = Array.length xs in
  if not stepped then A.Any (A.of_array dt [| n |] xs)
  else
    let spread i = if i mod 2 = 1 then xs.(i / 2) else 7. in
    let x = A.of_array dt [| 2 * n |] (Array.init (2 * n) spread) in
    let odd = { M.start = 1; count = n; step = 2 } in
    A.Any (Option.get (A.move (M.Slice [| odd |]) x))

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

let any_examples =
  [
    example Sum f32 [| 4 |] [| 0 |];
    example Sum (D.Any D.Float16) [| 4 |] [| 0 |];
  ]

let scan_examples = [ example ~scan:true Sum f32 [| 9000 |] [| 0 |] ]

let dot_examples =
  [ example Sum f32 [| 3000 |] [| 0 |]; example Sum f32 [| 3 |] [||] ]

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

(* The reference *)

(* A term: a float's value with its bits, or an integer's value as an
   [int64] of its bits, sign- or zero-extended. *)
type term = F of float * int64 | I of int64

(* [x]'s elements in C order of indices. *)
let terms (A.Any x) =
  let c = A.copy x in
  match A.dtype x with
  | D.Float32 ->
      Array.map
        (fun b -> F (Int32.float_of_bits b, Int64.of_int32 b))
        (A.to_array (Option.get (A.bitcast D.Int32 c)))
  | D.Float64 ->
      Array.map
        (fun b -> F (Int64.float_of_bits b, b))
        (A.to_array (Option.get (A.bitcast D.Int64 c)))
  | _ ->
      let d = A.create Rig.host D.Int64 (L.shape (A.layout c)) in
      (match Support.map1 (module Nx_cpu) Cast ~dst:d c with
      | A.Done -> ()
      | r -> failf "a cast answered %a" Nx_array_support.pp_answer r);
      Array.map (fun v -> I v) (A.to_array d)

(* An expected result: its bits, or some NaN. *)
type want = Bits of int64 | Some_nan

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* [v] in the integer dtype [dt]: its low bits, sign- or zero-extended. *)
let wrap (D.Any dt) v =
  let bits = D.bits dt in
  if bits >= 64 then v
  else
    let s = 64 - bits in
    if D.is D.Signed dt then Int64.shift_right (Int64.shift_left v s) s
    else Int64.shift_right_logical (Int64.shift_left v s) s

(* The monoid [m] at [dt] over two terms, and its identity. Floats round
   once to [dt], through a double, which holds float32's sums and products
   exactly or rounds them innocuously; a NaN operand gives a NaN. An
   extreme's identity is the least (greatest) element. *)
let op (D.Any dt as d) m =
  let float x =
    if D.equal dt D.Float32 then
      let x = round32 x in
      F (x, Int64.of_int32 (Int32.bits_of_float x))
    else F (x, Int64.bits_of_float x)
  in
  let unsigned = not (D.is D.Signed dt) in
  let cmp x y =
    if unsigned then Int64.unsigned_compare x y else Int64.compare x y
  in
  let least = if unsigned then 0L else wrap d (Int64.shift_left 1L (D.bits dt - 1)) in
  let greatest = Int64.lognot least |> wrap d in
  (* The greater of two floats, -0 below +0. *)
  let fmax x y =
    if x > y then x else if y > x then y else if Float.sign_bit x then y else x
  in
  let fmin x y =
    if x < y then x else if y < x then y else if Float.sign_bit x then x else y
  in
  let both fl it a b =
    match (a, b) with
    | F (x, _), _ when Float.is_nan x -> a
    | _, F (y, _) when Float.is_nan y -> b
    | F (x, _), F (y, _) -> float (fl x y)
    | I x, I y -> I (it x y)
    | _ -> invalid_arg "op: a float and an integer"
  in
  let is_float = D.is D.Float dt in
  match m with
  | Sum ->
      (both ( +. ) (fun x y -> wrap d (Int64.add x y)), if is_float then float 0. else I 0L)
  | Prod ->
      (both ( *. ) (fun x y -> wrap d (Int64.mul x y)), if is_float then float 1. else I 1L)
  | Max ->
      ( both fmax (fun x y -> if cmp x y >= 0 then x else y),
        if is_float then float neg_infinity else I least )
  | Min ->
      ( both fmin (fun x y -> if cmp x y <= 0 then x else y),
        if is_float then float infinity else I greatest )

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

let is_nan = function F (x, _) -> Float.is_nan x | I _ -> false
let bits_of = function F (_, b) -> b | I v -> v

(* [t]'s bits, or some NaN where it is one: a NaN result promises a NaN,
   never which one. *)
let want_of t = if is_nan t then Some_nan else Bits (bits_of t)

(* The terms of each output of [c], in C order of outputs, each in C order
   of the reduced axes' indices; a scan's are its slices, in order. *)
let outputs c ts =
  let s = shape_of c.x in
  let r = Array.length s in
  let strides = Array.make r 1 in
  for i = r - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * s.(i + 1)
  done;
  let kept = List.filter (fun i -> not (Array.mem i c.axes)) (List.init r Fun.id) in
  let ks = Array.of_list kept in
  let at idx = Array.fold_left ( + ) 0 (Array.mapi (fun i j -> j * strides.(i)) idx) in
  let outer = indices (Array.map (fun i -> s.(i)) ks) in
  let inner = indices (Array.map (fun a -> s.(a)) c.axes) in
  List.map
    (fun o ->
      let idx = Array.make r 0 in
      Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
      Array.of_list
        (List.map
           (fun t ->
             Array.iteri (fun q a -> idx.(a) <- t.(q)) c.axes;
             ts.(at idx))
           inner))
    outer

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

(* The position of each index of shape [s] in C order. *)
let position s idx =
  let p = ref 0 in
  Array.iteri (fun i j -> p := (!p * s.(i)) + j) idx;
  !p

(* The results of [c] in C order of the result's indices, from its terms
   [ts]: [reduce] of each output's terms, or [scan] of each slice's. [fill]
   fills the array before. *)
let results c ts ~reduce ~scan ~fill =
  if not c.scan then Array.of_list (List.map reduce (outputs c ts))
  else
    let s = shape_of c.x and a = c.axes.(0) in
    let r = Array.length s in
    let ks = Array.of_list (List.filter (( <> ) a) (List.init r Fun.id)) in
    let y = Array.make (total s) fill in
    List.iter
      (fun o ->
        let idx = Array.make r 0 in
        Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
        let at j =
          idx.(a) <- j;
          position s idx
        in
        let slice = Array.init s.(a) (fun j -> ts.(at j)) in
        Array.iteri (fun j w -> y.(at j) <- w) (scan slice))
      (indices (Array.map (fun i -> s.(i)) ks));
    y

(* The expected results of [c] in C order of the result's indices. *)
let expected c =
  let (A.Any x) = c.x in
  let m = op (D.Any (A.dtype x)) c.monoid in
  results c (terms c.x) ~fill:Some_nan
    ~reduce:(fun o -> want_of (ordered m o))
    ~scan:(scanned m)

(* The kernels' result's bits in C order. *)
let got y =
  Array.map
    (function F (x, b) -> if Float.is_nan x then Some_nan else Bits b | I v -> Bits v)
    (terms y)

let show = function Bits b -> Printf.sprintf "%Lx" b | Some_nan -> "nan"

(* The first results that differ from [want]: a [Some_nan] wanted matches
   any NaN. *)
let differ want y =
  let g = terms y in
  let bad = ref [] in
  Array.iteri
    (fun i w ->
      let ok =
        match (w, g.(i)) with
        | Some_nan, t -> is_nan t
        | Bits b, t -> bits_of t = b
      in
      if (not ok) && List.length !bad < 8 then
        bad := Printf.sprintf "result %d: %Lx, expected %s" i (bits_of g.(i)) (show w) :: !bad)
    want;
  List.rev !bad

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
let zsums c ts =
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
let zexpected c =
  let narrow = narrow c.x and first = c.monoid = Prod in
  let ts = zterms c.x in
  if c.monoid = Sum then zsums c ts
  else
    let m = zop narrow c.monoid in
    results c ts ~fill:Z_nan
      ~reduce:(fun o -> zwant_of (ordered ~first m o))
      ~scan:(zscanned ~first m)

(* Complex sums, products and extremes, reduced and scanned, in nx.cpu's
   order, bit for bit, a NaN as any NaN: a sum's NaN part, else a number
   with a NaN part. *)
let law_complex (b : Support.backend) c =
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
  | Some y ->
      let want = zexpected c and got = zterms y in
      let part w x b =
        match w with Some_nan -> Float.is_nan x | Bits w -> w = b
      in
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
            match w with
            | Z_nan -> "nan"
            | Z (wr, wi) -> show wr ^ " " ^ show wi
          in
          if (not ok) && List.length !bad < 8 then
            bad :=
              Printf.sprintf "result %d: %Lx %Lx, expected %s" i gr gi shown
              :: !bad)
        want;
      equal (list string) [] (List.rev !bad)

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

let covers c =
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

(* A float sum of [n] terms is within γ(n - 1) Σ|x| of the exact sum; a NaN
   or infinite sum is one where the plain sum is. *)
let law_bound (b : Support.backend) c =
  let (A.Any x) = c.x in
  match (A.dtype x, c.monoid, c.scan) with
  | (D.Float32 | D.Float64), Sum, false -> (
      match run b c with
      | None -> cover "declined" true
      | Some y ->
          cover "computed" true;
          let u = if D.bits (A.dtype x) = 32 then 0x1p-24 else 0x1p-53 in
          let per = outputs c (terms c.x) in
          let g = terms y in
          let bad =
            List.concat
              (List.mapi
                 (fun i o ->
                   let xs = Array.map (function F (v, _) -> v | I _ -> assert false) o in
                   let plain = Array.fold_left ( +. ) 0. xs in
                   let v = match g.(i) with F (v, _) -> v | I _ -> assert false in
                   if not (Float.is_finite plain) then
                     if Float.is_nan plain = Float.is_nan v && (Float.is_nan v || v = plain) then []
                     else [ Printf.sprintf "result %d: %h, the plain sum %h" i v plain ]
                   else
                     let n = float_of_int (max 0 (Array.length xs - 1)) in
                     let mag = Array.fold_left (fun a x -> a +. Float.abs x) 0. xs in
                     (* The exact sum: TwoSum's errors carried in a second part. *)
                     let hi = ref 0. and lo = ref 0. in
                     Array.iter
                       (fun x ->
                         let s = !hi +. x in
                         let z = s -. !hi in
                         lo := !lo +. (!hi -. (s -. z)) +. (x -. z);
                         hi := s)
                       xs;
                     let exact = !hi +. !lo in
                     let g = n *. u /. (1. -. (n *. u)) in
                     let bound = (g *. mag) +. (0x1p-100 *. mag) in
                     if Float.abs (v -. exact) <= bound then []
                     else [ Printf.sprintf "result %d: %h is %h from %h, past %h" i v (v -. exact) exact bound ])
                 per)
          in
          equal (list string) [] (List.filteri (fun i _ -> i < 8) bad))
  | _ -> cover "not a float sum" true

(* Each operand as a C-contiguous copy of its view gives the same bits. *)
let law_layouts b c =
  let (A.Any x) = c.x in
  let c' = { c with x = A.Any (A.copy x); views = [] } in
  match (run b c, run b c') with
  | Some y, Some y' ->
      cover "computed from views" (c.views <> []);
      equal (list string) [] (differ (got y') y)
  | None, None -> cover "declined" true
  | _ -> failf "a view and its copy were not both computed"

(* The scan of a prefix of the axis is the prefix of the scan. *)
let law_prefix (b : Support.backend) c =
  let s = shape_of c.x and a = c.axes.(0) in
  let m = s.(a) / 2 in
  let (A.Any x) = c.x in
  let cut =
    A.Any
      (Option.get
         (A.move
            (M.Slice
               (Iarray.of_array (Array.mapi
                  (fun i e ->
                    if i = a then { M.start = 0; count = m; step = 1 }
                    else { M.start = 0; count = e; step = 1 })
                  s)))
            x))
  in
  match (run b c, run b { c with x = cut }) with
  | Some (A.Any y), Some y' ->
      cover "a prefix past a chunk" (m > 4096);
      let prefix =
        A.Any
          (Option.get
             (A.move
                (M.Slice
                   (Iarray.of_array (Array.mapi
                      (fun i e ->
                        if i = a then { M.start = 0; count = m; step = 1 }
                        else { M.start = 0; count = e; step = 1 })
                      s)))
                y))
      in
      equal (list string) [] (differ (got prefix) y')
  | None, None -> cover "declined" true
  | _ -> failf "a scan and its prefix were not both computed"

(* A kernel that declines writes nothing. *)
(* CR: terms casts narrow floats and complex values to Int64, so writes
   can disappear (Float16 0.5 and 0.75 both become 0). Initialize dst
   with deterministic bytes, keeping spare packed bits zero, and compare
   its full buffer via Rig.Buffer.copy to host and blit_to_bytes. Use the
   same byte check in test_contract's refusal law. Keep the Float16 reduce
   example and add its nonempty scan counterpart. *)
let law_declined (b : Support.backend) c =
  let (A.Any x) = c.x in
  let dst = on b (A.Any (A.create Rig.host (A.dtype x) (result_shape c))) in
  let bits () = Array.map bits_of (terms (host dst)) in
  let before = bits () in
  match call b c dst with
  | A.Declined ->
      cover "declined" true;
      equal (array int64) before (bits ())
  | A.Done -> cover "computed" true
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* nx.cpu's cases and order *)

(* Whether a float result of [c] is a NaN made from numbers: one whose terms,
   a scan's up to it, hold no NaN. *)
let makes_nan c =
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

(* The dtype nx.cpu reduces terms of [dt] in: float32 for a narrow float,
   int8 for int4, uint8 for uint4, bool for bit, else [dt]. *)
let carrier (D.Any dt as d) =
  match dt with
  | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
  | D.Float4_e2m1fn ->
      D.Any D.Float32
  | D.Int4 -> D.Any D.Int8
  | D.Uint4 -> D.Any D.Uint8
  | D.Bit -> D.Any D.Bool
  | _ -> d

let same (D.Any a) (D.Any b) = D.equal a b
let staged d = not (same (carrier d) d)

(* The full blocks of 1024 terms of one of [outputs] outputs of [per]
   terms each that nx.cpu reduces in one call, before its cap of 32 KiB of
   terms in the carrier: below 64 outputs, an output's blocks fall into
   groups of a power of two, about 64 groups in all, and a call takes one
   group's full blocks. *)
let blocks_a_call ~outputs per =
  let blocks = (per + 1023) / 1024 in
  let group =
    if outputs >= 64 || blocks <= 1 then blocks
    else
      let want = (64 + outputs - 1) / outputs in
      let each = (blocks + want - 1) / want in
      let rec up g = if g < each then up (2 * g) else g in
      up 1
  in
  min group (per / 1024)

(* Covers nx.cpu's calls of several full blocks of one of [outputs]
   outputs of [per] terms in [path], and of more than a call holds of
   them in the carrier [c]. *)
let covers_blocks path (D.Any c) ~outputs per =
  let b = if path then blocks_a_call ~outputs per else 0 in
  cover "several full blocks of an output's terms a call" (b >= 2);
  cover "more full blocks of an output's terms than a call holds"
    (b > 256 / D.bits c)

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

let law_computes b c =
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
  let dst = A.Any (A.create Rig.host d (result_shape c)) in
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
let reached c =
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

(* A narrow operand's terms reduce in its carrier, and each result rounds
   once into its destination: the bits of the carrier's reduction of the
   operand cast into it, cast into the destination, a NaN as any NaN. A
   float result a NaN term reaches is a NaN. *)
let law_into (b : Support.backend) (c, d) =
  let x = dtype_of c.x in
  cover "narrow" (staged x);
  cover "into another dtype" (not (same d x));
  cover "scan" c.scan;
  let s = shape_of c.x in
  let per = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  cover "several blocks" (per > 1024);
  cover "a scan of several chunks" (c.scan && per > 4096);
  (* One output's terms a unit, staged by blocks: the last axis reduced,
     no kept axis stepping less than it. *)
  let r = Array.length s in
  let lanes =
    staged x && (not c.scan) && r > 0 && Array.mem (r - 1) c.axes
    && not (List.mem "permuted" c.views || List.mem "broadcast" c.views)
  in
  let outputs = if per = 0 then 0 else total s / per in
  covers_blocks lanes (carrier x) ~outputs per;
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  let got = run_into b.kernels d c c.x in
  let wide = run_into b.kernels (carrier x) c (cast_into (carrier x) c.x) in
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

(* A result of a narrow float into its own dtype that a NaN term reaches is
   a NaN, whether the term is signalling or quiet: the reduction of zeros
   with two NaNs, a few terms and past a block and a scan's chunk, by each
   monoid, Logsumexp and Moments, the running results from the first NaN
   on, and Arg's extreme, with that NaN's position, reduced and scanned. *)
let test_nan_bits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let code w c =
    String.init w (fun j -> Char.chr ((c lsr (8 * j)) land 0xFF))
  in
  let check (D.Any dt as d) first other (n, i, j) =
    let w = D.bits dt / 8 in
    let es =
      Array.init n (fun k ->
          code w (if k = i then first else if k = j then other else 0))
    in
    let x = Elements.of_elements d [| n |] es in
    let nan = es.(i) and same = array (Elements.same d) in
    (* Zeros' running results, then NaNs. *)
    let running = Array.init n (fun k -> if k < i then es.(0) else nan) in
    let run scan m =
      let c = { scan; monoid = m; axes = [| 0 |]; x; views = [] } in
      Elements.elements (run_into b.kernels d c x)
    in
    let msg r =
      Format.asprintf "%s of %d %a terms, %x first" r n D.pp dt first
    in
    List.iter
      (fun m ->
        let msg = msg (monoid_name m) in
        equal ~msg same [| nan |] (run false m);
        equal ~msg:(msg ^ ", scanned") same running (run true m))
      monoids;
    let stat ~scan r k =
      let s = if scan then [| n |] else [||] in
      let dsts = Array.init k (fun _ -> A.Any (A.create Rig.host dt s)) in
      let loads = [| plain_of x |] in
      let answer =
        if scan then K.scan (identity d) ~axis:0 (r, 0) ~dsts loads
        else
          K.reduce (identity d) ~shape:[| n |] ~axes:[| 0 |] [| (r, 0) |] ~dsts
            loads
      in
      equal ~msg:"its answer" bool true (answer = A.Done);
      Array.map Elements.elements dsts
    in
    equal ~msg:(msg "Logsumexp") (array same) [| [| nan |] |]
      (stat ~scan:false Logsumexp 1);
    equal ~msg:(msg "Moments") (array same)
      [| [| nan |]; [| nan |] |]
      (stat ~scan:false Moments 2);
    equal ~msg:(msg "Moments, scanned") (array same)
      [| running; running |]
      (stat ~scan:true Moments 2);
    let lse = (stat ~scan:true Logsumexp 1).(0) in
    equal ~msg:(msg "Logsumexp, scanned, from the first NaN") same
      (Array.sub running i (n - i))
      (Array.sub lse i (n - i));
    List.iter
      (fun (e : Nx_kernel.extreme) ->
        let v = A.create Rig.host dt [||]
        and at = A.create Rig.host D.Int64 [||] in
        let answer =
          K.reduce (identity d) ~shape:[| n |] ~axes:[| 0 |] [| (Arg e, 0) |]
            ~dsts:[| A.Any v; A.Any at |] [| plain_of x |]
        in
        equal ~msg:"its answer" bool true (answer = A.Done);
        equal
          ~msg:(msg (if e = Max then "Arg Max" else "Arg Min"))
          (pair same int64)
          ([| nan |], Int64.of_int i)
          (Elements.elements (A.Any v), (A.to_array at).(0));
        let v = A.create Rig.host dt [| n |]
        and at = A.create Rig.host D.Int64 [| n |] in
        let answer =
          K.scan (identity d) ~axis:0 (Arg e, 0) ~dsts:[| A.Any v; A.Any at |]
            [| plain_of x |]
        in
        equal ~msg:"its answer" bool true (answer = A.Done);
        equal
          ~msg:(msg (if e = Max then "Arg Max scanned" else "Arg Min scanned"))
          (pair same (array int64))
          ( running,
            Array.init n (fun k -> if k < i then 0L else Int64.of_int i) )
          (Elements.elements (A.Any v), A.to_array at))
      [ Max; Min ]
  in
  List.iter
    (fun (d, first, other) ->
      List.iter (check d first other) [ (4, 1, 2); (5000, 4500, 4700) ])
    D.
      [
        (Any Float16, 0x7C01, 0x7E00);
        (Any Bfloat16, 0x7F81, 0x7FC0);
        (Any Float8_e5m2, 0x7D, 0x7F);
        (Any Float8_e5m2, 0x7E, 0x7F);
        (Any Float8_e5m2, 0xFD, 0x7E);
        (Any Float8_e4m3fn, 0xFF, 0x7F);
      ]

(* A sum of one axis is a contraction of it with ones, in lane order: the
   two share their order. *)
let law_contract (b : Support.backend) c =
  let (A.Any x) = c.x in
  let s = shape_of c.x in
  let ones n : A.any option =
    match A.dtype x with
    | D.Float32 -> Some (A.Any (A.of_array D.Float32 [| n |] (Array.make n 1.)))
    | D.Float64 -> Some (A.Any (A.of_array D.Float64 [| n |] (Array.make n 1.)))
    | _ -> None
  in
  let sum_of_axis = (not c.scan) && c.monoid = Sum && Array.length s = 1 && c.axes = [| 0 |] in
  match if sum_of_axis then ones s.(0) else None with
  | Some (A.Any ones) -> (
      let module K = (val b.kernels) in
      let dst = A.create Rig.host (A.dtype x) [||] in
      let summed =
        K.contract ~order:Lane ~batch:[||] ~contracting:[| (0, 0) |]
          ~segments:None ~acc:(dtype_of c.x) ~dst ~init:None
          { Nx_kernel.array = A.Any x; scale = None }
          { Nx_kernel.array = A.Any ones; scale = None }
      in
      match (run b c, summed) with
      | Some y, A.Done ->
          cover "computed" true;
          equal (list string) [] (differ (got (A.Any dst)) y)
      | _ -> cover "declined" true)
  | _ -> cover "not a sum of one float axis" true

(* nx.cpu with its reductions and scans on one thread. *)
module One = struct
  include Nx_cpu

  let reduce p ~shape ~axes rs ~dsts loads =
    Nx_cpu.one_thread (fun () -> Nx_cpu.reduce p ~shape ~axes rs ~dsts loads)

  let scan ?carry p ~axis rk ~dsts loads =
    Nx_cpu.one_thread (fun () -> Nx_cpu.scan ?carry p ~axis rk ~dsts loads)
end

(* One thread gives the bits of the job's threads, a NaN as any NaN. *)
let law_threads (b : Support.backend) c =
  let (A.Any x) = c.x in
  let into k =
    let dst = A.create Rig.host (A.dtype x) (result_shape c) in
    let p = identity (dtype_of c.x) in
    match compute k ~scan:c.scan p c.monoid c.axes (A.Any dst) [| plain_of c.x |] with
    | A.Done -> A.Any dst
    | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r
  in
  let one = into (module One) and many = into b.kernels in
  let d = dtype_of c.x in
  cover "scan" c.scan;
  cover "few outputs" (total (result_shape c) < 64);
  cover "narrow" (staged d);
  (* A narrow result read through its carrier, which holds its value. *)
  let wide y = if staged d then cast_into (carrier d) y else y in
  equal (list string) [] (differ (got (wide many)) (wide one))

(* A NaN made from numbers before two NaN terms past the first block and
   chunk: the sum, the maximum, the running sum and the running product
   agree with the reference, NaN where a NaN reaches. *)
let test_far_nans (b : Support.backend) () =
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
  List.iter
    (fun (scan, monoid) ->
      let c = { scan; monoid; axes = [| 0 |]; x; views = [] } in
      match run b c with
      | None -> failf "%s declined a float32 %s" b.name (monoid_name monoid)
      | Some y ->
          equal ~msg:(monoid_name monoid) (list string) []
            (differ (expected c) y))
    [ (false, Sum); (false, Max); (true, Sum); (true, Prod) ]

(* Products and sums of 8- and 16-bit integers at their extremes wrap in
   their width: 0xFFFF · 0xFFFF is 1 at uint16. *)
let test_wrap (b : Support.backend) () =
  let fold (type s) (dt : (int, s) D.t) m xs want =
    let x = on b (A.Any (A.of_array dt [| Array.length xs |] xs)) in
    let dst = on b (A.Any (A.create Rig.host dt [||])) in
    let p = identity (D.Any dt) in
    (match compute b.kernels ~scan:false p m [| 0 |] dst [| plain_of x |] with
    | A.Done -> ()
    | r -> failf "reduce answered %a" Nx_array_support.pp_answer r);
    equal
      ~msg:(Printf.sprintf "%s of %s" (monoid_name m) (D.name dt))
      int want
      (A.get (A.expect dt (host dst)) [||])
  in
  fold D.Uint16 Prod [| 0xFFFF; 0xFFFF; 3 |] 3;
  fold D.Int16 Prod [| -32768; -1 |] (-32768);
  fold D.Int16 Prod [| 32767; 32767 |] 1;
  fold D.Uint8 Prod [| 255; 255 |] 1;
  fold D.Int8 Prod [| -128; -1 |] (-128);
  fold D.Uint16 Sum [| 0xFFFF; 0xFFFF |] 0xFFFE;
  fold D.Int16 Sum [| 32767; 1 |] (-32768)

let test_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let on x = on b (A.Any x) in
  let x = on (A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]) in
  let p = identity (D.Any D.Float32) in
  let reduce m dst x = compute b.kernels ~scan:false p m [| 1 |] dst [| plain_of x |] in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let wrong = on (A.create Rig.host D.Float32 [| 3 |]) in
  equal ~msg:"a destination of another shape" answer A.Shape_mismatch
    (reduce Sum wrong x);
  let empty = on (A.create Rig.host D.Float32 [| 2; 0 |]) in
  let dst = on (A.create Rig.host D.Float32 [| 2 |]) in
  equal ~msg:"a maximum of no term" answer A.Shape_mismatch (reduce Max dst empty);
  equal ~msg:"a sum of no term" answer A.Done (reduce Sum dst empty);
  equal ~msg:"is +0" (array float_exact) [| 0.; 0. |]
    (A.to_array (A.expect D.Float32 (host dst)));
  let sum = [| (Nx_kernel.Monoid Sum, 0) |] in
  equal ~msg:"a load of another shape than the loop's" answer A.Shape_mismatch
    (K.reduce p ~shape:[| 2; 4 |] ~axes:[| 1 |] sum
       ~dsts:[| on (A.create Rig.host D.Float32 [| 2 |]) |]
       [| plain_of x |]);
  let ints = on (A.of_array D.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]) in
  let mean = on (A.create Rig.host D.Int32 [| 2; 3 |]) in
  let var = on (A.create Rig.host D.Int32 [| 2; 3 |]) in
  equal ~msg:"a Moments scan of int32" answer A.Wrong_dtype
    (K.scan (identity (D.Any D.Int32)) ~axis:1 (Moments, 0)
       ~dsts:[| mean; var |] [| plain_of ints |]);
  let bools = on (A.create Rig.host D.Bool [| 2; 3 |]) in
  equal ~msg:"a Sum of bool" answer A.Wrong_dtype
    (K.reduce (identity (D.Any D.Bool)) ~shape:[| 2; 3 |] ~axes:[| 1 |]
       [| (Monoid Sum, 0) |]
       ~dsts:[| on (A.create Rig.host D.Bool [| 2 |]) |]
       [| plain_of bools |])

(* A padded load of more elements than a layout admits, or whose loop does
   not fit its destinations, is refused before the kernel copies it: padded
   to 2^50 elements, more than any host's memory, it answers Shape_mismatch
   and writes nothing. *)
let test_padded_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let on x = on b (A.Any x) in
  let x = A.expect D.Float32 (on (A.of_array D.Float32 [| 2 |] [| 1.; 2. |])) in
  let padded lo : Nx_kernel.load =
    Padded
      {
        x;
        fill = 0.;
        pad = { lo = [| lo |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] };
      }
  in
  let big = 1 lsl 50 in
  let huge = padded big and past = padded (L.max_numel - 1) in
  let f32 = D.Any D.Float32 in
  let id = identity f32 in
  let twice = P.v ~ins:[| f32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |] in
  let dst s = on (A.of_array D.Float32 s (Array.make (total s) 9.)) in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let refused msg call dsts =
    let before = Array.map (fun d -> A.to_array (A.expect D.Float32 (host d))) dsts in
    equal ~msg answer A.Shape_mismatch (call dsts);
    Array.iteri
      (fun i d ->
        equal ~msg:(msg ^ ": untouched") (array float_exact) before.(i)
          (A.to_array (A.expect D.Float32 (host d))))
      dsts
  in
  let sum = [| (Nx_kernel.Monoid Sum, 0) |] in
  let reduce p load dsts =
    K.reduce p ~shape:[| big + 2 |] ~axes:[| 0 |] sum ~dsts [| load |]
  in
  let scan p r load dsts = K.scan p ~axis:0 (r, 0) ~dsts [| load |] in
  List.iter
    (fun (name, p) ->
      refused (name ^ ": a reduction into another shape") (reduce p huge)
        [| dst [| 2 |] |];
      refused (name ^ ": a scan into another shape") (scan p (Monoid Sum) huge)
        [| dst [| 3 |] |];
      refused (name ^ ": a scan past max_numel") (scan p (Monoid Sum) past)
        [| dst [| 3 |] |];
      refused (name ^ ": a Moments scan into another shape")
        (scan p Moments huge)
        [| dst [| 3 |]; dst [| 3 |] |])
    [ ("identity", id); ("a program", twice) ]

(* Padded loads *)

(* What a padded load reads: the case of the array, built here, where
   Nx_kernel.Shape takes the padding and the destination; else the shape a
   reading of the record that checked nothing would give, its windows
   counted by truncating division, with the reduction's or scan's axes in
   it. *)
type loaded =
  | Reads of case
  | Refused of { scan : bool; monoid : monoid; axes : int array; read : int array }

(* A reduction or scan of a padded load: the operand, its padding and its
   fill's bits, and what it reads. *)
type padded = {
  x : A.any;
  view : view;
  fill : string;
  pad : Nx_kernel.pad;
  loaded : loaded;
}

let pp_padded ppf p =
  let pp_window ppf (w : M.window) =
    Format.fprintf ppf "(%d %d %d %d)" w.axis w.size w.step w.dilation
  in
  Format.fprintf ppf "lo %a hi %a interior %a windows %a of %a: " pp_ints
    (Iarray.to_array p.pad.lo) pp_ints (Iarray.to_array p.pad.hi) pp_ints
    (Iarray.to_array p.pad.interior)
    (Format.pp_print_list pp_window)
    (Iarray.to_list p.pad.windows) pp_ints (shape_of p.x);
  Format.fprintf ppf "(%s) " (String.concat ", " (view_names p.view));
  match p.loaded with
  | Reads c -> pp_case ppf c
  | Refused r ->
      Format.fprintf ppf "refused, %s %s along %a of %a"
        (if r.scan then "scan" else "reduce")
        (monoid_name r.monoid) pp_ints r.axes pp_ints r.read

(* The array [x] padded with the element [fill] (its bits) as [pad] says,
   C-contiguous, built element by element: along each axis, padded index
   [i] holds element [(i - lo) / (interior + 1)] where that divides and lies
   in the operand, the fill elsewhere; then the windows' view. *)
let pad_array (A.Any x) fill (pad : Nx_kernel.pad) =
  let dt = A.dtype x in
  let w = D.bits dt / 8 in
  let s = L.shape (A.layout x) in
  let r = Array.length s in
  let padded =
    Array.init r (fun i ->
        let d = s.(i) in
        Iarray.get pad.lo i + Iarray.get pad.hi i + d
        + if d > 0 then Iarray.get pad.interior i * (d - 1) else 0)
  in
  (* The operand's index at padded index [idx], if it lands on one. *)
  let source idx =
    let src = Array.make r 0 and inside = ref true in
    Array.iteri
      (fun i j ->
        let c = j - Iarray.get pad.lo i
        and step = Iarray.get pad.interior i + 1 in
        if c < 0 || c mod step <> 0 || c / step >= s.(i) then inside := false
        else src.(i) <- c / step)
      idx;
    if !inside then Some src else None
  in
  let a =
    if w = 0 then begin
      (* A sub-byte element is no byte of its own: element by element. *)
      let one = A.v dt (L.contiguous [||]) (Rig.Buffer.of_string fill) in
      let f = A.get one [||] in
      let a = A.create Rig.host dt padded in
      List.iter
        (fun idx ->
          A.set a idx
            (match source idx with Some src -> A.get x src | None -> f))
        (indices padded);
      a
    end
    else begin
      let bytes = A.to_array (Option.get (A.bitcast D.Uint8 (A.copy x))) in
      let b = Bytes.create (max 1 (w * total padded)) in
      List.iteri
        (fun k idx ->
          let flat src =
            let n = ref 0 in
            Array.iteri (fun i c -> n := (!n * s.(i)) + c) src;
            !n
          in
          let at = Option.map flat (source idx) in
          for q = 0 to w - 1 do
            Bytes.set b ((k * w) + q)
              (match at with
              | Some at -> Char.chr bytes.((at * w) + q)
              | None -> fill.[q])
          done)
        (indices padded);
      A.v dt (L.contiguous padded) (Rig.Buffer.of_string (Bytes.to_string b))
    end
  in
  if Iarray.length pad.windows = 0 then A.Any a
  else A.Any (Option.get (A.move (M.Window pad.windows) a))

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
  let* axis = option (int_range 0 (r - 1)) in
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

(* The padded extents of an operand of shape [s], below zero where a crop
   passes the axis. *)
let extents s (pad : Nx_kernel.pad) =
  Array.mapi
    (fun i d ->
      Iarray.get pad.lo i + Iarray.get pad.hi i + d
      + if d > 0 then Iarray.get pad.interior i * (d - 1) else 0)
    s

(* The shape a reading of [pad] that checked nothing would give: [s] for a
   padding of another rank, else the padded extents, each window's axis by
   its count from truncating division, then the windows' sizes. *)
let unchecked s (pad : Nx_kernel.pad) =
  if Iarray.length pad.lo <> Array.length s then s
  else
    let e = extents s pad in
    Iarray.iter
      (fun (w : M.window) ->
        e.(w.axis) <- ((e.(w.axis) - 1 - (w.dilation * (w.size - 1))) / w.step) + 1)
      pad.windows;
    Array.append e
        (Iarray.to_array (Iarray.map (fun (w : M.window) -> w.size) pad.windows))

(* [x] loaded padded with the element of bits [fill] as [pad] says. *)
let padded_load (A.Any x) fill pad =
  let dt = A.dtype x in
  let fill = A.get (A.v dt (L.contiguous [||]) (Rig.Buffer.of_string fill)) [||] in
  Nx_kernel.Padded { x; fill; pad }

(* The fill of bits [fill] of a float operand like [x], else None. *)
let float_fill (A.Any x) fill =
  let value (type s) (dt : (float, s) D.t) =
    Some (A.get (A.v dt (L.contiguous [||]) (Rig.Buffer.of_string fill)) [||])
  in
  match A.dtype x with
  | D.Float32 -> value D.Float32
  | D.Float64 -> value D.Float64
  | _ -> None

(* Whether Nx_kernel.Shape refuses the reduction or scan of [load]. *)
let refuses ~scan monoid axes load =
  let module Shape = Nx_kernel.Shape in
  let y = Shape.loaded [| load |] in
  Result.is_error
    (if scan then Result.bind y (Shape.scan ~axis:axes.(0))
     else Result.bind y
         (Shape.reduce ~axes:(Iarray.of_array axes) [| Monoid monoid |]))

(* The reduction or scan by [monoid] along [axes] of an operand of dtype
   [d] and shape [s] drawn from [seed], loaded as [pad] with [fill]. *)
let padded ?view ~scan d monoid s pad fill axes seed =
  let r = Array.length s in
  let plain =
    { perm = Array.init r Fun.id; stepped = false; reversed = None; broadcast = false }
  in
  let view = Option.value view ~default:plain in
  let x = operand d s view ~specials:20 seed in
  let loaded =
    if refuses ~scan monoid axes (padded_load x fill pad) then
      Refused { scan; monoid; axes; read = unchecked s pad }
    else Reads { scan; monoid; axes; x = pad_array x fill pad; views = [] }
  in
  { x; view; fill; pad; loaded }

(* The dtypes whose elements are no byte of their own, but float4_e2m1fn,
   which the reference does not compute in. *)
let sub_bytes = D.[ Any Int4; Any Uint4; Any Bit ]

let padded_case =
  Gen.with_pp pp_padded
    (let open Gen in
     let* scan = bool in
     let* d = of_list ~pp:pp_dtype (base @ sub_bytes) in
     let* monoid = of_list (List.filter (fun m -> accepts m d) monoids)
     in
     let* s = array ~size:(int_range 1 3) (int_range 0 6) in
     let* pad = pad_of s in
     (* One operand in three broadcast along axis 0, which half of them
        leave unpadded: drawn freely that is one case in a hundred. *)
     let* view = view_of (Array.length s) in
     let* broadcast = frequency [ (2, constant false); (1, constant true) ] in
     let* alone = bool in
     let view = { view with broadcast } in
     let pad =
       if not (broadcast && alone) then pad
       else
         let zero a = Iarray.mapi (fun i e -> if i = 0 then 0 else e) a in
         { pad with lo = zero pad.lo; hi = zero pad.hi; interior = zero pad.interior }
     in
     let* fill = of_list (fills d) in
     let* seed = int in
     let lr = Array.length (unchecked s pad) in
     let+ axes =
       if scan then map (fun a -> [| a |]) (int_range 0 (lr - 1))
       else
         let+ keep = array ~size:(constant lr) bool in
         Array.of_list (List.filter (fun i -> keep.(i)) (List.init lr Fun.id))
     in
     padded ~view ~scan d monoid s pad fill axes seed)

(* One padded load of each regime the law covers, float32 filled with
   one. *)
let padded_examples =
  let f32 = D.Any D.Float32 and one = "\000\000\128\063" in
  let pad ?(windows = [||]) lo hi interior =
      let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
      let interior = Iarray.of_array interior in
      { Nx_kernel.lo; hi; interior; windows = Iarray.of_array windows }
    in
  let w axis size step = { M.axis; size; step; dilation = 1 } in
  let reduce m s p axes = padded ~scan:false f32 m s p one axes 7 in
  [
    reduce Sum [| 4 |]
      (pad [| -1 |] [| 2 |] [| 1 |] ~windows:[| w 0 2 1 |])
      [| 1 |];
    (* A window of 4 over an axis of 3. *)
    reduce Sum [| 3 |] (pad [| 0 |] [| 0 |] [| 0 |] ~windows:[| w 0 4 2 |]) [| 1 |];
    (* Both axes cropped to -1. *)
    reduce Sum [| 2; 2 |] (pad [| -3; -3 |] [| 0; 0 |] [| 0; 0 |]) [| 0; 1 |];
    (* A padding of rank 2 over an operand of rank 1. *)
    reduce Sum [| 3 |] (pad [| 0; 5 |] [| 0; 0 |] [| 0; 0 |]) [| 0 |];
    (* A padding of more axes than an array has, each with a window. *)
    (let r = L.max_rank + 1 in
     reduce Sum [| 3 |]
       (pad (Array.make r 0) (Array.make r 0) (Array.make r 0)
          ~windows:(Array.init r (fun a -> w a 1 1)))
       [| 0 |]);
    reduce Max [| 0; 2 |] (pad [| 0; 0 |] [| 0; 0 |] [| 0; 0 |]) [| 0 |];
    (* Fill along a reduced axis. *)
    reduce Sum [| 4 |] (pad [| 1 |] [| 1 |] [| 0 |]) [| 0 |];
    (* Fill before axis 0 and after axis 1, axis 1 cropped before and
       windowed, its taps reduced. *)
    reduce Max [| 5; 6 |]
      (pad [| 1; -1 |] [| 0; 2 |] [| 0; 0 |] ~windows:[| w 1 3 2 |])
      [| 2 |];
    (* Broadcast along an axis the padding leaves alone. *)
    padded
      ~view:
        { perm = [| 0; 1 |]; stepped = false; reversed = None;
          broadcast = true }
      ~scan:false f32 Sum [| 4; 5 |]
      (pad [| 0; 1 |] [| 0; 1 |] [| 0; 0 |] ~windows:[| w 1 3 1 |])
      one [| 2 |] 7;
    (* Int4, a NaN and a -0 fill. *)
    padded ~scan:false (D.Any D.Int4) Sum [| 3; 5 |]
      (pad [| 0; 2 |] [| 0; 1 |] [| 0; 0 |] ~windows:[| w 1 2 1 |])
      "\136" [| 2 |] 7;
    padded ~scan:false f32 Max [| 3; 5 |]
      (pad [| 1; 0 |] [| 1; 0 |] [| 0; 0 |])
      "\052\018\192\127" [||] 7;
    padded ~scan:false f32 Sum [| 6 |]
      (pad [| 2 |] [| 1 |] [| 0 |] ~windows:[| w 0 2 1 |])
      "\000\000\000\128" [| 1 |] 7;
    (* Windows of one stepping 2, the first two or the last two in the fill
       alone. *)
    reduce Sum [| 4 |]
      (pad [| 3 |] [| 0 |] [| 0 |] ~windows:[| w 0 1 2 |])
      [| 1 |];
    reduce Sum [| 4 |]
      (pad [| 0 |] [| 3 |] [| 0 |] ~windows:[| w 0 1 2 |])
      [| 1 |];
    (* Each window of 3 over 2 elements padded by 1 reads the fill. *)
    reduce Sum [| 2 |]
      (pad [| 1 |] [| 1 |] [| 0 |] ~windows:[| w 0 3 1 |])
      [| 1 |];
    (* Through a reversed and a stepped view. *)
    padded
      ~view:
        { perm = [| 0; 1 |]; stepped = false; reversed = Some 1;
          broadcast = false }
      ~scan:false f32 Max [| 4; 5 |]
      (pad [| 1; 0 |] [| 2; 0 |] [| 0; 0 |])
      one [| 1 |] 7;
    padded
      ~view:
        { perm = [| 0; 1 |]; stepped = true; reversed = None;
          broadcast = false }
      ~scan:false f32 Sum [| 4; 5 |]
      (pad [| 1; 0 |] [| 2; 0 |] [| 0; 0 |])
      one [| 1 |] 7;
  ]

(* nx.cpu reduces and scans a padded load as the array it reads: the
   reference's bits over that array, built here. It refuses with
   Shape_mismatch a padding Nx_kernel.Shape refuses and an extreme of no term,
   here into a destination of the shape a reading that checked nothing
   would give; the sanitize profile shows that it reads nothing then. *)
let law_padded (b : Support.backend) p =
  let loads = [| padded_load p.x p.fill p.pad |] in
  let (A.Any x) = p.x in
  let s = shape_of p.x in
  let pad = p.pad in
  cover "windows" (pad.windows <> [||]);
  cover "cropped" (Iarray.exists (fun l -> l < 0) pad.lo);
  cover "interior" (Iarray.exists (fun i -> i > 0) pad.interior);
  let alone =
    p.view.broadcast && s.(0) > 1 && Iarray.get pad.lo 0 = 0
    && Iarray.get pad.hi 0 = 0 && Iarray.get pad.interior 0 = 0
  in
  cover "a broadcast axis the padding leaves alone" alone;
  match p.loaded with
  | Reads c ->
      let dst = A.Any (A.create Rig.host (A.dtype x) (result_shape c)) in
      cover "computed" true;
      (* The dtypes and the fills. *)
      let lo i = Iarray.get pad.lo i and hi i = Iarray.get pad.hi i in
      let axes = List.init (Array.length s) Fun.id in
      let filled = List.filter (fun i -> lo i > 0 || hi i > 0) axes in
      let nan, neg_zero =
        match float_fill p.x p.fill with
        | Some f -> (Float.is_nan f, f = 0. && Float.sign_bit f)
        | None -> (false, false)
      in
      cover "sub-byte" (D.bits (A.dtype x) < 8);
      cover "a NaN fill" nan;
      cover "a -0 fill" neg_zero;
      cover "a fill along a kept axis"
        (List.exists (fun i -> not (Array.mem i c.axes)) filled);
      cover "a fill along a reduced axis"
        ((not c.scan) && List.exists (fun i -> Array.mem i c.axes) filled);
      (* It gives no window along that axis. *)
      let e = extents s pad in
      cover "a window wider than its axis"
        (Iarray.exists
           (fun (w : M.window) -> e.(w.axis) < (w.dilation * (w.size - 1)) + 1)
           pad.windows);
      let p = identity (dtype_of p.x) in
      (match compute b.kernels ~scan:c.scan p c.monoid c.axes dst loads with
      | A.Done -> equal (list string) [] (differ (expected c) dst)
      | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r)
  | Refused r ->
      let same = Iarray.length pad.lo = Array.length s in
      let e = if same then extents s pad else s in
      cover "a padding of another rank" (not same);
      cover "an axis cropped below zero" (Array.exists (fun d -> d < 0) e);
      let extreme = r.monoid = Max || r.monoid = Min in
      let terms = Array.fold_left (fun n a -> n * r.read.(a)) 1 r.axes in
      cover "an extreme of no term" (same && (not r.scan) && extreme && terms = 0);
      let kept =
        if r.scan then r.read
        else
          Array.of_list
            (List.filteri (fun i _ -> not (Array.mem i r.axes)) (Array.to_list r.read))
      in
      let dst = A.Any (A.create Rig.host (A.dtype x) (Array.map (max 0) kept)) in
      let p = identity (dtype_of p.x) in
      (match compute b.kernels ~scan:r.scan p r.monoid r.axes dst loads with
      | A.Shape_mismatch -> ()
      | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a)

(* Moments scans *)

(* A Moments scan of [x] along [axis], [x] drawn through the views
   [views]. *)
type moments = { axis : int; x : A.any; views : string list }

let pp_moments ppf (c : moments) =
  let (A.Any x) = c.x in
  Format.fprintf ppf "scan Moments along %d of %a %a (%s)" c.axis D.pp
    (A.dtype x) L.pp (A.layout x)
    (String.concat ", " c.views)

(* Lengths around a multiple of nx.cpu's network tiles of 256 terms, from
   [lo] to [hi] tiles. *)
let tiled lo hi =
  let open Gen in
  let* k = int_range lo hi in
  let+ d = int_range (-2) 2 in
  (k * 256) + d

(* One in twelve operands is one axis of 2 to 32 tiles, the law's long
   case; one in twelve 2 or 3 slices of 12 to 24 tiles side by side, of
   finite terms, which a few threads run as one unit by tiles; one in
   twelve 65 to 72 slices of 1 to 8 tiles side by side, in units of 16
   slices, the last partial; and one in twelve slices of one term, its
   "one term" case, which [shape_gen] alone draws too rarely for every run
   to cover. Four in twelve are rows of 32-term tiles: 2 to 130 slices of
   32, 33 or more terms, scanned along the last axis, whose lanes the
   results hold a row apart, or along the first through a transposed
   operand, whose lanes it holds a row apart. *)
let moments_case =
  Gen.with_pp pp_moments
    (let open Gen in
     let* d = of_list ~pp:pp_dtype floats in
     let any =
       let* s = shape_gen in
       let s = if Array.length s = 0 then [| 3 |] else s in
       let+ axis = int_range 0 (Array.length s - 1) in
       (s, axis, false, false)
     in
     let long = map (fun n -> ([| n |], 0, false, false)) (tiled 2 32) in
     let few =
       let* n = tiled 12 24 in
       let+ k = int_range 2 3 in
       ([| n; k |], 0, false, true)
     in
     let wide =
       let* n = tiled 1 8 in
       let+ k = int_range 65 72 in
       ([| n; k |], 0, false, false)
     in
     let one = map (fun n -> ([| n; 1 |], 1, false, false)) (int_range 1 40) in
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
       if last then ([| slices; len |], 1, false, false)
       else ([| len; slices |], 0, true, false)
     in
     let* s, axis, transposed, finite =
       frequency
         [ (4, any); (1, long); (1, few); (1, wide); (1, one); (4, tiles) ]
     in
     let* v = view_of (Array.length s) in
     let v = if transposed then { v with perm = [| 1; 0 |] } else v in
     let* specials =
       if finite then constant 0
       else frequency [ (3, constant 0); (1, constant 300); (1, constant 4) ]
     in
     let+ seed = int in
     { axis; x = operand d s v ~specials seed; views = view_names v })

(* Slices of 15 to 25 tiles, one alone or three side by side, about 2^-20
   to 2^20 in magnitude, of both signs, whose joins of the tiles' states
   round differently in each order. *)
let moments_examples =
  let mixed d s seed =
    let n = Array.fold_left ( * ) 1 s in
    let r = Random.State.make [| seed |] in
    let term _ =
      let sign = if Random.State.bool r then 1. else -1. in
      let e = Random.State.int r 41 - 20 in
      sign *. Float.ldexp (1. +. Random.State.float r 1.) e
    in
    let xs = Array.init n term in
    match d with
    | D.Any D.Float32 -> A.Any (A.of_array D.Float32 s xs)
    | D.Any D.Float64 -> A.Any (A.of_array D.Float64 s xs)
    | _ -> invalid_arg "moments_examples"
  in
  [
    { axis = 0; x = mixed f32 [| 6400 |] 1; views = [] };
    { axis = 0; x = mixed f64 [| 5000 |] 2; views = [] };
    { axis = 0; x = mixed f32 [| 4000; 3 |] 3; views = [] };
  ]

(* [c] by [k] into fresh destinations, the mean's and the variance's terms;
   [None] if declined. *)
let run_moments k (c : moments) =
  let module K = (val k : Nx_kernel.S) in
  let (A.Any x) = c.x in
  let s = shape_of c.x in
  let mean = A.create Rig.host (A.dtype x) s in
  let var = A.create Rig.host (A.dtype x) s in
  let p = identity (dtype_of c.x) in
  match
    K.scan p ~axis:c.axis (Moments, 0)
      ~dsts:[| A.Any mean; A.Any var |]
      [| plain_of c.x |]
  with
  | A.Done -> Some (terms (A.Any mean), terms (A.Any var))
  | A.Declined -> None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* A result: some bits, or a value within a tolerance. *)
type expect = Is of want | Near of float * float

(* The most combinations a network scan's result [i] passes through:
   [max 0 (2 ⌈log₂ (i + 1)⌉ - 1)]. *)
let depth i =
  let rec log2 k p = if p >= k then 0 else 1 + log2 k (2 * p) in
  Float.of_int (max 0 ((2 * log2 (i + 1) 1) - 1))

(* The expected mean and variance at each index of [c], in C order: past a
   NaN term, NaNs; past an infinity, the infinities' sum and a NaN, the
   mean a NaN too past infinities of both signs; else the running mean and
   variance by float64 sums of the terms less the first and of their
   squares, within nx.cpu's bound at [x]'s unit roundoff [u], the depth [D]
   of the index and the range [R] and variance [V] of the terms so far, plus
   the sums' own bound in float64. *)
let expected_moments (c : moments) =
  let (A.Any x) = c.x in
  let u = if D.bits (A.dtype x) = 32 then 0x1p-24 else 0x1p-53 in
  let gamma k =
    Float.of_int k *. 0x1p-53 /. (1. -. (Float.of_int k *. 0x1p-53))
  in
  let s = shape_of c.x and a = c.axis in
  let ts = terms c.x in
  let r = Array.length s in
  let ks = Array.of_list (List.filter (( <> ) a) (List.init r Fun.id)) in
  let mean = Array.make (total s) (Is Some_nan) in
  let var = Array.make (total s) (Is Some_nan) in
  List.iter
    (fun o ->
      let idx = Array.make r 0 in
      Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
      let nan = ref false and pos = ref false and neg = ref false in
      let s1 = ref 0. and s2 = ref 0. and c = ref None in
      let lo = ref Float.infinity and hi = ref Float.neg_infinity in
      for j = 0 to s.(a) - 1 do
        idx.(a) <- j;
        let at = position s idx in
        let v = match ts.(at) with F (v, _) -> v | I _ -> assert false in
        if Float.is_nan v then nan := true
        else if v = Float.infinity then pos := true
        else if v = Float.neg_infinity then neg := true
        else begin
          let c = match !c with Some c -> c | None -> (c := Some v; v) in
          s1 := !s1 +. (v -. c);
          s2 := !s2 +. ((v -. c) *. (v -. c));
          lo := Float.min !lo v;
          hi := Float.max !hi v
        end;
        let n = Float.of_int (j + 1) in
        if !nan || (!pos && !neg) then begin
          mean.(at) <- Is Some_nan;
          var.(at) <- Is Some_nan
        end
        else if !pos || !neg then begin
          let inf = if !pos then Float.infinity else Float.neg_infinity in
          mean.(at) <- Near (inf, 0.);
          var.(at) <- Is Some_nan
        end
        else
          let c = Option.get !c and d = depth j and k = j + 1 in
          let m = !s1 /. n and r = !hi -. !lo in
          let v = Float.max 0. ((!s2 /. n) -. (m *. m)) in
          let slack = gamma (3 * k + 4) *. ((r *. r) +. r +. Float.abs c) in
          let tm =
            (((4. *. d) +. 1.) *. u *. r)
            +. (u *. Float.abs (c +. m))
            +. slack
          in
          let tv =
            (((7. *. d) +. 1.) *. u *. v)
            +. (((4. *. d *. d) -. (2. *. d)) *. u *. r *. Float.sqrt v)
            +. slack
          in
          mean.(at) <- Near (c +. m, tm);
          var.(at) <- Near (v, tv)
      done)
    (indices (Array.map (fun i -> s.(i)) ks));
  (mean, var)

(* The bits of each slice's mean and variance by nx_kernel.mli's network,
   replayed term by term as a stream: term t's state joins its level's
   waiting state while t's low bits are 1, and the result at the level it
   reaches is that state, or the next level's last result joined with it.
   Each operation rounds to [x]'s dtype. A slice with a term not finite
   takes nx_kinds.h's NaN rules, which this does not replay: its results
   are [None]. *)
let replayed_moments (c : moments) =
  let (A.Any x) = c.x in
  let f32 = D.bits (A.dtype x) = 32 in
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
  let s = shape_of c.x and a = c.axis in
  let ts = terms c.x in
  let r = Array.length s in
  let ks = Array.of_list (List.filter (( <> ) a) (List.init r Fun.id)) in
  let mean = Array.make (total s) None in
  let var = Array.make (total s) None in
  List.iter
    (fun o ->
      let idx = Array.make r 0 in
      Array.iteri (fun q i -> idx.(i) <- o.(q)) ks;
      let at j = idx.(a) <- j; position s idx in
      let xs =
        Array.init s.(a) (fun j ->
            match ts.(at j) with F (v, _) -> v | I _ -> assert false)
      in
      if Array.for_all Float.is_finite xs && Array.length xs > 0 then begin
        let c = xs.(0) in
        let held = Array.make 64 (0., 0., 0.) and got = Array.make 65 (0., 0., 0.) in
        Array.iteri
          (fun t v ->
            let st = ref (rnd (v -. c), 0., 1.) and k = ref 0 and j = ref t in
            while !j land 1 = 1 do
              st := join held.(!k) !st;
              incr k;
              j := !j lsr 1
            done;
            let res = if !j = 0 then !st else join got.(!k + 1) !st in
            held.(!k) <- !st;
            for h = 0 to !k do got.(h) <- res done;
            let mu, sv, _ = res in
            let n = Float.of_int (t + 1) in
            mean.(at t) <- Some (Is (Bits (bits (rnd (c +. mu)))));
            var.(at t) <- Some (Is (Bits (bits (rnd (sv /. n))))))
          xs
      end)
    (indices (Array.map (fun i -> s.(i)) ks));
  (mean, var)

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

(* A Moments scan has the bits of nx_kernel.mli's network, within nx.cpu's
   bound of the running mean and variance, its NaNs and infinities as
   stated, and the same bits on one thread and from a C-contiguous copy of
   its operand, a NaN as any NaN. *)
let law_moments (b : Support.backend) (c : moments) =
  let s = shape_of c.x in
  let len = s.(c.axis) in
  cover "no result" (total s = 0);
  cover "one term" (len = 1);
  cover "more than 64 slices" (len > 0 && total s / len > 64);
  cover "one slice past one tile, the last partial"
    (len > 256 && len mod 256 <> 0 && total s = len);
  cover "an axis around a multiple of a tile"
    (len >= 254 && (len + 2) mod 256 <= 4);
  cover "more than 64 slices side by side, past a tile"
    (len > 256 && total s / len > 64);
  cover "two or three slices side by side of 12 tiles or more"
    (len >= (12 * 256) - 2 && total s / len >= 2 && total s / len <= 3);
  cover "a tile's terms, or one more" (len = 32 || len = 33);
  (* Lanes a row apart go through rows of the scratch in tiles of 32 terms. *)
  let partial = len > 32 && len mod 32 <> 0 && total s / len > 1 in
  let rank = Array.length s in
  cover "a partial tile written through rows"
    (partial && rank >= 2 && c.axis = rank - 1);
  cover "a partial tile read through rows"
    (partial && rank = 2 && c.axis = 0
    && List.mem "permuted" c.views
    && not (List.mem "broadcast" c.views));
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  match run_moments b.kernels c with
  | None -> failf "nx.cpu declined a Moments scan"
  | Some (mean, var) ->
      let want_mean, want_var = expected_moments c in
      cover "a NaN" (Array.exists is_nan (terms c.x));
      cover "an infinity"
        (Array.exists
           (function Near (v, _) -> not (Float.is_finite v) | Is _ -> false)
           want_mean);
      equal ~msg:"mean" (list string) [] (misses want_mean mean);
      equal ~msg:"variance" (list string) [] (misses want_var var);
      let net_mean, net_var = replayed_moments c in
      let net want got =
        Array.mapi
          (fun i w -> Option.value w ~default:(Is (Bits (bits_of got.(i)))))
          want
      in
      cover "replayed" (Array.exists Option.is_some net_mean);
      equal ~msg:"the network's mean" (list string) []
        (misses (net net_mean mean) mean);
      equal ~msg:"the network's variance" (list string) []
        (misses (net net_var var) var);
      let exact ts = Array.map (fun t -> Is (want_of t)) ts in
      let (A.Any x) = c.x in
      let copy = { c with x = A.Any (A.copy x) } in
      List.iter
        (fun (msg, k, c) ->
          match run_moments k c with
          | None -> failf "%s declined" msg
          | Some (m, v) ->
              equal ~msg (list string) [] (misses (exact mean) m);
              equal ~msg (list string) [] (misses (exact var) v))
        [
          ("one thread", (module One : Nx_kernel.S), c);
          ("a copy", b.kernels, copy);
        ]

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

(* An [Arg], [Logsumexp] or [Moments] reduction of [x] along [axes], [x]
   drawn through the views [views] from values of the kind [values]. *)
type stat = {
  r : Nx_kernel.reduction;
  axes : int array;
  x : A.any;
  views : string list;
  values : string;
}

let reduction_name : Nx_kernel.reduction -> string = function
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"
  | Monoid m -> monoid_name m
  | Logsumexp -> "Logsumexp"
  | Moments -> "Moments"

let pp_stat ppf c =
  let (A.Any x) = c.x in
  Format.fprintf ppf "%s along %a of %a %a (%s, %s)" (reduction_name c.r)
    pp_ints c.axes D.pp (A.dtype x) L.pp (A.layout x) c.values
    (String.concat ", " c.views)

let is_float (D.Any dt) = D.is D.Float dt

(* The reduction [r] of [x] along [axes] drawn [make]: an Arg takes an
   element along each empty reduced axis, since one of no term is
   refused. *)
let stat_of r s axes v values make =
  let s =
    match r with
    | Nx_kernel.Arg _ ->
        Array.mapi (fun i e -> if Array.mem i axes then max 1 e else e) s
    | _ -> s
  in
  { r; axes; x = operand_with make s v; views = view_names v; values }

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

(* [c] as a reduction case of Sum, whose result's shape and outputs it
   shares. *)
let as_case c = { scan = false; monoid = Sum; axes = c.axes; x = c.x; views = [] }
let stat_shape c = result_shape (as_case c)

(* The destinations of [c]: the extreme and its int64 position for Arg, the
   mean and the variance for Moments, one for Logsumexp. *)
let stat_dsts c =
  let (A.Any x) = c.x in
  let s = stat_shape c in
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

(* A float's value. *)
let value = function F (x, _) -> x | I _ -> invalid_arg "value: an integer"

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

(* The extreme by [e] of the terms [ts] at [d], then its first position. *)
let arg_expect d (e : Nx_kernel.extreme) ts =
  let f, _ = op d (match e with Max -> Max | Min -> Min) in
  let best = Array.fold_left f ts.(0) ts in
  let same t = if is_nan best then is_nan t else bits_of t = bits_of best in
  let i = Option.get (Array.find_index same ts) in
  [| Is (want_of ts.(i)); Is (Bits (Int64.of_int i)) |]

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
let stat_expected c =
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
let law_stateful (b : Support.backend) c =
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
        want;
      let exact = Array.map (Array.map (fun t -> Is (want_of t))) got in
      let (A.Any x) = c.x in
      List.iter
        (fun (msg, k, c) ->
          match run_stat k c with
          | None -> failf "%s declined" msg
          | Some g ->
              Array.iteri (fun i w -> equal ~msg (list string) [] (misses w g.(i))) exact)
        [
          ("one thread", (module One : Nx_kernel.S), c);
          ("a copy", b.kernels, { c with x = A.Any (A.copy x) });
        ]

(* Arg and Logsumexp scans *)

(* A scan by Arg of a float, an integer or a boolean, or by Logsumexp of a
   float, along one axis: now and then a vector of 1 to 40 network tiles,
   past an Arg scan's chunk of 4096 from 16, or 65 to 72 slices of 1 to 8
   tiles, so that chunks' and tiles' carries run. *)
let stat_scan_case =
  Gen.with_pp pp_stat
    (let open Gen in
     let ints = List.filter (fun (D.Any dt) -> not (D.is D.Float dt)) base in
     let* d = of_list ~pp:pp_dtype (floats @ floats @ ints) in
     let* r =
       of_list
         (if is_float d then Nx_kernel.[ Arg Max; Arg Min; Logsumexp ]
          else Nx_kernel.[ Arg Max; Arg Min ])
     in
     let* s =
       frequency
         [
           (4, shape_gen);
           (1, map (fun n -> [| n |]) (tiled 1 40));
           ( 1,
             let* n = tiled 1 8 in
             let+ k = int_range 65 72 in
             [| n; k |] );
         ]
     in
     let s = if Array.length s = 0 then [| 3 |] else s in
     let rank = Array.length s in
     let* a = int_range 0 (rank - 1) in
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
     stat_of r s [| a |] v values.name make)

(* [c]'s scan by [k] into fresh destinations of [c.x]'s shape, their
   terms; [None] if declined. *)
let run_stat_scan k (c : stat) =
  let module K = (val k : Nx_kernel.S) in
  let (A.Any x) = c.x in
  let like dt = A.Any (A.create Rig.host dt (shape_of c.x)) in
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

(* The running Arg of a slice [ts] at [d]: each prefix's extreme and its
   first position. *)
let arg_scan d (e : Nx_kernel.extreme) ts =
  let f, _ = op d (match e with Max -> Max | Min -> Min) in
  let best = ref ts.(0) and at = ref 0 in
  Array.mapi
    (fun i t ->
      let g = f !best t in
      let same = if is_nan !best then is_nan g else bits_of g = bits_of !best in
      if not same then begin
        best := t;
        at := i
      end;
      [| Is (want_of !best); Is (Bits (Int64.of_int !at)) |])
    ts

(* nx_kernel.mli's bound of a Logsumexp scan's result [r] after [n] terms,
   at the unit roundoff [u]. *)
let lse_scan_bound ~n u r =
  (((7. *. depth (n - 1)) +. (6. *. log (Float.of_int n))) *. u)
  +. (u *. Float.abs r)

(* The running Logsumexp of a slice [ts] within nx_kernel.mli's bound at
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

(* The bits of a Logsumexp scan of a slice [ts], float32 where [f32] else
   float64, by nx_kernel.mli's network replayed term by term as a stream,
   as replayed_moments does, each operation as stated: a term [x] is the
   state [(x, e (x - c))], and states [a] then [b] join into
   [m = Maximum (ma, mb)] and [sa e (ma - c) + sb e (mb - c)], [c] being [m]
   where finite and [0] elsewhere and [e] nx_kinds.h's exp, [0] below the
   reduction's cut; the result is [m + log s], or [m] where [m] is not
   finite. Each operation rounds to the dtype. From the first NaN term on,
   the result is a NaN. *)
let lse_replay ~f32 ts =
  let rnd x = if f32 then round32 x else x in
  (* A kind over a vector, in one call. *)
  let kind name xs =
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
  in
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

(* The expected results of [c]'s scan in C order, by destination. *)
let stat_scan_expected (c : stat) =
  let (A.Any x) = c.x in
  let d = D.Any (A.dtype x) in
  let u = if D.bits (A.dtype x) = 32 then 0x1p-24 else 0x1p-53 in
  let slice ts =
    match c.r with
    | Arg e -> arg_scan d e ts
    | _ -> lse_scan u ts
  in
  let scanned = { (as_case c) with scan = true } in
  let k = match c.r with Arg _ -> 2 | _ -> 1 in
  Array.init k (fun j ->
      results scanned (terms c.x) ~fill:(Is Some_nan)
        ~reduce:(fun _ -> assert false)
        ~scan:(fun ts -> Array.map (fun e -> e.(j)) (slice ts)))

(* Arg scans are exact, Logsumexp scans within nx_kernel.mli's bound and
   the bits of its network replayed, NaNs and infinities as
   Nx_kernel.reduction states them, and both give the same bits on one
   thread and from a C-contiguous copy of the operand, a NaN as any NaN. *)
let law_stat_scan (b : Support.backend) (c : stat) =
  let s = shape_of c.x in
  let a = c.axes.(0) in
  let n = s.(a) in
  cover "an Arg scan's several chunks" (n > 4096);
  cover "an axis around a multiple of a tile" (n >= 254 && (n + 2) mod 256 <= 4);
  cover "one slice past one tile" (n > 256 && total s = n);
  cover "more than 64 slices side by side, past a tile"
    (n > 256 && total s / n > 64);
  let last = Array.length s - 1 in
  cover "slices side by side" (a < last && s.(last) > 1);
  List.iter (fun (k : values) -> cover k.name (c.values = k.name)) value_kinds;
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "permuted"; "stepped"; "reversed"; "broadcast" ];
  match run_stat_scan b.kernels c with
  | None -> failf "nx.cpu declined %s" (reduction_name c.r)
  | Some got ->
      let want = stat_scan_expected c in
      let wanted p = Array.exists (Array.exists p) want in
      let lse = c.r = Logsumexp in
      cover "a NaN" (lse && wanted (function Is Some_nan -> true | _ -> false));
      cover "a -∞ term"
        (lse
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
      (if lse then
         let (A.Any x) = c.x in
         let f32 = D.bits (A.dtype x) = 32 in
         let replay =
           results { (as_case c) with scan = true } (terms c.x)
             ~fill:(Is Some_nan) ~reduce:(fun _ -> assert false)
             ~scan:(lse_replay ~f32)
         in
         equal ~msg:"the replayed network" (list string) []
           (misses replay got.(0)));
      let exact = Array.map (Array.map (fun t -> Is (want_of t))) got in
      let (A.Any x) = c.x in
      List.iter
        (fun (msg, k, c) ->
          match run_stat_scan k c with
          | None -> failf "%s declined" msg
          | Some g ->
              Array.iteri (fun i w -> equal ~msg (list string) [] (misses w g.(i))) exact)
        [
          ("one thread", (module One : Nx_kernel.S), c);
          ("a copy", b.kernels, { c with x = A.Any (A.copy x) });
        ]

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

(* The two passes' arithmetic. Logsumexp and Moments reduce in two passes:
   drawn from these spreads of terms, they give the bits of that arithmetic
   run on the kernels' own Max, Sum and kinds. *)

let normal r =
  let u1 = Random.State.float r 1. and u2 = Random.State.float r 1. in
  Float.sqrt (-2. *. log (u1 +. 1e-300)) *. Float.cos (2. *. Float.pi *. u2)

let unit_ r = Random.State.float r 2. -. 1.

(* Spreads of terms: term [i] of [n] from the state [r]. *)
let spreads =
  [
    ("normal", fun r _ _ -> normal r);
    ("offset", fun r _ _ -> 1e4 +. (1e-2 *. unit_ r));
    ("wide", fun r _ _ -> 40. *. unit_ r);
    ("first-out", fun r i _ -> if i = 0 then 1e3 else unit_ r);
    ( "outliers",
      fun r _ _ ->
        let x = unit_ r in
        if Random.State.int r 50 = 0 then x *. Float.ldexp 1. (Random.State.int r 21)
        else x );
    ("ramp", fun r i n -> (-60. *. Float.of_int i /. Float.of_int n) +. (1e-3 *. unit_ r));
    ("lognormal", fun r _ _ -> Float.exp (3. *. normal r));
  ]


(* [r] of [outer] times [inner] outputs of [n] terms each: output [(o, i)]'s
   terms at [(o, t, i)] of an array of shape [[outer; n; inner]], [inner]
   apart beside their neighbours' where [inner > 1]. With [alone], [inner]
   is 1 and the array is the first column of one of shape [[outer; n; 2]]:
   each output's terms two apart, beside no output's. *)
type spread_case = {
  r : Nx_kernel.reduction;
  dt : D.any;
  spread : string;
  n : int;
  outer : int;
  inner : int;
  alone : bool;
  seed : int;
}

(* Long ramps, whose blocks' maxima differ: a sum rescaled block by block
   loses their bits. Many outputs of several blocks: along rows, side by
   side, and alone with their terms two apart. *)
let spread_examples =
  List.concat_map
    (fun dt ->
      List.map
        (fun seed ->
          { r = Nx_kernel.Logsumexp; dt; spread = "ramp"; n = 70_000; outer = 1;
            inner = 1; alone = false; seed })
        [ 0; 1; 2; 3 ])
    floats
  @ List.concat_map
      (fun (inner, alone) ->
        List.map
          (fun r ->
            { r; dt = f32; spread = "offset"; n = 1025; outer = 64; inner;
              alone; seed = 5 })
          Nx_kernel.[ Logsumexp; Moments ])
      [ (1, false); (2, false); (1, true) ]

let spread_case =
  Gen.with_pp
    (fun ppf c ->
      Format.fprintf ppf "%s of %dx%d outputs of %d %a terms, %s%s (seed %d)"
        (reduction_name c.r) c.outer c.inner c.n pp_dtype c.dt c.spread
        (if c.alone then ", alone" else "")
        c.seed)
    (let open Gen in
     let* r = of_list Nx_kernel.[ Logsumexp; Moments ] in
     let* dt = of_list floats in
     let* spread = of_list (List.map fst spreads) in
     let* outer, inner, alone, n =
       frequency
         [
           ( 3,
             map
               (fun n -> (1, 1, false, n))
               (of_list [ 1; 2; 17; 1023; 1025; 4097; 70_000 ]) );
           ( 1,
             let* outer = int_range 64 66 in
             let* inner, alone = of_list [ (1, false); (2, false); (1, true) ] in
             let+ n = of_list [ 17; 1025; 2100 ] in
             (outer, inner, alone, n) );
         ]
     in
     let+ seed = int in
     { r; dt; spread; n; outer; inner; alone; seed })

(* The float32 or float64 [dt], of [bits], array of shape [s] of [xs]. *)
let array_bits (type a b) (dt : (a, b) D.t) bits s xs =
  let b = Bytes.create (D.bytes dt (Array.length xs)) in
  Array.iteri
    (fun i x ->
      if bits = 32 then Bytes.set_int32_le b (4 * i) (Int32.bits_of_float x)
      else Bytes.set_int64_le b (8 * i) (Int64.bits_of_float x))
    xs;
  A.Any (A.v dt (L.contiguous s) (Rig.Buffer.of_string (Bytes.to_string b)))

(* [r] along [axes] of the array of shape [s] of [xs] by [k], each
   destination's terms. *)
let reduce_terms k (D.Any dt) r s axes xs =
  let x = array_bits dt (D.bits dt) s xs in
  match run_stat k { r; axes; x; views = []; values = "" } with
  | None -> failf "nx.cpu declined %s" (reduction_name r)
  | Some got -> got

(* [r] of the vector [xs] by [k]. *)
let reduce_vector k d r xs = reduce_terms k d r [| Array.length xs |] [| 0 |] xs

(* The kind [k] of each of [xs] at the float [dt], by the kernels. *)
let kind (D.Any dt) k xs =
  let (A.Any x) = array_bits dt (D.bits dt) [| Array.length xs |] xs in
  let d = A.create Rig.host (A.dtype x) [| Array.length xs |] in
  match Support.map1 (module Nx_cpu) (Unary k) ~dst:d x with
  | A.Done -> Array.map value (terms (A.Any d))
  | a -> failf "a kind answered %a" Nx_array_support.pp_answer a

(* Logsumexp: the maximum [m], [c] where it is finite or else [0], then
   [c + log] of the sum of [exp (x - c)], an exp below the cut taken as
   [0]. Moments: the sum over the count, the mean, then the sums [s1] of
   the deviations [d] from it and [s2] of their squares, and [(s2 - s1 (s1
   / n)) / n], +0 where that is negative, or [s2] where that is +inf. Each
   operation rounds to the dtype, and each output's sums are reductions of
   its terms alone. *)
let law_record (b : Support.backend) c =
  let (D.Any dt) = c.dt in
  let bits = D.bits dt in
  let gen = List.assoc c.spread spreads in
  let r = Random.State.make [| c.seed |] in
  let round x = if bits = 32 then Int32.float_of_bits (Int32.bits_of_float x) else x in
  let xs =
    Array.init (c.outer * c.n * c.inner) (fun k ->
        round (gen r (k / c.inner mod c.n) c.n))
  in
  cover "several blocks" (c.n > 1024);
  cover "groups of blocks" (c.n >= 70_000);
  cover "many outputs of several blocks" (c.outer >= 64 && c.n > 1024);
  cover "many outputs of several blocks, side by side"
    (c.outer >= 64 && c.n > 1024 && c.inner > 1);
  cover "many outputs of several blocks, each alone"
    (c.outer >= 64 && c.n > 1024 && c.alone);
  let got =
    if not c.alone then
      reduce_terms b.kernels c.dt c.r [| c.outer; c.n; c.inner |] [| 1 |] xs
    else
      (* Each term beside a NaN that no output reads. *)
      let ys =
        Array.init (2 * Array.length xs) (fun k ->
            if k mod 2 = 0 then xs.(k / 2) else Float.nan)
      in
      let (A.Any x) = array_bits dt bits [| c.outer; c.n; 2 |] ys in
      let whole e = { M.start = 0; count = e; step = 1 } in
      let first = { M.start = 0; count = 1; step = 1 } in
      let slice = M.Slice [| whole c.outer; whole c.n; first |] in
      let x = Option.get (A.move slice x) in
      match
        run_stat b.kernels
          { r = c.r; axes = [| 1 |]; x = A.Any x; views = []; values = "" }
      with
      | None -> failf "nx.cpu declined %s" (reduction_name c.r)
      | Some got -> got
  in
  let sum ys = value (reduce_vector b.kernels c.dt (Monoid Sum) ys).(0).(0) in
  let bits_at x =
    if bits = 32 then Int64.of_int32 (Int32.bits_of_float x) else Int64.bits_of_float x
  in
  let output o i =
    let at = (o * c.inner) + i in
    let xs = Array.init c.n (fun t -> xs.((((o * c.n) + t) * c.inner) + i)) in
    let msg s = Printf.sprintf "%s of output (%d, %d)" s o i in
    match c.r with
    | Logsumexp ->
        let m = value (reduce_vector b.kernels c.dt (Monoid Max) xs).(0).(0) in
        let c0 = if Float.is_finite m then m else 0. in
        let cut = if bits = 32 then -87. else -708. in
        let ys = Array.map (fun x -> round (x -. c0)) xs in
        let es = kind c.dt Exp (Array.map (Float.max cut) ys) in
        let es = Array.mapi (fun i e -> if ys.(i) < cut then 0. else e) es in
        let r = round (c0 +. (kind c.dt Log [| sum es |]).(0)) in
        equal ~msg:(msg "logsumexp") int64 (bits_at r) (bits_of got.(0).(at))
    | _ ->
        let n = round (Float.of_int c.n) in
        let mean = round (sum xs /. n) in
        let d x = round (x -. mean) in
        let s1 = sum (Array.map d xs) in
        let s2 = sum (Array.map (fun x -> round (d x *. d x)) xs) in
        let q = round (round (s2 -. round (s1 *. round (s1 /. n))) /. n) in
        let var = if s2 = Float.infinity then s2 else if q < 0. then 0. else q in
        equal ~msg:(msg "mean") int64 (bits_at mean) (bits_of got.(0).(at));
        equal ~msg:(msg "variance") int64 (bits_at var) (bits_of got.(1).(at))
  in
  for o = 0 to c.outer - 1 do
    for i = 0 to c.inner - 1 do
      output o i
    done
  done

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
  let fresh (A.Any x as a) = A.Any (A.create Rig.host (A.dtype x) (shape_of a)) in
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
          A.Any (A.create Rig.host D.Int64 (shape_of x))
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

(* nx.cpu refuses an Arg of no term, and a Moments scan into destinations
   of other shapes, float32 or narrow. *)
let test_stateful_cases (b : Support.backend) () =
  let module K = (val b.kernels) in
  let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = ) in
  let reduce r x dsts =
    K.reduce (identity (dtype_of x)) ~shape:(shape_of x) ~axes:[| 1 |]
      [| (r, 0) |] ~dsts [| plain_of x |]
  in
  let make dt s = A.Any (A.create Rig.host dt s) in
  let pos = make D.Int64 [| 2 |] in
  equal ~msg:"an Arg of no term" answer A.Shape_mismatch
    (reduce (Arg Min) (make D.Float32 [| 2; 0 |]) [| make D.Float32 [| 2 |]; pos |]);
  let scan dt mean var =
    let x = make dt [| 2; 3 |] in
    K.scan (identity (dtype_of x)) ~axis:1 (Moments, 0)
      ~dsts:[| make dt mean; make dt var |] [| plain_of x |]
  in
  equal ~msg:"a float32 Moments scan into a mean of another shape" answer
    A.Shape_mismatch
    (scan D.Float32 [| 3; 2 |] [| 2; 3 |]);
  equal ~msg:"a bfloat16 Moments scan into a variance of another shape" answer
    A.Shape_mismatch
    (scan D.Bfloat16 [| 2; 3 |] [| 3; 2 |])

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

(* A padded load's operand, which nx.cpu copies padded first, against the
   page, and the destination likewise. *)
let law_padded_guards (b : Support.backend) (p, side) =
  match p.loaded with
  | Refused _ -> reject ()
  | Reads c ->
      cover "windows" (p.pad.windows <> [||]);
      cover "cropped" (Iarray.exists (fun l -> l < 0) p.pad.lo);
      cover "at the page after" (side = Support.After);
      cover "at the page before" (side = Support.Before);
      let dt = dtype_of p.x in
      let call xs ds =
        compute b.kernels ~scan:c.scan (identity dt) c.monoid c.axes ds.(0)
          [| padded_load xs.(0) p.fill p.pad |]
      in
      equal Elements.answer A.Done
        (Elements.same_guarded ~any_nan:true side ~call [| p.x |]
           [| (dt, result_shape c) |])

(* Fused programs *)

(* A reduction of output [k] of [prog], a program over [loads] of one shape,
   by [red] along [axes], or with [scan] its scan along [axes.(0)]; [guard]
   puts each load against an inaccessible page on its side. *)
type fused = {
  name : string;
  shape : int array;
  scan : bool;
  prog : P.t;
  k : int;
  red : Nx_kernel.reduction;
  axes : int array;
  loads : A.any array;
  lanes : bool;
  guard : Support.side option;
  fviews : string list;
}

let reduction_label : Nx_kernel.reduction -> string = function
  | Monoid m -> monoid_name m
  | Logsumexp -> "Logsumexp"
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"
  | Moments -> "Moments"

let pp_fused ppf c =
  let shape = c.shape in
  Format.fprintf ppf "%s, %s %s of output %d along %a of %a%s (%s)" c.name
    (if c.scan then "scan" else "reduction")
    (reduction_label c.red) c.k pp_ints c.axes pp_ints shape
    (match c.guard with
    | None -> ""
    | Some Support.Before -> ", guarded before"
    | Some Support.After -> ", guarded after")
    (String.concat ", " c.fviews)

(* The programs, over loads of the float dtype [d]: each one's name, its
   loads, whether its second load is a row of lanes (broadcast along the
   reduced axes), its nodes, its outputs, and the output the reduction
   reads. [rank] bounds its coordinates. *)
let templates (D.Any dt as d) rank =
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

(* The reductions an output of [dt] takes, or with [scan] its scans. *)
let reductions_of ~scan (D.Any dt) : Nx_kernel.reduction list =
  match D.kind dt with
  | D.Float when scan ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Logsumexp;
        Arg Max; Arg Min; Moments ]
  | D.Float ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Logsumexp;
        Arg Max; Arg Min; Moments ]
  | D.Complex when scan ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Arg Max; Arg Min ]
  | D.Complex ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Arg Max; Arg Min ]
  | D.Boolean -> [ Monoid Max; Monoid Min ]
  | _ when scan ->
      [ Monoid Sum; Monoid Prod; Monoid Max; Monoid Min; Arg Max ]
  | _ -> [ Monoid Sum; Monoid Max; Monoid Min; Arg Max ]

let fused_shapes =
  let open Gen in
  let small = array ~size:(int_range 1 3) (int_range 1 5) in
  let rows =
    let* r = int_range 1 40 in
    let+ k = of_list ~pp:Format.pp_print_int [ 9; 17; 64; 1100; 5000 ] in
    [| r; k |]
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
    [ (3, small); (3, rows); (2, columns); (1, long); (5, apart);
      (4, constant ~pp:pp_ints [||]); (4, empty) ]

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
  let* s = fused_shapes in
  let r = Array.length s in
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
  let ts = templates d r in
  let* t = int_range 0 (List.length ts - 1) in
  let name, ins, lanes, nodes, outs, k = List.nth ts t in
  let prog = P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
      ~outs:(Iarray.of_array outs) in
  let o = P.dtype prog outs.(k) in
  let* red = of_list (reductions_of ~scan o) in
  let* v = view_of r in
  let* lv = view_of r in
  let* specials = frequency [ (3, constant 0); (1, constant 20) ] in
  let* guard =
    frequency
      [ (3, constant None); (1, constant (Some Support.Before));
        (1, constant (Some Support.After)) ]
  in
  let+ seed = int in
  let load i =
    if i = 1 && lanes then
      let row = Array.mapi (fun j e -> if Array.mem j axes then 1 else e) s in
      let (A.Any x) = operand d row lv ~specials (seed + i) in
      A.Any (Option.get (A.move (M.Broadcast (Iarray.of_array s)) x))
    else operand d s (if i = 0 then v else lv) ~specials (seed + i)
  in
  {
    name; shape = s; scan; prog; k; red; axes; lanes; guard;
    loads = Array.init (Array.length ins) load;
    fviews = view_names v;
  }

(* The case of template [i] over [d] of shape [s], plain or with
   [broadcast] its first load broadcast along its first axis, or with
   [stepped] read with a step of two along its last, by [red] along
   [axes], or with [scan] scanned. *)
let fused_example ?(d = D.Any D.Float32) ?(scan = false) ?(broadcast = false)
    ?(stepped = false) i red s axes =
  let r = Array.length s in
  let name, ins, lanes, nodes, outs, k = List.nth (templates d r) i in
  let plain =
    { perm = Array.init r Fun.id; stepped = false; reversed = None;
      broadcast = false }
  in
  let load j =
    if j = 1 && lanes then
      let row = Array.mapi (fun a e -> if Array.mem a axes then 1 else e) s in
      let (A.Any x) = operand d row plain ~specials:0 (7 + j) in
      A.Any (Option.get (A.move (M.Broadcast (Iarray.of_array s)) x))
    else
      operand d s
        { plain with broadcast = broadcast && j = 0; stepped = stepped && j = 0 }
        ~specials:0 (7 + j)
  in
  {
    name; shape = s; scan; k; red; axes; lanes; guard = None;
    fviews = (if stepped then [ "stepped" ] else []);
    prog =
      P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
        ~outs:(Iarray.of_array outs);
    loads = Array.init (Array.length ins) load;
  }

(* A constant's sum over a long axis and a broadcast load's own sum, whose
   terms map.c holds once for a whole row; complex products, whose lanes
   and scans start from their first terms:
   one output's block, rows of outputs side by side, two reduced axes split
   by a kept one, several blocks, and scans past a chunk; float32 scans
   past a chunk: of one slice, in quads of chunks, as Moments, and of a row
   of slices side by side; a reduction of several blocks; Logsumexp down
   columns, several outputs' blocks computed at once; and outputs
   narrower than their carrier: bfloat16's Arg, Moments scan and scan in
   quads of chunks, and int4's scan so; Logsumexp scans by tiles, of one
   slice and of three side by side, an Arg scan past a chunk, and
   bfloat16's Arg scan, its extreme the program's term; Moments of many
   outputs of several blocks, each staged once for both passes, plain and
   stepped. *)
let fused_examples =
  let c = D.Any D.Complex64 in
  [
    fused_example ~d:c 2 (Monoid Prod) [| 3; 40 |] [| 1 |];
    fused_example ~d:c 2 (Monoid Prod) [| 40; 3 |] [| 0 |];
    fused_example ~d:c 2 (Monoid Prod) [| 3; 4; 5 |] [| 0; 2 |];
    fused_example ~d:c 0 (Monoid Prod) [| 2; 2000 |] [| 1 |];
    fused_example ~d:c ~scan:true 2 (Monoid Prod) [| 9000 |] [| 0 |];
    fused_example ~d:c ~scan:true 2 (Monoid Prod) [| 3; 5000 |] [| 1 |];
    fused_example ~d:c ~scan:true 2 (Monoid Prod) [| 5000; 3 |] [| 0 |];
    fused_example 7 (Monoid Sum) [| 100000 |] [| 0 |];
    fused_example ~broadcast:true 8 (Monoid Sum) [| 100000 |] [| 0 |];
    fused_example ~scan:true 0 (Monoid Sum) [| 9000 |] [| 0 |];
    fused_example ~scan:true 1 Moments [| 3; 9000 |] [| 1 |];
    fused_example ~scan:true 1 Moments [| 9000 |] [| 0 |];
    fused_example ~scan:true 3 (Monoid Max) [| 9000; 3 |] [| 0 |];
    fused_example ~scan:true 2 (Monoid Prod) [| 20000 |] [| 0 |];
    fused_example 2 (Monoid Sum) [| 5000 |] [| 0 |];
    fused_example 0 Logsumexp [| 300; 20 |] [| 0 |];
    fused_example 9 (Arg Max) [| 3; 40 |] [| 1 |];
    fused_example ~scan:true 9 Moments [| 3; 100 |] [| 1 |];
    fused_example ~scan:true 9 (Monoid Sum) [| 20000 |] [| 0 |];
    fused_example ~scan:true 10 (Monoid Max) [| 20000 |] [| 0 |];
    fused_example ~scan:true 0 Logsumexp [| 20000 |] [| 0 |];
    fused_example ~scan:true 1 Logsumexp [| 9000; 3 |] [| 0 |];
    fused_example ~scan:true 3 (Arg Min) [| 20000 |] [| 0 |];
    fused_example ~scan:true 9 (Arg Max) [| 3; 40 |] [| 1 |];
    fused_example 1 Moments [| 64; 1100 |] [| 1 |];
    fused_example ~stepped:true 1 Moments [| 64; 1100 |] [| 1 |];
  ]

(* A case of [fused_any], or a float32 program's monoid along
   [wide_rows] often enough that every run reaches their calls. *)
let fused_case =
  let open Gen in
  let rows =
    let ts = templates (D.Any D.Float32) 2 in
    let* i = int_range 0 (List.length ts - 1) in
    let _, ins, _, nodes, outs, k = List.nth ts i in
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
    fused_example i red s [| 1 |]
  in
  with_pp pp_fused (frequency [ (6, fused_any); (1, rows) ])

(* The destinations of [c]'s reduction, fresh, of output [c.k]'s dtype. *)
let fused_dsts (b : Support.backend) c =
  let s = c.shape in
  let y =
    if c.scan then s
    else
      Array.of_list
        (List.filteri (fun i _ -> not (Array.mem i c.axes)) (Array.to_list s))
  in
  let (D.Any dt) = P.dtype c.prog (Iarray.get c.prog.outs c.k) in
  let make () = A.Any (A.create b.device dt y) in
  match c.red with
  | Monoid _ | Logsumexp -> [| make () |]
  | Moments -> [| make (); make () |]
  | Arg _ -> [| make (); A.Any (A.create b.device D.Int64 y) |]

(* Each destination's bytes. *)
let bytes_of (A.Any x) =
  let n = total (L.shape (A.layout x)) in
  let w = max 1 (D.bits (A.dtype x) / 8) in
  let u = A.copy x in
  match A.bitcast D.Uint8 u with
  | Some b -> Array.to_list (A.to_array b)
  | None -> List.init (n * w) (fun _ -> 0)

(* [c] fused, its loads guarded where it says, against [c]'s program mapped
   into fresh arrays and reduced as their outputs' identity, a NaN as any
   NaN. *)
let law_fused (b : Support.backend) c =
  let module K = (val b.kernels) in
  let s = c.shape in
  let terms = Array.fold_left (fun n a -> n * s.(a)) 1 c.axes in
  classify c.name true;
  cover "a loop of no load" (c.loads = [||]);
  let complex =
    let (D.Any dt) = P.dtype c.prog (Iarray.get c.prog.outs c.k) in
    D.is D.Complex dt
  in
  cover "a complex output" complex;
  cover "a complex product" (complex && c.red = Monoid Prod);
  cover "a narrow output" (List.mem c.name
    [ "an operand cast into bfloat16"; "an operand cast into int4" ]);
  cover (reduction_label c.red) true;
  cover "a scan" c.scan;
  cover "a Moments scan" (c.scan && c.red = Moments);
  cover "a stateful scan"
    (c.scan && (c.red = Logsumexp || c.red = Arg Max || c.red = Arg Min));
  cover "a scan past a chunk" (c.scan && s.(c.axes.(0)) > 4096);
  cover "a row of lanes" c.lanes;
  cover "a lane or fewer terms an output" (terms <= 16);
  cover "several blocks of terms" (terms > 1024);
  let r = Array.length s in
  let side = r > 1 && not (Array.mem (r - 1) c.axes) in
  (* A monoid's full blocks of one output's computed terms, several
     computed at once: one output's terms a unit, none side by side. *)
  let monoid = match c.red with Monoid _ -> true | _ -> false in
  let outputs = if terms = 0 then 0 else total s / terms in
  covers_blocks
    (monoid && (not complex) && (not c.scan) && not side)
    (carrier (P.dtype c.prog (Iarray.get c.prog.outs c.k)))
    ~outputs terms;
  let split =
    Array.exists (fun a -> a + 1 < r && not (Array.mem (a + 1) c.axes)
                           && Array.exists (fun b -> b > a + 1) c.axes) c.axes
  in
  cover "outputs side by side" side;
  cover "reduced axes split by a kept one" split;
  cover "reduced axes split, outputs side by side, more than a lane of terms"
    (split && side && terms > 16);
  cover "guarded" (c.guard <> None);
  cover "rank 0" (r = 0);
  cover "no element" (total s = 0);
  let plain = Array.map (fun (A.Any x) -> Nx_kernel.Plain x) in
  let outs =
    Array.map
      (fun o ->
        let (D.Any dt) = P.dtype c.prog o in
        A.Any (A.create b.device dt s))
      (Iarray.to_array c.prog.outs)
  in
  (match K.map c.prog ~dsts:outs (plain c.loads) with
  | A.Done -> ()
  | r -> failf "the map answered %a" Nx_array_support.pp_answer r);
  let (A.Any y) = outs.(c.k) in
  let want = fused_dsts b c in
  let compute_in (module M : Nx_kernel.S) p k ~dsts loads =
    if c.scan then M.scan p ~axis:c.axes.(0) (c.red, k) ~dsts loads
    else
      M.reduce p ~shape:c.shape ~axes:(Iarray.of_array c.axes) [| (c.red, k) |]
        ~dsts loads
  in
  let compute = compute_in b.kernels in
  (* An extreme of no term refuses alike. *)
  let expected =
    compute (identity (D.Any (A.dtype y))) 0 ~dsts:want [| Nx_kernel.Plain y |]
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
                      (Iarray.of_array (Array.mapi
                         (fun i e ->
                           { M.start = 0; count = (if st.(i) = 0 then min e 1 else e);
                             step = 1 })
                         (L.shape l))))
                   y)
            in
            A.blit ~src:(once x) ~dst:(once g);
            (A.Any g, free))
      c.loads
  in
  let got = fused_dsts b c in
  Fun.protect
    ~finally:(fun () -> Array.iter (fun (_, free) -> free ()) guarded)
    (fun () ->
      let answer = compute c.prog c.k ~dsts:got (plain (Array.map fst guarded)) in
      equal ~msg:"answer" Elements.answer expected answer;
      if answer = A.Done then begin
        (* One thread gives the job's bits. *)
        let one = fused_dsts b c in
        equal ~msg:"one thread's answer" Elements.answer A.Done
          (compute_in (module One) c.prog c.k ~dsts:one (plain c.loads));
        Array.iteri
          (fun i w ->
            let es = Elements.elements (host w) in
            let same = array (Elements.same (dtype_of w)) in
            equal ~msg:(Printf.sprintf "destination %d" i) same es
              (Elements.elements (host got.(i)));
            equal ~msg:(Printf.sprintf "destination %d, one thread" i)
              same es (Elements.elements (host one.(i))))
          want
      end)

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
  check "an Arg scan" numbers [| h; i64 |] (scan (Arg Max))

(* Several reductions of one loop: [c]'s and [red] of output [k], with
   [c]'s axes, one call. *)
type several = { first : fused; red2 : Nx_kernel.reduction; k2 : int }

let pp_several ppf s =
  Format.fprintf ppf "%a, then %s of output %d" pp_fused s.first
    (reduction_label s.red2) s.k2

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
  let loads = Array.map (fun (A.Any x) -> Nx_kernel.Plain x) c.loads in
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

(* Two reductions whose second destination has another shape refuse,
   writing neither. *)
let test_several_refuse (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let p = identity (D.Any D.Float32) in
  let y1 = A.of_array D.Float32 [| 2 |] [| 7.; 7. |] in
  let y2 = A.create b.device D.Float32 [| 3 |] in
  let answer =
    K.reduce p ~shape:[| 2; 3 |] ~axes:[| 1 |]
      [| (Monoid Sum, 0); (Monoid Max, 0) |]
      ~dsts:[| A.Any y1; A.Any y2 |]
      [| Nx_kernel.Plain x |]
  in
  equal ~msg:"answer" bool true (answer = A.Shape_mismatch);
  equal ~msg:"the first destination" (array float_exact) [| 7.; 7. |]
    (A.to_array y1)

(* Sums of products *)

let plain_view r =
  { perm = Array.init r Fun.id; stepped = false; reversed = None; broadcast = false }

(* A Sum along [paxes] of the products of two loads over the loop [pshape],
   or of one load by itself where [same]: each load of [ldt], read at the
   product's float dtype [pdt], through a cast where they differ. [loads]
   are what the reduction reads and [reads] the arrays they read, the
   first through a padded load where [pad] holds its padding. *)
type product = {
  ldt : D.any;
  pdt : D.any;
  pshape : int array;
  paxes : int array;
  same : bool;
  pad : Nx_kernel.pad option;
  loads : Nx_kernel.load array;
  reads : A.any array;
  pnames : string list;
}

let pp_product ppf c =
  Format.fprintf ppf "Sum along %a of %a of %s products of %a at %a%s (%s)"
    pp_ints c.paxes pp_ints c.pshape
    (if c.same then "squared," else "two loads'")
    pp_dtype c.ldt pp_dtype c.pdt
    (if c.pad <> None then ", the first padded" else "")
    (String.concat ", " c.pnames)

(* The product of two loads of [ldt] at [pdt], or of one by itself. *)
let product_prog ~same (D.Any l as ldt) (D.Any p as pdt) =
  let ins = if same then [| ldt |] else [| ldt; ldt |] in
  let nodes =
    match (same, D.equal_witness l p) with
    | true, Some _ -> P.[| In 0; Op2 (Binary Mul, 0, 0) |]
    | true, None -> P.[| In 0; Op1 (Cast, pdt, 0); Op2 (Binary Mul, 1, 1) |]
    | false, Some _ -> P.[| In 0; In 1; Op2 (Binary Mul, 0, 1) |]
    | false, None ->
        P.[| In 0; In 1; Op1 (Cast, pdt, 0); Op1 (Cast, pdt, 1);
             Op2 (Binary Mul, 2, 3) |]
  in
  P.v ~ins:(Iarray.of_array ins) (Iarray.of_array nodes)
    ~outs:(Iarray.of_array [| Array.length nodes - 1 |])

(* [n] elements of [ldt], of float32's values and specials cast into a
   narrow float. *)
let product_elements (D.Any dt as d) n ~specials seed =
  match dt with
  | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
  | D.Float4_e2m1fn ->
      cast_into d (A.Any (elements D.Float32 n ~specials seed))
  | _ -> A.Any (elements dt n ~specials seed)

let product_operand d s v ~specials seed =
  operand_with (fun n -> product_elements d n ~specials seed) s v

(* The loads' and the product's dtypes: each float at itself, the narrow
   floats and float64 cast into float32, and float32 into float64. *)
let product_dtypes =
  let open Gen in
  let pair a b = constant ~pp:(fun ppf _ -> pp_dtype ppf a) (a, b) in
  frequency
    [
      (3, pair f32 f32); (3, pair f64 f64);
      (2, pair (D.Any D.Bfloat16) f32); (2, pair (D.Any D.Float16) f32);
      (1, pair (D.Any D.Float8_e4m3fn) f32);
      (1, pair (D.Any D.Float8_e5m2) f32);
      (1, pair (D.Any D.Float4_e2m1fn) f32); (1, pair f64 f32);
      (1, pair f32 f64);
    ]

(* [n] terms an output, from 1 to 17 and past it, along one reduced axis
   or two; kept axes of a row of up to 130 outputs and up to two more; all
   in any order: the reduced axes and their positions. *)
let product_loop =
  let open Gen in
  let* n = frequency [ (8, int_range 1 16); (1, constant 17); (1, int_range 18 40) ] in
  let* split = bool in
  let divisors = List.filter (fun a -> n mod a = 0) (List.init n (fun i -> i + 1)) in
  let* a = of_list ~pp:Format.pp_print_int divisors in
  let red = if split && a > 1 && a < n then [| a; n / a |] else [| n |] in
  let* row = frequency [ (3, int_range 1 20); (1, of_list ~pp:Format.pp_print_int [ 64; 130 ]) ] in
  let* outer = array ~size:(int_range 0 2) (int_range 1 4) in
  let kept = Array.append outer [| row |] in
  let all = Array.append (Array.map (fun e -> (e, false)) kept) (Array.map (fun e -> (e, true)) red) in
  let r = Array.length all in
  let+ order =
    frequency
      [ (1, constant (List.init r Fun.id));
        (1, permutation ~pp:Format.pp_print_int (List.init r Fun.id)) ]
  in
  let placed = Array.of_list (List.map (fun i -> all.(i)) order) in
  let shape = Array.map fst placed in
  let axes =
    Array.of_list
      (List.filter (fun i -> snd placed.(i)) (List.init r Fun.id))
  in
  (shape, axes)

(* A load broadcast along the kept axes, as a convolution's kernel is,
   from a row of its reduced axes' elements. *)
let broadcast_factor d shape axes v ~specials seed =
  let row = Array.mapi (fun i e -> if Array.mem i axes then e else 1) shape in
  let (A.Any x) = product_operand d row v ~specials seed in
  A.Any (Option.get (A.move (M.Broadcast (Iarray.of_array shape)) x))

(* A case over the loop [shape] along [axes], its loads drawn from [seed];
   the second factor broadcast where [broadcast], and the product's first
   where [first]. *)
let product_of ?(first = false) ~same ~broadcast ldt pdt shape axes v lv
    ~specials seed =
  let x = product_operand ldt shape v ~specials seed in
  let y =
    if broadcast then broadcast_factor ldt shape axes lv ~specials (seed + 1)
    else product_operand ldt shape lv ~specials (seed + 1)
  in
  let reads =
    if same then [| x |] else if first then [| y; x |] else [| x; y |]
  in
  {
    ldt; pdt; pshape = shape; paxes = axes; same; pad = None;
    loads = Array.map plain_of reads; reads;
    pnames =
      view_names v
      @ (match (same, broadcast, first) with
        | true, _, _ | false, false, _ -> []
        | false, true, true -> [ "a broadcast first factor" ]
        | false, true, false -> [ "a broadcast factor" ])
      @ (if specials > 0 then [ "specials" ] else []);
  }

(* Sums along axis 1 of [160; 2; 64], more outputs than a thread's unit
   of a sum of products, of a broadcast first factor by one with a NaN at
   [0; 0; 5]: the NaN's output is a NaN while the units after it on the
   thread read the broadcast factor's terms. *)
let nan_before_units =
  let shape = [| 160; 2; 64 |] in
  let n = Array.fold_left ( * ) 1 shape in
  let r = Random.State.make [| 17 |] in
  let bits = Bytes.create (4 * n) in
  for i = 0 to n - 1 do
    Bytes.set_int32_le bits (4 * i) (Int32.bits_of_float (full r 23))
  done;
  Bytes.set_int32_le bits (4 * 5) (Int32.bits_of_float Float.nan);
  let b =
    A.Any
      (A.v D.Float32 (L.contiguous shape)
         (Rig.Buffer.of_string (Bytes.to_string bits)))
  in
  let a = broadcast_factor f32 shape [| 1 |] (plain_view 3) ~specials:0 19 in
  {
    ldt = f32; pdt = f32; pshape = shape; paxes = [| 1 |]; same = false;
    pad = None; loads = [| plain_of a; plain_of b |]; reads = [| a; b |];
    pnames = [ "a broadcast first factor"; "specials" ];
  }

(* A depthwise convolution's loop: [x], [[b; h; w; c]], padded by [lo] and
   [hi] along its spatial axes, windows of [k1] by [k2] taps of [step] and
   [dilation] appending the taps, times a kernel [[c; k1; k2]] broadcast
   over the batch and the windows, summed over the taps. *)
let depthwise ldt pdt ~b ~h ~w ~c ~lo ~hi ~k1 ~k2 ~step ~dilation ~fill ~specials
    seed =
  let x = product_operand ldt [| b; h; w; c |] (plain_view 4) ~specials seed in
  let sp v = Iarray.of_array [| 0; v; v; 0 |] in
  let win axis size : M.window = { axis; size; step; dilation } in
  let pad : Nx_kernel.pad =
    { lo = sp lo; hi = sp hi; interior = sp 0;
      windows = Iarray.of_list [ win 1 k1; win 2 k2 ] }
  in
  let loaded = unchecked [| b; h; w; c |] pad in
  let axes = [| 4; 5 |] in
  let y = broadcast_factor ldt loaded axes (plain_view 6) ~specials (seed + 1) in
  let (A.Any xa) = x in
  let fill_value =
    A.get (A.v (A.dtype xa) (L.contiguous [||]) (Rig.Buffer.of_string fill)) [||]
  in
  {
    ldt; pdt; pshape = loaded; paxes = axes; same = false; pad = Some pad;
    loads = [| Nx_kernel.Padded { x = xa; fill = fill_value; pad }; plain_of y |];
    reads = [| pad_array x fill pad; y |];
    pnames =
      [ "padded" ]
      @ (if lo <> hi then [ "padded unevenly" ] else [])
      @ (if (lo = 0) <> (hi = 0) then [ "a side unpadded" ] else [])
      @ (match float_fill x fill with
        | Some f when Float.is_nan f -> [ "a NaN fill" ]
        | Some f when f = 0. && Float.sign_bit f -> [ "a -0 fill" ]
        | _ -> [])
      @ if specials > 0 then [ "specials" ] else [];
  }

(* Zero's bits, and a NaN's and -0's for float32 and float64. *)
let product_fills (D.Any dt) =
  let w = D.bits dt / 8 in
  let zero = String.make (max 1 w) '\000' in
  let neg_zero = String.make (max 0 (w - 1)) '\000' ^ "\128" in
  match dt with
  | D.Float32 -> [ zero; "\052\018\192\127"; neg_zero ]
  | D.Float64 -> [ zero; "\035\001\000\000\000\000\248\127"; neg_zero ]
  | _ -> [ zero ]

let product_case =
  let open Gen in
  with_pp pp_product
  @@ let* ldt, pdt = product_dtypes in
     let* specials = frequency [ (3, constant 0); (1, constant 8) ] in
     let* seed = int in
     let* padded = frequency [ (3, constant false); (1, constant true) ] in
     if padded then
       let* b = int_range 1 2 in
       let* c = frequency [ (3, int_range 1 20); (1, constant 64) ] in
       let* lo = int_range 0 2 in
       let* hi = int_range 0 2 in
       let* k1 = int_range 1 4 in
       let* k2 = int_range 1 4 in
       let* step = int_range 1 2 in
       let* dilation = int_range 1 2 in
       (* Spatial axes at least a window long, so that each has one. *)
       let* h = int_range 1 7 in
       let* w = int_range 1 7 in
       let h = h + (dilation * (k1 - 1)) and w = w + (dilation * (k2 - 1)) in
       let+ fill = of_list (product_fills ldt) in
       depthwise ldt pdt ~b ~h ~w ~c ~lo ~hi ~k1 ~k2 ~step ~dilation ~fill
         ~specials seed
     else
       let* shape, axes = product_loop in
       let r = Array.length shape in
       let* v = view_of r in
       let* lv = view_of r in
       let* same = frequency [ (5, constant false); (1, constant true) ] in
       let* broadcast = bool in
       let+ first = bool in
       product_of ~first ~same ~broadcast ldt pdt shape axes v lv ~specials
         seed

(* Each count of terms from 1 to 17 at float32, rows of 64 outputs and of
   one, and the regimes the law covers. *)
let product_examples =
  let plain r = plain_view r in
  let flat ?(same = false) ?(broadcast = false) ?(v = plain 2) ?(specials = 0)
      ?(d = f32) ?(p = f32) shape axes =
    product_of ~same ~broadcast d p shape axes v (plain (Array.length shape))
      ~specials 11
  in
  let dw ?(d = f32) ?(p = f32) ?(c = 64) ?(lo = 1) ?(hi = 1) ?fill
      ?(specials = 0) k1 k2 =
    let fill = Option.value fill ~default:(List.hd (product_fills d)) in
    depthwise d p ~b:2 ~h:5 ~w:6 ~c ~lo ~hi ~k1 ~k2 ~step:1 ~dilation:1
      ~fill ~specials 13
  in
  List.init 17 (fun i -> flat ~broadcast:true [| 64; i + 1 |] [| 1 |])
  @ List.init 17 (fun i -> flat [| 3; i + 1 |] [| 1 |])
  @ [
      nan_before_units;
      flat ~d:f64 ~p:f64 [| 37; 9 |] [| 1 |];
      flat ~d:(D.Any D.Bfloat16) [| 70; 5 |] [| 1 |];
      flat ~d:(D.Any D.Float16) [| 70; 5 |] [| 1 |];
      flat ~d:(D.Any D.Float8_e4m3fn) ~specials:3 [| 70; 5 |] [| 1 |];
      flat ~d:(D.Any D.Float8_e5m2) ~specials:3 [| 70; 5 |] [| 1 |];
      flat ~d:(D.Any D.Float4_e2m1fn) ~specials:3 [| 70; 5 |] [| 1 |];
      flat ~d:f64 ~specials:3 [| 70; 5 |] [| 1 |];
      flat ~p:f64 [| 9; 3 |] [| 1 |];
      flat ~same:true [| 20; 7 |] [| 1 |];
      flat [| 3; 4; 5 |] [| 0; 2 |] ~v:(plain 3);
      flat [| 33; 6 |] [| 1 |] ~v:{ (plain 2) with stepped = true };
      flat [| 33; 6 |] [| 1 |] ~v:{ (plain 2) with reversed = Some 0 };
      flat [| 33; 6 |] [| 1 |] ~v:{ (plain 2) with perm = [| 1; 0 |] };
      flat [| 33; 6 |] [| 1 |] ~v:{ (plain 2) with broadcast = true };
      flat ~specials:3 [| 64; 9 |] [| 1 |];
      flat ~specials:3 ~d:f64 ~p:f64 [| 64; 9 |] [| 1 |];
      dw 3 3;
      dw ~d:f64 ~p:f64 3 3;
      dw ~d:(D.Any D.Bfloat16) 3 3;
      dw ~c:5 ~specials:4 2 3;
      dw ~fill:"\052\018\192\127" 4 4;
      dw ~fill:"\000\000\000\128" ~lo:2 ~hi:0 3 3;
      dw ~d:(D.Any D.Float4_e2m1fn) ~c:5 ~lo:0 ~hi:2 2 2;
      dw 4 5;
    ]

(* The elements of [c]'s Sum of [p]'s output over [loads] by [M]. *)
let sum_products (module M : Nx_kernel.S) (b : Support.backend) c p loads =
  let s = c.pshape in
  let kept =
    Array.of_list
      (List.filteri (fun i _ -> not (Array.mem i c.paxes)) (Array.to_list s))
  in
  let (D.Any pt) = c.pdt in
  let dst = A.Any (A.create b.device pt kept) in
  match
    M.reduce p ~shape:s ~axes:(Iarray.of_array c.paxes) [| (Monoid Sum, 0) |]
      ~dsts:[| dst |] loads
  with
  | A.Done -> Elements.elements (host dst)
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* The products summed as the program's output mapped into an array, then
   reduced: the same bits, a NaN as any NaN, on the job's threads and on
   one. *)
let law_products (b : Support.backend) c =
  let module K = (val b.kernels) in
  let s = c.pshape in
  let terms = Array.fold_left (fun n a -> n * s.(a)) 1 c.paxes in
  cover "float32" (same c.pdt f32 && same c.ldt f32);
  cover "float64" (same c.pdt f64 && same c.ldt f64);
  cover "bfloat16 factors" (same c.ldt (D.Any D.Bfloat16));
  cover "float16 factors" (same c.ldt (D.Any D.Float16));
  cover "float32 factors at float64" (same c.pdt f64 && same c.ldt f32);
  cover "float64 factors at float32" (same c.pdt f32 && same c.ldt f64);
  cover "float8_e4m3fn factors" (same c.ldt (D.Any D.Float8_e4m3fn));
  cover "float8_e5m2 factors" (same c.ldt (D.Any D.Float8_e5m2));
  cover "float4_e2m1fn factors" (same c.ldt (D.Any D.Float4_e2m1fn));
  cover "a load by itself" c.same;
  cover "a padded load" (c.pad <> None);
  cover "a padded load of few products" (c.pad <> None && terms <= 16);
  cover "a padded float4_e2m1fn load"
    (c.pad <> None && same c.ldt (D.Any D.Float4_e2m1fn));
  for n = 1 to 16 do
    cover (Printf.sprintf "%d terms" n) (terms = n)
  done;
  cover "17 terms" (terms = 17);
  cover "more terms" (terms > 17);
  cover "two reduced axes" (Array.length c.paxes = 2);
  List.iter (fun name -> cover name (List.mem name c.pnames))
    [ "a broadcast factor"; "a broadcast first factor"; "stepped"; "reversed";
      "permuted"; "broadcast"; "specials"; "padded unevenly";
      "a side unpadded"; "a NaN fill"; "a -0 fill" ];
  let prog = product_prog ~same:c.same c.ldt c.pdt in
  let (D.Any pt) = c.pdt in
  let y = A.create b.device pt s in
  (match K.map prog ~dsts:[| A.Any y |] (Array.map plain_of c.reads) with
  | A.Done -> ()
  | r -> failf "the map answered %a" Nx_array_support.pp_answer r);
  let reduce k p loads = sum_products k b c p loads in
  let want = reduce b.kernels (identity c.pdt) [| Nx_kernel.Plain y |] in
  let same = array (Elements.same c.pdt) in
  equal ~msg:"the job's threads" same want (reduce b.kernels prog c.loads);
  equal ~msg:"one thread" same want (reduce (module One) prog c.loads)

(* A Sum adds its terms from +0, so products of -0 by positive factors sum
   to +0: a row of 64 outputs of 9 terms, the positive factor broadcast. *)
let test_zero_products (b : Support.backend) () =
  let s = [| 64; 9 |] in
  let x = A.Any (A.of_array D.Float32 s (Array.make (64 * 9) (-0.))) in
  let row =
    A.of_array D.Float32 [| 1; 9 |]
      (Array.init 9 (fun i -> 0.5 +. Float.of_int i))
  in
  let y = A.Any (Option.get (A.move (M.Broadcast (Iarray.of_array s)) row)) in
  let c =
    {
      ldt = f32; pdt = f32; pshape = s; paxes = [| 1 |]; same = false;
      pad = None; loads = [| plain_of x; plain_of y |]; reads = [| x; y |];
      pnames = [];
    }
  in
  let prog = product_prog ~same:false f32 f32 in
  let zero =
    Elements.elements (A.Any (A.of_array D.Float32 [| 64 |] (Array.make 64 0.)))
  in
  equal ~msg:"the job's threads" (array string) zero
    (sum_products b.kernels b c prog c.loads);
  equal ~msg:"one thread" (array string) zero
    (sum_products (module One) b c prog c.loads)

(* A Sum of a program's output over padded loads gives the bits of the Sum
   over the arrays the loads read: a destination aliasing the operand, a
   coordinate read, two loads padded at different ends, a windowed load
   whose taps lie along an axis another load fills along, and an empty
   loop. *)
let test_padded_programs (b : Support.backend) () =
  let module K = (val b.kernels) in
  let one = "\000\000\128\063" in
  let pad ?(windows = [||]) lo hi =
    let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
    let interior = Iarray.map (fun _ -> 0) lo in
    { Nx_kernel.lo; hi; interior; windows = Iarray.of_array windows }
  in
  let w axis size = { M.axis; size; step = 1; dilation = 1 } in
  let operand s seed =
    operand f32 s (plain_view (Array.length s)) ~specials:0 seed
  in
  let neg =
    P.v ~ins:[| f32 |] P.[| In 0; Op1 (Unary Neg, f32, 0) |] ~outs:[| 1 |]
  in
  let coord =
    P.v ~ins:[| f32 |]
      P.[| In 0; Coord 0; Op1 (Cast, f32, 1); Op2 (Binary Add, 0, 2) |]
      ~outs:[| 3 |]
  in
  let product = product_prog ~same:false f32 f32 in
  let sum p ~shape axes dst loads =
    K.reduce p ~shape ~axes:(Iarray.of_array axes) [| (Monoid Sum, 0) |]
      ~dsts:[| dst |] loads
  in
  (* [p] summed along [axes] of the loads of [xs] padded by [pads] with
     one, into [into] where given, else a fresh array. *)
  let check ?into what p xs pads axes =
    let loads = Array.map2 (fun x pad -> padded_load x one pad) xs pads in
    let shape = loop_shape loads in
    let kept =
      Array.of_list
        (List.filteri (fun i _ -> not (Array.mem i axes)) (Array.to_list shape))
    in
    let fresh () = A.Any (A.create Rig.host D.Float32 kept) in
    let reads =
      Array.map2 (fun x pad -> plain_of (pad_array x one pad)) xs pads
    in
    let want = fresh () in
    equal ~msg:(what ^ ", from copies") Elements.answer A.Done
      (sum p ~shape axes want reads);
    let got = Option.value into ~default:(fresh ()) in
    equal ~msg:what Elements.answer A.Done (sum p ~shape axes got loads);
    equal ~msg:what (list int) (bytes_of want) (bytes_of got)
  in
  let x = operand [| 3 |] 1 in
  let y = operand [| 3 |] 1 in
  check "an aliased load" ~into:x (identity f32) [| x |]
    [| pad [| 1 |] [| 1 |] ~windows:[| w 0 3 |] |] [| 1 |];
  check "an aliased load of a program" ~into:y neg [| y |]
    [| pad [| 1 |] [| 1 |] ~windows:[| w 0 3 |] |] [| 1 |];
  check "a coordinate" coord [| operand [| 5; 2 |] 2 |]
    [| pad [| 1; 0 |] [| 1; 0 |] |] [| 1 |];
  check "two loads padded at different ends" product
    [| operand [| 5; 3 |] 5; operand [| 3; 3 |] 6 |]
    [| pad [| 1; 0 |] [| 0; 0 |] ~windows:[| w 0 2 |];
       pad [| 0; 0 |] [| 3; 0 |] ~windows:[| w 0 2 |] |]
    [| 1; 2 |];
  check "taps along another load's fill" product
    [| operand [| 5 |] 3; operand [| 3; 2 |] 4 |]
    [| pad [| 0 |] [| 0 |] ~windows:[| w 0 3 |]; pad [| 0; 0 |] [| 0; 1 |] |]
    [| 0 |];
  (* A window of 5 over 3 elements: a loop of [0; 5], whose Sum along its
     taps has no output, and along its windows five of no term, +0. *)
  let loads =
    [| padded_load (operand [| 3 |] 7) one
         (pad [| 0 |] [| 0 |] ~windows:[| w 0 5 |]) |]
  in
  let none = A.create Rig.host D.Float32 [| 0 |] in
  equal ~msg:"an empty loop" Elements.answer A.Done
    (sum neg ~shape:[| 0; 5 |] [| 1 |] (A.Any none) loads);
  let zeros = A.create Rig.host D.Float32 [| 5 |] in
  equal ~msg:"an empty loop, its taps kept" Elements.answer A.Done
    (sum neg ~shape:[| 0; 5 |] [| 0 |] (A.Any zeros) loads);
  equal ~msg:"an empty loop, its taps kept" (array float_exact)
    (Array.make 5 0.) (A.to_array zeros)

(* Carries *)

(* A dtype a carried scan draws: its finite elements, near one for a
   product, its NaNs, its other specials, and two elements whose sum, or
   product, is a NaN made from numbers. *)
type 'v drawn = {
  finite : monoid -> Random.State.t -> 'v;
  nans : 'v list;
  others : 'v list;
  made : monoid -> ('v * 'v) option;
}

type carried = Carried : ('v, 's) D.t * 'v drawn -> carried

let carried_kinds =
  let float_values =
    {
      finite =
        (fun m r ->
          match m with
          | Prod ->
              (if Random.State.bool r then 1. else -1.)
              *. (0.999 +. Random.State.float r 0.002)
          | _ ->
              Float.ldexp
                (Random.State.float r 2. -. 1.)
                (Random.State.int r 9 - 4));
      nans =
        [
          nan; -.nan; Int64.float_of_bits 0x7FF4_0000_0000_0000L;
          Int64.float_of_bits 0xFFF9_2340_0000_0000L;
        ];
      others = [ -0.; 0.; infinity; neg_infinity ];
      made =
        (function
        | Sum -> Some (infinity, neg_infinity)
        | Prod -> Some (0., infinity)
        | Max | Min -> None);
    }
  in
  let float dt = Carried (dt, float_values) in
  let complex dt =
    let c re im = { Complex.re; im } in
    Carried
      ( dt,
        {
          finite =
            (fun m r ->
              c (float_values.finite m r) (0.01 *. float_values.finite Sum r));
          nans = [ c nan 1.; c 1. (-.nan); c (-.nan) nan ];
          others = [ c (-0.) (-0.); c infinity 0.; c 0. neg_infinity ];
          made =
            (function
            | Sum -> Some (c infinity 0., c neg_infinity 1.)
            | Prod -> Some (c 0. 0., c infinity 0.)
            | Max | Min -> None);
        } )
  in
  let ints dt ~small ~edges =
    Carried
      ( dt,
        {
          finite = (fun _ r -> small (Random.State.int r 16 - 8));
          nans = [];
          others = edges;
          made = (fun _ -> None);
        } )
  in
  [
    float D.Float32; float D.Float64; float D.Float16; float D.Bfloat16;
    complex D.Complex64; complex D.Complex128;
    ints D.Int32 ~small:Int32.of_int ~edges:[ Int32.max_int; Int32.min_int ];
    ints D.Int8 ~small:Fun.id ~edges:[ 127; -128 ];
    ints D.Int4 ~small:Fun.id ~edges:[ 7; -8 ];
    Carried
      ( D.Bool,
        {
          finite = (fun _ r -> Random.State.bool r);
          nans = [];
          others = [ true; false ];
          made = (fun _ -> None);
        } );
  ]

(* A scan along axis 1 of [[outer; p + m; inner]] whose last [m] terms of
   each slice are scanned from the carry of its first [p]: [p] whole chunks,
   or, for an exact scan, [p + m] terms within one. The first [p] hold no
   NaN, but a NaN made from numbers where [made] and a NaN term where
   [nan_before], of a [Max] or [Min]; the last hold a NaN where
   [nan_after]. The terms are a program's copy of the operand where
   [through]. *)
type carry_case = {
  kind : carried;
  cm : monoid;
  through : bool;
  outer : int;
  inner : int;
  p : int;
  m : int;
  made : bool;
  nan_before : bool;
  nan_after : bool;
  cseed : int;
}

let pp_carry_case ppf c =
  let (Carried (dt, _)) = c.kind in
  Format.fprintf ppf
    "scan %s of %s%a [%d; %d + %d; %d]%s%s%s, seed %d" (monoid_name c.cm)
    (if c.through then "a copy of " else "")
    D.pp dt c.outer c.p c.m c.inner
    (if c.made then ", a made NaN before" else "")
    (if c.nan_before then ", a NaN before" else "")
    (if c.nan_after then ", a NaN after" else "")
    c.cseed

let carry_case =
  let open Gen in
  with_pp pp_carry_case
    (* One case in five carries a NaN made from numbers and holds a NaN
       term after it: few draws of independent choices meet all of them. *)
    (let* both = frequency [ (4, constant false); (1, constant true) ] in
     let* (Carried (dt, vs) as kind) =
       of_list
         (List.filter
            (fun (Carried (_, vs)) -> (not both) || vs.made Sum <> None)
            carried_kinds)
     in
     let* cm =
       of_list
         (List.filter
            (fun m -> accepts m (D.Any dt) && ((not both) || vs.made m <> None))
            [ Sum; Prod; Max; Min ])
     in
     let* outer = of_list [ 1; 2 ] in
     let* inner = of_list [ 1; 3 ] in
     let exact =
       (cm = Max || cm = Min) || not (D.is D.Float dt || D.is D.Complex dt)
     in
     let* aligned = if exact then bool else constant true in
     (* Such a case needs two terms before the carry and one after it. *)
     let least k l = if both then List.filter (fun n -> n >= k) l else l in
     let* p, m =
       if aligned then
         let* k = of_list [ 1; 2 ] in
         let+ m = of_list (least 1 [ 0; 1; 5; 4095; 4096; 4097; 9000 ]) in
         (4096 * k, m)
       else
         let* p = of_list (least 2 [ 1; 2; 17; 1024; 2049; 4095 ]) in
         let+ m = of_list (least 1 [ 0; 1; 4096 - p; (4096 - p) / 2 ]) in
         (p, m)
     in
     let* made = if both then constant true else bool in
     let* nan_before =
       if both || vs.nans = [] || (cm <> Max && cm <> Min) then constant false
       else bool
     in
     let* nan_after = if both then constant true else bool in
     let* through = bool in
     let+ cseed = int in
     { kind; cm; through; outer; inner; p; m; made; nan_before; nan_after; cseed })

(* The case's terms in C order of [[outer; p + m; inner]]. *)
let carry_terms (type v) (vs : v drawn) c : v array =
  let r = Random.State.make [| c.cseed |] in
  let n = c.p + c.m in
  let pick l = List.nth l (Random.State.int r (List.length l)) in
  let term t =
    let special = Random.State.int r 64 = 0 in
    if special && t >= c.p && c.nan_after && vs.nans <> [] then pick vs.nans
    else if special then pick vs.others
    else vs.finite c.cm r
  in
  let a =
    Array.init (c.outer * n * c.inner) (fun i -> term (i / c.inner mod n))
  in
  let set t v =
    for o = 0 to c.outer - 1 do
      for i = 0 to c.inner - 1 do
        a.((((o * n) + t) * c.inner) + i) <- v
      done
    done
  in
  (match vs.made c.cm with
  | Some (u, v) when c.made && c.p > 1 ->
      set (Random.State.int r (c.p - 1)) u;
      set (c.p - 1) v
  | _ -> ());
  if c.nan_before then set (Random.State.int r c.p) (pick vs.nans);
  (if c.nan_after && c.m > 0 && vs.nans <> [] then
     let t = c.p + Random.State.int r c.m in
     set t (pick vs.nans));
  a

(* Elements [lo, lo + count) along axis 1 of [x]. *)
let along1 (A.Any x) lo count =
  let s = L.shape (A.layout x) in
  let r =
    Array.mapi
      (fun a e ->
        if a = 1 then { M.start = lo; count; step = 1 }
        else { M.start = 0; count = e; step = 1 })
      s
  in
  A.Any (Option.get (A.move (M.Slice (Iarray.of_array r)) x))

let law_carry (b : Support.backend) c =
  let (Carried (dt, vs)) = c.kind in
  let (module K : Nx_kernel.S) = b.kernels in
  let n = c.p + c.m in
  let aligned = c.p mod 4096 = 0 in
  cover "a carry into a chunk's start" aligned;
  cover "an exact scan's carry within a chunk" (not aligned);
  cover "several chunks after a carry" (aligned && c.m > 4096);
  cover "no term after a carry" (c.m = 0);
  cover "several slices" (c.outer * c.inner > 1);
  let floating = D.is D.Float dt || D.is D.Complex dt in
  let made = c.made && c.p > 1 && vs.made c.cm <> None in
  cover "a NaN carry made from numbers" made;
  cover "a NaN carry and a NaN term after it" (made && c.nan_after && c.m > 0);
  cover "a NaN term after a carry" (floating && c.nan_after && c.m > 0);
  cover "a NaN term before an extreme's carry" (c.nan_before && c.m > 0);
  cover "a narrow float's carry" (D.bits dt = 16 && D.is D.Float dt);
  cover "a program's terms" c.through;
  let z =
    on b
      (A.Any (A.of_array dt [| c.outer; n; c.inner |] (carry_terms vs c)))
  in
  let (D.Any ct) = carrier (D.Any dt) in
  let ok what = function
    | A.Done -> ()
    | r -> failf "%s: the kernels answered %a" what Nx_array_support.pp_answer r
  in
  let p0 =
    if c.through then
      P.v ~ins:[| D.Any dt |]
        [| P.In 0; P.Op1 (Copy, D.Any dt, 0) |]
        ~outs:[| 1 |]
    else identity (D.Any dt)
  in
  (* The scan of [x] into a fresh destination of [into]. *)
  let scan (type v s) ?carry (into : (v, s) D.t) (A.Any x) =
    let y = A.Any (A.create b.device into (L.shape (A.layout x))) in
    ok "scan"
      (K.scan ?carry p0 ~axis:1 (Monoid c.cm, 0) ~dsts:[| y |] [| Plain x |]);
    y
  in
  let carry =
    if aligned then begin
      (* The carry chain over the chunks' totals, each a reduction in the
         carrier, combined by the monoid's Prog kind from the first. *)
      let total k =
        let (A.Any x) = along1 z (4096 * k) 4096 in
        let y = A.Any (A.create b.device ct [| c.outer; c.inner |]) in
        ok "reduce"
          (K.reduce p0 ~shape:(L.shape (A.layout x))
             ~axes:[| 1 |] [| (Monoid c.cm, 0) |] ~dsts:[| y |] [| Plain x |]);
        y
      in
      let kind : P.binary =
        match c.cm with
        | Sum -> Add
        | Prod -> Mul
        | Max -> Maximum
        | Min -> Minimum
      in
      let combine =
        P.v
          ~ins:[| D.Any ct; D.Any ct |]
          [| P.In 0; P.In 1; P.Op2 (Binary kind, 0, 1) |]
          ~outs:[| 2 |]
      in
      let rec chain acc k =
        if 4096 * k >= c.p then acc
        else
          let (A.Any a) = acc in
          let (A.Any t) = total k in
          let y = A.Any (A.create b.device ct [| c.outer; c.inner |]) in
          ok "map" (K.map combine ~dsts:[| y |] [| Plain a; Plain t |]);
          chain y (k + 1)
      in
      let (A.Any v) = chain (total 0) 1 in
      A.Any (Option.get (A.move (M.Reshape [| c.outer; 1; c.inner |]) v))
    end
    else along1 (scan ct (along1 z 0 c.p)) (c.p - 1) 1
  in
  let got = scan ~carry dt (along1 z c.p c.m) in
  let want = along1 (scan dt z) c.p c.m in
  (* A NaN as any NaN; a complex Sum's parts apart, as it adds them. *)
  let parts y =
    let (A.Any h as a) = host y in
    match A.dtype h with
    | D.Complex64 when c.cm = Sum ->
        A.Any (Option.get (A.bitcast D.Float32 (A.copy h)))
    | D.Complex128 when c.cm = Sum ->
        A.Any (Option.get (A.bitcast D.Float64 (A.copy h)))
    | _ -> a
  in
  let want = parts want and got = parts got in
  equal (array (Elements.same (dtype_of want)))
    (Elements.elements want) (Elements.elements got)

(* A carry of a reduction that takes none, or of another shape or dtype than
   the carrier's, is refused. *)
let test_carry_refusals (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let make dt s = A.Any (A.create b.device dt s) in
  let scan (type v s) (dt : (v, s) D.t) ?carry r dsts =
    let (A.Any x) = make dt [| 2; 3 |] in
    K.scan ?carry (identity (D.Any dt)) ~axis:1 (r, 0) ~dsts [| Plain x |]
  in
  let f32 s = make D.Float32 s in
  let refuses msg want got = equal ~msg Elements.answer want got in
  let carry = f32 [| 2; 1 |] in
  refuses "a Logsumexp's carry" A.Shape_mismatch
    (scan D.Float32 ~carry Logsumexp [| f32 [| 2; 3 |] |]);
  refuses "a Moments' carry" A.Shape_mismatch
    (scan D.Float32 ~carry Moments [| f32 [| 2; 3 |]; f32 [| 2; 3 |] |]);
  refuses "an Arg's carry" A.Shape_mismatch
    (scan D.Float32 ~carry (Arg Max)
       [| f32 [| 2; 3 |]; make D.Int64 [| 2; 3 |] |]);
  refuses "a carry of the destination's shape" A.Shape_mismatch
    (scan D.Float32 ~carry:(f32 [| 2; 3 |]) (Monoid Sum) [| f32 [| 2; 3 |] |]);
  refuses "a carry of one element" A.Shape_mismatch
    (scan D.Float32 ~carry:(f32 [| 1; 1 |]) (Monoid Sum) [| f32 [| 2; 3 |] |]);
  refuses "a carry of another dtype" A.Wrong_dtype
    (scan D.Float32 ~carry:(make D.Float64 [| 2; 1 |]) (Monoid Max)
       [| f32 [| 2; 3 |] |]);
  refuses "a float16 carry of float16 terms" A.Wrong_dtype
    (scan D.Float16 ~carry:(make D.Float16 [| 2; 1 |]) (Monoid Sum)
       [| make D.Float16 [| 2; 3 |] |]);
  refuses "a float32 carry of float16 terms" A.Done
    (scan D.Float16 ~carry (Monoid Sum) [| make D.Float16 [| 2; 3 |] |])

(* A Max's NaN carry that is its destination, a slice of one term, a
   number, makes the result a NaN, though the scan writes the carry's
   bytes. *)
let test_aliased_carry (b : Support.backend) () =
  let (module K : Nx_kernel.S) = b.kernels in
  let on_b x = if Rig.equal (A.device x) b.device then x else A.to_device b.device x in
  let f64 = Int64.float_of_bits in
  let y = on_b (A.of_array D.Float64 [| 1; 1 |] [| f64 0x7FF4_0000_0000_0001L |]) in
  let x = on_b (A.of_array D.Float64 [| 1; 1 |] [| 1. |]) in
  let answer =
    K.scan ~carry:(A.Any y) (identity (D.Any D.Float64)) ~axis:1
      (Monoid Max, 0) ~dsts:[| A.Any y |] [| Plain x |]
  in
  equal ~msg:"answer" Elements.answer A.Done answer;
  let y = if Rig.equal (A.device y) Rig.host then y else A.to_device Rig.host y in
  equal ~msg:"a NaN" bool true (Float.is_nan (A.to_array y).(0))

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "a float sum is within its bound of the exact sum" sums
        (run (law_bound b));
      prop ~examples:computed_examples "bits do not depend on layouts" computed
        (run (law_layouts b));
      prop ~examples:scan_examples "a scan of a prefix is the prefix of the scan"
        scans (run (law_prefix b));
      prop ~examples:any_examples "a declined case writes nothing" any_case
        (run (law_declined b));
      prop "a scan from a carry continues the scan whose terms it carries"
        carry_case (run (law_carry b));
      test "a carry the scan cannot take is refused" (fun () ->
          b.around (test_carry_refusals b));
      test "an extreme's carry its destination aliases" (fun () ->
          b.around (test_aliased_carry b));
    ]

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
  | d -> carrier d

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

(* [c]'s reduction, or its scan where [scan], of [x] by [k], into fresh
   destinations of [d], an Arg's positions of int64. *)
let stat_run k ~scan (c : stat) (D.Any d) x =
  let module K = (val k : Nx_kernel.S) in
  let s = if scan then shape_of x else stat_shape c in
  let make : type a b. (a, b) D.t -> A.any =
   fun dt -> A.Any (A.create Rig.host dt s)
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
  match answer with
  | A.Done -> dsts
  | a -> failf "the kernels answered %a" Nx_array_support.pp_answer a

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

let order (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu's cases and order, " ^ b.name)
    [
      prop ~examples:(computed_examples @ made_examples)
        "each result folds its terms in nx.cpu's order" computed
        (run (law_order b));
      prop ~examples:any_examples
        "computes every Sum, Prod, Max and Min"
        any_case (run (law_computes b));
      test "a bool of any non-zero byte is true" (fun () ->
          b.around (test_bool_bytes b));
      test "a program of Prog.of_node's operand computes" (fun () ->
          b.around (test_of_node b));
      test "a NaN past the first block and chunk makes the result a NaN"
        (fun () -> b.around (test_far_nans b));
      test "8- and 16-bit products and sums wrap at their extremes" (fun () ->
          b.around (test_wrap b));
      test
        "refuses a destination or a reduction's load of another shape and an \
         extreme of nothing, answers Wrong_dtype for dtypes a reduction does \
         not take"
        (fun () -> b.around (test_refusals b));
      test
        "refuses a padded load past max_numel or one that does not fit its \
         destinations before copying it"
        (fun () -> b.around (test_padded_refusals b));
    ]

(* A reduction and a scan of a program of as many loads as it holds, the sum
   of fifteen loads of [x]: fifteen times [x]'s sums along the last axis, and
   its running sums. *)
let test_most_loads (b : Support.backend) () =
  let module K = (val b.kernels) in
  let nl = P.max_operands - 1 in
  let nodes =
    List.init nl (fun k -> P.In k)
    @ List.init (nl - 1) (fun i ->
          P.Op2 (Binary Add, (if i = 0 then 0 else nl + i - 1), i + 1))
  in
  let p =
    P.v
      ~ins:(Iarray.init nl (fun _ -> D.Any D.Float32))
      (Iarray.of_list nodes)
      ~outs:[| List.length nodes - 1 |]
  in
  let x = A.of_array D.Float32 [| 4; 3 |] (Array.init 12 Float.of_int) in
  let loads = Array.make nl (Nx_kernel.Plain x) in
  let sums = A.create Rig.host D.Float32 [| 4 |] in
  equal ~msg:"reduce" Elements.answer A.Done
    (K.reduce p ~shape:[| 4; 3 |] ~axes:[| 1 |] [| (Monoid Sum, 0) |]
       ~dsts:[| A.Any sums |] loads);
  equal ~msg:"sums" (array float_exact) [| 45.; 180.; 315.; 450. |]
    (A.to_array sums);
  let running = A.create Rig.host D.Float32 [| 4; 3 |] in
  equal ~msg:"scan" Elements.answer A.Done
    (K.scan p ~axis:1 (Monoid Sum, 0) ~dsts:[| A.Any running |] loads);
  equal ~msg:"running sums" (array float_exact)
    (Array.init 12 (fun i ->
         let r = i / 3 and c = i mod 3 in
         Float.of_int (15 * (((c + 1) * 3 * r) + (c * (c + 1) / 2)))))
    (A.to_array running)

(* Nine Moments of one loop take 18 destinations, twenty Args 40, and many
   sums as many tables of reductions, more than one loop holds beside its
   load: each result has the bits of a call of few reductions. *)
let test_many_reductions (b : Support.backend) () =
  let module K = (val b.kernels) in
  let p = identity (D.Any D.Float32) and shape = [| 4; 3 |] in
  let x =
    A.of_array D.Float32 shape
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

(* A reduction or scan given destinations other than its reduction's is
   refused, never declined: nx.cpu computes every case its contract offers. *)
let test_malformed (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let f32 s = A.Any (A.create Rig.host D.Float32 s) in
  let p = identity (D.Any D.Float32) and shape = [| 2; 3 |] in
  let reduce r dsts =
    K.reduce p ~shape ~axes:[| 1 |] [| (r, 0) |] ~dsts [| plain_of (A.Any x) |]
  in
  equal ~msg:"a sum into two destinations" Elements.answer A.Shape_mismatch
    (reduce (Monoid Sum) [| f32 [| 2 |]; f32 [| 2 |] |]);
  equal ~msg:"Moments into one destination" Elements.answer A.Shape_mismatch
    (reduce Moments [| f32 [| 2 |] |]);
  equal ~msg:"Arg's positions in int32" Elements.answer A.Wrong_dtype
    (reduce (Arg Max) [| f32 [| 2 |]; A.Any (A.create Rig.host D.Int32 [| 2 |]) |]);
  equal ~msg:"a scan into no destination" Elements.answer A.Shape_mismatch
    (K.scan p ~axis:1 (Monoid Sum, 0) ~dsts:[||] [| plain_of (A.Any x) |])

(* A reduction and a scan of a program that is its one load's identity, of
   plain loads, allocate nothing: their tables live in C. *)
let test_identity_allocates_nothing (b : Support.backend) =
  let module K = (val b.kernels) in
  let n = 1024 in
  let x = A.of_array D.Float32 [| n |] (Array.make n 1.) in
  let p = identity (D.Any D.Float32) and loads = [| Nx_kernel.Plain x |] in
  let into = [| A.Any (A.create Rig.host D.Float32 [||]) |] in
  let along = [| A.Any (A.create Rig.host D.Float32 [| n |]) |] in
  let shape = [| n |] and axes = Iarray.of_array [| 0 |] in
  let rs = [| (Nx_kernel.Monoid Sum, 0) |] and rk = (Nx_kernel.Monoid Sum, 0) in
  let reduce () = K.reduce p ~shape ~axes rs ~dsts:into loads in
  let scan () = K.scan p ~axis:0 rk ~dsts:along loads in
  let words f =
    (match f () with
    | A.Done -> ()
    | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r);
    let before = Gc.minor_words () in
    for _ = 1 to 100 do
      ignore (Sys.opaque_identity (f ()))
    done;
    Gc.minor_words () -. before
  in
  equal ~msg:"reduce" (float 0.5) 0. (words reduce);
  equal ~msg:"scan" (float 0.5) 0. (words scan)

let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      test "reduces many reductions in one claim of their operands" (fun () ->
          b.around (test_many_claims b));
      test "reduces more reductions than one loop holds" (fun () ->
          b.around (test_many_reductions b));
      test "reduces and scans a program of as many loads as it holds"
        (fun () -> b.around (test_most_loads b));
      test "refuses destinations other than a reduction's" (fun () ->
          b.around (test_malformed b));
      test "an identity reduction and scan allocate nothing" (fun () ->
          b.around (fun () -> test_identity_allocates_nothing b));
      prop ~examples:dot_examples "a sum of one axis is a contraction with ones"
        dots (run (law_contract b));
      prop ~examples:into_examples
        "narrow terms reduce in their carrier, rounded once" into_case
        (run (law_into b));
      test "a narrow result a NaN term reaches is a NaN" (fun () ->
          b.around (test_nan_bits b));
      prop ~examples:stat_into_examples
        "narrow Arg, Logsumexp and Moments reduce in their carrier"
        stat_into_case (run (law_stat_into b));
      prop ~examples:complex_examples
        "complex sums, products and extremes fold in nx.cpu's order"
        complex_case (run (law_complex b));
      prop ~count:20 ~examples:large_examples "one thread gives the job's bits"
        large (run (law_threads b));
      prop ~examples:padded_examples "a padded load reduces as the array it reads"
        padded_case (run (law_padded b));
      prop ~examples:fused_examples
        "a reduction or scan of a program's output gives the bits of its \
         map's"
        fused_case (run (law_fused b));
      prop ~examples:product_examples
        "a sum of few products of two loads gives the bits of their map's"
        product_case (run (law_products b));
      test "a sum of products of -0 by positive factors is +0"
        (test_zero_products b);
      test "a program over padded loads sums as over the arrays they read"
        (fun () -> b.around (test_padded_programs b));
      test "a destination aliasing a load computes as a fresh one" (fun () ->
          b.around (test_program_in_place b));
      prop
        ~examples:
          [
            { first = fused_example 4 Moments [| 5; 40 |] [| 1 |];
              red2 = Monoid Max; k2 = 0 };
          ]
        "several reductions in one call give each one's bits" several_case
        (run (law_several b));
      test "several reductions refuse a misfit, writing nothing" (fun () ->
          b.around (test_several_refuse b));
      prop ~examples:moments_examples
        "a Moments scan is within its bound, NaNs and infinities as stated"
        moments_case (run (law_moments b));
      prop ~examples:stat_examples
        "Arg is exact, Logsumexp and Moments within their bounds, each the \
         same bits on one thread and from a copy"
        stat_case (run (law_stateful b));
      prop ~examples:spread_examples
        "Logsumexp and Moments give the bits of their two passes" spread_case
        (run (law_record b));
      test "a reduction or scan into its own operand gives fresh bits"
        (fun () -> b.around (test_in_place b));

      prop ~examples:stat_scan_examples
        "Arg scans are exact, Logsumexp scans their network's bits within \
         their bound, each the same on one thread and from a copy"
        stat_scan_case (run (law_stat_scan b));
      prop ~examples:lanes_examples
        "a scan by Arg, Logsumexp or Moments gives each slice its bits alone"
        lanes_case (run (law_lanes b));
      test "a complex Arg scan holds its first extreme and first NaN"
        (fun () -> b.around (test_complex_arg_scan b));
      test "refuses an Arg of nothing and a Moments scan's misshapen \
            destinations"
        (fun () -> b.around (test_stateful_cases b));
      prop ~count:500 ~examples:guard_examples
        "reads and writes no byte past the operand and destinations, \
         whatever pages lie beside them"
        (Gen.pair guard_case Elements.sides)
        (run (law_guards b));
      prop ~count:200
        "reads no byte past a padded load's operand, whatever pages lie \
         beside it"
        (Gen.pair padded_case Elements.sides)
        (run (law_padded_guards b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.reduce"
       (List.map laws Support.backends
       @ List.map order Support.cpus
       @ List.map cpu Support.cpus))
